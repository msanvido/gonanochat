package main

import (
	"fmt"
	"math"
)

// GPTConfig mirrors the Python GPTConfig dataclass.
type GPTConfig struct {
	SequenceLen   int    `json:"sequence_len"`
	VocabSize     int    `json:"vocab_size"`
	NLayer        int    `json:"n_layer"`
	NHead         int    `json:"n_head"`
	NKVHead       int    `json:"n_kv_head"`
	NEmbd         int    `json:"n_embd"`
	WindowPattern string `json:"window_pattern"`
}

func (c GPTConfig) HeadDim() int {
	return c.NEmbd / c.NHead
}

// Attention holds weights for CausalSelfAttention.
type Attention struct {
	CQ    *Tensor // (n_head * head_dim, n_embd)
	CK    *Tensor // (n_kv_head * head_dim, n_embd)
	CV    *Tensor // (n_kv_head * head_dim, n_embd)
	CProj *Tensor // (n_embd, n_embd)

	VEGate *Tensor // (n_kv_head, ve_gate_channels) or nil
}

// MLP holds weights for the feed-forward network.
type MLP struct {
	CFC   *Tensor // (4 * n_embd, n_embd)
	CProj *Tensor // (n_embd, 4 * n_embd)
}

// Block holds a single transformer block.
type Block struct {
	Attn Attention
	MLP  MLP
}

// GPT holds the full model weights for inference.
type GPT struct {
	Config GPTConfig

	// Token embeddings: (padded_vocab_size, n_embd)
	WTE *Tensor
	// Output projection: (padded_vocab_size, n_embd)
	LMHead *Tensor

	// Per-layer scalars
	ResidLambdas []float32 // (n_layer,)
	X0Lambdas    []float32 // (n_layer,)

	// Smear gate
	SmearGate   *Tensor // (1, 24) weight matrix
	SmearLambda float32
	// Backout lambda
	BackoutLambda float32

	// Transformer blocks
	Blocks []Block

	// Value embeddings: layer_idx -> (padded_vocab_size, n_kv_head * head_dim)
	ValueEmbeds map[int]*Tensor

	// Precomputed rotary embeddings
	Cos *Tensor // (max_seq_len, head_dim/2)
	Sin *Tensor // (max_seq_len, head_dim/2)

	// Per-layer window sizes (left context length, -1 = full)
	WindowSizes []int

	// Padded vocab size (used internally)
	PaddedVocabSize int
}

// hasVE returns true if a layer should have Value Embedding (alternating layers).
func hasVE(layerIdx, nLayer int) bool {
	return layerIdx%2 == (nLayer-1)%2
}

// NewGPT creates a GPT model from config and initializes rotary embeddings and window sizes.
func NewGPT(config GPTConfig) *GPT {
	// Compute padded vocab size (pad to 64)
	paddedVocab := ((config.VocabSize + 63) / 64) * 64

	// Compute window sizes
	pattern := config.WindowPattern
	if pattern == "" {
		pattern = "SSSL"
	}
	longWindow := config.SequenceLen
	shortWindow := ceilDiv(longWindow, 4*128) * 128 // ceil to 128 tile

	windowSizes := make([]int, config.NLayer)
	for i := 0; i < config.NLayer; i++ {
		ch := pattern[i%len(pattern)]
		switch ch {
		case 'L', 'l':
			windowSizes[i] = longWindow
		case 'S', 's':
			windowSizes[i] = shortWindow
		default:
			windowSizes[i] = longWindow
		}
	}
	// Final layer always gets full context
	windowSizes[config.NLayer-1] = longWindow

	// Precompute rotary embeddings
	maxSeqLen := config.SequenceLen * 10
	cos, sin := PrecomputeRotaryEmbeddings(maxSeqLen, config.HeadDim(), 100000.0)

	return &GPT{
		Config:          config,
		Blocks:          make([]Block, config.NLayer),
		ValueEmbeds:     make(map[int]*Tensor),
		Cos:             cos,
		Sin:             sin,
		WindowSizes:     windowSizes,
		PaddedVocabSize: paddedVocab,
	}
}

func ceilDiv(a, b int) int {
	return (a + b - 1) / b
}

// Forward runs the model forward pass for inference.
// idx is a slice of token IDs (length T).
// kvCache may be nil (naive mode) or a KVCache for efficient generation.
// Returns logits of shape (vocab_size,) for the last position only.
func (m *GPT) Forward(idx []int, kvCache *KVCache) []float32 {
	T := len(idx)
	C := m.Config.NEmbd
	nHead := m.Config.NHead
	nKVHead := m.Config.NKVHead
	headDim := m.Config.HeadDim()
	nLayer := m.Config.NLayer
	halfDim := headDim / 2
	veGateChannels := 12

	// Determine rotary position offset
	T0 := 0
	if kvCache != nil {
		T0 = kvCache.Pos
	}

	// 1. Embed tokens: x has shape (T, C)
	x := make([]float32, T*C)
	for t := 0; t < T; t++ {
		copy(x[t*C:(t+1)*C], m.WTE.Row(idx[t]))
	}

	// 2. RMSNorm each position
	for t := 0; t < T; t++ {
		RMSNorm(x[t*C : (t+1)*C])
	}

	// 3. Smear: mix previous token's embedding into current position
	if kvCache != nil {
		// Save current last position embedding for next step
		prevEmb := kvCache.PrevEmbedding
		kvCache.PrevEmbedding = make([]float32, C)
		copy(kvCache.PrevEmbedding, x[(T-1)*C:T*C])

		if T > 1 {
			// Prefill: apply smear to positions 1+
			for t := 1; t < T; t++ {
				xt := x[t*C : (t+1)*C]
				xPrev := x[(t-1)*C : t*C]
				gateInput := xt[:24]
				gateVal := m.SmearLambda * Sigmoid(vecDotSmall(m.SmearGate.Data, gateInput, 24))
				VecMulAdd(xt, gateVal, xPrev)
			}
		} else if prevEmb != nil {
			// Decode: single token, use cached prev embedding
			xt := x[0:C]
			gateInput := xt[:24]
			gateVal := m.SmearLambda * Sigmoid(vecDotSmall(m.SmearGate.Data, gateInput, 24))
			VecMulAdd(xt, gateVal, prevEmb)
		}
	} else if T > 1 {
		// Training-style (no cache): apply smear to positions 1+
		for t := 1; t < T; t++ {
			xt := x[t*C : (t+1)*C]
			xPrev := x[(t-1)*C : t*C]
			gateInput := xt[:24]
			gateVal := m.SmearLambda * Sigmoid(vecDotSmall(m.SmearGate.Data, gateInput, 24))
			VecMulAdd(xt, gateVal, xPrev)
		}
	}

	// Save x0 for x0 residual blending
	x0 := make([]float32, T*C)
	copy(x0, x)

	// 4. Transformer blocks
	backoutLayer := nLayer / 2
	var xBackout []float32

	for layer := 0; layer < nLayer; layer++ {
		block := &m.Blocks[layer]

		// Apply per-layer residual scaling: x = resid_lambda * x + x0_lambda * x0
		rl := m.ResidLambdas[layer]
		xl := m.X0Lambdas[layer]
		for i := range x {
			x[i] = rl*x[i] + xl*x0[i]
		}

		// --- Attention sublayer ---
		// RMSNorm for attention input
		normX := make([]float32, T*C)
		copy(normX, x)
		for t := 0; t < T; t++ {
			RMSNorm(normX[t*C : (t+1)*C])
		}

		// Project Q, K, V for all positions
		q := make([]float32, T*nHead*headDim)
		k := make([]float32, T*nKVHead*headDim)
		v := make([]float32, T*nKVHead*headDim)
		for t := 0; t < T; t++ {
			nx := normX[t*C : (t+1)*C]
			qRow := MatVecMul(block.Attn.CQ, nx)
			kRow := MatVecMul(block.Attn.CK, nx)
			vRow := MatVecMul(block.Attn.CV, nx)
			copy(q[t*nHead*headDim:(t+1)*nHead*headDim], qRow)
			copy(k[t*nKVHead*headDim:(t+1)*nKVHead*headDim], kRow)
			copy(v[t*nKVHead*headDim:(t+1)*nKVHead*headDim], vRow)
		}

		// Value embedding (ResFormer-style)
		if ve, ok := m.ValueEmbeds[layer]; ok && block.Attn.VEGate != nil {
			for t := 0; t < T; t++ {
				veRow := ve.Row(idx[t]) // (nKVHead * headDim,)
				// Compute gate: 3 * sigmoid(ve_gate @ normX[:veGateChannels]) per kv_head
				gateInput := normX[t*C : t*C+veGateChannels]
				gates := MatVecMul(block.Attn.VEGate, gateInput) // (nKVHead,)
				for h := 0; h < nKVHead; h++ {
					gate := 3.0 * Sigmoid(gates[h])
					vOff := t*nKVHead*headDim + h*headDim
					veOff := h * headDim
					for d := 0; d < headDim; d++ {
						v[vOff+d] += gate * veRow[veOff+d]
					}
				}
			}
		}

		// Apply rotary embeddings to Q and K
		for t := 0; t < T; t++ {
			pos := T0 + t
			cosRow := m.Cos.Data[pos*halfDim : (pos+1)*halfDim]
			sinRow := m.Sin.Data[pos*halfDim : (pos+1)*halfDim]
			for h := 0; h < nHead; h++ {
				ApplyRotaryEmb(q[t*nHead*headDim+h*headDim:t*nHead*headDim+(h+1)*headDim], cosRow, sinRow)
			}
			for h := 0; h < nKVHead; h++ {
				ApplyRotaryEmb(k[t*nKVHead*headDim+h*headDim:t*nKVHead*headDim+(h+1)*headDim], cosRow, sinRow)
			}
		}

		// QK norm + scale by 1.2
		for t := 0; t < T; t++ {
			for h := 0; h < nHead; h++ {
				qHead := q[t*nHead*headDim+h*headDim : t*nHead*headDim+(h+1)*headDim]
				RMSNorm(qHead)
				VecScale(qHead, 1.2)
			}
			for h := 0; h < nKVHead; h++ {
				kHead := k[t*nKVHead*headDim+h*headDim : t*nKVHead*headDim+(h+1)*headDim]
				RMSNorm(kHead)
				VecScale(kHead, 1.2)
			}
		}

		// Attention computation
		var attnOut []float32
		if kvCache != nil {
			attnOut = m.attentionWithCache(layer, q, k, v, T, kvCache)
		} else {
			attnOut = m.attentionNaive(q, k, v, T, m.WindowSizes[layer])
		}

		// Output projection and residual
		for t := 0; t < T; t++ {
			yProj := MatVecMul(block.Attn.CProj, attnOut[t*C:(t+1)*C])
			VecAdd(x[t*C:(t+1)*C], x[t*C:(t+1)*C], yProj)
		}

		// --- MLP sublayer ---
		normX2 := make([]float32, T*C)
		copy(normX2, x)
		for t := 0; t < T; t++ {
			RMSNorm(normX2[t*C : (t+1)*C])
		}

		mlpDim := 4 * C
		for t := 0; t < T; t++ {
			nx := normX2[t*C : (t+1)*C]
			h := MatVecMul(block.MLP.CFC, nx)
			ReLUSquared(h)
			y := MatVecMul(block.MLP.CProj, h)
			xSlice := x[t*C : (t+1)*C]
			VecAdd(xSlice, xSlice, y)
		}
		_ = mlpDim

		// Save backout layer
		if layer == backoutLayer {
			xBackout = make([]float32, T*C)
			copy(xBackout, x)
		}

		// Advance KV cache position after the last layer
		if kvCache != nil && layer == nLayer-1 {
			kvCache.Pos += T
		}
	}

	// 5. Backout: subtract mid-layer residual
	if xBackout != nil {
		bl := m.BackoutLambda
		for i := range x {
			x[i] -= bl * xBackout[i]
		}
	}

	// 6. Final RMSNorm
	for t := 0; t < T; t++ {
		RMSNorm(x[t*C : (t+1)*C])
	}

	// 7. Compute logits for the last position only
	lastX := x[(T-1)*C : T*C]
	logits := MatVecMul(m.LMHead, lastX) // (padded_vocab_size,)

	// Crop to actual vocab size
	logits = logits[:m.Config.VocabSize]

	// 8. Softcap
	TanhSoftcap(logits, 15.0)

	return logits
}

// attentionNaive computes causal self-attention without KV cache (for prefill or naive mode).
func (m *GPT) attentionNaive(q, k, v []float32, T, windowSize int) []float32 {
	nHead := m.Config.NHead
	nKVHead := m.Config.NKVHead
	headDim := m.Config.HeadDim()
	C := m.Config.NEmbd
	headsPerKVGroup := nHead / nKVHead
	scale := float32(1.0 / math.Sqrt(float64(headDim)))

	attnOut := make([]float32, T*C)

	for h := 0; h < nHead; h++ {
		kvH := h / headsPerKVGroup // GQA: which KV head this query head uses

		for t := 0; t < T; t++ {
			// Compute attention scores for position t
			qHead := q[t*nHead*headDim+h*headDim : t*nHead*headDim+(h+1)*headDim]

			// Determine attention window
			startPos := 0
			if windowSize > 0 && t-windowSize > 0 {
				startPos = t - windowSize
			}

			scores := make([]float32, t+1-startPos)
			for s := startPos; s <= t; s++ {
				kHead := k[s*nKVHead*headDim+kvH*headDim : s*nKVHead*headDim+(kvH+1)*headDim]
				scores[s-startPos] = VecDot(qHead, kHead) * scale
			}

			Softmax(scores)

			// Weighted sum of values
			outHead := attnOut[t*C+h*headDim : t*C+(h+1)*headDim]
			for s := startPos; s <= t; s++ {
				vHead := v[s*nKVHead*headDim+kvH*headDim : s*nKVHead*headDim+(kvH+1)*headDim]
				VecMulAdd(outHead, scores[s-startPos], vHead)
			}
		}
	}

	return attnOut
}

// attentionWithCache computes attention using the KV cache (for incremental decode).
func (m *GPT) attentionWithCache(layer int, q, k, v []float32, T int, kvc *KVCache) []float32 {
	nHead := m.Config.NHead
	nKVHead := m.Config.NKVHead
	headDim := m.Config.HeadDim()
	C := m.Config.NEmbd
	headsPerKVGroup := nHead / nKVHead
	scale := float32(1.0 / math.Sqrt(float64(headDim)))
	windowSize := m.WindowSizes[layer]

	// Store new K, V into the cache
	for t := 0; t < T; t++ {
		pos := kvc.Pos + t
		for h := 0; h < nKVHead; h++ {
			kSrc := k[t*nKVHead*headDim+h*headDim : t*nKVHead*headDim+(h+1)*headDim]
			vSrc := v[t*nKVHead*headDim+h*headDim : t*nKVHead*headDim+(h+1)*headDim]
			kDst := kvc.K[layer][pos*nKVHead*headDim+h*headDim : pos*nKVHead*headDim+(h+1)*headDim]
			vDst := kvc.V[layer][pos*nKVHead*headDim+h*headDim : pos*nKVHead*headDim+(h+1)*headDim]
			copy(kDst, kSrc)
			copy(vDst, vSrc)
		}
	}

	totalSeqLen := kvc.Pos + T // Total positions including new ones
	attnOut := make([]float32, T*C)

	for h := 0; h < nHead; h++ {
		kvH := h / headsPerKVGroup

		for t := 0; t < T; t++ {
			queryPos := kvc.Pos + t
			qHead := q[t*nHead*headDim+h*headDim : t*nHead*headDim+(h+1)*headDim]

			// Determine attention window (causal: can attend to positions <= queryPos)
			startPos := 0
			if windowSize > 0 && queryPos-windowSize+1 > 0 {
				startPos = queryPos - windowSize + 1
			}
			endPos := queryPos + 1 // exclusive, causal
			if endPos > totalSeqLen {
				endPos = totalSeqLen
			}

			nScores := endPos - startPos
			scores := make([]float32, nScores)

			for s := startPos; s < endPos; s++ {
				kHead := kvc.K[layer][s*nKVHead*headDim+kvH*headDim : s*nKVHead*headDim+(kvH+1)*headDim]
				scores[s-startPos] = VecDot(qHead, kHead) * scale
			}

			Softmax(scores)

			outHead := attnOut[t*C+h*headDim : t*C+(h+1)*headDim]
			for s := startPos; s < endPos; s++ {
				vHead := kvc.V[layer][s*nKVHead*headDim+kvH*headDim : s*nKVHead*headDim+(kvH+1)*headDim]
				VecMulAdd(outHead, scores[s-startPos], vHead)
			}
		}
	}

	return attnOut
}

// vecDotSmall computes dot product of weight[0:n] and input[0:n].
// Used for small gate computations (smear gate with 24 channels).
func vecDotSmall(weight, input []float32, n int) float32 {
	var sum float64
	for i := 0; i < n; i++ {
		sum += float64(weight[i]) * float64(input[i])
	}
	return float32(sum)
}

// NumParams returns total number of parameters.
func (m *GPT) NumParams() int {
	total := 0
	total += m.WTE.Size()
	total += m.LMHead.Size()
	total += len(m.ResidLambdas)
	total += len(m.X0Lambdas)
	total += m.SmearGate.Size()
	total += 1 // SmearLambda
	total += 1 // BackoutLambda
	for _, b := range m.Blocks {
		total += b.Attn.CQ.Size()
		total += b.Attn.CK.Size()
		total += b.Attn.CV.Size()
		total += b.Attn.CProj.Size()
		if b.Attn.VEGate != nil {
			total += b.Attn.VEGate.Size()
		}
		total += b.MLP.CFC.Size()
		total += b.MLP.CProj.Size()
	}
	for _, ve := range m.ValueEmbeds {
		total += ve.Size()
	}
	return total
}

// String returns a summary of the model configuration.
func (m *GPT) String() string {
	return fmt.Sprintf("GPT(layers=%d, heads=%d, kv_heads=%d, embd=%d, vocab=%d, seq=%d, params=%.1fM)",
		m.Config.NLayer, m.Config.NHead, m.Config.NKVHead, m.Config.NEmbd,
		m.Config.VocabSize, m.Config.SequenceLen,
		float64(m.NumParams())/1e6)
}
