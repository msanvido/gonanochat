package main

import "math"

// --- Backward pass primitives ---
// Each function computes gradients for a specific operation.
// Naming convention: <Op>Backward takes the upstream gradient (dy)
// and cached forward values, returns downstream gradient (dx)
// and accumulates parameter gradients where applicable.

// RMSNormBackward computes gradient through y = x / rms(x).
// dy, x, y are all length n. Returns dx of length n.
func RMSNormBackward(dy, x, y []float32) []float32 {
	n := len(x)
	// Recompute rms
	var sumSq float64
	for i := 0; i < n; i++ {
		sumSq += float64(x[i]) * float64(x[i])
	}
	rms := float32(math.Sqrt(sumSq/float64(n) + 1e-6))
	invRms := 1.0 / rms

	// dx = (dy - y * mean(dy * y)) / rms
	var dotDyY float64
	for i := 0; i < n; i++ {
		dotDyY += float64(dy[i]) * float64(y[i])
	}
	meanDotDyY := float32(dotDyY / float64(n))

	dx := make([]float32, n)
	for i := 0; i < n; i++ {
		dx[i] = (dy[i] - y[i]*meanDotDyY) * invRms
	}
	return dx
}

// LinearBackward computes gradients for y = W @ x (matrix-vector product).
// W is (outDim, inDim), x is (inDim,), dy is (outDim,).
// Returns dx (inDim,) and accumulates dW (outDim, inDim).
func LinearBackward(dy []float32, W *Tensor, x []float32, dW *Tensor) []float32 {
	outDim := W.Shape[0]
	inDim := W.Shape[1]

	// dx = W.T @ dy
	dx := make([]float32, inDim)
	for j := 0; j < inDim; j++ {
		var sum float64
		for i := 0; i < outDim; i++ {
			sum += float64(W.Data[i*inDim+j]) * float64(dy[i])
		}
		dx[j] = float32(sum)
	}

	// dW += outer(dy, x)
	for i := 0; i < outDim; i++ {
		dyi := dy[i]
		off := i * inDim
		for j := 0; j < inDim; j++ {
			dW.Data[off+j] += dyi * x[j]
		}
	}

	return dx
}

// EmbeddingBackward accumulates gradient for embedding lookup.
// dy is (C,), idx is the token index, dWeight is (vocabSize, C).
func EmbeddingBackward(dy []float32, idx int, dWeight *Tensor) {
	C := dWeight.Shape[1]
	off := idx * C
	for i := 0; i < C; i++ {
		dWeight.Data[off+i] += dy[i]
	}
}

// ReLUSquaredBackward computes gradient through relu(x)^2.
// dy is the upstream gradient, xPre is the pre-activation value.
// d/dx[relu(x)^2] = 2*relu(x) = 2*max(0,x)
func ReLUSquaredBackward(dy, xPre []float32) []float32 {
	dx := make([]float32, len(dy))
	for i, v := range xPre {
		if v > 0 {
			dx[i] = dy[i] * 2 * v
		}
		// else: dx[i] = 0 (already zero)
	}
	return dx
}

// TanhSoftcapBackward computes gradient through y = cap * tanh(x / cap).
// dy/dx = 1 - (y/cap)^2 = 1 - tanh(x/cap)^2
func TanhSoftcapBackward(dy, y []float32, cap float32) []float32 {
	dx := make([]float32, len(dy))
	invCap := 1.0 / cap
	for i, yi := range y {
		t := yi * invCap // tanh(x/cap) = y/cap
		dx[i] = dy[i] * (1 - t*t)
	}
	return dx
}

// SoftmaxCrossEntropyGrad computes the gradient of cross-entropy loss w.r.t. logits.
// logits is (vocabSize,), target is the correct token index.
// Returns d_logits = softmax(logits) - one_hot(target), already divided by nothing
// (the loss is per-token, not summed).
func SoftmaxCrossEntropyGrad(logits []float32, target int) ([]float32, float32) {
	n := len(logits)

	// Compute softmax
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	probs := make([]float32, n)
	var sum float64
	for i, v := range logits {
		e := math.Exp(float64(v - maxVal))
		probs[i] = float32(e)
		sum += e
	}
	invSum := float32(1.0 / sum)
	for i := range probs {
		probs[i] *= invSum
	}

	// Loss = -log(probs[target])
	loss := float32(-math.Log(float64(probs[target]) + 1e-10))

	// Gradient: probs - one_hot(target)
	grad := make([]float32, n)
	copy(grad, probs)
	grad[target] -= 1.0

	return grad, loss
}

// RotaryEmbBackward computes gradient through rotary embedding.
// The rotary embedding is an orthogonal transformation, so the backward
// is just the inverse (transpose) rotation.
// dy has length headDim, cos/sin have length headDim/2.
func RotaryEmbBackward(dy []float32, cos, sin []float32) []float32 {
	d := len(dy) / 2
	dx := make([]float32, len(dy))
	// Forward:  y[:d] = x[:d]*cos + x[d:]*sin
	//           y[d:] = x[:d]*(-sin) + x[d:]*cos
	// Backward (inverse rotation):
	//           dx[:d] = dy[:d]*cos + dy[d:]*(-sin)
	//           dx[d:] = dy[:d]*sin + dy[d:]*cos
	for i := 0; i < d; i++ {
		dx[i] = dy[i]*cos[i] + dy[d+i]*(-sin[i])
		dx[d+i] = dy[i]*sin[i] + dy[d+i]*cos[i]
	}
	return dx
}

// ScaleBackward computes gradient through y = alpha * x.
// Returns dx = alpha * dy.
func ScaleBackward(dy []float32, alpha float32) []float32 {
	dx := make([]float32, len(dy))
	for i, v := range dy {
		dx[i] = alpha * v
	}
	return dx
}

// AttentionBackward computes gradients through scaled dot-product attention.
// Q: (nHead, headDim), K: (seqLen, nKVHead, headDim), V: (seqLen, nKVHead, headDim)
// attnWeights: (nHead, seqLen) - softmax weights
// dy: (nHead*headDim,) - upstream gradient of attention output
//
// For a single query position attending to seqLen key positions (causal attention).
// Returns dQ (nHead*headDim,), and accumulates into dK, dV.
func AttentionBackward(
	dy []float32,
	Q []float32, // (nHead * headDim,) - query for this position
	K []float32, // stored keys up to this position: flat (seqLen * nKVHead * headDim)
	V []float32, // stored values: flat (seqLen * nKVHead * headDim)
	attnWeights []float32, // (nHead * seqLen) - attention weights
	nHead, nKVHead, headDim, seqLen int,
	scale float32,
	dK []float32, // accumulate: (seqLen * nKVHead * headDim)
	dV []float32, // accumulate: (seqLen * nKVHead * headDim)
) []float32 {
	headsPerGroup := nHead / nKVHead
	dQ := make([]float32, nHead*headDim)

	for h := 0; h < nHead; h++ {
		kvH := h / headsPerGroup
		qOff := h * headDim
		wOff := h * seqLen
		dyHead := dy[qOff : qOff+headDim]

		// dV: dV[s, kvH] += attnWeights[h, s] * dy[h]
		for s := 0; s < seqLen; s++ {
			w := attnWeights[wOff+s]
			vOff := s*nKVHead*headDim + kvH*headDim
			for d := 0; d < headDim; d++ {
				dV[vOff+d] += w * dyHead[d]
			}
		}

		// d_attn = dy @ V[s].T -> (seqLen,) per head
		dAttn := make([]float32, seqLen)
		for s := 0; s < seqLen; s++ {
			vOff := s*nKVHead*headDim + kvH*headDim
			var dot float64
			for d := 0; d < headDim; d++ {
				dot += float64(dyHead[d]) * float64(V[vOff+d])
			}
			dAttn[s] = float32(dot)
		}

		// Softmax backward: d_scores = attn * (d_attn - sum(d_attn * attn))
		var sumDA float64
		for s := 0; s < seqLen; s++ {
			sumDA += float64(dAttn[s]) * float64(attnWeights[wOff+s])
		}
		dScores := make([]float32, seqLen)
		for s := 0; s < seqLen; s++ {
			dScores[s] = attnWeights[wOff+s] * (dAttn[s] - float32(sumDA))
		}

		// Apply scale
		for s := 0; s < seqLen; s++ {
			dScores[s] *= scale
		}

		// dQ[h] = sum_s(dScores[s] * K[s, kvH])
		for s := 0; s < seqLen; s++ {
			ds := dScores[s]
			kOff := s*nKVHead*headDim + kvH*headDim
			for d := 0; d < headDim; d++ {
				dQ[qOff+d] += ds * K[kOff+d]
			}
		}

		// dK[s, kvH] += dScores[s] * Q[h]
		for s := 0; s < seqLen; s++ {
			ds := dScores[s]
			kOff := s*nKVHead*headDim + kvH*headDim
			for d := 0; d < headDim; d++ {
				dK[kOff+d] += ds * Q[qOff+d]
			}
		}
	}

	return dQ
}

// SmearGateBackward computes gradients through the smear gate.
// Forward: gate = smear_lambda * sigmoid(w @ x[:channels])
//          y = x + gate * x_prev
// Given dy, returns dx, dx_prev, and accumulates dW, dSmearLambda.
func SmearGateBackward(
	dy []float32, // (C,)
	x []float32, // (C,) current position (pre-smear, after norm)
	xPrev []float32, // (C,) previous position embedding
	gateVal float32, // computed gate value
	smearLambda float32,
	wData []float32, // smear gate weight (channels,)
	channels int,
	dW []float32, // accumulate gradient for smear gate weight
	dSmearLambda *float64, // accumulate gradient for smear_lambda
) (dx, dxPrev []float32) {
	C := len(dy)
	dx = make([]float32, C)
	dxPrev = make([]float32, C)

	// y = x + gate * x_prev
	// dy/dx = I + d(gate)/dx * x_prev  (gate depends on x[:channels])
	// dy/dx_prev = gate * I
	// dy/d(gate) = x_prev

	// d_gate from residual contribution
	var dGate float64
	for i := 0; i < C; i++ {
		dGate += float64(dy[i]) * float64(xPrev[i])
	}

	// gate = smear_lambda * sigmoid(dot(w, x[:channels]))
	// Let z = dot(w, x[:channels])
	// gate = smear_lambda * sigmoid(z)
	// d_gate/d_smear_lambda = sigmoid(z)
	// d_gate/dz = smear_lambda * sigmoid(z) * (1 - sigmoid(z))

	var z float64
	for i := 0; i < channels; i++ {
		z += float64(wData[i]) * float64(x[i])
	}
	sigZ := 1.0 / (1.0 + math.Exp(-z))
	sigZf := float32(sigZ)

	// Accumulate d_smear_lambda
	*dSmearLambda += dGate * sigZ

	// dz = d_gate * smear_lambda * sigmoid(z) * (1 - sigmoid(z))
	dz := float32(dGate) * smearLambda * sigZf * (1 - sigZf)

	// dW += dz * x[:channels]
	for i := 0; i < channels; i++ {
		dW[i] += dz * x[i]
	}

	// dx[:channels] += dz * w
	// Also dx = dy (pass-through from y = x + ...)
	copy(dx, dy)
	for i := 0; i < channels; i++ {
		dx[i] += dz * wData[i]
	}

	// dxPrev = dy * gate
	for i := 0; i < C; i++ {
		dxPrev[i] = dy[i] * gateVal
	}

	return dx, dxPrev
}
