package main

import (
	"encoding/binary"
	"fmt"
	"io"
	"log"
	"math"
	"math/rand/v2"
	"os"
	"time"
)

// TrainConfig holds training hyperparameters.
type TrainConfig struct {
	// Data
	DataPath    string // path to pre-tokenized binary data (uint16 tokens)
	ValDataPath string // path to validation data (optional)

	// Model (if training from scratch)
	Depth     int // number of layers
	VocabSize int // vocabulary size (must match tokenizer)

	// Training
	SeqLen       int     // sequence length
	BatchSize    int     // number of sequences per batch
	LR           float32 // peak learning rate
	MinLR        float32 // minimum LR (end of cosine decay)
	WarmupSteps  int     // warmup steps
	TotalSteps   int     // total training steps
	WeightDecay  float32
	Beta1        float32
	Beta2        float32

	// Eval/Save
	EvalInterval int    // evaluate every N steps
	SaveInterval int    // save checkpoint every N steps
	SaveDir      string // checkpoint save directory

	// Resume
	ResumeFrom   string // resume from this model directory
}

// DefaultTrainConfig returns sensible defaults for small model training.
func DefaultTrainConfig() TrainConfig {
	return TrainConfig{
		Depth:        4,
		VocabSize:    32768,
		SeqLen:       256,
		BatchSize:    4,
		LR:           3e-4,
		MinLR:        1e-5,
		WarmupSteps:  100,
		TotalSteps:   5000,
		WeightDecay:  0.01,
		Beta1:        0.9,
		Beta2:        0.999,
		EvalInterval: 100,
		SaveInterval: 500,
		SaveDir:      "checkpoints",
	}
}

// TokenReader reads pre-tokenized data from a binary file (uint16 or uint32 tokens).
type TokenReader struct {
	data []int
	rng  *rand.Rand
}

// NewTokenReader loads tokenized data from a binary file.
// File format: 4-byte header ("TK" + dtype byte + pad) followed by token data.
func NewTokenReader(path string) (*TokenReader, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open %s: %w", path, err)
	}
	defer f.Close()

	info, err := f.Stat()
	if err != nil {
		return nil, err
	}
	fileSize := info.Size()

	// Read 4-byte header: "TK" + dtype (2 or 4) + padding
	header := make([]byte, 4)
	if _, err := io.ReadFull(f, header); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}
	if string(header[:2]) != "TK" {
		return nil, fmt.Errorf("invalid token file (missing 'TK' magic). Re-run prepare_data.py to regenerate")
	}
	dtype := int(header[2]) // 2 = uint16, 4 = uint32
	dataSize := fileSize - 4

	var data []int
	switch dtype {
	case 2:
		numTokens := int(dataSize / 2)
		raw := make([]uint16, numTokens)
		if err := binary.Read(f, binary.LittleEndian, raw); err != nil {
			return nil, fmt.Errorf("read uint16 tokens: %w", err)
		}
		data = make([]int, numTokens)
		for i, v := range raw {
			data[i] = int(v)
		}
	case 4:
		numTokens := int(dataSize / 4)
		raw := make([]uint32, numTokens)
		if err := binary.Read(f, binary.LittleEndian, raw); err != nil {
			return nil, fmt.Errorf("read uint32 tokens: %w", err)
		}
		data = make([]int, numTokens)
		for i, v := range raw {
			data[i] = int(v)
		}
	default:
		return nil, fmt.Errorf("unsupported token dtype: %d (expected 2 or 4)", dtype)
	}

	log.Printf("Loaded %d tokens (uint%d, %.1f MB) from %s", len(data), dtype*8, float64(fileSize)/1e6, path)
	return &TokenReader{data: data, rng: rand.New(rand.NewPCG(42, 0))}, nil
}

// GetBatch returns a batch of token sequences, each of length seqLen+1 (for targets).
func (tr *TokenReader) GetBatch(batchSize, seqLen int) [][]int {
	maxStart := len(tr.data) - seqLen - 1
	batch := make([][]int, batchSize)
	for b := 0; b < batchSize; b++ {
		start := tr.rng.IntN(maxStart)
		seq := make([]int, seqLen+1)
		copy(seq, tr.data[start:start+seqLen+1])
		batch[b] = seq
	}
	return batch
}

// Train runs the training loop.
func Train(cfg TrainConfig) {
	log.Println("=== NanoChat Training (Go) ===")
	log.Printf("Config: depth=%d, seq_len=%d, batch=%d, steps=%d, lr=%.1e",
		cfg.Depth, cfg.SeqLen, cfg.BatchSize, cfg.TotalSteps, cfg.LR)

	// Load data
	trainData, err := NewTokenReader(cfg.DataPath)
	if err != nil {
		log.Fatalf("Failed to load training data: %v", err)
	}
	var valData *TokenReader
	if cfg.ValDataPath != "" {
		valData, err = NewTokenReader(cfg.ValDataPath)
		if err != nil {
			log.Fatalf("Failed to load validation data: %v", err)
		}
	}

	// Create or load model
	var model *GPT
	if cfg.ResumeFrom != "" {
		log.Printf("Resuming from %s", cfg.ResumeFrom)
		var loadErr error
		model, _, loadErr = LoadModel(cfg.ResumeFrom)
		if loadErr != nil {
			log.Fatalf("Failed to load model: %v", loadErr)
		}
	} else {
		// Create model from scratch with compute-optimal config
		config := makeConfig(cfg.Depth, cfg.VocabSize, cfg.SeqLen)
		model = NewGPT(config)
		initWeights(model)
		log.Printf("Created %s", model)
	}

	// Create trainable wrapper and optimizer
	tg := NewTrainableGPT(model)
	opt := NewAdamW(cfg.LR, cfg.Beta1, cfg.Beta2, 1e-10, cfg.WeightDecay)

	// Build parameter map for optimizer
	paramMap := buildParamMap(model)

	// Create save directory
	os.MkdirAll(cfg.SaveDir, 0755)

	// Training loop
	log.Println("Starting training...")
	var totalTokens int64
	startTime := time.Now()

	for step := 0; step < cfg.TotalSteps; step++ {
		stepStart := time.Now()

		// Update learning rate
		lr := LRSchedule(step, cfg.WarmupSteps, cfg.TotalSteps, cfg.LR, cfg.MinLR)
		opt.LR = lr

		// Get batch
		batch := trainData.GetBatch(cfg.BatchSize, cfg.SeqLen)

		// Forward + backward for each sequence (accumulate gradients)
		tg.ZeroGrad()
		var batchLoss float64
		for _, seq := range batch {
			loss := tg.ForwardTrain(seq)
			tg.Backward()
			batchLoss += float64(loss)
		}
		batchLoss /= float64(cfg.BatchSize)

		// Apply scalar gradients
		applyScalarGrads(tg, paramMap)

		// Optimizer step
		opt.Step(paramMap, tg.Grads)

		// Sync scalar params back to model
		syncScalarsToModel(model, paramMap)

		totalTokens += int64(cfg.BatchSize * cfg.SeqLen)
		stepTime := time.Since(stepStart)
		tokPerSec := float64(cfg.BatchSize*cfg.SeqLen) / stepTime.Seconds()

		// Log
		if step%10 == 0 || step == cfg.TotalSteps-1 {
			elapsed := time.Since(startTime)
			log.Printf("step %5d/%d | loss %.4f | lr %.2e | %.0f tok/s | %s elapsed",
				step, cfg.TotalSteps, batchLoss, lr, tokPerSec, elapsed.Round(time.Second))
		}

		// Eval
		if valData != nil && cfg.EvalInterval > 0 && (step%cfg.EvalInterval == 0 || step == cfg.TotalSteps-1) {
			valLoss := evaluate(tg, valData, cfg.SeqLen, 10)
			log.Printf("  val_loss: %.4f", valLoss)
		}

		// Save
		if cfg.SaveInterval > 0 && ((step+1)%cfg.SaveInterval == 0 || step == cfg.TotalSteps-1) {
			savePath := fmt.Sprintf("%s/step_%06d", cfg.SaveDir, step+1)
			saveTrainingCheckpoint(model, savePath)
			log.Printf("  Saved checkpoint to %s", savePath)
		}
	}

	totalTime := time.Since(startTime)
	log.Printf("Training complete! %d steps, %d tokens, %.1f minutes",
		cfg.TotalSteps, totalTokens, totalTime.Minutes())
}

// makeConfig creates a GPTConfig from depth with compute-optimal settings.
func makeConfig(depth, vocabSize, seqLen int) GPTConfig {
	aspectRatio := 64
	nEmbd := depth * aspectRatio
	headDim := 64
	if nEmbd < 64 {
		headDim = nEmbd
	}
	nHead := nEmbd / headDim
	nKVHead := nHead // full attention (no GQA for small models)

	return GPTConfig{
		SequenceLen:   seqLen,
		VocabSize:     vocabSize,
		NLayer:        depth,
		NHead:         nHead,
		NKVHead:       nKVHead,
		NEmbd:         nEmbd,
		WindowPattern: "L", // full context for small models
	}
}

// initWeights initializes model weights (matching Python init_weights).
func initWeights(m *GPT) {
	rng := rand.New(rand.NewPCG(42, 0))
	normalFill := func(data []float32, std float64) {
		for i := range data {
			data[i] = float32(rng.NormFloat64() * std)
		}
	}
	uniformFill := func(data []float32, low, high float64) {
		for i := range data {
			data[i] = float32(low + rng.Float64()*(high-low))
		}
	}

	// Allocate all weight tensors
	paddedVocab := m.PaddedVocabSize
	C := m.Config.NEmbd
	nLayer := m.Config.NLayer
	nKVHead := m.Config.NKVHead
	headDim := m.Config.HeadDim()
	kvDim := nKVHead * headDim

	m.WTE = NewTensor(paddedVocab, C)
	normalFill(m.WTE.Data, 0.8)

	m.LMHead = NewTensor(paddedVocab, C)
	normalFill(m.LMHead.Data, 0.001)

	m.ResidLambdas = make([]float32, nLayer)
	m.X0Lambdas = make([]float32, nLayer)
	for i := 0; i < nLayer; i++ {
		m.ResidLambdas[i] = 1.15 - 0.10*float32(i)/float32(max(nLayer-1, 1))
		m.X0Lambdas[i] = 0.20 - 0.15*float32(i)/float32(max(nLayer-1, 1))
	}

	m.SmearGate = NewTensor(1, 24)
	m.SmearLambda = 0
	m.BackoutLambda = 0.2

	s := math.Sqrt(3.0) * math.Pow(float64(C), -0.5)
	for i := 0; i < nLayer; i++ {
		b := &m.Blocks[i]
		b.Attn.CQ = NewTensor(C, C)
		uniformFill(b.Attn.CQ.Data, -s, s)
		b.Attn.CK = NewTensor(kvDim, C)
		uniformFill(b.Attn.CK.Data, -s, s)
		b.Attn.CV = NewTensor(kvDim, C)
		uniformFill(b.Attn.CV.Data, -s, s)
		b.Attn.CProj = NewTensor(C, C)
		// zeros for projection
		b.MLP.CFC = NewTensor(4*C, C)
		uniformFill(b.MLP.CFC.Data, -s*0.4, s*0.4)
		b.MLP.CProj = NewTensor(C, 4*C)
		// zeros for projection

		if hasVE(i, nLayer) {
			b.Attn.VEGate = NewTensor(nKVHead, 12)
			uniformFill(b.Attn.VEGate.Data, 0.0, 0.02)
			m.ValueEmbeds[i] = NewTensor(paddedVocab, kvDim)
			uniformFill(m.ValueEmbeds[i].Data, -s, s)
		}
	}
}

// buildParamMap creates a flat map of parameter name -> data slice for the optimizer.
func buildParamMap(m *GPT) map[string][]float32 {
	pm := make(map[string][]float32)
	pm["transformer.wte.weight"] = m.WTE.Data
	pm["lm_head.weight"] = m.LMHead.Data
	pm["resid_lambdas"] = m.ResidLambdas
	pm["x0_lambdas"] = m.X0Lambdas
	pm["smear_gate.weight"] = m.SmearGate.Data

	for i := 0; i < m.Config.NLayer; i++ {
		p := fmt.Sprintf("transformer.h.%d", i)
		pm[p+".attn.c_q.weight"] = m.Blocks[i].Attn.CQ.Data
		pm[p+".attn.c_k.weight"] = m.Blocks[i].Attn.CK.Data
		pm[p+".attn.c_v.weight"] = m.Blocks[i].Attn.CV.Data
		pm[p+".attn.c_proj.weight"] = m.Blocks[i].Attn.CProj.Data
		pm[p+".mlp.c_fc.weight"] = m.Blocks[i].MLP.CFC.Data
		pm[p+".mlp.c_proj.weight"] = m.Blocks[i].MLP.CProj.Data
		if m.Blocks[i].Attn.VEGate != nil {
			pm[p+".attn.ve_gate.weight"] = m.Blocks[i].Attn.VEGate.Data
		}
	}
	for idx, ve := range m.ValueEmbeds {
		pm[fmt.Sprintf("value_embeds.%d.weight", idx)] = ve.Data
	}

	// Scalar params stored as single-element slices
	pm["smear_lambda"] = []float32{m.SmearLambda}
	pm["backout_lambda"] = []float32{m.BackoutLambda}
	return pm
}

// applyScalarGrads converts accumulated scalar gradients into the grad tensors.
func applyScalarGrads(tg *TrainableGPT, pm map[string][]float32) {
	// SmearLambda and BackoutLambda don't have Tensor grads, so we handle them
	// by creating temporary gradient entries.
	if _, ok := tg.Grads["smear_lambda"]; !ok {
		tg.Grads["smear_lambda"] = NewTensor(1)
	}
	if _, ok := tg.Grads["backout_lambda"]; !ok {
		tg.Grads["backout_lambda"] = NewTensor(1)
	}
	tg.Grads["smear_lambda"].Data[0] = float32(tg.GradSmearLambda)
	tg.Grads["backout_lambda"].Data[0] = float32(tg.GradBackoutLambda)
}

// syncScalarsToModel copies scalar param values back to model fields.
func syncScalarsToModel(m *GPT, pm map[string][]float32) {
	m.SmearLambda = pm["smear_lambda"][0]
	m.BackoutLambda = pm["backout_lambda"][0]
}

// evaluate computes mean loss on validation data.
func evaluate(tg *TrainableGPT, valData *TokenReader, seqLen, numBatches int) float32 {
	var totalLoss float64
	for i := 0; i < numBatches; i++ {
		batch := valData.GetBatch(1, seqLen)
		loss := tg.ForwardTrain(batch[0])
		totalLoss += float64(loss)
	}
	return float32(totalLoss / float64(numBatches))
}

// saveTrainingCheckpoint saves the model in the Go binary format.
func saveTrainingCheckpoint(model *GPT, dir string) {
	os.MkdirAll(dir, 0755)
	path := dir + "/model.bin"

	f, err := os.Create(path)
	if err != nil {
		log.Printf("Failed to create %s: %v", path, err)
		return
	}
	defer f.Close()

	// Write header
	f.Write([]byte(modelMagic))
	binary.Write(f, binary.LittleEndian, uint32(modelVersion))

	// Config as JSON
	configJSON := fmt.Sprintf(`{"sequence_len":%d,"vocab_size":%d,"n_layer":%d,"n_head":%d,"n_kv_head":%d,"n_embd":%d,"window_pattern":"%s"}`,
		model.Config.SequenceLen, model.Config.VocabSize, model.Config.NLayer,
		model.Config.NHead, model.Config.NKVHead, model.Config.NEmbd, model.Config.WindowPattern)
	binary.Write(f, binary.LittleEndian, uint32(len(configJSON)))
	f.Write([]byte(configJSON))

	// Collect tensors
	tensors := make(map[string]*Tensor)
	tensors["transformer.wte.weight"] = model.WTE
	tensors["lm_head.weight"] = model.LMHead
	tensors["resid_lambdas"] = TensorFrom(model.ResidLambdas, len(model.ResidLambdas))
	tensors["x0_lambdas"] = TensorFrom(model.X0Lambdas, len(model.X0Lambdas))
	tensors["smear_gate.weight"] = model.SmearGate
	tensors["smear_lambda"] = TensorFrom([]float32{model.SmearLambda}, 1)
	tensors["backout_lambda"] = TensorFrom([]float32{model.BackoutLambda}, 1)

	for i := 0; i < model.Config.NLayer; i++ {
		p := fmt.Sprintf("transformer.h.%d", i)
		tensors[p+".attn.c_q.weight"] = model.Blocks[i].Attn.CQ
		tensors[p+".attn.c_k.weight"] = model.Blocks[i].Attn.CK
		tensors[p+".attn.c_v.weight"] = model.Blocks[i].Attn.CV
		tensors[p+".attn.c_proj.weight"] = model.Blocks[i].Attn.CProj
		tensors[p+".mlp.c_fc.weight"] = model.Blocks[i].MLP.CFC
		tensors[p+".mlp.c_proj.weight"] = model.Blocks[i].MLP.CProj
		if model.Blocks[i].Attn.VEGate != nil {
			tensors[p+".attn.ve_gate.weight"] = model.Blocks[i].Attn.VEGate
		}
	}
	for idx, ve := range model.ValueEmbeds {
		tensors[fmt.Sprintf("value_embeds.%d.weight", idx)] = ve
	}

	binary.Write(f, binary.LittleEndian, uint32(len(tensors)))
	for name, t := range tensors {
		writeTensor(f, name, t)
	}
}

func writeTensor(w io.Writer, name string, t *Tensor) {
	nameBytes := []byte(name)
	binary.Write(w, binary.LittleEndian, uint32(len(nameBytes)))
	w.Write(nameBytes)
	binary.Write(w, binary.LittleEndian, uint32(len(t.Shape)))
	for _, dim := range t.Shape {
		binary.Write(w, binary.LittleEndian, uint32(dim))
	}
	binary.Write(w, binary.LittleEndian, t.Data)
}

func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
