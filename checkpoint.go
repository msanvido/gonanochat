package main

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"path/filepath"
)

// Binary model file format:
//   Magic:      "NANO" (4 bytes)
//   Version:    uint32 (4 bytes)
//   ConfigLen:  uint32 (4 bytes)
//   ConfigJSON: bytes (ConfigLen bytes)
//   NumTensors: uint32 (4 bytes)
//   For each tensor:
//     NameLen:  uint32 (4 bytes)
//     Name:    bytes (NameLen bytes)
//     NDim:    uint32 (4 bytes)
//     Shape:   uint32[NDim] (NDim * 4 bytes)
//     Data:    float32[] (product(Shape) * 4 bytes, little-endian)

const modelMagic = "NANO"
const modelVersion = 1

// LoadModel loads a model and tokenizer from a directory containing:
//   - model.bin (binary weights)
//   - tokenizer.json (BPE vocabulary)
func LoadModel(modelDir string) (*GPT, *Tokenizer, error) {
	// Load tokenizer
	tokPath := filepath.Join(modelDir, "tokenizer.json")
	tokenizer, err := LoadTokenizer(tokPath)
	if err != nil {
		return nil, nil, fmt.Errorf("load tokenizer: %w", err)
	}

	// Load model
	modelPath := filepath.Join(modelDir, "model.bin")
	model, err := LoadModelBinary(modelPath)
	if err != nil {
		return nil, nil, fmt.Errorf("load model: %w", err)
	}

	// Sanity check
	if tokenizer.VocabSize() != model.Config.VocabSize {
		return nil, nil, fmt.Errorf("vocab size mismatch: tokenizer=%d, model=%d",
			tokenizer.VocabSize(), model.Config.VocabSize)
	}

	return model, tokenizer, nil
}

// LoadModelBinary loads model weights from the binary format.
func LoadModelBinary(path string) (*GPT, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()

	// Read magic
	magic := make([]byte, 4)
	if _, err := io.ReadFull(f, magic); err != nil {
		return nil, fmt.Errorf("read magic: %w", err)
	}
	if string(magic) != modelMagic {
		return nil, fmt.Errorf("invalid magic: %q (expected %q)", magic, modelMagic)
	}

	// Read version
	var version uint32
	if err := binary.Read(f, binary.LittleEndian, &version); err != nil {
		return nil, fmt.Errorf("read version: %w", err)
	}
	if version != modelVersion {
		return nil, fmt.Errorf("unsupported version: %d (expected %d)", version, modelVersion)
	}

	// Read config
	var configLen uint32
	if err := binary.Read(f, binary.LittleEndian, &configLen); err != nil {
		return nil, fmt.Errorf("read config length: %w", err)
	}
	configBytes := make([]byte, configLen)
	if _, err := io.ReadFull(f, configBytes); err != nil {
		return nil, fmt.Errorf("read config: %w", err)
	}
	var config GPTConfig
	if err := json.Unmarshal(configBytes, &config); err != nil {
		return nil, fmt.Errorf("parse config: %w", err)
	}

	// Create model
	model := NewGPT(config)

	// Read tensors
	var numTensors uint32
	if err := binary.Read(f, binary.LittleEndian, &numTensors); err != nil {
		return nil, fmt.Errorf("read num_tensors: %w", err)
	}

	tensors := make(map[string]*Tensor, numTensors)
	for i := uint32(0); i < numTensors; i++ {
		name, tensor, err := readTensor(f)
		if err != nil {
			return nil, fmt.Errorf("read tensor %d: %w", i, err)
		}
		tensors[name] = tensor
	}

	// Assign tensors to model
	if err := assignWeights(model, tensors); err != nil {
		return nil, err
	}

	return model, nil
}

func readTensor(r io.Reader) (string, *Tensor, error) {
	// Read name
	var nameLen uint32
	if err := binary.Read(r, binary.LittleEndian, &nameLen); err != nil {
		return "", nil, fmt.Errorf("read name length: %w", err)
	}
	nameBytes := make([]byte, nameLen)
	if _, err := io.ReadFull(r, nameBytes); err != nil {
		return "", nil, fmt.Errorf("read name: %w", err)
	}
	name := string(nameBytes)

	// Read shape
	var ndim uint32
	if err := binary.Read(r, binary.LittleEndian, &ndim); err != nil {
		return "", nil, fmt.Errorf("read ndim: %w", err)
	}
	shape := make([]int, ndim)
	size := 1
	for d := uint32(0); d < ndim; d++ {
		var dim uint32
		if err := binary.Read(r, binary.LittleEndian, &dim); err != nil {
			return "", nil, fmt.Errorf("read dim %d: %w", d, err)
		}
		shape[d] = int(dim)
		size *= int(dim)
	}

	// Read data
	data := make([]float32, size)
	if err := binary.Read(r, binary.LittleEndian, data); err != nil {
		return "", nil, fmt.Errorf("read data for %q (%d floats): %w", name, size, err)
	}

	return name, &Tensor{Data: data, Shape: shape}, nil
}

func assignWeights(model *GPT, tensors map[string]*Tensor) error {
	get := func(name string) (*Tensor, error) {
		t, ok := tensors[name]
		if !ok {
			return nil, fmt.Errorf("missing tensor: %q", name)
		}
		return t, nil
	}

	var err error

	// WTE
	model.WTE, err = get("transformer.wte.weight")
	if err != nil {
		return err
	}

	// LMHead
	model.LMHead, err = get("lm_head.weight")
	if err != nil {
		return err
	}

	// Per-layer scalars
	rl, err := get("resid_lambdas")
	if err != nil {
		return err
	}
	model.ResidLambdas = rl.Data

	xl, err := get("x0_lambdas")
	if err != nil {
		return err
	}
	model.X0Lambdas = xl.Data

	// Smear gate
	model.SmearGate, err = get("smear_gate.weight")
	if err != nil {
		return err
	}

	sl, err := get("smear_lambda")
	if err != nil {
		return err
	}
	model.SmearLambda = sl.Data[0]

	bl, err := get("backout_lambda")
	if err != nil {
		return err
	}
	model.BackoutLambda = bl.Data[0]

	// Transformer blocks
	nLayer := model.Config.NLayer
	for i := 0; i < nLayer; i++ {
		prefix := fmt.Sprintf("transformer.h.%d", i)

		model.Blocks[i].Attn.CQ, err = get(prefix + ".attn.c_q.weight")
		if err != nil {
			return err
		}
		model.Blocks[i].Attn.CK, err = get(prefix + ".attn.c_k.weight")
		if err != nil {
			return err
		}
		model.Blocks[i].Attn.CV, err = get(prefix + ".attn.c_v.weight")
		if err != nil {
			return err
		}
		model.Blocks[i].Attn.CProj, err = get(prefix + ".attn.c_proj.weight")
		if err != nil {
			return err
		}
		model.Blocks[i].MLP.CFC, err = get(prefix + ".mlp.c_fc.weight")
		if err != nil {
			return err
		}
		model.Blocks[i].MLP.CProj, err = get(prefix + ".mlp.c_proj.weight")
		if err != nil {
			return err
		}

		// VE gate (optional, only alternating layers)
		if hasVE(i, nLayer) {
			veGateName := prefix + ".attn.ve_gate.weight"
			if t, ok := tensors[veGateName]; ok {
				model.Blocks[i].Attn.VEGate = t
			}
		}
	}

	// Value embeddings
	for i := 0; i < nLayer; i++ {
		if hasVE(i, nLayer) {
			name := fmt.Sprintf("value_embeds.%d.weight", i)
			if t, ok := tensors[name]; ok {
				model.ValueEmbeds[i] = t
			}
		}
	}

	return nil
}
