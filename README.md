# gonanochat

A port of [nanochat](https://github.com/karpathy/nanochat) from Python/PyTorch to pure Go.

nanochat is Andrej Karpathy's minimal LLM training harness that covers the full pipeline: tokenization, pretraining, fine-tuning, evaluation, inference, and a chat UI. **gonanochat** reimplements this entire pipeline in ~4,300 lines of Go with zero CGo dependencies, compiling to a single static binary.

## What was ported

The original nanochat is built on Python and PyTorch. This Go port reimplements everything from scratch:

| Component | Python (nanochat) | Go (gonanochat) |
|---|---|---|
| **Tensor operations** | PyTorch | Hand-written matmul, RMSNorm, softmax, RoPE, etc. |
| **GPT model** | `nn.Module` subclasses | Struct with manual forward pass |
| **Backpropagation** | PyTorch autograd | Hand-derived gradients for every operation |
| **Optimizer** | Custom Muon + AdamW | AdamW with bias correction |
| **KV cache inference** | Flash Attention 3 / SDPA | Manual attention with KV cache |
| **BPE tokenizer** | tiktoken / rustbpe | Pure Go BPE encoder/decoder |
| **Web server** | FastAPI + uvicorn | `net/http` with SSE streaming |
| **CLI chat** | Python readline | Go `bufio.Scanner` |
| **Model format** | PyTorch `.pt` (pickle) | Custom binary format |

All model features are faithfully preserved:

- **Grouped-Query Attention** (GQA) for efficient inference
- **Rotary Position Embeddings** (RoPE)
- **QK normalization** with 1.2x sharpening
- **Sliding window attention** (configurable per-layer pattern)
- **Value embeddings** (ResFormer-style, alternating layers)
- **Smear gate** (bigram mixing from previous token)
- **Backout lambda** (subtract mid-layer residual before output)
- **ReLU² activation** in MLP
- **Logit softcap** (15.0 tanh capping)
- **Tool use** (Python expression calculator during generation)

## Getting started

### Prerequisites

- Go 1.23+
- Python 3.10+ with nanochat installed (only for data preparation and model conversion)

### Build

```bash
go build -o gonanochat .
```

This produces a single ~9MB static binary with no external dependencies.

### Train a model from scratch

**Step 1: Download training data.** Grab a text corpus (here we use *The Adventures of Sherlock Holmes* from Project Gutenberg):

```bash
curl -sL "https://www.gutenberg.org/cache/epub/1661/pg1661.txt" > /tmp/sherlock.txt
```

**Step 2: Tokenize.** Convert the text into binary token format using the nanochat tokenizer, splitting 5% for validation:

```bash
python scripts/prepare_data.py \
  -i /tmp/sherlock.txt \
  -o /tmp/train.bin \
  --val /tmp/val.bin \
  --val-fraction 0.05
```

**Step 3: Train.** Run the Go training loop:

```bash
./gonanochat train \
  -data /tmp/train.bin \
  -val /tmp/val.bin \
  -depth 4 \
  -vocab 32768 \
  -seq 256 \
  -batch 4 \
  -lr 1e-3 \
  -steps 5000 \
  -save-dir my_model
```

The `depth` flag is the single complexity dial (matching nanochat): it sets the number of transformer layers, and all other hyperparameters (width, heads, etc.) are computed automatically.

### Use a model trained in Python

If you already have a nanochat model trained with PyTorch, convert it:

```bash
python scripts/convert.py \
  --source sft \
  -o model_export
```

This exports `model_export/model.bin` and `model_export/tokenizer.json`.

### Chat with a model

**Interactive CLI:**

```bash
./gonanochat chat -m model_export
```

**Single prompt:**

```bash
./gonanochat chat -m model_export -p "Why is the sky blue?"
```

**Web server** (serves a chat UI at `http://localhost:8000`):

```bash
./gonanochat serve -m model_export
```

The web server exposes an OpenAI-compatible `/chat/completions` endpoint with SSE streaming, plus a built-in chat UI.

## Commands

```
gonanochat train   Train a model from scratch
gonanochat chat    Interactive CLI chat or single prompt
gonanochat serve   Start the web server with chat UI
gonanochat help    Show help
```

### Train flags

| Flag | Default | Description |
|------|---------|-------------|
| `-data` | (required) | Pre-tokenized training data (`.bin`) |
| `-val` | | Validation data (`.bin`) |
| `-depth` | 4 | Model depth (number of layers) |
| `-vocab` | 32768 | Vocabulary size |
| `-seq` | 256 | Sequence length |
| `-batch` | 4 | Batch size |
| `-lr` | 3e-4 | Peak learning rate |
| `-steps` | 5000 | Total training steps |
| `-warmup` | 100 | Warmup steps |
| `-save-dir` | checkpoints | Checkpoint directory |
| `-resume` | | Resume from model directory |

### Chat/Serve flags

| Flag | Default | Description |
|------|---------|-------------|
| `-m` | model_export | Model directory |
| `-t` | 0.6 / 0.8 | Temperature |
| `-k` | 50 | Top-k sampling |
| `-n` | 256 / 512 | Max tokens |
| `-p` | | Prompt (chat only, non-interactive) |
| `-port` | 8000 | Port (serve only) |

## Project structure

```
gonanochat/
├── main.go            Entry point with subcommands
├── tensor.go          Tensor type, matmul, RMSNorm, softmax, RoPE, sampling, PRNG
├── model.go           GPT model: config, attention, MLP, forward pass
├── engine.go          KV cache, streaming generation, tool use
├── tokenizer.go       BPE tokenizer (encode/decode, special tokens)
├── checkpoint.go      Binary model format loader
├── backward.go        Gradient functions for all operations
├── train_model.go     Trainable GPT with forward cache and backward pass
├── optim.go           AdamW optimizer, LR schedule
├── train.go           Training loop, data loading, checkpointing
├── server.go          HTTP server with SSE streaming
├── cli.go             Interactive terminal chat
├── ui.go              Embedded web UI (HTML/CSS/JS)
└── scripts/
    ├── convert.py     Convert PyTorch checkpoints to Go format
    └── prepare_data.py Tokenize text into binary training data
```

## Design notes

**Why Go?** A single static binary with no runtime dependencies makes deployment trivial. The inference engine can run on any machine without Python, CUDA, or PyTorch.

**Pure Go, no CGo.** All tensor operations are implemented in plain Go. This means no BLAS, no GPU - everything runs on CPU. For small models (depth 2-8), inference is fast enough for interactive use. Training is slower than PyTorch but fully functional.

**Manual backpropagation.** Since Go has no autograd framework, every gradient is hand-derived and implemented. The backward pass mirrors the forward pass in reverse, computing gradients for: RMSNorm, linear projections, ReLU², rotary embeddings, scaled dot-product attention (with causal masking), softmax cross-entropy, tanh softcap, and the smear gate.

**Binary model format.** A simple custom format: 4-byte magic (`NANO`), config as JSON, then named tensors (name + shape + float32 data). Easy to read in any language.

## Limitations

- **CPU only.** No GPU support. Training large models is impractical; inference for models up to ~100M params is reasonable.
- **No distributed training.** Single-process only (nanochat supports multi-GPU DDP).
- **AdamW only.** The Python version uses a Muon + AdamW mixed optimizer which trains faster.
- **No Flash Attention.** Uses naive O(T²) attention. Fine for short sequences, slow for long ones.
- **No torch.compile equivalent.** No JIT or kernel fusion optimizations.

## License

Same as nanochat (MIT).
