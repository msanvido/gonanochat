package main

import (
	"flag"
	"fmt"
	"log"
	"os"
)

func main() {
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	command := os.Args[1]
	args := os.Args[2:]

	switch command {
	case "chat":
		runChat(args)
	case "serve":
		runServe(args)
	case "train":
		runTrain(args)
	case "help", "-h", "--help":
		printUsage()
	default:
		fmt.Fprintf(os.Stderr, "Unknown command: %s\n\n", command)
		printUsage()
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`gonanochat - NanoChat in Go (training + inference)

Usage:
  gonanochat <command> [options]

Commands:
  train   Train a model from scratch
  chat    Interactive CLI chat
  serve   Start the web server
  help    Show this help

Train a model:
  python scripts/prepare_data.py -i data.txt -o train.bin
  gonanochat train -data train.bin -depth 4 -steps 5000

Inference (from exported Python model or Go-trained model):
  gonanochat chat -m model_export
  gonanochat serve -m model_export`)
}

func runChat(args []string) {
	fs := flag.NewFlagSet("chat", flag.ExitOnError)
	modelDir := fs.String("m", "model_export", "Path to exported model directory")
	temperature := fs.Float64("t", 0.6, "Temperature for generation")
	topK := fs.Int("k", 50, "Top-k sampling parameter")
	maxTokens := fs.Int("n", 256, "Max tokens to generate")
	prompt := fs.String("p", "", "Single prompt (non-interactive mode)")
	fs.Parse(args)

	model, tokenizer, err := LoadModel(*modelDir)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Printf("Loaded %s\n", model)

	engine := NewEngine(model, tokenizer)

	if *prompt != "" {
		RunPrompt(engine, tokenizer, *prompt, float32(*temperature), *topK, *maxTokens)
	} else {
		RunCLI(engine, tokenizer, float32(*temperature), *topK, *maxTokens)
	}
}

func runServe(args []string) {
	fs := flag.NewFlagSet("serve", flag.ExitOnError)
	modelDir := fs.String("m", "model_export", "Path to exported model directory")
	host := fs.String("host", "0.0.0.0", "Host to bind to")
	port := fs.Int("port", 8000, "Port to listen on")
	temperature := fs.Float64("t", 0.8, "Default temperature")
	topK := fs.Int("k", 50, "Default top-k")
	maxTokens := fs.Int("n", 512, "Default max tokens")
	fs.Parse(args)

	model, tokenizer, err := LoadModel(*modelDir)
	if err != nil {
		log.Fatalf("Failed to load model: %v", err)
	}
	fmt.Printf("Loaded %s\n", model)

	engine := NewEngine(model, tokenizer)

	config := ServerConfig{
		Host:        *host,
		Port:        *port,
		ModelDir:    *modelDir,
		Temperature: float32(*temperature),
		TopK:        *topK,
		MaxTokens:   *maxTokens,
	}

	if err := StartServer(engine, tokenizer, config); err != nil {
		log.Fatalf("Server error: %v", err)
	}
}

func runTrain(args []string) {
	fs := flag.NewFlagSet("train", flag.ExitOnError)
	dataPath := fs.String("data", "", "Path to pre-tokenized training data (.bin)")
	valPath := fs.String("val", "", "Path to pre-tokenized validation data (.bin)")
	depth := fs.Int("depth", 4, "Model depth (number of layers)")
	vocabSize := fs.Int("vocab", 32768, "Vocabulary size (must match tokenizer)")
	seqLen := fs.Int("seq", 256, "Sequence length")
	batchSize := fs.Int("batch", 4, "Batch size")
	lr := fs.Float64("lr", 3e-4, "Peak learning rate")
	steps := fs.Int("steps", 5000, "Total training steps")
	warmup := fs.Int("warmup", 100, "Warmup steps")
	evalInterval := fs.Int("eval-interval", 100, "Evaluate every N steps (0 to disable)")
	saveInterval := fs.Int("save-interval", 500, "Save every N steps (0 to disable)")
	saveDir := fs.String("save-dir", "checkpoints", "Checkpoint save directory")
	resume := fs.String("resume", "", "Resume from model directory")
	fs.Parse(args)

	if *dataPath == "" {
		fmt.Fprintln(os.Stderr, "Error: -data is required")
		fmt.Fprintln(os.Stderr, "\nPrepare data with: python scripts/prepare_data.py -i text.txt -o train.bin")
		fs.Usage()
		os.Exit(1)
	}

	cfg := TrainConfig{
		DataPath:     *dataPath,
		ValDataPath:  *valPath,
		Depth:        *depth,
		VocabSize:    *vocabSize,
		SeqLen:       *seqLen,
		BatchSize:    *batchSize,
		LR:           float32(*lr),
		MinLR:        float32(*lr) * 0.1,
		WarmupSteps:  *warmup,
		TotalSteps:   *steps,
		WeightDecay:  0.01,
		Beta1:        0.9,
		Beta2:        0.999,
		EvalInterval: *evalInterval,
		SaveInterval: *saveInterval,
		SaveDir:      *saveDir,
		ResumeFrom:   *resume,
	}

	Train(cfg)
}
