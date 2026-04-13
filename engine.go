package main

import (
	"fmt"
	"math"
	"strings"
)

// KVCache stores key/value tensors for all layers during inference.
type KVCache struct {
	// K[layer] is a flat slice of shape (max_seq_len, n_kv_head, head_dim)
	K [][]float32
	// V[layer] is a flat slice of shape (max_seq_len, n_kv_head, head_dim)
	V [][]float32

	Pos       int       // Current sequence position
	MaxSeqLen int
	NLayers   int
	NKVHead   int
	HeadDim   int

	// Previous token's normalized embedding for smear (set by model forward pass)
	PrevEmbedding []float32
}

// NewKVCache creates a new KV cache.
func NewKVCache(config GPTConfig, maxSeqLen int) *KVCache {
	nKVHead := config.NKVHead
	headDim := config.HeadDim()
	nLayers := config.NLayer
	entrySize := maxSeqLen * nKVHead * headDim

	kvc := &KVCache{
		K:         make([][]float32, nLayers),
		V:         make([][]float32, nLayers),
		Pos:       0,
		MaxSeqLen: maxSeqLen,
		NLayers:   nLayers,
		NKVHead:   nKVHead,
		HeadDim:   headDim,
	}
	for i := 0; i < nLayers; i++ {
		kvc.K[i] = make([]float32, entrySize)
		kvc.V[i] = make([]float32, entrySize)
	}
	return kvc
}

// Reset clears the cache.
func (kvc *KVCache) Reset() {
	kvc.Pos = 0
	kvc.PrevEmbedding = nil
	for i := range kvc.K {
		for j := range kvc.K[i] {
			kvc.K[i][j] = 0
		}
		for j := range kvc.V[i] {
			kvc.V[i][j] = 0
		}
	}
}

// RowState tracks per-sample generation state for tool use.
type RowState struct {
	Tokens          []int
	ForcedTokens    []int
	InPythonBlock   bool
	PythonExprTokens []int
	Completed       bool
}

// Engine orchestrates model inference with KV caching and tool use.
type Engine struct {
	Model     *GPT
	Tokenizer *Tokenizer
}

// NewEngine creates a new inference engine.
func NewEngine(model *GPT, tokenizer *Tokenizer) *Engine {
	return &Engine{Model: model, Tokenizer: tokenizer}
}

// GenerateToken is a single step result from the generator.
type GenerateToken struct {
	Token  int
	Forced bool // true if this token was force-injected (tool output)
}

// Generate runs streaming token generation with tool use support.
// Yields tokens one at a time through the callback function.
// The callback returns false to stop generation.
func (e *Engine) Generate(tokens []int, maxTokens int, temperature float32, topK int, seed uint64, callback func(GenerateToken) bool) {
	config := e.Model.Config
	maxSeqLen := len(tokens) + maxTokens
	if maxSeqLen > config.SequenceLen*10 {
		maxSeqLen = config.SequenceLen * 10
	}

	kvCache := NewKVCache(config, maxSeqLen)
	rng := NewRNG(seed)

	// Special tokens
	assistantEnd := e.Tokenizer.EncodeSpecial(TokenAssistantEnd)
	bos := e.Tokenizer.BOSTokenID()
	pythonStart := e.Tokenizer.EncodeSpecial(TokenPythonStart)
	pythonEnd := e.Tokenizer.EncodeSpecial(TokenPythonEnd)
	outputStart := e.Tokenizer.EncodeSpecial(TokenOutputStart)
	outputEnd := e.Tokenizer.EncodeSpecial(TokenOutputEnd)

	state := &RowState{Tokens: append([]int{}, tokens...)}

	// Prefill: process all prompt tokens at once
	logits := e.Model.Forward(tokens, kvCache)

	for i := 0; i < maxTokens; i++ {
		if state.Completed {
			break
		}

		var token int
		var forced bool

		if len(state.ForcedTokens) > 0 {
			// Pop from forced token queue
			token = state.ForcedTokens[0]
			state.ForcedTokens = state.ForcedTokens[1:]
			forced = true
		} else {
			// Sample from logits
			token = TopKSample(logits, topK, temperature, rng)
			forced = false
		}

		state.Tokens = append(state.Tokens, token)

		// Check for end of generation
		if token == assistantEnd || token == bos {
			state.Completed = true
		}

		// Tool use state machine
		if token == pythonStart {
			state.InPythonBlock = true
			state.PythonExprTokens = nil
		} else if token == pythonEnd && state.InPythonBlock {
			state.InPythonBlock = false
			if len(state.PythonExprTokens) > 0 {
				expr := e.Tokenizer.Decode(state.PythonExprTokens)
				result := UseCalculator(expr)
				if result != "" {
					resultTokens := e.Tokenizer.Encode(result)
					state.ForcedTokens = append(state.ForcedTokens, outputStart)
					state.ForcedTokens = append(state.ForcedTokens, resultTokens...)
					state.ForcedTokens = append(state.ForcedTokens, outputEnd)
				}
			}
			state.PythonExprTokens = nil
		} else if state.InPythonBlock {
			state.PythonExprTokens = append(state.PythonExprTokens, token)
		}

		// Yield the token
		if !callback(GenerateToken{Token: token, Forced: forced}) {
			break
		}

		// Compute logits for next step
		logits = e.Model.Forward([]int{token}, kvCache)
	}
}

// GenerateBatch generates a complete response (non-streaming).
// Returns the generated tokens (excluding prompt and terminal tokens).
func (e *Engine) GenerateBatch(tokens []int, maxTokens int, temperature float32, topK int, seed uint64) []int {
	assistantEnd := e.Tokenizer.EncodeSpecial(TokenAssistantEnd)
	bos := e.Tokenizer.BOSTokenID()

	var result []int
	e.Generate(tokens, maxTokens, temperature, topK, seed, func(gt GenerateToken) bool {
		if gt.Token != assistantEnd && gt.Token != bos {
			result = append(result, gt.Token)
		}
		return true
	})
	return result
}

// --- Calculator tool ---

// UseCalculator evaluates a simple math expression or string.count() call.
// Returns the result as a string, or empty string on failure.
func UseCalculator(expr string) string {
	expr = strings.ReplaceAll(expr, ",", "")
	expr = strings.TrimSpace(expr)

	if expr == "" {
		return ""
	}

	// Try pure math expression
	if isMathExpr(expr) {
		if strings.Contains(expr, "**") {
			return "" // disallow power operator
		}
		result, ok := evalMathExpr(expr)
		if ok {
			return formatNumber(result)
		}
		return ""
	}

	// Try string.count() operations
	if strings.Contains(expr, ".count(") {
		result, ok := evalStringCount(expr)
		if ok {
			return fmt.Sprintf("%d", result)
		}
	}

	return ""
}

func isMathExpr(expr string) bool {
	for _, c := range expr {
		if !strings.ContainsRune("0123456789*+-/.() ", c) {
			return false
		}
	}
	return true
}

// evalMathExpr evaluates a simple arithmetic expression using a recursive descent parser.
func evalMathExpr(expr string) (float64, bool) {
	p := &mathParser{input: expr, pos: 0}
	result := p.parseExpr()
	p.skipSpaces()
	if p.pos != len(expr) {
		return 0, false
	}
	if math.IsInf(result, 0) || math.IsNaN(result) {
		return 0, false
	}
	return result, true
}

type mathParser struct {
	input string
	pos   int
}

func (p *mathParser) skipSpaces() {
	for p.pos < len(p.input) && p.input[p.pos] == ' ' {
		p.pos++
	}
}

func (p *mathParser) parseExpr() float64 {
	result := p.parseTerm()
	for {
		p.skipSpaces()
		if p.pos >= len(p.input) {
			break
		}
		op := p.input[p.pos]
		if op != '+' && op != '-' {
			break
		}
		p.pos++
		right := p.parseTerm()
		if op == '+' {
			result += right
		} else {
			result -= right
		}
	}
	return result
}

func (p *mathParser) parseTerm() float64 {
	result := p.parseFactor()
	for {
		p.skipSpaces()
		if p.pos >= len(p.input) {
			break
		}
		op := p.input[p.pos]
		if op != '*' && op != '/' {
			break
		}
		p.pos++
		right := p.parseFactor()
		if op == '*' {
			result *= right
		} else {
			result /= right
		}
	}
	return result
}

func (p *mathParser) parseFactor() float64 {
	p.skipSpaces()
	if p.pos >= len(p.input) {
		return 0
	}

	// Handle unary minus
	if p.input[p.pos] == '-' {
		p.pos++
		return -p.parseFactor()
	}

	// Handle parentheses
	if p.input[p.pos] == '(' {
		p.pos++
		result := p.parseExpr()
		p.skipSpaces()
		if p.pos < len(p.input) && p.input[p.pos] == ')' {
			p.pos++
		}
		return result
	}

	// Parse number
	start := p.pos
	for p.pos < len(p.input) && (p.input[p.pos] >= '0' && p.input[p.pos] <= '9' || p.input[p.pos] == '.') {
		p.pos++
	}
	if start == p.pos {
		return 0
	}
	var val float64
	fmt.Sscanf(p.input[start:p.pos], "%f", &val)
	return val
}

// evalStringCount handles expressions like "hello".count("l")
func evalStringCount(expr string) (int, bool) {
	// Simple parser for "string".count("substr")
	expr = strings.TrimSpace(expr)

	// Find the string and the count argument
	dotCount := strings.Index(expr, ".count(")
	if dotCount < 0 {
		return 0, false
	}

	strPart := strings.TrimSpace(expr[:dotCount])
	argPart := strings.TrimSpace(expr[dotCount+7:]) // skip ".count("

	// Remove trailing )
	if !strings.HasSuffix(argPart, ")") {
		return 0, false
	}
	argPart = argPart[:len(argPart)-1]

	// Extract string values (remove quotes)
	s := unquote(strPart)
	sub := unquote(argPart)
	if s == "" && sub == "" {
		return 0, false
	}

	return strings.Count(s, sub), true
}

func unquote(s string) string {
	s = strings.TrimSpace(s)
	if len(s) >= 2 {
		if (s[0] == '\'' && s[len(s)-1] == '\'') || (s[0] == '"' && s[len(s)-1] == '"') {
			return s[1 : len(s)-1]
		}
	}
	return s
}

func formatNumber(f float64) string {
	if f == math.Trunc(f) && !math.IsInf(f, 0) {
		return fmt.Sprintf("%.0f", f)
	}
	return fmt.Sprintf("%g", f)
}
