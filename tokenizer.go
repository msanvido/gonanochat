package main

import (
	"encoding/hex"
	"encoding/json"
	"fmt"
	"os"
	"regexp"
	"sort"
	"strings"
	"unicode/utf8"
)

// Special token names used in nanochat conversation format.
const (
	TokenBOS            = "<|bos|>"
	TokenUserStart      = "<|user_start|>"
	TokenUserEnd        = "<|user_end|>"
	TokenAssistantStart = "<|assistant_start|>"
	TokenAssistantEnd   = "<|assistant_end|>"
	TokenPythonStart    = "<|python_start|>"
	TokenPythonEnd      = "<|python_end|>"
	TokenOutputStart    = "<|output_start|>"
	TokenOutputEnd      = "<|output_end|>"
)

// Tokenizer implements BPE tokenization compatible with tiktoken/rustbpe.
type Tokenizer struct {
	encoder       map[string]int    // byte sequence (as string of raw bytes) -> token ID
	decoder       map[int][]byte    // token ID -> byte sequence
	specialTokens map[string]int    // special token name -> token ID
	bpeRanks      map[string]int    // byte-pair (as concatenated strings) -> merge rank
	vocabSize     int
	pattern       *regexp.Regexp    // pre-tokenization split pattern (simplified RE2)
}

// TokenizerJSON is the JSON format exported by scripts/convert.py
type TokenizerJSON struct {
	VocabSize     int               `json:"vocab_size"`
	Pattern       string            `json:"pattern"`
	MergeableRanks map[string]int   `json:"mergeable_ranks"` // hex-encoded bytes -> rank
	SpecialTokens  map[string]int   `json:"special_tokens"`  // name -> token ID
}

// LoadTokenizer loads a tokenizer from a JSON file.
func LoadTokenizer(path string) (*Tokenizer, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("tokenizer: read %s: %w", path, err)
	}
	var tj TokenizerJSON
	if err := json.Unmarshal(data, &tj); err != nil {
		return nil, fmt.Errorf("tokenizer: parse %s: %w", path, err)
	}

	encoder := make(map[string]int, len(tj.MergeableRanks))
	decoder := make(map[int][]byte, len(tj.MergeableRanks)+len(tj.SpecialTokens))
	bpeRanks := make(map[string]int, len(tj.MergeableRanks))

	for hexBytes, rank := range tj.MergeableRanks {
		raw, err := hex.DecodeString(hexBytes)
		if err != nil {
			return nil, fmt.Errorf("tokenizer: decode hex %q: %w", hexBytes, err)
		}
		key := string(raw)
		encoder[key] = rank
		decoder[rank] = raw
		bpeRanks[key] = rank
	}

	for name, id := range tj.SpecialTokens {
		decoder[id] = []byte(name)
	}

	// Simplified pre-tokenization pattern compatible with Go RE2.
	// The original uses possessive quantifiers and lookahead which RE2 doesn't support.
	// This is a close approximation that works for most text.
	pat := regexp.MustCompile(`'(?i:[sdmt]|ll|ve|re)|[^\r\n\pL\pN]?\pL+|\pN{1,2}| ?[^\s\pL\pN]+[\r\n]*|\s*[\r\n]|\s+`)

	return &Tokenizer{
		encoder:       encoder,
		decoder:       decoder,
		specialTokens: tj.SpecialTokens,
		bpeRanks:      bpeRanks,
		vocabSize:     tj.VocabSize,
		pattern:       pat,
	}, nil
}

// VocabSize returns the total vocabulary size (regular + special tokens).
func (t *Tokenizer) VocabSize() int {
	return t.vocabSize
}

// EncodeSpecial returns the token ID for a named special token.
func (t *Tokenizer) EncodeSpecial(name string) int {
	id, ok := t.specialTokens[name]
	if !ok {
		panic(fmt.Sprintf("tokenizer: unknown special token %q", name))
	}
	return id
}

// BOSTokenID returns the BOS token ID.
func (t *Tokenizer) BOSTokenID() int {
	return t.EncodeSpecial(TokenBOS)
}

// Encode tokenizes a string into a list of token IDs.
func (t *Tokenizer) Encode(text string) []int {
	if text == "" {
		return nil
	}
	var tokens []int
	// Pre-tokenize: split text into words using regex
	words := t.pattern.FindAllString(text, -1)
	for _, word := range words {
		wordTokens := t.bpeEncode([]byte(word))
		tokens = append(tokens, wordTokens...)
	}
	return tokens
}

// Decode converts token IDs back to a string.
func (t *Tokenizer) Decode(ids []int) string {
	var buf []byte
	for _, id := range ids {
		if b, ok := t.decoder[id]; ok {
			buf = append(buf, b...)
		}
	}
	// Validate UTF-8; replace invalid bytes
	if utf8.Valid(buf) {
		return string(buf)
	}
	return strings.ToValidUTF8(string(buf), "\ufffd")
}

// bpeEncode applies BPE to a single word (as raw bytes).
func (t *Tokenizer) bpeEncode(word []byte) []int {
	if len(word) == 0 {
		return nil
	}

	// Check if the whole word is a single token
	if id, ok := t.encoder[string(word)]; ok {
		return []int{id}
	}

	// Start with individual bytes as tokens
	parts := make([][]byte, len(word))
	for i, b := range word {
		parts[i] = []byte{b}
	}

	// Iteratively merge the pair with the lowest rank
	for len(parts) >= 2 {
		// Find the pair with the lowest merge rank
		bestRank := -1
		bestIdx := -1
		for i := 0; i < len(parts)-1; i++ {
			merged := string(append(append([]byte{}, parts[i]...), parts[i+1]...))
			if rank, ok := t.encoder[merged]; ok {
				if bestIdx == -1 || rank < bestRank {
					bestRank = rank
					bestIdx = i
				}
			}
		}
		if bestIdx == -1 {
			break // No more merges possible
		}
		// Merge the best pair
		merged := append(append([]byte{}, parts[bestIdx]...), parts[bestIdx+1]...)
		newParts := make([][]byte, 0, len(parts)-1)
		newParts = append(newParts, parts[:bestIdx]...)
		newParts = append(newParts, merged)
		newParts = append(newParts, parts[bestIdx+2:]...)
		parts = newParts
	}

	// Convert byte sequences to token IDs
	ids := make([]int, len(parts))
	for i, part := range parts {
		if id, ok := t.encoder[string(part)]; ok {
			ids[i] = id
		} else {
			// Fallback: encode individual bytes (byte_fallback)
			// Single bytes should always be in the vocabulary (byte-level BPE)
			ids[i] = t.encoder[string(part)]
		}
	}
	return ids
}

// RenderConversation tokenizes a chat conversation into token IDs.
// Returns token IDs ready for model input.
func (t *Tokenizer) RenderConversation(messages []ChatMessage) []int {
	bos := t.BOSTokenID()
	userStart := t.EncodeSpecial(TokenUserStart)
	userEnd := t.EncodeSpecial(TokenUserEnd)
	assistantStart := t.EncodeSpecial(TokenAssistantStart)
	assistantEnd := t.EncodeSpecial(TokenAssistantEnd)

	tokens := []int{bos}
	for _, msg := range messages {
		switch msg.Role {
		case "user":
			tokens = append(tokens, userStart)
			tokens = append(tokens, t.Encode(msg.Content)...)
			tokens = append(tokens, userEnd)
		case "assistant":
			tokens = append(tokens, assistantStart)
			tokens = append(tokens, t.Encode(msg.Content)...)
			tokens = append(tokens, assistantEnd)
		}
	}
	// Prime for assistant completion
	tokens = append(tokens, assistantStart)
	return tokens
}

// ChatMessage represents a single message in a conversation.
type ChatMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// --- Tokenizer vocabulary export helpers (used by convert.py) ---

// VocabEntry is a token ID and its byte representation.
type VocabEntry struct {
	ID    int
	Bytes []byte
}

// SortedVocab returns the vocabulary sorted by token ID.
func (t *Tokenizer) SortedVocab() []VocabEntry {
	var entries []VocabEntry
	for bytes, id := range t.encoder {
		entries = append(entries, VocabEntry{id, []byte(bytes)})
	}
	sort.Slice(entries, func(i, j int) bool {
		return entries[i].ID < entries[j].ID
	})
	return entries
}
