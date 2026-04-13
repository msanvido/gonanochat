package main

import (
	"encoding/json"
	"fmt"
	"log"
	"math/rand/v2"
	"net/http"
	"strings"
	"sync"
	"time"
)

// Abuse prevention limits (matching Python)
const (
	maxMessagesPerRequest       = 500
	maxMessageLength            = 8000
	maxTotalConversationLength  = 32000
	minTemperature              = 0.0
	maxTemperatureLimit         = 2.0
	minTopK                     = 0
	maxTopKLimit                = 200
	minMaxTokens                = 1
	maxMaxTokensLimit           = 4096
)

// ServerConfig holds configuration for the web server.
type ServerConfig struct {
	Host        string
	Port        int
	ModelDir    string
	Temperature float32
	TopK        int
	MaxTokens   int
}

// Server is the web chat server.
type Server struct {
	config    ServerConfig
	engine    *Engine
	tokenizer *Tokenizer
	mu        sync.Mutex // serialize inference (single worker)
}

// ChatRequest is the request body for /chat/completions.
type ChatCompletionRequest struct {
	Messages    []ChatMessage `json:"messages"`
	Temperature *float32      `json:"temperature,omitempty"`
	MaxTokens   *int          `json:"max_tokens,omitempty"`
	TopK        *int          `json:"top_k,omitempty"`
}

// StartServer starts the HTTP server.
func StartServer(engine *Engine, tokenizer *Tokenizer, config ServerConfig) error {
	srv := &Server{
		config:    config,
		engine:    engine,
		tokenizer: tokenizer,
	}

	mux := http.NewServeMux()
	mux.HandleFunc("GET /", srv.handleRoot)
	mux.HandleFunc("POST /chat/completions", srv.handleChatCompletions)
	mux.HandleFunc("GET /health", srv.handleHealth)
	mux.HandleFunc("GET /stats", srv.handleStats)

	addr := fmt.Sprintf("%s:%d", config.Host, config.Port)
	log.Printf("Starting NanoChat server at http://%s", addr)
	log.Printf("  Temperature: %.1f, Top-k: %d, Max tokens: %d", config.Temperature, config.TopK, config.MaxTokens)
	return http.ListenAndServe(addr, mux)
}

func (s *Server) handleRoot(w http.ResponseWriter, r *http.Request) {
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Write([]byte(uiHTML))
}

func (s *Server) handleChatCompletions(w http.ResponseWriter, r *http.Request) {
	var req ChatCompletionRequest
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, `{"error": "invalid JSON"}`, http.StatusBadRequest)
		return
	}

	// Validate
	if err := validateChatRequest(req); err != "" {
		http.Error(w, fmt.Sprintf(`{"error": %q}`, err), http.StatusBadRequest)
		return
	}

	// Apply defaults
	temperature := s.config.Temperature
	if req.Temperature != nil {
		temperature = *req.Temperature
	}
	topK := s.config.TopK
	if req.TopK != nil {
		topK = *req.TopK
	}
	maxTokens := s.config.MaxTokens
	if req.MaxTokens != nil {
		maxTokens = *req.MaxTokens
	}

	// Log the conversation
	log.Println(strings.Repeat("=", 40))
	for _, msg := range req.Messages {
		log.Printf("[%s]: %s", strings.ToUpper(msg.Role), truncate(msg.Content, 200))
	}
	log.Println(strings.Repeat("-", 40))

	// Build conversation tokens
	tokens := s.tokenizer.RenderConversation(req.Messages)

	// Set up SSE streaming
	w.Header().Set("Content-Type", "text/event-stream")
	w.Header().Set("Cache-Control", "no-cache")
	w.Header().Set("Connection", "keep-alive")
	w.Header().Set("Access-Control-Allow-Origin", "*")

	flusher, ok := w.(http.Flusher)
	if !ok {
		http.Error(w, "streaming not supported", http.StatusInternalServerError)
		return
	}

	// Serialize access to the model
	s.mu.Lock()
	defer s.mu.Unlock()

	assistantEnd := s.tokenizer.EncodeSpecial(TokenAssistantEnd)
	bos := s.tokenizer.BOSTokenID()
	seed := rand.Uint64()

	// Token accumulation for proper UTF-8 handling
	var accumulatedTokens []int
	var lastCleanText string
	var fullResponse strings.Builder

	s.engine.Generate(tokens, maxTokens, temperature, topK, seed, func(gt GenerateToken) bool {
		// Check if client disconnected
		if r.Context().Err() != nil {
			return false
		}

		if gt.Token == assistantEnd || gt.Token == bos {
			return false
		}

		accumulatedTokens = append(accumulatedTokens, gt.Token)
		currentText := s.tokenizer.Decode(accumulatedTokens)

		// Only emit if no replacement character at end (incomplete UTF-8)
		if !strings.HasSuffix(currentText, "\ufffd") {
			newText := currentText[len(lastCleanText):]
			if newText != "" {
				data, _ := json.Marshal(map[string]interface{}{
					"token": newText,
				})
				fmt.Fprintf(w, "data: %s\n\n", data)
				flusher.Flush()
				fullResponse.WriteString(newText)
				lastCleanText = currentText
			}
		}
		return true
	})

	// Send done
	fmt.Fprintf(w, "data: %s\n\n", `{"done":true}`)
	flusher.Flush()

	log.Printf("[ASSISTANT]: %s", truncate(fullResponse.String(), 200))
	log.Println(strings.Repeat("=", 40))
}

func (s *Server) handleHealth(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"status": "ok",
		"ready":  true,
	})
}

func (s *Server) handleStats(w http.ResponseWriter, r *http.Request) {
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(map[string]interface{}{
		"model":   s.engine.Model.String(),
		"uptime":  time.Since(startTime).String(),
	})
}

var startTime = time.Now()

func validateChatRequest(req ChatCompletionRequest) string {
	if len(req.Messages) == 0 {
		return "at least one message is required"
	}
	if len(req.Messages) > maxMessagesPerRequest {
		return fmt.Sprintf("too many messages (max %d)", maxMessagesPerRequest)
	}

	totalLen := 0
	for i, msg := range req.Messages {
		if msg.Content == "" {
			return fmt.Sprintf("message %d has empty content", i)
		}
		if len(msg.Content) > maxMessageLength {
			return fmt.Sprintf("message %d too long (max %d chars)", i, maxMessageLength)
		}
		if msg.Role != "user" && msg.Role != "assistant" {
			return fmt.Sprintf("message %d has invalid role %q", i, msg.Role)
		}
		totalLen += len(msg.Content)
	}
	if totalLen > maxTotalConversationLength {
		return fmt.Sprintf("total conversation too long (max %d chars)", maxTotalConversationLength)
	}

	if req.Temperature != nil && (*req.Temperature < minTemperature || *req.Temperature > maxTemperatureLimit) {
		return fmt.Sprintf("temperature must be %.1f-%.1f", minTemperature, maxTemperatureLimit)
	}
	if req.TopK != nil && (*req.TopK < minTopK || *req.TopK > maxTopKLimit) {
		return fmt.Sprintf("top_k must be %d-%d", minTopK, maxTopKLimit)
	}
	if req.MaxTokens != nil && (*req.MaxTokens < minMaxTokens || *req.MaxTokens > maxMaxTokensLimit) {
		return fmt.Sprintf("max_tokens must be %d-%d", minMaxTokens, maxMaxTokensLimit)
	}

	return ""
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n] + "..."
	}
	return s
}

// uiHTML is the embedded web UI (see ui.go for the content).
// This is set by the ui.go file using go:embed or a string literal.
var uiHTML string
