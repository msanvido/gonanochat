package main

import (
	"bufio"
	"fmt"
	"math/rand/v2"
	"os"
	"strings"
)

// RunCLI starts an interactive chat session in the terminal.
func RunCLI(engine *Engine, tokenizer *Tokenizer, temperature float32, topK int, maxTokens int) {
	bos := tokenizer.BOSTokenID()
	userStart := tokenizer.EncodeSpecial(TokenUserStart)
	userEnd := tokenizer.EncodeSpecial(TokenUserEnd)
	assistantStart := tokenizer.EncodeSpecial(TokenAssistantStart)
	assistantEnd := tokenizer.EncodeSpecial(TokenAssistantEnd)

	fmt.Println()
	fmt.Println("NanoChat Interactive Mode (Go)")
	fmt.Println(strings.Repeat("-", 50))
	fmt.Println("Type 'quit' or 'exit' to end the conversation")
	fmt.Println("Type 'clear' to start a new conversation")
	fmt.Println(strings.Repeat("-", 50))

	conversationTokens := []int{bos}
	scanner := bufio.NewScanner(os.Stdin)
	scanner.Buffer(make([]byte, 1024*1024), 1024*1024) // 1MB buffer for long inputs

	for {
		fmt.Print("\nUser: ")
		if !scanner.Scan() {
			fmt.Println("\nGoodbye!")
			break
		}
		userInput := strings.TrimSpace(scanner.Text())

		// Handle special commands
		switch strings.ToLower(userInput) {
		case "quit", "exit":
			fmt.Println("Goodbye!")
			return
		case "clear":
			conversationTokens = []int{bos}
			fmt.Println("Conversation cleared.")
			continue
		case "":
			continue
		}

		// Encode user message
		conversationTokens = append(conversationTokens, userStart)
		conversationTokens = append(conversationTokens, tokenizer.Encode(userInput)...)
		conversationTokens = append(conversationTokens, userEnd)

		// Prime assistant
		conversationTokens = append(conversationTokens, assistantStart)

		// Generate response
		fmt.Print("\nAssistant: ")
		var responseTokens []int
		seed := rand.Uint64()

		engine.Generate(conversationTokens, maxTokens, temperature, topK, seed, func(gt GenerateToken) bool {
			responseTokens = append(responseTokens, gt.Token)
			// Decode and print incrementally
			text := tokenizer.Decode([]int{gt.Token})
			fmt.Print(text)
			return true
		})
		fmt.Println()

		// Ensure assistant_end is the last token
		if len(responseTokens) == 0 || responseTokens[len(responseTokens)-1] != assistantEnd {
			responseTokens = append(responseTokens, assistantEnd)
		}
		conversationTokens = append(conversationTokens, responseTokens...)
	}
}

// RunPrompt runs a single prompt and prints the response.
func RunPrompt(engine *Engine, tokenizer *Tokenizer, prompt string, temperature float32, topK int, maxTokens int) {
	tokens := tokenizer.RenderConversation([]ChatMessage{
		{Role: "user", Content: prompt},
	})

	seed := rand.Uint64()
	engine.Generate(tokens, maxTokens, temperature, topK, seed, func(gt GenerateToken) bool {
		assistantEnd := tokenizer.EncodeSpecial(TokenAssistantEnd)
		bos := tokenizer.BOSTokenID()
		if gt.Token == assistantEnd || gt.Token == bos {
			return false
		}
		text := tokenizer.Decode([]int{gt.Token})
		fmt.Print(text)
		return true
	})
	fmt.Println()
}
