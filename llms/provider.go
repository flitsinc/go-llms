package llms

import (
	"context"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/tools"
)

// Usage represents token usage information from an LLM response.
type Usage struct {
	CachedInputTokens int
	InputTokens       int
	OutputTokens      int
}

type ProviderStream interface {
	Err() error
	Iter() func(yield func(StreamStatus) bool)
	Message() Message
	Text() string
	Thought() content.Thought
	ToolCall() ToolCall
	Usage() Usage
}

type Provider interface {
	Company() string
	Model() string
	// Generate takes a system prompt, message history, and optional toolbox,
	// returning a stream for the LLM's response. The provided context should
	// be respected for cancellation.
	Generate(
		ctx context.Context,
		systemPrompt content.Content,
		messages []Message,
		toolbox *tools.Toolbox,
		jsonOutputSchema *tools.ValueSchema,
	) ProviderStream
}
