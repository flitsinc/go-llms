package llms

import (
	"context"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/tools"
)

// Usage represents token usage information from an LLM response.
type Usage struct {
	CachedInputTokens        int // Number of cached input tokens (cache reads)
	CacheCreationInputTokens int // Number of input tokens used to create cache (cache writes)
	InputTokens              int // Number of input tokens
	OutputTokens             int // Number of output tokens
}

func (u *Usage) Add(other Usage) {
	u.CachedInputTokens += other.CachedInputTokens
	u.CacheCreationInputTokens += other.CacheCreationInputTokens
	u.InputTokens += other.InputTokens
	u.OutputTokens += other.OutputTokens
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
