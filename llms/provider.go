package llms

import (
	"context"

	"github.com/blixt/go-llms/content"
	"github.com/blixt/go-llms/tools"
)

type ProviderStream interface {
	Err() error
	Iter() func(yield func(StreamStatus) bool)
	Message() Message
	Text() string
	ToolCall() ToolCall
	CostUSD() float64
	Usage() (inputTokens, outputTokens int)
}

type Provider interface {
	Company() string
	// Generate takes a system prompt, message history, and optional toolbox,
	// returning a stream for the LLM's response. The provided context should
	// be respected for cancellation.
	Generate(ctx context.Context, systemPrompt content.Content, messages []Message, toolbox *tools.Toolbox) ProviderStream
}
