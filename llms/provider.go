package llms

import (
	"context"

	"github.com/blixt/go-llms/content"
	"github.com/blixt/go-llms/tools"
)

// TODO: The CostUSD() should go away because it's too complex to keep pricing accurate.
// TODO: The Usage() API should become detailed enough to track things like:
//       - Cached vs non-cached input tokens
//       - Reasoning vs non-reasoning output tokens
//       - Large inputs (>200K tokens) vs not
//       - Text/image/audio tokens

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
	Model() string
	// Generate takes a system prompt, message history, and optional toolbox,
	// returning a stream for the LLM's response. The provided context should
	// be respected for cancellation.
	Generate(ctx context.Context, systemPrompt content.Content, messages []Message, toolbox *tools.Toolbox) ProviderStream
}
