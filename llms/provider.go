package llms

import (
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
	Generate(systemPrompt content.Content, messages []Message, toolbox *tools.Toolbox) ProviderStream
}
