package openrouter

import "github.com/flitsinc/go-llms/openai"

// Reasoning configures thinking/reasoning behavior for OpenRouter requests.
type Reasoning struct {
	Effort string `json:"effort,omitempty"` // "xhigh", "high", "medium", "low", "minimal", "none"
}

// New creates an OpenAI-compatible ChatCompletionsAPI configured for OpenRouter.
// Cache control is enabled by default so that CacheHint items are passed through
// to upstream providers (e.g. Anthropic) for prompt caching.
func New(apiKey, model string) *openai.ChatCompletionsAPI {
	return openai.New(apiKey, model).
		WithEndpoint("https://openrouter.ai/api/v1/chat/completions", "OpenRouter").
		WithCacheControl(true)
}

// NewWithReasoning creates an OpenRouter provider with reasoning/thinking enabled.
func NewWithReasoning(apiKey, model string, reasoning Reasoning) *openai.ChatCompletionsAPI {
	return New(apiKey, model).
		WithCustomPayloadValue("reasoning", reasoning)
}
