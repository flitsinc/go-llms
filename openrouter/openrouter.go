package openrouter

import "github.com/flitsinc/go-llms/openai"

// Provider is an OpenRouter-configured Chat Completions client.
type Provider = openai.ChatCompletionsAPI

// Reasoning configures OpenRouter's top-level reasoning request parameter.
type Reasoning struct {
	Effort string `json:"effort,omitempty"` // "xhigh", "high", "medium", "low", "minimal", "none"
}

// New returns a Chat Completions client configured for OpenRouter.
func New(apiKey, model string) *Provider {
	return openai.New(apiKey, model).
		WithEndpoint("https://openrouter.ai/api/v1/chat/completions", "OpenRouter").
		WithCacheControlPromptHints().
		WithAssistantReasoningReplay()
}

// NewWithReasoning returns an OpenRouter-configured Chat Completions client
// with the top-level "reasoning" request parameter set.
func NewWithReasoning(apiKey, model string, reasoning Reasoning) *Provider {
	return New(apiKey, model).
		WithCustomPayloadValue("reasoning", reasoning)
}
