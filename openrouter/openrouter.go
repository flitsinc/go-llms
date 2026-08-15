package openrouter

import "github.com/flitsinc/go-llms/openai"

// Provider is an OpenRouter-configured Chat Completions client.
type Provider = openai.ChatCompletionsAPI

// Reasoning configures OpenRouter's top-level reasoning request parameter.
type Reasoning struct {
	Effort string `json:"effort,omitempty"` // "xhigh", "high", "medium", "low", "minimal", "none"
}

// New returns a Chat Completions client configured for OpenRouter.
// Custom (grammar/text) tools use the flat Responses-style declaration
// because OpenRouter forwards the tools array into a Responses API request
// upstream, where the nested Chat Completions wrapper is rejected.
func New(apiKey, model string) *Provider {
	return openai.New(apiKey, model).
		WithEndpoint("https://openrouter.ai/api/v1/chat/completions", "OpenRouter").
		WithCacheControlPromptHints().
		WithAssistantReasoningReplay().
		WithFlatCustomTools()
}

// NewWithReasoning returns an OpenRouter-configured Chat Completions client
// with the top-level "reasoning" request parameter set.
func NewWithReasoning(apiKey, model string, reasoning Reasoning) *Provider {
	return New(apiKey, model).
		WithCustomPayloadValue("reasoning", reasoning)
}
