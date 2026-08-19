package openrouter

import (
	"strings"

	"github.com/flitsinc/go-llms/openai"
)

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
	provider := openai.New(apiKey, model).
		WithEndpoint("https://openrouter.ai/api/v1/chat/completions", "OpenRouter").
		WithCacheControlPromptHints().
		WithAssistantReasoningReplay().
		WithFlatCustomTools()
	if isGeminiModel(model) {
		provider = provider.WithGeminiToolSchemas()
	}
	return provider
}

// isGeminiModel reports whether an OpenRouter model id is served by Google's
// Gemini API upstream, which accepts a narrower tool schema than the rest of
// OpenRouter's catalogue.
//
// The check is on the model id rather than a caller-supplied option on purpose:
// a Gemini row that forgets to opt in does not degrade, it fails every
// tool-calling request with a 400, so the safe behaviour has to be the one you
// get by default. Only "google/gemini*" qualifies — Google's other listings
// (Gemma) are served by third-party endpoints that take the full schema.
func isGeminiModel(model string) bool {
	return strings.HasPrefix(model, "google/gemini")
}

// NewWithReasoning returns an OpenRouter-configured Chat Completions client
// with the top-level "reasoning" request parameter set.
func NewWithReasoning(apiKey, model string, reasoning Reasoning) *Provider {
	return New(apiKey, model).
		WithCustomPayloadValue("reasoning", reasoning)
}
