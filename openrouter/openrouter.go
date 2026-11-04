package openrouter

import (
	"strings"

	"github.com/flitsinc/go-llms/anthropic"
	"github.com/flitsinc/go-llms/google"
	"github.com/flitsinc/go-llms/llms"
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
// upstream, where the nested Chat Completions wrapper is rejected. Requests are
// checked against the image limits of the model's vendor (see ImageLimitsFor).
func New(apiKey, model string) *Provider {
	return openai.New(apiKey, model).
		WithEndpoint("https://openrouter.ai/api/v1/chat/completions", "OpenRouter").
		WithCacheControlPromptHints().
		WithAssistantReasoningReplay().
		WithFlatCustomTools().
		WithImageLimits(ImageLimitsFor(model))
}

// ImageLimitsFor returns the image limits of an OpenRouter model ID
// ("<vendor>/<model>") that hold on every upstream OpenRouter may route it
// to, or no limits for a vendor whose limits go-llms does not know. Per-image
// size limits differ between the upstreams of one model (Claude allows 10 MiB
// on Anthropic and 5 MiB on Amazon Bedrock), so they are left out.
func ImageLimitsFor(model string) llms.ImageLimits {
	vendor, _, _ := strings.Cut(model, "/")
	switch vendor {
	case "anthropic":
		// The image count and dimension limits are the same on Anthropic,
		// Amazon Bedrock and Vertex AI (checked 2026-11-04).
		return anthropic.ImageLimitsFor(model)
	case "openai":
		// OpenRouter serves OpenAI models from OpenAI, Azure and Bedrock;
		// the refusal relayed from OpenAI names OpenAI's own limit.
		return openai.ImageLimits()
	case "google":
		// OpenRouter serves Gemini from Vertex AI and Google AI Studio (the
		// Gemini Developer API); Vertex AI's limits are the stricter.
		return google.ImageLimitsFor(strings.TrimPrefix(model, "google/"), true)
	}
	return llms.ImageLimits{}
}

// NewWithReasoning returns an OpenRouter-configured Chat Completions client
// with the top-level "reasoning" request parameter set.
func NewWithReasoning(apiKey, model string, reasoning Reasoning) *Provider {
	return New(apiKey, model).
		WithCustomPayloadValue("reasoning", reasoning)
}
