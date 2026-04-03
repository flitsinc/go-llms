package openrouter

import "github.com/flitsinc/go-llms/openai"

// New creates an OpenAI-compatible ChatCompletionsAPI configured for OpenRouter.
// Cache control is enabled by default so that CacheHint items are passed through
// to upstream providers (e.g. Anthropic) for prompt caching.
func New(apiKey, model string) *openai.ChatCompletionsAPI {
	return openai.New(apiKey, model).
		WithEndpoint("https://openrouter.ai/api/v1/chat/completions", "OpenRouter").
		WithCacheControl(true)
}
