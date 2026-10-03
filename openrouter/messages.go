package openrouter

import "github.com/flitsinc/go-llms/anthropic"

// MessagesEndpoint is OpenRouter's Anthropic Messages API endpoint.
const MessagesEndpoint = "https://openrouter.ai/api/v1/messages"

// NewMessages returns an Anthropic Messages client for OpenRouter's
// /api/v1/messages endpoint, which authenticates with a bearer token. Unlike
// Chat Completions, it carries Anthropic-native request and response blocks,
// so features such as context management compaction keep their artifacts
// (see https://openrouter.ai/docs/api/api-reference/anthropic-messages/create-a-message).
func NewMessages(apiKey, model string) *anthropic.Model {
	return anthropic.New(apiKey, model).
		WithEndpoint(MessagesEndpoint, "OpenRouter").
		WithBearerAuth()
}
