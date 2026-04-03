package openrouter

import (
	"context"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/openai"
	"github.com/flitsinc/go-llms/tools"
)

// Reasoning configures thinking/reasoning behavior for OpenRouter requests.
type Reasoning struct {
	Effort string `json:"effort,omitempty"` // "xhigh", "high", "medium", "low", "minimal", "none"
}

// Provider wraps ChatCompletionsAPI with OpenRouter-specific behavior:
// cache_control on content parts, no prompt_cache_retention, and optional reasoning.
type Provider struct {
	*openai.ChatCompletionsAPI
	reasoning *Reasoning
}

// New creates an OpenRouter provider.
func New(apiKey, model string) *Provider {
	return &Provider{
		ChatCompletionsAPI: openai.New(apiKey, model).
			WithEndpoint("https://openrouter.ai/api/v1/chat/completions", "OpenRouter"),
	}
}

// NewWithReasoning creates an OpenRouter provider with reasoning/thinking enabled.
func NewWithReasoning(apiKey, model string, reasoning Reasoning) *Provider {
	p := New(apiKey, model)
	p.reasoning = &reasoning
	return p
}

func (p *Provider) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []llms.Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) llms.ProviderStream {
	payload, err := p.ChatCompletionsAPI.BuildPayload(systemPrompt, messages, toolbox, jsonOutputSchema)
	if err != nil {
		return &errorStream{err: err}
	}

	// Replace messages with cache_control-aware versions.
	var apiMessages []openai.Message
	if systemPrompt != nil {
		apiMessages = append(apiMessages, openai.Message{
			Role:    "system",
			Content: convertContentWithCacheControl(systemPrompt),
		})
	}
	for _, msg := range messages {
		apiMessages = append(apiMessages, messagesWithCacheControl(msg)...)
	}
	payload["messages"] = apiMessages

	// OpenRouter doesn't support prompt_cache_retention; remove it.
	delete(payload, "prompt_cache_retention")

	// Add reasoning parameter if configured.
	if p.reasoning != nil {
		payload["reasoning"] = p.reasoning
	}

	return p.ChatCompletionsAPI.DoRequest(ctx, payload)
}

// convertContentWithCacheControl converts content.Content to a ContentList,
// attaching cache_control: {"type": "ephemeral"} to content parts that precede
// a CacheHint. This enables prompt caching for Anthropic models via OpenRouter.
func convertContentWithCacheControl(c content.Content) openai.ContentList {
	cl := openai.ConvertContent(c)

	// Walk through original content to find CacheHint positions and attach
	// cache_control to the corresponding preceding content part.
	clIdx := 0
	for _, item := range c {
		switch item.(type) {
		case *content.CacheHint:
			if clIdx > 0 {
				cl[clIdx-1].CacheControl = &openai.CacheControl{Type: "ephemeral"}
			}
		case *content.Thought:
			// Skipped by ConvertContent, don't advance index.
		default:
			clIdx++
		}
	}
	return cl
}

// messagesWithCacheControl converts an llms.Message to OpenAI API messages
// with cache_control support on content parts.
func messagesWithCacheControl(m llms.Message) []openai.Message {
	msgs := openai.MessagesFromLLM(m)
	// For non-tool messages, re-convert content with cache control.
	if m.Role != "tool" && len(m.Content) > 0 {
		for i := range msgs {
			msgs[i].Content = convertContentWithCacheControl(m.Content)
		}
	}
	return msgs
}

// errorStream is a minimal ProviderStream that only returns an error.
type errorStream struct {
	err error
}

func (s *errorStream) Err() error                                    { return s.err }
func (s *errorStream) Iter() func(yield func(llms.StreamStatus) bool) { return func(func(llms.StreamStatus) bool) {} }
func (s *errorStream) Message() llms.Message                         { return llms.Message{} }
func (s *errorStream) Text() string                                  { return "" }
func (s *errorStream) Image() (string, string)                       { return "", "" }
func (s *errorStream) Thought() content.Thought                      { return content.Thought{} }
func (s *errorStream) ToolCall() llms.ToolCall                       { return llms.ToolCall{} }
func (s *errorStream) Usage() llms.Usage                             { return llms.Usage{} }
