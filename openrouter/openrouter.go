package openrouter

import (
	"context"
	"encoding/base64"
	"encoding/json"

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

// Builder method overrides — these delegate to ChatCompletionsAPI but return
// *Provider so that method chaining doesn't silently downcast to the base type.

func (p *Provider) WithMaxCompletionTokens(n int) *Provider {
	p.ChatCompletionsAPI.WithMaxCompletionTokens(n)
	return p
}

func (p *Provider) WithThinking(effort openai.Effort) *Provider {
	p.ChatCompletionsAPI.WithThinking(effort)
	return p
}

func (p *Provider) WithVerbosity(verbosity openai.Verbosity) *Provider {
	p.ChatCompletionsAPI.WithVerbosity(verbosity)
	return p
}

func (p *Provider) WithIncludeUsage(include bool) *Provider {
	p.ChatCompletionsAPI.WithIncludeUsage(include)
	return p
}

func (p *Provider) WithCustomPayloadValue(key string, value any) *Provider {
	p.ChatCompletionsAPI.WithCustomPayloadValue(key, value)
	return p
}

func (p *Provider) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []llms.Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) llms.ProviderStream {
	// BuildPayload handles tools, response_format, and other non-message fields.
	// We rebuild messages below with cache_control and reasoning_details, so the
	// messages built by BuildPayload are replaced.
	payload, err := p.ChatCompletionsAPI.BuildPayload(systemPrompt, messages, toolbox, jsonOutputSchema)
	if err != nil {
		return &errorStream{err: err}
	}

	// Replace messages with cache_control-aware versions and reasoning continuity.
	var apiMessages []any
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

	// Add reasoning parameter if configured.
	if p.reasoning != nil {
		payload["reasoning"] = p.reasoning
	}

	return p.ChatCompletionsAPI.DoRequest(ctx, payload)
}

// convertContentWithCacheControl converts content.Content to a ContentList,
// attaching cache_control: {"type": "ephemeral"} to content parts that precede
// a CacheHint. This enables prompt caching for Anthropic models via OpenRouter.
//
// Note: this walks the original content to find CacheHint positions while
// tracking which items ConvertContent skips (Thought, CacheHint). If
// ConvertContent starts skipping new types, this function must be updated.
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

// messageWithReasoning wraps a base message and adds reasoning_details.
// It uses custom JSON marshaling to preserve all base message fields.
type messageWithReasoning struct {
	base             openai.Message
	reasoningDetails []openai.ReasoningDetail
}

func (m messageWithReasoning) MarshalJSON() ([]byte, error) {
	// Marshal base message to a map, then add reasoning_details.
	data, err := json.Marshal(m.base)
	if err != nil {
		return nil, err
	}
	var obj map[string]json.RawMessage
	if err := json.Unmarshal(data, &obj); err != nil {
		return nil, err
	}
	rd, err := json.Marshal(m.reasoningDetails)
	if err != nil {
		return nil, err
	}
	obj["reasoning_details"] = rd
	return json.Marshal(obj)
}

// messagesWithCacheControl converts an llms.Message to API messages with
// cache_control support and reasoning continuity for assistant messages.
func messagesWithCacheControl(m llms.Message) []any {
	msgs := openai.MessagesFromLLM(m)

	// For non-tool messages, re-convert content with cache control.
	if m.Role != "tool" && len(m.Content) > 0 {
		for i := range msgs {
			msgs[i].Content = convertContentWithCacheControl(m.Content)
		}
	}

	// For assistant messages, extract thoughts and emit as reasoning_details.
	if m.Role == "assistant" {
		var details []openai.ReasoningDetail
		var thoughtText string
		var signature string
		for _, item := range m.Content {
			t, ok := item.(*content.Thought)
			if !ok {
				continue
			}
			if len(t.Encrypted) > 0 {
				// Encrypted/redacted thinking — emit as reasoning.encrypted.
				details = append(details, openai.ReasoningDetail{
					Type: "reasoning.encrypted",
					Data: base64.StdEncoding.EncodeToString(t.Encrypted),
				})
			} else {
				thoughtText += t.Text
				if t.Signature != "" {
					signature = t.Signature
				}
			}
		}
		if thoughtText != "" || signature != "" {
			detail := openai.ReasoningDetail{
				Type: "reasoning.text",
				Text: thoughtText,
			}
			if signature != "" {
				detail.Signature = signature
			}
			details = append(details, detail)
		}
		if len(details) > 0 {
			result := make([]any, len(msgs))
			for i, msg := range msgs {
				result[i] = messageWithReasoning{
					base:             msg,
					reasoningDetails: details,
				}
			}
			return result
		}
	}

	// Default: return as-is
	result := make([]any, len(msgs))
	for i, msg := range msgs {
		result[i] = msg
	}
	return result
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
func (s *errorStream) Audio() (string, string)                       { return "", "" }
func (s *errorStream) Thought() content.Thought                      { return content.Thought{} }
func (s *errorStream) ToolCall() llms.ToolCall                       { return llms.ToolCall{} }
func (s *errorStream) Usage() llms.Usage                             { return llms.Usage{} }
