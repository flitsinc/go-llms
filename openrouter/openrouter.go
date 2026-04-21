package openrouter

import (
	"context"
	"encoding/base64"
	"strconv"

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
			WithEndpoint("https://openrouter.ai/api/v1/chat/completions", "OpenRouter").
			WithoutPromptCacheRetention(),
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
	if p.reasoning == nil {
		p.reasoning = &Reasoning{}
	}
	p.reasoning.Effort = string(effort)
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
	payload, err := p.buildPayload(systemPrompt, messages, toolbox, jsonOutputSchema)
	if err != nil {
		return &errorStream{err: err}
	}
	return p.ChatCompletionsAPI.DoRequest(ctx, payload)
}

func (p *Provider) buildPayload(
	systemPrompt content.Content,
	messages []llms.Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) (map[string]any, error) {
	// BuildPayload handles tools, response_format, and other non-message fields.
	// We rebuild messages below with cache_control and reasoning_details, so the
	// messages built by BuildPayload are replaced.
	payload, err := p.ChatCompletionsAPI.BuildPayload(systemPrompt, messages, toolbox, jsonOutputSchema)
	if err != nil {
		return nil, err
	}

	// Replace messages with cache_control-aware versions and reasoning continuity.
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
	delete(payload, "prompt_cache_retention")

	// Add reasoning parameter if configured.
	if p.reasoning != nil {
		payload["reasoning"] = p.reasoning
	}

	return payload, nil
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

// messagesWithCacheControl converts an llms.Message to API messages with
// cache_control support and reasoning continuity for assistant messages.
func messagesWithCacheControl(m llms.Message) []openai.Message {
	msgs := openai.MessagesFromLLM(m)

	// For non-tool messages, re-convert content with cache control.
	if m.Role != "tool" && len(m.Content) > 0 {
		for i := range msgs {
			msgs[i].Content = convertContentWithCacheControl(m.Content)
		}
	}

	// For assistant messages, extract thoughts and emit as reasoning_details.
	if m.Role == "assistant" {
		details := reasoningDetailsFromContent(m.Content)
		if len(details) > 0 {
			for i := range msgs {
				msgs[i].ReasoningDetails = details
			}
		}
	}

	return msgs
}

func reasoningDetailsFromContent(c content.Content) []openai.ReasoningDetail {
	var details []openai.ReasoningDetail
	for _, item := range c {
		t, ok := item.(*content.Thought)
		if !ok {
			continue
		}
		detail := openai.ReasoningDetail{
			ID:        t.ID,
			Signature: t.Signature,
		}
		if format := t.Metadata["openai:reasoning_format"]; format != "" {
			detail.Format = format
		}
		if idx, ok := thoughtReasoningIndex(t); ok {
			detail.Index = &idx
		}
		switch {
		case len(t.Encrypted) > 0:
			detail.Type = "reasoning.encrypted"
			detail.Data = base64.StdEncoding.EncodeToString(t.Encrypted)
		case t.Summary:
			detail.Type = "reasoning.summary"
			detail.Summary = t.Text
		default:
			detail.Type = "reasoning.text"
			detail.Text = t.Text
		}
		if detail.Type == "reasoning.text" || detail.Type == "reasoning.summary" || detail.Type == "reasoning.encrypted" {
			details = append(details, detail)
		}
	}
	return details
}

func thoughtReasoningIndex(thought *content.Thought) (int, bool) {
	if thought == nil || thought.Metadata == nil {
		return 0, false
	}
	val, ok := thought.Metadata["openai:reasoning_index"]
	if !ok || val == "" {
		return 0, false
	}
	idx, err := strconv.Atoi(val)
	if err != nil {
		return 0, false
	}
	return idx, true
}

// errorStream is a minimal ProviderStream that only returns an error.
type errorStream struct {
	err error
}

var _ llms.ProviderStream = (*errorStream)(nil)

func (s *errorStream) Err() error { return s.err }
func (s *errorStream) Iter() func(yield func(llms.StreamStatus) bool) {
	return func(func(llms.StreamStatus) bool) {}
}
func (s *errorStream) Message() llms.Message    { return llms.Message{} }
func (s *errorStream) Text() string             { return "" }
func (s *errorStream) Image() (string, string)  { return "", "" }
func (s *errorStream) Audio() (string, string)  { return "", "" }
func (s *errorStream) Thought() content.Thought { return content.Thought{} }
func (s *errorStream) ToolCall() llms.ToolCall  { return llms.ToolCall{} }
func (s *errorStream) Usage() llms.Usage        { return llms.Usage{} }
