package llms

import (
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

type beforeResponseCaptureProvider struct {
	mu    sync.Mutex
	calls [][]Message
}

func (p *beforeResponseCaptureProvider) Company() string      { return "BeforeResponseCapture" }
func (p *beforeResponseCaptureProvider) Model() string        { return "before-response-model" }
func (p *beforeResponseCaptureProvider) SetDebugger(Debugger) {}
func (p *beforeResponseCaptureProvider) SetHTTPClient(_ *http.Client) {
}

func (p *beforeResponseCaptureProvider) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) ProviderStream {
	p.mu.Lock()
	p.calls = append(p.calls, cloneMessages(messages))
	callNumber := len(p.calls)
	p.mu.Unlock()

	if callNumber == 1 {
		return newBeforeResponseToolStream()
	}
	return &beforeResponseFinalStream{text: "done"}
}

func (p *beforeResponseCaptureProvider) Calls() [][]Message {
	p.mu.Lock()
	defer p.mu.Unlock()
	out := make([][]Message, len(p.calls))
	for i := range p.calls {
		out[i] = cloneMessages(p.calls[i])
	}
	return out
}

type beforeResponseToolStream struct {
	message Message
}

func newBeforeResponseToolStream() *beforeResponseToolStream {
	return &beforeResponseToolStream{
		message: Message{
			Role:    "assistant",
			Content: content.FromText("planning"),
			ToolCalls: []ToolCall{{
				ID:        "test_tool-id-0",
				Name:      "test_tool",
				Arguments: json.RawMessage(`{"test_param":"injected"}`),
			}},
		},
	}
}

func (s *beforeResponseToolStream) Err() error { return nil }

func (s *beforeResponseToolStream) Iter() func(func(StreamStatus) bool) {
	return func(yield func(StreamStatus) bool) {
		if !yield(StreamStatusText) {
			return
		}
		if !yield(StreamStatusToolCallBegin) {
			return
		}
		_ = yield(StreamStatusToolCallReady)
	}
}

func (s *beforeResponseToolStream) Message() Message         { return s.message }
func (s *beforeResponseToolStream) Text() string             { return "planning" }
func (s *beforeResponseToolStream) Image() (string, string)  { return "", "" }
func (s *beforeResponseToolStream) Thought() content.Thought { return content.Thought{} }
func (s *beforeResponseToolStream) ToolCall() ToolCall       { return s.message.ToolCalls[0] }
func (s *beforeResponseToolStream) Usage() Usage             { return Usage{} }

type beforeResponseFinalStream struct {
	text string
}

func (s *beforeResponseFinalStream) Err() error { return nil }

func (s *beforeResponseFinalStream) Iter() func(func(StreamStatus) bool) {
	return func(yield func(StreamStatus) bool) {
		_ = yield(StreamStatusText)
	}
}

func (s *beforeResponseFinalStream) Message() Message {
	return Message{Role: "assistant", Content: content.FromText(s.text)}
}

func (s *beforeResponseFinalStream) Text() string             { return s.text }
func (s *beforeResponseFinalStream) Image() (string, string)  { return "", "" }
func (s *beforeResponseFinalStream) Thought() content.Thought { return content.Thought{} }
func (s *beforeResponseFinalStream) ToolCall() ToolCall       { return ToolCall{} }
func (s *beforeResponseFinalStream) Usage() Usage             { return Usage{} }

func TestBeforeResponseHookCanAppendMessages(t *testing.T) {
	provider := &beforeResponseCaptureProvider{}
	llm := New(provider, testTool)
	llm.BeforeResponse = func(ctx context.Context, state BeforeResponseState) error {
		state.Append(Message{
			Role:    "user",
			Content: content.FromText("injected-turn"),
		})
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	_ = runTestChat(ctx, t, llm, "original user message")

	require.NoError(t, llm.Err())

	calls := provider.Calls()
	require.Len(t, calls, 2, "expected two provider calls")

	require.GreaterOrEqual(t, len(calls[0]), 2)
	assert.Equal(t, "original user message", firstText(calls[0][0]))
	assert.Equal(t, "injected-turn", firstText(calls[0][len(calls[0])-1]))

	require.GreaterOrEqual(t, len(calls[1]), 2)
	assert.Equal(t, "injected-turn", firstText(calls[1][len(calls[1])-1]))
}

func TestBeforeResponseHookCanReplaceMessages(t *testing.T) {
	provider := &beforeResponseCaptureProvider{}
	llm := New(provider, testTool)
	llm.BeforeResponse = func(ctx context.Context, state BeforeResponseState) error {
		if state.Turn() == 1 {
			state.Replace(Message{
				Role:    "user",
				Content: content.FromText("replacement message"),
			})
		}
		return nil
	}

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_ = runTestChat(ctx, t, llm, "should be replaced")

	require.NoError(t, llm.Err())
	calls := provider.Calls()
	require.NotEmpty(t, calls)
	require.NotEmpty(t, calls[0])
	assert.Equal(t, "replacement message", firstText(calls[0][0]))
}

func TestBeforeResponseHookCanAbort(t *testing.T) {
	provider := &beforeResponseCaptureProvider{}
	llm := New(provider)

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	const reason = "abort before provider request"
	llm.BeforeResponse = func(ctx context.Context, state BeforeResponseState) error {
		return errors.New(reason)
	}
	_ = runTestChat(ctx, t, llm, "abort please")

	require.Error(t, llm.Err())
	assert.Contains(t, llm.Err().Error(), reason)
	assert.Len(t, provider.Calls(), 0, "provider should not be called after before-response abort")
}

func firstText(msg Message) string {
	for _, item := range msg.Content {
		if text, ok := item.(*content.Text); ok {
			return text.Text
		}
	}
	return ""
}
