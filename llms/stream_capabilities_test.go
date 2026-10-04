package llms_test

import (
	"context"
	"encoding/json"
	"net/http"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// capableStream is a two-turn provider stream with every optional
// capability: the first turn reports a search, a compaction checkpoint and a
// tool call with final arguments; the second answers in text.
type capableStream struct {
	turn int
}

var (
	testSearch     = llms.SearchActivity{Source: "web", ResultCount: 2}
	testCompaction = content.Compaction{Provider: content.CompactionProviderAnthropic, Text: "summary"}
	testToolCall   = llms.ToolCall{ID: "call_1", Name: "echo", Arguments: json.RawMessage(`{}`)}
	testFinalArgs  = json.RawMessage(`{"final":true}`)
)

func (s *capableStream) Iter() func(yield func(llms.StreamStatus) bool) {
	statuses := []llms.StreamStatus{llms.StreamStatusText}
	if s.turn == 0 {
		statuses = []llms.StreamStatus{
			llms.StreamStatusSearch,
			llms.StreamStatusCompaction,
			llms.StreamStatusToolCallBegin,
			llms.StreamStatusToolCallReady,
		}
	}
	return func(yield func(llms.StreamStatus) bool) {
		for _, status := range statuses {
			if !yield(status) {
				return
			}
		}
	}
}

func (s *capableStream) Message() llms.Message {
	if s.turn == 0 {
		return llms.Message{Role: "assistant", Content: content.Content{&testCompaction}, ToolCalls: []llms.ToolCall{testToolCall}}
	}
	return llms.Message{Role: "assistant", Content: content.FromText("done")}
}

func (s *capableStream) Err() error                  { return nil }
func (s *capableStream) Text() string                { return "done" }
func (s *capableStream) Image() (string, string)     { return "", "" }
func (s *capableStream) Audio() (string, string)     { return "", "" }
func (s *capableStream) Thought() content.Thought    { return content.Thought{} }
func (s *capableStream) ToolCall() llms.ToolCall     { return testToolCall }
func (s *capableStream) Usage() llms.Usage           { return llms.Usage{InputTokens: 300} }
func (s *capableStream) ContextUsage() llms.Usage    { return llms.Usage{InputTokens: 100} }
func (s *capableStream) Search() llms.SearchActivity { return testSearch }
func (s *capableStream) Compaction() content.Compaction {
	return testCompaction
}
func (s *capableStream) ToolArgumentFinalization() (json.RawMessage, bool) {
	return testFinalArgs, true
}

type wrappingProvider struct {
	turns int
}

func (p *wrappingProvider) Company() string            { return "test" }
func (p *wrappingProvider) Model() string              { return "test" }
func (p *wrappingProvider) SetHTTPClient(*http.Client) {}
func (p *wrappingProvider) Generate(context.Context, content.Content, []llms.Message, *tools.Toolbox, *tools.ValueSchema) llms.ProviderStream {
	stream := &capableStream{turn: p.turns}
	p.turns++
	return llms.StreamWrapper{ProviderStream: stream}
}

// The turn loop discovers each capability with a type assertion on the
// stream Generate returned, so this runs a real turn through a StreamWrapper:
// a capability the wrapper does not forward, or one the turn loop stops
// consulting, drops its update and fails here.
func TestStreamWrapperForwardsEveryCapabilityThroughTheTurnLoop(t *testing.T) {
	echo := tools.Func("Echo", "Echoes.", "echo", func(tools.Runner, struct{}) tools.Result {
		return tools.SuccessFromString("ok")
	})
	llm := llms.New(&wrappingProvider{}, echo)

	var search []llms.SearchActivity
	var compactions []content.Compaction
	var finalArgs []json.RawMessage
	for update := range llm.Chat("go") {
		switch u := update.(type) {
		case llms.SearchUpdate:
			search = append(search, u.SearchActivity)
		case llms.CompactionUpdate:
			compactions = append(compactions, u.Compaction)
		case llms.ToolArgumentFinalizationUpdate:
			finalArgs = append(finalArgs, u.Arguments)
		}
	}
	require.NoError(t, llm.Err())

	assert.Equal(t, []llms.SearchActivity{testSearch}, search)
	assert.Equal(t, []content.Compaction{testCompaction}, compactions)
	assert.Equal(t, []json.RawMessage{testFinalArgs}, finalArgs)
	assert.Equal(t, llms.Usage{InputTokens: 100}, llm.LastContextUsage)
	assert.Equal(t, llms.Usage{InputTokens: 600}, llm.TotalUsage)
}

type plainStream struct {
	llms.ProviderStream
}

func (plainStream) Usage() llms.Usage { return llms.Usage{InputTokens: 300} }

func TestStreamWrapperReportsZeroCapabilitiesForAPlainStream(t *testing.T) {
	w := llms.StreamWrapper{ProviderStream: plainStream{}}
	assert.Zero(t, w.Search())
	assert.Zero(t, w.Compaction())
	args, expected := w.ToolArgumentFinalization()
	assert.Nil(t, args)
	assert.False(t, expected)
	assert.Equal(t, llms.Usage{InputTokens: 300}, w.ContextUsage(), "a stream without passes reports its whole usage")
}
