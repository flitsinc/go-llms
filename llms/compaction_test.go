package llms

import (
	"context"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/tools"
)

type compactionMockProvider struct{}

func (compactionMockProvider) Company() string              { return "Test" }
func (compactionMockProvider) Model() string                { return "test-model" }
func (compactionMockProvider) SetHTTPClient(_ *http.Client) {}
func (compactionMockProvider) Generate(context.Context, content.Content, []Message, *tools.Toolbox, *tools.ValueSchema) ProviderStream {
	return &compactionMockStream{}
}

type compactionMockStream struct{}

var testCompaction = content.Compaction{Provider: "anthropic", Text: "Summary."}

func (*compactionMockStream) Err() error { return nil }
func (*compactionMockStream) Iter() func(func(StreamStatus) bool) {
	return func(yield func(StreamStatus) bool) {
		if !yield(StreamStatusCompaction) {
			return
		}
		yield(StreamStatusText)
	}
}
func (*compactionMockStream) Message() Message {
	compaction := testCompaction
	return Message{Role: "assistant", Content: content.Content{&compaction, &content.Text{Text: "Hi"}}}
}
func (*compactionMockStream) Text() string                   { return "Hi" }
func (*compactionMockStream) Audio() (string, string)        { return "", "" }
func (*compactionMockStream) Image() (string, string)        { return "", "" }
func (*compactionMockStream) Thought() content.Thought       { return content.Thought{} }
func (*compactionMockStream) ToolCall() ToolCall             { return ToolCall{} }
func (*compactionMockStream) Usage() Usage                   { return Usage{} }
func (*compactionMockStream) Compaction() content.Compaction { return testCompaction }

func TestChat_EmitsCompactionUpdate(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	llm := New(compactionMockProvider{})
	updates := runTestChat(ctx, t, llm, "hello")
	require.NoError(t, llm.Err())

	var compactionUpdates []CompactionUpdate
	for _, u := range updates {
		if cu, ok := u.(CompactionUpdate); ok {
			compactionUpdates = append(compactionUpdates, cu)
		}
	}
	require.Len(t, compactionUpdates, 1)
	assert.Equal(t, UpdateTypeCompaction, compactionUpdates[0].Type())
	assert.Equal(t, testCompaction, compactionUpdates[0].Compaction)
}
