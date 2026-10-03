package openai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
)

func TestResponsesStream_CompactionItem(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"response.created"}`,
		`data: {"type":"response.output_item.done","item":{"type":"compaction","id":"cmp_1","encrypted_content":"gAAAAenc"}}`,
		`data: {"type":"response.output_item.added","item":{"type":"message","role":"assistant"}}`,
		`data: {"type":"response.content_part.added","part":{"type":"text","text":"Hello"},"item_id":"msg_1","content_index":0}`,
		`data: {"type":"response.content_part.done","part":{"type":"text","text":"Hello"},"item_id":"msg_1","content_index":0}`,
		`data: {"type":"response.completed","response":{"usage":{"input_tokens":100,"output_tokens":50,"input_tokens_details":{"cached_tokens":25}}}}`,
		"",
	}, "\n")
	stream := &ResponsesStream{ctx: context.Background(), model: "gpt-5.4", stream: strings.NewReader(sse)}

	var reported content.Compaction
	var statuses []llms.StreamStatus
	stream.Iter()(func(status llms.StreamStatus) bool {
		statuses = append(statuses, status)
		if status == llms.StreamStatusCompaction {
			reported = stream.Compaction()
		}
		return true
	})
	require.NoError(t, stream.Err())

	want := content.Compaction{Provider: "openai", ID: "cmp_1", Encrypted: "gAAAAenc"}
	assert.Contains(t, statuses, llms.StreamStatusCompaction)
	assert.Equal(t, want, reported)
	require.NotEmpty(t, stream.Message().Content)
	assert.Equal(t, &want, stream.Message().Content[0])
}

func TestConvertMessageToInput_ReplaysCompactionBeforeOutput(t *testing.T) {
	items, err := convertMessageToInput(llms.Message{
		Role: "assistant",
		ID:   "msg_1",
		Content: content.Content{
			&content.Compaction{Provider: "openai", ID: "cmp_1", Encrypted: "gAAAAenc"},
			&content.Text{Text: "Hello"},
		},
	}, nil)
	require.NoError(t, err)
	raw, err := json.Marshal(items)
	require.NoError(t, err)
	assert.JSONEq(t, `[
		{"type":"compaction","id":"cmp_1","encrypted_content":"gAAAAenc"},
		{"type":"message","id":"msg_1","role":"assistant","content":[{"type":"output_text","text":"Hello"}]}
	]`, string(raw))
}

func TestConvertMessageToInput_RejectsForeignCompaction(t *testing.T) {
	_, err := convertMessageToInput(llms.Message{
		Role:    "assistant",
		Content: content.Content{&content.Compaction{Provider: "anthropic", Text: "Summary."}},
	}, nil)
	require.ErrorIs(t, err, content.ErrForeignCompaction)
}

func TestResponsesStream_UnreadableCompactionItemFailsTheStream(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"response.created"}`,
		`data: {"type":"response.output_item.done","item":{"type":"compaction","id":"cmp_1"}}`,
		`data: {"type":"response.completed","response":{"usage":{"input_tokens":100,"output_tokens":50}}}`,
		"",
	}, "\n")
	stream := &ResponsesStream{ctx: context.Background(), model: "gpt-5.4", stream: strings.NewReader(sse)}
	stream.Iter()(func(llms.StreamStatus) bool { return true })
	require.ErrorContains(t, stream.Err(), "has no encrypted_content")
}

func TestChatCompletions_RejectsCompaction(t *testing.T) {
	_, err := convertContentWithOptions(content.Content{&content.Compaction{Provider: content.CompactionProviderOpenAI, ID: "cmp_1", Encrypted: "enc"}}, chatMessageEncodingOptions{})
	require.ErrorIs(t, err, content.ErrForeignCompaction)
}

func TestBuildResponsesPayload_Compaction(t *testing.T) {
	m := NewResponsesAPI("key", "gpt-5.4").WithCompaction(400000)
	payload, err := m.buildResponsesPayload(nil, "", nil, nil)
	require.NoError(t, err)
	raw, err := json.Marshal(payload["context_management"])
	require.NoError(t, err)
	assert.JSONEq(t, `[{"type":"compaction","compact_threshold":400000}]`, string(raw))

	payload, err = NewResponsesAPI("key", "gpt-5.4").buildResponsesPayload(nil, "", nil, nil)
	require.NoError(t, err)
	assert.NotContains(t, payload, "context_management")
}
