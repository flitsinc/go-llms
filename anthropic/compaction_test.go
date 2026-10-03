package anthropic

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
)

// rawString encodes s as a JSON string for the raw wire fields.
func rawString(s string) json.RawMessage {
	b, _ := json.Marshal(s)
	return b
}

func TestAnthropicStream_ThresholdCompaction(t *testing.T) {
	var sse strings.Builder
	sse.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{ID: "msg_1", Role: "assistant", Usage: &usage{InputTokens: numPtr(1)}}}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_start", Index: 0, ContentBlock: &contentBlock{Type: "compaction"}}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 0, Delta: delta{Type: "compaction_delta", Content: rawString("Summary of earlier work.")}}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 0}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_start", Index: 1, ContentBlock: &contentBlock{Type: "text"}}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 1, Delta: delta{Type: "text_delta", Text: "Continuing."}}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 1}))
	sse.WriteString(`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":23000,"output_tokens":1000,"cache_read_input_tokens":500,"iterations":[{"type":"compaction","input_tokens":180000,"output_tokens":3500,"cache_read_input_tokens":170000},{"type":"message","input_tokens":23000,"output_tokens":1000,"cache_read_input_tokens":500}]}}` + "\n\n")
	sse.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

	stream := newTestAnthropicStream(context.Background(), "claude-opus-4-6", sse.String())
	var statuses []llms.StreamStatus
	var reported content.Compaction
	stream.Iter()(func(status llms.StreamStatus) bool {
		statuses = append(statuses, status)
		if status == llms.StreamStatusCompaction {
			reported = stream.Compaction()
		}
		return true
	})
	require.NoError(t, stream.Err())

	assert.Equal(t, []llms.StreamStatus{llms.StreamStatusMessageStart, llms.StreamStatusCompaction, llms.StreamStatusText}, statuses)
	want := content.Compaction{Provider: "anthropic", Text: "Summary of earlier work."}
	assert.Equal(t, want, reported)

	msg := stream.Message()
	require.Len(t, msg.Content, 2)
	assert.Equal(t, &want, msg.Content[0])
	assert.Equal(t, &content.Text{Text: "Continuing."}, msg.Content[1])

	// Top-level usage excludes the compaction iteration, so it is added back.
	assert.Equal(t, llms.Usage{InputTokens: 203000, OutputTokens: 4500, CachedInputTokens: 170500}, stream.Usage())
	// The context footprint is the final iteration alone.
	assert.Equal(t, llms.Usage{InputTokens: 23000, OutputTokens: 1000, CachedInputTokens: 500}, stream.ContextUsage())
}

func TestAnthropicStream_ContextUsageWithoutCompaction(t *testing.T) {
	var sse strings.Builder
	sse.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{ID: "msg_1", Role: "assistant", Usage: &usage{InputTokens: numPtr(1)}}}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_start", Index: 0, ContentBlock: &contentBlock{Type: "text"}}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 0, Delta: delta{Type: "text_delta", Text: "Hi."}}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 0}))
	sse.WriteString(`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":600,"output_tokens":20,"cache_read_input_tokens":590000,"cache_creation_input_tokens":400}}` + "\n\n")
	sse.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

	stream := newTestAnthropicStream(context.Background(), "claude-opus-4-6", sse.String())
	stream.Iter()(func(llms.StreamStatus) bool { return true })
	require.NoError(t, stream.Err())

	want := llms.Usage{InputTokens: 600, OutputTokens: 20, CachedInputTokens: 590000, CacheCreationInputTokens: 400}
	assert.Equal(t, want, stream.Usage())
	assert.Equal(t, want, stream.ContextUsage())
}

func TestAnthropicStream_OnDemandCompactionBlockArrivesWhole(t *testing.T) {
	var sse strings.Builder
	sse.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_start", Index: 0, ContentBlock: &contentBlock{Type: "compaction", Content: rawString("Summary."), Signature: "sig_1"}}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 0}))
	sse.WriteString(sseEvent(streamEvent{Type: "message_delta", Delta: delta{StopReason: "compaction"}}))
	sse.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

	stream := newTestAnthropicStream(context.Background(), "claude-opus-4-6", sse.String())
	stream.Iter()(func(llms.StreamStatus) bool { return true })
	require.NoError(t, stream.Err())
	assert.Equal(t, content.Content{&content.Compaction{Provider: "anthropic", Text: "Summary.", Signature: "sig_1"}}, stream.Message().Content)
}

func TestAnthropicStream_FailedCompactionIsNotReplayed(t *testing.T) {
	var sse strings.Builder
	sse.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
	sse.WriteString(`data: {"type":"content_block_start","index":0,"content_block":{"type":"compaction","content":null}}` + "\n\n")
	sse.WriteString(`data: {"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":null}}` + "\n\n")
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 0}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_start", Index: 1, ContentBlock: &contentBlock{Type: "text"}}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 1, Delta: delta{Type: "text_delta", Text: "Hi"}}))
	sse.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 1}))
	sse.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

	stream := newTestAnthropicStream(context.Background(), "claude-opus-4-6", sse.String())
	var statuses []llms.StreamStatus
	stream.Iter()(func(status llms.StreamStatus) bool { statuses = append(statuses, status); return true })
	require.NoError(t, stream.Err())
	assert.NotContains(t, statuses, llms.StreamStatusCompaction)
	assert.Equal(t, content.Content{&content.Text{Text: "Hi"}}, stream.Message().Content)
}

type capturedRequest struct {
	Headers http.Header
	Body    map[string]json.RawMessage
}

func captureGenerate(t *testing.T, m *Model, messages []llms.Message) capturedRequest {
	t.Helper()
	ch := make(chan capturedRequest, 1)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(raw, &body))
		ch <- capturedRequest{Headers: r.Header.Clone(), Body: body}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte("data: {\"type\":\"message_start\",\"message\":{\"role\":\"assistant\"}}\n\ndata: {\"type\":\"message_stop\"}\n\n"))
	}))
	defer ts.Close()
	m.endpoint = ts.URL
	stream := m.Generate(context.Background(), nil, messages, nil, nil)
	stream.Iter()(func(llms.StreamStatus) bool { return true })
	require.NoError(t, stream.Err())
	return <-ch
}

func TestAnthropic_ContextCompactionRequest(t *testing.T) {
	m := New("key", "claude-opus-4-6").WithBeta("context-1m-2025-08-07").WithContextCompaction(ContextCompaction{
		TriggerInputTokens: 600000,
		Instructions:       "Do not call tools.",
	})
	req := captureGenerate(t, m, []llms.Message{{Role: "user", Content: content.FromText("hi")}})

	assert.ElementsMatch(t, []string{"context-1m-2025-08-07", "compact-2026-01-12"}, req.Headers.Values("Anthropic-Beta"))
	assert.JSONEq(t, `{"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":600000},"instructions":"Do not call tools."}]}`, string(req.Body["context_management"]))
	// The configured betas are not mutated by the per-request addition.
	assert.Equal(t, []string{"context-1m-2025-08-07"}, m.betaFeatures)
}

func TestAnthropic_ReplaysCompactionBlockVerbatim(t *testing.T) {
	m := New("key", "claude-opus-4-6")
	req := captureGenerate(t, m, []llms.Message{
		{Role: "assistant", Content: content.Content{
			&content.Compaction{Provider: "anthropic", Text: "Summary."},
			&content.CacheHint{Duration: "long"},
			&content.Text{Text: "Continuing."},
		}},
		{Role: "user", Content: content.FromText("next")},
	})

	// Replaying a compaction block requires the beta even without new compaction.
	assert.Equal(t, []string{"compact-2026-01-12"}, req.Headers.Values("Anthropic-Beta"))
	_, hasContextManagement := req.Body["context_management"]
	assert.False(t, hasContextManagement)
	assert.JSONEq(t, `[
		{"role":"assistant","content":[
			{"type":"compaction","content":"Summary.","cache_control":{"type":"ephemeral","ttl":"1h"}},
			{"type":"text","text":"Continuing."}
		]},
		{"role":"user","content":[{"type":"text","text":"next"}]}
	]`, string(req.Body["messages"]))
}

func TestAnthropic_RejectsForeignCompaction(t *testing.T) {
	stream := New("key", "claude-opus-4-6").Generate(context.Background(), nil, []llms.Message{
		{Role: "assistant", Content: content.Content{&content.Compaction{Provider: "openai", Encrypted: "enc"}}},
	}, nil, nil)
	require.ErrorIs(t, stream.Err(), content.ErrForeignCompaction)
}

// Server tool results carry a non-string "content"; it must not be decoded
// as a compaction summary.
func TestAnthropicStream_ServerToolResultContentIsNotCompaction(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"message_start","message":{"id":"msg_1","role":"assistant","usage":{"input_tokens":10}}}`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"server_tool_use","id":"srvtoolu_1","name":"web_search","input":{}}}`,
		`data: {"type":"content_block_stop","index":0}`,
		`data: {"type":"content_block_start","index":1,"content_block":{"type":"web_search_tool_result","tool_use_id":"srvtoolu_1","content":[{"type":"web_search_result","url":"https://example.com","title":"x","encrypted_content":"abc"}]}}`,
		`data: {"type":"content_block_stop","index":1}`,
		`data: {"type":"content_block_start","index":2,"content_block":{"type":"text"}}`,
		`data: {"type":"content_block_delta","index":2,"delta":{"type":"text_delta","text":"Found it."}}`,
		`data: {"type":"content_block_stop","index":2}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
		`data: {"type":"message_stop"}`,
		"",
	}, "\n\n")

	stream := newTestAnthropicStream(context.Background(), "claude-opus-4-6", sse)
	stream.Iter()(func(llms.StreamStatus) bool { return true })
	require.NoError(t, stream.Err())
	assert.Equal(t, content.Content{&content.Text{Text: "Found it."}}, stream.Message().Content)
	assert.Equal(t, content.Compaction{}, stream.Compaction())
}

// OpenRouter's Messages API relays an encrypted form of the summary, which is
// replayed alongside it.
func TestAnthropicStream_CompactionEncryptedContentRoundTrip(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"message_start","message":{"id":"msg_1","role":"assistant","usage":{"input_tokens":10}}}`,
		`data: {"type":"content_block_start","index":0,"content_block":{"type":"compaction","content":null,"encrypted_content":null}}`,
		`data: {"type":"content_block_delta","index":0,"delta":{"type":"compaction_delta","content":"Summary.","encrypted_content":"enc_1"}}`,
		`data: {"type":"content_block_stop","index":0}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":5}}`,
		`data: {"type":"message_stop"}`,
		"",
	}, "\n\n")

	stream := newTestAnthropicStream(context.Background(), "claude-opus-4-6", sse)
	stream.Iter()(func(llms.StreamStatus) bool { return true })
	require.NoError(t, stream.Err())
	want := content.Compaction{Provider: content.CompactionProviderAnthropic, Text: "Summary.", Encrypted: "enc_1"}
	assert.Equal(t, want, stream.Compaction())

	req := captureGenerate(t, New("key", "claude-opus-4-6"), []llms.Message{
		{Role: "assistant", Content: content.Content{&want}},
		{Role: "user", Content: content.FromText("next")},
	})
	assert.JSONEq(t, `[
		{"role":"assistant","content":[{"type":"compaction","content":"Summary.","encrypted_content":"enc_1"}]},
		{"role":"user","content":[{"type":"text","text":"next"}]}
	]`, string(req.Body["messages"]))
}

func TestAnthropic_RejectsCompactionTriggerBelowMinimum(t *testing.T) {
	stream := New("key", "claude-opus-4-6").
		WithContextCompaction(ContextCompaction{TriggerInputTokens: 49_999}).
		Generate(context.Background(), nil, []llms.Message{{Role: "user", Content: content.FromText("hi")}}, nil, nil)
	require.ErrorContains(t, stream.Err(), "below the API minimum")
}

// Iterations may omit the cache buckets; a single message iteration takes
// them from the top-level usage, which covers exactly that iteration.
func TestAnthropicStream_ContextUsageFillsMissingIterationCacheBuckets(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"message_start","message":{"id":"msg_1","role":"assistant","usage":{"input_tokens":1}}}`,
		`data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"input_tokens":40,"output_tokens":10,"cache_read_input_tokens":589960,"cache_creation_input_tokens":10000,"iterations":[{"type":"compaction","input_tokens":700000,"output_tokens":3000},{"type":"message","input_tokens":40,"output_tokens":10}]}}`,
		`data: {"type":"message_stop"}`,
		"",
	}, "\n\n")

	stream := newTestAnthropicStream(context.Background(), "claude-opus-4-6", sse)
	stream.Iter()(func(llms.StreamStatus) bool { return true })
	require.NoError(t, stream.Err())
	assert.Equal(t, llms.Usage{InputTokens: 40, OutputTokens: 10, CachedInputTokens: 589960, CacheCreationInputTokens: 10000}, stream.ContextUsage())
	assert.Equal(t, llms.Usage{InputTokens: 700040, OutputTokens: 3010, CachedInputTokens: 589960, CacheCreationInputTokens: 10000}, stream.Usage())
}

func TestAnthropic_BearerAuth(t *testing.T) {
	req := captureGenerate(t, New("key", "claude-opus-4-6").WithBearerAuth(), []llms.Message{{Role: "user", Content: content.FromText("hi")}})
	assert.Equal(t, "Bearer key", req.Headers.Get("Authorization"))
	assert.Empty(t, req.Headers.Get("X-API-Key"))

	req = captureGenerate(t, New("key", "claude-opus-4-6"), []llms.Message{{Role: "user", Content: content.FromText("hi")}})
	assert.Equal(t, "key", req.Headers.Get("X-API-Key"))
	assert.Empty(t, req.Headers.Get("Authorization"))
}
