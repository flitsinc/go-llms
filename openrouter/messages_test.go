package openrouter

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/flitsinc/go-llms/anthropic"
	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNewMessages_SendsBearerAuthAndContextManagement(t *testing.T) {
	type captured struct {
		header http.Header
		body   map[string]json.RawMessage
	}
	ch := make(chan captured, 1)
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		raw, _ := io.ReadAll(r.Body)
		var body map[string]json.RawMessage
		require.NoError(t, json.Unmarshal(raw, &body))
		ch <- captured{header: r.Header.Clone(), body: body}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, ": OPENROUTER PROCESSING\n\n"+
			"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"gen-1\",\"role\":\"assistant\",\"usage\":{\"input_tokens\":10,\"output_tokens\":1}}}\n\n"+
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"compaction\",\"content\":\"\"}}\n\n"+
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"compaction_delta\",\"content\":\"Summary.\"}}\n\n"+
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"+
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"+
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hi\"}}\n\n"+
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":1}\n\n"+
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\"},\"usage\":{\"input_tokens\":213,\"output_tokens\":24,\"cost\":0.2,\"iterations\":[{\"type\":\"compaction\",\"input_tokens\":65839,\"output_tokens\":123},{\"type\":\"message\",\"input_tokens\":213,\"output_tokens\":24}]}}\n\n"+
			"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")
	}))
	defer ts.Close()

	m := NewMessages("or-key", "anthropic/claude-sonnet-4.6").
		WithEndpoint(ts.URL, "OpenRouter").
		WithContextCompaction(anthropic.ContextCompaction{TriggerInputTokens: 600000})
	assert.Equal(t, "OpenRouter", m.Company())

	stream := m.Generate(context.Background(), nil, []llms.Message{{Role: "user", Content: content.FromText("hi")}}, nil, nil)
	stream.Iter()(func(llms.StreamStatus) bool { return true })
	require.NoError(t, stream.Err())

	req := <-ch
	assert.Equal(t, "Bearer or-key", req.header.Get("Authorization"))
	assert.Empty(t, req.header.Get("X-API-Key"))
	assert.JSONEq(t, `"anthropic/claude-sonnet-4.6"`, string(req.body["model"]))
	assert.JSONEq(t, `{"edits":[{"type":"compact_20260112","trigger":{"type":"input_tokens","value":600000}}]}`, string(req.body["context_management"]))

	assert.Equal(t, content.Content{
		&content.Compaction{Provider: "anthropic", Text: "Summary."},
		&content.Text{Text: "Hi"},
	}, stream.Message().Content)
	assert.Equal(t, 65839+213, stream.Usage().InputTokens)
	cs, ok := stream.(llms.ContextUsageStream)
	require.True(t, ok)
	assert.Equal(t, 213, cs.ContextUsage().InputTokens)
}
