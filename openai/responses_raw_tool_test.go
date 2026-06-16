package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
)

// TestResponsesRawToolSerialized verifies a RawTool is emitted verbatim into the
// request's "tools" array, so callers can express provider-specific server-side
// tools (e.g. xAI's web_search / x_search) that have no dedicated type here.
func TestResponsesRawToolSerialized(t *testing.T) {
	var capturedBody []byte
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		capturedBody, _ = io.ReadAll(r.Body)
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.completed\"}\n\n")
	}))
	defer ts.Close()

	stream := NewResponsesAPI("key", "grok-4.3").
		WithEndpoint(ts.URL, "xAI").
		WithTool(RawTool{Body: map[string]any{"type": "web_search"}}).
		WithTool(RawTool{Body: map[string]any{"type": "x_search", "allowed_x_handles": []string{"paulg"}}}).
		Generate(context.Background(), content.FromText("system"), []llms.Message{{
			Role:    "user",
			Content: content.FromText("hi"),
		}}, nil, nil)

	for range stream.Iter() {
	}
	require.NoError(t, stream.Err())

	var payload struct {
		Tools []map[string]any `json:"tools"`
	}
	require.NoError(t, json.Unmarshal(capturedBody, &payload))
	require.Len(t, payload.Tools, 2)
	assert.Equal(t, "web_search", payload.Tools[0]["type"])
	assert.Equal(t, "x_search", payload.Tools[1]["type"])
	assert.Equal(t, []any{"paulg"}, payload.Tools[1]["allowed_x_handles"])
}

// TestXAIResponsesLiveSearch is an opt-in end-to-end check against the live xAI
// Responses API: a RawTool expresses xAI's web_search + x_search Agent Tools and
// the streamed answer parses back. Skipped unless XAI_API_KEY is set, so normal
// CI never hits the network.
func TestXAIResponsesLiveSearch(t *testing.T) {
	apiKey := os.Getenv("XAI_API_KEY")
	if apiKey == "" {
		t.Skip("XAI_API_KEY not set; skipping live xAI Responses test")
	}

	provider := NewResponsesAPI(apiKey, "grok-4.3").
		WithEndpoint("https://api.x.ai/v1/responses", "xAI").
		WithMaxOutputTokens(2048).
		WithTool(RawTool{Body: map[string]any{"type": "web_search"}}).
		WithTool(RawTool{Body: map[string]any{"type": "x_search", "allowed_x_handles": []string{"paulg"}}})

	ctx, cancel := context.WithTimeout(context.Background(), 3*time.Minute)
	defer cancel()

	stream := provider.Generate(ctx, content.FromText("You research people from public web and X signals."), []llms.Message{{
		Role:    "user",
		Content: content.FromText("Who is @paulg on X? Answer in one sentence and include the source URL you used."),
	}}, nil, nil)

	// Text() returns the latest delta on each StreamStatusText, so accumulate.
	var sb strings.Builder
	for status := range stream.Iter() {
		if status == llms.StreamStatusText {
			sb.WriteString(stream.Text())
		}
	}
	require.NoError(t, stream.Err())

	got := strings.TrimSpace(sb.String())
	t.Logf("xAI live response: %s", got)
	require.NotEmpty(t, got, "expected a non-empty answer from xAI Responses + Agent Tools")
	assert.Contains(t, got, "Graham", "expected the answer to identify Paul Graham")
}
