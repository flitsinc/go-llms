package openai

import (
	"context"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/llms"
)

// xaiHostedXSearchSSE mirrors a real xAI Responses stream where the x_search Agent Tool runs
// its server-side x_user_search sub-tool (a custom_tool_call with call_id prefixed "xs_") and
// then completes with a final message. Before the fix the turn loop tried to execute
// x_user_search against the (empty) client toolbox and failed with "tool not found".
const xaiHostedXSearchSSE = `data: {"type":"response.created","response":{"id":"resp_1"}}

data: {"type":"response.output_item.added","item":{"type":"web_search_call","id":"ws_1","status":"in_progress","action":{"type":"search","query":"Paul Graham Y Combinator"}}}

data: {"type":"response.output_item.done","item":{"type":"web_search_call","id":"ws_1","status":"completed","action":{"type":"search","query":"Paul Graham Y Combinator"}}}

data: {"type":"response.output_item.added","item":{"type":"custom_tool_call","id":"ctc_1","call_id":"xs_call-abc-0","name":"x_user_search","input":"","status":"in_progress"},"output_index":1}

data: {"type":"response.custom_tool_call_input.delta","item_id":"ctc_1","delta":"{\"query\":\"paulg\"}"}

data: {"type":"response.custom_tool_call_input.done","item_id":"ctc_1"}

data: {"type":"response.output_item.done","item":{"type":"custom_tool_call","id":"ctc_1","call_id":"xs_call-abc-0","name":"x_user_search","status":"completed"}}

data: {"type":"response.output_item.added","item":{"type":"message","id":"msg_1","status":"in_progress"}}

data: {"type":"response.output_text.delta","item_id":"msg_1","content_index":0,"delta":"Paul Graham co-founded Y Combinator."}

data: {"type":"response.output_item.done","item":{"type":"message","id":"msg_1","role":"assistant","status":"completed","content":[{"type":"output_text","text":"Paul Graham co-founded Y Combinator."}]}}

data: {"type":"response.completed"}

`

// The parser must not surface xAI's hosted x_search sub-tool calls as client tool calls.
func TestResponsesXAIHostedToolCallNotSurfaced(t *testing.T) {
	stream := &ResponsesStream{ctx: context.Background(), model: "grok-4.3", stream: strings.NewReader(xaiHostedXSearchSSE)}
	for range stream.Iter() {
	}
	require.NoError(t, stream.Err())
	assert.Empty(t, stream.Message().ToolCalls, "hosted x_search sub-tool calls must not surface as client tool calls")
}

// The full turn loop must complete with the final message instead of failing "tool not found".
func TestResponsesXAIHostedToolCallTurnLoop(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, xaiHostedXSearchSSE)
	}))
	defer ts.Close()

	llm := llms.New(NewResponsesAPI("k", "grok-4.3").WithEndpoint(ts.URL, "xAI")).WithMaxTurns(1)

	var text strings.Builder
	for update := range llm.ChatWithContext(context.Background(), "Who is @paulg?") {
		if tu, ok := update.(llms.TextUpdate); ok {
			text.WriteString(tu.Text)
		}
	}

	require.NoError(t, llm.Err())
	assert.Contains(t, text.String(), "Y Combinator")
}

// The turn loop must surface the provider-run web_search and x_search as SearchUpdates, carrying
// their queries, so a UI can show what the model looked up.
func TestResponsesXAISearchActivitySurfaced(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, xaiHostedXSearchSSE)
	}))
	defer ts.Close()

	llm := llms.New(NewResponsesAPI("k", "grok-4.3").WithEndpoint(ts.URL, "xAI")).WithMaxTurns(1)

	var searches []llms.SearchActivity
	for update := range llm.ChatWithContext(context.Background(), "Who is @paulg?") {
		if su, ok := update.(llms.SearchUpdate); ok {
			searches = append(searches, su.SearchActivity)
		}
	}

	require.NoError(t, llm.Err())
	require.Len(t, searches, 2)
	assert.Equal(t, llms.SearchActivity{Source: "web", Query: "Paul Graham Y Combinator"}, searches[0])
	assert.Equal(t, llms.SearchActivity{Source: "x", Query: "paulg"}, searches[1])
}
