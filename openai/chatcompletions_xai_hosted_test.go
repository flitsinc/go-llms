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

// openRouterHostedXSearchSSE mirrors an OpenRouter chat-completions stream (provider xAI) where the
// x_search Agent Tool runs its x_user_search sub-tool server-side and streams it as a tool_call with
// an "xs_"-prefixed id, then continues to the final answer with finish_reason "tool_calls". Without
// the hosted-tool skip the turn loop tries to execute x_user_search and fails "tool not found".
const openRouterHostedXSearchSSE = `data: {"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"xs_call-abc-0","type":"function","function":{"name":"x_user_search","arguments":""}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"function":{"arguments":"{\"query\":\"paulg\"}"}}]},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"role":"assistant","content":"Paul Graham co-founded Y Combinator."},"finish_reason":null}]}

data: {"choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":"tool_calls"}]}

data: [DONE]

`

// The turn loop must complete with the final answer instead of trying to run the hosted x_user_search.
func TestChatCompletionsHostedXSearchNotExecuted(t *testing.T) {
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, openRouterHostedXSearchSSE)
	}))
	defer ts.Close()

	llm := llms.New(New("k", "grok-4.3").WithEndpoint(ts.URL, "OpenRouter")).WithMaxTurns(1)

	var text strings.Builder
	for update := range llm.ChatWithContext(context.Background(), "Who is @paulg?") {
		if tu, ok := update.(llms.TextUpdate); ok {
			text.WriteString(tu.Text)
		}
	}

	// No error means the hosted x_user_search was skipped, not looked up in the (empty) toolbox.
	require.NoError(t, llm.Err())
	assert.Contains(t, text.String(), "Y Combinator")
}
