package main

import (
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/openrouter"
	"github.com/flitsinc/go-llms/tools"
)

const integrationPatchGrammar = `start: begin_patch hunk+ end_patch
begin_patch: "*** Begin Patch" LF
end_patch: "*** End Patch" LF?

hunk: add_hunk
add_hunk: "*** Add File: " filename LF add_line+

filename: /(.+)/
add_line: "+" /(.*)/ LF -> line

%import common.LF`

// requestRecorder captures every raw provider request so the test can assert
// what actually went over the wire on each turn.
type requestRecorder struct {
	mu       sync.Mutex
	requests []string
}

func (r *requestRecorder) RawRequest(endpoint string, data []byte) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.requests = append(r.requests, string(data))
}

func (r *requestRecorder) RawEvent([]byte) {}

// TestIntegration_OpenRouterFlatCustomGrammarTool runs a Lark-grammar custom
// tool through the full client against live OpenRouter (openai/gpt-5.6-luna),
// covering the whole flat-custom-tools surface in one two-turn conversation:
//
//   - declaration: the flat Responses-style custom tool + forced tool_choice
//     are accepted (the nested Chat Completions wrapper is rejected upstream);
//   - generation: the streamed tool input is raw grammar-conformant text,
//     delivered incrementally, and constrained decoding is genuinely applied;
//   - replay: the follow-up request carries the assistant call's raw
//     arguments verbatim — the review-found bug was these being sanitized to
//     "{}" — and the upstream accepts the replayed call + result.
func TestIntegration_OpenRouterFlatCustomGrammarTool(t *testing.T) {
	key := os.Getenv("OPENROUTER_API_KEY")
	if key == "" {
		t.Skip("OPENROUTER_API_KEY required")
	}

	var toolInputs []string
	deltaCount := 0
	patchTool := tools.FuncGrammar(
		tools.Lark(integrationPatchGrammar),
		"apply_patch",
		"Create files by emitting a patch. The input is the raw patch text, not JSON.",
		"apply_patch",
		func(r tools.Runner, input string) tools.Result {
			toolInputs = append(toolInputs, input)
			return tools.SuccessFromString("created")
		},
	)

	recorder := &requestRecorder{}
	llm := llms.New(openrouter.New(key, "openai/gpt-5.6-luna"), patchTool).
		WithMaxTurns(2).
		WithDebugger(recorder)
	llm.SystemPrompt = func() content.Content {
		return content.FromText("You create files using the apply_patch tool.")
	}
	llm.SetToolChoice(tools.Choice{Mode: tools.ChoiceRequireOneOf, AllowedTools: []string{"apply_patch"}})

	done := make(chan struct{})
	go func() {
		defer close(done)
		for update := range llm.Chat("Create hello.txt containing the single line: hi from the integration test") {
			if _, ok := update.(llms.ToolDeltaUpdate); ok {
				deltaCount++
			}
		}
	}()
	select {
	case <-done:
	case <-time.After(120 * time.Second):
		t.Fatal("conversation did not complete within 120s")
	}
	if err := llm.Err(); err != nil {
		require.ErrorIs(t, err, llms.ErrMaxTurnsReached, "only the max-turns stop condition is acceptable")
	}

	// Generation: raw, grammar-conformant, streamed incrementally.
	require.NotEmpty(t, toolInputs, "the forced tool must have executed")
	firstInput := toolInputs[0]
	assert.True(t, strings.HasPrefix(firstInput, "*** Begin Patch\n"),
		"tool input must be the raw patch text, got %q", firstInput)
	assert.Contains(t, firstInput, "*** Add File: ")
	assert.False(t, strings.HasPrefix(firstInput, "{"), "input must not be JSON-wrapped")
	assert.Greater(t, deltaCount, 1, "input must stream as incremental deltas")

	// Wire: turn 1 declares the flat custom tool; turn 2 replays the call's
	// raw arguments verbatim instead of sanitizing them to "{}".
	recorder.mu.Lock()
	defer recorder.mu.Unlock()
	require.GreaterOrEqual(t, len(recorder.requests), 2, "expected a request per turn")
	turn1, turn2 := recorder.requests[0], recorder.requests[len(recorder.requests)-1]
	assert.Contains(t, turn1, `"type":"custom"`, "turn 1 must declare the custom tool")
	assert.Contains(t, turn1, `"syntax":"lark"`)
	assert.NotContains(t, turn1, `"custom":{`, "flat mode must not emit the nested wrapper")
	assert.Contains(t, turn2, "*** Add File: ", "turn 2 must replay the raw patch arguments")
	assert.NotContains(t, turn2, `"arguments":"{}"`, "raw arguments must not be sanitized away on replay")

	t.Logf("turn-1 tool input (%d deltas):\n%s", deltaCount, firstInput)
}
