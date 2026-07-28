package llms

import (
	"context"
	"testing"
	"time"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestErrorHandling tests that errors from the ProviderStream are propagated correctly.
func TestErrorHandling(t *testing.T) {
	// Arrange: Error provider and LLM
	errorProvider := &errorMockProvider{
		errorMessage: "test provider stream error",
	}
	llm := New(errorProvider)

	// Act: Run chat (will error)
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	_ = runTestChat(ctx, t, llm, "Test message")

	// Assert: Check for the expected error
	require.Error(t, llm.Err(), "LLM.Err() should return an error")
	assert.Contains(t, llm.Err().Error(), "test provider stream error", "Error should contain provider's error message")
	assert.Contains(t, llm.Err().Error(), "LLM returned error response", "Error should indicate it came from the LLM layer")
}

// TestEmptyToolCallIDError tests that an empty ToolCall ID from the stream causes an error.
func TestEmptyToolCallIDError(t *testing.T) {
	// Arrange: Empty ID provider and LLM
	mockProv := &mockEmptyIDProvider{}
	llm := New(mockProv, testTool)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: Limited updates before error
	require.Equal(t, 1, len(updates), "Should receive exactly 1 update")
	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "First update should be TextUpdate")

	// Assert: Correct error details
	require.Error(t, llm.Err(), "LLM.Err() should return an error")
	assert.Contains(t, llm.Err().Error(), "missing tool call ID", "Error should mention missing tool call ID")
	assert.Contains(t, llm.Err().Error(), "test_tool", "Error should include the tool name")
}

// TestTurnToolNotFound tests that calling a tool that is not in the toolbox
// produces an error tool result the model can recover from, instead of aborting
// the turn.
func TestTurnToolNotFound(t *testing.T) {
	// Arrange: Provider that calls a non-existent tool, LLM with *some other* tool
	provider := &mockProvider{toolCallsToMake: []string{"tool_does_not_exist"}}
	llm, _ := setupTestLLM(t, provider, testTool) // LLM has 'test_tool', provider calls 'tool_does_not_exist'

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: The turn ran to completion and continued into a second turn.
	assert.NoError(t, llm.Err(), "An unknown tool should not abort the chat")

	// Turn 1: Text, ToolStart, 2xToolDelta, ToolDone. Turn 2: Text.
	require.Len(t, updates, 6)
	_, isText := updates[0].(TextUpdate)
	assert.True(t, isText, "First update should be text")

	startUpdate, ok := updates[1].(ToolStartUpdate)
	require.True(t, ok, "Update 1 should be ToolStartUpdate")
	require.NotNil(t, startUpdate.Tool, "Unknown tools still need a Tool for consumers to display")
	assert.True(t, tools.IsUnknown(startUpdate.Tool), "The started tool should be flagged as unknown")
	assert.Equal(t, "tool_does_not_exist", startUpdate.Tool.FuncName())

	doneUpdate, ok := updates[4].(ToolDoneUpdate)
	require.True(t, ok, "Update 4 should be ToolDoneUpdate")
	assert.True(t, tools.IsUnknown(doneUpdate.Tool), "The finished tool should be flagged as unknown")
	require.Error(t, doneUpdate.Result.Error(), "The result should carry the not-found error")
	assert.Contains(t, doneUpdate.Result.Error().Error(), "tool \"tool_does_not_exist\" not found")
	var notFound *tools.NotFoundError
	require.ErrorAs(t, doneUpdate.Result.Error(), &notFound, "The error should be identifiable as a not-found error")
	assert.Equal(t, "tool_does_not_exist", notFound.FuncName)

	_, isText = updates[5].(TextUpdate)
	assert.True(t, isText, "The model should have gotten another turn after the error result")

	// Assert: The error was fed back to the model as a tool result.
	var toolMessages []Message
	for _, msg := range provider.messages {
		if msg.Role == "tool" {
			toolMessages = append(toolMessages, msg)
		}
	}
	require.Len(t, toolMessages, 1, "The unknown call should still produce a tool result")
	assert.Equal(t, "tool_does_not_exist", toolMessages[0].ToolCallName)
	assert.True(t, toolMessages[0].IsError, "The tool result should be marked as an error")
}

// TestTurnUnknownToolLoopStops tests that a model which only ever calls tools
// that don't exist eventually gives up instead of looping forever.
func TestTurnUnknownToolLoopStops(t *testing.T) {
	// Arrange: Provider that calls a non-existent tool on every single turn.
	provider := &mockAlwaysUnknownToolProvider{}
	llm, _ := setupTestLLM(t, provider, testTool)
	llm.WithMaxUnknownToolTurns(2)

	var turnSuccess []bool
	llm.TrackUsage = func(ctx context.Context, usage Usage, success bool) {
		turnSuccess = append(turnSuccess, success)
	}

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: Stopped after the configured number of turns, with a clear error.
	require.ErrorIs(t, llm.Err(), ErrTooManyUnknownTools)
	assert.Equal(t, 2, provider.generateCount, "Should have stopped after 2 unknown-only turns")
	// Each turn: Text, ToolStart, ToolDelta, ToolDone.
	assert.Len(t, updates, 8)

	// Assert: The turn that gave up is reported as unsuccessful, so usage
	// tracking agrees with the error the caller sees.
	assert.Equal(t, []bool{true, false}, turnSuccess)
}

// TestTurnUnknownToolLoopResetsBetweenChats tests that the unknown tool streak
// doesn't leak into the next chat on the same LLM, which would cut that chat off
// on its first unknown tool call.
func TestTurnUnknownToolLoopResetsBetweenChats(t *testing.T) {
	// Arrange: Provider that calls a non-existent tool on every single turn.
	provider := &mockAlwaysUnknownToolProvider{}
	llm, _ := setupTestLLM(t, provider, testTool)
	llm.WithMaxUnknownToolTurns(2)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	// Act: Run a chat that exhausts the allowance, then start a fresh one.
	_ = runTestChat(ctx, t, llm, "Test message")
	require.ErrorIs(t, llm.Err(), ErrTooManyUnknownTools)
	require.Equal(t, 2, provider.generateCount)

	_ = runTestChat(ctx, t, llm, "Another message")

	// Assert: The second chat got the full allowance again, rather than
	// stopping after a single turn.
	require.ErrorIs(t, llm.Err(), ErrTooManyUnknownTools)
	assert.Equal(t, 4, provider.generateCount, "The new chat should start the streak over")
}

// TestTurnUnknownToolLoopResets tests that turns which call a real tool reset
// the unknown tool counter, so recovering models aren't cut off.
func TestTurnUnknownToolLoopResets(t *testing.T) {
	// Arrange: Provider that alternates between an unknown tool and a real one.
	provider := &mockAlwaysUnknownToolProvider{realToolEveryOtherTurn: true, maxTurns: 6}
	llm, _ := setupTestLLM(t, provider, testTool)
	llm.WithMaxUnknownToolTurns(2)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = runTestChat(ctx, t, llm, "Test message")

	// Assert: The chat was never cut short by the unknown tool guard.
	assert.NoError(t, llm.Err())
	assert.Equal(t, 6, provider.generateCount)
}

// TestRunToolCallWithError tests the behavior when a called tool returns an error.
func TestRunToolCallWithError(t *testing.T) {
	// Arrange: Provider that calls the erroring tool
	mockProv := &mockProvider{
		toolCallsToMake: []string{"error_tool"},
	}
	llm, _ := setupTestLLM(t, mockProv, mockToolWithError)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: No LLM-level error (tool error shouldn't stop the flow)
	assert.NoError(t, llm.Err(), "LLM should not error just because a tool failed")

	// Assert: Correct updates received.
	// Turn 1: Text, ToolStart, 2xToolDelta, ToolDone
	// Turn 2: Final Text
	require.Equal(t, 6, len(updates), "Should receive 6 updates")
	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "Update 0 should be TextUpdate")
	_, ok = updates[1].(ToolStartUpdate)
	require.True(t, ok, "Update 1 should be ToolStartUpdate")

	// ToolDeltaUpdates at index 2 and 3 are skipped by these checks

	doneUpdate, ok := updates[4].(ToolDoneUpdate)
	require.True(t, ok, "Update 4 should be ToolDoneUpdate")

	_, ok = updates[5].(TextUpdate)
	require.True(t, ok, "Update 5 should be TextUpdate")

	// Assert: ToolDoneUpdate contains the error
	assert.Equal(t, "error_tool", doneUpdate.Tool.FuncName())
	require.NotNil(t, doneUpdate.Result, "Result should not be nil")
	assert.Error(t, doneUpdate.Result.Error(), "Result error should not be nil")
	assert.Contains(t, doneUpdate.Result.Error().Error(), "internal tool error detail", "Result error should contain tool's internal error")
	assert.Equal(t, "Error: internal tool error detail", doneUpdate.Result.Label(), "Result label should match the error message") // Default label from Error()
	// Check JSON representation - only contains the detail error
	resultJSON := extractJSONFromResult(t, doneUpdate.Result)
	assert.JSONEq(t, `{"error":"internal tool error detail"}`, string(resultJSON))

	// Assert: Message history includes the tool result message with error
	require.Len(t, llm.lastSentMessages, 4, "Should have 4 messages in history")
	toolResultMessage := llm.lastSentMessages[2]
	assert.Equal(t, "tool", toolResultMessage.Role)
	require.NotNil(t, toolResultMessage.Content)
	require.Len(t, toolResultMessage.Content, 1, "Tool result content should have 1 part")
	jsonPart, ok := toolResultMessage.Content[0].(*content.JSON)
	require.True(t, ok, "Tool result content should be JSON")
	// Check JSON in history - only contains the detail error
	assert.JSONEq(t, `{"error":"internal tool error detail"}`, string(jsonPart.Data))
}
