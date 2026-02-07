package llms

import (
	"context"
	"testing"
	"time"

	"github.com/flitsinc/go-llms/content"
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

// TestTurnToolNotFound tests the error path when a called tool is not in the toolbox.
func TestTurnToolNotFound(t *testing.T) {
	// Arrange: Provider that yields a non-existent tool, LLM with *some other* tool
	provider := &mockProviderToolNotFound{}
	llm, _ := setupTestLLM(t, provider, testTool) // LLM has 'test_tool', provider calls 'tool_does_not_exist'

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: Should get text update, then an error
	require.NotEmpty(t, updates, "Should receive at least one update before error")
	require.Len(t, updates, 1)
	_, isText := updates[0].(TextUpdate)
	assert.True(t, isText, "First update should be text")

	require.Error(t, llm.Err(), "LLM should have an error")
	assert.Contains(t, llm.Err().Error(), "tool \"tool_does_not_exist\" not found", "Error message should indicate tool not found")
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
