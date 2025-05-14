package llms

import (
	"context"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestMaxTurnsReached tests that WithMaxTurns limits turns and returns ErrMaxTurnsReached.
func TestMaxTurnsReached(t *testing.T) {
	// Arrange: Mock provider forcing tool calls, LLM with MaxTurns(1)
	mockProv := &mockProvider{
		toolCallsToMake: []string{"test_tool"},
	}
	llm, _ := setupTestLLM(t, mockProv, testTool)
	llm.WithMaxTurns(1)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: Updates received (Text, ToolStart, 2xToolDelta, ToolDone)
	// MaxTurns(1) means the first turn completes, including tool execution, then stops.
	require.Equal(t, 5, len(updates), "Should receive exactly 5 updates")
	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "First update should be TextUpdate")
	_, ok = updates[1].(ToolStartUpdate)
	require.True(t, ok, "Second update should be ToolStartUpdate")

	// ToolDeltaUpdates at index 2 and 3 are skipped by these checks

	_, ok = updates[4].(ToolDoneUpdate)
	require.True(t, ok, "Update at index 4 should be ToolDoneUpdate")

	// Assert: Max turns error and turn count
	require.Error(t, llm.Err(), "LLM.Err() should return an error")
	assert.ErrorIs(t, llm.Err(), ErrMaxTurnsReached, "Error should be ErrMaxTurnsReached")
	assert.Equal(t, 1, llm.turns, "Turn count should be 1")
}

// TestMaxTurnsAllowsCompletion tests that an LLM with sufficient max turns completes successfully.
func TestMaxTurnsAllowsCompletion(t *testing.T) {
	// Arrange: Mock provider forcing tool calls, LLM with MaxTurns(2)
	mockProv := &mockProvider{
		toolCallsToMake: []string{"test_tool"},
	}
	llm, _ := setupTestLLM(t, mockProv, testTool)
	llm.WithMaxTurns(2)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: Updates received (Turn1: Text, TS, 2xTDelta, TD; Turn2: Text)
	// Total 5 + 1 = 6 updates
	require.Equal(t, 6, len(updates), "Should receive exactly 6 updates")

	// Individual checks for T1 Text, TS, TD could be added if needed, but the key is the final text
	// For minimality, we only adjust the final text index and overall count.

	finalTextUpdate, ok := updates[5].(TextUpdate)
	require.True(t, ok, "Final update (index 5) should be TextUpdate")
	assert.Equal(t, "I've processed the results from the tool.", finalTextUpdate.Text)

	// Assert: No error and correct turn count
	assert.NoError(t, llm.Err(), "No error should occur in this test")
	assert.Equal(t, 2, llm.turns, "Turn count should be 2")
}
