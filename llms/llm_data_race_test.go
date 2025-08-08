package llms

import (
	"context"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/flitsinc/go-llms/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// drainChannel reads from an Update channel until it's closed.
func drainChannel(ch <-chan Update) {
	for range ch {
	}
}

// TestRapidCancellation tests robustness against immediate context cancellation after starting a chat involving tools.
// This is less about the -race detector and more about ensuring cancellation paths are robust.
func TestRapidCancellation(t *testing.T) {
	mockProv := &mockProvider{
		toolCallsToMake: []string{"test_tool"}, // Ensure a tool call is attempted
	}
	// Need tool definition accessible here
	testTool := tools.Func("Test Tool", "A test tool for testing", "test_tool",
		func(r tools.Runner, p TestToolParams) tools.Result {
			return tools.SuccessWithLabel("Test Tool Ran", map[string]any{
				"result": fmt.Sprintf("Processed: %s", p.TestParam),
			})
		})

	llm, _ := setupTestLLM(t, mockProv, testTool)

	ctx, cancel := context.WithCancel(context.Background())

	// Start the chat
	chatChan := llm.ChatWithContext(ctx, "Test rapid cancellation")

	// Cancel immediately
	cancel()

	// Drain the channel
	drainChannel(chatChan)

	// Assert: Expect a cancellation error
	require.Error(t, llm.Err(), "Expected an error due to cancellation")
	// The exact error might be context.Canceled or potentially the stream error if cancellation happens mid-stream
	// assert.ErrorIs(t, llm.Err(), context.Canceled, "Error should be context.Canceled")
	assert.Contains(t, llm.Err().Error(), "context canceled", "Error message should indicate cancellation")
}

// TestCancellationDuringToolExecution verifies that cancelling the context *during*
// the execution of a tool call (within runToolCall) doesn't cause a
// "send on closed channel" panic when sending ToolDoneUpdate.
func TestCancellationDuringToolExecution(t *testing.T) {
	// 1. Slow Tool
	slowToolDuration := 150 * time.Millisecond
	slowTool := tools.Func("Slow Tool", "A tool that takes time", "slow_tool",
		func(r tools.Runner, p TestToolParams) tools.Result {
			// Simulate work, check context periodically
			select {
			case <-time.After(slowToolDuration):
				return tools.Success(map[string]any{"status": "completed"})
			case <-r.Context().Done():
				return tools.Errorf("slow tool cancelled: %v", r.Context().Err())
			}
		})

	// 2. Mock Provider to call the slow tool
	mockProv := &mockProvider{
		toolCallsToMake: []string{"slow_tool"},
	}

	// 3. LLM Setup
	llm, _ := setupTestLLM(t, mockProv, slowTool)

	// 4. Context that cancels during tool execution
	// Cancel slightly after the tool call starts but before it finishes
	cancelDelay := 75 * time.Millisecond
	ctx, cancel := context.WithCancel(context.Background())

	// 5. Execution
	chatChan := llm.ChatWithContext(ctx, "Run the slow tool")

	// Schedule cancellation
	cancelTimer := time.AfterFunc(cancelDelay, cancel)
	defer cancelTimer.Stop() // Clean up timer if test finishes early

	// Collect updates until channel closes
	var updates []Update
	for update := range chatChan {
		updates = append(updates, update)
	}

	// 6. Assertion
	// The primary assertion is that the test completes *without panicking*.
	// We also expect a context cancellation error from the LLM.
	require.Error(t, llm.Err(), "Expected an error due to cancellation")
	// The error might originate from the stream processing loop or the tool context itself
	assert.Contains(t, llm.Err().Error(), "context canceled", "LLM error should indicate cancellation")

	// Check updates received before cancellation
	// We expect TextUpdate and ToolStartUpdate. ToolDoneUpdate should *not* be sent.
	foundText := false
	foundToolStart := false
	foundToolDone := false
	for _, update := range updates {
		switch update.(type) {
		case TextUpdate:
			foundText = true
		case ToolStartUpdate:
			foundToolStart = true
		case ToolDoneUpdate:
			foundToolDone = true
		}
	}
	assert.True(t, foundText, "Should have received TextUpdate before cancellation")
	assert.True(t, foundToolStart, "Should have received ToolStartUpdate before cancellation")
	assert.False(t, foundToolDone, "Should NOT have received ToolDoneUpdate due to cancellation")
}

// TestSlowReceiverLosesToolDoneUpdate demonstrates that the `default` case in the
// select statement for sending ToolDoneUpdate in runToolCall can cause the update
// to be lost if the receiver channel is blocked, even without context cancellation.
func TestSlowReceiverLosesToolDoneUpdate(t *testing.T) {
	// 1. Use a fast tool
	fastTool := tools.Func("Fast Tool", "A tool that returns instantly", "fast_tool",
		func(r tools.Runner, p TestToolParams) tools.Result {
			return tools.Success(map[string]any{"status": "done instantly"})
		})

	// 2. Mock Provider calls the fast tool
	mockProv := &mockProvider{
		toolCallsToMake: []string{"fast_tool"},
	}

	// 3. LLM Setup
	llm, _ := setupTestLLM(t, mockProv, fastTool)

	// 4. Slow Receiver Simulation
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second) // Test timeout
	defer cancel()

	chatChan := llm.ChatWithContext(ctx, "Run fast tool with slow listener")

	receiverSleepDuration := 50 * time.Millisecond
	var updates []Update
	var wg sync.WaitGroup
	wg.Add(1)

	go func() {
		defer wg.Done()
		for update := range chatChan {
			updates = append(updates, update)
			// If we just received the start signal for our tool, pause briefly
			if _, ok := update.(ToolStartUpdate); ok {
				time.Sleep(receiverSleepDuration)
			}
		}
	}()

	// Wait for the receiver goroutine to finish processing all updates
	wg.Wait()

	// 5. Assertions
	// Check that the LLM itself didn't encounter an error (like cancellation)
	assert.NoError(t, llm.Err(), "LLM should not report an error in this scenario")

	// Check received updates: We expect Text, ToolStart, 2x ToolDelta, ToolDone, and the final Text
	// because the sends are now blocking.
	require.Len(t, updates, 6, "Expected exactly 6 updates (Text, ToolStart, 2x ToolDelta, ToolDone, Final Text)")

	foundText := false
	foundToolStart := false
	foundToolDone := false
	foundFinalText := false
	finalTextContent := ""

	for i, update := range updates {
		switch u := update.(type) {
		case TextUpdate:
			if i == 0 { // Initial text
				foundText = true
			} else { // Final text
				foundFinalText = true
				finalTextContent = u.Text // Capture the final text
			}
		case ToolStartUpdate:
			foundToolStart = true
			assert.Equal(t, "fast_tool", u.Tool.FuncName())
		case ToolDoneUpdate:
			foundToolDone = true
		}
	}

	assert.True(t, foundText, "Should have received the initial TextUpdate")
	assert.True(t, foundToolStart, "Should have received the ToolStartUpdate")
	assert.True(t, foundToolDone, "Should have received the ToolDoneUpdate")
	assert.True(t, foundFinalText, "Should have received the final TextUpdate after the tool sequence")
	// The final text will be the default mock response after processing tool results
	assert.Equal(t, "I've processed the results from the tool.", finalTextContent, "Final text content mismatch")
}
