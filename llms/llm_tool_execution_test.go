package llms

import (
	"context"
	"encoding/json"
	"fmt"
	"strings"
	"testing"
	"time"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestToolCallIDsFlow tests that tool call IDs flow correctly through the update chain.
func TestToolCallIDsFlow(t *testing.T) {
	// Arrange: Mock provider forcing multiple tool calls
	mockProv := &mockProvider{
		toolCallsToMake: []string{"tool1", "tool2", "tool3"},
	}

	// Define tools locally within the test
	type Tool1Params struct {
		TestParam string `json:"test_param"`
	}
	type Tool2Params struct {
		TestParam string `json:"test_param"`
	}
	type Tool3Params struct {
		TestParam string `json:"test_param"`
	}
	tool1 := tools.Func("Tool 1", "Test tool 1", "tool1", func(r tools.Runner, p Tool1Params) tools.Result {
		return tools.Success(map[string]any{"result": "tool1 result"})
	})
	tool2 := tools.Func("Tool 2", "Test tool 2", "tool2", func(r tools.Runner, p Tool2Params) tools.Result {
		return tools.Success(map[string]any{"result": "tool2 result"})
	})
	tool3 := tools.Func("Tool 3", "Test tool 3", "tool3", func(r tools.Runner, p Tool3Params) tools.Result {
		return tools.Success(map[string]any{"result": "tool3 result"})
	})

	llm, _ := setupTestLLM(t, mockProv, tool1, tool2, tool3)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test multiple tool calls")

	// Assert: No error
	assert.NoError(t, llm.Err(), "No error should occur in this test")

	// Assert: Correct number and type of updates
	require.Equal(t, 8, len(updates), "Should receive exactly 8 updates")
	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "First update should be TextUpdate")
	_, ok = updates[7].(TextUpdate)
	require.True(t, ok, "Final update should be TextUpdate")

	// Assert: Tool call ID propagation and uniqueness
	var lastToolCallID string
	for i := 1; i < 7; i += 2 { // Iterate through start/done pairs
		start, startOK := updates[i].(ToolStartUpdate)
		done, doneOK := updates[i+1].(ToolDoneUpdate)
		require.True(t, startOK, "Update %d should be ToolStartUpdate", i)
		require.True(t, doneOK, "Update %d should be ToolDoneUpdate", i+1)

		expectedToolName := fmt.Sprintf("tool%d", (i/2)+1)
		expectedID := fmt.Sprintf("%s-id-%d", expectedToolName, i/2)

		assert.Equal(t, expectedToolName, start.Tool.FuncName(), "Tool name mismatch in start update %d", i)
		assert.Equal(t, expectedID, start.ToolCallID, "ToolCallID mismatch in start update %d", i)
		assert.Equal(t, expectedToolName, done.Tool.FuncName(), "Tool name mismatch in done update %d", i+1)
		assert.Equal(t, start.ToolCallID, done.ToolCallID, "ToolCallID should match in start/done pair %d", (i/2)+1)

		if i > 1 {
			assert.NotEqual(t, lastToolCallID, start.ToolCallID, "ToolCallIDs should be unique (pair %d vs previous)", (i/2)+1)
		}
		lastToolCallID = start.ToolCallID
	}
}

// TestRunToolCallImageHandling tests returning content including images.
func TestRunToolCallImageHandling(t *testing.T) {
	// Arrange: Image tool that uses SuccessWithContent
	imageTestTool := tools.Func("Image Tool", "A tool that returns an image", "image_tool",
		func(r tools.Runner, p TestToolParams) tools.Result {
			// Simulate generating JSON and adding an image
			jsonData, _ := json.Marshal(map[string]string{"status": "image_included"})
			resultContent := content.FromRawJSON(jsonData)
			// In a real tool, you might use content.ImageToDataURI here
			resultContent.AddImage("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII=")
			// Use SuccessWithContent to return multiple content types with a label
			return tools.SuccessWithContent("Generated test image", resultContent)
		})

	// Arrange: Mock provider configured to call the image tool
	mockProv := &mockProvider{
		toolCallsToMake: []string{"image_tool"},
	}
	llm, _ := setupTestLLM(t, mockProv, imageTestTool)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Generate a test image")

	// Assert: No error
	assert.NoError(t, llm.Err(), "Should not return an error")

	// Assert: ToolDoneUpdate has the correct content
	require.Len(t, updates, 4, "Should have 4 updates")
	doneUpdate, ok := updates[2].(ToolDoneUpdate)
	require.True(t, ok, "Update 2 should be ToolDoneUpdate")
	assert.Equal(t, "Generated test image", doneUpdate.Result.Label())
	require.NoError(t, doneUpdate.Result.Error())
	require.NotNil(t, doneUpdate.Result.Content())
	require.Len(t, doneUpdate.Result.Content(), 2, "Result content should have 2 parts (JSON + Image)")

	// Check JSON part
	jsonPart, ok := doneUpdate.Result.Content()[0].(*content.JSON)
	require.True(t, ok, "First part should be JSON")
	assert.JSONEq(t, `{"status":"image_included"}`, string(jsonPart.Data))

	// Check Image part
	imgPart, ok := doneUpdate.Result.Content()[1].(*content.ImageURL)
	require.True(t, ok, "Second part should be ImageURL")
	assert.True(t, strings.HasPrefix(imgPart.URL, "data:image/png;base64"), "Image URL should be data URI")

	// Assert: Message history contains the tool message with both content types
	require.Len(t, llm.lastSentMessages, 4, "Should have 4 messages in history")
	toolMsg := llm.lastSentMessages[2]
	assert.Equal(t, "tool", toolMsg.Role)
	require.Len(t, toolMsg.Content, 2, "Tool message content should have 2 parts")
	_, jsonOK := toolMsg.Content[0].(*content.JSON)
	_, imgOK := toolMsg.Content[1].(*content.ImageURL)
	assert.True(t, jsonOK, "First part in history message should be JSON")
	assert.True(t, imgOK, "Second part in history message should be ImageURL")
}

// TestToolStatusUpdates verifies the basic flow when a tool (potentially with status updates) runs.
// NOTE: This test *does not* verify the reception of ToolStatusUpdate messages,
// as that requires mocking the internal runner callback mechanism.
// It verifies that runToolCall is exercised with a tool call.
func TestToolStatusUpdates(t *testing.T) {
	// Arrange: Provider that calls the status tool
	mockProv := &mockProvider{
		toolCallsToMake: []string{"status_tool"},
	}
	llm, _ := setupTestLLM(t, mockProv, mockToolForStatusTest)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: No LLM-level error
	assert.NoError(t, llm.Err())

	// Assert: Correct basic updates received (Text, ToolStart, ToolDone, Final Text)
	require.Equal(t, 4, len(updates), "Should receive 4 standard updates")
	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "Update 0 should be TextUpdate")
	startUpdate, ok := updates[1].(ToolStartUpdate)
	require.True(t, ok, "Update 1 should be ToolStartUpdate")
	doneUpdate, ok := updates[2].(ToolDoneUpdate)
	require.True(t, ok, "Update 2 should be ToolDoneUpdate")
	_, ok = updates[3].(TextUpdate)
	require.True(t, ok, "Update 3 should be TextUpdate")

	// Assert: ToolDoneUpdate is correct
	assert.Equal(t, "status_tool", doneUpdate.Tool.FuncName())
	assert.Equal(t, startUpdate.ToolCallID, doneUpdate.ToolCallID, "Done ID should match start ID")
	assert.NoError(t, doneUpdate.Result.Error())
	resultJSON := extractJSONFromResult(t, doneUpdate.Result)
	assert.JSONEq(t, `{"status":"done"}`, string(resultJSON))
}

// TestToolCallInContext verifies that the ToolCall struct is correctly placed in the context
// and accessible by the tool function via the runner.
func TestToolCallInContext(t *testing.T) {
	// Arrange: Provider that calls the context checking tool
	expectedToolCallID := "context_checker_tool-id-0"
	mockProv := &mockProvider{
		toolCallsToMake: []string{"context_checker_tool"},
	}
	llm, _ := setupTestLLM(t, mockProv, toolThatChecksContext)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: No LLM-level error
	assert.NoError(t, llm.Err())

	// Assert: Correct updates received (Text, ToolStart, ToolDone, Final Text)
	require.Equal(t, 4, len(updates), "Should receive 4 standard updates")
	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "Update 0 should be TextUpdate")
	startUpdate, ok := updates[1].(ToolStartUpdate)
	require.True(t, ok, "Update 1 should be ToolStartUpdate")
	assert.Equal(t, expectedToolCallID, startUpdate.ToolCallID, "ToolCallID in start update should match expected")
	doneUpdate, ok := updates[2].(ToolDoneUpdate)
	require.True(t, ok, "Update 2 should be ToolDoneUpdate")
	_, ok = updates[3].(TextUpdate)
	require.True(t, ok, "Update 3 should be TextUpdate")

	// Assert: ToolDoneUpdate contains the correct ToolCallID extracted from the context
	assert.Equal(t, "context_checker_tool", doneUpdate.Tool.FuncName())
	assert.Equal(t, expectedToolCallID, doneUpdate.ToolCallID, "ToolCallID in done update should match expected")
	require.NoError(t, doneUpdate.Result.Error())
	resultJSON := extractJSONFromResult(t, doneUpdate.Result)
	assert.JSONEq(t, fmt.Sprintf(`{"tool_call_id":%q}`, expectedToolCallID), string(resultJSON))
}
