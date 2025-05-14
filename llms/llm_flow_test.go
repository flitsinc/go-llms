package llms

import (
	"context"
	"testing"
	"time"

	"github.com/flitsinc/go-llms/content"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestChatFlow tests the complete chat flow with a tool call and response.
func TestChatFlow(t *testing.T) {
	// Arrange: Mock provider, LLM with test tool, and system prompt
	mockProv := &mockProvider{
		toolCallsToMake: []string{"test_tool"},
	}
	llm, prov := setupTestLLM(t, mockProv, testTool)
	require.NotNil(t, prov, "Cast to *mockProvider should succeed") // Ensure cast worked
	timeNow := time.Now()
	expectedTimeString := timeNow.Format(time.RFC3339)
	llm.SystemPrompt = func() content.Content {
		return content.Textf("This is a test system prompt. Current time: %s", expectedTimeString)
	}

	// Act: Run the chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: Check for errors and provider calls
	assert.NoError(t, llm.Err(), "No error should occur in the chat flow")
	require.True(t, prov.generateCalled, "Provider's Generate method should be called")
	require.NotNil(t, prov.systemPrompt, "System prompt should be passed to provider")
	assert.Nil(t, prov.jsonOutputSchema, "JSON schema should be nil for standard chat") // Use correct field name

	// Assert: System prompt content
	var systemPromptText string
	for _, part := range prov.systemPrompt {
		if part.Type() == content.TypeText {
			textPart, ok := part.(*content.Text)
			if ok {
				systemPromptText += textPart.Text
			}
		}
	}
	assert.Contains(t, systemPromptText, "This is a test system prompt")
	assert.Contains(t, systemPromptText, expectedTimeString)

	// Assert: Updates received
	require.Equal(t, 6, len(updates), "Should receive 6 updates")
	textUpdate, ok := updates[0].(TextUpdate)
	require.True(t, ok, "First update should be TextUpdate")
	assert.Equal(t, "This is a test message.", textUpdate.Text)

	toolStartUpdate, ok := updates[1].(ToolStartUpdate)
	require.True(t, ok, "Second update should be ToolStartUpdate")
	assert.Equal(t, "Test Tool", toolStartUpdate.Tool.Label())
	assert.Equal(t, "test_tool", toolStartUpdate.Tool.FuncName())
	assert.Equal(t, "test_tool-id-0", toolStartUpdate.ToolCallID, "ToolCallID should match the ID from the message")

	toolDoneUpdate, ok := updates[4].(ToolDoneUpdate)
	require.True(t, ok, "Update at index 4 should be ToolDoneUpdate")
	assert.Equal(t, "Test Tool", toolDoneUpdate.Tool.Label())
	assert.Equal(t, "test_tool-id-0", toolDoneUpdate.ToolCallID, "ToolCallID should match the ID from the message")
	resultJSON := extractJSONFromResult(t, toolDoneUpdate.Result)
	assert.JSONEq(t, `{"result":"Processed: test_value_test_tool"}`, string(resultJSON))

	secondTextUpdate, ok := updates[5].(TextUpdate)
	require.True(t, ok, "Update at index 5 should be TextUpdate")
	assert.Equal(t, "I've processed the results from the tool.", secondTextUpdate.Text)

	// Assert: Message history
	assert.Equal(t, 4, len(llm.lastSentMessages), "Should have 4 messages in history")
	assert.Equal(t, "user", llm.lastSentMessages[0].Role, "First message should be from user")
	assert.Equal(t, "assistant", llm.lastSentMessages[1].Role, "Second message should be from assistant (initiating tool call)")
	assert.True(t, len(llm.lastSentMessages[1].ToolCalls) > 0, "Second assistant message should contain tool calls")
	assert.Equal(t, "tool", llm.lastSentMessages[2].Role, "Third message should be from tool")
	assert.Equal(t, "assistant", llm.lastSentMessages[3].Role, "Fourth message should be from assistant (final response)")
	assert.True(t, len(llm.lastSentMessages[3].ToolCalls) == 0, "Final assistant message should not contain tool calls")
}

// TestSuccessfulChatNoError tests that a successful chat returns nil from Err().
func TestSuccessfulChatNoError(t *testing.T) {
	// Arrange: Standard provider and LLM
	mockProv := &mockProvider{}
	llm, _ := setupTestLLM(t, mockProv)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	_ = runTestChat(ctx, t, llm, "Test message")

	// Assert: No error
	assert.NoError(t, llm.Err(), "LLM.Err() should return nil for successful chat")
}

// TestChatWrapper tests the simple llms.Chat wrapper function.
func TestChatWrapper(t *testing.T) {
	// Arrange: Use a standard mock provider and LLM
	mockProv := &mockProvider{}
	llm, _ := setupTestLLM(t, mockProv)

	// Act: Run chat using the simple wrapper
	// We don't need a long timeout or context here, just verifying the call.
	// Use a simple channel read to ensure it doesn't block indefinitely.
	chatChan := llm.Chat("Simple chat message")
	select {
	case update, ok := <-chatChan:
		require.True(t, ok, "Should receive at least one update")
		_, isText := update.(TextUpdate)
		assert.True(t, isText, "First update should be text")
		// Drain the rest to prevent goroutine leak
		go func() {
			for range chatChan {
			}
		}()
	case <-time.After(1 * time.Second):
		t.Fatal("llm.Chat did not produce an update within the timeout")
	}

	// Assert: Check provider was called
	assert.NoError(t, llm.Err(), "Simple chat should not produce an error")
	assert.True(t, mockProv.generateCalled, "Provider generate should be called")
	require.Len(t, mockProv.messages, 1, "Provider should receive one message")
	assert.Equal(t, "user", mockProv.messages[0].Role)
	require.Len(t, mockProv.messages[0].Content, 1)
	textPart, ok := mockProv.messages[0].Content[0].(*content.Text)
	require.True(t, ok)
	assert.Equal(t, "Simple chat message", textPart.Text)
}

// TestNilSystemPrompt tests chat behavior when LLM.SystemPrompt is nil.
func TestNilSystemPrompt(t *testing.T) {
	// Arrange: Provider and LLM, ensure SystemPrompt is nil
	mockProv := &mockProvider{}
	llm, _ := setupTestLLM(t, mockProv)
	llm.SystemPrompt = nil // Explicitly set to nil

	// Act: Run a simple chat
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	_ = runTestChat(ctx, t, llm, "Test message")

	// Assert: No error and provider received a nil system prompt
	assert.NoError(t, llm.Err())
	assert.True(t, mockProv.generateCalled, "Provider generate should be called")
	assert.Nil(t, mockProv.systemPrompt, "Provider should receive nil system prompt")
}
