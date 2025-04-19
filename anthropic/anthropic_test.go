package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
	"testing"

	"github.com/blixt/go-llms/content"
	"github.com/blixt/go-llms/llms"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Helper to create SSE data lines
func sseEvent(data any) string {
	jsonData, err := json.Marshal(data)
	if err != nil {
		panic(fmt.Sprintf("Failed to marshal SSE event data: %v", err))
	}
	// Anthropic uses event: event_type\ndata: json_payload\n\n format
	// We just need the data: part for the scanner used in Iter
	return fmt.Sprintf("data: %s\n\n", string(jsonData))
}

// Helper to create a NopCloser for testing streams
type stringNopCloser struct {
	io.Reader
}

func (s *stringNopCloser) Close() error { return nil }

func newTestStream(content string) io.ReadCloser {
	return &stringNopCloser{strings.NewReader(content)}
}

// newTestAnthropicStream is a helper for creating Anthropic streams for testing,
// ensuring required fields like context are initialized.
func newTestAnthropicStream(ctx context.Context, model, content string) *Stream {
	if ctx == nil {
		ctx = context.Background() // Default to background context if nil
	}
	return &Stream{
		ctx:    ctx, // Initialize context
		model:  model,
		stream: newTestStream(content),
	}
}

func TestAnthropicStreamHandling(t *testing.T) {
	t.Run("Zero Argument Tool Call", func(t *testing.T) {
		var streamContent strings.Builder

		// Simulate Anthropic SSE stream for a zero-arg tool call
		streamContent.WriteString(sseEvent(streamEvent{
			Type: "message_start",
			Message: &messageEvent{
				Role:  "assistant",
				Usage: &usage{InputTokens: 10, OutputTokens: 1},
			},
		}))
		streamContent.WriteString(sseEvent(streamEvent{
			Type:  "content_block_start",
			Index: 0,
			ContentBlock: &contentBlock{
				Type:  "tool_use",
				ID:    "toolu_01",
				Name:  "getDiagnostics",
				Input: json.RawMessage(`{}`), // Key: Anthropic sends {} for zero args
			},
		}))
		streamContent.WriteString(sseEvent(streamEvent{
			Type:  "content_block_stop",
			Index: 0,
		}))
		streamContent.WriteString(sseEvent(streamEvent{
			Type: "message_delta",
			Delta: delta{
				StopReason: "tool_use",
				Usage:      &usage{OutputTokens: 5}, // Simulate usage delta
			},
		}))
		streamContent.WriteString(sseEvent(streamEvent{
			Type: "message_stop",
		}))

		stream := newTestAnthropicStream(context.Background(), "claude-3-haiku", streamContent.String())
		var yieldedStatuses []llms.StreamStatus
		var toolCallAtBegin llms.ToolCall
		var toolCallAtReady llms.ToolCall

		iter := stream.Iter()
		iter(func(status llms.StreamStatus) bool {
			yieldedStatuses = append(yieldedStatuses, status)
			// Capture tool call state at relevant points
			if status == llms.StreamStatusToolCallBegin {
				toolCallAtBegin = stream.ToolCall()
			}
			if status == llms.StreamStatusToolCallReady {
				toolCallAtReady = stream.ToolCall()
			}
			return true
		})

		require.NoError(t, stream.Err(), "Stream iteration should succeed")

		// Assertions
		expectedStatuses := []llms.StreamStatus{
			llms.StreamStatusToolCallBegin,
			llms.StreamStatusToolCallReady,
		}
		assert.Equal(t, expectedStatuses, yieldedStatuses, "Expected stream statuses sequence")

		require.NotNil(t, stream.Message(), "Final message should not be nil")
		msg := stream.Message()
		assert.Equal(t, "assistant", msg.Role, "Message role should be assistant")
		require.Len(t, msg.ToolCalls, 1, "Should have one tool call")

		finalToolCall := msg.ToolCalls[0]
		assert.Equal(t, "toolu_01", finalToolCall.ID, "Tool call ID mismatch")
		assert.Equal(t, "getDiagnostics", finalToolCall.Name, "Tool call name mismatch")
		assert.JSONEq(t, `{}`, string(finalToolCall.Arguments), "Final tool call arguments should be '{}'")

		// Also check the ToolCall state captured during iteration
		assert.Equal(t, "toolu_01", toolCallAtBegin.ID, "Begin tool call ID mismatch")
		assert.JSONEq(t, `{}`, string(toolCallAtBegin.Arguments), "Begin tool call arguments should be '{}'")

		assert.Equal(t, "toolu_01", toolCallAtReady.ID, "Ready tool call ID mismatch")
		assert.JSONEq(t, `{}`, string(toolCallAtReady.Arguments), "Ready tool call arguments should be '{}'")

		inTokens, outTokens := stream.Usage()
		assert.Equal(t, 10, inTokens, "Input tokens mismatch")
		// Note: Output tokens accumulate: 1 (start) + 5 (delta) = 6
		assert.Equal(t, 6, outTokens, "Output tokens mismatch")
	})

	t.Run("Simple Argument Tool Call (Single Delta)", func(t *testing.T) {
		var streamContent strings.Builder
		toolArgsJSON := `{"query": "search terms"}`

		streamContent.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
		streamContent.WriteString(sseEvent(streamEvent{
			Type:  "content_block_start",
			Index: 0,
			ContentBlock: &contentBlock{
				Type:  "tool_use",
				ID:    "toolu_02",
				Name:  "searchTool",
				Input: json.RawMessage(`{}`), // Starts with empty object
			},
		}))
		// Simulate arguments arriving in the *first* delta
		streamContent.WriteString(sseEvent(streamEvent{
			Type:  "content_block_delta",
			Index: 0,
			Delta: delta{
				Type:        "input_json_delta",
				PartialJSON: toolArgsJSON,
			},
		}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 0}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_delta", Delta: delta{StopReason: "tool_use"}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

		stream := newTestAnthropicStream(context.Background(), "claude-3-haiku", streamContent.String())
		var yieldedStatuses []llms.StreamStatus
		var argsAtData string
		var argsAtReady string

		iter := stream.Iter()
		iter(func(status llms.StreamStatus) bool {
			yieldedStatuses = append(yieldedStatuses, status)
			// Capture arguments state specifically after the data delta and at ready
			if status == llms.StreamStatusToolCallData {
				argsAtData = string(stream.ToolCall().Arguments)
			}
			if status == llms.StreamStatusToolCallReady {
				argsAtReady = string(stream.ToolCall().Arguments)
			}
			return true
		})

		require.NoError(t, stream.Err(), "Stream iteration should succeed")

		expectedStatuses := []llms.StreamStatus{
			llms.StreamStatusToolCallBegin,
			llms.StreamStatusToolCallData, // Expect data status
			llms.StreamStatusToolCallReady,
		}
		assert.Equal(t, expectedStatuses, yieldedStatuses, "Expected stream statuses sequence")

		// Check intermediate state captured during iteration
		assert.JSONEq(t, toolArgsJSON, argsAtData, "Arguments after delta should match")
		assert.JSONEq(t, toolArgsJSON, argsAtReady, "Arguments at ready should match")

		// Check final state
		require.NotNil(t, stream.Message(), "Final message should not be nil")
		require.Len(t, stream.Message().ToolCalls, 1, "Should have one tool call")
		finalToolCall := stream.Message().ToolCalls[0]
		assert.Equal(t, "toolu_02", finalToolCall.ID, "Tool call ID mismatch")
		assert.Equal(t, "searchTool", finalToolCall.Name, "Tool call name mismatch")
		assert.JSONEq(t, toolArgsJSON, string(finalToolCall.Arguments), "Final tool call arguments should match")
	})

	t.Run("Complex Argument Tool Call (Multi Delta)", func(t *testing.T) {
		var streamContent strings.Builder
		delta1 := `{"query": "search terms", "filter":`
		delta2 := `{"type": "recent", "count":`
		delta3 := ` 10}}`
		finalArgsJSON := delta1 + delta2 + delta3 // `{"query": "search terms", "filter":{"type": "recent", "count": 10}}`

		streamContent.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
		streamContent.WriteString(sseEvent(streamEvent{
			Type:  "content_block_start",
			Index: 0,
			ContentBlock: &contentBlock{
				Type:  "tool_use",
				ID:    "toolu_03",
				Name:  "complexSearch",
				Input: json.RawMessage(`{}`), // Starts empty
			},
		}))
		// Simulate arguments arriving over multiple deltas
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 0, Delta: delta{Type: "input_json_delta", PartialJSON: delta1}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 0, Delta: delta{Type: "input_json_delta", PartialJSON: delta2}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 0, Delta: delta{Type: "input_json_delta", PartialJSON: delta3}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 0}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_delta", Delta: delta{StopReason: "tool_use"}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

		stream := newTestAnthropicStream(context.Background(), "claude-3-haiku", streamContent.String())
		var yieldedStatuses []llms.StreamStatus
		var argsAfterDelta1 string
		var argsAfterDelta2 string
		var argsAfterDelta3 string
		var argsAtReady string

		callCount := 0
		iter := stream.Iter()
		iter(func(status llms.StreamStatus) bool {
			yieldedStatuses = append(yieldedStatuses, status)
			if status == llms.StreamStatusToolCallData {
				callCount++
				currentArgs := string(stream.ToolCall().Arguments)
				switch callCount {
				case 1:
					argsAfterDelta1 = currentArgs
				case 2:
					argsAfterDelta2 = currentArgs
				case 3:
					argsAfterDelta3 = currentArgs
				}
			}
			if status == llms.StreamStatusToolCallReady {
				argsAtReady = string(stream.ToolCall().Arguments)
			}
			return true
		})

		require.NoError(t, stream.Err(), "Stream iteration should succeed")

		expectedStatuses := []llms.StreamStatus{
			llms.StreamStatusToolCallBegin,
			llms.StreamStatusToolCallData,
			llms.StreamStatusToolCallData,
			llms.StreamStatusToolCallData,
			llms.StreamStatusToolCallReady,
		}
		assert.Equal(t, expectedStatuses, yieldedStatuses, "Expected stream statuses sequence")

		// Check intermediate states
		assert.Equal(t, delta1, argsAfterDelta1, "Args after delta 1 mismatch (replacement)")
		assert.Equal(t, delta1+delta2, argsAfterDelta2, "Args after delta 2 mismatch (append)")

		// Final checks use assert.JSONEq as they should be complete JSON
		assert.JSONEq(t, finalArgsJSON, argsAfterDelta3, "Args after delta 3 mismatch (final append)")
		assert.JSONEq(t, finalArgsJSON, argsAtReady, "Args at ready mismatch")

		// Check final message state
		require.NotNil(t, stream.Message(), "Final message should not be nil")
		require.Len(t, stream.Message().ToolCalls, 1, "Should have one tool call")
		finalToolCall := stream.Message().ToolCalls[0]
		assert.Equal(t, "toolu_03", finalToolCall.ID)
		assert.Equal(t, "complexSearch", finalToolCall.Name)
		assert.JSONEq(t, finalArgsJSON, string(finalToolCall.Arguments), "Final tool call arguments mismatch")
	})

	t.Run("Text Followed by Zero Argument Tool Call", func(t *testing.T) {
		var streamContent strings.Builder

		streamContent.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
		// Text block
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_start", Index: 0, ContentBlock: &contentBlock{Type: "text"}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 0, Delta: delta{Type: "text_delta", Text: "Okay, I "}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 0, Delta: delta{Type: "text_delta", Text: "can do that."}})) // Split text
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 0}))
		// Tool block
		streamContent.WriteString(sseEvent(streamEvent{
			Type:  "content_block_start",
			Index: 1, // Next block index
			ContentBlock: &contentBlock{
				Type:  "tool_use",
				ID:    "toolu_04", // New ID
				Name:  "getDiagnostics",
				Input: json.RawMessage(`{}`),
			},
		}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 1}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_delta", Delta: delta{StopReason: "tool_use"}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

		stream := newTestAnthropicStream(context.Background(), "claude-3-haiku", streamContent.String())
		var yieldedStatuses []llms.StreamStatus
		var capturedTextParts []string
		var finalToolCall llms.ToolCall

		iter := stream.Iter()
		iter(func(status llms.StreamStatus) bool {
			yieldedStatuses = append(yieldedStatuses, status)
			if status == llms.StreamStatusText {
				capturedTextParts = append(capturedTextParts, stream.Text()) // Capture each text part
			}
			if status == llms.StreamStatusToolCallReady {
				finalToolCall = stream.ToolCall() // Capture final tool call state
			}
			return true
		})

		require.NoError(t, stream.Err(), "Stream iteration should succeed")

		expectedStatuses := []llms.StreamStatus{
			llms.StreamStatusText, // "Okay, I "
			llms.StreamStatusText, // "can do that."
			llms.StreamStatusToolCallBegin,
			llms.StreamStatusToolCallReady,
		}
		assert.Equal(t, expectedStatuses, yieldedStatuses, "Expected stream statuses sequence")

		// Check final message content
		require.NotNil(t, stream.Message(), "Final message should not be nil")
		msg := stream.Message()
		assert.Equal(t, "assistant", msg.Role, "Message role should be assistant")

		// Check accumulated text content part
		// Note: Content.Append accumulates text
		require.Len(t, msg.Content, 1, "Should have one content item (text)")
		expectedTextContent := content.FromText("Okay, I can do that.")
		assert.Equal(t, expectedTextContent, msg.Content, "Accumulated text content mismatch")

		// Check captured text parts during iteration
		assert.Equal(t, []string{"Okay, I ", "can do that."}, capturedTextParts, "Captured text parts mismatch")

		// Check tool call part
		require.Len(t, msg.ToolCalls, 1, "Should have one tool call")
		finalToolCallInMsg := msg.ToolCalls[0]
		assert.Equal(t, "toolu_04", finalToolCallInMsg.ID, "Final message tool call ID mismatch")
		assert.Equal(t, "getDiagnostics", finalToolCallInMsg.Name, "Final message tool call name mismatch")
		assert.JSONEq(t, `{}`, string(finalToolCallInMsg.Arguments), "Final message tool call arguments should be '{}'")

		// Check captured tool call state
		assert.Equal(t, "toolu_04", finalToolCall.ID, "Captured tool call ID mismatch")
		assert.Equal(t, "getDiagnostics", finalToolCall.Name, "Captured tool call name mismatch")
		assert.JSONEq(t, `{}`, string(finalToolCall.Arguments), "Captured tool call arguments should be '{}'")
	})

	t.Run("Error Event Handling", func(t *testing.T) {
		var streamContent strings.Builder
		errMsg := "Something went wrong"
		errType := "internal_error"

		streamContent.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
		// Simulate an error event
		streamContent.WriteString(sseEvent(streamEvent{
			Type: "error",
			Error: &errorInfo{
				Type:    errType,
				Message: errMsg,
			},
		}))
		// Anthropic might send message_stop even after error, depends on error type
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

		stream := newTestAnthropicStream(context.Background(), "claude-3-haiku", streamContent.String())
		var yieldedStatuses []llms.StreamStatus

		iter := stream.Iter()
		iter(func(status llms.StreamStatus) bool {
			yieldedStatuses = append(yieldedStatuses, status)
			return true // Keep iterating even if error is processed internally
		})

		require.Error(t, stream.Err(), "Stream iteration should have resulted in an error")
		assert.Contains(t, stream.Err().Error(), errMsg, "Error message should contain the API error message")
		assert.Contains(t, stream.Err().Error(), errType, "Error message should contain the API error type")
		// Note: Depending on exactly when the error is detected vs yielded, statuses might be non-empty.
		// Let's check it *doesn't* contain statuses *after* the point the error should occur.
		// An error event should immediately stop processing and prevent further yields.
		assert.Empty(t, yieldedStatuses, "No statuses should be yielded in this stream containing only start and error")
	})

	t.Run("Stream Parsing Error - Immediate", func(t *testing.T) {
		// *** FIX: Test invalid JSON immediately after message_start ***
		streamContent := strings.Builder{}
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
		streamContent.WriteString("data: invalid json\n\n") // Invalid data right away

		stream := newTestAnthropicStream(context.Background(), "claude-3-haiku", streamContent.String())
		var yieldedStatuses []llms.StreamStatus

		iter := stream.Iter()
		iter(func(status llms.StreamStatus) bool {
			yieldedStatuses = append(yieldedStatuses, status)
			return true
		})

		require.Error(t, stream.Err(), "Stream iteration should fail on invalid JSON")
		assert.Contains(t, stream.Err().Error(), "error unmarshalling event", "Error message should indicate unmarshalling failure")
		// Check that *no* statuses were yielded because the error happened on the first data line
		assert.Empty(t, yieldedStatuses, "No statuses should be yielded when the first data event is invalid")
	})

	t.Run("Stream Parsing Error - After Valid Event", func(t *testing.T) {
		// Test invalid JSON *after* a valid, yield-producing event
		streamContent := strings.Builder{}
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_start", Index: 0, ContentBlock: &contentBlock{Type: "text"}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 0, Delta: delta{Type: "text_delta", Text: "Hello"}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 0}))
		streamContent.WriteString("data: invalid json\n\n") // Add invalid data after valid events

		stream := newTestAnthropicStream(context.Background(), "claude-3-haiku", streamContent.String())
		var yieldedStatuses []llms.StreamStatus
		iter := stream.Iter()
		iter(func(status llms.StreamStatus) bool {
			yieldedStatuses = append(yieldedStatuses, status)
			return true
		})

		require.Error(t, stream.Err(), "Stream iteration should fail on invalid JSON")
		assert.Contains(t, stream.Err().Error(), "error unmarshalling event", "Error message should indicate unmarshalling failure")
		// Should have yielded the text status *before* hitting the error
		assert.Equal(t, []llms.StreamStatus{llms.StreamStatusText}, yieldedStatuses, "Should yield status for valid events before error")
	})
}

func TestContentFromLLMEdgeCases(t *testing.T) {
	t.Run("Empty Text Content", func(t *testing.T) {
		llmContent := content.FromText("")
		apiContent := contentFromLLM(llmContent)
		require.Len(t, apiContent, 0, "Empty text should result in an empty content list from contentFromLLM")
	})

	t.Run("Whitespace Only Text Content", func(t *testing.T) {
		llmContent := content.FromText("   \n\t ")
		apiContent := contentFromLLM(llmContent)
		require.Len(t, apiContent, 0, "Whitespace-only text should result in an empty content list from contentFromLLM")
	})

	t.Run("JSON Content", func(t *testing.T) {
		jsonValue := `{"key": "value", "num": 1}`
		llmContent := content.FromRawJSON(json.RawMessage(jsonValue))
		apiContent := contentFromLLM(llmContent)
		require.Len(t, apiContent, 1)
		assert.Equal(t, "text", apiContent[0].Type)
		assert.Equal(t, jsonValue, apiContent[0].Text, "JSON content should be converted to text")
	})

	// Add more edge cases for contentFromLLM if needed (e.g., invalid image data URI)
}

func TestMessageFromLLMEdgeCases(t *testing.T) {
	t.Run("Assistant Message No Tool Calls", func(t *testing.T) {
		llmMsg := llms.Message{
			Role:    "assistant",
			Content: content.FromText("Just text."),
		}
		apiMsg := messageFromLLM(llmMsg)
		assert.Equal(t, "assistant", apiMsg.Role)
		require.Len(t, apiMsg.Content, 1)
		assert.Equal(t, "text", apiMsg.Content[0].Type)
		assert.Equal(t, "Just text.", apiMsg.Content[0].Text)
		// Ensure ToolUse specific fields are absent or nil for text blocks
		assert.Nil(t, apiMsg.Content[0].Input, "Input should be nil for text content")
		assert.Empty(t, apiMsg.Content[0].ID, "ID should be empty for text content")
		assert.Empty(t, apiMsg.Content[0].Name, "Name should be empty for text content")
	})

	t.Run("Tool Message Content", func(t *testing.T) {
		llmMsg := llms.Message{
			Role:       "tool",
			ToolCallID: "toolu_test_123",
			Content:    content.Textf("Tool result text"),
		}
		apiMsg := messageFromLLM(llmMsg)
		assert.Equal(t, "user", apiMsg.Role, "Tool result message should have role 'user'")
		require.Len(t, apiMsg.Content, 1, "Tool result message should have one content item")
		toolResultItem := apiMsg.Content[0]
		assert.Equal(t, "tool_result", toolResultItem.Type, "Content type should be 'tool_result'")
		assert.Equal(t, "toolu_test_123", toolResultItem.ToolUseID, "ToolUseID should match")
		require.NotNil(t, toolResultItem.Content, "Inner content should not be nil")
		require.Len(t, toolResultItem.Content, 1, "Inner content should have one item")
		innerContentItem := toolResultItem.Content[0]
		assert.Equal(t, "text", innerContentItem.Type, "Inner content item type should be text")
		assert.Equal(t, "Tool result text", innerContentItem.Text, "Inner content text mismatch")
	})

	t.Run("Tool Message with JSON Content", func(t *testing.T) {
		jsonResult := `{"status": "ok", "value": 42}`
		llmMsg := llms.Message{
			Role:       "tool",
			ToolCallID: "toolu_json_456",
			Content:    content.FromRawJSON(json.RawMessage(jsonResult)),
		}
		apiMsg := messageFromLLM(llmMsg)
		assert.Equal(t, "user", apiMsg.Role)
		require.Len(t, apiMsg.Content, 1)
		toolResultItem := apiMsg.Content[0]
		assert.Equal(t, "tool_result", toolResultItem.Type)
		assert.Equal(t, "toolu_json_456", toolResultItem.ToolUseID)
		require.Len(t, toolResultItem.Content, 1)
		innerContentItem := toolResultItem.Content[0]
		// Check anthropic.go: JSON content becomes a text block
		assert.Equal(t, "text", innerContentItem.Type)
		assert.Equal(t, jsonResult, innerContentItem.Text)
	})

	t.Run("Assistant Message With Tool Calls", func(t *testing.T) {
		llmMsg := llms.Message{
			Role:    "assistant",
			Content: content.FromText("Thinking..."), // Optional preceding text
			ToolCalls: []llms.ToolCall{
				{ID: "t_1", Name: "toolA", Arguments: json.RawMessage(`{"a":1}`)},
				{ID: "t_2", Name: "toolB", Arguments: json.RawMessage(`{}`)},
			},
		}
		apiMsg := messageFromLLM(llmMsg)
		assert.Equal(t, "assistant", apiMsg.Role)
		// Check the logic in messageFromLLM: it appends tool_use blocks after content blocks
		require.Len(t, apiMsg.Content, 3, "Expected 1 text + 2 tool_use content items")

		// Check text part (index 0)
		assert.Equal(t, "text", apiMsg.Content[0].Type)
		assert.Equal(t, "Thinking...", apiMsg.Content[0].Text)

		// Check first tool_use part (index 1)
		assert.Equal(t, "tool_use", apiMsg.Content[1].Type)
		assert.Equal(t, "t_1", apiMsg.Content[1].ID)
		assert.Equal(t, "toolA", apiMsg.Content[1].Name)
		require.NotNil(t, apiMsg.Content[1].Input, "Input should not be nil for tool_use")
		assert.JSONEq(t, `{"a":1}`, string(apiMsg.Content[1].Input))

		// Check second tool_use part (index 2)
		assert.Equal(t, "tool_use", apiMsg.Content[2].Type)
		assert.Equal(t, "t_2", apiMsg.Content[2].ID)
		assert.Equal(t, "toolB", apiMsg.Content[2].Name)
		require.NotNil(t, apiMsg.Content[2].Input, "Input should not be nil for tool_use")
		assert.JSONEq(t, `{}`, string(apiMsg.Content[2].Input))
	})
}
