package openai

import (
	"encoding/json"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMessagesFromLLM_OpenAI(t *testing.T) {
	testCases := []struct {
		name     string
		input    llms.Message
		expected []message
	}{
		{
			name: "User message - text only",
			input: llms.Message{
				Role:    "user",
				Content: content.FromText("Hello there"),
			},
			expected: []message{
				{
					Role: "user",
					Content: contentList{
						{Type: "text", Text: ptr("Hello there")},
					},
				},
			},
		},
		{
			name: "User message - text and image",
			input: llms.Message{
				Role:    "user",
				Content: content.FromTextAndImage("Look at this:", "data:image/png;base64,abc"),
			},
			expected: []message{
				{
					Role: "user",
					Content: contentList{
						{Type: "text", Text: ptr("Look at this:")},
						{Type: "image_url", ImageURL: &imageURL{URL: "data:image/png;base64,abc", Detail: "auto"}},
					},
				},
			},
		},
		{
			name: "Assistant message - text only",
			input: llms.Message{
				Role:    "assistant",
				Content: content.FromText("I am here to help."),
			},
			expected: []message{
				{
					Role: "assistant",
					Content: contentList{
						{Type: "text", Text: ptr("I am here to help.")},
					},
				},
			},
		},
		{
			name: "Assistant message - with tool call",
			input: llms.Message{
				Role: "assistant",
				ToolCalls: []llms.ToolCall{
					{ID: "call_123", Name: "get_weather", Arguments: json.RawMessage(`{"location": "Paris"}`)},
				},
				Content: content.FromText("Okay, getting weather."), // Content often empty but test just in case
			},
			expected: []message{
				{
					Role: "assistant",
					Content: contentList{ // Ensure content is still converted
						{Type: "text", Text: ptr("Okay, getting weather.")},
					},
					ToolCalls: []toolCall{
						{ID: "call_123", Type: "function", Function: toolCallFunction{Name: "get_weather", Arguments: `{"location": "Paris"}`}},
					},
				},
			},
		},
		{
			name: "Assistant message - with tool call and empty content",
			input: llms.Message{
				Role: "assistant",
				ToolCalls: []llms.ToolCall{
					{ID: "call_456", Name: "web_search", Arguments: json.RawMessage(`{"query": "OpenAI"}`)},
				},
				Content: content.Content{}, // Empty content
			},
			expected: []message{
				{
					Role:    "assistant",
					Content: nil, // Content should be nil so it's omitted from JSON
					ToolCalls: []toolCall{
						{ID: "call_456", Type: "function", Function: toolCallFunction{Name: "web_search", Arguments: `{"query": "OpenAI"}`}},
					},
				},
			},
		},
		{
			name: "Tool result - JSON only",
			input: llms.Message{
				Role:       "tool",
				ToolCallID: "call_123",
				Content:    content.FromRawJSON(json.RawMessage(`{"temp": 25}`)),
			},
			expected: []message{
				{
					Role:       "tool",
					ToolCallID: "call_123",
					Content:    contentList{{Type: "text", Text: ptr(`{"temp": 25}`)}}, // Result stringified
				},
			},
		},
		{
			name: "Tool result - Text only (converted to string)",
			input: llms.Message{
				Role:       "tool",
				ToolCallID: "call_456",
				Content:    content.FromText("Command executed successfully"),
			},
			expected: []message{
				{
					Role:       "tool",
					ToolCallID: "call_456",
					Content:    contentList{{Type: "text", Text: ptr("Command executed successfully")}},
				},
			},
		},
		{
			name: "Tool result - JSON primary + Text secondary",
			input: llms.Message{
				Role:       "tool",
				ToolCallID: "call_789",
				Content: content.Content{ // Multi-part content
					&content.JSON{Data: json.RawMessage(`{"status": "done"}`)},
					&content.Text{Text: "Process finished."},
				},
			},
			expected: []message{
				{ // Primary tool result message
					Role:       "tool",
					ToolCallID: "call_789",
					Content:    contentList{{Type: "text", Text: ptr(`{"status": "done"}`)}},
				},
				{ // Secondary user message for the extra text
					Role: "user",
					Content: contentList{
						{Type: "text", Text: ptr("Process finished.")},
					},
				},
			},
		},
		{
			name: "Tool result - JSON primary + Image secondary",
			input: llms.Message{
				Role:       "tool",
				ToolCallID: "call_abc",
				Content: content.Content{ // Multi-part content
					&content.JSON{Data: json.RawMessage(`{"plot_generated": true}`)},
					&content.ImageURL{URL: "data:image/png;base64,xyz"},
				},
			},
			expected: []message{
				{ // Primary tool result message
					Role:       "tool",
					ToolCallID: "call_abc",
					Content:    contentList{{Type: "text", Text: ptr(`{"plot_generated": true}`)}},
				},
				{ // Secondary user message for the extra image
					Role: "user",
					Content: contentList{
						{Type: "image_url", ImageURL: &imageURL{URL: "data:image/png;base64,xyz", Detail: "auto"}},
					},
				},
			},
		},
		{
			name:     "Empty message",
			input:    llms.Message{Role: "user", Content: nil},
			expected: []message{}, // Should produce no message if content and tool calls are empty
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := messagesFromLLM(tc.input)
			assert.Equal(t, len(tc.expected), len(actual), "Number of messages mismatch")

			// Use require for slice length check before iterating
			require.Equal(t, len(tc.expected), len(actual), "Number of messages mismatch")

			for i := range tc.expected {
				// Compare Role and ToolCallID directly
				assert.Equal(t, tc.expected[i].Role, actual[i].Role, "Role mismatch in message %d", i)
				assert.Equal(t, tc.expected[i].ToolCallID, actual[i].ToolCallID, "ToolCallID mismatch in message %d", i)

				// Compare Content (contentList)
				assert.Equal(t, len(tc.expected[i].Content), len(actual[i].Content), "Content length mismatch in message %d", i)
				require.Equal(t, len(tc.expected[i].Content), len(actual[i].Content), "Content length mismatch in message %d", i)
				for j := range tc.expected[i].Content {
					assert.Equal(t, tc.expected[i].Content[j], actual[i].Content[j], "Content mismatch at index %d in message %d", j, i)
				}

				// Compare ToolCalls ([]toolCall)
				assert.Equal(t, len(tc.expected[i].ToolCalls), len(actual[i].ToolCalls), "ToolCalls length mismatch in message %d", i)
				require.Equal(t, len(tc.expected[i].ToolCalls), len(actual[i].ToolCalls), "ToolCalls length mismatch in message %d", i)
				for j := range tc.expected[i].ToolCalls {
					// Compare individual toolCall fields
					assert.Equal(t, tc.expected[i].ToolCalls[j].ID, actual[i].ToolCalls[j].ID, "ToolCall ID mismatch at index %d in message %d", j, i)
					assert.Equal(t, tc.expected[i].ToolCalls[j].Type, actual[i].ToolCalls[j].Type, "ToolCall Type mismatch at index %d in message %d", j, i)
					assert.Equal(t, tc.expected[i].ToolCalls[j].Function.Name, actual[i].ToolCalls[j].Function.Name, "ToolCall Function Name mismatch at index %d in message %d", j, i)
					// Use JSONEq for arguments as order might not matter
					assert.JSONEq(t, tc.expected[i].ToolCalls[j].Function.Arguments, actual[i].ToolCalls[j].Function.Arguments, "ToolCall Function Arguments mismatch at index %d in message %d", j, i)
				}
			}
		})
	}
}

// Helper function to get a pointer to a string
func ptr(s string) *string {
	return &s
}
