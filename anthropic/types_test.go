package anthropic

import (
	"encoding/json"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMessageFromLLM_Anthropic(t *testing.T) {
	testCases := []struct {
		name     string
		input    llms.Message
		expected message // Anthropic conversion always results in a single message
	}{
		{
			name: "User message - text only",
			input: llms.Message{
				Role:    "user",
				Content: content.FromText("Hi Claude"),
			},
			expected: message{
				Role:    "user",
				Content: contentList{{Type: "text", Text: "Hi Claude"}},
			},
		},
		{
			name: "User message - text and image (data URI)",
			input: llms.Message{
				Role:    "user",
				Content: content.FromTextAndImage("What is this?", "data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=="),
			},
			expected: message{
				Role: "user",
				Content: contentList{
					{Type: "text", Text: "What is this?"},
					{Type: "image", Source: &source{Type: "base64", MediaType: "image/gif", Data: "R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw=="}},
				},
			},
		},
		{
			name: "Assistant message - text only",
			input: llms.Message{
				Role:    "assistant",
				Content: content.FromText("Certainly!"),
			},
			expected: message{
				Role:    "assistant",
				Content: contentList{{Type: "text", Text: "Certainly!"}},
			},
		},
		{
			name: "Assistant message - with tool call",
			input: llms.Message{
				Role: "assistant",
				ToolCalls: []llms.ToolCall{
					{ID: "toolu_1", Name: "get_stock", Arguments: json.RawMessage(`{"ticker": "GOOG"}`)},
				},
				Content: content.FromText("Checking stock price..."), // Include text part too
			},
			expected: message{
				Role: "assistant",
				Content: contentList{
					{Type: "text", Text: "Checking stock price..."},                                                    // Text comes first
					{Type: "tool_use", ID: "toolu_1", Name: "get_stock", Input: json.RawMessage(`{"ticker": "GOOG"}`)}, // Then tool use
				},
			},
		},
		{
			name: "Tool result - JSON only",
			input: llms.Message{
				Role:       "tool",
				ToolCallID: "toolu_1",
				Content:    content.FromRawJSON(json.RawMessage(`{"price": 180.5}`)),
			},
			expected: message{
				Role: "user", // Tool results become user messages
				Content: contentList{
					{
						Type:      "tool_result",
						ToolUseID: "toolu_1",
						Content:   contentList{{Type: "text", Text: `{"price": 180.5}`}}, // JSON stringified
					},
				},
			},
		},
		{
			name: "Tool result - Text only",
			input: llms.Message{
				Role:       "tool",
				ToolCallID: "toolu_2",
				Content:    content.FromText("Stock not found"),
			},
			expected: message{
				Role: "user",
				Content: contentList{
					{
						Type:      "tool_result",
						ToolUseID: "toolu_2",
						Content:   contentList{{Type: "text", Text: "Stock not found"}},
					},
				},
			},
		},
		{
			name: "Tool result - JSON + Text + Image",
			input: llms.Message{
				Role:       "tool",
				ToolCallID: "toolu_3",
				Content: content.Content{
					&content.JSON{Data: json.RawMessage(`{"chart_generated": true}`)},
					&content.Text{Text: "Here is the chart:"},
					&content.ImageURL{URL: "data:image/png;base64,ghi"},
				},
			},
			expected: message{
				Role: "user",
				Content: contentList{
					{
						Type:      "tool_result",
						ToolUseID: "toolu_3",
						Content: contentList{ // All parts converted inside tool_result content
							{Type: "text", Text: `{"chart_generated": true}`},
							{Type: "text", Text: "Here is the chart:"},
							{Type: "image", Source: &source{Type: "base64", MediaType: "image/png", Data: "ghi"}},
						},
					},
				},
			},
		},
		{
			name:  "Empty message",
			input: llms.Message{Role: "assistant", Content: nil, ToolCalls: nil},
			// The new logic in messageFromLLM ensures empty content gets [{type: "text", text: ""}].
			expected: message{Role: "assistant", Content: contentList{{Type: "text", Text: ""}}},
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := messageFromLLM(tc.input)
			assert.Equal(t, tc.expected.Role, actual.Role, "Role mismatch")
			require.Equal(t, len(tc.expected.Content), len(actual.Content), "Content length mismatch")

			// Detailed comparison of content items
			for i := range tc.expected.Content {
				expectedItem := tc.expected.Content[i]
				actualItem := actual.Content[i]
				assert.Equal(t, expectedItem.Type, actualItem.Type, "Content item type mismatch at index %d", i)
				assert.Equal(t, expectedItem.Text, actualItem.Text, "Content item text mismatch at index %d", i)
				assert.Equal(t, expectedItem.Source, actualItem.Source, "Content item source mismatch at index %d", i)
				assert.Equal(t, expectedItem.ID, actualItem.ID, "Content item ID mismatch at index %d", i)
				assert.Equal(t, expectedItem.Name, actualItem.Name, "Content item Name mismatch at index %d", i)
				// Conditional check for Input field
				if expectedItem.Input != nil {
					assert.JSONEq(t, string(expectedItem.Input), string(actualItem.Input), "Content item Input mismatch at index %d", i)
				} else {
					// Expect actual input to be nil or empty if expected is nil
					assert.True(t, len(actualItem.Input) == 0, "Expected nil or empty Input, got %s at index %d", actualItem.Input, i)
				}
				assert.Equal(t, expectedItem.ToolUseID, actualItem.ToolUseID, "Content item ToolUseID mismatch at index %d", i)
				// Compare nested content for tool_result
				if expectedItem.Type == "tool_result" {
					require.Equal(t, len(expectedItem.Content), len(actualItem.Content), "Nested content length mismatch in tool_result at index %d", i)
					for j := range expectedItem.Content {
						assert.Equal(t, expectedItem.Content[j], actualItem.Content[j], "Nested content mismatch at index %d in tool_result %d", j, i)
					}
				}
			}
		})
	}
}
