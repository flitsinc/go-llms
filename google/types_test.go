package google

import (
	"encoding/json"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestMessagesFromLLM_Google(t *testing.T) {
	testCases := []struct {
		name     string
		input    llms.Message
		expected []message // Google conversion can also result in multiple messages
	}{
		{
			name: "User message - text only",
			input: llms.Message{
				Role:    "user",
				Content: content.FromText("Hello Gemini"),
			},
			expected: []message{
				{
					Role:  "user",
					Parts: parts{{Text: ptr("Hello Gemini")}},
				},
			},
		},
		{
			name: "User message - text and image (data URI)",
			input: llms.Message{
				Role:    "user",
				Content: content.FromTextAndImage("Describe this:", "data:image/jpeg;base64,def"),
			},
			expected: []message{
				{
					Role: "user",
					Parts: parts{
						{Text: ptr("Describe this:")},
						{InlineData: &inlineData{MimeType: "image/jpeg", Data: "def"}},
					},
				},
			},
		},
		{
			name: "Assistant message - text only",
			input: llms.Message{
				Role:    "assistant",
				Content: content.FromText("Thinking..."),
			},
			expected: []message{
				{
					Role:  "model", // Assistant maps to model
					Parts: parts{{Text: ptr("Thinking...")}},
				},
			},
		},
		{
			name: "Assistant message - with tool call",
			input: llms.Message{
				Role: "assistant",
				ToolCalls: []llms.ToolCall{
					{ID: "call_g1", Name: "search_web", Arguments: json.RawMessage(`{"query": "Go LLMs"}`)},
				},
				Content: nil, // Content often nil with tool calls
			},
			expected: []message{
				{
					Role: "model",
					Parts: parts{
						{
							FunctionCall: &functionCall{Name: "search_web", Args: json.RawMessage(`{"query": "Go LLMs"}`)}},
					},
				},
			},
		},
		{
			name: "Tool result - JSON only",
			input: llms.Message{
				Role:       "tool",
				ToolCallID: "call_g1",
				Content:    content.FromRawJSON(json.RawMessage(`{"result": "found"}`)),
			},
			expected: []message{
				{
					Role: "function", // Tool maps to function role
					Parts: parts{
						{
							FunctionResponse: &functionResponse{
								Name:     "call_g1",
								Response: mustMarshal(map[string]any{"name": "call_g1", "content": json.RawMessage(`{"result": "found"}`)}),
							},
						},
					},
				},
			},
		},
		{
			name: "Tool result - Non-JSON primary (Error generated)",
			input: llms.Message{
				Role:       "tool",
				ToolCallID: "call_g2",
				Content:    content.FromText("Just text, not JSON"),
			},
			expected: []message{
				{
					Role: "function",
					Parts: parts{
						{
							FunctionResponse: &functionResponse{
								Name:     "call_g2",
								Response: mustMarshal(map[string]any{"name": "call_g2", "content": json.RawMessage(`{"error":"Primary tool result must be JSON for Google Gemini"}`)}),
							},
						},
					},
				},
				{
					Role:  "user", // Original text becomes secondary user message
					Parts: parts{{Text: ptr("Just text, not JSON")}},
				},
			},
		},
		{
			name: "Tool result - JSON primary + Text secondary",
			input: llms.Message{
				Role:       "tool",
				ToolCallID: "call_g3",
				Content: content.Content{
					&content.JSON{Data: json.RawMessage(`{"status": "complete"}`)},
					&content.Text{Text: "Secondary info."},
				},
			},
			expected: []message{
				{
					Role: "function",
					Parts: parts{
						{
							FunctionResponse: &functionResponse{
								Name:     "call_g3",
								Response: mustMarshal(map[string]any{"name": "call_g3", "content": json.RawMessage(`{"status": "complete"}`)}),
							},
						},
					},
				},
				{
					Role:  "user", // Secondary content becomes user message
					Parts: parts{{Text: ptr("Secondary info.")}},
				},
			},
		},
		{
			name:     "Empty message",
			input:    llms.Message{Role: "user", Content: nil},
			expected: []message{}, // Should produce no message
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			actual := messagesFromLLM(tc.input)

			require.Equal(t, len(tc.expected), len(actual), "Number of messages mismatch")

			for i := range tc.expected {
				assert.Equal(t, tc.expected[i].Role, actual[i].Role, "Role mismatch in message %d", i)
				require.Equal(t, len(tc.expected[i].Parts), len(actual[i].Parts), "Parts length mismatch in message %d", i)

				for j := range tc.expected[i].Parts {
					expectedPart := tc.expected[i].Parts[j]
					actualPart := actual[i].Parts[j]

					// Compare individual fields within the part
					assert.Equal(t, expectedPart.Text, actualPart.Text, "Part Text mismatch at index %d in message %d", j, i)
					assert.Equal(t, expectedPart.InlineData, actualPart.InlineData, "Part InlineData mismatch at index %d in message %d", j, i)
					assert.Equal(t, expectedPart.FileData, actualPart.FileData, "Part FileData mismatch at index %d in message %d", j, i)

					// Compare FunctionCall
					if expectedPart.FunctionCall != nil {
						require.NotNil(t, actualPart.FunctionCall, "Expected FunctionCall, got nil at index %d in message %d", j, i)
						assert.Equal(t, expectedPart.FunctionCall.Name, actualPart.FunctionCall.Name, "FunctionCall Name mismatch at index %d in message %d", j, i)
						assert.JSONEq(t, string(expectedPart.FunctionCall.Args), string(actualPart.FunctionCall.Args), "FunctionCall Args mismatch at index %d in message %d", j, i)
					} else {
						assert.Nil(t, actualPart.FunctionCall, "Expected nil FunctionCall, got non-nil at index %d in message %d", j, i)
					}

					// Compare FunctionResponse
					if expectedPart.FunctionResponse != nil {
						require.NotNil(t, actualPart.FunctionResponse, "Expected FunctionResponse, got nil at index %d in message %d", j, i)
						assert.Equal(t, expectedPart.FunctionResponse.Name, actualPart.FunctionResponse.Name, "FunctionResponse Name mismatch at index %d in message %d", j, i)
						assert.JSONEq(t, string(expectedPart.FunctionResponse.Response), string(actualPart.FunctionResponse.Response), "FunctionResponse Response mismatch at index %d in message %d", j, i)
					} else {
						assert.Nil(t, actualPart.FunctionResponse, "Expected nil FunctionResponse, got non-nil at index %d in message %d", j, i)
					}

					assert.Equal(t, expectedPart.VideoMetadata, actualPart.VideoMetadata, "Part VideoMetadata mismatch at index %d in message %d", j, i)
				}
			}
		})
	}
}

// Helper function to get a pointer to a string
func ptr(s string) *string {
	return &s
}

// Helper function to marshal JSON, panicking on error (for test setup)
func mustMarshal(v any) json.RawMessage {
	data, err := json.Marshal(v)
	if err != nil {
		panic(err)
	}
	return json.RawMessage(data)
}
