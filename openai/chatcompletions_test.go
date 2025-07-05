package openai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestThinkingSplitter(t *testing.T) {
	tests := []struct {
		name         string
		inputs       []string
		wantTexts    []string
		wantThoughts []string
		wantFlush    string
	}{
		{
			name:         "simple thinking block",
			inputs:       []string{"Before <think>thinking</think> After"},
			wantTexts:    []string{"Before  After"},
			wantThoughts: []string{"thinking"},
			wantFlush:    "",
		},
		{
			name:         "split marker across chunks",
			inputs:       []string{"Before <thi", "nk>thinking</", "think> After"},
			wantTexts:    []string{"Before ", " After"},
			wantThoughts: []string{"thinking"},
			wantFlush:    "",
		},
		{
			name:         "multiple thinking blocks",
			inputs:       []string{"A <think>1</think> B <think>2</think> C"},
			wantTexts:    []string{"A  B  C"},
			wantThoughts: []string{"12"},
			wantFlush:    "",
		},
		{
			name:         "incomplete thinking block flushed",
			inputs:       []string{"Text <think>incomplete"},
			wantTexts:    []string{"Text "},
			wantThoughts: []string{"incomplete"},
			wantFlush:    "",
		},
		{
			name:         "partial marker flushed",
			inputs:       []string{"Text <thi"},
			wantTexts:    []string{"Text "},
			wantThoughts: []string{},
			wantFlush:    "<thi",
		},
		{
			name:         "no thinking blocks",
			inputs:       []string{"Just regular text"},
			wantTexts:    []string{"Just regular text"},
			wantThoughts: []string{},
			wantFlush:    "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			splitter := &thinkingSplitter{
				startBlock: "<think>",
				endBlock:   "</think>",
			}

			var gotTexts []string
			var gotThoughts []string

			for _, input := range tt.inputs {
				thinking, text := splitter.Split(input)
				if thinking != "" {
					gotThoughts = append(gotThoughts, thinking)
				}
				if text != "" {
					gotTexts = append(gotTexts, text)
				}
			}

			// Get flush result
			gotFlush := splitter.Flush()

			// Check texts
			if len(gotTexts) != len(tt.wantTexts) {
				t.Errorf("got %d texts, want %d", len(gotTexts), len(tt.wantTexts))
			}
			for i := range tt.wantTexts {
				if i < len(gotTexts) && gotTexts[i] != tt.wantTexts[i] {
					t.Errorf("text[%d] = %q, want %q", i, gotTexts[i], tt.wantTexts[i])
				}
			}

			// Check thoughts
			if len(gotThoughts) != len(tt.wantThoughts) {
				t.Errorf("got %d thoughts, want %d", len(gotThoughts), len(tt.wantThoughts))
			}
			for i := range tt.wantThoughts {
				if i < len(gotThoughts) && gotThoughts[i] != tt.wantThoughts[i] {
					t.Errorf("thought[%d] = %q, want %q", i, gotThoughts[i], tt.wantThoughts[i])
				}
			}

			// Check flush
			if gotFlush != tt.wantFlush {
				t.Errorf("flush = %q, want %q", gotFlush, tt.wantFlush)
			}
		})
	}
}

func Test_convertContent(t *testing.T) {
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

func TestCustomThinkingBlocks(t *testing.T) {
	// Create a mock response with custom thinking blocks
	mockResponse := `data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"Let me think about this "},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"<think>"},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"I need to analyze the problem"},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":" step by step"},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"</think>"},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":" The answer is 42."},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"finish_reason":"stop"}]}
data: [DONE]
`

	stream := &ChatCompletionsStream{
		ctx:    context.Background(),
		stream: strings.NewReader(mockResponse),
		splitter: &thinkingSplitter{
			startBlock: "<think>",
			endBlock:   "</think>",
		},
	}

	var updates []llms.StreamStatus
	var texts []string
	var thoughts []string

	for status := range stream.Iter() {
		updates = append(updates, status)
		switch status {
		case llms.StreamStatusText:
			texts = append(texts, stream.Text())
		case llms.StreamStatusThinking:
			thoughts = append(thoughts, stream.Thought().Text)
		}
	}

	// Verify we got the expected updates
	expectedTexts := []string{"Let me think about this ", " The answer is 42."}
	expectedThoughts := []string{"I need to analyze the problem", " step by step"}

	if len(texts) != len(expectedTexts) {
		t.Errorf("Expected %d text updates, got %d", len(expectedTexts), len(texts))
	}

	for i, expected := range expectedTexts {
		if i < len(texts) && texts[i] != expected {
			t.Errorf("Text %d: expected %q, got %q", i, expected, texts[i])
		}
	}

	// The thinking content comes in two parts due to how it's chunked
	if len(thoughts) != len(expectedThoughts) {
		t.Errorf("Expected %d thinking updates, got %d", len(expectedThoughts), len(thoughts))
	}

	for i, expected := range expectedThoughts {
		if i < len(thoughts) && thoughts[i] != expected {
			t.Errorf("Thought %d: expected %q, got %q", i, expected, thoughts[i])
		}
	}

	// Verify the final message content includes both text and thoughts
	msg := stream.Message()
	if msg.Content == nil {
		t.Fatal("Expected message content to be non-nil")
	}

	// Check that we have both text and thought items in the content
	var hasText, hasThought bool
	for _, item := range msg.Content {
		switch item.Type() {
		case content.TypeText:
			hasText = true
		case content.TypeThought:
			hasThought = true
		}
	}

	if !hasText {
		t.Error("Expected message content to contain text items")
	}
	if !hasThought {
		t.Error("Expected message content to contain thought items")
	}
}

func TestCustomThinkingBlocksSplitMarkers(t *testing.T) {
	// Test case where thinking block markers are split across chunks
	// With the new thinkingSplitter, this should be handled correctly
	mockResponse := `data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"Before <thi"},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"nk>Inside thinking</"},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"think> After"},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"finish_reason":"stop"}]}
data: [DONE]
`

	stream := &ChatCompletionsStream{
		ctx:    context.Background(),
		stream: strings.NewReader(mockResponse),
		splitter: &thinkingSplitter{
			startBlock: "<think>",
			endBlock:   "</think>",
		},
	}

	var texts []string
	var thoughts []string

	for status := range stream.Iter() {
		switch status {
		case llms.StreamStatusText:
			texts = append(texts, stream.Text())
		case llms.StreamStatusThinking:
			thoughts = append(thoughts, stream.Thought().Text)
		}
	}

	// Now the splitter correctly handles split markers
	expectedTexts := []string{"Before ", " After"}
	expectedThoughts := []string{"Inside thinking"}

	if len(texts) != len(expectedTexts) {
		t.Errorf("Expected %d text updates, got %d", len(expectedTexts), len(texts))
	}

	for i, expected := range expectedTexts {
		if i < len(texts) && texts[i] != expected {
			t.Errorf("Text %d: expected %q, got %q", i, expected, texts[i])
		}
	}

	if len(thoughts) != len(expectedThoughts) {
		t.Errorf("Expected %d thinking updates, got %d", len(expectedThoughts), len(thoughts))
	}

	for i, expected := range expectedThoughts {
		if i < len(thoughts) && thoughts[i] != expected {
			t.Errorf("Thought %d: expected %q, got %q", i, expected, thoughts[i])
		}
	}
}

func TestCustomThinkingBlocksMultiple(t *testing.T) {
	// Test multiple thinking blocks in the same response
	mockResponse := `data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"delta":{"content":"First <think>thought 1</think> then <think>thought 2</think> end"},"finish_reason":null}]}
data: {"id":"1","object":"chat.completion.chunk","created":1234567890,"model":"test","choices":[{"index":0,"finish_reason":"stop"}]}
data: [DONE]
`

	stream := &ChatCompletionsStream{
		ctx:    context.Background(),
		stream: strings.NewReader(mockResponse),
		splitter: &thinkingSplitter{
			startBlock: "<think>",
			endBlock:   "</think>",
		},
	}

	var texts []string
	var thoughts []string

	for status := range stream.Iter() {
		switch status {
		case llms.StreamStatusText:
			texts = append(texts, stream.Text())
		case llms.StreamStatusThinking:
			thoughts = append(thoughts, stream.Thought().Text)
		}
	}

	// Verify we got all text parts
	// Since all content comes in one chunk, the splitter returns joined strings
	expectedTexts := []string{"First  then  end"}
	if len(texts) != len(expectedTexts) {
		t.Errorf("Expected %d text updates, got %d", len(expectedTexts), len(texts))
	}

	for i, expected := range expectedTexts {
		if i < len(texts) && texts[i] != expected {
			t.Errorf("Text %d: expected %q, got %q", i, expected, texts[i])
		}
	}

	// Both thoughts are joined when processed in the same chunk
	expectedThoughts := []string{"thought 1thought 2"}
	if len(thoughts) != len(expectedThoughts) {
		t.Errorf("Expected %d thinking updates, got %d", len(expectedThoughts), len(thoughts))
	}

	for i, expected := range expectedThoughts {
		if i < len(thoughts) && thoughts[i] != expected {
			t.Errorf("Thought %d: expected %q, got %q", i, expected, thoughts[i])
		}
	}
}
