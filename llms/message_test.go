package llms

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/blixt/go-llms/content"
)

func TestMessageMarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		message Message
		want    string
	}{
		{
			name: "user message with text",
			message: Message{
				Role:    "user",
				Content: content.FromText("Hello, world!"),
			},
			want: `{"role":"user","content":[{"text":"Hello, world!","type":"text"}]}`,
		},
		{
			name: "assistant message with text",
			message: Message{
				Role:    "assistant",
				Name:    "AI",
				Content: content.FromText("How can I help you?"),
			},
			want: `{"role":"assistant","name":"AI","content":[{"text":"How can I help you?","type":"text"}]}`,
		},
		{
			name: "tool message with JSON",
			message: Message{
				Role:       "tool",
				Content:    content.FromRawJSON(json.RawMessage(`{"result":"success"}`)),
				ToolCallID: "tool-call-123",
			},
			want: `{"role":"tool","content":[{"data":{"result":"success"},"type":"json"}],"tool_call_id":"tool-call-123"}`,
		},
		{
			name: "assistant message with tool calls",
			message: Message{
				Role:    "assistant",
				Content: content.FromText("I'll help you with that."),
				ToolCalls: []ToolCall{
					{
						ID:        "tool-call-123",
						Name:      "getWeather",
						Arguments: json.RawMessage(`{"location":"New York"}`),
					},
				},
			},
			want: `{"role":"assistant","content":[{"text":"I'll help you with that.","type":"text"}],"tool_calls":[{"id":"tool-call-123","name":"getWeather","arguments":{"location":"New York"}}]}`,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := json.Marshal(tt.message)
			if err != nil {
				t.Errorf("Marshal() error = %v", err)
				return
			}
			if string(got) != tt.want {
				t.Errorf("Marshal() got = %v, want %v", string(got), tt.want)
			}
		})
	}
}

func TestMessageUnmarshalJSON(t *testing.T) {
	tests := []struct {
		name    string
		json    string
		want    Message
		wantErr bool
	}{
		{
			name: "user message with text",
			json: `{"role":"user","content":[{"type":"text","text":"Hello, world!"}]}`,
			want: Message{
				Role:    "user",
				Content: content.FromText("Hello, world!"),
			},
		},
		{
			name: "assistant message with text",
			json: `{"role":"assistant","name":"AI","content":[{"type":"text","text":"How can I help you?"}]}`,
			want: Message{
				Role:    "assistant",
				Name:    "AI",
				Content: content.FromText("How can I help you?"),
			},
		},
		{
			name: "tool message with JSON",
			json: `{"role":"tool","content":[{"type":"json","data":{"result":"success"}}],"tool_call_id":"tool-call-123"}`,
			want: Message{
				Role:       "tool",
				Content:    content.FromRawJSON(json.RawMessage(`{"result":"success"}`)),
				ToolCallID: "tool-call-123",
			},
		},
		{
			name: "assistant message with tool calls",
			json: `{"role":"assistant","content":[{"type":"text","text":"I'll help you with that."}],"tool_calls":[{"id":"tool-call-123","name":"getWeather","arguments":{"location":"New York"}}]}`,
			want: Message{
				Role:    "assistant",
				Content: content.FromText("I'll help you with that."),
				ToolCalls: []ToolCall{
					{
						ID:        "tool-call-123",
						Name:      "getWeather",
						Arguments: json.RawMessage(`{"location":"New York"}`),
					},
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			var got Message
			err := json.Unmarshal([]byte(tt.json), &got)

			if (err != nil) != tt.wantErr {
				t.Errorf("Unmarshal() error = %v, wantErr %v", err, tt.wantErr)
				return
			}

			if tt.wantErr {
				return
			}

			// For JSON content, we need to compare the marshaled string
			if len(got.Content) == 1 && got.Content[0].Type() == content.TypeJSON {
				gotJSON, _ := json.Marshal(got.Content)
				wantJSON, _ := json.Marshal(tt.want.Content)
				if !reflect.DeepEqual(gotJSON, wantJSON) {
					t.Errorf("Unmarshal() content got = %v, want %v", string(gotJSON), string(wantJSON))
				}
				// Compare other fields
				got.Content = nil
				tt.want.Content = nil
				if !reflect.DeepEqual(got, tt.want) {
					t.Errorf("Unmarshal() other fields got = %v, want %v", got, tt.want)
				}
				return
			}

			// For tool calls, we need special comparison for Arguments field
			if len(got.ToolCalls) > 0 || len(tt.want.ToolCalls) > 0 {
				if len(got.ToolCalls) != len(tt.want.ToolCalls) {
					t.Errorf("Unmarshal() tool calls count mismatch got = %v, want %v", len(got.ToolCalls), len(tt.want.ToolCalls))
					return
				}

				for i := range got.ToolCalls {
					gotArgs, _ := json.Marshal(got.ToolCalls[i].Arguments)
					wantArgs, _ := json.Marshal(tt.want.ToolCalls[i].Arguments)
					if !reflect.DeepEqual(gotArgs, wantArgs) {
						t.Errorf("Unmarshal() tool call %d arguments got = %v, want %v", i, string(gotArgs), string(wantArgs))
					}

					// Compare other fields
					got.ToolCalls[i].Arguments = nil
					tt.want.ToolCalls[i].Arguments = nil
					if !reflect.DeepEqual(got.ToolCalls[i], tt.want.ToolCalls[i]) {
						t.Errorf("Unmarshal() tool call %d other fields got = %v, want %v", i, got.ToolCalls[i], tt.want.ToolCalls[i])
					}
				}

				// Compare other fields
				got.ToolCalls = nil
				tt.want.ToolCalls = nil
			}

			if !reflect.DeepEqual(got, tt.want) {
				t.Errorf("Unmarshal() got = %v, want %v", got, tt.want)
			}
		})
	}
}

func TestMessageSliceRoundTrip(t *testing.T) {
	tests := []struct {
		name     string
		messages []Message
	}{
		{
			name: "conversation",
			messages: []Message{
				{
					Role:    "user",
					Content: content.FromText("What's the weather in New York?"),
				},
				{
					Role:    "assistant",
					Content: content.FromText("I'll check that for you."),
					ToolCalls: []ToolCall{
						{
							ID:        "tool-call-123",
							Name:      "getWeather",
							Arguments: json.RawMessage(`{"location":"New York"}`),
						},
					},
				},
				{
					Role:       "tool",
					Content:    content.FromRawJSON(json.RawMessage(`{"temperature":72,"condition":"sunny"}`)),
					ToolCallID: "tool-call-123",
				},
				{
					Role:    "assistant",
					Content: content.FromText("The weather in New York is sunny with a temperature of 72°F."),
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			data, err := json.Marshal(tt.messages)
			if err != nil {
				t.Errorf("Marshal() error = %v", err)
				return
			}

			var got []Message
			err = json.Unmarshal(data, &got)
			if err != nil {
				t.Errorf("Unmarshal() error = %v", err)
				return
			}

			// Compare message by message
			if len(got) != len(tt.messages) {
				t.Errorf("Round trip message count mismatch: got %d, want %d", len(got), len(tt.messages))
				return
			}

			for i := range got {
				// For Content field with JSON data
				if len(got[i].Content) == 1 && got[i].Content[0].Type() == content.TypeJSON {
					gotJSON, _ := json.Marshal(got[i].Content)
					wantJSON, _ := json.Marshal(tt.messages[i].Content)
					if !reflect.DeepEqual(gotJSON, wantJSON) {
						t.Errorf("Round trip message[%d] content got = %v, want %v", i, string(gotJSON), string(wantJSON))
					}
					// Compare other fields
					got[i].Content = nil
					tt.messages[i].Content = nil
				}

				// For ToolCalls field
				if len(got[i].ToolCalls) > 0 || len(tt.messages[i].ToolCalls) > 0 {
					if len(got[i].ToolCalls) != len(tt.messages[i].ToolCalls) {
						t.Errorf("Round trip message[%d] tool calls count mismatch got = %v, want %v",
							i, len(got[i].ToolCalls), len(tt.messages[i].ToolCalls))
						continue
					}

					for j := range got[i].ToolCalls {
						gotArgs, _ := json.Marshal(got[i].ToolCalls[j].Arguments)
						wantArgs, _ := json.Marshal(tt.messages[i].ToolCalls[j].Arguments)
						if !reflect.DeepEqual(gotArgs, wantArgs) {
							t.Errorf("Round trip message[%d] tool call %d arguments got = %v, want %v",
								i, j, string(gotArgs), string(wantArgs))
						}

						// Compare other fields
						got[i].ToolCalls[j].Arguments = nil
						tt.messages[i].ToolCalls[j].Arguments = nil
						if !reflect.DeepEqual(got[i].ToolCalls[j], tt.messages[i].ToolCalls[j]) {
							t.Errorf("Round trip message[%d] tool call %d other fields got = %v, want %v",
								i, j, got[i].ToolCalls[j], tt.messages[i].ToolCalls[j])
						}
					}

					// Done comparing tool calls
					got[i].ToolCalls = nil
					tt.messages[i].ToolCalls = nil
				}

				// Compare remaining fields
				if !reflect.DeepEqual(got[i], tt.messages[i]) {
					t.Errorf("Round trip message[%d] got = %v, want %v", i, got[i], tt.messages[i])
				}
			}
		})
	}
}
