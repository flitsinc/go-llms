package openai

import (
	"context"
	"encoding/json"
	"strings"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
)

// Test that reasoning summary events stream deltas via Thought() and aggregate full text in Message().
func TestResponsesStream_ReasoningSummaries_DeltaAndAggregate(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"response.created"}`,
		`data: {"type":"response.output_item.added","item":{"type":"reasoning","id":"r_1"}}`,
		`data: {"type":"response.reasoning_summary_part.added","part":{"type":"summary_text","text":"Hello"},"item_id":"r_1"}`,
		`data: {"type":"response.reasoning_summary_text.delta","delta":" world","item_id":"r_1"}`,
		`data: {"type":"response.reasoning_summary_text.done","text":"Hello world","item_id":"r_1"}`,
		`data: {"type":"response.completed"}`,
		"",
	}, "\n")

	stream := &ResponsesStream{ctx: context.Background(), model: "gpt-5", stream: strings.NewReader(sse)}

	var deltas []string
	for yield := range stream.Iter() {
		switch yield {
		case llms.StreamStatusThinking:
			deltas = append(deltas, stream.Thought().Text)
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	// Expect two deltas: "Hello" then " world"
	if len(deltas) != 2 {
		t.Fatalf("expected 2 thinking deltas, got %d (%v)", len(deltas), deltas)
	}
	if deltas[0] != "Hello" {
		t.Fatalf("expected first delta 'Hello', got %q", deltas[0])
	}
	if deltas[1] != " world" {
		t.Fatalf("expected second delta ' world', got %q", deltas[1])
	}

	// Verify the aggregated thought in the final message content is full text
	msg := stream.Message()
	var found *content.Thought
	for _, it := range msg.Content {
		if th, ok := it.(*content.Thought); ok && th.ID == "r_1" {
			found = th
			break
		}
	}
	if found == nil {
		t.Fatalf("expected aggregated thought with ID r_1 in message content")
	}
	if got := found.Text; got != "Hello world" {
		t.Fatalf("expected aggregated text 'Hello world', got %q", got)
	}
	if !found.Summary {
		t.Fatalf("expected aggregated thought to be marked as summary")
	}
}

// Test that assistant history conversion uses output_text and includes reasoning summary as full text.
func TestConvertMessageToInput_AssistantReasoningAndOutput(t *testing.T) {
	msg := llms.Message{
		Role: "assistant",
		Content: content.Content{
			&content.Text{Text: "Final answer."},
			&content.Thought{ID: "r_99", Text: "Concise internal reasoning summary.", Summary: true},
		},
	}

	inputs, err := convertMessageToInput(msg)
	if err != nil {
		t.Fatalf("convertMessageToInput returned error: %v", err)
	}
	if len(inputs) == 0 {
		t.Fatalf("expected non-empty inputs")
	}

	var haveOutputMessage bool
	var haveReasoning bool
	for _, in := range inputs {
		switch v := in.(type) {
		case OutputMessage:
			haveOutputMessage = true
			if v.Role != "assistant" {
				t.Fatalf("expected OutputMessage role 'assistant', got %q", v.Role)
			}
			if len(v.Content) == 0 {
				t.Fatalf("expected OutputMessage to have content")
			}
			// Ensure content items are output_text
			if ot, ok := v.Content[0].(OutputText); !ok || ot.Type != "output_text" || ot.Text == "" {
				t.Fatalf("expected first content to be OutputText with non-empty text, got %#v", v.Content[0])
			}
		case Reasoning:
			haveReasoning = true
			if v.ID != "r_99" {
				t.Fatalf("expected reasoning ID 'r_99', got %q", v.ID)
			}
			if len(v.Summary) == 0 || v.Summary[0].Text != "Concise internal reasoning summary." {
				t.Fatalf("expected reasoning summary text, got %#v", v.Summary)
			}
		}
	}
	if !haveOutputMessage {
		t.Fatalf("expected an OutputMessage with output_text content")
	}
	if !haveReasoning {
		t.Fatalf("expected a Reasoning item with summary text")
	}
}

func TestResponsesStream_UsageWithCachedTokens(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"response.created"}`,
		`data: {"type":"response.output_item.added","item":{"type":"message","role":"assistant"}}`,
		`data: {"type":"response.content_part.added","part":{"type":"text","text":"Hello"},"item_id":"msg_1","content_index":0}`,
		`data: {"type":"response.content_part.done","part":{"type":"text","text":"Hello"},"item_id":"msg_1","content_index":0}`,
		`data: {"type":"response.completed","response":{"usage":{"input_tokens":100,"output_tokens":50,"total_tokens":150,"input_tokens_details":{"cached_tokens":25},"output_tokens_details":{"reasoning_tokens":10}}}}`,
		"",
	}, "\n")

	stream := &ResponsesStream{ctx: context.Background(), model: "gpt-4o", stream: strings.NewReader(sse)}

	for yield := range stream.Iter() {
		_ = yield
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	usage := stream.Usage()
	if usage.InputTokens != 100 {
		t.Errorf("expected InputTokens=100, got %d", usage.InputTokens)
	}
	if usage.OutputTokens != 50 {
		t.Errorf("expected OutputTokens=50, got %d", usage.OutputTokens)
	}
	if usage.CachedInputTokens != 25 {
		t.Errorf("expected CachedInputTokens=25, got %d", usage.CachedInputTokens)
	}
}

func TestConvertMessageToInput_MissingToolExtraIDReturnsError(t *testing.T) {
	msg := llms.Message{
		Role: "assistant",
		Content: content.Content{
			&content.Thought{ID: "rs_missing", Text: "Planning..."},
		},
		ToolCalls: []llms.ToolCall{
			{
				ID:        "call_missing",
				Name:      "run_shell_cmd",
				Arguments: json.RawMessage(`{"command":"ls"}`),
			},
		},
	}

	_, err := convertMessageToInput(msg)
	if err == nil {
		t.Fatalf("expected error for tool call missing ExtraID, got nil")
	}
	if !strings.Contains(err.Error(), "missing output item id") {
		t.Fatalf("expected missing output item id error, got %v", err)
	}
}

func TestConvertMessageToInput_ReasoningPairedWithToolCall(t *testing.T) {
	msg := llms.Message{
		Role: "assistant",
		Content: content.Content{
			&content.Thought{ID: "rs_pair", Text: "Need to inspect files"},
		},
		ToolCalls: []llms.ToolCall{
			{
				ID:        "call1",
				Name:      "run_shell_cmd",
				Arguments: json.RawMessage(`{"command":"ls"}`),
				ExtraID:   "fc_123",
			},
		},
	}

	items, err := convertMessageToInput(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 2 {
		t.Fatalf("expected 2 items (reasoning + tool), got %d", len(items))
	}
	if _, ok := items[0].(Reasoning); !ok {
		t.Fatalf("expected first item to be Reasoning, got %T", items[0])
	}
	fc, ok := items[1].(FunctionCall)
	if !ok {
		t.Fatalf("expected second item to be FunctionCall, got %T", items[1])
	}
	if fc.ID != "fc_123" {
		t.Fatalf("expected FunctionCall ID 'fc_123', got %q", fc.ID)
	}
	if fc.CallID != "call1" {
		t.Fatalf("expected FunctionCall CallID 'call1', got %q", fc.CallID)
	}
}
