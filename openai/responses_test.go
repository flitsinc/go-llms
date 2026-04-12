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

	// Expect three deltas: initial empty (from output_item.added), "Hello", then " world"
	if len(deltas) != 3 {
		t.Fatalf("expected 3 thinking deltas, got %d (%v)", len(deltas), deltas)
	}
	if deltas[0] != "" {
		t.Fatalf("expected first delta '' (initial reasoning), got %q", deltas[0])
	}
	if deltas[1] != "Hello" {
		t.Fatalf("expected second delta 'Hello', got %q", deltas[1])
	}
	if deltas[2] != " world" {
		t.Fatalf("expected third delta ' world', got %q", deltas[2])
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

func TestResponsesStream_ReasoningSummaryDoneWithoutDeltas(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"response.created"}`,
		`data: {"type":"response.output_item.added","item":{"type":"reasoning","id":"r_done_only"},"output_index":0}`,
		`data: {"type":"response.reasoning_summary_text.done","item_id":"r_done_only","text":"Final reasoning summary."}`,
		`data: {"type":"response.output_item.done","item":{"type":"reasoning","id":"r_done_only","summary":[{"type":"summary_text","text":"Final reasoning summary."}]},"output_index":0}`,
		`data: {"type":"response.completed"}`,
		"",
	}, "\n")

	stream := &ResponsesStream{ctx: context.Background(), model: "gpt-5", stream: strings.NewReader(sse)}

	var thinkingDone int
	for status := range stream.Iter() {
		if status == llms.StreamStatusThinkingDone {
			thinkingDone++
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}
	if thinkingDone != 1 {
		t.Fatalf("expected 1 thinking done event, got %d", thinkingDone)
	}

	msg := stream.Message()
	var summary *content.Thought
	for _, it := range msg.Content {
		if th, ok := it.(*content.Thought); ok && th.ID == "r_done_only" {
			summary = th
			break
		}
	}
	if summary == nil {
		t.Fatalf("expected reasoning thought for r_done_only")
	}
	if summary.Text != "Final reasoning summary." {
		t.Fatalf("expected reasoning text to match done payload, got %q", summary.Text)
	}
	if !summary.Summary {
		t.Fatalf("expected reasoning thought to be marked summary")
	}
}

func TestResponsesStream_ReasoningOutputDoneSummaryFallback(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"response.created"}`,
		`data: {"type":"response.output_item.added","item":{"type":"reasoning","id":"r_done_only"},"output_index":0}`,
		`data: {"type":"response.output_item.done","item":{"type":"reasoning","id":"r_done_only","summary":[{"type":"summary_text","text":"Summary from output item."}]},"output_index":0}`,
		`data: {"type":"response.completed"}`,
		"",
	}, "\n")

	stream := &ResponsesStream{ctx: context.Background(), model: "gpt-5", stream: strings.NewReader(sse)}
	for range stream.Iter() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	msg := stream.Message()
	var summary *content.Thought
	for _, it := range msg.Content {
		if th, ok := it.(*content.Thought); ok && th.ID == "r_done_only" {
			summary = th
			break
		}
	}
	if summary == nil {
		t.Fatalf("expected reasoning thought for r_done_only")
	}
	if summary.Text != "Summary from output item." {
		t.Fatalf("expected reasoning text to come from output item, got %q", summary.Text)
	}
	if !summary.Summary {
		t.Fatalf("expected reasoning thought to be marked summary")
	}
}

// Test that reasoning without summary text still emits Thinking + ThinkingDone
// so callers always see the reasoning ID for round-tripping with OpenAI.
func TestResponsesStream_ReasoningWithoutSummary(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"response.created"}`,
		`data: {"type":"response.output_item.added","item":{"type":"reasoning","id":"rs_nosummary"},"output_index":0}`,
		`data: {"type":"response.output_item.done","item":{"type":"reasoning","id":"rs_nosummary","summary":[]},"output_index":0}`,
		`data: {"type":"response.output_item.added","item":{"type":"function_call","id":"fc_123","name":"test_tool","call_id":"call_1","arguments":"{}"},"output_index":1}`,
		`data: {"type":"response.output_item.done","item":{"type":"function_call","id":"fc_123","name":"test_tool","call_id":"call_1","arguments":"{}"},"output_index":1}`,
		`data: {"type":"response.completed"}`,
		"",
	}, "\n")

	stream := &ResponsesStream{ctx: context.Background(), model: "gpt-5", stream: strings.NewReader(sse)}

	var thinkingCount, thinkingDoneCount int
	for status := range stream.Iter() {
		switch status {
		case llms.StreamStatusThinking:
			thinkingCount++
		case llms.StreamStatusThinkingDone:
			thinkingDoneCount++
		}
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	if thinkingCount != 1 {
		t.Fatalf("expected 1 thinking event (from output_item.added), got %d", thinkingCount)
	}
	if thinkingDoneCount != 1 {
		t.Fatalf("expected 1 thinking done event (from output_item.done), got %d", thinkingDoneCount)
	}

	// Verify the thought exists in the message with the ID
	msg := stream.Message()
	var found *content.Thought
	for _, it := range msg.Content {
		if th, ok := it.(*content.Thought); ok && th.ID == "rs_nosummary" {
			found = th
			break
		}
	}
	if found == nil {
		t.Fatalf("expected thought with ID rs_nosummary in message content")
	}
	if !found.Summary {
		t.Fatalf("expected thought to be marked as summary")
	}
}

// Test that assistant history conversion uses output_text and includes reasoning summary as full text.
func TestConvertMessageToInput_AssistantReasoningAndOutput(t *testing.T) {
	msg := llms.Message{
		Role: "assistant",
		Content: content.Content{
			&content.Thought{ID: "r_99", Text: "Concise internal reasoning summary.", Summary: true},
			&content.Text{Text: "Final answer."},
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

func TestConvertMessageToInput_MissingOpenAIItemIDGeneratesSyntheticID(t *testing.T) {
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

	items, err := convertMessageToInput(msg)
	if err != nil {
		t.Fatalf("expected no error for tool call missing openai:item_id metadata, got %v", err)
	}

	// Should have reasoning + function_call
	if len(items) != 2 {
		t.Fatalf("expected 2 items, got %d", len(items))
	}

	fc, ok := items[1].(FunctionCall)
	if !ok {
		t.Fatalf("expected FunctionCall, got %T", items[1])
	}
	if fc.ID != "fc_synthetic_call_missing" {
		t.Fatalf("expected synthetic item ID 'fc_synthetic_call_missing', got %q", fc.ID)
	}
	if fc.CallID != "call_missing" {
		t.Fatalf("expected CallID 'call_missing', got %q", fc.CallID)
	}
}

func TestConvertMessageToInput_CrossProviderToolCallIDs(t *testing.T) {
	// Simulates tool calls from Anthropic (toolu_01*) being replayed through OpenAI
	msg := llms.Message{
		Role: "assistant",
		Content: content.Content{
			&content.Text{Text: "I'll read the module."},
		},
		ToolCalls: []llms.ToolCall{
			{
				ID:        "toolu_01GRVUqdmDwoW27uDQJ378yi",
				Name:      "readModule",
				Arguments: json.RawMessage(`{"module_name":"Views.Home"}`),
			},
		},
	}

	items, err := convertMessageToInput(msg)
	if err != nil {
		t.Fatalf("expected no error for cross-provider tool call, got %v", err)
	}

	// Should have output_text + function_call
	if len(items) != 2 {
		t.Fatalf("expected 2 items, got %d", len(items))
	}

	fc, ok := items[1].(FunctionCall)
	if !ok {
		t.Fatalf("expected FunctionCall, got %T", items[1])
	}
	if fc.ID != "fc_synthetic_toolu_01GRVUqdmDwoW27uDQJ378yi" {
		t.Fatalf("expected synthetic item ID, got %q", fc.ID)
	}
	if fc.CallID != "toolu_01GRVUqdmDwoW27uDQJ378yi" {
		t.Fatalf("expected original CallID preserved, got %q", fc.CallID)
	}
	if fc.Name != "readModule" {
		t.Fatalf("expected tool name 'readModule', got %q", fc.Name)
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
				Metadata: map[string]string{
					"openai:item_id":   "fc_123",
					"openai:item_type": "function_call",
				},
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

func TestConvertMessageToInput_PreservesReasoningOrderAcrossToolCalls(t *testing.T) {
	msg := llms.Message{
		Role: "assistant",
		Content: content.Content{
			&content.Thought{ID: "rs_tool", Text: "Need a tool call", Summary: true},
			&content.Text{Text: "Working..."},
		},
		ToolCalls: []llms.ToolCall{
			{
				ID:        "call1",
				Name:      "run_shell_cmd",
				Arguments: json.RawMessage(`{"command":"ls"}`),
				Metadata: map[string]string{
					"openai:item_id":   "fc_999",
					"openai:item_type": "function_call",
				},
			},
			{
				ID:        "call2",
				Name:      "run_shell_cmd",
				Arguments: json.RawMessage(`{"command":"pwd"}`),
				Metadata: map[string]string{
					"openai:item_id":   "fc_1000",
					"openai:item_type": "function_call",
				},
			},
		},
	}

	items, err := convertMessageToInput(msg)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(items) != 4 {
		t.Fatalf("expected 4 items, got %d (%#v)", len(items), items)
	}

	if r, ok := items[0].(Reasoning); !ok || r.ID != "rs_tool" {
		t.Fatalf("expected first item to be reasoning rs_tool, got %#v", items[0])
	}
	if _, ok := items[1].(OutputMessage); !ok {
		t.Fatalf("expected second item to be OutputMessage, got %T", items[1])
	}
	if fc, ok := items[2].(FunctionCall); !ok || fc.ID != "fc_999" {
		t.Fatalf("expected third item to be FunctionCall fc_999, got %#v", items[2])
	}
	if fc, ok := items[3].(FunctionCall); !ok || fc.ID != "fc_1000" {
		t.Fatalf("expected fourth item to be FunctionCall fc_1000, got %#v", items[3])
	}
}

func TestConvertMessageToInput_MultiMessageReasoningToolTextSequence(t *testing.T) {
	messages := []llms.Message{
		{
			Role: "assistant",
			Content: content.Content{
				&content.Thought{ID: "rs_tool", Text: "Need tool output", Summary: true},
				&content.Text{Text: "I'll run pwd and ls before summarizing."},
			},
			ToolCalls: []llms.ToolCall{
				{
					ID:        "tool_call_1",
					Name:      "run_shell_cmd",
					Arguments: json.RawMessage(`{"command":"ls"}`),
					Metadata: map[string]string{
						"openai:item_id":   "fc_111",
						"openai:item_type": "function_call",
					},
				},
			},
		},
		{
			Role:    "user",
			Content: content.FromText("Here are the tool results."),
		},
		{
			Role:       "tool",
			ToolCallID: "tool_call_1",
			Content:    content.FromText("file_a\nfile_b"),
		},
		{
			Role: "assistant",
			Content: content.Content{
				&content.Thought{ID: "rs_final", Text: "Summarize results", Summary: true},
				&content.Text{Text: "Listed files successfully."},
			},
		},
	}

	var sequence []ResponseInput
	for i, msg := range messages {
		items, err := convertMessageToInput(msg)
		if err != nil {
			t.Fatalf("message %d conversion failed: %v", i+1, err)
		}
		sequence = append(sequence, items...)
	}

	if len(sequence) != 7 {
		t.Fatalf("expected 7 replay items, got %d (%#v)", len(sequence), sequence)
	}

	if r, ok := sequence[0].(Reasoning); !ok || r.ID != "rs_tool" {
		t.Fatalf("expected first item reasoning rs_tool, got %#v", sequence[0])
	}
	if _, ok := sequence[1].(OutputMessage); !ok {
		t.Fatalf("expected second item assistant OutputMessage announcing plan, got %#v", sequence[1])
	}
	if fc, ok := sequence[2].(FunctionCall); !ok || fc.ID != "fc_111" {
		t.Fatalf("expected third item function call fc_111, got %#v", sequence[2])
	}
	if inMsg, ok := sequence[3].(InputMessage); !ok || inMsg.Role != "user" {
		t.Fatalf("expected fourth item user InputMessage, got %#v", sequence[3])
	}
	if out, ok := sequence[4].(FunctionCallOutput); !ok || out.CallID != "tool_call_1" {
		t.Fatalf("expected fifth item FunctionCallOutput for tool_call_1, got %#v", sequence[4])
	}
	if r, ok := sequence[5].(Reasoning); !ok || r.ID != "rs_final" {
		t.Fatalf("expected sixth item reasoning rs_final, got %#v", sequence[5])
	}
	if outMsg, ok := sequence[6].(OutputMessage); !ok || len(outMsg.Content) == 0 {
		t.Fatalf("expected seventh item assistant OutputMessage, got %#v", sequence[6])
	}
}

// Test that the phase field is parsed from streaming events and stored in message metadata.
func TestResponsesStream_PhaseFieldParsed(t *testing.T) {
	sse := strings.Join([]string{
		`data: {"type":"response.created","response":{"id":"resp_1"}}`,
		`data: {"type":"response.output_item.added","item":{"type":"message","id":"msg_1","role":"assistant","phase":"commentary"},"output_index":0}`,
		`data: {"type":"response.output_text.delta","delta":"I'll start by analyzing...","item_id":"msg_1","content_index":0}`,
		`data: {"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`,
		"",
	}, "\n")

	stream := &ResponsesStream{ctx: context.Background(), model: "gpt-5.4", stream: strings.NewReader(sse)}
	for range stream.Iter() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	msg := stream.Message()
	if msg.Metadata == nil {
		t.Fatalf("expected message metadata to be non-nil")
	}
	if got := msg.Metadata["openai:phase"]; got != "commentary" {
		t.Fatalf("expected openai:phase='commentary', got %q", got)
	}
}

// Test that phase is preserved through the full round-trip: stream -> Message -> convertMessageToInput -> OutputMessage.
func TestConvertMessageToInput_PhasePreservedRoundTrip(t *testing.T) {
	msg := llms.Message{
		Role: "assistant",
		Content: content.Content{
			&content.Text{Text: "Here's the refactored code..."},
		},
		Metadata: map[string]string{
			"openai:phase": "final_answer",
		},
	}

	inputs, err := convertMessageToInput(msg)
	if err != nil {
		t.Fatalf("convertMessageToInput returned error: %v", err)
	}
	if len(inputs) != 1 {
		t.Fatalf("expected 1 input, got %d", len(inputs))
	}

	outMsg, ok := inputs[0].(OutputMessage)
	if !ok {
		t.Fatalf("expected OutputMessage, got %T", inputs[0])
	}
	if outMsg.Phase != "final_answer" {
		t.Fatalf("expected Phase='final_answer', got %q", outMsg.Phase)
	}
}

// Test that phase is correctly set to "commentary" on replay.
func TestConvertMessageToInput_CommentaryPhase(t *testing.T) {
	msg := llms.Message{
		Role: "assistant",
		Content: content.Content{
			&content.Text{Text: "I'll start by analyzing the auth module..."},
		},
		Metadata: map[string]string{
			"openai:phase": "commentary",
		},
	}

	inputs, err := convertMessageToInput(msg)
	if err != nil {
		t.Fatalf("convertMessageToInput returned error: %v", err)
	}
	if len(inputs) != 1 {
		t.Fatalf("expected 1 input, got %d", len(inputs))
	}

	outMsg, ok := inputs[0].(OutputMessage)
	if !ok {
		t.Fatalf("expected OutputMessage, got %T", inputs[0])
	}
	if outMsg.Phase != "commentary" {
		t.Fatalf("expected Phase='commentary', got %q", outMsg.Phase)
	}
}

// Test that messages without phase metadata produce OutputMessage with empty Phase (omitted in JSON).
func TestConvertMessageToInput_NoPhaseOmitted(t *testing.T) {
	msg := llms.Message{
		Role: "assistant",
		Content: content.Content{
			&content.Text{Text: "Hello!"},
		},
	}

	inputs, err := convertMessageToInput(msg)
	if err != nil {
		t.Fatalf("convertMessageToInput returned error: %v", err)
	}
	if len(inputs) != 1 {
		t.Fatalf("expected 1 input, got %d", len(inputs))
	}

	outMsg, ok := inputs[0].(OutputMessage)
	if !ok {
		t.Fatalf("expected OutputMessage, got %T", inputs[0])
	}
	if outMsg.Phase != "" {
		t.Fatalf("expected empty Phase for message without metadata, got %q", outMsg.Phase)
	}

	// Verify it's omitted from JSON
	data, err := json.Marshal(outMsg)
	if err != nil {
		t.Fatalf("json.Marshal error: %v", err)
	}
	if strings.Contains(string(data), "phase") {
		t.Fatalf("expected 'phase' to be omitted from JSON when empty, got %s", data)
	}
}

// Test that phase is preserved alongside reasoning and tool calls in a complex multi-item sequence.
func TestConvertMessageToInput_PhaseWithReasoningAndToolCalls(t *testing.T) {
	msg := llms.Message{
		Role: "assistant",
		Content: content.Content{
			&content.Thought{ID: "rs_1", Text: "Planning...", Summary: true},
			&content.Text{Text: "I'll run this command."},
		},
		ToolCalls: []llms.ToolCall{
			{
				ID:        "call_1",
				Name:      "run_cmd",
				Arguments: json.RawMessage(`{"cmd":"ls"}`),
				Metadata: map[string]string{
					"openai:item_id":   "fc_1",
					"openai:item_type": "function_call",
				},
			},
		},
		Metadata: map[string]string{
			"openai:phase": "commentary",
		},
	}

	inputs, err := convertMessageToInput(msg)
	if err != nil {
		t.Fatalf("convertMessageToInput returned error: %v", err)
	}

	// Should be: Reasoning, OutputMessage (with phase), FunctionCall
	if len(inputs) != 3 {
		t.Fatalf("expected 3 inputs, got %d (%#v)", len(inputs), inputs)
	}

	if _, ok := inputs[0].(Reasoning); !ok {
		t.Fatalf("expected first item to be Reasoning, got %T", inputs[0])
	}
	outMsg, ok := inputs[1].(OutputMessage)
	if !ok {
		t.Fatalf("expected second item to be OutputMessage, got %T", inputs[1])
	}
	if outMsg.Phase != "commentary" {
		t.Fatalf("expected OutputMessage Phase='commentary', got %q", outMsg.Phase)
	}
	if _, ok := inputs[2].(FunctionCall); !ok {
		t.Fatalf("expected third item to be FunctionCall, got %T", inputs[2])
	}
}
