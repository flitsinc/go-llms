package openai

import (
	"encoding/json"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

func TestConvertMessageToInput_CustomToolCallResultUsesCustomOutput(t *testing.T) {
	messages := []llms.Message{
		{
			Role: "assistant",
			ToolCalls: []llms.ToolCall{
				{
					ID:        "call_custom",
					Name:      "apply_patch",
					Arguments: json.RawMessage("*** Begin Patch\n*** End Patch"),
					Metadata:  map[string]string{"openai:item_type": "custom_tool_call", "openai:item_id": "item_1"},
				},
				{
					ID:        "call_fn",
					Name:      "read_file",
					Arguments: json.RawMessage(`{"path":"a"}`),
					Metadata:  map[string]string{"openai:item_type": "function_call", "openai:item_id": "item_2"},
				},
			},
		},
		{Role: "tool", ToolCallID: "call_custom", Content: content.FromText("ok")},
		{Role: "tool", ToolCallID: "call_fn", Content: content.FromText("data")},
	}

	customCallIDs := customToolCallIDs(messages)
	if len(customCallIDs) != 1 || !customCallIDs["call_custom"] {
		t.Fatalf("expected only call_custom to be collected, got %#v", customCallIDs)
	}

	items, err := convertMessageToInput(messages[1], customCallIDs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(items) != 1 {
		t.Fatalf("expected 1 item, got %d (%#v)", len(items), items)
	}
	customOut, ok := items[0].(CustomToolCallOutput)
	if !ok {
		t.Fatalf("a custom tool call's result must be a custom_tool_call_output (function_call_output is rejected by the API), got %#v", items[0])
	}
	if customOut.Type != "custom_tool_call_output" || customOut.CallID != "call_custom" || customOut.Output != "ok" {
		t.Fatalf("unexpected custom output: %#v", customOut)
	}

	items, err = convertMessageToInput(messages[2], customCallIDs)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, ok := items[0].(FunctionCallOutput); !ok {
		t.Fatalf("a function call's result must stay a function_call_output, got %#v", items[0])
	}
}

func TestBuildToolChoice_CustomToolsKeepTheirDeclaredType(t *testing.T) {
	toolsArr := []any{
		map[string]any{"type": "custom", "name": "apply_patch", "format": map[string]any{"type": "grammar"}},
		FunctionTool{Type: "function", Name: "read_file"},
	}

	forced, err := buildToolChoice(tools.Choice{Mode: tools.ChoiceRequireOneOf, AllowedTools: []string{"apply_patch"}}, toolsArr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	forcedMap, ok := forced.(map[string]any)
	if !ok || forcedMap["type"] != "custom" || forcedMap["name"] != "apply_patch" {
		t.Fatalf("forcing a custom tool must reference it as type custom, got %#v", forced)
	}

	allowed, err := buildToolChoice(tools.Choice{Mode: tools.ChoiceAllowOnly, AllowedTools: []string{"apply_patch", "read_file"}}, toolsArr)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	allowedChoice, ok := allowed.(AllowedToolsToolChoice)
	if !ok {
		t.Fatalf("expected AllowedToolsToolChoice, got %#v", allowed)
	}
	types := map[string]string{}
	for _, entry := range allowedChoice.Tools {
		m, ok := entry.(map[string]any)
		if !ok {
			t.Fatalf("unexpected entry %#v", entry)
		}
		types[m["name"].(string)] = m["type"].(string)
	}
	if types["apply_patch"] != "custom" || types["read_file"] != "function" {
		t.Fatalf("allow-list must keep each tool's declared type, got %#v", types)
	}
}
