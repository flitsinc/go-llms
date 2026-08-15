package openai

import (
	"encoding/json"
	"testing"

	"github.com/flitsinc/go-llms/tools"
)

func flatModeToolbox() *tools.Toolbox {
	patch := tools.FuncGrammar(tools.Lark("start: \"x\""), "apply_patch", "Apply a patch.", "apply_patch",
		func(r tools.Runner, input string) tools.Result { return tools.SuccessFromString("ok") })
	read := tools.Func("Read", "Read a file.", "read_file",
		func(r tools.Runner, params struct {
			Path string `json:"path"`
		}) tools.Result {
			return tools.SuccessFromString("ok")
		})
	return tools.Box(patch, read)
}

// marshalToolsArray round-trips the tools array through JSON, which is what
// the wire sees; the flat/nested distinction only matters in that form.
func marshalToolsArray(t *testing.T, apiTools []Tool) []map[string]any {
	t.Helper()
	data, err := json.Marshal(apiTools)
	if err != nil {
		t.Fatalf("marshal tools: %v", err)
	}
	var decoded []map[string]any
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("unmarshal tools: %v", err)
	}
	return decoded
}

func TestToolsFromToolbox_FlatCustomTools(t *testing.T) {
	apiTools, err := toolsFromToolbox(flatModeToolbox(), true)
	if err != nil {
		t.Fatalf("toolsFromToolbox: %v", err)
	}
	byName := map[string]map[string]any{}
	for _, entry := range marshalToolsArray(t, apiTools) {
		if name, ok := entry["name"].(string); ok {
			byName[name] = entry
		} else if fn, ok := entry["function"].(map[string]any); ok {
			byName[fn["name"].(string)] = entry
		}
	}

	patch := byName["apply_patch"]
	if patch == nil || patch["type"] != "custom" {
		t.Fatalf("expected flat custom apply_patch, got %#v", byName)
	}
	if _, nested := patch["custom"]; nested {
		t.Fatalf("flat mode must not emit the nested custom wrapper (OpenRouter's Responses upstream rejects it): %#v", patch)
	}
	format, ok := patch["format"].(map[string]any)
	if !ok || format["type"] != "grammar" || format["syntax"] != "lark" || format["definition"] != "start: \"x\"" {
		t.Fatalf("flat custom tool must carry the flat Responses-style format, got %#v", patch["format"])
	}
	if byName["read_file"]["type"] != "function" {
		t.Fatalf("JSON tools must stay function-typed, got %#v", byName["read_file"])
	}
}

func TestToolsFromToolbox_NestedCustomToolsUnchanged(t *testing.T) {
	apiTools, err := toolsFromToolbox(flatModeToolbox(), false)
	if err != nil {
		t.Fatalf("toolsFromToolbox: %v", err)
	}
	for _, entry := range marshalToolsArray(t, apiTools) {
		if entry["type"] != "custom" {
			continue
		}
		custom, ok := entry["custom"].(map[string]any)
		if !ok {
			t.Fatalf("nested mode must keep the custom wrapper, got %#v", entry)
		}
		format := custom["format"].(map[string]any)
		grammar, ok := format["grammar"].(map[string]any)
		if !ok || grammar["syntax"] != "lark" {
			t.Fatalf("nested mode must keep the nested grammar format, got %#v", format)
		}
		return
	}
	t.Fatal("no custom tool found in nested mode")
}

func TestChatToolChoice_FlatCustomForceAndAllowList(t *testing.T) {
	m := New("key", "gpt-5.6-luna").WithFlatCustomTools()
	toolbox := flatModeToolbox()

	toolbox.Choice = tools.Choice{Mode: tools.ChoiceRequireOneOf, AllowedTools: []string{"apply_patch"}}
	payload, err := m.BuildPayload(nil, nil, toolbox, nil)
	if err != nil {
		t.Fatalf("BuildPayload: %v", err)
	}
	forced, ok := payload["tool_choice"].(ChatToolChoice)
	if !ok || forced.Type != "custom" || forced.Name != "apply_patch" || forced.Function != nil {
		t.Fatalf("forcing a flat custom tool must use {type:custom,name}, got %#v", payload["tool_choice"])
	}

	toolbox.Choice = tools.Choice{Mode: tools.ChoiceAllowOnly, AllowedTools: []string{"apply_patch", "read_file"}}
	payload, err = m.BuildPayload(nil, nil, toolbox, nil)
	if err != nil {
		t.Fatalf("BuildPayload: %v", err)
	}
	allowed, ok := payload["tool_choice"].(ChatAllowedToolsChoice)
	if !ok {
		t.Fatalf("expected allowed_tools choice, got %#v", payload["tool_choice"])
	}
	types := map[string]string{}
	for _, entry := range allowed.Tools {
		if entry.Name != "" {
			types[entry.Name] = entry.Type
		} else if entry.Function != nil {
			types[entry.Function.Name] = entry.Type
		}
	}
	if types["apply_patch"] != "custom" || types["read_file"] != "function" {
		t.Fatalf("allow-list must keep declared types in flat mode, got %#v", types)
	}
}
