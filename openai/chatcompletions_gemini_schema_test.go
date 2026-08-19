package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/flitsinc/go-llms/tools"
)

// editorShapedToolbox reproduces the tool declaration an OpenAI-strict client
// sends: "additionalProperties": false on every object, optional fields encoded
// as anyOf with a null branch, and validation keywords Gemini has never
// implemented. Properties arrive as decoded JSON (not typed ValueSchema)
// because that is what an external caller's schema looks like after the API
// unmarshals it off the wire.
func editorShapedToolbox(t *testing.T) *tools.Toolbox {
	t.Helper()
	const raw = `{
		"name": "edit_module",
		"description": "Edit a module.",
		"parameters": {
			"type": "object",
			"additionalProperties": false,
			"required": ["moduleId", "edits"],
			"properties": {
				"moduleId": {"type": "string", "pattern": "^mod_[a-z]+$", "minLength": 4},
				"kind": {"const": "component"},
				"edits": {
					"type": "array",
					"items": {
						"type": "object",
						"additionalProperties": false,
						"properties": {
							"find": {"type": "string"},
							"replace": {"anyOf": [{"type": "string"}, {"type": "null"}]}
						}
					}
				}
			}
		}
	}`
	var schema tools.FunctionSchema
	if err := json.Unmarshal([]byte(raw), &schema); err != nil {
		t.Fatalf("unmarshal schema: %v", err)
	}
	return tools.Box(tools.External("Edit module", &schema,
		func(r tools.Runner, params json.RawMessage) tools.Result { return tools.SuccessFromString("ok") }))
}

func toolsJSON(t *testing.T, toolbox *tools.Toolbox, geminiToolSchemas bool) string {
	t.Helper()
	apiTools, err := toolsFromToolbox(toolbox, true, geminiToolSchemas)
	if err != nil {
		t.Fatalf("toolsFromToolbox: %v", err)
	}
	data, err := json.Marshal(apiTools)
	if err != nil {
		t.Fatalf("marshal tools: %v", err)
	}
	return string(data)
}

// Gemini answers a declaration carrying any of these with a 400 before billing
// a token, so none may survive into the request.
var geminiRejectedKeywords = []string{"additionalProperties", "pattern", "minLength", "const"}

func TestToolsFromToolbox_GeminiSchemasDropRejectedKeywords(t *testing.T) {
	payload := toolsJSON(t, editorShapedToolbox(t), true)

	for _, keyword := range geminiRejectedKeywords {
		if strings.Contains(payload, keyword) {
			t.Errorf("Gemini mode still sends %q, which Google rejects with a 400: %s", keyword, payload)
		}
	}

	// Narrowing must not cost the model the information it needs to call the
	// tool: the properties and their nesting have to survive intact.
	for _, kept := range []string{"moduleId", "edits", "find", "replace", "items"} {
		if !strings.Contains(payload, kept) {
			t.Errorf("Gemini mode dropped %q, which the model needs to call the tool: %s", kept, payload)
		}
	}
}

// The narrowing is a workaround for one upstream, not an improvement. Applying
// it to every OpenRouter model would silently weaken strict-mode tool calling
// on providers that honour these keywords.
func TestToolsFromToolbox_NonGeminiKeepsStrictKeywords(t *testing.T) {
	payload := toolsJSON(t, editorShapedToolbox(t), false)

	if !strings.Contains(payload, "additionalProperties") {
		t.Errorf("non-Gemini mode must leave strict-mode schemas alone: %s", payload)
	}
}

// A toolbox is shared across providers and across turns. Narrowing a schema for
// Gemini must not reach back into it, or an unrelated request would silently
// inherit the weakened copy.
func TestToolsFromToolbox_GeminiSchemasDoNotMutateToolbox(t *testing.T) {
	toolbox := editorShapedToolbox(t)

	before := toolsJSON(t, toolbox, false)
	toolsJSON(t, toolbox, true) // the pass that could write through to the toolbox
	after := toolsJSON(t, toolbox, false)

	if before != after {
		t.Errorf("narrowing for Gemini mutated the shared toolbox\nbefore: %s\nafter:  %s", before, after)
	}
}
