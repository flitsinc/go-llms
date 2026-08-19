package openrouter

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

// strictToolbox is a tool declared the way an OpenAI-strict client declares
// one: "additionalProperties": false on every object plus validation keywords
// that Gemini rejects outright.
func strictToolbox(t *testing.T) *tools.Toolbox {
	t.Helper()
	const raw = `{
		"name": "edit_module",
		"description": "Edit a module.",
		"parameters": {
			"type": "object",
			"additionalProperties": false,
			"required": ["moduleId"],
			"properties": {
				"moduleId": {"type": "string", "pattern": "^mod_"},
				"edits": {
					"type": "array",
					"items": {"type": "object", "additionalProperties": false,
						"properties": {"find": {"type": "string"}}}
				}
			}
		}
	}`
	var schema tools.FunctionSchema
	require.NoError(t, json.Unmarshal([]byte(raw), &schema))
	return tools.Box(tools.External("Edit module", &schema,
		func(r tools.Runner, params json.RawMessage) tools.Result { return tools.SuccessFromString("ok") }))
}

func payloadJSON(t *testing.T, model string) string {
	t.Helper()
	payload, err := New("", model).BuildPayload(
		content.FromText("system"),
		[]llms.Message{{Role: "user", Content: content.FromText("hi")}},
		strictToolbox(t),
		nil,
	)
	require.NoError(t, err)
	data, err := json.Marshal(payload)
	require.NoError(t, err)
	return string(data)
}

// Every Gemini model reached through OpenRouter answered a strict-mode tool
// declaration with a 400 before this narrowing existed, so the request built
// for one must not carry the keywords Google refuses.
func TestNew_GeminiModelsNarrowToolSchemas(t *testing.T) {
	for _, model := range []string{
		"google/gemini-3.7-flash",
		"google/gemini-3-flash-preview",
		"google/gemini-3.1-pro-preview",
	} {
		t.Run(model, func(t *testing.T) {
			payload := payloadJSON(t, model)
			for _, keyword := range []string{"additionalProperties", "pattern"} {
				require.NotContains(t, payload, keyword,
					"Gemini rejects %q with a 400; it must not reach the wire", keyword)
			}
			require.Contains(t, payload, "moduleId", "the tool's properties must survive narrowing")
		})
	}
}

// Narrowing is a Gemini workaround, not a default. Models whose upstream
// honours strict mode must keep receiving the full schema.
func TestNew_NonGeminiModelsKeepStrictToolSchemas(t *testing.T) {
	for _, model := range []string{
		"openai/gpt-5.6-luna",
		"anthropic/claude-sonnet-4",
		"moonshotai/kimi-k2.7-code",
		// Google's non-Gemini listings are served by third-party endpoints
		// that take the full schema, so the prefix check must not catch them.
		"google/gemma-3-27b-it",
	} {
		t.Run(model, func(t *testing.T) {
			require.Contains(t, payloadJSON(t, model), "additionalProperties",
				"strict-mode schemas must reach providers that honour them")
		})
	}
}

func TestIsGeminiModel(t *testing.T) {
	for model, want := range map[string]bool{
		"google/gemini-3.7-flash":       true,
		"google/gemini-3-flash-preview": true,
		"google/gemma-3-27b-it":         false,
		"openai/gpt-5.6-luna":           false,
		"gemini-3.7-flash":              false, // not an OpenRouter id
	} {
		require.Equalf(t, want, isGeminiModel(model), "isGeminiModel(%q)", model)
	}
}

// Guards the reason the check lives on the model id: a Gemini row added later
// gets the narrowing without anyone remembering to ask for it.
func TestNew_GeminiNarrowingNeedsNoOptIn(t *testing.T) {
	payload := payloadJSON(t, "google/gemini-99-future-preview")
	require.False(t, strings.Contains(payload, "additionalProperties"),
		"a future Gemini row must be narrowed by default, not by opt-in")
}
