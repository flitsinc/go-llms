package openai

import (
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/tools"
)

// declaredTask is a declaration as a TypeScript caller writes it: required
// before additionalProperties, a keyword ValueSchema has no field for
// (minProperties), and a member beside the function's own (grammar). It is
// indented and holds HTML characters, which the wire compacts and escapes.
const declaredTask = `{
  "name": "task",
  "description": "Start a task",
  "parameters": {
    "type": "object",
    "properties": {"title": {"description": "What to do, as <verb> <object>", "type": "string"}},
    "required": ["title"],
    "additionalProperties": false,
    "minProperties": 1
  },
  "grammar": {"syntax": "lark"}
}`

// sentTask is declaredTask as WithDeclaredTools sends it: the same JSON value,
// members in the declared order, compacted, with HTML characters escaped.
const sentTask = `{"type":"function","function":{"name":"task","description":"Start a task","parameters":{"type":"object","properties":{"title":{"description":"What to do, as \u003cverb\u003e \u003cobject\u003e","type":"string"}},"required":["title"],"additionalProperties":false,"minProperties":1},"grammar":{"syntax":"lark"}}}`

// sentNote is the plain External note tool as every encoding without strict
// sends it.
const sentNote = `{"type":"function","function":{"name":"note","description":"Write a note","parameters":{"type":"object","properties":{"text":{"type":"string"}}}}}`

func runNothing(r tools.Runner, params json.RawMessage) tools.Result {
	return tools.SuccessFromString("ok")
}

func declaredTaskSchema(t *testing.T) *tools.FunctionSchema {
	t.Helper()
	var task tools.FunctionSchema
	require.NoError(t, json.Unmarshal([]byte(declaredTask), &task))
	return &task
}

func noteTool(t *testing.T) tools.Tool {
	t.Helper()
	note := tools.FunctionSchema{Name: "note", Description: "Write a note", Parameters: decodeValueSchema(t, `{"type":"object","properties":{"text":{"type":"string"}}}`)}
	return tools.External("Note", &note, runNothing)
}

func declaredToolbox(t *testing.T) *tools.Toolbox {
	t.Helper()
	return tools.Box(
		tools.ExternalDeclared("Task", declaredTaskSchema(t), json.RawMessage(declaredTask), runNothing),
		noteTool(t),
	)
}

func encodedTools(t *testing.T, payload map[string]any) []string {
	t.Helper()
	raw, err := json.Marshal(payload["tools"])
	require.NoError(t, err)
	var encoded []json.RawMessage
	require.NoError(t, json.Unmarshal(raw, &encoded))
	strs := make([]string, 0, len(encoded))
	for _, tool := range encoded {
		strs = append(strs, string(tool))
	}
	return strs
}

// WithDeclaredTools sends a declared tool as its declaration, and every other
// tool as before.
func TestChatCompletionsPayloadWithDeclaredToolsSendsTheDeclaration(t *testing.T) {
	payload, err := NewChatCompletionsAPI("", "served").WithDeclaredTools().BuildPayload(nil, nil, declaredToolbox(t), nil)
	require.NoError(t, err)

	assert.Equal(t, []string{sentTask, sentNote}, encodedTools(t, payload))
}

// Without the option a declared tool encodes from its schema, as any tool
// does: in ValueSchema's member order, without the keywords it drops.
func TestChatCompletionsPayloadEncodesADeclaredToolFromItsSchemaByDefault(t *testing.T) {
	payload, err := NewChatCompletionsAPI("", "served").BuildPayload(nil, nil, declaredToolbox(t), nil)
	require.NoError(t, err)

	assert.Equal(t, []string{
		`{"type":"function","function":{"name":"task","description":"Start a task","parameters":{"type":"object","properties":{"title":{"description":"What to do, as \u003cverb\u003e \u003cobject\u003e","type":"string"}},"additionalProperties":false,"required":["title"]}}}`,
		sentNote,
	}, encodedTools(t, payload))
}

// Strict mode pads a schema, which a declaration cannot be, so BuildPayload
// refuses a declared tool under both options and names it.
func TestChatCompletionsPayloadRefusesADeclaredToolThatIsStrict(t *testing.T) {
	_, err := NewChatCompletionsAPI("", "served").WithDeclaredTools().WithStrictTools().BuildPayload(nil, nil, declaredToolbox(t), nil)
	require.ErrorIs(t, err, ErrStrictDeclaredTools)
	assert.Contains(t, err.Error(), `"task"`)
}

// The refusal is about a tool, not the options: under both, a request without
// declared tools is sent strict, including an ExternalDeclared tool built
// without a declaration.
func TestChatCompletionsPayloadWithStrictAndDeclaredToolsSendsUndeclaredToolsStrict(t *testing.T) {
	m := NewChatCompletionsAPI("", "served").WithDeclaredTools().WithStrictTools()
	toolbox := tools.Box(
		tools.ExternalDeclared("Task", declaredTaskSchema(t), nil, runNothing),
		noteTool(t),
	)

	payload, err := m.BuildPayload(nil, nil, toolbox, nil)
	require.NoError(t, err)
	toolsArr := payloadField(t, payload["tools"]).([]any)
	require.Len(t, toolsArr, 2)
	for _, tool := range toolsArr {
		assert.Equal(t, true, tool.(map[string]any)["function"].(map[string]any)["strict"])
	}

	_, err = m.BuildPayload(nil, nil, nil, nil)
	require.NoError(t, err)
}

// A Tool built by hand cannot be both either: its encoding refuses rather
// than drop strict.
func TestToolRefusesADeclarationThatIsStrict(t *testing.T) {
	_, err := json.Marshal(Tool{Type: "function", Function: declaredTaskSchema(t), Strict: true, Declaration: json.RawMessage(declaredTask)})
	require.ErrorIs(t, err, ErrStrictDeclaredTools)
}
