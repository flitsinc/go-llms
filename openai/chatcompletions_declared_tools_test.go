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
// (minProperties), and a member beside the function's own (grammar).
const declaredTask = `{"name":"task","description":"Start a task","parameters":{"type":"object","properties":{"title":{"description":"What to do","type":"string"}},"required":["title"],"additionalProperties":false,"minProperties":1},"grammar":{"syntax":"lark"}}`

func declaredToolbox(t *testing.T) (*tools.Toolbox, tools.FunctionSchema) {
	t.Helper()
	var task tools.FunctionSchema
	require.NoError(t, json.Unmarshal([]byte(declaredTask), &task))
	note := tools.FunctionSchema{Name: "note", Description: "Write a note", Parameters: decodeValueSchema(t, `{"type":"object","properties":{"text":{"type":"string"}}}`)}
	run := func(r tools.Runner, params json.RawMessage) tools.Result { return tools.SuccessFromString("ok") }
	return tools.Box(
		tools.ExternalDeclared("Task", &task, json.RawMessage(declaredTask), run),
		tools.External("Note", &note, run),
	), note
}

func encodedTools(t *testing.T, payload map[string]any) []json.RawMessage {
	t.Helper()
	raw, err := json.Marshal(payload["tools"])
	require.NoError(t, err)
	var encoded []json.RawMessage
	require.NoError(t, json.Unmarshal(raw, &encoded))
	return encoded
}

// WithDeclaredTools sends a declared tool as its caller wrote it, and every
// other tool as before.
func TestChatCompletionsPayloadWithDeclaredToolsSendsTheDeclaration(t *testing.T) {
	toolbox, note := declaredToolbox(t)
	payload, err := NewChatCompletionsAPI("", "served").WithDeclaredTools().BuildPayload(nil, nil, toolbox, nil)
	require.NoError(t, err)

	encoded := encodedTools(t, payload)
	require.Len(t, encoded, 2)
	assert.Equal(t, `{"type":"function","function":`+declaredTask+`}`, string(encoded[0]))
	plainNote, err := json.Marshal(Tool{Type: "function", Function: &note})
	require.NoError(t, err)
	assert.Equal(t, string(plainNote), string(encoded[1]))
}

// Without the option a declared tool encodes from its schema, as any tool
// does: in ValueSchema's member order, without the keywords it drops.
func TestChatCompletionsPayloadEncodesADeclaredToolFromItsSchemaByDefault(t *testing.T) {
	toolbox, _ := declaredToolbox(t)
	payload, err := NewChatCompletionsAPI("", "served").BuildPayload(nil, nil, toolbox, nil)
	require.NoError(t, err)

	encoded := encodedTools(t, payload)
	require.Len(t, encoded, 2)
	assert.Equal(t,
		`{"type":"function","function":{"name":"task","description":"Start a task","parameters":{"type":"object","properties":{"title":{"description":"What to do","type":"string"}},"additionalProperties":false,"required":["title"]}}}`,
		string(encoded[0]))
}

// Strict mode pads a schema, which a declaration sent as written cannot be.
func TestChatCompletionsPayloadRefusesDeclaredToolsThatAreStrict(t *testing.T) {
	toolbox, _ := declaredToolbox(t)
	_, err := NewChatCompletionsAPI("", "served").WithDeclaredTools().WithStrictTools().BuildPayload(nil, nil, toolbox, nil)
	require.EqualError(t, err, "openai chat: declared tools cannot also be strict")
}
