package openai

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/metalim/jsonmap"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/internal/schematree"
	"github.com/flitsinc/go-llms/tools"
)

// decodeValueSchema builds a schema the way a caller decoding JSON does, so
// property values take the jsonmap shape ValueSchema.UnmarshalJSON produces.
func decodeValueSchema(t *testing.T, raw string) tools.ValueSchema {
	t.Helper()
	var schema tools.ValueSchema
	require.NoError(t, json.Unmarshal([]byte(raw), &schema))
	return schema
}

func encodeValueSchema(t *testing.T, schema *tools.ValueSchema) string {
	t.Helper()
	raw, err := json.Marshal(schema)
	require.NoError(t, err)
	return string(raw)
}

func strictSchemaJSON(t *testing.T, raw string) string {
	t.Helper()
	schema := decodeValueSchema(t, raw)
	tree, err := schematree.Of(&schema)
	require.NoError(t, err)
	schematree.WalkObjects(tree, padStrictObject)
	encoded, err := json.Marshal(tree)
	require.NoError(t, err)
	return string(encoded)
}

func TestStrictSchemaWrapsOptionalPropertiesAndRequiresEverything(t *testing.T) {
	padded := strictSchemaJSON(t, `{
		"type": "object",
		"properties": {
			"name": {"type": "string"},
			"count": {"type": "number", "description": "An optional count"}
		},
		"required": ["name"],
		"additionalProperties": false
	}`)

	assert.JSONEq(t, `{
		"type": "object",
		"properties": {
			"name": {"type": "string"},
			"count": {"anyOf": [{"type": "number", "description": "An optional count"}, {"type": "null"}]}
		},
		"required": ["name", "count"],
		"additionalProperties": false
	}`, padded)
}

func TestStrictSchemaRecursesIntoNestedObjectsArraysAndAnyOf(t *testing.T) {
	padded := strictSchemaJSON(t, `{
		"type": "object",
		"properties": {
			"outer": {"type": "object", "properties": {"inner": {"type": "string"}}},
			"items": {
				"type": "array",
				"items": {"type": "object", "properties": {"desc": {"type": "string"}}}
			},
			"either": {
				"anyOf": [
					{"type": "object", "properties": {"x": {"type": "string"}}},
					{"type": "string"}
				]
			}
		},
		"required": ["outer", "items", "either"]
	}`)

	assert.JSONEq(t, `{
		"type": "object",
		"properties": {
			"outer": {
				"type": "object",
				"properties": {"inner": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
				"required": ["inner"]
			},
			"items": {
				"type": "array",
				"items": {
					"type": "object",
					"properties": {"desc": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
					"required": ["desc"]
				}
			},
			"either": {
				"anyOf": [
					{"type": "object", "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}}, "required": ["x"]},
					{"type": "string"}
				]
			}
		},
		"required": ["outer", "items", "either"]
	}`, padded)
}

// Optional properties are wrapped even when they look nullable, because a
// sibling keyword can still exclude null; required ones are left alone.
func TestStrictSchemaWrapsEveryOptionalProperty(t *testing.T) {
	padded := strictSchemaJSON(t, `{
		"type": "object",
		"properties": {
			"size": {"type": ["string", "null"], "enum": ["small", "large"]},
			"variant": {"anyOf": [{"type": "string"}, {"type": "null"}]},
			"kept": {"anyOf": [{"type": "string"}, {"type": "null"}]}
		},
		"required": ["kept"]
	}`)

	assert.JSONEq(t, `{
		"type": "object",
		"properties": {
			"size": {"anyOf": [{"type": ["string", "null"], "enum": ["small", "large"]}, {"type": "null"}]},
			"variant": {"anyOf": [{"anyOf": [{"type": "string"}, {"type": "null"}]}, {"type": "null"}]},
			"kept": {"anyOf": [{"type": "string"}, {"type": "null"}]}
		},
		"required": ["size", "variant", "kept"]
	}`, padded)
}

func TestStrictSchemaTypesAdditionalProperties(t *testing.T) {
	padded := strictSchemaJSON(t, `{
		"type": "object",
		"properties": {
			"anything": {"type": "object", "additionalProperties": {}},
			"counts": {
				"type": "object",
				"additionalProperties": {"type": "object", "properties": {"note": {"type": "string"}}}
			}
		},
		"required": ["anything", "counts"]
	}`)

	// An any-value record has no type, which strict mode rejects, so it closes;
	// a typed record value is padded like any other object.
	assert.JSONEq(t, `{
		"type": "object",
		"properties": {
			"anything": {"type": "object", "additionalProperties": false},
			"counts": {
				"type": "object",
				"additionalProperties": {
					"type": "object",
					"properties": {"note": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
					"required": ["note"]
				}
			}
		},
		"required": ["anything", "counts"]
	}`, padded)
}

func TestStrictSchemaKeepsUnmodelledKeywordsAndPropertyOrder(t *testing.T) {
	padded := strictSchemaJSON(t, `{
		"type": "object",
		"properties": {
			"zeta": {"type": "string", "const": "fixed"},
			"alpha": {"type": "string", "format": "date-time"},
			"mid": {"type": "integer", "minimum": 1},
			"tags": {"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string"}}
		},
		"required": ["zeta"]
	}`)

	assert.JSONEq(t, `{
		"type": "object",
		"properties": {
			"zeta": {"type": "string", "const": "fixed"},
			"alpha": {"anyOf": [{"type": "string", "format": "date-time"}, {"type": "null"}]},
			"mid": {"anyOf": [{"type": "integer", "minimum": 1}, {"type": "null"}]},
			"tags": {"anyOf": [{"type": "array", "minItems": 1, "maxItems": 3, "items": {"type": "string"}}, {"type": "null"}]}
		},
		"required": ["zeta", "alpha", "mid", "tags"]
	}`, padded)
	assert.Less(t, strings.Index(padded, `"zeta"`), strings.Index(padded, `"alpha"`))
	assert.Less(t, strings.Index(padded, `"alpha"`), strings.Index(padded, `"mid"`))
}

// A tool without arguments keeps its empty properties object, which strict
// mode needs on every object schema.
func TestStrictSchemaKeepsEmptyProperties(t *testing.T) {
	const raw = `{"type":"object","properties":{},"additionalProperties":false}`
	schema := decodeValueSchema(t, raw)

	assert.Equal(t, raw, encodeValueSchema(t, &schema))
	assert.Equal(t, raw, strictSchemaJSON(t, raw))
}

// Objects typed as a list are padded; non-objects keep an any-value
// additionalProperties, as only object schemas need a closed one.
func TestStrictSchemaPadsTypeListObjectsAndLeavesNonObjects(t *testing.T) {
	padded := strictSchemaJSON(t, `{
		"type": "object",
		"properties": {
			"maybe": {"type": ["object", "null"], "properties": {"x": {"type": "string"}}},
			"loose": {"additionalProperties": {}}
		},
		"required": ["maybe", "loose"]
	}`)

	assert.JSONEq(t, `{
		"type": "object",
		"properties": {
			"maybe": {"type": ["object", "null"], "properties": {"x": {"anyOf": [{"type": "string"}, {"type": "null"}]}}, "required": ["x"]},
			"loose": {"additionalProperties": {}}
		},
		"required": ["maybe", "loose"]
	}`, padded)
}

// Callers that pad their own schemas must keep sending the same bytes.
func TestStrictSchemaIsIdempotent(t *testing.T) {
	once := strictSchemaJSON(t, `{
		"type": "object",
		"properties": {
			"taskTodoId": {"type": "string", "description": "Optional link to a todo"},
			"name": {"type": "string"},
			"options": {
				"type": "object",
				"properties": {"transparent": {"type": "boolean"}, "tags": {"type": "array", "items": {"type": "string"}}},
				"additionalProperties": false
			},
			"scope": {"type": "array", "items": {"type": "object", "properties": {"kind": {"type": "string", "enum": ["view", "widget"]}}}}
		},
		"required": ["name"],
		"additionalProperties": false
	}`)

	assert.Equal(t, once, strictSchemaJSON(t, once))
}

// Toolboxes and output schemas are shared across providers, so padding for
// one strict request must not change what another provider is sent. The
// schema is built in Go, so its properties hold ValueSchema values the way a
// reflected tool's do.
func TestStrictSchemaDoesNotMutateItsInput(t *testing.T) {
	inner := jsonmap.New()
	inner.Set("note", tools.ValueSchema{Type: "string"})
	properties := jsonmap.New()
	properties.Set("name", tools.ValueSchema{Type: "string"})
	properties.Set("details", tools.ValueSchema{Type: "object", Properties: inner, AdditionalProperties: false})
	schema := tools.ValueSchema{Type: "object", Properties: properties, Required: []string{"name"}}
	before := encodeValueSchema(t, &schema)

	tool := FunctionTool{Type: "function", Name: "task", Parameters: &schema, Strict: true}
	padded := schemaJSON(t, tool)

	assert.Equal(t, before, encodeValueSchema(t, &schema))
	assert.Contains(t, padded, `"required":["name","details"]`)
}

// The strict types encode their fields in the plain order, so a schema that
// was already strict produces the same bytes as before padding existed.
// The strict types pad their own encoding, so a schema that was already
// strict produces the same bytes as the plain type, including exact large
// integers in enums and constants.
func TestStrictTypesKeepTheirPlainEncoding(t *testing.T) {
	properties := jsonmap.New()
	properties.Set("id", tools.ValueSchema{Type: "integer", Enum: []any{int64(9007199254740993)}})
	properties.Set("name", map[string]any{"type": "string", "const": "task", "maxLength": json.Number("9007199254740993")})
	schema := tools.ValueSchema{Type: "object", Properties: properties, Required: []string{"id", "name"}, AdditionalProperties: false}

	tool := FunctionTool{Type: "function", Name: "task", Description: "Start a task", Parameters: &schema, Strict: true}
	type plainTool FunctionTool
	assert.Equal(t, schemaJSON(t, plainTool(tool)), schemaJSON(t, tool))
	assert.Contains(t, schemaJSON(t, tool), "9007199254740993")

	format := TextResponseFormat{Type: "json_schema", Name: "structured_output", Schema: &schema, Description: "Out", Strict: true}
	type plainFormat TextResponseFormat
	assert.Equal(t, schemaJSON(t, plainFormat(format)), schemaJSON(t, format))

	definition := jsonSchemaDefinition{Name: "structured_output", Schema: &schema, Strict: true}
	type plainDefinition jsonSchemaDefinition
	assert.Equal(t, schemaJSON(t, plainDefinition(definition)), schemaJSON(t, definition))
}

const optionalFieldSchema = `{
	"type": "object",
	"properties": {"name": {"type": "string"}, "todoId": {"type": "string"}},
	"required": ["name"],
	"additionalProperties": false
}`

const optionalFieldSchemaPadded = `{
	"type": "object",
	"properties": {"name": {"type": "string"}, "todoId": {"anyOf": [{"type": "string"}, {"type": "null"}]}},
	"required": ["name", "todoId"],
	"additionalProperties": false
}`

func optionalFieldToolbox(t *testing.T) *tools.Toolbox {
	t.Helper()
	schema := tools.FunctionSchema{Name: "task", Description: "Start a task", Parameters: decodeValueSchema(t, optionalFieldSchema)}
	return tools.Box(tools.External("Task", &schema, func(r tools.Runner, params json.RawMessage) tools.Result {
		return tools.SuccessFromString("ok")
	}))
}

// payloadField returns a payload value re-decoded as generic JSON, so tests
// compare what goes on the wire.
func payloadField(t *testing.T, value any) any {
	t.Helper()
	raw, err := json.Marshal(value)
	require.NoError(t, err)
	var decoded any
	require.NoError(t, json.Unmarshal(raw, &decoded))
	return decoded
}

func schemaJSON(t *testing.T, value any) string {
	t.Helper()
	raw, err := json.Marshal(value)
	require.NoError(t, err)
	return string(raw)
}

func TestResponsesPayloadPadsStrictToolsAndJSONOutput(t *testing.T) {
	output := decodeValueSchema(t, optionalFieldSchema)
	api := NewResponsesAPI("", "gpt-5")

	payload, err := api.buildResponsesPayload(nil, "", optionalFieldToolbox(t), &output)
	require.NoError(t, err)

	toolsArr := payloadField(t, payload["tools"]).([]any)
	require.Len(t, toolsArr, 1)
	tool := toolsArr[0].(map[string]any)
	assert.Equal(t, true, tool["strict"])
	assert.JSONEq(t, optionalFieldSchemaPadded, schemaJSON(t, tool["parameters"]))

	format := payloadField(t, payload["text"]).(map[string]any)["format"].(map[string]any)
	assert.Equal(t, true, format["strict"])
	assert.JSONEq(t, optionalFieldSchemaPadded, schemaJSON(t, format["schema"]))
}

// Chat Completions function tools are not sent strict, so they go out as the
// caller wrote them. This is the path Claude takes through OpenRouter, and it
// must see optional fields as optional.
func TestChatCompletionsPayloadSendsToolsAsWrittenAndPadsStrictJSONOutput(t *testing.T) {
	output := decodeValueSchema(t, optionalFieldSchema)
	m := NewChatCompletionsAPI("", "anthropic/claude-sonnet-5.5")

	payload, err := m.BuildPayload(nil, nil, optionalFieldToolbox(t), &output)
	require.NoError(t, err)

	toolsArr := payloadField(t, payload["tools"]).([]any)
	require.Len(t, toolsArr, 1)
	function := toolsArr[0].(map[string]any)["function"].(map[string]any)
	assert.NotContains(t, function, "strict")
	assert.JSONEq(t, optionalFieldSchema, schemaJSON(t, function["parameters"]))

	jsonSchema := payloadField(t, payload["response_format"]).(map[string]any)["json_schema"].(map[string]any)
	assert.Equal(t, true, jsonSchema["strict"])
	assert.JSONEq(t, optionalFieldSchemaPadded, schemaJSON(t, jsonSchema["schema"]))
}

// A strict FunctionTool supplied directly is padded too: strict never reaches
// the wire with an unpadded schema, whoever built the tool.
func TestResponsesPayloadPadsStrictToolsSuppliedDirectly(t *testing.T) {
	schema := decodeValueSchema(t, optionalFieldSchema)
	api := NewResponsesAPI("", "gpt-5").WithTool(FunctionTool{Type: "function", Name: "task", Parameters: &schema, Strict: true})

	payload, err := api.buildResponsesPayload(nil, "", nil, nil)
	require.NoError(t, err)

	toolsArr := payloadField(t, payload["tools"]).([]any)
	require.Len(t, toolsArr, 1)
	assert.JSONEq(t, optionalFieldSchemaPadded, schemaJSON(t, toolsArr[0].(map[string]any)["parameters"]))
}

type reflectedTaskParams struct {
	Name    string  `json:"name"`
	TodoID  *string `json:"todoId,omitempty"`
	Options *struct {
		Tags []string `json:"tags,omitempty"`
	} `json:"options,omitempty"`
}

// A reflected tool accepts the null that strict padding lets the model send
// for an optional field, at any depth, and still rejects null for a required
// one. The advertised schema and the tool's own validation must agree.
func TestStrictReflectedToolRunsWithNullOptionalFields(t *testing.T) {
	var got reflectedTaskParams
	tool := tools.Func("Task", "Start a task", "task", func(r tools.Runner, p reflectedTaskParams) tools.Result {
		got = p
		return tools.SuccessFromString("ok")
	})
	api := NewResponsesAPI("", "gpt-5")
	payload, err := api.buildResponsesPayload(nil, "", tools.Box(tool), nil)
	require.NoError(t, err)
	parameters := schemaJSON(t, payloadField(t, payload["tools"]).([]any)[0].(map[string]any)["parameters"])
	assert.Contains(t, parameters, `"required":["name","todoId","options"]`)
	assert.Contains(t, parameters, `"required":["tags"]`)

	result := tool.Run(nil, json.RawMessage(`{"name": "Menu", "todoId": null, "options": {"tags": null}}`))
	require.NoError(t, result.Error())
	assert.Equal(t, "Menu", got.Name)
	assert.Nil(t, got.TodoID)
	require.NotNil(t, got.Options)
	assert.Nil(t, got.Options.Tags)

	result = tool.Run(nil, json.RawMessage(`{"name": null}`))
	assert.Error(t, result.Error())
}

// WithStrictTools sends Chat Completions function tools strict with padded
// parameters, for endpoints that serve OpenAI models; custom tools are left as
// declared.
func TestChatCompletionsPayloadWithStrictToolsPadsFunctionTools(t *testing.T) {
	functionSchema := tools.FunctionSchema{Name: "task", Description: "Start a task", Parameters: decodeValueSchema(t, optionalFieldSchema)}
	toolbox := tools.Box(
		tools.External("Task", &functionSchema, func(r tools.Runner, params json.RawMessage) tools.Result {
			return tools.SuccessFromString("ok")
		}),
		tools.FuncGrammar(tools.Text(), "Note", "Write a note", "note", func(r tools.Runner, input string) tools.Result {
			return tools.SuccessFromString("ok")
		}),
	)
	m := NewChatCompletionsAPI("", "openai/gpt-6-luna").WithStrictTools().WithFlatCustomTools()

	payload, err := m.BuildPayload(nil, nil, toolbox, nil)
	require.NoError(t, err)

	toolsArr := payloadField(t, payload["tools"]).([]any)
	require.Len(t, toolsArr, 2)
	function := toolsArr[0].(map[string]any)["function"].(map[string]any)
	assert.Equal(t, true, function["strict"])
	assert.JSONEq(t, optionalFieldSchemaPadded, schemaJSON(t, function["parameters"]))
	assert.Equal(t, map[string]any{"type": "custom", "name": "note", "description": "Write a note", "format": map[string]any{"type": "text"}}, toolsArr[1])
}
