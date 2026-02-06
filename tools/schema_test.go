package tools

import (
	"encoding/json"
	"math/rand"
	"reflect"
	"strings"
	"testing"

	"github.com/metalim/jsonmap"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestValueSchema_AnyOf tests that ValueSchema correctly handles the anyOf keyword
// during JSON marshalling and unmarshalling.
func TestValueSchema_AnyOf(t *testing.T) {
	// Simplified schema structure mimicking the user's input for the 'value' field
	inputJSON := `{
		"anyOf": [
			{ "type": "string" },
			{ "type": "number" },
			{ "type": "boolean" },
			{ "type": "null" }
		]
	}`

	var schema ValueSchema
	err := json.Unmarshal([]byte(inputJSON), &schema)
	require.NoError(t, err, "Failed to unmarshal input JSON into ValueSchema")

	// Check that the AnyOf field was populated correctly
	require.NotNil(t, schema.AnyOf, "Schema AnyOf field should not be nil after unmarshal")
	require.Len(t, schema.AnyOf, 4, "Schema AnyOf field should have 4 items")
	assert.Equal(t, "string", schema.AnyOf[0].Type, "First anyOf type should be string")
	assert.Equal(t, "number", schema.AnyOf[1].Type, "Second anyOf type should be number")
	assert.Equal(t, "boolean", schema.AnyOf[2].Type, "Third anyOf type should be boolean")
	assert.Equal(t, "null", schema.AnyOf[3].Type, "Fourth anyOf type should be null")

	// Marshal the schema back to JSON
	outputJSON, err := json.Marshal(schema)
	require.NoError(t, err, "Failed to marshal ValueSchema back to JSON")

	// Assert that the output JSON is equivalent to the input JSON
	// Note: JSONEq handles potential key order differences and whitespace
	assert.JSONEq(t, inputJSON, string(outputJSON), "Output JSON should match input JSON for anyOf structure")
}

// TestGenerateSchema checks that the JSON schema is generated correctly from the Params struct.
// Moved from tool_test.go
func TestGenerateSchema(t *testing.T) {
	// Define a local struct equivalent to Params for testing schema generation
	type testParams struct {
		Name    string `json:"name"`
		Age     int    `json:"age"`
		Email   string `json:"email,omitempty"`
		IsAdmin bool   `json:"isAdmin"`
	}
	typ := reflect.TypeOf(testParams{})
	schema := generateSchema("TestFunction", "Test function description", typ)

	expectedSchema := map[string]any{
		"name":        "TestFunction",
		"description": "Test function description",
		"parameters": map[string]any{
			"type": "object",
			"properties": map[string]any{
				"name":    map[string]any{"type": "string"},
				"age":     map[string]any{"type": "integer"},
				"email":   map[string]any{"type": "string"}, // Email is optional
				"isAdmin": map[string]any{"type": "boolean"},
			},
			"required":             []string{"name", "age", "isAdmin"}, // Email is not required due to omitempty
			"additionalProperties": false,
		},
	}

	schemaJSON, err := json.Marshal(schema)
	require.NoError(t, err, "Failed to marshal generated schema")

	expectedSchemaJSON, err := json.Marshal(expectedSchema)
	require.NoError(t, err, "Failed to marshal expected schema")

	var schemaMap, expectedSchemaMap map[string]any
	err = json.Unmarshal(schemaJSON, &schemaMap)
	require.NoError(t, err, "Failed to unmarshal generated schema")
	err = json.Unmarshal(expectedSchemaJSON, &expectedSchemaMap)
	require.NoError(t, err, "Failed to unmarshal expected schema")

	assert.Equal(t, expectedSchemaMap, schemaMap, "Generated schema does not match expected schema")
}

// TestGenerateSchema_AdvancedTypes tests schema generation for more complex types and tags.
func TestGenerateSchema_AdvancedTypes(t *testing.T) {
	type Nested struct {
		NestedField string `json:"nested_field" description:"A nested field."`
	}
	type advancedParams struct {
		StringSlice []string          `json:"string_slice"`
		IntPtr      *int              `json:"int_ptr,omitempty"`
		StructMap   map[string]Nested `json:"struct_map"`
		Ignored     string            `json:"-"` // Should be ignored
		unexported  string            // Should be ignored (lowercase)
		Renamed     bool              `json:"renamed_field"`
	}
	// Avoid a lint warning for the unexported field.
	_ = advancedParams{unexported: ""}

	typ := reflect.TypeOf(advancedParams{})
	schema := generateSchema("AdvancedTest", "Testing advanced types", typ)

	// Basic structural checks
	require.NotNil(t, schema.Parameters.Properties, "Properties map should not be nil")
	props := schema.Parameters.Properties

	getProp := func(key string) (ValueSchema, bool) {
		raw, ok := props.Get(key)
		if !ok {
			return ValueSchema{}, false
		}
		val, ok := raw.(ValueSchema)
		require.True(t, ok, "Property %q should be ValueSchema, got %T", key, raw)
		return val, true
	}

	// Check specific field types
	stringSlice, ok := getProp("string_slice")
	require.True(t, ok, "string_slice should exist")
	assert.Equal(t, "array", stringSlice.Type, "string_slice type mismatch")
	require.NotNil(t, stringSlice.Items, "string_slice items should not be nil")
	assert.Equal(t, "string", stringSlice.Items.Type, "string_slice item type mismatch")

	intPtr, ok := getProp("int_ptr")
	require.True(t, ok, "int_ptr should exist")
	assert.Equal(t, "integer", intPtr.Type, "int_ptr type mismatch") // Note: Ptr unwraps

	structMap, ok := getProp("struct_map")
	require.True(t, ok, "struct_map should exist")
	assert.Equal(t, "object", structMap.Type, "struct_map type mismatch")
	require.NotNil(t, structMap.AdditionalProperties, "struct_map additionalProperties should not be nil")
	// Type assert AdditionalProperties to access its fields
	apSchema, ok := structMap.AdditionalProperties.(ValueSchema)
	require.True(t, ok, "AdditionalProperties should be of type ValueSchema")
	assert.Equal(t, "object", apSchema.Type, "struct_map value type mismatch")
	require.NotNil(t, apSchema.Properties, "struct_map value properties should not be nil")
	nestedRaw, ok := apSchema.Properties.Get("nested_field")
	require.True(t, ok, "nested_field should exist")
	nested, ok := nestedRaw.(ValueSchema)
	require.True(t, ok, "nested_field should be ValueSchema")
	assert.Equal(t, "string", nested.Type, "nested_field type mismatch")
	assert.Equal(t, "A nested field.", nested.Description, "nested_field description mismatch")
	assert.Equal(t, false, apSchema.AdditionalProperties, "Nested struct should also have AdditionalProperties false")

	// Check ignored/unexported fields are absent
	_, ignoredExists := getProp("Ignored")
	assert.False(t, ignoredExists, "Ignored field should not be in schema")
	_, ignoredJsonExists := getProp("-")
	assert.False(t, ignoredJsonExists, "json: '-' field should not be in schema")
	_, unexportedExists := getProp("unexported") // Check lowercase name
	assert.False(t, unexportedExists, "Unexported field should not be in schema")

	// Check renamed field
	_, originalRenamedExists := getProp("Renamed")
	assert.False(t, originalRenamedExists, "Original name of renamed field should not exist")
	renamedSchema, renamedExists := getProp("renamed_field")
	assert.True(t, renamedExists, "Renamed field should exist")
	assert.Equal(t, "boolean", renamedSchema.Type, "Renamed field type mismatch")

	// Check required fields (only non-omitempty and non-ignored)
	expectedRequired := []string{"string_slice", "struct_map", "renamed_field"}
	assert.ElementsMatch(t, expectedRequired, schema.Parameters.Required, "Required fields mismatch")

	// Check that AdditionalProperties is set to false for struct-generated schemas
	assert.Equal(t, false, schema.Parameters.AdditionalProperties, "AdditionalProperties should be false for struct schemas")
}

func TestValueSchema_OrderPreservedOnMarshal(t *testing.T) {
	r := rand.New(rand.NewSource(42))

	randomKey := func(n int) string {
		const letters = "abcdefghijklmnopqrstuvwxyz"
		var b strings.Builder
		for i := 0; i < n; i++ {
			b.WriteByte(letters[r.Intn(len(letters))])
		}
		return b.String()
	}

	// Build many random schemas and assert marshal output matches the original bytes.
	for i := 0; i < 200; i++ {
		propCount := 5 + r.Intn(10)
		names := make([]string, propCount)
		for j := range names {
			names[j] = randomKey(6 + r.Intn(6))
		}

		var b strings.Builder
		b.WriteString(`{"type":"object","properties":{`)
		for j, n := range names {
			if j > 0 {
				b.WriteByte(',')
			}
			b.WriteString(`"`)
			b.WriteString(n)
			b.WriteString(`":{"type":"string"}`)
		}
		b.WriteString(`},"required":[`)
		for j, n := range names {
			if j > 0 {
				b.WriteByte(',')
			}
			b.WriteString(`"`)
			b.WriteString(n)
			b.WriteString(`"`)
		}
		b.WriteString(`]}`)
		orig := b.String()

		var schema ValueSchema
		require.NoError(t, json.Unmarshal([]byte(orig), &schema))
		out, err := json.Marshal(schema)
		require.NoError(t, err)
		assert.Equal(t, orig, string(out), "schema order changed on case %d", i)
	}
}

func TestValueSchema_AdditionalPropertiesOrderPreservedOnMarshal(t *testing.T) {
	orig := `{"type":"object","additionalProperties":{"type":"object","properties":{"reasoning":{"type":"string"},"answer":{"type":"string"},"confidence":{"type":"number"}}}}`

	var schema ValueSchema
	require.NoError(t, json.Unmarshal([]byte(orig), &schema))

	out, err := json.Marshal(schema)
	require.NoError(t, err)
	body := string(out)

	posReasoning := strings.Index(body, `"reasoning"`)
	posAnswer := strings.Index(body, `"answer"`)
	posConfidence := strings.Index(body, `"confidence"`)
	require.Greater(t, posReasoning, -1)
	require.Greater(t, posAnswer, -1)
	require.Greater(t, posConfidence, -1)
	assert.True(t, posReasoning < posAnswer && posAnswer < posConfidence, "additionalProperties field order changed")

	apMap, ok := schema.AdditionalProperties.(*jsonmap.Map)
	require.True(t, ok, "additionalProperties should decode to ordered map")
	rawProps, ok := apMap.Get("properties")
	require.True(t, ok)
	props, ok := rawProps.(*jsonmap.Map)
	require.True(t, ok)
	assert.Equal(t, []string{"reasoning", "answer", "confidence"}, props.Keys())
}

func TestValidateJSON_AdditionalPropertiesOrderedMapSchema(t *testing.T) {
	props := jsonmap.New()
	props.Set("name", ValueSchema{Type: "string"})

	additional := jsonmap.New()
	additional.Set("type", "string")

	schema := FunctionSchema{
		Name: "OrderedAdditionalProperties",
		Parameters: ValueSchema{
			Type:                 "object",
			Properties:           props,
			Required:             []string{"name"},
			AdditionalProperties: additional,
		},
	}

	err := validateJSON(&schema, json.RawMessage(`{"name":"Alice","extra":"ok"}`))
	assert.NoError(t, err)

	err = validateJSON(&schema, json.RawMessage(`{"name":"Alice","extra":123}`))
	require.Error(t, err)
	assert.Contains(t, err.Error(), `additional property "extra": type mismatch: expected string`)
}

// TestValidateJSON tests the validateJSON function with various scenarios.
func TestValidateJSON(t *testing.T) {
	// Use the schema generated from TestGenerateSchema's local testParams
	type testParams struct {
		Name    string `json:"name"`
		Age     int    `json:"age"`
		Email   string `json:"email,omitempty"`
		IsAdmin bool   `json:"isAdmin"`
	}
	funcSchema := generateSchema("TestFunction", "Desc", reflect.TypeOf(testParams{}))

	tests := []struct {
		name          string
		jsonData      string
		expectError   bool
		errorContains string // Substring to check for in error message
	}{
		{
			name:        "Valid data",
			jsonData:    `{"name":"Alice", "age":30, "isAdmin":true, "email":"alice@example.com"}`,
			expectError: false,
		},
		{
			name:        "Valid data missing optional field",
			jsonData:    `{"name":"Bob", "age":25, "isAdmin":false}`,
			expectError: false,
		},
		{
			name:          "Missing required field age",
			jsonData:      `{"name":"Charlie", "isAdmin":true}`,
			expectError:   true,
			errorContains: "missing required field: \"age\"",
		},
		{
			name:          "Missing required field isAdmin",
			jsonData:      `{"name":"David", "age":40}`,
			expectError:   true,
			errorContains: "missing required field: \"isAdmin\"",
		},
		{
			name:          "Invalid type for age (string)",
			jsonData:      `{"name":"Eve", "age":"thirty", "isAdmin":false}`,
			expectError:   true,
			errorContains: "field \"age\": type mismatch: expected integer",
		},
		{
			name:          "Invalid type for isAdmin (number)",
			jsonData:      `{"name":"Frank", "age":50, "isAdmin":1}`,
			expectError:   true,
			errorContains: "field \"isAdmin\": type mismatch: expected boolean",
		},
		{
			name:          "Invalid JSON format",
			jsonData:      `{"name":"Grace", "age":60, "isAdmin":true`, // Missing closing brace
			expectError:   true,
			errorContains: "invalid JSON format",
		},
		{
			name:          "Extra field (not allowed for structs)",
			jsonData:      `{"name":"Heidi", "age":35, "isAdmin":true, "extra":"field"}`,
			expectError:   true,
			errorContains: "additional property \"extra\" not allowed",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rawJSON := json.RawMessage(tt.jsonData)
			err := validateJSON(&funcSchema, rawJSON)
			if tt.expectError {
				assert.Error(t, err)
				if tt.errorContains != "" {
					assert.Contains(t, err.Error(), tt.errorContains)
				}
			} else {
				assert.NoError(t, err)
			}
		})
	}

	// Add specific tests for AdditionalProperties variations
	t.Run("AdditionalProperties False - Extra Field", func(t *testing.T) {
		schema := funcSchema // Use the base schema
		schema.Parameters.AdditionalProperties = false
		jsonData := json.RawMessage(`{"name":"Alice", "age":30, "isAdmin":true, "extra":"field"}`)
		err := validateJSON(&schema, jsonData)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "additional property \"extra\" not allowed")
		schema.Parameters.AdditionalProperties = nil // Reset for other tests
	})

	t.Run("AdditionalProperties True - Extra Field", func(t *testing.T) {
		schema := funcSchema // Use the base schema
		schema.Parameters.AdditionalProperties = true
		jsonData := json.RawMessage(`{"name":"Alice", "age":30, "isAdmin":true, "extra":"field"}`)
		err := validateJSON(&schema, jsonData)
		assert.NoError(t, err)
		schema.Parameters.AdditionalProperties = nil // Reset for other tests
	})

	t.Run("AdditionalProperties Schema - Valid Extra Field", func(t *testing.T) {
		schema := funcSchema // Use the base schema
		extraSchema := ValueSchema{Type: "string"}
		schema.Parameters.AdditionalProperties = extraSchema
		jsonData := json.RawMessage(`{"name":"Alice", "age":30, "isAdmin":true, "extra":"field"}`)
		err := validateJSON(&schema, jsonData)
		assert.NoError(t, err)
		schema.Parameters.AdditionalProperties = nil // Reset for other tests
	})

	t.Run("AdditionalProperties Schema - Invalid Extra Field Type", func(t *testing.T) {
		schema := funcSchema // Use the base schema
		extraSchema := ValueSchema{Type: "string"}
		schema.Parameters.AdditionalProperties = extraSchema
		jsonData := json.RawMessage(`{"name":"Alice", "age":30, "isAdmin":true, "extra":123}`)
		err := validateJSON(&schema, jsonData)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "additional property \"extra\": type mismatch: expected string")
		schema.Parameters.AdditionalProperties = nil // Reset for other tests
	})

	t.Run("AnyOf Validation - String Matches", func(t *testing.T) {
		props := jsonmap.New()
		props.Set("value", ValueSchema{
			AnyOf: []ValueSchema{
				{Type: "string"},
				{Type: "integer"},
			},
		})
		schema := FunctionSchema{
			Name: "AnyOfTest",
			Parameters: ValueSchema{
				Type:       "object",
				Properties: props,
				Required:   []string{"value"},
			},
		}
		jsonData := json.RawMessage(`{"value": "hello"}`)
		err := validateJSON(&schema, jsonData)
		assert.NoError(t, err)
	})

	t.Run("AnyOf Validation - Integer Matches", func(t *testing.T) {
		props := jsonmap.New()
		props.Set("value", ValueSchema{
			AnyOf: []ValueSchema{
				{Type: "string"},
				{Type: "integer"},
			},
		})
		schema := FunctionSchema{
			Name: "AnyOfTest",
			Parameters: ValueSchema{
				Type:       "object",
				Properties: props,
				Required:   []string{"value"},
			},
		}
		jsonData := json.RawMessage(`{"value": 123}`)
		err := validateJSON(&schema, jsonData)
		assert.NoError(t, err)
	})

	t.Run("AnyOf Validation - Boolean Fails (No Match)", func(t *testing.T) {
		props := jsonmap.New()
		props.Set("value", ValueSchema{
			AnyOf: []ValueSchema{
				{Type: "string"},
				{Type: "integer"},
			},
		})
		schema := FunctionSchema{
			Name: "AnyOfTest",
			Parameters: ValueSchema{
				Type:       "object",
				Properties: props,
				Required:   []string{"value"},
			},
		}
		jsonData := json.RawMessage(`{"value": true}`)
		err := validateJSON(&schema, jsonData)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "field \"value\": data does not match any of the schemas in anyOf")
	})

	t.Run("Invalid Schema - Not Object Type", func(t *testing.T) {
		invalidSchema := FunctionSchema{
			Name: "InvalidFunc",
			Parameters: ValueSchema{
				Type: "string", // Invalid: Should be object
			},
		}
		dummyJSON := json.RawMessage(`{}`)
		err := validateJSON(&invalidSchema, dummyJSON)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "schema error: received an invalid object schema")
	})

	t.Run("Invalid Schema - Nil Properties", func(t *testing.T) {
		invalidSchema := FunctionSchema{
			Name: "InvalidFunc",
			Parameters: ValueSchema{
				Type:       "object",
				Properties: nil, // Invalid: Should be non-nil map pointer
			},
		}
		dummyJSON := json.RawMessage(`{}`)
		err := validateJSON(&invalidSchema, dummyJSON)
		assert.Error(t, err)
		assert.Contains(t, err.Error(), "schema error: received an invalid object schema")
	})
}
