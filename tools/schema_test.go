package tools

import (
	"encoding/json"
	"reflect"
	"testing"

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
			"required": []string{"name", "age", "isAdmin"}, // Email is not required due to omitempty
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
	props := *schema.Parameters.Properties

	// Check specific field types
	assert.Equal(t, "array", props["string_slice"].Type, "string_slice type mismatch")
	require.NotNil(t, props["string_slice"].Items, "string_slice items should not be nil")
	assert.Equal(t, "string", props["string_slice"].Items.Type, "string_slice item type mismatch")

	assert.Equal(t, "integer", props["int_ptr"].Type, "int_ptr type mismatch") // Note: Ptr unwraps

	assert.Equal(t, "object", props["struct_map"].Type, "struct_map type mismatch")
	require.NotNil(t, props["struct_map"].AdditionalProperties, "struct_map additionalProperties should not be nil")
	assert.Equal(t, "object", props["struct_map"].AdditionalProperties.Type, "struct_map value type mismatch")
	require.NotNil(t, props["struct_map"].AdditionalProperties.Properties, "struct_map value properties mismatch")
	nestedProps := *props["struct_map"].AdditionalProperties.Properties
	assert.Equal(t, "string", nestedProps["nested_field"].Type, "nested_field type mismatch")
	assert.Equal(t, "A nested field.", nestedProps["nested_field"].Description, "nested_field description mismatch")

	// Check ignored/unexported fields are absent
	_, ignoredExists := props["Ignored"]
	assert.False(t, ignoredExists, "Ignored field should not be in schema")
	_, ignoredJsonExists := props["-"]
	assert.False(t, ignoredJsonExists, "json: '-' field should not be in schema")
	_, unexportedExists := props["unexported"] // Check lowercase name
	assert.False(t, unexportedExists, "Unexported field should not be in schema")

	// Check renamed field
	_, originalRenamedExists := props["Renamed"]
	assert.False(t, originalRenamedExists, "Original name of renamed field should not exist")
	renamedSchema, renamedExists := props["renamed_field"]
	assert.True(t, renamedExists, "Renamed field should exist")
	assert.Equal(t, "boolean", renamedSchema.Type, "Renamed field type mismatch")

	// Check required fields (only non-omitempty and non-ignored)
	expectedRequired := []string{"string_slice", "struct_map", "renamed_field"}
	assert.ElementsMatch(t, expectedRequired, schema.Parameters.Required, "Required fields mismatch")
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
			errorContains: "missing required field: age",
		},
		{
			name:          "Missing required field isAdmin",
			jsonData:      `{"name":"David", "age":40}`,
			expectError:   true,
			errorContains: "missing required field: isAdmin",
		},
		{
			name:          "Invalid type for age (string)",
			jsonData:      `{"name":"Eve", "age":"thirty", "isAdmin":false}`,
			expectError:   true,
			errorContains: "age: type mismatch: expected integer",
		},
		{
			name:          "Invalid type for isAdmin (number)",
			jsonData:      `{"name":"Frank", "age":50, "isAdmin":1}`,
			expectError:   true,
			errorContains: "isAdmin: type mismatch: expected boolean",
		},
		{
			name:          "Invalid JSON format",
			jsonData:      `{"name":"Grace", "age":60, "isAdmin":true`, // Missing closing brace
			expectError:   true,
			errorContains: "invalid JSON format",
		},
		{
			name:        "Extra field (should be ignored)",
			jsonData:    `{"name":"Heidi", "age":35, "isAdmin":true, "extra":"field"}`,
			expectError: false,
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

	// Add specific tests for invalid schema structure fed into validateParameters
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
