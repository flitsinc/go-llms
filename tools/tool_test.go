package tools

import (
	"context"
	"encoding/json"
	"reflect"
	"strings"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// Params defines a struct type with various fields for testing tool functionality.
type Params struct {
	Name    string `json:"name"`
	Age     int    `json:"age"`
	Email   string `json:"email,omitempty"` // Optional field
	IsAdmin bool   `json:"isAdmin"`
}

// Helper to extract JSON data from result content for testing
func extractJSONFromResult(t *testing.T, r Result) json.RawMessage {
	t.Helper()
	require.NotNil(t, r.Content(), "Result content should not be nil")
	require.NotEmpty(t, r.Content(), "Result content should not be empty")
	jsonItem, ok := r.Content()[0].(*content.JSON)
	require.True(t, ok, "First content item should be JSON")
	return jsonItem.Data
}

// TestGenerateSchema checks that the JSON schema is generated correctly from the Params struct.
func TestGenerateSchema(t *testing.T) {
	typ := reflect.TypeOf(Params{})
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

// TestToolRun_CorrectData verifies that the tool functions correctly with valid input data.
func TestToolRun_CorrectData(t *testing.T) {
	testFunc := func(r Runner, p Params) Result {
		return Success(map[string]any{
			"name":    p.Name,
			"age":     p.Age,
			"email":   p.Email,
			"isAdmin": p.IsAdmin,
		})
	}
	tool := Func("Test Tool", "Test function for Params", "test_tool", testFunc)

	params := json.RawMessage(`{"name":"Bob", "age":30, "email":"bob@example.com", "isAdmin":false}`)
	result := tool.Run(&runner{}, params)

	require.NoError(t, result.Error(), "Expected no error")
	resultJSON := extractJSONFromResult(t, result)
	assert.JSONEq(t, `{"name":"Bob","age":30,"email":"bob@example.com","isAdmin":false}`, string(resultJSON))
}

// TestToolRun_OptionalFieldAbsent verifies that the tool handles the absence of optional fields correctly.
func TestToolRun_OptionalFieldAbsent(t *testing.T) {
	testFunc := func(r Runner, p Params) Result {
		return Success(map[string]any{
			"name":    p.Name,
			"age":     p.Age,
			"email":   p.Email,
			"isAdmin": p.IsAdmin,
		})
	}
	tool := Func("Test Tool", "Test function for Params", "test_tool", testFunc)

	params := json.RawMessage(`{"name":"Alice", "age":28, "isAdmin":true}`)
	result := tool.Run(&runner{}, params)

	require.NoError(t, result.Error(), "Expected no error")
	resultJSON := extractJSONFromResult(t, result)
	assert.JSONEq(t, `{"name":"Alice","age":28,"email":"","isAdmin":true}`, string(resultJSON))
}

// TestToolRun_MissingRequiredField verifies that the tool correctly handles missing required fields.
func TestToolRun_MissingRequiredField(t *testing.T) {
	testFunc := func(r Runner, p Params) Result {
		// This part of the function shouldn't be reached if validation works
		return Success(map[string]any{})
	}
	tool := Func("Test Tool", "Test function for Params", "test_tool", testFunc)

	params := json.RawMessage(`{"name":"John"}`) // Missing 'age' and 'isAdmin', which are required
	result := tool.Run(&runner{}, params)

	assert.Error(t, result.Error(), "Expected an error for missing required fields")
	assert.Contains(t, result.Error().Error(), "missing required field", "Error should mention missing required field")
	// Check the content of the error result
	resultJSON := extractJSONFromResult(t, result)
	assert.Contains(t, string(resultJSON), "missing required field")
}

// TestToolRun_InvalidDataType checks that the tool correctly identifies incorrect data types in input.
func TestToolRun_InvalidDataType(t *testing.T) {
	testFunc := func(r Runner, p Params) Result {
		// This part shouldn't be reached
		return Success(map[string]any{})
	}
	tool := Func("Test Tool", "Test function for Params", "test_tool", testFunc)

	// Invalid data type for 'isAdmin', expecting a boolean but providing a string
	params := json.RawMessage(`{"name":"Alice", "age":28, "isAdmin":"yes"}`)
	result := tool.Run(&runner{}, params)

	assert.Error(t, result.Error(), "Expected a type mismatch error")
	assert.Contains(t, result.Error().Error(), "type mismatch", "Error should mention type mismatch")
	// Check the content of the error result
	resultJSON := extractJSONFromResult(t, result)
	assert.Contains(t, string(resultJSON), "type mismatch")
}

// TestToolRun_UnexpectedFields verifies that the tool ignores fields that are not defined in the schema.
func TestToolRun_UnexpectedFields(t *testing.T) {
	testFunc := func(r Runner, p Params) Result {
		return Success(map[string]any{
			"name":    p.Name,
			"age":     p.Age,
			"email":   p.Email,
			"isAdmin": p.IsAdmin,
		})
	}
	tool := Func("Test Tool", "Test function for Params", "test_tool", testFunc)

	// Including an unexpected 'location' field
	params := json.RawMessage(`{"name":"Alice", "age":28, "isAdmin":true, "location":"unknown"}`)
	result := tool.Run(&runner{}, params)

	require.NoError(t, result.Error(), "Expected no error for unexpected field")
	resultJSON := extractJSONFromResult(t, result)
	assert.JSONEq(t, `{"name":"Alice","age":28,"email":"","isAdmin":true}`, string(resultJSON))
}

type AdvancedParams struct {
	ID       int      `json:"id"`
	Features []string `json:"features"`
	Profile  struct {
		Username string `json:"username"`
		Active   bool   `json:"active"`
	} `json:"profile"`
}

// TestValidateJSONWithArrayAndObject tests validation of both array and nested object fields.
func TestValidateJSONWithArrayAndObject(t *testing.T) {
	testFunc := func(r Runner, p AdvancedParams) Result {
		return Success(map[string]any{
			"id":       p.ID,
			"features": p.Features,
			"profile":  p.Profile,
		})
	}
	tool := Func("Advanced Tool", "Test function for Advanced Params", "advanced_tool", testFunc)

	t.Run("Valid Input", func(t *testing.T) {
		validParams := json.RawMessage(`{"id":101, "features":["fast", "reliable", "secure"], "profile":{"username":"user01", "active":true}}`)
		result := tool.Run(&runner{}, validParams)

		require.NoError(t, result.Error(), "Expected no error")
		resultJSON := extractJSONFromResult(t, result)
		assert.JSONEq(t, `{"id":101,"features":["fast","reliable","secure"],"profile":{"username":"user01","active":true}}`, string(resultJSON))
	})

	t.Run("Invalid Input", func(t *testing.T) {
		invalidParams := json.RawMessage(`{"id":101, "features":"fast", "profile":{"username":123, "active":"yes"}}`)
		result := tool.Run(&runner{}, invalidParams)

		assert.Error(t, result.Error(), "Expected a type mismatch or validation error")
		assert.True(t, strings.Contains(result.Error().Error(), "type mismatch") || strings.Contains(result.Error().Error(), "validation error"))
		// Check the content of the error result
		resultJSON := extractJSONFromResult(t, result)
		assert.Contains(t, string(resultJSON), "type mismatch")
	})
}

func TestToolFunctionErrorHandling(t *testing.T) {
	testFunc := func(r Runner, p AdvancedParams) Result {
		if p.ID == 0 {
			return Errorf("ID cannot be zero")
		}
		return Success(map[string]any{
			"id":       p.ID,
			"features": p.Features,
			"profile":  p.Profile,
		})
	}
	tool := Func("Error Handling Tool", "Test function for error handling in Params", "error_handling_tool", testFunc)

	t.Run("Error Case", func(t *testing.T) {
		errorParams := json.RawMessage(`{"id":0, "features":["fast", "reliable"], "profile":{"username":"user01", "active":true}}`)
		result := tool.Run(&runner{}, errorParams)

		assert.Error(t, result.Error(), "Expected error 'ID cannot be zero'")
		assert.Contains(t, result.Error().Error(), "ID cannot be zero")
		// Check the content of the error result
		resultJSON := extractJSONFromResult(t, result)
		assert.JSONEq(t, `{"error":"ID cannot be zero"}`, string(resultJSON))
	})

	t.Run("Valid Case", func(t *testing.T) {
		validParams := json.RawMessage(`{"id":101, "features":["fast", "reliable"], "profile":{"username":"user01", "active":true}}`)
		result := tool.Run(&runner{}, validParams)

		require.NoError(t, result.Error(), "Expected no error")
		resultJSON := extractJSONFromResult(t, result)
		assert.JSONEq(t, `{"id":101,"features":["fast","reliable"],"profile":{"username":"user01","active":true}}`, string(resultJSON))
	})
}

func TestToolFunctionReport(t *testing.T) {
	reportCalled := false
	runner := &runner{
		report: func(status string) {
			reportCalled = true
			assert.Equal(t, "running", status, "Expected status 'running'")
		},
		ctx: context.Background(), // Provide a default context
	}

	testFunc := func(r Runner, p Params) Result {
		r.Report("running")
		return Success(map[string]any{
			"name":    p.Name,
			"age":     p.Age,
			"email":   p.Email,
			"isAdmin": p.IsAdmin,
		})
	}
	tool := Func("Report Tool", "Test function for report functionality", "report_tool", testFunc)

	params := json.RawMessage(`{"name":"Alice", "age":28, "email":"alice@example.com", "isAdmin":true}`)
	result := tool.Run(runner, params)

	require.NoError(t, result.Error(), "Expected no error")
	assert.True(t, reportCalled, "Expected report function to be called")
	resultJSON := extractJSONFromResult(t, result)
	assert.JSONEq(t, `{"name":"Alice","age":28,"email":"alice@example.com","isAdmin":true}`, string(resultJSON))
}
