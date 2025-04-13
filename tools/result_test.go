package tools

import (
	"encoding/json"
	"errors"
	"strings"
	"testing"

	"github.com/blixt/go-llms/content"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// NOTE: extractJSONFromResult helper moved to tool_test.go to avoid redeclaration.
// func extractJSONFromResult(t *testing.T, r Result) json.RawMessage { ... }

// Simple struct implementing Stringer for label testing
type stringerStruct struct {
	Value string
}

func (s stringerStruct) String() string {
	return "StringerLabel: " + s.Value
}

// Struct that causes JSON marshal error (contains a channel)
type unmarshalableStruct struct {
	FailChan chan int
}

func TestSuccess(t *testing.T) {
	t.Run("With Stringer", func(t *testing.T) {
		val := stringerStruct{Value: "Test"}
		res := Success(val)
		assert.NoError(t, res.Error())
		assert.Equal(t, "StringerLabel: Test", res.Label())
		// Assuming extractJSONFromResult exists in tool_test.go or similar helper file
		// For isolated test, we'd check manually:
		require.Len(t, res.Content(), 1)
		jsonItem, ok := res.Content()[0].(*content.JSON)
		require.True(t, ok)
		assert.JSONEq(t, `{"Value":"Test"}`, string(jsonItem.Data))
	})

	t.Run("Without Stringer", func(t *testing.T) {
		val := map[string]int{"count": 5}
		res := Success(val)
		assert.NoError(t, res.Error())
		assert.Equal(t, "Success", res.Label()) // Default label
		require.Len(t, res.Content(), 1)
		jsonItem, ok := res.Content()[0].(*content.JSON)
		require.True(t, ok)
		assert.JSONEq(t, `{"count":5}`, string(jsonItem.Data))
	})

	t.Run("With Marshal Error", func(t *testing.T) {
		val := unmarshalableStruct{FailChan: make(chan int)}
		res := Success(val)
		assert.Error(t, res.Error())
		assert.Contains(t, res.Error().Error(), "json: unsupported type: chan int")
		// Assert the DESIRED behavior: Label should indicate the error using the original label.
		// The label should now be exactly "Error (Success)"
		assert.Equal(t, "Error (Success)", res.Label(), "Label should be 'Error (Success)'")
		require.Len(t, res.Content(), 1)
		jsonItem, ok := res.Content()[0].(*content.JSON)
		require.True(t, ok)
		assert.Contains(t, string(jsonItem.Data), "failed to marshal success result to JSON")
	})
}

func TestSuccessWithLabel(t *testing.T) {
	t.Run("Valid Value", func(t *testing.T) {
		val := map[string]bool{"ok": true}
		res := SuccessWithLabel("Custom Label", val)
		assert.NoError(t, res.Error())
		assert.Equal(t, "Custom Label", res.Label())
		require.Len(t, res.Content(), 1)
		jsonItem, ok := res.Content()[0].(*content.JSON)
		require.True(t, ok)
		assert.JSONEq(t, `{"ok":true}`, string(jsonItem.Data))
	})

	t.Run("With Marshal Error", func(t *testing.T) {
		val := unmarshalableStruct{FailChan: make(chan int)}
		res := SuccessWithLabel("My Successful Result", val)
		assert.Error(t, res.Error())
		assert.Contains(t, res.Error().Error(), "json: unsupported type: chan int")
		// Assert the DESIRED behavior: Label should indicate the error using the original label.
		// The label should now be exactly "Error (My Successful Result)"
		assert.Equal(t, "Error (My Successful Result)", res.Label(), "Label should be 'Error (My Successful Result)'")
		require.Len(t, res.Content(), 1)
		jsonItem, ok := res.Content()[0].(*content.JSON)
		require.True(t, ok)
		assert.Contains(t, string(jsonItem.Data), "failed to marshal success result to JSON")
	})
}

func TestSuccessFromString(t *testing.T) {
	res := SuccessFromString("Simple output string")
	assert.NoError(t, res.Error())
	assert.Equal(t, "Simple output string", res.Label())
	require.Len(t, res.Content(), 1)
	jsonItem, ok := res.Content()[0].(*content.JSON)
	require.True(t, ok)
	assert.JSONEq(t, `{"output":"Simple output string"}`, string(jsonItem.Data))

	// Test truncation
	longString := strings.Repeat("a", 100)
	res = SuccessFromString(longString)
	assert.Equal(t, strings.Repeat("a", 77)+"...", res.Label())
}

func TestSuccessFromStringf(t *testing.T) {
	res := Successf("Count: %d, Status: %s", 10, "active")
	assert.NoError(t, res.Error())
	assert.Equal(t, "Count: 10, Status: active", res.Label())
	require.Len(t, res.Content(), 1)
	jsonItem, ok := res.Content()[0].(*content.JSON)
	require.True(t, ok)
	assert.JSONEq(t, `{"output":"Count: 10, Status: active"}`, string(jsonItem.Data))
}

func TestSuccessFromStringWithLabel(t *testing.T) {
	res := SuccessFromStringWithLabel("Specific Task", "Task output details")
	assert.NoError(t, res.Error())
	assert.Equal(t, "Specific Task", res.Label())
	require.Len(t, res.Content(), 1)
	jsonItem, ok := res.Content()[0].(*content.JSON)
	require.True(t, ok)
	assert.JSONEq(t, `{"output":"Task output details"}`, string(jsonItem.Data))
}

func TestSuccessWithContent(t *testing.T) {
	jsonData, _ := json.Marshal(map[string]string{"field": "value"})
	testContent := content.FromRawJSON(jsonData)
	testContent.AddImage("data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==")

	res := SuccessWithContent("Multi-Content Result", testContent)
	assert.NoError(t, res.Error())
	assert.Equal(t, "Multi-Content Result", res.Label())
	require.Len(t, res.Content(), 2)
	// Correct type assertions using concrete types
	_, ok1 := res.Content()[0].(*content.JSON)
	_, ok2 := res.Content()[1].(*content.ImageURL)
	assert.True(t, ok1, "First item should be *content.JSON")
	assert.True(t, ok2, "Second item should be *content.ImageURL")
}

func TestError(t *testing.T) {
	err := errors.New("standard error")
	res := Error(err)
	assert.Error(t, res.Error())
	assert.Same(t, err, res.Error()) // Check if it's the exact same error object
	assert.Equal(t, "Error: standard error", res.Label())
	require.Len(t, res.Content(), 1)
	jsonItem, ok := res.Content()[0].(*content.JSON)
	require.True(t, ok)
	assert.JSONEq(t, `{"error":"standard error"}`, string(jsonItem.Data))
}

func TestErrorf(t *testing.T) {
	res := Errorf("formatted error: %d", 123)
	assert.Error(t, res.Error())
	assert.EqualError(t, res.Error(), "formatted error: 123")
	assert.Equal(t, "Error: formatted error: 123", res.Label())
	require.Len(t, res.Content(), 1)
	jsonItem, ok := res.Content()[0].(*content.JSON)
	require.True(t, ok)
	assert.JSONEq(t, `{"error":"formatted error: 123"}`, string(jsonItem.Data))
}

func TestErrorWithLabel(t *testing.T) {
	err := errors.New("internal error")
	res := ErrorWithLabel("Failed Operation", err)
	assert.Error(t, res.Error())
	assert.Same(t, err, res.Error())
	assert.Equal(t, "Failed Operation", res.Label())
	require.Len(t, res.Content(), 1)
	jsonItem, ok := res.Content()[0].(*content.JSON)
	require.True(t, ok)
	assert.JSONEq(t, `{"error":"internal error"}`, string(jsonItem.Data))
}
