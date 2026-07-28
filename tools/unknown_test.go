package tools

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/stretchr/testify/require"
)

func TestUnknown_RunsNothingAndReportsNotFound(t *testing.T) {
	// Arrange
	tool := Unknown("does_not_exist")

	// Act
	result := tool.Run(NewRunner(context.Background(), Box(), func(string) {}), json.RawMessage(`{"a":1}`))

	// Assert
	require.True(t, IsUnknown(tool))
	require.Equal(t, "does_not_exist", tool.FuncName())
	require.NotNil(t, tool.Grammar(), "consumers should be able to inspect the grammar uniformly")
	require.EqualError(t, result.Error(), `tool "does_not_exist" not found`)

	var notFound *NotFoundError
	require.ErrorAs(t, result.Error(), &notFound)
	require.Equal(t, "does_not_exist", notFound.FuncName)
}

func TestIsUnknown_RealToolIsNotUnknown(t *testing.T) {
	tool := Func("A", "desc", "a", func(r Runner, params struct{}) Result { return Success(nil) })
	require.False(t, IsUnknown(tool))
	require.False(t, IsUnknown(nil))
}

func TestToolbox_Run_MissingToolReportsNotFound(t *testing.T) {
	// Arrange
	tb := Box(Func("A", "desc", "a", func(r Runner, params struct{}) Result { return Success(nil) }))

	// Act
	result := tb.Run(NewRunner(context.Background(), tb, func(string) {}), "b", json.RawMessage(`{}`))

	// Assert: same error the Unknown placeholder produces, so the streaming and
	// non-streaming paths agree.
	require.EqualError(t, result.Error(), `tool "b" not found`)
	var notFound *NotFoundError
	require.ErrorAs(t, result.Error(), &notFound)
	require.Equal(t, "b", notFound.FuncName)
}
