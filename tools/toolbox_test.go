package tools

import (
	"testing"

	"github.com/stretchr/testify/require"
)

func TestToolbox_All_PreservesInsertionOrder(t *testing.T) {
	// Arrange
	toolA := Func("A", "desc", "a", func(r Runner, params struct{}) Result { return Success(map[string]any{"ok": true}) })
	toolB := Func("B", "desc", "b", func(r Runner, params struct{}) Result { return Success(map[string]any{"ok": true}) })
	toolC := Func("C", "desc", "c", func(r Runner, params struct{}) Result { return Success(map[string]any{"ok": true}) })

	tb := Box()
	tb.Add(toolA)
	tb.Add(toolB)
	tb.Add(toolC)

	// Act
	all := tb.All()

	// Assert
	require.Len(t, all, 3)
	require.Equal(t, "a", all[0].FuncName())
	require.Equal(t, "b", all[1].FuncName())
	require.Equal(t, "c", all[2].FuncName())
}

func TestToolbox_Add_DuplicatePanics(t *testing.T) {
	toolA1 := Func("A1", "desc", "dup", func(r Runner, params struct{}) Result { return Success(nil) })
	toolA2 := Func("A2", "desc", "dup", func(r Runner, params struct{}) Result { return Success(nil) })
	tb := Box(toolA1)
	require.Panics(t, func() { tb.Add(toolA2) })
}
