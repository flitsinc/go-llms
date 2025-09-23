package llms

import (
	"context"
	"encoding/json"
)

// contextKey is a value for use with context.WithValue. It's used as a pointer
// so it fits in an interface{} without allocation.
type contextKey struct {
	name string
}

// ToolCallContextKey is a context key. It can be used in tool functions with
// Runner.Context().Value() to access the specific ToolCall instance that
// triggered the current tool execution. The associated value will be of type
// llms.ToolCall.
var ToolCallContextKey = &contextKey{"tool-call"}

type ToolCall struct {
	ID        string          `json:"id"`
	Name      string          `json:"name"`
	Arguments json.RawMessage `json:"arguments"`
	// Some providers require passing a second id for tool calls.
	ExtraID string `json:"extra_id,omitempty"`
}

// GetToolCall retrieves the ToolCall associated with the context, if present.
func GetToolCall(ctx context.Context) (ToolCall, bool) {
	val := ctx.Value(ToolCallContextKey)
	if val == nil {
		return ToolCall{}, false
	}
	tc, ok := val.(ToolCall)
	return tc, ok
}
