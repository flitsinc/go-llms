package llms

import "context"

var debuggerContextKey = &contextKey{"debugger"}

// WithDebugger returns a copy of ctx carrying the given Debugger.
func WithDebugger(ctx context.Context, d Debugger) context.Context {
	return context.WithValue(ctx, debuggerContextKey, d)
}

// GetDebugger returns the Debugger stored in ctx, or nil if none is set.
func GetDebugger(ctx context.Context) Debugger {
	d, _ := ctx.Value(debuggerContextKey).(Debugger)
	return d
}
