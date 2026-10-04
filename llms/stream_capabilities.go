package llms

import (
	"encoding/json"

	"github.com/flitsinc/go-llms/content"
)

// The interfaces below are optional ProviderStream capabilities. Only the
// providers that have them implement them, and the turn loop finds each one
// with a type assertion on the stream Generate returned. A stream that wraps
// another must therefore declare every one of them or the wrapped stream's
// capabilities are silently lost: embed StreamWrapper to forward them all.
// A new capability gets its interface here, a place in StreamCapabilities and
// a forwarding method on StreamWrapper, so every wrapper picks it up.

// ContextUsageStream is implemented by provider streams whose Usage also
// counts sampling passes that do not carry the request's context forward, such
// as a context-compaction pass. ContextUsage reports only the final pass: the
// context the request ended with, after any compaction.
type ContextUsageStream interface {
	ContextUsage() Usage
}

// SearchStream is implemented by provider streams that report provider-run
// searches. Search returns the activity behind the latest StreamStatusSearch.
type SearchStream interface {
	Search() SearchActivity
}

// CompactionStream is implemented by provider streams that return native
// context compaction checkpoints. Compaction returns the checkpoint behind the
// latest StreamStatusCompaction.
type CompactionStream interface {
	Compaction() content.Compaction
}

// ToolArgumentFinalizationStream is implemented by provider streams whose
// protocol sends an independent final snapshot of a tool call's arguments.
// ToolArgumentFinalization returns it for the active tool call, and whether
// the protocol expects one at all.
type ToolArgumentFinalizationStream interface {
	ToolArgumentFinalization() (json.RawMessage, bool)
}

// StreamCapabilities is every optional ProviderStream capability. StreamWrapper
// implements it; a wrapper type can assert it does too.
type StreamCapabilities interface {
	ContextUsageStream
	SearchStream
	CompactionStream
	ToolArgumentFinalizationStream
}

// StreamWrapper embeds a ProviderStream and forwards every optional
// capability to it, reporting what the turn loop would see for an unwrapped
// stream when the inner stream lacks one. Embed it in a stream wrapper in place
// of ProviderStream and override only the methods the wrapper changes.
//
// A wrapper that overrides Usage must override ContextUsage too:
// StreamWrapper always implements ContextUsage, so the turn loop never falls
// back to the outer Usage, and StreamWrapper's own fallback reads the inner
// stream's.
type StreamWrapper struct {
	ProviderStream
}

var _ StreamCapabilities = StreamWrapper{}

// ContextUsage forwards the final sampling pass's usage, or the inner
// stream's whole Usage when it has no separate passes.
func (w StreamWrapper) ContextUsage() Usage {
	if cs, ok := w.ProviderStream.(ContextUsageStream); ok {
		return cs.ContextUsage()
	}
	return w.Usage()
}

// Search forwards provider-run search activity.
func (w StreamWrapper) Search() SearchActivity {
	if s, ok := w.ProviderStream.(SearchStream); ok {
		return s.Search()
	}
	return SearchActivity{}
}

// Compaction forwards the native compaction checkpoint.
func (w StreamWrapper) Compaction() content.Compaction {
	if c, ok := w.ProviderStream.(CompactionStream); ok {
		return c.Compaction()
	}
	return content.Compaction{}
}

// ToolArgumentFinalization forwards the final tool-argument snapshot.
func (w StreamWrapper) ToolArgumentFinalization() (json.RawMessage, bool) {
	if f, ok := w.ProviderStream.(ToolArgumentFinalizationStream); ok {
		return f.ToolArgumentFinalization()
	}
	return nil, false
}
