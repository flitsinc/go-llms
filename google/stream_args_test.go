package google

import (
	"encoding/json"
	"testing"
)

func ptrFloat64(v float64) *float64 { return &v }
func ptrString(v string) *string    { return &v }

// Ensure the builder matches the single-call streaming example in Google's docs:
// numberValue, then a streaming string with an empty continuation chunk, then done.
func TestStreamArgs_DocSingleCallExample(t *testing.T) {
	b := newStreamingArgsBuilder()

	// $.brightness numberValue 50
	if err := b.applyPartialArg(partialFunctionArgument{
		JSONPath:    "$.brightness",
		NumberValue: ptrFloat64(50),
	}); err != nil {
		t.Fatalf("apply brightness: %v", err)
	}
	if got := string(b.marshalSnapshot()); got != `{"brightness":50}` {
		t.Fatalf("snapshot after brightness = %s", got)
	}
	if got := b.buf.String(); got != `{"brightness":50` {
		t.Fatalf("buffer after brightness = %q", got)
	}

	// $.colorTemperature stringValue "warm", willContinue=true (no flush yet)
	if err := b.applyPartialArg(partialFunctionArgument{
		JSONPath:     "$.colorTemperature",
		StringValue:  ptrString("warm"),
		WillContinue: true,
	}); err != nil {
		t.Fatalf("apply colorTemperature warm: %v", err)
	}
	if got := string(b.marshalSnapshot()); got != `{"brightness":50,"colorTemperature":"warm"}` {
		t.Fatalf("snapshot after warm chunk = %s", got)
	}
	if got := b.buf.String(); got != `{"brightness":50` {
		t.Fatalf("buffer should not flush streaming string yet, got %q", got)
	}

	// Empty chunk for the same path (doc shows an empty partialArg to continue)
	if err := b.applyPartialArg(partialFunctionArgument{
		JSONPath:     "$.colorTemperature",
		WillContinue: true,
	}); err != nil {
		t.Fatalf("apply empty continuation: %v", err)
	}
	if got := string(b.marshalSnapshot()); got != `{"brightness":50,"colorTemperature":"warm"}` {
		t.Fatalf("snapshot after empty continuation = %s", got)
	}
	if got := b.buf.String(); got != `{"brightness":50` {
		t.Fatalf("buffer should still be unchanged before finalize, got %q", got)
	}

	// Final chunk closes the string and flushes to stream
	if err := b.applyPartialArg(partialFunctionArgument{
		JSONPath:    "$.colorTemperature",
		StringValue: ptrString(""),
	}); err != nil {
		t.Fatalf("apply final empty string chunk: %v", err)
	}
	changed := b.finalize()

	wantSnapshot := `{"brightness":50,"colorTemperature":"warm"}`
	if got := string(b.marshalSnapshot()); got != wantSnapshot {
		t.Fatalf("final snapshot = %s, want %s", got, wantSnapshot)
	}
	wantBuf := `{"brightness":50,"colorTemperature":"warm"}`
	if got := b.buf.String(); got != wantBuf {
		t.Fatalf("final buffer = %q, want %q", got, wantBuf)
	}
	if !changed {
		t.Fatalf("expected finalize to report changes (closing the object)")
	}
}

// Ensure nested paths and arrays are handled (supports $.location.latitude and $.cities[1].name).
func TestStreamArgs_NestedAndArrayPaths(t *testing.T) {
	b := newStreamingArgsBuilder()

	// Nested number
	if err := b.applyPartialArg(partialFunctionArgument{
		JSONPath:    "$.location.latitude",
		NumberValue: ptrFloat64(12.34),
	}); err != nil {
		t.Fatalf("apply latitude: %v", err)
	}

	// Array string streaming across two chunks
	if err := b.applyPartialArg(partialFunctionArgument{
		JSONPath:     "$.cities[1].name",
		StringValue:  ptrString("Bang"),
		WillContinue: true,
	}); err != nil {
		t.Fatalf("apply cities[1] first chunk: %v", err)
	}
	if err := b.applyPartialArg(partialFunctionArgument{
		JSONPath:    "$.cities[1].name",
		StringValue: ptrString("kok"),
	}); err != nil {
		t.Fatalf("apply cities[1] final chunk: %v", err)
	}

	changed := b.finalize()

	var snapshot map[string]any
	if err := json.Unmarshal(b.marshalSnapshot(), &snapshot); err != nil {
		t.Fatalf("unmarshal snapshot: %v", err)
	}

	loc, ok := snapshot["location"].(map[string]any)
	if !ok || loc["latitude"] != 12.34 {
		t.Fatalf("location.latitude not set correctly: %#v", snapshot["location"])
	}

	cities, ok := snapshot["cities"].([]any)
	if !ok || len(cities) < 2 {
		t.Fatalf("cities array not extended: %#v", snapshot["cities"])
	}
	city1, ok := cities[1].(map[string]any)
	if !ok || city1["name"] != "Bangkok" {
		t.Fatalf("cities[1].name not set correctly: %#v", cities[1])
	}

	wantBuf := `{"location":{"latitude":12.34},"cities":[null,{"name":"Bangkok"}]}`
	if got := b.buf.String(); got != wantBuf {
		t.Fatalf("final buffer = %q, want %q", got, wantBuf)
	}
	if !changed {
		t.Fatalf("expected finalize to report changes (closing the object)")
	}
}

// Regression: when args arrive first (with a closing brace) and partials follow,
// the builder must keep the buffer open and avoid producing malformed JSON slices.
func TestStreamArgs_ArgsThenPartialAppendKeepsBufferOpen(t *testing.T) {
	b := newStreamingArgsBuilder()

	// Simulate args with full JSON object.
	b.setFullArgs(json.RawMessage(`{"foo":1}`))
	if got := b.buf.String(); got != `{"foo":1` {
		t.Fatalf("buffer after setFullArgs should be open object, got %q", got)
	}

	// Now append a partial for another field.
	if err := b.applyPartialArg(partialFunctionArgument{
		JSONPath:    "$.bar",
		NumberValue: ptrFloat64(2),
	}); err != nil {
		t.Fatalf("apply partial: %v", err)
	}

	// Finalize should close properly.
	if !b.finalize() {
		t.Fatalf("expected finalize to report changes")
	}

	wantBuf := `{"foo":1,"bar":2}`
	if got := b.buf.String(); got != wantBuf {
		t.Fatalf("final buffer = %q, want %q", got, wantBuf)
	}

	var snapshot map[string]any
	if err := json.Unmarshal(b.marshalSnapshot(), &snapshot); err != nil {
		t.Fatalf("unmarshal snapshot: %v", err)
	}
	if snapshot["foo"] != float64(1) || snapshot["bar"] != float64(2) {
		t.Fatalf("snapshot mismatch: %#v", snapshot)
	}
}
