package google

import (
	"bytes"
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// argumentPath represents a parsed JSONPath like $.foo.bar or $.items[0].baz.
// Only supports simple dot segments and optional [index] on a segment.
type argumentPath struct {
	segments []pathSegment
}

type pathSegment struct {
	key   string
	index *int // nil if not an indexed segment
}

func parseArgumentPath(path string) (argumentPath, error) {
	path = strings.TrimSpace(path)
	path = strings.TrimPrefix(path, "$")
	path = strings.TrimPrefix(path, ".")
	if path == "" {
		return argumentPath{}, fmt.Errorf("empty path")
	}

	parts := strings.Split(path, ".")
	segments := make([]pathSegment, 0, len(parts))
	for _, part := range parts {
		if part == "" {
			return argumentPath{}, fmt.Errorf("empty segment in path %q", path)
		}
		seg := pathSegment{key: part}
		if strings.Contains(part, "[") {
			// support foo[0], bar[12]
			base, idxPart, ok := strings.Cut(part, "[")
			if !ok || !strings.HasSuffix(idxPart, "]") {
				return argumentPath{}, fmt.Errorf("unsupported index syntax in %q", part)
			}
			idxStr := strings.TrimSuffix(idxPart, "]")
			idx, err := strconv.Atoi(idxStr)
			if err != nil || idx < 0 {
				return argumentPath{}, fmt.Errorf("invalid index %q", idxStr)
			}
			seg.key = base
			seg.index = &idx
		}
		segments = append(segments, seg)
	}
	return argumentPath{segments: segments}, nil
}

// streamingArgsBuilder keeps both an execution snapshot (as map/array) and an
// append-only streaming buffer that is always a valid JSON prefix.
type streamingArgsBuilder struct {
	// execution snapshot
	root any
	// streaming buffer (prefix-safe, object form)
	buf bytes.Buffer
	// tracks if any fields written (for commas)
	hasFields bool
	// fields under construction for streaming (string accumulation)
	stringAccumulators map[string]string
}

func newStreamingArgsBuilder() *streamingArgsBuilder {
	b := &streamingArgsBuilder{
		root:               make(map[string]any),
		stringAccumulators: make(map[string]string),
	}
	b.resetBufferOpen()
	return b
}

func (b *streamingArgsBuilder) marshalSnapshot() json.RawMessage {
	data, err := json.Marshal(b.root)
	if err != nil {
		return json.RawMessage("{}")
	}
	return data
}

// setValue applies a partial argument to the execution snapshot.
func (b *streamingArgsBuilder) setValue(path argumentPath, value any) {
	current := b.root
	for i, seg := range path.segments {
		last := i == len(path.segments)-1
		obj, ok := current.(map[string]any)
		if !ok {
			obj = make(map[string]any)
		}
		// ensure nested container
		next, exists := obj[seg.key]
		if seg.index == nil {
			if last {
				obj[seg.key] = value
				b.root = updateRoot(b.root, path.segments[:i+1], value)
				return
			}
			if !exists {
				next = make(map[string]any)
				obj[seg.key] = next
			}
			current = next
			b.root = updateRoot(b.root, path.segments[:i+1], next)
			continue
		}

		// Array handling
		var arr []any
		if exists {
			if cast, ok := next.([]any); ok {
				arr = cast
			}
		}
		if arr == nil {
			arr = []any{}
		}
		idx := *seg.index
		if idx >= len(arr) {
			// extend
			ext := make([]any, idx+1)
			copy(ext, arr)
			arr = ext
		}
		if last {
			arr[idx] = value
		} else {
			if arr[idx] == nil {
				arr[idx] = make(map[string]any)
			}
			current = arr[idx]
		}
		obj[seg.key] = arr
		b.root = updateRoot(b.root, path.segments[:i+1], arr)
	}
}

// updateRoot writes value at the given path (segments) into root.
func updateRoot(root any, segs []pathSegment, value any) any {
	if len(segs) == 0 {
		return value
	}
	seg := segs[0]
	obj, ok := root.(map[string]any)
	if !ok {
		obj = make(map[string]any)
	}
	if seg.index == nil {
		if len(segs) == 1 {
			obj[seg.key] = value
		} else {
			obj[seg.key] = updateRoot(obj[seg.key], segs[1:], value)
		}
		return obj
	}
	// array case
	var arr []any
	if existing, ok := obj[seg.key].([]any); ok {
		arr = existing
	}
	idx := *seg.index
	if idx >= len(arr) {
		ext := make([]any, idx+1)
		copy(ext, arr)
		arr = ext
	}
	if len(segs) == 1 {
		arr[idx] = value
	} else {
		arr[idx] = updateRoot(arr[idx], segs[1:], value)
	}
	obj[seg.key] = arr
	return obj
}

// flushValueToStream appends a complete field to the streaming buffer.
func (b *streamingArgsBuilder) flushValueToStream(key string, value any) {
	if b.hasFields {
		b.buf.WriteByte(',')
	}
	b.hasFields = true
	keyBytes, _ := json.Marshal(key)
	b.buf.Write(keyBytes)
	b.buf.WriteByte(':')
	valBytes, err := json.Marshal(value)
	if err != nil {
		b.buf.WriteString("null")
		return
	}
	b.buf.Write(valBytes)
}

// applyPartialArg merges the fragment into snapshot and, if the field is complete,
// appends it to the streaming buffer. Incomplete string chunks are held until finished.
func (b *streamingArgsBuilder) applyPartialArg(p partialFunctionArgument) error {
	path, err := parseArgumentPath(p.JSONPath)
	if err != nil {
		return err
	}

	var val any
	switch {
	case p.StringValue != nil:
		acc := b.stringAccumulators[p.JSONPath] + *p.StringValue
		b.stringAccumulators[p.JSONPath] = acc
		if p.WillContinue {
			// Update snapshot with current string so tool execution sees progress.
			val = acc
			b.setValue(path, val)
			return nil
		}
		// Final chunk; flush
		val = acc
		delete(b.stringAccumulators, p.JSONPath)
	case p.NumberValue != nil:
		val = *p.NumberValue
	case p.BoolValue != nil:
		val = *p.BoolValue
	case p.NullValue != nil:
		val = nil
	default:
		// Treat a valueless partialArg as a signal that streaming continues but without a new value.
		// If we have an in-flight string, keep snapshot up to current accumulator.
		if acc, ok := b.stringAccumulators[p.JSONPath]; ok {
			b.setValue(path, acc)
		}
		return nil
	}

	// Update execution snapshot.
	b.setValue(path, val)
	// Append to streaming buffer.
	b.flushValueToStream(path.segments[0].key, extractTopValue(b.root, path.segments[0].key))
	return nil
}

// resetBufferOpen resets the streaming buffer to an open object prefix.
func (b *streamingArgsBuilder) resetBufferOpen() {
	b.buf.Reset()
	b.buf.WriteByte('{')
	b.hasFields = false
}

// setFullArgs replaces the builder state with a complete JSON object, preserving
// an open buffer so that subsequent partials can append without producing
// malformed JSON.
func (b *streamingArgsBuilder) setFullArgs(raw json.RawMessage) {
	b.stringAccumulators = make(map[string]string)
	b.root = make(map[string]any)
	b.resetBufferOpen()

	if len(raw) == 0 {
		return
	}

	var obj map[string]any
	if err := json.Unmarshal(raw, &obj); err != nil {
		return
	}
	b.root = obj

	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) < 2 || trimmed[0] != '{' || trimmed[len(trimmed)-1] != '}' {
		return
	}

	// Write existing fields into the open buffer (omit the trailing "}").
	body := bytes.TrimSpace(trimmed[1 : len(trimmed)-1])
	if len(body) > 0 {
		b.buf.Write(body)
		b.hasFields = len(bytes.TrimSpace(body)) > 0
	}
}

// extractTopValue returns root[key] without descending arrays/objects;
// used for streaming top-level fields.
func extractTopValue(root any, key string) any {
	if obj, ok := root.(map[string]any); ok {
		return obj[key]
	}
	return nil
}

// finalize closes any open strings and closes the object in the streaming buffer,
// ensuring the buffer forms valid JSON. It returns true if new argument content
// (not just the closing brace) was written to the stream.
func (b *streamingArgsBuilder) finalize() bool {
	changed := false
	before := b.buf.Len()
	// Flush any in-progress strings to both snapshot and stream.
	for pathStr, acc := range b.stringAccumulators {
		path, err := parseArgumentPath(pathStr)
		if err != nil {
			continue
		}
		b.setValue(path, acc)
		b.flushValueToStream(path.segments[0].key, extractTopValue(b.root, path.segments[0].key))
		delete(b.stringAccumulators, pathStr)
		changed = true
	}
	if !bytes.HasSuffix(b.buf.Bytes(), []byte("}")) {
		b.buf.WriteByte('}')
	}
	return changed || b.buf.Len() != before
}
