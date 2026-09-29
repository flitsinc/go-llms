package tools

import (
	"bytes"
	"encoding/json"
	"slices"
)

// omitOptionalNulls returns arguments with every null-valued optional property
// removed, at any depth the schema describes (object properties and array
// items). OpenAI strict mode makes every property required and has the model
// send null for the optional ones it leaves out, so a null for a property the
// schema does not require is an omission. Removing it before validation and
// decoding means both see the call as the model meant it; decoding null
// instead would still run a field's own UnmarshalJSON, which may reject it.
//
// Arguments that need no change are returned as they are. Otherwise they are
// re-encoded from a decode that keeps exact numbers.
func omitOptionalNulls(schema ValueSchema, arguments json.RawMessage) json.RawMessage {
	// Leave malformed arguments, including trailing input after the first
	// value, to validation, which rejects them. Decoding reads only the first
	// value, so rewriting it would silently drop the rest.
	if !json.Valid(arguments) {
		return arguments
	}
	decoder := json.NewDecoder(bytes.NewReader(arguments))
	decoder.UseNumber()
	var value any
	if err := decoder.Decode(&value); err != nil {
		return arguments
	}
	if !dropOptionalNulls(schema, value) {
		return arguments
	}
	normalized, err := json.Marshal(value)
	if err != nil {
		return arguments
	}
	return normalized
}

// dropOptionalNulls removes null-valued optional properties from value in
// place and reports whether it removed any.
func dropOptionalNulls(schema ValueSchema, value any) bool {
	changed := false
	switch v := value.(type) {
	case map[string]any:
		if schema.Properties == nil {
			return false
		}
		for key, property := range v {
			rawPropertySchema, known := schema.Properties.Get(key)
			if !known {
				continue
			}
			if property == nil && !slices.Contains(schema.Required, key) {
				delete(v, key)
				changed = true
				continue
			}
			if propertySchema, err := PropertySchema(rawPropertySchema); err == nil && dropOptionalNulls(propertySchema, property) {
				changed = true
			}
		}
	case []any:
		if schema.Items == nil {
			return false
		}
		for _, item := range v {
			if dropOptionalNulls(*schema.Items, item) {
				changed = true
			}
		}
	}
	return changed
}
