package openai

import (
	"encoding/json"
	"slices"

	"github.com/metalim/jsonmap"

	"github.com/flitsinc/go-llms/internal/schematree"
)

// OpenAI strict mode rejects a schema unless every property is listed in
// `required`, so an optional property has to travel as required-but-nullable,
// `anyOf: [<schema>, {"type": "null"}]`, and the model sends null for it.
//
// The types that carry `strict` (FunctionTool, TextResponseFormat,
// jsonSchemaDefinition) pad their schema when they are encoded with Strict
// set, so a strict flag never reaches the wire with an unpadded schema, and
// callers write ordinary optional properties that every other provider
// receives as written. That matters for Claude: forced to write a value for an
// optional field it wants to leave out, it sometimes writes an empty one,
// which Anthropic streams as invalid argument JSON.
//
// Padding is idempotent, so a caller that pads its own schemas sends the same
// bytes as before. The model sends null for the optional properties it omits;
// tools.Func validation reads null for an optional property as left out.

// padStrictObject pads one object schema: every property becomes required,
// and one that was optional is wrapped as `anyOf: [<schema>, {"type": "null"}]`
// so the model can still leave it out. The wrap is unconditional, because a
// schema that looks nullable can still exclude null through a sibling keyword
// (an enum, say); a property that was already required is never wrapped, which
// makes padding idempotent. An empty additionalProperties schema (any value,
// which has no type) becomes false. A node that is itself an array or an anyOf
// is left as it is.
func padStrictObject(node *jsonmap.Map) {
	if _, ok := node.Get("anyOf"); ok || schematree.TypeIncludes(node, "array") || !schematree.TypeIncludes(node, "object") {
		return
	}

	if additional, ok := node.Get("additionalProperties"); ok {
		if additionalNode, ok := additional.(*jsonmap.Map); ok && additionalNode.Len() == 0 {
			node.Set("additionalProperties", false)
		}
	}

	properties, _ := node.Get("properties")
	propertiesNode, ok := properties.(*jsonmap.Map)
	if !ok || propertiesNode.Len() == 0 {
		// No properties (a tool without arguments), so nothing to require;
		// ValueSchema encodes such an object without `required`.
		return
	}
	requiredValue, _ := node.Get("required")
	wasRequired, _ := requiredValue.([]any)

	required := make([]any, 0, propertiesNode.Len())
	for _, key := range propertiesNode.Keys() {
		required = append(required, key)
		if slices.Contains(wasRequired, any(key)) {
			continue
		}
		property, _ := propertiesNode.Get(key)
		nullSchema := jsonmap.New()
		nullSchema.Set("type", "null")
		wrapped := jsonmap.New()
		wrapped.Set("anyOf", []any{property, nullSchema})
		propertiesNode.Set(key, wrapped)
	}
	node.Set("required", required)
}

// marshalStrict encodes value (a strict-carrying type converted to a type
// without this MarshalJSON) with the schema under schemaKey padded. It pads
// the value's own encoding, so field order and omission rules stay those of
// the canonical type.
func marshalStrict(value any, schemaKey string) ([]byte, error) {
	tree, err := schematree.Of(value)
	if err != nil {
		return nil, err
	}
	if schema, ok := tree.Get(schemaKey); ok {
		schematree.WalkObjects(schema, padStrictObject)
	}
	return json.Marshal(tree)
}

// MarshalJSON pads Parameters for strict mode when Strict is set.
func (t FunctionTool) MarshalJSON() ([]byte, error) {
	type plain FunctionTool
	if !t.Strict {
		return json.Marshal(plain(t))
	}
	return marshalStrict(plain(t), "parameters")
}

// MarshalJSON pads Schema for strict mode when Strict is set.
func (f TextResponseFormat) MarshalJSON() ([]byte, error) {
	type plain TextResponseFormat
	if !f.Strict {
		return json.Marshal(plain(f))
	}
	return marshalStrict(plain(f), "schema")
}

// MarshalJSON pads Schema for strict mode when Strict is set.
func (d jsonSchemaDefinition) MarshalJSON() ([]byte, error) {
	type plain jsonSchemaDefinition
	if !d.Strict {
		return json.Marshal(plain(d))
	}
	return marshalStrict(plain(d), "schema")
}
