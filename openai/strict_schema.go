package openai

import (
	"encoding/json"
	"slices"

	"github.com/metalim/jsonmap"

	"github.com/flitsinc/go-llms/internal/schematree"
	"github.com/flitsinc/go-llms/tools"
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
// bytes as before.

// strictSchemaTree returns the strict-mode form of schema as a JSON tree,
// leaving schema itself untouched.
func strictSchemaTree(schema *tools.ValueSchema) (*jsonmap.Map, error) {
	tree, err := schematree.Of(schema)
	if err != nil {
		return nil, err
	}
	schematree.WalkObjects(tree, padStrictObject)
	return tree, nil
}

// padStrictObject pads one object schema whose subschemas are already padded:
// every property becomes required, an optional one that does not already admit
// null is wrapped as `anyOf: [<schema>, {"type": "null"}]`, and an empty
// additionalProperties schema (any value, which has no type) becomes false.
// A node that is itself an array or an anyOf is left as it is.
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
	if !ok {
		return
	}
	if propertiesNode.Len() == 0 {
		// A tool without arguments; ValueSchema encodes it without `required`.
		return
	}
	requiredValue, _ := node.Get("required")
	wasRequired, _ := requiredValue.([]any)

	required := make([]any, 0, propertiesNode.Len())
	for _, key := range propertiesNode.Keys() {
		required = append(required, key)
		property, _ := propertiesNode.Get(key)
		propertyNode, ok := property.(*jsonmap.Map)
		if !ok || slices.Contains(wasRequired, any(key)) || admitsNull(propertyNode) {
			continue
		}
		nullSchema := jsonmap.New()
		nullSchema.Set("type", "null")
		wrapped := jsonmap.New()
		wrapped.Set("anyOf", []any{propertyNode, nullSchema})
		propertiesNode.Set(key, wrapped)
	}
	node.Set("required", required)
}

// admitsNull reports whether a schema already accepts null: it is
// `{"type": "null"}` or has such a branch (at any depth) in its anyOf.
func admitsNull(node *jsonmap.Map) bool {
	if schematree.TypeIncludes(node, "null") {
		return true
	}
	anyOf, _ := node.Get("anyOf")
	branches, _ := anyOf.([]any)
	return slices.ContainsFunc(branches, func(branch any) bool {
		branchNode, ok := branch.(*jsonmap.Map)
		return ok && admitsNull(branchNode)
	})
}

// The MarshalJSON methods below spell out their wire fields in the order the
// plain encoding uses, so a padded schema that was already strict encodes to
// the same bytes as before (TestStrictTypesKeepTheirPlainEncoding).

// MarshalJSON pads Parameters for strict mode when Strict is set.
func (t FunctionTool) MarshalJSON() ([]byte, error) {
	type plain FunctionTool
	if !t.Strict || t.Parameters == nil {
		return json.Marshal(plain(t))
	}
	parameters, err := strictSchemaTree(t.Parameters)
	if err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		Type        string       `json:"type"`
		Name        string       `json:"name"`
		Description string       `json:"description,omitempty"`
		Parameters  *jsonmap.Map `json:"parameters"`
		Strict      bool         `json:"strict"`
	}{t.Type, t.Name, t.Description, parameters, t.Strict})
}

// MarshalJSON pads Schema for strict mode when Strict is set.
func (f TextResponseFormat) MarshalJSON() ([]byte, error) {
	type plain TextResponseFormat
	if !f.Strict || f.Schema == nil {
		return json.Marshal(plain(f))
	}
	schema, err := strictSchemaTree(f.Schema)
	if err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		Type        string       `json:"type"`
		Name        string       `json:"name,omitempty"`
		Schema      *jsonmap.Map `json:"schema,omitempty"`
		Description string       `json:"description,omitempty"`
		Strict      bool         `json:"strict,omitempty"`
	}{f.Type, f.Name, schema, f.Description, f.Strict})
}

// MarshalJSON pads Schema for strict mode when Strict is set.
func (d jsonSchemaDefinition) MarshalJSON() ([]byte, error) {
	type plain jsonSchemaDefinition
	if !d.Strict || d.Schema == nil {
		return json.Marshal(plain(d))
	}
	schema, err := strictSchemaTree(d.Schema)
	if err != nil {
		return nil, err
	}
	return json.Marshal(struct {
		Name   string       `json:"name"`
		Schema *jsonmap.Map `json:"schema"`
		Strict bool         `json:"strict,omitempty"`
	}{d.Name, schema, d.Strict})
}
