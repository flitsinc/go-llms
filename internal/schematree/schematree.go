// Package schematree rewrites JSON schemas for a provider's wire format on an
// ordered JSON tree.
package schematree

import (
	"bytes"
	"encoding/json"
	"fmt"
	"slices"

	"github.com/metalim/jsonmap"
)

// Of returns schema's JSON encoding as a deep, order-preserving tree: objects
// are *jsonmap.Map, arrays []any, numbers json.Number (so a large integer
// keeps its exact value). Providers rewrite schemas on the tree, so the
// caller's schema, which toolboxes share across providers and requests, is
// never touched. The tree holds what the schema encodes to: for a
// tools.ValueSchema that is the fields it models, plus every keyword inside
// the raw properties and additionalProperties subtrees it carries.
func Of(schema any) (*jsonmap.Map, error) {
	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}
	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	value, err := decodeValue(decoder)
	if err != nil {
		return nil, fmt.Errorf("failed to decode schema into ordered map: %w", err)
	}
	tree, ok := value.(*jsonmap.Map)
	if !ok {
		return nil, fmt.Errorf("schema encodes to %T, not an object", value)
	}
	return tree, nil
}

// decodeValue decodes the next JSON value, keeping object key order and exact
// numbers (the decoder uses UseNumber).
func decodeValue(decoder *json.Decoder) (any, error) {
	token, err := decoder.Token()
	if err != nil {
		return nil, err
	}
	switch token {
	case json.Delim('{'):
		object := jsonmap.New()
		for decoder.More() {
			keyToken, err := decoder.Token()
			if err != nil {
				return nil, err
			}
			key, _ := keyToken.(string)
			value, err := decodeValue(decoder)
			if err != nil {
				return nil, err
			}
			object.Push(key, value)
		}
		_, err := decoder.Token() // '}'
		return object, err
	case json.Delim('['):
		array := []any{}
		for decoder.More() {
			value, err := decodeValue(decoder)
			if err != nil {
				return nil, err
			}
			array = append(array, value)
		}
		_, err := decoder.Token() // ']'
		return array, err
	default:
		return token, nil
	}
}

// WalkObjects calls visit on every schema object in a tree from Of,
// children before their parent, so visit may replace a node's subschemas
// without the walk descending into the replacements.
func WalkObjects(node any, visit func(*jsonmap.Map)) {
	switch n := node.(type) {
	case *jsonmap.Map:
		for _, key := range n.Keys() {
			child, _ := n.Get(key)
			switch schemaChildKind(key) {
			case schemaChildMap:
				if children, ok := child.(*jsonmap.Map); ok {
					for _, name := range children.Keys() {
						grandchild, _ := children.Get(name)
						WalkObjects(grandchild, visit)
					}
				}
			case schemaChildList, schemaChildDirect:
				WalkObjects(child, visit)
			}
		}
		visit(n)
	case []any:
		for _, item := range n {
			WalkObjects(item, visit)
		}
	}
}

// TypeIncludes reports whether a schema node's "type" (a name or a list
// of names) includes want.
func TypeIncludes(node *jsonmap.Map, want string) bool {
	switch t, _ := node.Get("type"); types := t.(type) {
	case string:
		return types == want
	case []any:
		return slices.Contains(types, any(want))
	}
	return false
}

type schemaChild uint8

const (
	schemaChildNone schemaChild = iota
	// schemaChildMap holds subschemas by name (properties, $defs, ...).
	schemaChildMap
	// schemaChildList holds a list of subschemas (anyOf, prefixItems, ...).
	schemaChildList
	// schemaChildDirect holds one subschema (items, not, ...).
	schemaChildDirect
)

func schemaChildKind(key string) schemaChild {
	switch key {
	case "properties", "patternProperties", "dependentSchemas", "$defs", "definitions", "dependencies":
		return schemaChildMap
	case "anyOf", "allOf", "oneOf", "prefixItems":
		return schemaChildList
	case "items", "additionalProperties", "additionalItems", "contains", "propertyNames", "not", "if", "then", "else", "unevaluatedItems", "unevaluatedProperties":
		return schemaChildDirect
	default:
		return schemaChildNone
	}
}
