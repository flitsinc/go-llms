// Package schematree rewrites JSON schemas for a provider's wire format on an
// ordered JSON tree.
package schematree

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/metalim/jsonmap"
)

// Of returns schema's JSON encoding as a deep, order-preserving tree: objects
// are *jsonmap.Map, arrays []any. Providers rewrite schemas on the tree, so
// the caller's schema, which toolboxes share across providers and requests, is
// never touched. The tree holds what the schema encodes to: for a
// tools.ValueSchema that is the fields it models, plus every keyword inside the
// raw properties and additionalProperties subtrees it carries.
func Of(schema any) (*jsonmap.Map, error) {
	raw, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}
	tree := jsonmap.New()
	if err := json.Unmarshal(raw, tree); err != nil {
		return nil, fmt.Errorf("failed to decode schema into ordered map: %w", err)
	}
	return tree, nil
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
