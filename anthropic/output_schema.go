package anthropic

import (
	"encoding/json"

	"github.com/metalim/jsonmap"

	"github.com/flitsinc/go-llms/internal/schematree"
	"github.com/flitsinc/go-llms/tools"
)

// normalizeOutputSchemaForAnthropic returns a deep-normalized copy of schema
// for Anthropic structured outputs without mutating the caller's schema. Only
// the output schema is rewritten: a tool's input_schema goes out as declared,
// because Anthropic accepts array limits there as hints.
func normalizeOutputSchemaForAnthropic(schema *tools.ValueSchema) (any, error) {
	tree, err := schematree.Of(schema)
	if err != nil {
		return nil, err
	}
	schematree.WalkObjects(tree, normalizeOutputNodeForAnthropic)
	return tree, nil
}

// normalizeOutputNodeForAnthropic rewrites one schema node into what Anthropic
// structured outputs accept
// (https://platform.claude.com/docs/en/build-with-claude/structured-outputs):
//
//   - Every object needs additionalProperties: false.
//   - Arrays accept only a minItems of 0 or 1 and no maxItems, so maxItems is
//     dropped and a larger minItems is lowered to 1, which still admits every
//     array the original limits admit. The keywords mean nothing on a
//     non-array node, so they are rewritten wherever they appear. A minItems
//     that is not a number above 1 is sent as written: an invalid limit is the
//     caller's to fix and Anthropic's to reject.
func normalizeOutputNodeForAnthropic(node *jsonmap.Map) {
	if schematree.TypeIncludes(node, "object") || jsonMapLooksLikeObject(node) {
		node.Set("additionalProperties", false)
	}

	node.Delete("maxItems")
	value, _ := node.Get("minItems")
	minItems, _ := value.(json.Number)
	if count, err := minItems.Float64(); err == nil && count > 1 {
		node.Set("minItems", json.Number("1"))
	}
}

func jsonMapLooksLikeObject(node *jsonmap.Map) bool {
	for _, key := range []string{"properties", "patternProperties", "required", "dependencies", "dependentSchemas"} {
		if _, ok := node.Get(key); ok {
			return true
		}
	}
	return false
}
