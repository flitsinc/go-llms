package schematree

import (
	"encoding/json"
	"testing"

	"github.com/metalim/jsonmap"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// visitOrder walks the tree of raw and returns the titles of the visited
// objects, in visit order.
func visitOrder(t *testing.T, raw string) []string {
	t.Helper()
	tree, err := Of(json.RawMessage(raw))
	require.NoError(t, err)
	var titles []string
	WalkObjects(tree, func(node *jsonmap.Map) {
		title, _ := node.Get("title")
		s, _ := title.(string)
		titles = append(titles, s)
	})
	return titles
}

func TestWalkObjectsVisitsEverySubschemaKeywordChildFirst(t *testing.T) {
	cases := map[string]string{}
	for _, keyword := range []string{"properties", "patternProperties", "dependentSchemas", "$defs", "definitions", "dependencies"} {
		cases[keyword] = `{"title": "parent", "` + keyword + `": {"a": {"title": "child"}}}`
	}
	for _, keyword := range []string{"anyOf", "allOf", "oneOf", "prefixItems"} {
		cases[keyword] = `{"title": "parent", "` + keyword + `": [{"title": "child"}]}`
	}
	for _, keyword := range []string{"items", "additionalProperties", "additionalItems", "contains", "propertyNames", "not", "if", "then", "else", "unevaluatedItems", "unevaluatedProperties"} {
		cases[keyword] = `{"title": "parent", "` + keyword + `": {"title": "child"}}`
	}
	for keyword, raw := range cases {
		t.Run(keyword, func(t *testing.T) {
			assert.Equal(t, []string{"child", "parent"}, visitOrder(t, raw))
		})
	}
}

// Keyword values that are data, not schemas, are never visited.
func TestWalkObjectsSkipsDataValues(t *testing.T) {
	titles := visitOrder(t, `{"title": "parent", "const": {"title": "data"}, "default": {"title": "data"}, "examples": [{"title": "data"}]}`)

	assert.Equal(t, []string{"parent"}, titles)
}

func TestTypeIncludes(t *testing.T) {
	tree, err := Of(json.RawMessage(`{"a": {"type": "object"}, "b": {"type": ["object", "null"]}, "c": {"type": "string"}, "d": {}}`))
	require.NoError(t, err)
	node := func(key string) *jsonmap.Map {
		value, _ := tree.Get(key)
		return value.(*jsonmap.Map)
	}

	assert.True(t, TypeIncludes(node("a"), "object"))
	assert.True(t, TypeIncludes(node("b"), "object"))
	assert.True(t, TypeIncludes(node("b"), "null"))
	assert.False(t, TypeIncludes(node("c"), "object"))
	assert.False(t, TypeIncludes(node("d"), "object"))
}
