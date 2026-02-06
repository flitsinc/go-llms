package anthropic

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/metalim/jsonmap"

	"github.com/flitsinc/go-llms/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func normalizedSchemaAsMap(t *testing.T, normalized any) map[string]any {
	t.Helper()
	data, err := json.Marshal(normalized)
	require.NoError(t, err)
	var out map[string]any
	require.NoError(t, json.Unmarshal(data, &out))
	return out
}

func keyPositions(body string, keys ...string) []int {
	pos := make([]int, len(keys))
	for i, key := range keys {
		pos[i] = strings.Index(body, fmt.Sprintf(`"%s"`, key))
	}
	return pos
}

func positionsOrdered(pos []int) bool {
	for i := 1; i < len(pos); i++ {
		if pos[i-1] < 0 || pos[i] < 0 || pos[i-1] >= pos[i] {
			return false
		}
	}
	return true
}

func TestNormalizeOutputSchemaForAnthropic_OverridesPresetAndPreservesFields(t *testing.T) {
	props := jsonmap.New()
	props.Set("status", map[string]any{
		"type":    "string",
		"enum":    []any{"open", "closed"},
		"pattern": "^(open|closed)$",
	})
	props.Set("dict", tools.ValueSchema{
		Type: "object",
		AdditionalProperties: tools.ValueSchema{
			Type: "string",
		},
	})
	props.Set("metadata", map[string]any{
		"type": "object",
		"properties": map[string]any{
			"count": map[string]any{
				"type":    "integer",
				"minimum": 0,
			},
		},
		"additionalProperties": map[string]any{
			"type": "string",
		},
		"anyOf": []any{
			map[string]any{
				"type":                 "object",
				"additionalProperties": true,
				"properties": map[string]any{
					"id": map[string]any{
						"type":    "string",
						"pattern": "^[a-z]+$",
					},
				},
			},
			map[string]any{
				"type": "string",
			},
		},
	})

	schema := tools.ValueSchema{
		Type:                 "object",
		Properties:           props,
		AdditionalProperties: true,
	}

	normalizedAny := normalizeOutputSchemaForAnthropic(&schema)
	normalized := normalizedSchemaAsMap(t, normalizedAny)
	assert.Equal(t, false, normalized["additionalProperties"])

	normalizedProps, ok := normalized["properties"].(map[string]any)
	require.True(t, ok)

	rawStatus, ok := normalizedProps["status"]
	require.True(t, ok)
	status, ok := rawStatus.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, []any{"open", "closed"}, status["enum"])
	assert.Equal(t, "^(open|closed)$", status["pattern"])

	rawMetadata, ok := normalizedProps["metadata"]
	require.True(t, ok)
	metadata, ok := rawMetadata.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, false, metadata["additionalProperties"])

	rawDict, ok := normalizedProps["dict"]
	require.True(t, ok)
	dict, ok := rawDict.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, false, dict["additionalProperties"])

	metadataProps, ok := metadata["properties"].(map[string]any)
	require.True(t, ok)
	count, ok := metadataProps["count"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, float64(0), count["minimum"])

	metadataAnyOf, ok := metadata["anyOf"].([]any)
	require.True(t, ok)
	require.GreaterOrEqual(t, len(metadataAnyOf), 1)
	firstAnyOfObject, ok := metadataAnyOf[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, false, firstAnyOfObject["additionalProperties"])
}

func TestNormalizeOutputSchemaForAnthropic_DoesNotMutateInputSchema(t *testing.T) {
	props := jsonmap.New()
	props.Set("nested", map[string]any{
		"type":                 "object",
		"additionalProperties": true,
	})

	schema := tools.ValueSchema{
		Type:       "object",
		Properties: props,
		AnyOf: []tools.ValueSchema{
			{
				Type:                 "object",
				AdditionalProperties: true,
			},
		},
	}

	normalizedAny := normalizeOutputSchemaForAnthropic(&schema)
	normalized := normalizedSchemaAsMap(t, normalizedAny)
	normalizedAnyOf, ok := normalized["anyOf"].([]any)
	require.True(t, ok)
	require.GreaterOrEqual(t, len(normalizedAnyOf), 1)
	firstAnyOf, ok := normalizedAnyOf[0].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, false, firstAnyOf["additionalProperties"])

	// Ensure the original schema graph is unchanged.
	assert.Equal(t, true, schema.AnyOf[0].AdditionalProperties)
	rawNested, ok := schema.Properties.Get("nested")
	require.True(t, ok)
	_, isMap := rawNested.(map[string]any)
	assert.True(t, isMap)
	_, isValueSchema := rawNested.(tools.ValueSchema)
	assert.False(t, isValueSchema)
	nested, ok := rawNested.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, true, nested["additionalProperties"])
}

func TestNormalizeOutputSchemaForAnthropic_NormalizesRawMessageWithoutFieldLoss(t *testing.T) {
	rawChild := json.RawMessage(`{
		"type":"object",
		"additionalProperties":{"type":"string"},
		"properties":{
			"label":{"type":"string","minLength":1}
		}
	}`)

	props := jsonmap.New()
	props.Set("child", rawChild)

	schema := tools.ValueSchema{
		Type:       "object",
		Properties: props,
	}

	normalizedAny := normalizeOutputSchemaForAnthropic(&schema)
	normalized := normalizedSchemaAsMap(t, normalizedAny)
	normalizedProps, ok := normalized["properties"].(map[string]any)
	require.True(t, ok)
	rawChildNormalized, ok := normalizedProps["child"]
	require.True(t, ok)
	child, ok := rawChildNormalized.(map[string]any)
	require.True(t, ok)
	assert.Equal(t, false, child["additionalProperties"])
	childProps, ok := child["properties"].(map[string]any)
	require.True(t, ok)
	label, ok := childProps["label"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, float64(1), label["minLength"])
}

func TestNormalizeOutputSchemaForAnthropic_PreservesPropertyOrder(t *testing.T) {
	props := jsonmap.New()
	props.Set("reasoning", tools.ValueSchema{Type: "string"})
	props.Set("answer", tools.ValueSchema{Type: "string"})
	props.Set("confidence", tools.ValueSchema{Type: "number"})

	schema := tools.ValueSchema{
		Type:       "object",
		Properties: props,
	}

	normalizedAny := normalizeOutputSchemaForAnthropic(&schema)
	data, err := json.Marshal(normalizedAny)
	require.NoError(t, err)

	pos := keyPositions(string(data), "reasoning", "answer", "confidence")
	assert.True(t, positionsOrdered(pos), "schema property order should be preserved in serialized output")
}
