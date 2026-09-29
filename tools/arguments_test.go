package tools

import (
	"encoding/json"
	"errors"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// nonNullString decodes a JSON string and rejects null, like a custom field
// decoder that has no meaning for null.
type nonNullString string

func (s *nonNullString) UnmarshalJSON(data []byte) error {
	var value *string
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	if value == nil {
		return errors.New("expected a string value")
	}
	*s = nonNullString(*value)
	return nil
}

type argumentsItem struct {
	Label string `json:"label"`
	Hint  *int   `json:"hint,omitempty"`
}

type argumentsParams struct {
	Name    string        `json:"name"`
	Count   int64         `json:"count,omitempty"`
	Note    nonNullString `json:"note,omitempty"`
	Options *struct {
		Tags []string `json:"tags,omitempty"`
	} `json:"options,omitempty"`
	Items []argumentsItem `json:"items,omitempty"`
}

func argumentsTool(got *argumentsParams) Tool {
	return Func("Task", "Start a task", "task", func(r Runner, p argumentsParams) Result {
		*got = p
		return SuccessFromString("ok")
	})
}

// An optional field is left out whether the model omits it or, as OpenAI
// strict mode has it do, sends null; its own decoder never sees the null.
func TestToolRun_OptionalNullIsOmission(t *testing.T) {
	for name, arguments := range map[string]string{
		"omitted": `{"name": "task"}`,
		"string":  `{"name": "task", "note": "hi"}`,
		"null":    `{"name": "task", "note": null}`,
	} {
		t.Run(name, func(t *testing.T) {
			var got argumentsParams
			result := argumentsTool(&got).Run(nil, json.RawMessage(arguments))

			require.NoError(t, result.Error())
			assert.Equal(t, "task", got.Name)
		})
	}
}

func TestToolRun_OptionalNullIsOmissionAtAnyDepth(t *testing.T) {
	var got argumentsParams
	result := argumentsTool(&got).Run(nil, json.RawMessage(
		`{"name": "task", "count": 9007199254740993, "options": {"tags": null}, "items": [{"label": "a", "hint": null}]}`,
	))

	require.NoError(t, result.Error())
	require.NotNil(t, got.Options)
	assert.Nil(t, got.Options.Tags)
	require.Len(t, got.Items, 1)
	assert.Equal(t, "a", got.Items[0].Label)
	assert.Nil(t, got.Items[0].Hint)
	// Re-encoding after dropping the nulls keeps exact numbers.
	assert.Equal(t, int64(9007199254740993), got.Count)
}

func TestToolRun_RequiredNullIsRejected(t *testing.T) {
	var got argumentsParams
	tool := argumentsTool(&got)

	assert.Error(t, tool.Run(nil, json.RawMessage(`{"name": null}`)).Error())
	assert.Error(t, tool.Run(nil, json.RawMessage(`{"name": "task", "items": [{"label": null}]}`)).Error())
}

func TestOmitOptionalNullsLeavesOtherArgumentsUntouched(t *testing.T) {
	var got argumentsParams
	grammar, ok := argumentsTool(&got).Grammar().(JSONGrammar)
	require.True(t, ok)
	parameters := grammar.Schema().Parameters

	unchanged := json.RawMessage(`{ "name": "task",  "count": 9007199254740993 }`)
	assert.Equal(t, string(unchanged), string(omitOptionalNulls(parameters, unchanged)))

	malformed := json.RawMessage(`{"name": `)
	assert.Equal(t, string(malformed), string(omitOptionalNulls(parameters, malformed)))

	// An unknown property is not the schema's to drop.
	unknown := json.RawMessage(`{"name": "task", "extra": null}`)
	assert.Equal(t, string(unknown), string(omitOptionalNulls(parameters, unknown)))
}
