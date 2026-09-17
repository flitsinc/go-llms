package typesafe

import (
	"encoding/json"
	"errors"
	"fmt"

	"github.com/metalim/jsonmap"

	"github.com/flitsinc/go-llms/tools"
)

// ErrUnsupportedSchema is wrapped by every error about an output schema the
// provider cannot turn into questions. A System One model answers questions;
// it does not generate values, so the schema has to be flat and every property
// has to be a question with a bounded answer.
var ErrUnsupportedSchema = errors.New("typesafe: unsupported output schema")

// questionKind records how a schema property maps to a question and back, so
// the answer can be rendered in the property's declared type.
type questionKind uint8

const (
	// kindNoulBoolean is a boolean property: a Noul whose answer is rendered
	// as true when the probability of yes is at least 0.5.
	kindNoulBoolean questionKind = iota
	// kindNoulNumber is a number property: a Noul whose answer is rendered as
	// the raw probability of yes, for callers that threshold themselves.
	kindNoulNumber
	// kindChoice is a string property with an enum: a Choice over the enum
	// members, rendered as the chosen member.
	kindChoice
)

type boundQuestion struct {
	kind     questionKind
	question Question
}

// questionsFromSchema turns a flat JSON object schema into one question per
// property. The property description is the question's instructions and is
// required, because the property name is never sent to the model.
//
//   - boolean            → Noul, rendered as true/false
//   - number             → Noul, rendered as the probability of yes
//   - string with enum   → Choice over the enum members
//
// Anything else (nested objects, arrays, free strings, integers) is rejected
// with [ErrUnsupportedSchema], because the model cannot produce it.
func questionsFromSchema(schema *tools.ValueSchema) (*jsonmap.Map, map[string]Question, error) {
	if schema == nil {
		return nil, nil, fmt.Errorf("%w: a JSON output schema is required; questions are derived from its properties", ErrUnsupportedSchema)
	}
	if schema.Type != "object" || schema.Properties == nil || schema.Properties.Len() == 0 {
		return nil, nil, fmt.Errorf("%w: the root must be an object with at least one property", ErrUnsupportedSchema)
	}

	bound := jsonmap.New()
	questions := make(map[string]Question, schema.Properties.Len())
	for el := schema.Properties.First(); el != nil; el = el.Next() {
		name := el.Key()
		property, err := propertySchema(el.Value())
		if err != nil {
			return nil, nil, fmt.Errorf("%w: property %q: %w", ErrUnsupportedSchema, name, err)
		}
		bq, err := questionFromProperty(property)
		if err != nil {
			return nil, nil, fmt.Errorf("%w: property %q: %w", ErrUnsupportedSchema, name, err)
		}
		bound.Set(name, bq)
		questions[name] = bq.question
	}
	return bound, questions, nil
}

// propertySchema normalizes a property value into a ValueSchema. The
// properties map holds a ValueSchema when built in Go and a decoded JSON
// object when the schema arrived over the wire, so the value is re-encoded
// rather than type-switched.
func propertySchema(raw any) (tools.ValueSchema, error) {
	if vs, ok := raw.(tools.ValueSchema); ok {
		return vs, nil
	}
	if vs, ok := raw.(*tools.ValueSchema); ok && vs != nil {
		return *vs, nil
	}
	data, err := json.Marshal(raw)
	if err != nil {
		return tools.ValueSchema{}, fmt.Errorf("encoding property schema: %w", err)
	}
	var vs tools.ValueSchema
	if err := json.Unmarshal(data, &vs); err != nil {
		return tools.ValueSchema{}, fmt.Errorf("decoding property schema: %w", err)
	}
	return vs, nil
}

func questionFromProperty(property tools.ValueSchema) (boundQuestion, error) {
	if property.Description == "" {
		return boundQuestion{}, errors.New("a description is required; it is the question sent to the model")
	}
	switch property.Type {
	case "boolean":
		return boundQuestion{kind: kindNoulBoolean, question: Question{Type: "noul", Instructions: property.Description}}, nil
	case "number":
		return boundQuestion{kind: kindNoulNumber, question: Question{Type: "noul", Instructions: property.Description}}, nil
	case "string":
		if len(property.Enum) == 0 {
			return boundQuestion{}, errors.New("a string property needs an enum; the model selects, it does not generate")
		}
		criteria := make(map[string]*string, len(property.Enum))
		for _, member := range property.Enum {
			option, ok := member.(string)
			if !ok {
				return boundQuestion{}, fmt.Errorf("enum member %v is not a string", member)
			}
			criteria[option] = nil
		}
		return boundQuestion{kind: kindChoice, question: Question{Type: "choice", Instructions: property.Description, Criteria: criteria}}, nil
	default:
		return boundQuestion{}, fmt.Errorf("type %q cannot be answered by a System One model; use boolean, number, or a string enum", property.Type)
	}
}

// renderAnswers builds the JSON object the caller's schema describes from the
// answers, in the schema's property order.
func renderAnswers(bound *jsonmap.Map, answers map[string]Answer) ([]byte, error) {
	out := jsonmap.New()
	for el := bound.First(); el != nil; el = el.Next() {
		name := el.Key()
		bq := el.Value().(boundQuestion)
		answer, ok := answers[name]
		if !ok {
			return nil, fmt.Errorf("typesafe: response has no answer for %q", name)
		}
		switch bq.kind {
		case kindNoulBoolean, kindNoulNumber:
			if answer.Type != "noul" || answer.Noul == nil {
				return nil, fmt.Errorf("typesafe: answer for %q is not a noul", name)
			}
			if bq.kind == kindNoulBoolean {
				out.Set(name, *answer.Noul >= 0.5)
			} else {
				out.Set(name, *answer.Noul)
			}
		case kindChoice:
			if answer.Type != "choice" || answer.Choice == "" {
				return nil, fmt.Errorf("typesafe: answer for %q is not a choice", name)
			}
			out.Set(name, answer.Choice)
		}
	}
	return json.Marshal(out)
}
