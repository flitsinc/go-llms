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

// binding is one schema property, bound to the question it is sent as and to
// the function that renders the model's answer back into the property's
// declared type. Bindings are kept in schema order, which is the order the
// rendered object's keys come out in.
type binding struct {
	name     string
	question Question
	required bool
	render   func(Answer) (any, error)
}

// questionsFromSchema turns a flat JSON object schema into one binding per
// property, in schema order. The property description is the question's
// instructions and is required, because the property name is never sent to the
// model.
//
//   - boolean            → Noul, rendered as true when the probability is ≥ 0.5
//   - number             → Noul, rendered as the probability itself, 0 to 1
//   - string with enum   → Choice over the enum members, rendered as the member
//
// Anything else (nested objects, arrays, free strings, integers) is rejected
// with [ErrUnsupportedSchema], because the model cannot produce it.
func questionsFromSchema(schema *tools.ValueSchema) ([]binding, error) {
	if schema == nil {
		return nil, fmt.Errorf("%w: a JSON output schema is required; questions are derived from its properties", ErrUnsupportedSchema)
	}
	if schema.Type != "object" || schema.Properties == nil || schema.Properties.Len() == 0 {
		return nil, fmt.Errorf("%w: the root must be an object with at least one property", ErrUnsupportedSchema)
	}

	required := make(map[string]bool, len(schema.Required))
	for _, name := range schema.Required {
		required[name] = true
	}

	bindings := make([]binding, 0, schema.Properties.Len())
	for el := schema.Properties.First(); el != nil; el = el.Next() {
		name := el.Key()
		property, err := tools.PropertySchema(el.Value())
		if err != nil {
			return nil, fmt.Errorf("%w: property %q: %w", ErrUnsupportedSchema, name, err)
		}
		b, err := bindProperty(name, property, required[name])
		if err != nil {
			return nil, fmt.Errorf("%w: property %q: %w", ErrUnsupportedSchema, name, err)
		}
		bindings = append(bindings, b)
	}
	return bindings, nil
}

func bindProperty(name string, property tools.ValueSchema, required bool) (binding, error) {
	if property.Description == "" {
		return binding{}, errors.New("a description is required; it is the question sent to the model")
	}
	b := binding{name: name, required: required}
	switch property.Type {
	case "boolean":
		if len(property.Enum) > 0 {
			return binding{}, errors.New("a boolean property cannot have an enum; it is asked as a yes/no question")
		}
		b.question = Question{Type: "noul", Instructions: property.Description}
		b.render = func(answer Answer) (any, error) {
			noul, err := noulAnswer(name, answer)
			if err != nil {
				return nil, err
			}
			return noul >= 0.5, nil
		}
	case "number":
		if len(property.Enum) > 0 {
			return binding{}, errors.New("a number property cannot have an enum; it is read as the probability that its description holds")
		}
		b.question = Question{Type: "noul", Instructions: property.Description}
		b.render = func(answer Answer) (any, error) {
			return noulAnswer(name, answer)
		}
	case "string":
		if len(property.Enum) == 0 {
			return binding{}, errors.New("a string property needs an enum; the model selects, it does not generate")
		}
		criteria := make(map[string]*string, len(property.Enum))
		for _, member := range property.Enum {
			option, ok := member.(string)
			if !ok {
				return binding{}, fmt.Errorf("enum member %v is not a string", member)
			}
			criteria[option] = nil
		}
		b.question = Question{Type: "choice", Instructions: property.Description, Criteria: criteria}
		b.render = func(answer Answer) (any, error) {
			if answer.Type != "choice" || answer.Choice == "" {
				return nil, fmt.Errorf("typesafe: answer for %q is not a choice", name)
			}
			if _, ok := criteria[answer.Choice]; !ok {
				return nil, fmt.Errorf("typesafe: answer for %q is %q, which is not one of the declared options", name, answer.Choice)
			}
			return answer.Choice, nil
		}
	default:
		return binding{}, fmt.Errorf("type %q cannot be answered by a System One model; use boolean, number, or a string enum", property.Type)
	}
	return b, nil
}

func noulAnswer(name string, answer Answer) (float64, error) {
	if answer.Type != "noul" || answer.Noul == nil {
		return 0, fmt.Errorf("typesafe: answer for %q is not a noul", name)
	}
	return *answer.Noul, nil
}

// renderAnswers builds the JSON object the caller's schema describes, in the
// schema's property order. A required property without an answer is an error;
// an optional one is left out. Answers for names that were never asked are
// ignored.
func renderAnswers(bindings []binding, answers map[string]Answer) ([]byte, error) {
	out := jsonmap.New()
	for _, b := range bindings {
		answer, ok := answers[b.name]
		if !ok {
			if b.required {
				return nil, fmt.Errorf("typesafe: response has no answer for %q", b.name)
			}
			continue
		}
		value, err := b.render(answer)
		if err != nil {
			return nil, err
		}
		out.Set(b.name, value)
	}
	return json.Marshal(out)
}
