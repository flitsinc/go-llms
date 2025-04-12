package tools

import (
	"encoding/json"
	"fmt"
	"reflect"
	"sync"
)

type Tool interface {
	// Label returns a nice human readable title for the tool.
	Label() string
	// Description returns the description of the tool.
	Description() string
	// FuncName returns the function name for the tool.
	FuncName() string
	// Run runs the tool with the provided parameters.
	Run(r Runner, params json.RawMessage) Result
	// Schema returns the JSON schema for the tool.
	Schema() *FunctionSchema
}

var jsonRawMessageType = reflect.TypeOf(json.RawMessage{})

// Func returns a tool for a function implementation with the given name and description.
func Func[Params any](label, description, funcName string, fn func(r Runner, params Params) Result) Tool {
	var zeroParams Params
	schemaType := reflect.TypeOf(zeroParams)
	if schemaType.Kind() != reflect.Struct && schemaType != jsonRawMessageType {
		panic("Params must be a struct or json.RawMessage")
	}
	var t *tool
	t = &tool{
		label:       label,
		description: description,
		schemaType:  schemaType,
		funcName:    funcName,
		fn: func(r Runner, params json.RawMessage) Result {
			if err := t.validateParams(params); err != nil {
				return ErrorWithLabel("LLM misbehaved", fmt.Errorf("validation error for %s: %w", funcName, err))
			}
			var p Params
			if err := json.Unmarshal(params, &p); err != nil {
				return ErrorWithLabel("LLM misbehaved", fmt.Errorf("unmarshal error for %s: %w", funcName, err))
			}
			return fn(r, p)
		},
	}
	return t
}

type tool struct {
	label, description, funcName string

	fn func(r Runner, params json.RawMessage) Result

	// Note: Lazily initialized.
	schema     *FunctionSchema
	schemaOnce sync.Once
	schemaType reflect.Type
}

func (t *tool) Label() string {
	return t.label
}

func (t *tool) Description() string {
	return t.description
}

func (t *tool) FuncName() string {
	return t.funcName
}

func (t *tool) Run(r Runner, params json.RawMessage) Result {
	return t.fn(r, params)
}

func (t *tool) Schema() *FunctionSchema {
	if t.schemaType == jsonRawMessageType {
		// We cannot generate schemas from json.RawMessage.
		return nil
	}
	t.schemaOnce.Do(func() {
		schema := generateSchema(t.funcName, t.description, t.schemaType)
		t.schema = &schema
	})
	return t.schema
}

func (t *tool) validateParams(params json.RawMessage) error {
	if t.schemaType == jsonRawMessageType {
		// All data is valid json.RawMessage data.
		return nil
	}
	return validateJSON(t.Schema(), params)
}
