// Package geminischema narrows JSON Schema tool declarations to the subset
// Google's function-calling API accepts.
//
// Gemini rejects a request outright (HTTP 400) when a function declaration
// carries JSON Schema keywords it does not implement — most notably
// "additionalProperties", but also "const", "pattern", "exclusiveMinimum" and
// the rest of the validation vocabulary. Callers that target OpenAI's strict
// mode emit exactly those keywords, so their toolbox is unusable against
// Gemini until it is narrowed here.
//
// This lives in its own package because Gemini is reachable by two different
// transports: the native API in the google package, and OpenAI-compatible
// gateways such as OpenRouter in the openai package. Both must narrow schemas
// identically, and a single implementation is the only way that stays true.
package geminischema

import (
	"encoding/json"

	"github.com/metalim/jsonmap"

	"github.com/flitsinc/go-llms/tools"
)

// SanitizeFunction returns a copy of schema whose parameters are safe to send
// to Gemini. The input is never modified, so the caller's toolbox stays intact
// for other providers sharing it.
func SanitizeFunction(schema tools.FunctionSchema) tools.FunctionSchema {
	schema.Parameters = SanitizeValue(schema.Parameters)
	return schema
}

// SanitizeValue returns a deep copy of schema with the keywords Gemini rejects
// removed:
//
//   - "additionalProperties" is dropped at every level.
//   - Every other keyword outside tools.ValueSchema is dropped by round-tripping
//     nested property schemas through ValueSchema, which only has fields Gemini
//     understands.
func SanitizeValue(schema tools.ValueSchema) tools.ValueSchema {
	schema.AdditionalProperties = nil

	if schema.Items != nil {
		items := SanitizeValue(*schema.Items)
		schema.Items = &items
	}

	if schema.Required != nil {
		schema.Required = append([]string(nil), schema.Required...)
	}

	if schema.AnyOf != nil {
		anyOf := make([]tools.ValueSchema, len(schema.AnyOf))
		for i, alternative := range schema.AnyOf {
			anyOf[i] = SanitizeValue(alternative)
		}
		schema.AnyOf = anyOf
	}

	if schema.Properties != nil {
		properties := jsonmap.New()
		for _, key := range schema.Properties.Keys() {
			raw, ok := schema.Properties.Get(key)
			if !ok {
				continue
			}
			value, ok := asValueSchema(raw)
			if !ok {
				// Not schema-shaped, so there is nothing to narrow and no safe
				// way to rewrite it. Carry it over untouched rather than drop a
				// property the model is expected to fill.
				properties.Set(key, raw)
				continue
			}
			properties.Set(key, SanitizeValue(value))
		}
		schema.Properties = properties
	}

	return schema
}

// asValueSchema reinterprets one property entry as a ValueSchema. Properties
// arrive as decoded JSON rather than typed values whenever the schema came off
// the wire, so the concrete type varies by caller.
func asValueSchema(raw any) (tools.ValueSchema, bool) {
	switch value := raw.(type) {
	case tools.ValueSchema:
		return value, true
	case *tools.ValueSchema:
		if value == nil {
			return tools.ValueSchema{}, false
		}
		return *value, true
	case json.RawMessage:
		var schema tools.ValueSchema
		if json.Unmarshal(value, &schema) != nil {
			return tools.ValueSchema{}, false
		}
		return schema, true
	case *jsonmap.Map, map[string]any:
		data, err := json.Marshal(value)
		if err != nil {
			return tools.ValueSchema{}, false
		}
		var schema tools.ValueSchema
		if json.Unmarshal(data, &schema) != nil {
			return tools.ValueSchema{}, false
		}
		return schema, true
	default:
		return tools.ValueSchema{}, false
	}
}
