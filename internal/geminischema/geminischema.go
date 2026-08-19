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
// to Gemini:
//
//   - "additionalProperties" is dropped at every level.
//   - Every other keyword outside tools.ValueSchema is dropped, because
//     ValueSchema only has fields Gemini understands.
//
// The input is never modified, so a toolbox shared with other providers keeps
// its full schema.
func SanitizeFunction(schema tools.FunctionSchema) tools.FunctionSchema {
	schema.Parameters = sanitizeValue(schema.Parameters)
	return schema
}

func sanitizeValue(schema tools.ValueSchema) tools.ValueSchema {
	schema.AdditionalProperties = nil

	if schema.Items != nil {
		items := sanitizeValue(*schema.Items)
		schema.Items = &items
	}

	if schema.AnyOf != nil {
		anyOf := make([]tools.ValueSchema, len(schema.AnyOf))
		for i, alternative := range schema.AnyOf {
			anyOf[i] = sanitizeValue(alternative)
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
				// way to rewrite it. Carrying it over keeps a property the
				// model needs, at the cost of Gemini still refusing it — which
				// is what would happen anyway.
				properties.Set(key, raw)
				continue
			}
			properties.Set(key, sanitizeValue(value))
		}
		schema.Properties = properties
	}

	return schema
}

// asValueSchema reinterprets one property entry as a ValueSchema.
//
// Properties are typed as `any` (jsonmap preserves declaration order but not
// types), so the concrete type depends on how the schema was built: decoding
// one off the wire yields *jsonmap.Map, while tools.Func builds ValueSchema
// directly. Round-tripping through JSON handles every shape without a type
// switch that would have to guess at the full set, and it is the same
// marshalling the request itself performs a moment later.
func asValueSchema(raw any) (tools.ValueSchema, bool) {
	data, err := json.Marshal(raw)
	if err != nil {
		return tools.ValueSchema{}, false
	}
	var schema tools.ValueSchema
	if json.Unmarshal(data, &schema) != nil {
		return tools.ValueSchema{}, false
	}
	return schema, true
}
