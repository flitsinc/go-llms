package tools

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
)

// FunctionSchema defines the structure for a function tool that an LLM can call.
// It aligns with the format expected by many LLM providers for function calling.
// Based on a subset of the JSON Schema specification.
type FunctionSchema struct {
	// Name is the name of the function to be called.
	Name string `json:"name"`
	// Description is a description of what the function does.
	Description string `json:"description"`
	// Parameters is the schema for the arguments object that the function expects.
	Parameters ValueSchema `json:"parameters"`
}

// ValueSchema represents a schema for a value within the function's parameters.
// It corresponds to a subset of the JSON Schema specification, defining the type
// and constraints of a parameter field or the parameter object itself.
type ValueSchema struct {
	// Type specifies the data type of the value (e.g., "string", "integer", "object", "array", "boolean", "number").
	Type string `json:"type,omitempty"`
	// Description provides a brief explanation of the value or field.
	Description string `json:"description,omitempty"`
	// Items defines the schema for elements within an array. Only used when Type is "array".
	Items *ValueSchema `json:"items,omitempty"`
	// Properties defines the schema for properties within an object. Only used when Type is "object".
	// Note: We use a pointer to the map here to differentiate "no map" from "empty map".
	// See: https://github.com/golang/go/issues/22480
	Properties *map[string]ValueSchema `json:"properties,omitempty"`
	// AdditionalProperties specifies the schema for additional properties in an object, or allows/disallows them.
	// It can be a boolean (true to allow any, false to disallow) or a ValueSchema defining the type of allowed additional properties.
	// Only used when Type is "object".
	AdditionalProperties any `json:"additionalProperties,omitempty"`
	// Required lists the names of properties that must be present when Type is "object".
	Required []string `json:"required,omitempty"`
	// AnyOf specifies that the value must conform to at least one of the provided schemas.
	AnyOf []ValueSchema `json:"anyOf,omitempty"`
}

// generateSchema initializes and returns the main structure of a function's JSON Schema
func generateSchema(name, description string, typ reflect.Type) FunctionSchema {
	parameters := generateObjectSchema(typ)
	return FunctionSchema{
		Name:        name,
		Description: description,
		Parameters:  parameters,
	}
}

// fieldTypeToJSONSchema maps Go data types to corresponding JSON Schema properties consistently
func fieldTypeToJSONSchema(t reflect.Type) ValueSchema {
	switch t.Kind() {
	case reflect.String:
		return ValueSchema{Type: "string"}
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64,
		reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		return ValueSchema{Type: "integer"}
	case reflect.Bool:
		return ValueSchema{Type: "boolean"}
	case reflect.Float32, reflect.Float64:
		return ValueSchema{Type: "number"}
	case reflect.Slice, reflect.Array:
		itemSchema := fieldTypeToJSONSchema(t.Elem())
		return ValueSchema{Type: "array", Items: &itemSchema}
	case reflect.Map:
		additionalPropertiesSchema := fieldTypeToJSONSchema(t.Elem())
		return ValueSchema{Type: "object", AdditionalProperties: additionalPropertiesSchema}
	case reflect.Struct:
		return generateObjectSchema(t)
	case reflect.Ptr:
		return fieldTypeToJSONSchema(t.Elem())
	default:
		panic("unsupported type: " + t.Kind().String())
	}
}

// generateObjectSchema constructs a JSON Schema for structs
func generateObjectSchema(typ reflect.Type) ValueSchema {
	properties := make(map[string]ValueSchema)
	required := []string{}

	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		if !field.IsExported() { // Skip unexported fields
			continue
		}
		jsonTag := field.Tag.Get("json")
		if jsonTag == "-" { // Field is explicitly ignored
			continue
		}
		parts := strings.Split(jsonTag, ",")
		fieldName := field.Name
		if parts[0] != "" {
			fieldName = parts[0]
		}

		fieldSchema := fieldTypeToJSONSchema(field.Type)
		if description := field.Tag.Get("description"); description != "" {
			fieldSchema.Description = description
		}
		properties[fieldName] = fieldSchema
		if len(parts) == 1 || (len(parts) > 1 && parts[1] != "omitempty") {
			required = append(required, fieldName)
		}
	}
	return ValueSchema{
		Type:       "object",
		Properties: &properties,
		Required:   required,
	}
}

// validateJSON checks if jsonData conforms to the structure defined in the schema from generateSchema
func validateJSON(schema *FunctionSchema, jsonData json.RawMessage) error {
	return validateParameters(schema.Parameters, jsonData)
}

// validateParameters validates JSON data against the provided parameters schema
func validateParameters(schema ValueSchema, jsonData json.RawMessage) error {
	if schema.Type != "object" || schema.Properties == nil {
		return errors.New("schema error: received an invalid object schema")
	}

	var dataMap map[string]any
	if err := json.Unmarshal(jsonData, &dataMap); err != nil {
		return errors.New("invalid JSON format")
	}

	for key, val := range dataMap {
		fieldSchema, found := (*schema.Properties)[key]
		if found {
			// Validate known property
			if err := validateField(fieldSchema, val); err != nil {
				return fmt.Errorf("field \"%s\": %w", key, err)
			}
			continue
		}

		// Handle additional property based on schema.AdditionalProperties
		switch ap := schema.AdditionalProperties.(type) {
		case nil:
			// Default behavior: If AdditionalProperties is not specified or nil, allow extra fields.
			continue
		case bool:
			if !ap {
				return fmt.Errorf("additional property %q not allowed", key)
			}
			// If ap is true, allow the property.
			continue
		case ValueSchema:
			// Validate against the provided schema for additional properties
			if err := validateField(ap, val); err != nil {
				return fmt.Errorf("additional property %q: %w", key, err)
			}
		default:
			return fmt.Errorf("invalid schema: AdditionalProperties has unexpected type %T", schema.AdditionalProperties)
		}
	}

	for _, field := range schema.Required {
		if _, exists := dataMap[field]; !exists {
			return fmt.Errorf("missing required field: %q", field)
		}
	}

	return nil
}

// validateField checks a single field against its schema
func validateField(fieldSchema ValueSchema, data any) error {
	// Check AnyOf first
	if len(fieldSchema.AnyOf) > 0 {
		valid := false
		for _, subSchema := range fieldSchema.AnyOf {
			if err := validateField(subSchema, data); err == nil {
				valid = true
				break
			}
		}
		if valid {
			return nil // Valid against at least one schema in AnyOf
		}
		// If none matched, return a generic error or potentially the last error encountered
		// TODO: Improve error message for AnyOf failure?
		return fmt.Errorf("data does not match any of the schemas in anyOf")
	}

	dataType := fieldSchema.Type
	if dataType == "" {
		// If type is empty and AnyOf is not used, consider it an error or permissive?
		// For now, let's assume it should match based on inferred type if possible, or error if ambiguous.
		// This part might need refinement based on exact JSON Schema spec interpretation for empty type.
		// Let's treat it as an error for stricter validation initially.
		return fmt.Errorf("schema type is missing and anyOf is not specified")
	}

	switch dataType {
	case "integer":
		num, ok := data.(float64)
		if !ok || num != float64(int(num)) {
			return fmt.Errorf("type mismatch: expected integer, got %T", data)
		}
	case "number":
		if _, ok := data.(float64); !ok {
			return fmt.Errorf("type mismatch: expected number, got %T", data)
		}
	case "string":
		if _, ok := data.(string); !ok {
			return fmt.Errorf("type mismatch: expected string, got %T", data)
		}
	case "boolean":
		if _, ok := data.(bool); !ok {
			return fmt.Errorf("type mismatch: expected boolean, got %T", data)
		}
	case "array":
		items, ok := data.([]any)
		if !ok {
			return fmt.Errorf("type mismatch: expected array, got %T", data)
		}
		if fieldSchema.Items == nil {
			return errors.New("schema error: missing item schema for array")
		}
		itemSchema := *fieldSchema.Items
		for _, item := range items {
			if err := validateField(itemSchema, item); err != nil {
				return err
			}
		}
	case "object":
		properties, ok := data.(map[string]any)
		if !ok {
			return fmt.Errorf("type mismatch: expected object, got %T", data)
		}
		jsonData, err := json.Marshal(properties)
		if err != nil {
			return errors.New("failed to marshal object data for validation")
		}
		return validateParameters(fieldSchema, json.RawMessage(jsonData))
	default:
		return fmt.Errorf("unsupported type: %s", dataType)
	}
	return nil
}
