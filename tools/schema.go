package tools

import (
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"strings"
)

type FunctionSchema struct {
	Name        string      `json:"name"`
	Description string      `json:"description"`
	Parameters  ValueSchema `json:"parameters"`
}

type ValueSchema struct {
	Type        string       `json:"type"`
	Description string       `json:"description,omitempty"`
	Items       *ValueSchema `json:"items,omitempty"`
	// Note: We use a pointer to the map here to differentiate "no map" from "empty map".
	// See: https://github.com/golang/go/issues/22480
	Properties           *map[string]ValueSchema `json:"properties,omitempty"`
	AdditionalProperties *ValueSchema            `json:"additionalProperties,omitempty"`
	Required             []string                `json:"required,omitempty"`
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
		return ValueSchema{Type: "object", AdditionalProperties: &additionalPropertiesSchema}
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
		if field.PkgPath != "" { // Skip unexported fields
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
		if !found {
			continue // Ignoring extra fields
		}
		if err := validateField(fieldSchema, val); err != nil {
			return fmt.Errorf("%s: %w", key, err)
		}
	}

	for _, field := range schema.Required {
		if _, exists := dataMap[field]; !exists {
			return fmt.Errorf("missing required field: %s", field)
		}
	}

	return nil
}

// validateField checks a single field against its schema
func validateField(fieldSchema ValueSchema, data any) error {
	dataType := fieldSchema.Type

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
