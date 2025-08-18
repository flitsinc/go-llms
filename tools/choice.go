package tools

import (
	"encoding/json"
	"fmt"
)

type Choice struct {
	// Mode controls how the model may use tools.
	// Semantics:
	// - Any: The model may choose any available tool (or none).
	// - AllowOnly: The model may only choose from AllowedTools. If the list is empty, no tools may be used.
	// - RequireOneOf: The model must choose one of AllowedTools. If the list is empty, this effectively forbids tool use.
	Mode ChoiceMode `json:"mode"`
	// AllowedTools are tool function names that are permitted/required depending on Mode.
	AllowedTools []string `json:"allowed_tools,omitempty"`
}

// ChoiceMode enumerates tool choice policies.
type ChoiceMode string

const (
	// ChoiceAny lets the model decide whether to call tools and which one.
	ChoiceAny ChoiceMode = "any"
	// ChoiceAllowOnly restricts tool use to a subset (or none if empty).
	ChoiceAllowOnly ChoiceMode = "allow_only"
	// ChoiceRequireOneOf forces the model to use one of the provided tool names.
	ChoiceRequireOneOf ChoiceMode = "require_one_of"
)

// UnmarshalJSON parses string values for ChoiceMode.
func (m *ChoiceMode) UnmarshalJSON(data []byte) error {
	var s string
	if err := json.Unmarshal(data, &s); err != nil {
		return err
	}
	switch s {
	case string(ChoiceAny):
		*m = ChoiceAny
	case string(ChoiceAllowOnly):
		*m = ChoiceAllowOnly
	case string(ChoiceRequireOneOf):
		*m = ChoiceRequireOneOf
	default:
		return fmt.Errorf("invalid ChoiceMode: %q", s)
	}
	return nil
}

// AnyTool configures tool choice as fully open (model may pick any tool or none).
func AnyTool() Choice { return Choice{Mode: ChoiceAny} }

// AllowOnly configures a whitelist of allowed tools; empty means no tools can be used.
func AllowOnly(toolNames ...string) Choice {
	return Choice{Mode: ChoiceAllowOnly, AllowedTools: append([]string{}, toolNames...)}
}

// RequireOneOf configures a set of tools from which the model must pick one.
func RequireOneOf(toolNames ...string) Choice {
	return Choice{Mode: ChoiceRequireOneOf, AllowedTools: append([]string{}, toolNames...)}
}
