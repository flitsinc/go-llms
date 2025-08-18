package tools

type Choice struct {
	// Mode controls how the model may use tools.
	// Semantics:
	// - Any: The model may choose any available tool (or none).
	// - AllowOnly: The model may only choose from AllowedTools. If the list is empty, no tools may be used.
	// - RequireOneOf: The model must choose one of AllowedTools. If the list is empty, this effectively forbids tool use.
	Mode ChoiceMode
	// AllowedTools are tool function names that are permitted/required depending on Mode.
	AllowedTools []string
}

// ChoiceMode enumerates tool choice policies.
type ChoiceMode int

const (
	// ChoiceAny lets the model decide whether to call tools and which one.
	ChoiceAny ChoiceMode = iota
	// ChoiceAllowOnly restricts tool use to a subset (or none if empty).
	ChoiceAllowOnly
	// ChoiceRequireOneOf forces the model to use one of the provided tool names.
	ChoiceRequireOneOf
)

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
