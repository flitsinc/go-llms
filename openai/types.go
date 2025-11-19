package openai

import "github.com/flitsinc/go-llms/tools"

type Effort string

const (
	EffortMinimal Effort = "minimal"
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
	EffortXHigh   Effort = "xhigh"
)

type Verbosity string

const (
	VerbosityLow    Verbosity = "low"
	VerbosityMedium Verbosity = "medium"
	VerbosityHigh   Verbosity = "high"
)

type Tool struct {
	Type     string                `json:"type"`
	Function *tools.FunctionSchema `json:"function,omitempty"`
	Custom   *CustomToolSchema     `json:"custom,omitempty"`
}

type CustomToolSchema struct {
	Name        string         `json:"name"`
	Description string         `json:"description,omitempty"`
	Format      map[string]any `json:"format,omitempty"`
}

// ChatToolChoice for Chat Completions API
// https://platform.openai.com/docs/guides/function-calling#additional-configurations
// Accepts: "none" | "auto" | {"type":"function","function":{"name":string}} | {"type":"tool","name":string}
type ChatToolChoice struct {
	Type     string              `json:"type"` // "function" or "tool"; for string modes, send string directly
	Function *ChatToolChoiceFunc `json:"function,omitempty"`
	Name     string              `json:"name,omitempty"`
}

type ChatToolChoiceFunc struct {
	Name string `json:"name"`
}

type ChatAllowedToolsChoice struct {
	Type  string            `json:"type"` // "allowed_tools"
	Mode  string            `json:"mode"` // "auto" | "required"
	Tools []ChatAllowedTool `json:"tools"`
}

type ChatAllowedTool struct {
	Type     string                 `json:"type"` // "function" | "custom"
	Function *ChatAllowedToolFunc   `json:"function,omitempty"`
	Custom   *ChatAllowedToolCustom `json:"custom,omitempty"`
}

type ChatAllowedToolFunc struct {
	Name string `json:"name"`
}

type ChatAllowedToolCustom struct {
	Name string `json:"name"`
}
