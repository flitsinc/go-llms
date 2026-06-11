package openai

import (
	"encoding/json"

	"github.com/flitsinc/go-llms/tools"
)

// errorWrappedToolOutput encodes a failed tool result for OpenAI APIs, which
// have no native error flag on tool outputs. Plain text is wrapped in the
// conventional {"error": ...} JSON payload; JSON payloads pass through
// unchanged because error-producing callers (e.g. tools.Error) already encode
// an "error" key themselves.
func errorWrappedToolOutput(output string, outputIsJSON bool) string {
	if outputIsJSON {
		return output
	}
	wrapped, _ := json.Marshal(map[string]string{"error": output})
	return string(wrapped)
}

type Effort string

const (
	// EffortNone disables reasoning entirely. The GPT-5.x reasoning
	// family (5.2, 5.4, 5.5, …) accepts it; the older o-series /
	// gpt-5 generation expose EffortMinimal as their lightest tier
	// instead. Which one a given model accepts is documented on its
	// model page; OpenAI returns a 400 if you send the wrong one.
	EffortNone    Effort = "none"
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
