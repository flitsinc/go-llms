package openai

import "github.com/flitsinc/go-llms/tools"

type Effort string

const (
	EffortMinimal Effort = "minimal"
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
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
