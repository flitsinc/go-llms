package openai

import "github.com/flitsinc/go-llms/tools"

type Effort string

const (
	EffortLow     Effort = "low"
	EffortMedium  Effort = "medium"
	EffortHigh    Effort = "high"
	EffortMinimal Effort = "minimal"
)

const (
	ModelGPT5     = "gpt-5"
	ModelGPT5Mini = "gpt-5-mini"
	ModelGPT5Nano = "gpt-5-nano"
)

type Tool struct {
	Type     string                `json:"type"`
	Function *tools.FunctionSchema `json:"function,omitempty"`
	Custom   *CustomToolSchema     `json:"custom,omitempty"`
}

type CustomToolSchema struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description,omitempty"`
	InputSchema map[string]interface{} `json:"input_schema,omitempty"`
}
