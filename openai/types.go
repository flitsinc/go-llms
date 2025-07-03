package openai

import "github.com/flitsinc/go-llms/tools"

type Effort string

const (
	EffortLow    Effort = "low"
	EffortMedium Effort = "medium"
	EffortHigh   Effort = "high"
)

type Tool struct {
	Type     string               `json:"type"`
	Function tools.FunctionSchema `json:"function"`
}
