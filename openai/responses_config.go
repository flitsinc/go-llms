package openai

import "github.com/flitsinc/go-llms/tools"

// responsesConfig holds configuration fields shared by ResponsesAPI and
// WebSocketResponsesAPI. Both provider types embed this struct.
type responsesConfig struct {
	model             string
	temperature       float64
	topP              *float64
	maxOutputTokens   int
	topLogprobs       int
	reasoningEffort   Effort
	verbosity         Verbosity
	parallelToolCalls bool
	serviceTier       string
	store             bool
	truncation        string
	user              string
	metadata          map[string]string
	promptCacheKey    string
	specialTools      []ResponseTool
}

// buildResponsesPayload builds the API payload map from config fields, input,
// instructions, tools, and JSON output schema. Callers add transport-specific
// fields (e.g. "stream":true for SSE, "type":"response.create" for WS) and
// previousResponseID.
func (c *responsesConfig) buildResponsesPayload(
	input []ResponseInput,
	instructions string,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) (map[string]any, error) {
	payload := map[string]any{
		"model":               c.model,
		"temperature":         c.temperature,
		"parallel_tool_calls": c.parallelToolCalls,
		"store":               c.store,
		"truncation":          c.truncation,
	}

	if input != nil {
		payload["input"] = input
	}

	if c.topP != nil {
		payload["top_p"] = *c.topP
	}

	if instructions != "" {
		payload["instructions"] = instructions
	}

	if c.maxOutputTokens > 0 {
		payload["max_output_tokens"] = c.maxOutputTokens
	}

	if c.topLogprobs > 0 {
		payload["top_logprobs"] = c.topLogprobs
	}

	if c.reasoningEffort == "" {
		payload["reasoning"] = map[string]any{
			"summary": "auto",
		}
	} else {
		payload["reasoning"] = map[string]any{
			"effort":  c.reasoningEffort,
			"summary": "auto",
		}
	}

	// Set up .text related settings.
	text := map[string]any{}

	if c.verbosity != "" {
		text["verbosity"] = c.verbosity
	}

	if jsonOutputSchema != nil {
		text["format"] = TextResponseFormat{
			Type:   "json_schema",
			Name:   "structured_output",
			Schema: jsonOutputSchema,
			Strict: true,
		}
	}

	if len(text) > 0 {
		payload["text"] = text
	}

	if c.serviceTier != "" {
		payload["service_tier"] = c.serviceTier
	}

	if c.user != "" {
		payload["user"] = c.user
	}

	if c.metadata != nil {
		payload["metadata"] = c.metadata
	}

	if c.promptCacheKey != "" {
		payload["prompt_cache_key"] = c.promptCacheKey
	}

	if toolbox != nil || len(c.specialTools) > 0 {
		toolsArr := buildResponsesToolsArray(c.specialTools, toolbox)
		if len(toolsArr) > 0 {
			payload["tools"] = toolsArr
			if toolbox != nil {
				tc, err := buildToolChoice(toolbox.Choice, toolsArr)
				if err != nil {
					return nil, err
				}
				payload["tool_choice"] = tc
			}
		}
	}

	return payload, nil
}
