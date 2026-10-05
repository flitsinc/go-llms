package openai

import (
	"fmt"

	"github.com/flitsinc/go-llms/tools"
)

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
	compaction        *ContextCompaction
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

	// Ask for each reasoning item's encrypted content so the history can
	// replay reasoning with it. A reasoning item replayed by ID alone only
	// resolves in the organization that stored it, so that history breaks
	// when the API key moves to another organization.
	payload["include"] = []string{"reasoning.encrypted_content"}

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

	if c.compaction != nil {
		contextManagement, err := c.compaction.contextManagement()
		if err != nil {
			return nil, err
		}
		payload["context_management"] = contextManagement
	}

	if c.promptCacheKey != "" {
		payload["prompt_cache_key"] = c.promptCacheKey
	}

	if toolbox != nil || len(c.specialTools) > 0 {
		toolsArr, err := buildResponsesToolsArray(c.specialTools, toolbox)
		if err != nil {
			return nil, err
		}
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

// ContextCompaction configures OpenAI Responses server-side compaction: once
// the rendered input reaches TriggerInputTokens, the API compacts the context
// and emits an encrypted "compaction" output item, surfaced as a
// content.Compaction in the assistant message, which replaces the history
// before it on later requests.
// https://developers.openai.com/api/docs/guides/compaction
type ContextCompaction struct {
	// TriggerInputTokens is the input size that triggers compaction
	// (compact_threshold). Required: the API documents no default.
	TriggerInputTokens int
}

func (c *ContextCompaction) contextManagement() ([]map[string]any, error) {
	if c.TriggerInputTokens <= 0 {
		return nil, fmt.Errorf("compaction requires a positive TriggerInputTokens, got %d", c.TriggerInputTokens)
	}
	return []map[string]any{{"type": "compaction", "compact_threshold": c.TriggerInputTokens}}, nil
}
