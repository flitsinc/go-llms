package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

type ResponsesAPI struct {
	accessToken string
	model       string
	endpoint    string
	company     string
	debug       bool

	maxOutputTokens    int
	reasoningEffort    Effort
	verbosity          Verbosity
	temperature        float64
	topP               float64
	topLogprobs        int
	parallelToolCalls  bool
	serviceTier        string
	store              bool
	truncation         string
	user               string
	metadata           map[string]string
	previousResponseID string
}

func NewResponsesAPI(accessToken, model string) *ResponsesAPI {
	return &ResponsesAPI{
		accessToken:       accessToken,
		model:             model,
		endpoint:          "https://api.openai.com/v1/responses",
		company:           "OpenAI",
		temperature:       1.0,
		topP:              1.0,
		parallelToolCalls: true,
		store:             true,
		truncation:        "disabled",
	}
}

func (m *ResponsesAPI) WithDebug() *ResponsesAPI {
	m.debug = true
	return m
}

// WithEndpoint sets the endpoint (and company name) so OpenAI-compatible API
// endpoints can be used.
func (m *ResponsesAPI) WithEndpoint(endpoint, company string) *ResponsesAPI {
	m.endpoint = endpoint
	m.company = company
	return m
}

func (m *ResponsesAPI) WithMaxOutputTokens(maxOutputTokens int) *ResponsesAPI {
	m.maxOutputTokens = maxOutputTokens
	return m
}

func (m *ResponsesAPI) WithThinking(effort Effort) *ResponsesAPI {
	m.reasoningEffort = effort
	return m
}

func (m *ResponsesAPI) WithTemperature(temperature float64) *ResponsesAPI {
	m.temperature = temperature
	return m
}

func (m *ResponsesAPI) WithTopP(topP float64) *ResponsesAPI {
	m.topP = topP
	return m
}

func (m *ResponsesAPI) WithTopLogprobs(topLogprobs int) *ResponsesAPI {
	m.topLogprobs = topLogprobs
	return m
}

func (m *ResponsesAPI) WithParallelToolCalls(parallel bool) *ResponsesAPI {
	m.parallelToolCalls = parallel
	return m
}

func (m *ResponsesAPI) WithServiceTier(tier string) *ResponsesAPI {
	m.serviceTier = tier
	return m
}

func (m *ResponsesAPI) WithStore(store bool) *ResponsesAPI {
	m.store = store
	return m
}

func (m *ResponsesAPI) WithTruncation(truncation string) *ResponsesAPI {
	m.truncation = truncation
	return m
}

func (m *ResponsesAPI) WithUser(user string) *ResponsesAPI {
	m.user = user
	return m
}

func (m *ResponsesAPI) WithMetadata(metadata map[string]string) *ResponsesAPI {
	m.metadata = metadata
	return m
}

func (m *ResponsesAPI) WithPreviousResponseID(id string) *ResponsesAPI {
	m.previousResponseID = id
	return m
}

func (m *ResponsesAPI) WithVerbosity(verbosity Verbosity) *ResponsesAPI {
	m.verbosity = verbosity
	return m
}

func (m *ResponsesAPI) Company() string {
	return m.company
}

func (m *ResponsesAPI) Model() string {
	return m.model
}

func (m *ResponsesAPI) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []llms.Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) llms.ProviderStream {
	// Build the input array
	var input []ResponseInput

	// Handle system prompt
	var instructions string
	// Check if system prompt is a single text item
	if text, ok := systemPrompt.AsString(); ok {
		// Single text item - use instructions field
		instructions = text
	} else {
		// Multiple items or non-text content - add as system message
		systemMsg := llms.Message{
			Role:    "system",
			Content: systemPrompt,
		}
		input = append(input, convertMessageToInput(systemMsg)...)
	}

	// Convert messages to input items
	for _, msg := range messages {
		input = append(input, convertMessageToInput(msg)...)
	}

	payload := map[string]any{
		"model":               m.model,
		"input":               input,
		"stream":              true,
		"temperature":         m.temperature,
		"top_p":               m.topP,
		"parallel_tool_calls": m.parallelToolCalls,
		"store":               m.store,
		"truncation":          m.truncation,
	}

	if instructions != "" {
		payload["instructions"] = instructions
	}

	if m.maxOutputTokens > 0 {
		payload["max_output_tokens"] = m.maxOutputTokens
	}

	if m.topLogprobs > 0 {
		payload["top_logprobs"] = m.topLogprobs
	}

	if m.reasoningEffort != "" {
		payload["reasoning"] = map[string]any{
			"effort":  m.reasoningEffort,
			"summary": "auto",
		}
	}

	// Set up .text related settings.
	text := map[string]any{}

	if m.verbosity != "" {
		text["verbosity"] = m.verbosity
	}

	// Handle JSON output schema
	if jsonOutputSchema != nil {
		text["format"] = TextResponseFormat{
			Type:   "json_schema",
			Name:   "structured_output",
			Schema: jsonOutputSchema,
			Strict: true,
		}
	}

	// Only include .text if there's anything configured.
	if len(text) > 0 {
		payload["text"] = text
	}

	if m.serviceTier != "" {
		payload["service_tier"] = m.serviceTier
	}

	if m.user != "" {
		payload["user"] = m.user
	}

	if m.metadata != nil {
		payload["metadata"] = m.metadata
	}

	if m.previousResponseID != "" {
		payload["previous_response_id"] = m.previousResponseID
	}

	// Handle tools
	if toolbox != nil {
		// Build tools for Responses API
		var toolsArr []any
		for _, t := range toolbox.All() {
			switch g := t.Grammar().(type) {
			case tools.JSONGrammar:
				if schema := g.Schema(); schema != nil {
					toolsArr = append(toolsArr, FunctionTool{
						Type:        "function",
						Name:        schema.Name,
						Description: schema.Description,
						Parameters:  &schema.Parameters,
						Strict:      true,
					})
				}
			case tools.TextGrammar:
				toolsArr = append(toolsArr, map[string]any{
					"type":        "custom",
					"name":        t.FuncName(),
					"description": t.Description(),
					"format":      map[string]any{"type": "text"},
				})
			case tools.LarkGrammar:
				toolsArr = append(toolsArr, map[string]any{
					"type":        "custom",
					"name":        t.FuncName(),
					"description": t.Description(),
					"format": map[string]any{
						"type":       "grammar",
						"definition": g.Definition,
						"syntax":     "lark",
					},
				})
			case tools.RegexGrammar:
				toolsArr = append(toolsArr, map[string]any{
					"type":        "custom",
					"name":        t.FuncName(),
					"description": t.Description(),
					"format": map[string]any{
						"type":       "grammar",
						"definition": g.Definition,
						"syntax":     "regex",
					},
				})
			default:
				panic(fmt.Sprintf("unsupported grammar type: %T", g))
			}
		}
		if len(toolsArr) > 0 {
			payload["tools"] = toolsArr
			payload["tool_choice"] = "auto"
		}
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return &ResponsesStream{err: fmt.Errorf("error encoding JSON: %w", err)}
	}

	if m.debug {
		fmt.Printf("\033[1;90m%s\033[0m\n", m.endpoint)
		fmt.Printf("-> \033[2;34m%s\033[0m\n", string(jsonData))
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return &ResponsesStream{err: fmt.Errorf("error creating request: %w", err)}
	}
	if m.accessToken != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", m.accessToken))
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return &ResponsesStream{err: fmt.Errorf("error making request: %w", err)}
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)

		if readErr == nil && len(bodyBytes) > 0 {
			var openAIError struct {
				Error struct {
					Message string `json:"message"`
					Type    string `json:"type"`
				} `json:"error"`
			}

			if jsonErr := json.Unmarshal(bodyBytes, &openAIError); jsonErr == nil && openAIError.Error.Message != "" {
				// Successfully parsed the OpenAI error format
				return &ResponsesStream{err: fmt.Errorf("%s: %s: %s", resp.Status, openAIError.Error.Type, openAIError.Error.Message)}
			}
			// Body read okay, but JSON parsing failed or structure mismatch.
			// Fall through to return status only.
		}
		// Default fallback: Read error, empty body, or failed/unexpected JSON parse.
		return &ResponsesStream{err: fmt.Errorf("%s", resp.Status)}
	}

	return &ResponsesStream{ctx: ctx, model: m.model, stream: resp.Body, debug: m.debug, lastThought: &content.Thought{}}
}

type ResponsesStream struct {
	ctx         context.Context
	model       string
	stream      io.Reader
	debug       bool
	err         error
	message     llms.Message
	lastText    string
	usage       *responsesUsage
	reasoning   string
	lastThought *content.Thought
}

func (s *ResponsesStream) Err() error {
	return s.err
}

func (s *ResponsesStream) Message() llms.Message {
	return s.message
}

func (s *ResponsesStream) Text() string {
	return s.lastText
}

func (s *ResponsesStream) ToolCall() llms.ToolCall {
	if len(s.message.ToolCalls) == 0 {
		return llms.ToolCall{}
	}
	return s.message.ToolCalls[len(s.message.ToolCalls)-1]
}

func (s *ResponsesStream) Thought() content.Thought {
	if s.lastThought != nil {
		return *s.lastThought
	}
	// The Responses API can stream thoughts/reasoning
	if s.reasoning != "" {
		return content.Thought{Text: s.reasoning}
	}
	return content.Thought{}
}

func (s *ResponsesStream) Usage() llms.Usage {
	if s.usage == nil {
		return llms.Usage{}
	}
	return llms.Usage{
		CachedInputTokens: s.usage.InputTokensDetails.CachedTokens,
		InputTokens:       s.usage.InputTokens,
		OutputTokens:      s.usage.OutputTokens,
	}
}

func (s *ResponsesStream) Iter() func(yield func(llms.StreamStatus) bool) {
	reader := bufio.NewReader(s.stream)
	var activeToolCall *llms.ToolCall

	return func(yield func(llms.StreamStatus) bool) {
		defer io.Copy(io.Discard, s.stream)
		for {
			select {
			case <-s.ctx.Done():
				s.err = s.ctx.Err()
				return
			default:
			}

			// Read a full logical line using ReadLine to support very long lines
			var lineBuilder strings.Builder
			for {
				part, isPrefix, err := reader.ReadLine()
				if err != nil {
					if err == io.EOF {
						// If we have accumulated partial data, process it before returning
						if lineBuilder.Len() == 0 {
							return
						}
						break
					}
					s.err = fmt.Errorf("error reading stream: %w", err)
					return
				}
				lineBuilder.Write(part)
				if !isPrefix {
					break
				}
			}

			rawLine := lineBuilder.String()
			line, ok := strings.CutPrefix(rawLine, "data: ")
			if !ok {
				continue
			}
			if line == "[DONE]" {
				if activeToolCall != nil {
					if !yield(llms.StreamStatusToolCallReady) {
						return
					}
					activeToolCall = nil
				}
				continue
			}

			if s.debug && strings.TrimSpace(line) != "" {
				fmt.Printf("<- \033[2;32m%s\033[0m\n", line)
			}

			var event ResponseStreamEvent
			if err := json.Unmarshal([]byte(line), &event); err != nil {
				s.err = fmt.Errorf("error unmarshalling chunk: %w", err)
				return
			}

			// Handle different chunk types
			switch event.Type {
			case "response.created":
				s.message.Role = "assistant"

			case "response.output_item.added":
				var item struct {
					Type string `json:"type"`
					ID   string `json:"id"`
				}
				if err := json.Unmarshal(event.Item, &item); err == nil {
					switch item.Type {
					case "function_call":
						if activeToolCall != nil {
							if !yield(llms.StreamStatusToolCallReady) {
								return
							}
						}
						var fc FunctionCall
						if err := json.Unmarshal(event.Item, &fc); err == nil {
							// The ID -> CallID, ExtraID -> ID mapping might
							// look weird, but it's because for most providers
							// the call ID is the main ID, but the Responses API
							// gives each item a unique ID too.
							llmToolCall := llms.ToolCall{
								ID:        fc.CallID,
								Name:      fc.Name,
								Arguments: json.RawMessage{},
								ExtraID:   fc.ID,
							}
							s.message.ToolCalls = append(s.message.ToolCalls, llmToolCall)
							activeToolCall = &s.message.ToolCalls[len(s.message.ToolCalls)-1]
							if !yield(llms.StreamStatusToolCallBegin) {
								return
							}
						}
					case "custom_tool_call":
						// Start of a custom tool call. Capture name and call_id so we can route to the correct tool.
						if activeToolCall != nil {
							if !yield(llms.StreamStatusToolCallReady) {
								return
							}
						}
						var ctc struct {
							Type   string `json:"type"`
							ID     string `json:"id"`
							Name   string `json:"name"`
							CallID string `json:"call_id"`
							Input  string `json:"input"`
						}
						if err := json.Unmarshal(event.Item, &ctc); err != nil {
							// Fallback to minimal fields if shape differs
							ctc.ID = item.ID
							ctc.Name = "custom"
						}
						llmToolCall := llms.ToolCall{
							ID:        ctc.CallID,
							Name:      ctc.Name,
							Arguments: json.RawMessage(ctc.Input),
							ExtraID:   ctc.ID,
						}
						s.message.ToolCalls = append(s.message.ToolCalls, llmToolCall)
						activeToolCall = &s.message.ToolCalls[len(s.message.ToolCalls)-1]
						if !yield(llms.StreamStatusToolCallBegin) {
							return
						}
					case "reasoning":
						// Create a new thought with ID and mark it as summary
						// The Responses API only sends reasoning summaries, not full traces
						s.lastThought = s.message.Content.AppendThoughtWithID(item.ID, "", true)
					}
				}

			case "response.output_text.delta":
				var delta struct {
					Delta string `json:"delta"`
				}
				if err := json.Unmarshal([]byte(line), &delta); err == nil {
					s.lastText = delta.Delta
					s.message.Content.Append(s.lastText)
					if s.lastText != "" {
						if !yield(llms.StreamStatusText) {
							return
						}
					}
				}

			case "response.function_call_arguments.delta":
				if activeToolCall != nil {
					var delta struct {
						Delta string `json:"delta"`
					}
					if err := json.Unmarshal([]byte(line), &delta); err == nil {
						activeToolCall.Arguments = append(activeToolCall.Arguments, []byte(delta.Delta)...)
						if !yield(llms.StreamStatusToolCallDelta) {
							return
						}
					}
				}

			case "response.function_call_arguments.done":
				if activeToolCall != nil {
					if !yield(llms.StreamStatusToolCallReady) {
						return
					}
					activeToolCall = nil
				}

				// Support custom tool calling input streaming using the events from the docs
				// https://platform.openai.com/docs/guides/function-calling#custom-tools
			case "response.custom_tool_call_input.delta":
				if activeToolCall != nil {
					var delta struct {
						Delta string `json:"delta"`
					}
					if err := json.Unmarshal([]byte(line), &delta); err == nil {
						// The delta is a plain string chunk; append
						activeToolCall.Arguments = append(activeToolCall.Arguments, []byte(delta.Delta)...)
						if !yield(llms.StreamStatusToolCallDelta) {
							return
						}
					}
				}

			case "response.custom_tool_call_input.done":
				if activeToolCall != nil {
					if !yield(llms.StreamStatusToolCallReady) {
						return
					}
					activeToolCall = nil
				}

			case "response.reasoning_summary_part.added":
				// This event contains initial text for a reasoning summary part
				var partEvent struct {
					Part struct {
						Type string `json:"type"`
						Text string `json:"text"`
					} `json:"part"`
					ItemID string `json:"item_id"`
				}
				if err := json.Unmarshal([]byte(line), &partEvent); err == nil && partEvent.Part.Text != "" {
					// Append to the thought with matching ID (creates if needed)
					s.lastThought = s.message.Content.AppendThoughtWithID(partEvent.ItemID, partEvent.Part.Text, true)
					if !yield(llms.StreamStatusThinking) {
						return
					}
				}

			case "response.reasoning_summary_part.done":
				// This event contains the complete text for a reasoning summary part
				var partEvent struct {
					Part struct {
						Type string `json:"type"`
						Text string `json:"text"`
					} `json:"part"`
					ItemID string `json:"item_id"`
				}
				if err := json.Unmarshal([]byte(line), &partEvent); err == nil {
					// The done event contains the complete text, but we've been streaming it
					// via delta events, so we don't need to append it again
				}

			case "response.reasoning_summary_text.delta":
				var deltaEvent struct {
					Delta  string `json:"delta"`
					ItemID string `json:"item_id"`
				}
				if err := json.Unmarshal([]byte(line), &deltaEvent); err == nil && deltaEvent.Delta != "" {
					// Append to the thought with matching ID (creates if needed)
					s.lastThought = s.message.Content.AppendThoughtWithID(deltaEvent.ItemID, deltaEvent.Delta, true)
					if !yield(llms.StreamStatusThinking) {
						return
					}
				}

			case "response.reasoning_summary_text.done":
				// The thought is already marked as complete summary, just reset our tracking
				var doneEvent struct {
					Text   string `json:"text"`
					ItemID string `json:"item_id"`
				}
				if err := json.Unmarshal([]byte(line), &doneEvent); err == nil {
					// The done event contains the complete text
					// For summaries, we could validate or replace, but we've been building it correctly
					s.lastThought = nil // Reset for next reasoning item
				}

			case "response.usage":
				if event.Usage != nil {
					s.usage = event.Usage
				}

			case "response.completed":
				if activeToolCall != nil {
					if !yield(llms.StreamStatusToolCallReady) {
						return
					}
					activeToolCall = nil
				}
				return

			case "error":
				if event.Error != nil {
					s.err = fmt.Errorf("stream error (%s): %s", event.Error.Code, event.Error.Message)
				}
				return
			}
		}
	}
}

// convertMessageToInput converts an llms.Message to ResponseInput items
func convertMessageToInput(msg llms.Message) []ResponseInput {
	var items []ResponseInput

	switch msg.Role {
	case "user", "system", "developer":
		content := convertContentToInputContent(msg.Content)
		items = append(items, InputMessage{
			Type:    "message",
			Role:    msg.Role,
			Content: content, // can be empty slice
		})

	case "assistant":
		// An assistant message from history can have both content and tool calls.
		// First, check if there are any thoughts with IDs that need to be included as reasoning items
		var hasReasoningWithID bool
		var reasoningID string
		reasoningSummary := []ReasoningSummary{}

		// Check for thoughts with IDs that need to be included as separate reasoning items
		for _, item := range msg.Content {
			if thought, ok := item.(*content.Thought); ok && thought.ID != "" {
				hasReasoningWithID = true
				reasoningID = thought.ID
				// Include a summary part only if we have text; an empty summary is still valid
				if thought.Text != "" {
					reasoningSummary = append(reasoningSummary, ReasoningSummary{
						Type: "summary_text",
						Text: thought.Text,
					})
				}
				// Only include the first reasoning item for now.
				// The Responses API requires the prior turn's reasoning item to precede
				// the replayed tool call, and that association typically refers to the
				// first reasoning item (output_index 0). We don't yet track mappings
				// between multiple reasoning IDs and specific tool calls; if we add that,
				// we should replay all reasoning items in order or map them per call.
				break
			}
		}

		// Convert content to input content (thoughts will be skipped)
		content := convertContentToInputContent(msg.Content)

		// Only include an assistant message if there's non-empty content
		if len(content) > 0 {
			items = append(items, InputMessage{
				Type:    "message",
				Role:    "assistant",
				Content: content,
			})
		}

		// If we found a thought with an ID, add it as a reasoning item before any tool calls.
		// The API requires the reasoning item to be present even if the summary is empty.
		if hasReasoningWithID {
			items = append(items, Reasoning{
				Type:    "reasoning",
				ID:      reasoningID,
				Summary: reasoningSummary,
			})
		}

		// Then add any tool calls
		for _, tc := range msg.ToolCalls {
			// If ExtraID looks like a custom tool call id (ctc_), send as custom_tool_call
			if strings.HasPrefix(tc.ExtraID, "ctc_") {
				items = append(items, CustomToolCall{
					Type:   "custom_tool_call",
					ID:     tc.ExtraID,
					Name:   tc.Name,
					Input:  string(tc.Arguments),
					CallID: tc.ID,
				})
				continue
			}
			// Otherwise, treat as a JSON function_call
			items = append(items, FunctionCall{
				Type:      "function_call",
				ID:        tc.ExtraID, // The item id is stored as an extra ID on the tool call.
				Name:      tc.Name,
				Arguments: string(tc.Arguments),
				CallID:    tc.ID,
			})
		}

	case "tool":
		// A tool result message.
		var outputStr string
		if len(msg.Content) > 0 {
			switch v := msg.Content[0].(type) {
			case *content.Text:
				outputStr = v.Text
			case *content.JSON:
				outputStr = string(v.Data)
			}
		}

		// Tool results can also have auxiliary content that should be sent
		// from the user role.
		if len(msg.Content) > 1 {
			secondaryContent := convertContentToInputContent(msg.Content[1:])
			if len(secondaryContent) > 0 {
				items = append(items, InputMessage{
					Type:    "message",
					Role:    "user",
					Content: secondaryContent,
				})
			}
		}

		items = append(items, FunctionCallOutput{
			Type:   "function_call_output",
			CallID: msg.ToolCallID,
			Output: outputStr,
		})
	}
	return items
}

// convertContentToInputContent converts content.Content to InputContent array
func convertContentToInputContent(c content.Content) []InputContent {
	var inputContent []InputContent

	for _, item := range c {
		switch v := item.(type) {
		case *content.Text:
			inputContent = append(inputContent, InputText{
				Type: "input_text",
				Text: v.Text,
			})
		case *content.ImageURL:
			inputContent = append(inputContent, InputImage{
				Type:     "input_image",
				ImageURL: v.URL,
				Detail:   "auto",
			})
		case *content.JSON:
			inputContent = append(inputContent, InputText{
				Type: "input_text",
				Text: string(v.Data),
			})
		case *content.Thought:
			// Skip thoughts in input
		case *content.CacheHint:
			// Skip cache hints
		}
	}

	return inputContent
}
