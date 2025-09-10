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
	debugger    llms.Debugger

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
	promptCacheKey     string

	specialTools []ResponseTool
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

func (m *ResponsesAPI) WithTool(tool ResponseTool) *ResponsesAPI {
	m.specialTools = append(m.specialTools, tool)
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

func (m *ResponsesAPI) WithPromptCacheKey(key string) *ResponsesAPI {
	m.promptCacheKey = key
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

func (m *ResponsesAPI) SetDebugger(d llms.Debugger) {
	m.debugger = d
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

	if m.reasoningEffort == "" {
		payload["reasoning"] = map[string]any{
			"summary": "auto",
		}
	} else {
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

	if m.promptCacheKey != "" {
		payload["prompt_cache_key"] = m.promptCacheKey
	}

	// Handle tools
	if toolbox != nil {
		// Build tools for Responses API
		var toolsArr []any
		for _, t := range m.specialTools {
			toolsArr = append(toolsArr, t)
		}
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
			// Include all tools by default for cacheability.
			payload["tools"] = toolsArr
			// Map tool choice using explicit allowed_tools object when constraining,
			// which Responses API supports.
			choice := toolbox.Choice
			switch choice.Mode {
			case tools.ChoiceAllowOnly:
				if len(choice.AllowedTools) == 0 {
					payload["tool_choice"] = "none"
				} else {
					// Validate that at least one of the allowed tools exists in toolsArr.
					exists := false
					for _, n := range choice.AllowedTools {
						for _, it := range toolsArr {
							switch v := it.(type) {
							case FunctionTool:
								if v.Name == n {
									exists = true
								}
							case map[string]any:
								if name, ok := v["name"].(string); ok && name == n {
									exists = true
								}
							}
							if exists {
								break
							}
						}
						if exists {
							break
						}
					}
					if !exists {
						return &ResponsesStream{err: fmt.Errorf("openai responses: no allowed tools found in toolbox")}
					}
					// Use allowed_tools with mode:auto
					var allowedEntries []any
					for _, n := range choice.AllowedTools {
						allowedEntries = append(allowedEntries, map[string]any{"type": "function", "name": n})
					}
					payload["tool_choice"] = AllowedToolsToolChoice{Type: "allowed_tools", Mode: "auto", Tools: allowedEntries}
				}
			case tools.ChoiceRequireOneOf:
				switch len(choice.AllowedTools) {
				case 0:
					payload["tool_choice"] = "none"
				case 1:
					// Force single tool by name; validate it exists.
					name := choice.AllowedTools[0]
					exists := false
					for _, it := range toolsArr {
						switch v := it.(type) {
						case FunctionTool:
							if v.Name == name {
								exists = true
							}
						case map[string]any:
							if n, ok := v["name"].(string); ok && n == name {
								exists = true
							}
						}
						if exists {
							break
						}
					}
					if !exists {
						return &ResponsesStream{err: fmt.Errorf("openai responses: required tool %q not found in toolbox", name)}
					}
					payload["tool_choice"] = map[string]any{
						"type": "function",
						"name": name,
					}
				default:
					// Multiple allowed: use allowed_tools with mode:required
					// Validate at least one exists.
					exists := false
					for _, n := range choice.AllowedTools {
						for _, it := range toolsArr {
							switch v := it.(type) {
							case FunctionTool:
								if v.Name == n {
									exists = true
								}
							case map[string]any:
								if name, ok := v["name"].(string); ok && name == n {
									exists = true
								}
							}
							if exists {
								break
							}
						}
						if exists {
							break
						}
					}
					if !exists {
						return &ResponsesStream{err: fmt.Errorf("openai responses: none of the required tools are present in toolbox")}
					}
					var allowedEntries []any
					for _, n := range choice.AllowedTools {
						allowedEntries = append(allowedEntries, map[string]any{"type": "function", "name": n})
					}
					payload["tool_choice"] = AllowedToolsToolChoice{Type: "allowed_tools", Mode: "required", Tools: allowedEntries}
				}
			default:
				payload["tool_choice"] = "auto"
			}
		}
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return &ResponsesStream{err: fmt.Errorf("error encoding JSON: %w", err)}
	}

	if m.debugger != nil {
		m.debugger.RawRequest(m.endpoint, jsonData)
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

	return &ResponsesStream{ctx: ctx, model: m.model, stream: resp.Body, debugger: m.debugger, lastThought: &content.Thought{}}
}

type ResponsesStream struct {
	ctx         context.Context
	model       string
	stream      io.Reader
	debugger    llms.Debugger
	err         error
	message     llms.Message
	lastText    string
	lastImage   struct{ URL, MIME string }
	usage       *responsesUsage
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

func (s *ResponsesStream) Image() (string, string) {
	return s.lastImage.URL, s.lastImage.MIME
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

			if s.debugger != nil && strings.TrimSpace(line) != "" {
				s.debugger.RawEvent([]byte(line))
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
				// TODO: We may want to handle other types like "image_generation_call" here.
				var item struct {
					Type string `json:"type"`
					ID   string `json:"id"`
				}
				if err := json.Unmarshal(event.Item, &item); err == nil {
					switch item.Type {
					case "message":
						// Capture the assistant message ID so it can be replayed later
						s.message.ID = item.ID
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
						// Initialize an aggregated thought in message content; lastThought will carry only deltas
						s.message.Content.AppendThoughtWithID(item.ID, "", true)
						s.lastThought = &content.Thought{ID: item.ID, Summary: true}
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
					// Update aggregated thought
					s.message.Content.AppendThoughtWithID(partEvent.ItemID, partEvent.Part.Text, true)
					// Set lastThought delta
					s.lastThought = &content.Thought{ID: partEvent.ItemID, Text: partEvent.Part.Text, Summary: true}
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
					// Update aggregated thought
					s.message.Content.AppendThoughtWithID(deltaEvent.ItemID, deltaEvent.Delta, true)
					// Set lastThought delta
					s.lastThought = &content.Thought{ID: deltaEvent.ItemID, Text: deltaEvent.Delta, Summary: true}
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
					if !yield(llms.StreamStatusThinkingDone) {
						return
					}
				}

			case "response.output_item.done":
				// Images are finalized on output_item.done for image_generation_call
				var itemHdr struct {
					Type   string `json:"type"`
					Status string `json:"status"`
				}
				if err := json.Unmarshal(event.Item, &itemHdr); err == nil {
					if itemHdr.Type == "image_generation_call" && itemHdr.Status == "completed" {
						var img struct {
							ID            string `json:"id"`
							Background    string `json:"background"`
							OutputFormat  string `json:"output_format"`
							Quality       string `json:"quality"`
							Result        string `json:"result"`
							RevisedPrompt string `json:"revised_prompt"`
							Size          string `json:"size"`
						}
						err := json.Unmarshal(event.Item, &img)
						if err != nil {
							s.err = fmt.Errorf("failed to parse image generation result: %w", s.err)
							return
						}
						if img.Result == "" {
							continue
						}
						mime := "image/png"
						switch img.OutputFormat {
						case "png":
							mime = "image/png"
						case "jpeg", "jpg":
							mime = "image/jpeg"
						case "webp":
							mime = "image/webp"
						case "gif":
							mime = "image/gif"
						}
						dataURI := content.BuildDataURI(mime, img.Result)
						s.lastImage.URL = dataURI
						s.lastImage.MIME = mime
						// Add to aggregated message content for history
						s.message.Content = append(s.message.Content, &content.ImageURL{URL: dataURI, MimeType: mime})
						if !yield(llms.StreamStatusImage) {
							return
						}
						if event.Usage != nil {
							s.usage = event.Usage
						}
					}
				}

			case "response.image_generation_call.in_progress":
			case "response.image_generation_call.generating":
			case "response.image_generation_call.partial_image":
				// TODO: Support streaming partial image frames during generation.
				// For now, ignore partial frames; we'll only emit on completed.

			case "response.completed":
				if event.Response != nil {
					var response struct {
						Usage *responsesUsage `json:"usage"`
					}
					if err := json.Unmarshal(event.Response, &response); err == nil && response.Usage != nil {
						s.usage = response.Usage
					}
				}

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
		// Replaying assistant messages into Responses API must use output item types,
		// not input content types. Build an output message with output_text parts.
		var outParts []OutputContent
		var hasReasoningWithID bool
		var reasoningID string
		reasoningSummary := []ReasoningSummary{}

		for _, item := range msg.Content {
			switch v := item.(type) {
			case *content.Text:
				outParts = append(outParts, OutputText{Type: "output_text", Text: v.Text})
			case *content.JSON:
				outParts = append(outParts, OutputText{Type: "output_text", Text: string(v.Data)})
			case *content.Thought:
				// Only include the first reasoning item for now.
				// The Responses API requires the prior turn's reasoning item to precede
				// the replayed tool call, and that association typically refers to the
				// first reasoning item (output_index 0). We don't yet track mappings
				// between multiple reasoning IDs and specific tool calls; if we add that,
				// we should replay all reasoning items in order or map them per call.
				if !hasReasoningWithID && v.ID != "" {
					hasReasoningWithID = true
					reasoningID = v.ID
					if v.Text != "" {
						reasoningSummary = append(reasoningSummary, ReasoningSummary{Type: "summary_text", Text: v.Text})
					}
				}
			}
		}

		if len(outParts) > 0 {
			items = append(items, OutputMessage{Type: "message", ID: msg.ID, Role: "assistant", Content: outParts})
		}

		if hasReasoningWithID {
			items = append(items, Reasoning{Type: "reasoning", ID: reasoningID, Summary: reasoningSummary})
		}

		// Then add any tool calls. The item id is stored as an extra ID on the tool call.
		for _, tc := range msg.ToolCalls {
			// If ExtraID looks like a custom tool call id (ctc_), send as custom_tool_call.
			if strings.HasPrefix(tc.ExtraID, "ctc_") {
				items = append(items, CustomToolCall{Type: "custom_tool_call", ID: tc.ExtraID, Name: tc.Name, Input: string(tc.Arguments), CallID: tc.ID})
				continue
			}
			// Otherwise, treat as a JSON function_call.
			items = append(items, FunctionCall{Type: "function_call", ID: tc.ExtraID, Name: tc.Name, Arguments: string(tc.Arguments), CallID: tc.ID})
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
