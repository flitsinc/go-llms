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
	httpClient  *http.Client

	maxOutputTokens    int
	reasoningEffort    Effort
	verbosity          Verbosity
	temperature        float64
	topP               *float64
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
	m.topP = &topP
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

func (m *ResponsesAPI) SetHTTPClient(client *http.Client) {
	m.httpClient = client
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
		systemInputs, err := convertMessageToInput(systemMsg)
		if err != nil {
			return newResponsesStreamError(fmt.Errorf("responses: failed to convert system message: %w", err))
		}
		input = append(input, systemInputs...)
	}

	// Convert messages to input items
	for _, msg := range messages {
		msgInputs, err := convertMessageToInput(msg)
		if err != nil {
			return newResponsesStreamError(fmt.Errorf("responses: failed to convert message role=%s: %w", msg.Role, err))
		}
		input = append(input, msgInputs...)
	}

	payload := map[string]any{
		"model":               m.model,
		"input":               input,
		"stream":              true,
		"temperature":         m.temperature,
		"parallel_tool_calls": m.parallelToolCalls,
		"store":               m.store,
		"truncation":          m.truncation,
	}

	if m.topP != nil {
		payload["top_p"] = *m.topP
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
						return newResponsesStreamError(fmt.Errorf("openai responses: no allowed tools found in toolbox"))
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
						return newResponsesStreamError(fmt.Errorf("openai responses: required tool %q not found in toolbox", name))
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
						return newResponsesStreamError(fmt.Errorf("openai responses: none of the required tools are present in toolbox"))
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
		return newResponsesStreamError(fmt.Errorf("error encoding JSON: %w", err))
	}

	if m.debugger != nil {
		m.debugger.RawRequest(m.endpoint, jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return newResponsesStreamError(fmt.Errorf("error creating request: %w", err))
	}
	if m.accessToken != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", m.accessToken))
	}
	req.Header.Set("Content-Type", "application/json")

	client := m.httpClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(req)
	if err != nil {
		return newResponsesStreamError(fmt.Errorf("error making request: %w", err))
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
				return newResponsesStreamError(&llms.HTTPError{
					StatusCode: resp.StatusCode,
					Status:     resp.Status,
					ErrorType:  openAIError.Error.Type,
					Message:    openAIError.Error.Message,
				})
			}
		}
		return newResponsesStreamError(&llms.HTTPError{
			StatusCode: resp.StatusCode,
			Status:     resp.Status,
		})
	}

	return &ResponsesStream{
		responsesEventProcessor: responsesEventProcessor{
			debugger:    m.debugger,
			lastThought: &content.Thought{},
		},
		ctx:    ctx,
		model:  m.model,
		stream: resp.Body,
	}
}

type ResponsesStream struct {
	responsesEventProcessor // shared event processing state
	ctx    context.Context
	model  string
	stream io.Reader
}

func newResponsesStreamError(err error) *ResponsesStream {
	return &ResponsesStream{responsesEventProcessor: responsesEventProcessor{err: err}}
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
	if s.err != nil {
		return func(yield func(llms.StreamStatus) bool) {}
	}
	reader := bufio.NewReader(s.stream)

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
				if s.activeToolCall != nil {
					if !yield(llms.StreamStatusToolCallReady) {
						return
					}
					s.activeToolCall = nil
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

			if s.processEvent(event, []byte(line), yield) {
				return
			}
		}
	}
}

// convertMessageToInput converts an llms.Message to ResponseInput items
func convertMessageToInput(msg llms.Message) ([]ResponseInput, error) {
	var items []ResponseInput

	switch msg.Role {
	case "user", "system", "developer":
		content := convertContentToInputContent(msg.Content)
		items = append(items, InputMessage{
			Type:    "message",
			Role:    msg.Role,
			Content: content, // can be empty slice
		})
		return items, nil

	case "assistant":
		seenReasoningIDs := map[string]bool{}
		var pendingOutParts []OutputContent
		firstOutputMessage := true

		flushOutput := func() {
			if len(pendingOutParts) == 0 {
				return
			}
			msgID := ""
			if firstOutputMessage {
				msgID = msg.ID
				firstOutputMessage = false
			}
			items = append(items, OutputMessage{Type: "message", ID: msgID, Role: "assistant", Content: pendingOutParts})
			pendingOutParts = nil
		}

		for _, item := range msg.Content {
			switch v := item.(type) {
			case *content.Text:
				pendingOutParts = append(pendingOutParts, OutputText{Type: "output_text", Text: v.Text})
			case *content.JSON:
				pendingOutParts = append(pendingOutParts, OutputText{Type: "output_text", Text: string(v.Data)})
			case *content.Thought:
				if v.ID == "" || seenReasoningIDs[v.ID] {
					continue
				}
				flushOutput()
				reasoning := Reasoning{Type: "reasoning", ID: v.ID, Summary: []ReasoningSummary{}}
				if v.Text != "" {
					reasoning.Summary = append(reasoning.Summary, ReasoningSummary{Type: "summary_text", Text: v.Text})
				}
				items = append(items, reasoning)
				seenReasoningIDs[v.ID] = true
			}
		}

		flushOutput()

		for _, tc := range msg.ToolCalls {
			itemID := tc.Metadata["openai:item_id"]
			if itemID == "" {
				return nil, fmt.Errorf("tool call %q is missing openai:item_id metadata for replay", tc.ID)
			}
			itemType := tc.Metadata["openai:item_type"]
			var toolItem ResponseInput
			switch itemType {
			case "custom_tool_call":
				toolItem = CustomToolCall{Type: "custom_tool_call", ID: itemID, Name: tc.Name, Input: string(tc.Arguments), CallID: tc.ID}
			default:
				toolItem = FunctionCall{Type: "function_call", ID: itemID, Name: tc.Name, Arguments: string(tc.Arguments), CallID: tc.ID}
			}
			items = append(items, toolItem)
		}

		return items, nil

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
		return items, nil
	}

	return nil, fmt.Errorf("unsupported role %q", msg.Role)
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
