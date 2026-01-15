package anthropic

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

const (
	jsonModeToolName = "json_output" // Name for the simulated tool in JSON mode
)

type Model struct {
	apiKey            string
	model             string
	endpoint          string
	company           string
	debugger          llms.Debugger
	maxTokens         int
	maxThinkingTokens int
	betaFeatures      []string
	httpClient        *http.Client
}

func New(apiKey, model string) *Model {
	return &Model{
		apiKey:    apiKey,
		model:     model,
		endpoint:  "https://api.anthropic.com/v1/messages",
		company:   "Anthropic",
		maxTokens: 1024,
	}
}

// WithBeta adds an Anthropic beta feature flag to the request.
// Multiple beta features can be added by calling this method multiple times.
func (m *Model) WithBeta(betaFeature string) *Model {
	m.betaFeatures = append(m.betaFeatures, betaFeature)
	return m
}

// WithEndpoint sets the endpoint (and company name) so Anthropic-compatible
// endpoints can be used.
func (m *Model) WithEndpoint(endpoint, company string) *Model {
	m.endpoint = endpoint
	m.company = company
	return m
}

func (m *Model) WithMaxTokens(maxTokens int) *Model {
	m.maxTokens = maxTokens
	return m
}

func (m *Model) WithThinking(budgetTokens int) *Model {
	m.maxThinkingTokens = budgetTokens
	return m
}

func (m *Model) Company() string {
	return m.company
}

func (m *Model) Model() string {
	return m.model
}

func (m *Model) SetDebugger(d llms.Debugger) {
	m.debugger = d
}

func (m *Model) SetHTTPClient(client *http.Client) {
	m.httpClient = client
}

func (m *Model) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []llms.Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) llms.ProviderStream {
	var apiMessages []message
	for _, msg := range messages {
		apiMessages = append(apiMessages, messageFromLLM(msg))
	}

	// Use a boolean flag to track if JSON mode is active for stream processing
	isJSONMode := jsonOutputSchema != nil

	payload := map[string]any{
		"model":    m.model,
		"messages": apiMessages,
		"stream":   true,
		// We make an opinionated choice here to calculate thinking tokens on
		// top of the max output tokens.
		"max_tokens": m.maxTokens + m.maxThinkingTokens,
	}

	if systemPrompt != nil {
		payload["system"] = contentFromLLM(systemPrompt)
	}

	if isJSONMode {
		// Simulate JSON mode using a forced tool
		jsonModeTool := Tool{
			Name:        jsonModeToolName,
			Description: "This tool receives your response in a specific JSON format.",
			InputSchema: *jsonOutputSchema, // Use the provided schema
		}
		payload["tools"] = []Tool{jsonModeTool}
		payload["tool_choice"] = ToolChoice{Type: "tool", Name: jsonModeToolName}
	} else if toolbox != nil {
		// Build full tool list first.
		allTools := Tools(toolbox)
		choice := toolbox.Choice

		// Map Choice to Anthropic tool_choice.
		// Anthropic does NOT support an allow-list in tool_choice; it supports:
		// - type: "auto" (model may or may not call a tool)
		// - type: "any" (model must call one of the provided tools)
		// - type: "tool", name: "..." (force a particular tool)
		// - type: "none" (disallow any tool use)
		// Therefore, whenever we must restrict the set of usable tools (AllowOnly or RequireOneOf with multiple),
		// we FILTER the tools array to the allowed subset, because Anthropic cannot take an allowed list separately.
		var toolChoice any
		switch choice.Mode {
		case tools.ChoiceAllowOnly:
			if len(choice.AllowedTools) == 0 {
				// Explicitly disallow tool use for this call; keep the tools list intact for cacheability.
				toolChoice = ToolChoice{Type: "none"}
			} else {
				// Filter to allowed tools; error if none of the allowed tools exist.
				allowed := map[string]bool{}
				for _, name := range choice.AllowedTools {
					allowed[name] = true
				}
				filtered := make([]Tool, 0, len(allTools))
				for _, t := range allTools {
					if allowed[t.Name] {
						filtered = append(filtered, t)
					}
				}
				if len(filtered) == 0 {
					return &Stream{err: fmt.Errorf("anthropic: no allowed tools found in toolbox")}
				}
				allTools = filtered
				toolChoice = ToolChoice{Type: "auto"}
			}
		case tools.ChoiceRequireOneOf:
			switch len(choice.AllowedTools) {
			case 0:
				// Require one-of with empty list means no tools may be used.
				toolChoice = ToolChoice{Type: "none"}
			case 1:
				// Force the single allowed tool by name. Validate that it exists; otherwise return an error.
				name := choice.AllowedTools[0]
				exists := false
				for _, t := range allTools {
					if t.Name == name {
						exists = true
						break
					}
				}
				if !exists {
					return &Stream{err: fmt.Errorf("anthropic: required tool %q not found in toolbox", name)}
				}
				toolChoice = ToolChoice{Type: "tool", Name: name}
				// Note: No need to filter here; forcing a specific tool is supported by Anthropic.
			default:
				// Multiple acceptable tools: filter tools list down to the allowed subset,
				// then set tool_choice to any (must use one of provided).
				allowed := map[string]bool{}
				for _, name := range choice.AllowedTools {
					allowed[name] = true
				}
				filtered := make([]Tool, 0, len(allTools))
				for _, t := range allTools {
					if allowed[t.Name] {
						filtered = append(filtered, t)
					}
				}
				if len(filtered) == 0 {
					return &Stream{err: fmt.Errorf("anthropic: none of the required tools are present in toolbox")}
				}
				allTools = filtered
				toolChoice = ToolChoice{Type: "any"}
			}
		default:
			// ChoiceAny: let the model decide.
			toolChoice = ToolChoice{Type: "auto"}
		}

		payload["tools"] = allTools
		payload["tool_choice"] = toolChoice
	}

	// Note: Anthropic does not support thinking when forcing tool use (which we
	// do to simulate JSON mode from other providers).
	if m.maxThinkingTokens > 0 && !isJSONMode {
		payload["thinking"] = map[string]any{
			"type":          "enabled",
			"budget_tokens": m.maxThinkingTokens,
		}
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return &Stream{err: fmt.Errorf("error encoding JSON: %w", err)}
	}

	if m.debugger != nil {
		m.debugger.RawRequest(m.endpoint, jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return &Stream{err: fmt.Errorf("error creating request: %w", err)}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", m.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	// Add beta feature headers if any are configured
	for _, beta := range m.betaFeatures {
		req.Header.Add("anthropic-beta", beta)
	}

	client := m.httpClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(req)
	if err != nil {
		return &Stream{err: fmt.Errorf("error making request: %w", err)}
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)
		if readErr == nil && len(bodyBytes) > 0 {
			var anthropicErr struct {
				Type  string `json:"type"`
				Error struct {
					Type    string `json:"type"`
					Message string `json:"message"`
				} `json:"error"`
			}
			if jsonErr := json.Unmarshal(bodyBytes, &anthropicErr); jsonErr == nil && anthropicErr.Type == "error" {
				// Successfully parsed the Anthropic error format
				return &Stream{err: &llms.HTTPError{
					StatusCode: resp.StatusCode,
					Status:     resp.Status,
					ErrorType:  anthropicErr.Error.Type,
					Message:    anthropicErr.Error.Message,
				}}
			}
			// Body read okay, but JSON parsing failed or structure mismatch.
			// Fall through to return status only.
		}
		// Default fallback: Read error, empty body, or failed/unexpected JSON parse.
		return &Stream{err: &llms.HTTPError{
			StatusCode: resp.StatusCode,
			Status:     resp.Status,
		}}
	}

	return &Stream{ctx: ctx, model: m.model, stream: resp.Body, isJSONMode: isJSONMode, debugger: m.debugger}
}

type Stream struct {
	ctx         context.Context
	model       string
	stream      io.Reader
	err         error
	message     llms.Message
	lastText    string
	lastThought *content.Thought
	isJSONMode  bool // Flag to indicate if JSON mode was used for generation
	debugger    llms.Debugger

	cachedInputTokens, cacheCreationInputTokens, inputTokens, outputTokens int
}

func (s *Stream) Err() error {
	return s.err
}

func (s *Stream) Message() llms.Message {
	return s.message
}

func (s *Stream) Text() string {
	return s.lastText
}

func (s *Stream) Image() (string, string) {
	// Anthropic doesn't generate any images as of this writing.
	return "", ""
}

func (s *Stream) Thought() content.Thought {
	if s.lastThought != nil {
		return *s.lastThought
	}
	return content.Thought{}
}

func (s *Stream) ToolCall() llms.ToolCall {
	if len(s.message.ToolCalls) == 0 {
		return llms.ToolCall{}
	}
	return s.message.ToolCalls[len(s.message.ToolCalls)-1]
}

func (s *Stream) Usage() llms.Usage {
	return llms.Usage{
		CachedInputTokens:        s.cachedInputTokens,
		CacheCreationInputTokens: s.cacheCreationInputTokens,
		InputTokens:              s.inputTokens,
		OutputTokens:             s.outputTokens,
	}
}

func (s *Stream) Iter() func(yield func(llms.StreamStatus) bool) {
	scanner := bufio.NewScanner(s.stream)
	return func(yield func(llms.StreamStatus) bool) {
		defer io.Copy(io.Discard, s.stream)
		lastToolCallIndex := -1
		var handlingJsonModeTool bool // Flag if we are inside the JSON mode tool block
		var resetNextArgumentsDelta bool
		// Track content block types by index so we can signal when a thinking block ends
		contentBlockTypeByIndex := map[int]string{}
		// The Anthropic SSE stream follows this pattern:
		// 1. message_start - contains initial message metadata
		// 2. For each content block:
		//    a. content_block_start - signals the beginning of a content block
		//    b. content_block_delta - streams content updates (potentially many)
		//    c. content_block_stop - signals the end of a content block
		// 3. message_delta - contains updates to the message object
		// 4. message_stop - signals the end of the stream
		//
		// There may also be:
		// - ping events throughout (no action needed)
		// - error events (should abort with error)
		for {
			select {
			case <-s.ctx.Done():
				s.err = s.ctx.Err()
				return
			default:
				// Context OK, keep scanning.
			}
			if !scanner.Scan() {
				if err := scanner.Err(); err != nil {
					s.err = fmt.Errorf("error scanning stream: %w", err)
				}
				return
			}

			if s.debugger != nil && strings.TrimSpace(scanner.Text()) != "" {
				s.debugger.RawEvent([]byte(scanner.Text()))
			}

			line, ok := strings.CutPrefix(scanner.Text(), "data: ")
			if !ok {
				continue
			}
			var event streamEvent
			if err := json.Unmarshal([]byte(line), &event); err != nil {
				s.err = fmt.Errorf("error unmarshalling event: %w", err)
				return
			}

			switch event.Type {
			case "message_start":
				// Initialize the message with the role from the message_start event
				s.message.Role = event.Message.Role
				if event.Message.ID != "" {
					s.message.ID = event.Message.ID
					if !yield(llms.StreamStatusMessageStart) {
						return
					}
				}
				if u := event.Message.Usage; u != nil {
					// Values are cumulative, so we overwrite the numbers instead of adding.
					// https://docs.anthropic.com/en/docs/build-with-claude/streaming
					if u.CacheReadInputTokens != nil {
						s.cachedInputTokens = *u.CacheReadInputTokens
					}
					if u.CacheCreationInputTokens != nil {
						s.cacheCreationInputTokens = *u.CacheCreationInputTokens
					}
					if u.InputTokens != nil {
						s.inputTokens = *u.InputTokens
					}
					if u.OutputTokens != nil {
						s.outputTokens = *u.OutputTokens
					}
				}
			case "content_block_start":
				// For now, we only need special handling for tool_use and thinking blocks.
				// Record content block type so we can detect when it stops later.
				contentBlockTypeByIndex[event.Index] = event.ContentBlock.Type
				switch event.ContentBlock.Type {
				case "tool_use":
					if s.isJSONMode && event.ContentBlock.Name == jsonModeToolName {
						// Start handling the JSON mode tool.
						handlingJsonModeTool = true
						continue // Don't process as regular tool call.
					}
					lastToolCallIndex = event.Index
					resetNextArgumentsDelta = true
					s.message.ToolCalls = append(s.message.ToolCalls, llms.ToolCall{
						ID:        event.ContentBlock.ID,
						Name:      event.ContentBlock.Name,
						Arguments: event.ContentBlock.Input,
					})
					if !yield(llms.StreamStatusToolCallBegin) {
						return
					}
				case "thinking":
					s.lastThought = &content.Thought{
						Text:      event.ContentBlock.Thinking,
						Signature: event.ContentBlock.Signature,
					}
					s.message.Content.AppendThought(event.ContentBlock.Thinking)
					if event.ContentBlock.Signature != "" {
						s.message.Content.SetThoughtSignature(event.ContentBlock.Signature)
					}
					if !yield(llms.StreamStatusThinking) {
						return
					}
				case "redacted_thinking":
					if event.ContentBlock.Data != "" {
						decodedData, err := base64.StdEncoding.DecodeString(event.ContentBlock.Data)
						if err != nil {
							s.err = fmt.Errorf("error decoding redacted_thinking data: %w", err)
							return
						}
						thought := &content.Thought{
							Text:      "(Redacted)",
							Encrypted: decodedData,
							Summary:   true,
						}
						s.lastThought = thought
						s.message.Content = append(s.message.Content, thought)
					}
					if !yield(llms.StreamStatusThinking) {
						return
					}
				}
			case "content_block_delta":
				switch event.Delta.Type {
				case "text_delta":
					// Regular text delta - append to content
					s.lastText = event.Delta.Text
					s.message.Content.Append(s.lastText)
					if !yield(llms.StreamStatusText) {
						return
					}
				case "input_json_delta":
					if event.Delta.PartialJSON == "" {
						continue
					}
					if handlingJsonModeTool {
						// Accumulate JSON delta as if it was text (because we're in JSON mode).
						s.lastText = event.Delta.PartialJSON
						s.message.Content.Append(s.lastText)
						if !yield(llms.StreamStatusText) {
							return
						}
						continue // Don't process as regular tool call.
					}
					index := len(s.message.ToolCalls) - 1
					if resetNextArgumentsDelta {
						s.message.ToolCalls[index].Arguments = json.RawMessage(event.Delta.PartialJSON)
						resetNextArgumentsDelta = false
					} else {
						s.message.ToolCalls[index].Arguments = append(
							s.message.ToolCalls[index].Arguments,
							[]byte(event.Delta.PartialJSON)...,
						)
					}
					if !yield(llms.StreamStatusToolCallDelta) {
						return
					}
				case "thinking_delta":
					s.lastThought = &content.Thought{Text: event.Delta.Thinking}
					s.message.Content.AppendThought(event.Delta.Thinking)
					if !yield(llms.StreamStatusThinking) {
						return
					}
					continue
				case "signature_delta":
					s.lastThought = &content.Thought{Signature: event.Delta.Signature}
					s.message.Content.SetThoughtSignature(event.Delta.Signature)
					if !yield(llms.StreamStatusThinking) {
						return
					}
					continue
				}
			case "content_block_stop":
				if handlingJsonModeTool {
					// Stop handling the JSON mode tool.
					handlingJsonModeTool = false
					continue // Don't process as regular tool.
				}
				// Signal the end of a content block
				// For tool calls, signal that the tool call is ready
				if event.Index == lastToolCallIndex {
					if !yield(llms.StreamStatusToolCallReady) {
						return
					}
				}
				// For thinking blocks, signal that thinking has finished
				if blockType, ok := contentBlockTypeByIndex[event.Index]; ok {
					if blockType == "thinking" || blockType == "redacted_thinking" {
						if !yield(llms.StreamStatusThinkingDone) {
							return
						}
					}
					delete(contentBlockTypeByIndex, event.Index)
				}
			case "message_delta":
				// Update usage statistics
				if u := event.Usage; u != nil {
					// Values are cumulative, so we overwrite the numbers instead of adding.
					// https://docs.anthropic.com/en/docs/build-with-claude/streaming
					if u.CacheReadInputTokens != nil {
						s.cachedInputTokens = *u.CacheReadInputTokens
					}
					if u.CacheCreationInputTokens != nil {
						s.cacheCreationInputTokens = *u.CacheCreationInputTokens
					}
					if u.InputTokens != nil {
						s.inputTokens = *u.InputTokens
					}
					if u.OutputTokens != nil {
						s.outputTokens = *u.OutputTokens
					}
				}
				// Check stop reason, but allow tool_use and end_turn
				if event.Delta.StopReason != "" &&
					event.Delta.StopReason != "tool_use" &&
					event.Delta.StopReason != "end_turn" {
					s.err = fmt.Errorf("unexpected stop reason: %q", event.Delta.StopReason)
					return
				}
			case "message_stop":
				// End of the message stream
				return
			case "ping":
				// Ignore ping events
				continue
			case "error":
				// Handle error events
				if event.Error != nil {
					s.err = fmt.Errorf("API error: %s - %s", event.Error.Type, event.Error.Message)
					return
				}
			default:
				// TODO: Log unknown event type
			}
		}
	}
}

func Tools(toolbox *tools.Toolbox) []Tool {
	toolDefs := []Tool{}
	for _, t := range toolbox.All() {
		switch g := t.Grammar().(type) {
		case tools.JSONGrammar:
			schema := g.Schema()
			toolDefs = append(toolDefs, Tool{
				Name:        schema.Name,
				Description: schema.Description,
				InputSchema: schema.Parameters,
			})
		default:
			panic(fmt.Sprintf("unsupported grammar type: %T", g))
		}
	}
	return toolDefs
}

func contentFromLLM(llmContent content.Content) (cl contentList) {
	cl = []contentItem{}
	for _, item := range llmContent {
		var ci contentItem
		switch v := item.(type) {
		case *content.Text:
			// Skip text blocks that are empty or contain only whitespace.
			if strings.TrimSpace(v.Text) == "" {
				continue
			}
			ci.Type = "text"
			ci.Text = v.Text
		case *content.ImageURL:
			ci.Type = "image"
			if dataValue, found := strings.CutPrefix(v.URL, "data:"); found {
				mimeType, data, found := strings.Cut(dataValue, ";base64,")
				if !found {
					panic(fmt.Sprintf("unsupported data URI format %q", v.URL))
				}
				ci.Source = &source{
					Type:      "base64",
					MediaType: mimeType,
					Data:      data,
				}
			} else {
				ci.Source = &source{
					Type: "url",
					URL:  v.URL,
				}
			}
		case *content.JSON:
			ci.Type = "text"
			ci.Text = string(v.Data)
		case *content.Thought:
			if len(v.Encrypted) > 0 {
				ci.Type = "redacted_thinking"
				ci.Data = base64.StdEncoding.EncodeToString(v.Encrypted)
			} else {
				ci.Type = "thinking"
				ci.Thinking = v.Text
				ci.Signature = v.Signature
			}
		case *content.CacheHint:
			// Add cache control to the previous content item.
			if i := len(cl) - 1; i >= 0 {
				cc := &cacheControl{Type: "ephemeral"}
				switch v.Duration {
				case "short":
					cc.TTL = "5m"
				case "long":
					cc.TTL = "1h"
				}
				cl[i].CacheControl = cc
			}
			continue // Skip appending empty content item
		default:
			panic(fmt.Sprintf("unhandled content item type %T", item))
		}
		cl = append(cl, ci)
	}
	return cl
}

func messageFromLLM(m llms.Message) message {
	apiContent := contentFromLLM(m.Content)
	switch m.Role {
	case "tool":
		// Anthropic expects tool results to be from the user, wrapped in a specific structure.
		return message{
			Role: "user",
			Content: []contentItem{
				{
					Type:      "tool_result",
					ToolUseID: m.ToolCallID,
					Content:   apiContent,
				},
			},
		}
	case "assistant":
		for _, toolCall := range m.ToolCalls {
			apiContent = append(apiContent, contentItem{
				Type:  "tool_use",
				ID:    toolCall.ID,
				Name:  toolCall.Name,
				Input: toolCall.Arguments,
			})
		}
	}
	// Only add a default empty text content if apiContent is genuinely empty
	// (i.e., no original content AND no tool calls were added).
	if len(apiContent) == 0 {
		apiContent = []contentItem{{Type: "text", Text: ""}}
	}
	return message{
		Role:    m.Role,
		Content: apiContent,
	}
}
