package anthropic

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/blixt/go-llms/content"
	"github.com/blixt/go-llms/llms"
	"github.com/blixt/go-llms/tools"
)

type Model struct {
	apiKey            string
	model             string
	endpoint          string
	debug             bool
	maxTokens         int
	maxThinkingTokens int
}

func New(apiKey, model string) *Model {
	return &Model{
		apiKey:    apiKey,
		model:     model,
		endpoint:  "https://api.anthropic.com/v1/messages",
		maxTokens: 1024,
	}
}

func (m *Model) WithDebug() *Model {
	m.debug = true
	return m
}

func (m *Model) WithEndpoint(endpoint string) *Model {
	m.endpoint = endpoint
	return m
}

func (m *Model) WithMaxTokens(maxTokens int) *Model {
	m.maxTokens = maxTokens
	return m
}

func (m *Model) WithThinking(budgetTokens int) *Model {
	// FIXME: The codebase needs to be updated to support thinking models.
	if budgetTokens > 0 {
		panic("thinking models are not yet supported")
	}
	m.maxThinkingTokens = budgetTokens
	return m
}

func (m *Model) Company() string {
	return "Anthropic"
}

func (m *Model) Generate(systemPrompt content.Content, messages []llms.Message, tools *tools.Toolbox) llms.ProviderStream {
	var apiMessages []message
	for _, msg := range messages {
		apiMessages = append(apiMessages, messageFromLLM(msg))
	}

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

	if tools != nil {
		payload["tools"] = Tools(tools)
		payload["tool_choice"] = map[string]string{"type": "auto"}
	}

	if m.maxThinkingTokens > 0 {
		payload["thinking"] = map[string]any{
			"type":          "enabled",
			"budget_tokens": m.maxThinkingTokens,
		}
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return &Stream{err: fmt.Errorf("error encoding JSON: %w", err)}
	}

	if m.debug {
		fmt.Printf("Request: %s\n%s\n", m.endpoint, string(jsonData))
	}

	req, err := http.NewRequest("POST", m.endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return &Stream{err: fmt.Errorf("error creating request: %w", err)}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-API-Key", m.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return &Stream{err: fmt.Errorf("error making request: %w", err)}
	}
	if resp.StatusCode != http.StatusOK {
		if m.debug {
			data, _ := io.ReadAll(resp.Body)
			return &Stream{err: fmt.Errorf("%s\n%s", resp.Status, data)}
		} else {
			return &Stream{err: fmt.Errorf("%s", resp.Status)}
		}
	}

	return &Stream{model: m.model, stream: resp.Body}
}

type Stream struct {
	model    string
	stream   io.Reader
	err      error
	message  llms.Message
	lastText string

	inputTokens, outputTokens int
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

func (s *Stream) ToolCall() llms.ToolCall {
	if len(s.message.ToolCalls) == 0 {
		return llms.ToolCall{}
	}
	return s.message.ToolCalls[len(s.message.ToolCalls)-1]
}

type pricing struct {
	inputCost  float64 // per million tokens
	outputCost float64 // per million tokens
}

var modelPricing = map[string]pricing{
	// Claude 3.7 models
	"claude-3-7-sonnet": {3.00, 15.00},

	// Claude 3.5 models
	"claude-3-5-sonnet": {3.00, 15.00},
	"claude-3-5-haiku":  {0.80, 4.00},

	// Claude 3 models
	"claude-3-opus":   {15.00, 75.00},
	"claude-3-sonnet": {3.00, 15.00},
	"claude-3-haiku":  {0.25, 1.25},
}

func (s *Stream) CostUSD() float64 {
	// First try exact model name
	if pricing, ok := modelPricing[s.model]; ok {
		return float64(s.inputTokens)*pricing.inputCost/1e6 + float64(s.outputTokens)*pricing.outputCost/1e6
	}

	// Then try prefix matching
	for prefix, pricing := range modelPricing {
		if strings.HasPrefix(s.model, prefix) {
			return float64(s.inputTokens)*pricing.inputCost/1e6 + float64(s.outputTokens)*pricing.outputCost/1e6
		}
	}

	// Default return 0 for unknown models
	return 0.0
}

func (s *Stream) Usage() (inputTokens, outputTokens int) {
	return s.inputTokens, s.outputTokens
}

func (s *Stream) Iter() func(yield func(llms.StreamStatus) bool) {
	scanner := bufio.NewScanner(s.stream)
	return func(yield func(llms.StreamStatus) bool) {
		defer io.Copy(io.Discard, s.stream)
		lastToolCallIndex := -1
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
		for scanner.Scan() {
			line, ok := strings.CutPrefix(scanner.Text(), "data: ")
			if !ok {
				continue
			}
			var event streamEvent
			if err := json.Unmarshal([]byte(line), &event); err != nil {
				s.err = fmt.Errorf("error unmarshalling event: %w", err)
				break
			}

			switch event.Type {
			case "message_start":
				// Initialize the message with the role from the message_start event
				s.message.Role = event.Message.Role
				if event.Message.Usage != nil {
					s.inputTokens += event.Message.Usage.InputTokens
					s.outputTokens += event.Message.Usage.OutputTokens
				}
			case "content_block_start":
				// For now, we only need special handling for tool_use and thinking blocks
				switch event.ContentBlock.Type {
				case "tool_use":
					lastToolCallIndex = event.Index
					s.message.ToolCalls = append(s.message.ToolCalls, llms.ToolCall{
						ID:   event.ContentBlock.ID,
						Name: event.ContentBlock.Name,
					})
					if !yield(llms.StreamStatusToolCallBegin) {
						return
					}
				case "redacted_thinking":
					// TODO: We need to track thinking blocks.
				case "thinking":
					// TODO: We need to track thinking blocks.
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
					// Tool use JSON delta - append to tool call arguments
					s.message.ToolCalls[len(s.message.ToolCalls)-1].Arguments = append(
						s.message.ToolCalls[len(s.message.ToolCalls)-1].Arguments,
						[]byte(event.Delta.PartialJSON)...,
					)
					if !yield(llms.StreamStatusToolCallData) {
						return
					}
				case "thinking_delta":
					// TODO: We need to track thinking blocks.
					continue
				case "signature_delta":
					// TODO: We need to track thinking blocks.
					continue
				}
			case "content_block_stop":
				// Signal the end of a content block
				// For tool calls, signal that the tool call is ready
				if event.Index == lastToolCallIndex {
					if !yield(llms.StreamStatusToolCallReady) {
						return
					}
				}
			case "message_delta":
				// Update usage statistics
				if event.Delta.Usage != nil {
					s.inputTokens += event.Delta.Usage.InputTokens
					s.outputTokens += event.Delta.Usage.OutputTokens
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
	tools := []Tool{}
	for _, t := range toolbox.All() {
		schema := t.Schema()
		tools = append(tools, Tool{
			Name:        schema.Name,
			Description: schema.Description,
			InputSchema: schema.Parameters,
		})
	}
	return tools
}

func contentFromLLM(llmContent content.Content) (cl contentList) {
	cl = []contentItem{}
	for _, item := range llmContent {
		var ci contentItem
		switch v := item.(type) {
		case *content.Text:
			ci.Type = "text"
			if strings.TrimSpace(v.Text) == "" {
				ci.Text = "(Empty)"
			} else {
				ci.Text = v.Text
			}
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
				// TODO: Download the image URL and turn it into base64.
				panic("Anthropic does not support URLs for images")
			}
		case *content.JSON:
			ci.Type = "text"
			ci.Text = string(v.Data)
		default:
			panic(fmt.Sprintf("unhandled content item type %T", item))
		}
		cl = append(cl, ci)
	}
	return cl
}

func messageFromLLM(m llms.Message) message {
	if m.Role == "tool" {
		// Anthropic considers tool responses to be from the user.
		return message{
			Role: "user",
			Content: []contentItem{
				{
					Type:      "tool_result",
					ToolUseID: m.ToolCallID,
					Content:   contentFromLLM(m.Content),
				},
			},
		}
	}
	content := contentFromLLM(m.Content)
	for _, toolCall := range m.ToolCalls {
		content = append(content, contentItem{
			Type:  "tool_use",
			ID:    toolCall.ID,
			Name:  toolCall.Name,
			Input: toolCall.Arguments,
		})
	}
	return message{
		Role:    m.Role,
		Content: content,
	}
}
