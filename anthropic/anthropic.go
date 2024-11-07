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
	apiKey   string
	model    string
	endpoint string
	debug    bool
}

func New(apiKey, model string) *Model {
	return &Model{
		apiKey:   apiKey,
		model:    model,
		endpoint: "https://api.anthropic.com/v1/messages",
	}
}

func (m *Model) WithDebug(debug bool) *Model {
	m.debug = debug
	return m
}

func (m *Model) WithEndpoint(endpoint string) *Model {
	m.endpoint = endpoint
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
		"model":      m.model,
		"max_tokens": 1024,
		"messages":   apiMessages,
		"stream":     true,
	}

	if systemPrompt != nil {
		payload["system"] = contentFromLLM(systemPrompt)
	}

	if tools != nil {
		payload["tools"] = Tools(tools)
		payload["tool_choice"] = map[string]string{"type": "auto"}
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
	usage    *usage
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

func (s *Stream) CostUSD() float64 {
	// TODO: Implement cost calculation based on Anthropic's pricing
	return 0.0
}

func (s *Stream) Usage() (inputTokens, outputTokens int) {
	if s.usage == nil {
		return 0, 0
	}
	return s.usage.InputTokens, s.usage.OutputTokens
}

func (s *Stream) Iter() func(yield func(llms.StreamStatus) bool) {
	scanner := bufio.NewScanner(s.stream)
	return func(yield func(llms.StreamStatus) bool) {
		defer io.Copy(io.Discard, s.stream)
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
				s.message.Role = event.Message.Role
			case "content_block_delta":
				switch event.Delta.Type {
				case "text_delta":
					s.lastText = event.Delta.Text
					s.message.Content.Append(s.lastText)
					if !yield(llms.StreamStatusText) {
						return
					}
				case "input_json_delta":
					s.message.ToolCalls[len(s.message.ToolCalls)-1].Arguments = append(s.message.ToolCalls[len(s.message.ToolCalls)-1].Arguments, []byte(event.Delta.PartialJSON)...)
					if !yield(llms.StreamStatusToolCallData) {
						return
					}
				}
			case "content_block_start":
				if event.ContentBlock.Type == "tool_use" {
					s.message.ToolCalls = append(s.message.ToolCalls, llms.ToolCall{
						ID:   event.ContentBlock.ID,
						Name: event.ContentBlock.Name,
					})
					if !yield(llms.StreamStatusToolCallBegin) {
						return
					}
				}
			case "content_block_stop":
				if len(s.message.ToolCalls) > 0 {
					if !yield(llms.StreamStatusToolCallReady) {
						return
					}
				}
			case "message_delta":
				if event.Delta.Usage != nil {
					s.usage = event.Delta.Usage
				}
			case "message_stop":
				return
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
