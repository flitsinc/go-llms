package openai

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
	accessToken string
	model       string
	endpoint    string
}

func New(accessToken, model string) *Model {
	return &Model{
		accessToken: accessToken,
		model:       model,
		endpoint:    "https://api.openai.com/v1/chat/completions",
	}
}

func (m *Model) WithEndpoint(endpoint string) *Model {
	m.endpoint = endpoint
	return m
}

func (m *Model) Company() string {
	return "OpenAI"
}

func (m *Model) Generate(systemPrompt content.Content, messages []llms.Message, tools *tools.Toolbox) llms.ProviderStream {
	var apiMessages []message
	if systemPrompt != nil {
		apiMessages = make([]message, 0, len(messages)+1)
		apiMessages = append(apiMessages, message{
			Role:    "system",
			Content: convertContent(systemPrompt),
		})
	} else {
		apiMessages = make([]message, 0, len(messages))
	}
	for _, msg := range messages {
		apiMessages = append(apiMessages, messageFromLLM(msg))
	}

	payload := map[string]any{
		"model":          m.model,
		"messages":       apiMessages,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}

	if tools != nil {
		payload["tools"] = Tools(tools)
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return &Stream{err: fmt.Errorf("error encoding JSON: %w", err)}
	}

	req, err := http.NewRequest("POST", m.endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return &Stream{err: fmt.Errorf("error creating request: %w", err)}
	}
	if m.accessToken != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", m.accessToken))
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return &Stream{err: fmt.Errorf("error making request: %w", err)}
	}
	if resp.StatusCode != http.StatusOK {
		// TODO: Consider parsing the body for a more specific error.
		return &Stream{err: fmt.Errorf("%s", resp.Status)}
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
	switch s.model {
	case "gpt-4o":
		const inputCost = 5   // per million tokens
		const outputCost = 15 // per million tokens
		inputTokens, outputTokens := s.Usage()
		return float64(inputTokens)*inputCost/1e6 + float64(outputTokens)*outputCost/1e6
	default:
		// FIXME
		panic(fmt.Sprintf("unknown model: %q", s.model))
	}
}

func (s *Stream) Usage() (inputTokens, outputTokens int) {
	if s.usage == nil {
		return 0, 0
	}
	return s.usage.PromptTokens, s.usage.CompletionTokens
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
			if line == "[DONE]" {
				continue
			}
			var chunk chatCompletionChunk
			if err := json.Unmarshal([]byte(line), &chunk); err != nil {
				s.err = fmt.Errorf("error unmarshalling chunk: %w", err)
				break
			}
			if chunk.Usage != nil {
				s.usage = chunk.Usage
			}
			if len(chunk.Choices) < 1 {
				continue
			}
			delta := chunk.Choices[0].Delta
			if delta.Role != "" {
				s.message.Role = delta.Role
			}
			s.lastText = delta.Content
			if s.lastText != "" {
				s.message.Content.Append(s.lastText)
				if !yield(llms.StreamStatusText) {
					return
				}
			}
			if len(delta.ToolCalls) > 1 {
				panic("received more than one tool call in a single chunk")
			}
			if len(delta.ToolCalls) == 0 {
				continue
			}
			toolDelta := delta.ToolCalls[0]
			if toolDelta.Index < len(s.message.ToolCalls) {
				if toolDelta.Index != len(s.message.ToolCalls)-1 {
					panic("tool call index mismatch")
				}
				s.message.ToolCalls[toolDelta.Index].Arguments = append(s.message.ToolCalls[toolDelta.Index].Arguments, toolDelta.Function.Arguments...)
				if !yield(llms.StreamStatusToolCallData) {
					return
				}
			} else {
				if toolDelta.Index > 0 {
					if !yield(llms.StreamStatusToolCallReady) {
						return
					}
				}
				s.message.ToolCalls = append(s.message.ToolCalls, toolDelta.ToLLM())
				if !yield(llms.StreamStatusToolCallBegin) {
					return
				}
			}
		}
		if len(s.message.ToolCalls) > 0 {
			if !yield(llms.StreamStatusToolCallReady) {
				return
			}
		}
	}
}

func Tools(toolbox *tools.Toolbox) []Tool {
	tools := []Tool{}
	for _, tool := range toolbox.All() {
		tools = append(tools, Tool{
			Type:     "function",
			Function: *tool.Schema(),
		})
	}
	return tools
}
