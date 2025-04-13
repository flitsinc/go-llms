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

	maxCompletionTokens int
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

func (m *Model) WithMaxCompletionTokens(maxCompletionTokens int) *Model {
	m.maxCompletionTokens = maxCompletionTokens
	return m
}

func (m *Model) Company() string {
	return "OpenAI"
}

func (m *Model) Generate(systemPrompt content.Content, messages []llms.Message, toolbox *tools.Toolbox) llms.ProviderStream {
	var apiMessages []message
	if systemPrompt != nil {
		apiMessages = append(apiMessages, message{
			Role:    "system",
			Content: convertContent(systemPrompt),
		})
	}

	for _, msg := range messages {
		convertedMsgs := messagesFromLLM(msg)
		apiMessages = append(apiMessages, convertedMsgs...)
	}

	payload := map[string]any{
		"model":          m.model,
		"messages":       apiMessages,
		"stream":         true,
		"stream_options": map[string]any{"include_usage": true},
	}

	if m.maxCompletionTokens > 0 {
		payload["max_completion_tokens"] = m.maxCompletionTokens
	}

	if toolbox != nil {
		payload["tools"] = Tools(toolbox)
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

type pricing struct {
	inputCost  float64 // per million tokens
	outputCost float64 // per million tokens
}

var modelPricing = map[string]pricing{
	// GPT-4.5 models
	"gpt-4.5-preview":            {75.00, 150.00},
	"gpt-4.5-preview-2025-02-27": {75.00, 150.00},

	// GPT-4o models
	"gpt-4o":                             {2.50, 10.00},
	"gpt-4o-2024-08-06":                  {2.50, 10.00},
	"gpt-4o-2024-11-20":                  {2.50, 10.00},
	"gpt-4o-2024-05-13":                  {5.00, 15.00},
	"gpt-4o-audio-preview":               {2.50, 10.00},
	"gpt-4o-audio-preview-2024-12-17":    {2.50, 10.00},
	"gpt-4o-audio-preview-2024-10-01":    {2.50, 10.00},
	"gpt-4o-realtime-preview":            {5.00, 20.00},
	"gpt-4o-realtime-preview-2024-12-17": {5.00, 20.00},
	"gpt-4o-realtime-preview-2024-10-01": {5.00, 20.00},
	"chatgpt-4o-latest":                  {5.00, 15.00},

	// GPT-4o mini models
	"gpt-4o-mini":                             {0.15, 0.60},
	"gpt-4o-mini-2024-07-18":                  {0.15, 0.60},
	"gpt-4o-mini-audio-preview":               {0.15, 0.60},
	"gpt-4o-mini-audio-preview-2024-12-17":    {0.15, 0.60},
	"gpt-4o-mini-realtime-preview":            {0.60, 2.40},
	"gpt-4o-mini-realtime-preview-2024-12-17": {0.60, 2.40},

	// O1 models
	"o1":                    {15.00, 60.00},
	"o1-2024-12-17":         {15.00, 60.00},
	"o1-preview-2024-09-12": {15.00, 60.00},
	"o1-pro":                {150.00, 600.00},
	"o1-pro-2025-03-19":     {150.00, 600.00},
	"o1-mini":               {1.10, 4.40},
	"o1-mini-2024-09-12":    {1.10, 4.40},

	// O3 models
	"o3-mini":            {1.10, 4.40},
	"o3-mini-2025-01-31": {1.10, 4.40},

	// GPT-4 Turbo models
	"gpt-4-turbo":               {10.00, 30.00},
	"gpt-4-turbo-2024-04-09":    {10.00, 30.00},
	"gpt-4-0125-preview":        {10.00, 30.00},
	"gpt-4-1106-preview":        {10.00, 30.00},
	"gpt-4-1106-vision-preview": {10.00, 30.00},

	// GPT-4 models
	"gpt-4":          {30.00, 60.00},
	"gpt-4-0613":     {30.00, 60.00},
	"gpt-4-0314":     {30.00, 60.00},
	"gpt-4-32k":      {60.00, 120.00},
	"gpt-4-32k-0613": {60.00, 120.00},

	// GPT-3.5 models
	"gpt-3.5-turbo":          {0.50, 1.50},
	"gpt-3.5-turbo-0125":     {0.50, 1.50},
	"gpt-3.5-turbo-1106":     {1.00, 2.00},
	"gpt-3.5-turbo-0613":     {1.50, 2.00},
	"gpt-3.5-0301":           {1.50, 2.00},
	"gpt-3.5-turbo-instruct": {1.50, 2.00},
	"gpt-3.5-turbo-16k-0613": {3.00, 4.00},

	// Older models
	"davinci-002": {2.00, 2.00},
	"babbage-002": {0.40, 0.40},
}

func (s *Stream) CostUSD() float64 {
	pricing, ok := modelPricing[s.model]
	if !ok {
		return 0 // Unknown model
	}

	inputTokens, outputTokens := s.Usage()
	return float64(inputTokens)*pricing.inputCost/1e6 + float64(outputTokens)*pricing.outputCost/1e6
}

func (s *Stream) Usage() (inputTokens, outputTokens int) {
	if s.usage == nil {
		return 0, 0
	}
	return s.usage.PromptTokens, s.usage.CompletionTokens
}

func (s *Stream) Iter() func(yield func(llms.StreamStatus) bool) {
	scanner := bufio.NewScanner(s.stream)
	var activeToolCallIndex = -1 // Track the index of the tool call being processed

	return func(yield func(llms.StreamStatus) bool) {
		defer io.Copy(io.Discard, s.stream)
		for scanner.Scan() {
			line, ok := strings.CutPrefix(scanner.Text(), "data: ")
			if !ok {
				continue
			}
			if line == "[DONE]" {
				// Stream ended. If a tool call was active, mark it as ready.
				if activeToolCallIndex != -1 {
					if !yield(llms.StreamStatusToolCallReady) {
						return
					}
					activeToolCallIndex = -1 // Reset active tool call
				}
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
			// Content is nullable string in delta
			if delta.Content != nil {
				s.lastText = *delta.Content
				if s.lastText != "" {
					s.message.Content.Append(s.lastText)
					if !yield(llms.StreamStatusText) {
						return
					}
				}
			}

			// Handle Tool Calls Delta
			if len(delta.ToolCalls) > 0 {
				for _, toolDelta := range delta.ToolCalls {
					if toolDelta.Index >= len(s.message.ToolCalls) {
						// This is a new tool call starting
						if toolDelta.Index != len(s.message.ToolCalls) {
							panic(fmt.Sprintf("tool call index mismatch: expected %d, got %d", len(s.message.ToolCalls), toolDelta.Index))
						}
						// If a previous tool call was active, mark it as ready now.
						if activeToolCallIndex != -1 {
							if !yield(llms.StreamStatusToolCallReady) {
								return // Abort if yield fails
							}
						}
						// Add the new tool call (converting from toolCallDelta)
						llmToolCall := toolDelta.ToLLM()
						s.message.ToolCalls = append(s.message.ToolCalls, llmToolCall)
						activeToolCallIndex = toolDelta.Index // Mark new tool call as active
						if !yield(llms.StreamStatusToolCallBegin) {
							return // Abort if yield fails
						}
					} else {
						// This is appending arguments to an existing tool call
						if toolDelta.Function.Arguments != "" {
							s.message.ToolCalls[toolDelta.Index].Arguments = append(s.message.ToolCalls[toolDelta.Index].Arguments, toolDelta.Function.Arguments...)
							if !yield(llms.StreamStatusToolCallData) {
								return // Abort if yield fails
							}
						}
					}
				}
			}
			// Check if the overall message is finished (might indicate last tool call is ready)
			if chunk.Choices[0].FinishReason != nil && *chunk.Choices[0].FinishReason == "tool_calls" {
				if activeToolCallIndex != -1 {
					if !yield(llms.StreamStatusToolCallReady) {
						return // Abort if yield fails
					}
					activeToolCallIndex = -1 // Reset active tool call
				}
			}
		}
	}
}

func Tools(toolbox *tools.Toolbox) []Tool {
	apiTools := []Tool{}
	for _, t := range toolbox.All() {
		// Get the schema which is *tools.FunctionSchema
		schema := t.Schema()
		if schema == nil {
			// Handle cases where a tool might not have a schema (though unlikely with current structure)
			continue
		}
		apiTools = append(apiTools, Tool{
			Type: "function",
			// Dereference the pointer to get tools.FunctionSchema
			Function: *schema,
		})
	}
	return apiTools
}
