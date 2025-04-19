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

	"github.com/blixt/go-llms/content"
	"github.com/blixt/go-llms/llms"
	"github.com/blixt/go-llms/tools"
)

type Model struct {
	accessToken string
	model       string
	endpoint    string
	company     string

	maxCompletionTokens int
}

func New(accessToken, model string) *Model {
	return &Model{
		accessToken: accessToken,
		model:       model,
		endpoint:    "https://api.openai.com/v1/chat/completions",
		company:     "OpenAI",
	}
}

// WithEndpoint sets the endpoint (and company name) so OpenAI-compatible API
// endpoints can be used.
func (m *Model) WithEndpoint(endpoint, company string) *Model {
	m.endpoint = endpoint
	m.company = company
	return m
}

func (m *Model) WithMaxCompletionTokens(maxCompletionTokens int) *Model {
	m.maxCompletionTokens = maxCompletionTokens
	return m
}

func (m *Model) Company() string {
	return m.company
}

func (m *Model) Model() string {
	return m.model
}

func (m *Model) Generate(ctx context.Context, systemPrompt content.Content, messages []llms.Message, toolbox *tools.Toolbox) llms.ProviderStream {
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

	req, err := http.NewRequestWithContext(ctx, "POST", m.endpoint, bytes.NewReader(jsonData))
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
				return &Stream{err: fmt.Errorf("%s: %s: %s", resp.Status, openAIError.Error.Type, openAIError.Error.Message)}
			}
			// Body read okay, but JSON parsing failed or structure mismatch.
			// Fall through to return status only.
		}
		// Default fallback: Read error, empty body, or failed/unexpected JSON parse.
		return &Stream{err: fmt.Errorf("%s", resp.Status)}
	}

	return &Stream{ctx: ctx, model: m.model, stream: resp.Body}
}

type Stream struct {
	ctx      context.Context
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
		// Add a loop to handle both context cancellation and scanner operations.
		for {
			select {
			case <-s.ctx.Done():
				s.err = s.ctx.Err()
				return // Exit if context is cancelled
			default:
				// Context OK, keep scanning.
			}
			// Try to scan the next line.
			if !scanner.Scan() {
				// If scanning fails (e.g., EOF or error), check for scanner error.
				if err := scanner.Err(); err != nil {
					s.err = fmt.Errorf("error scanning stream: %w", err)
				}
				return // Exit loop on scan failure or EOF
			}

			// Process the scanned line.
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
				continue // Continue the outer loop to check context or EOF
			}
			var chunk chatCompletionChunk
			if err := json.Unmarshal([]byte(line), &chunk); err != nil {
				s.err = fmt.Errorf("error unmarshalling chunk: %w", err)
				return
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
