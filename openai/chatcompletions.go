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

// thinkingSplitter handles splitting content into thinking and regular text
// based on configurable start/end block markers.
type thinkingSplitter struct {
	startBlock string
	endBlock   string
	remainder  string // Buffer for incomplete content
	inThinking bool   // Whether we're currently inside a thinking block
}

// Split processes content and returns any thinking content and regular text content.
// It handles cases where block markers might be split across chunks.
func (ts *thinkingSplitter) Split(content string) (thinking, text string) {
	// If no blocks configured, passthrough everything as text
	if ts.startBlock == "" || ts.endBlock == "" {
		return "", content
	}

	// Prepend any remainder from previous chunk
	content = ts.remainder + content
	ts.remainder = ""

	var thinkingParts []string
	var textParts []string

	for content != "" {
		if ts.inThinking {
			// Look for end marker
			if idx := strings.Index(content, ts.endBlock); idx >= 0 {
				// Found complete end marker
				thinkingParts = append(thinkingParts, content[:idx])
				ts.inThinking = false
				content = content[idx+len(ts.endBlock):]
			} else {
				// No complete end marker found
				// Check if we have a partial match at the end
				potentialMatch := ts.findPartialMatch(content, ts.endBlock)
				if potentialMatch > 0 {
					// Save the potential partial match as remainder
					ts.remainder = content[len(content)-potentialMatch:]
					content = content[:len(content)-potentialMatch]
				}
				// Add remaining content as thinking
				if content != "" {
					thinkingParts = append(thinkingParts, content)
				}
				break
			}
		} else {
			// Look for start marker
			if idx := strings.Index(content, ts.startBlock); idx >= 0 {
				// Found complete start marker
				if idx > 0 {
					textParts = append(textParts, content[:idx])
				}
				ts.inThinking = true
				content = content[idx+len(ts.startBlock):]
			} else {
				// No complete start marker found
				// Check if we have a partial match at the end
				potentialMatch := ts.findPartialMatch(content, ts.startBlock)
				if potentialMatch > 0 {
					// Save the potential partial match as remainder
					ts.remainder = content[len(content)-potentialMatch:]
					content = content[:len(content)-potentialMatch]
				}
				// Add remaining content as text
				if content != "" {
					textParts = append(textParts, content)
				}
				break
			}
		}
	}

	return strings.Join(thinkingParts, ""), strings.Join(textParts, "")
}

// Flush returns any remaining content in the buffer as text and clears the buffer.
// This should be called when the stream ends.
func (ts *thinkingSplitter) Flush() string {
	remainder := ts.remainder
	ts.remainder = ""
	ts.inThinking = false
	return remainder
}

// findPartialMatch checks if the end of content could be the beginning of marker.
// Returns the length of the potential partial match.
func (ts *thinkingSplitter) findPartialMatch(content, marker string) int {
	maxLen := len(marker) - 1
	if maxLen > len(content) {
		maxLen = len(content)
	}

	for i := maxLen; i > 0; i-- {
		if strings.HasSuffix(content, marker[:i]) {
			return i
		}
	}
	return 0
}

type ChatCompletionsAPI struct {
	accessToken string
	model       string
	endpoint    string
	company     string
	debug       bool

	maxCompletionTokens int
	reasoningEffort     Effort

	// Custom thinking block support
	thinkingBlockStart string
	thinkingBlockEnd   string
}

func NewChatCompletionsAPI(accessToken, model string) *ChatCompletionsAPI {
	return &ChatCompletionsAPI{
		accessToken: accessToken,
		model:       model,
		endpoint:    "https://api.openai.com/v1/chat/completions",
		company:     "OpenAI",
	}
}

func (m *ChatCompletionsAPI) WithDebug() *ChatCompletionsAPI {
	m.debug = true
	return m
}

// WithEndpoint sets the endpoint (and company name) so OpenAI-compatible API
// endpoints can be used.
func (m *ChatCompletionsAPI) WithEndpoint(endpoint, company string) *ChatCompletionsAPI {
	m.endpoint = endpoint
	m.company = company
	return m
}

func (m *ChatCompletionsAPI) WithMaxCompletionTokens(maxCompletionTokens int) *ChatCompletionsAPI {
	m.maxCompletionTokens = maxCompletionTokens
	return m
}

func (m *ChatCompletionsAPI) WithThinking(effort Effort) *ChatCompletionsAPI {
	m.reasoningEffort = effort
	return m
}

// WithCustomThinkBlock configures custom start/end markers for thinking blocks.
// Some OpenAI-compatible providers stream thinking inside custom blocks
// (e.g. DeepSeek uses <think> and </think>).
func (m *ChatCompletionsAPI) WithCustomThinkBlock(start, end string) *ChatCompletionsAPI {
	m.thinkingBlockStart = start
	m.thinkingBlockEnd = end
	return m
}

func (m *ChatCompletionsAPI) Company() string {
	return m.company
}

func (m *ChatCompletionsAPI) Model() string {
	return m.model
}

func (m *ChatCompletionsAPI) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []llms.Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) llms.ProviderStream {
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

	if m.reasoningEffort != "" {
		payload["reasoning_effort"] = m.reasoningEffort
	}

	if toolbox != nil {
		payload["tools"] = Tools(toolbox)
	}

	// Add response_format if JSON schema is provided
	if jsonOutputSchema != nil {
		// Use the type "json_schema" and provide the schema definition
		payload["response_format"] = responseFormat{
			Type: "json_schema",
			JSONSchema: &jsonSchemaDefinition{
				Name:   "structured_output", // Provide a default name
				Schema: jsonOutputSchema,
				Strict: true, // Default to strict enforcement, can be made configurable
			},
		}
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return &ChatCompletionsStream{err: fmt.Errorf("error encoding JSON: %w", err)}
	}

	if m.debug {
		fmt.Printf("\033[1;90m%s\033[0m\n", m.endpoint)
		fmt.Printf("-> \033[2;34m%s\033[0m\n", string(jsonData))
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return &ChatCompletionsStream{err: fmt.Errorf("error creating request: %w", err)}
	}
	if m.accessToken != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", m.accessToken))
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return &ChatCompletionsStream{err: fmt.Errorf("error making request: %w", err)}
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
				return &ChatCompletionsStream{err: fmt.Errorf("%s: %s: %s", resp.Status, openAIError.Error.Type, openAIError.Error.Message)}
			}
			// Body read okay, but JSON parsing failed or structure mismatch.
			// Fall through to return status only.
		}
		// Default fallback: Read error, empty body, or failed/unexpected JSON parse.
		return &ChatCompletionsStream{err: fmt.Errorf("%s", resp.Status)}
	}

	return &ChatCompletionsStream{
		ctx:    ctx,
		model:  m.model,
		stream: resp.Body,
		debug:  m.debug,
		splitter: &thinkingSplitter{
			startBlock: m.thinkingBlockStart,
			endBlock:   m.thinkingBlockEnd,
		},
	}
}

type ChatCompletionsStream struct {
	ctx      context.Context
	model    string
	stream   io.Reader
	debug    bool
	err      error
	message  llms.Message
	lastText string
	usage    *usage

	// Thinking splitter for handling custom thinking blocks
	splitter    *thinkingSplitter
	lastThought *content.Thought
}

func (s *ChatCompletionsStream) Err() error {
	return s.err
}

func (s *ChatCompletionsStream) Message() llms.Message {
	return s.message
}

func (s *ChatCompletionsStream) Text() string {
	return s.lastText
}

func (s *ChatCompletionsStream) ToolCall() llms.ToolCall {
	if len(s.message.ToolCalls) == 0 {
		return llms.ToolCall{}
	}
	return s.message.ToolCalls[len(s.message.ToolCalls)-1]
}

func (s *ChatCompletionsStream) Thought() content.Thought {
	// If custom thinking blocks are configured, return the latest thought
	if s.lastThought != nil {
		return *s.lastThought
	}
	// Otherwise, return empty thought (standard OpenAI Chat Completions API doesn't support thinking)
	return content.Thought{}
}

func (s *ChatCompletionsStream) Usage() (inputTokens, outputTokens int) {
	if s.usage == nil {
		return 0, 0
	}
	return s.usage.PromptTokens, s.usage.CompletionTokens
}

func (s *ChatCompletionsStream) Iter() func(yield func(llms.StreamStatus) bool) {
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
				// Stream ended, flush any remaining content from the splitter
				if remainder := s.splitter.Flush(); remainder != "" {
					// Treat any remainder as text since we couldn't determine its type
					s.message.Content.Append(remainder)
					s.lastText = remainder
					yield(llms.StreamStatusText)
				}

				// If scanning fails (e.g., EOF or error), check for scanner error.
				if err := scanner.Err(); err != nil {
					s.err = fmt.Errorf("error scanning stream: %w", err)
				}
				return // Exit loop on scan failure or EOF
			}

			if s.debug && strings.TrimSpace(scanner.Text()) != "" {
				fmt.Printf("<- \033[2;32m%s\033[0m\n", scanner.Text())
			}

			// Process the scanned line.
			line, ok := strings.CutPrefix(scanner.Text(), "data: ")
			if !ok {
				continue
			}
			if line == "[DONE]" {
				// Stream ended. Flush any remaining content from the splitter
				if remainder := s.splitter.Flush(); remainder != "" {
					s.message.Content.Append(remainder)
					s.lastText = remainder
					if !yield(llms.StreamStatusText) {
						return
					}
				}

				// If a tool call was active, mark it as ready.
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
					// Use the splitter to separate thinking and text content
					thinking, text := s.splitter.Split(s.lastText)

					// Handle any thinking content
					if thinking != "" {
						s.message.Content.AppendThought(thinking)
						s.lastThought = &content.Thought{Text: thinking}
						if !yield(llms.StreamStatusThinking) {
							return
						}
					}

					// Handle any text content
					if text != "" {
						s.message.Content.Append(text)
						s.lastText = text
						if !yield(llms.StreamStatusText) {
							return
						}
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
							if !yield(llms.StreamStatusToolCallDelta) {
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
