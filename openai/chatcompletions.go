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

type ChatCompletionsAPI struct {
	accessToken string
	model       string
	endpoint    string
	company     string
	debugger    llms.Debugger
	httpClient  *http.Client

	maxCompletionTokens int
	reasoningEffort     Effort
	verbosity           Verbosity

	// When true, include stream_options.include_usage in requests; default true.
	includeUsage bool

	// clearThink controls whether reasoning content persists across multi-turn
	// conversations. When false, thinking from previous turns is preserved,
	// improving consistency and cache efficiency in agentic workflows.
	// nil means don't send the parameter. Used by Cerebras GLM models.
	clearThink *bool
}

func NewChatCompletionsAPI(accessToken, model string) *ChatCompletionsAPI {
	return &ChatCompletionsAPI{
		accessToken:  accessToken,
		model:        model,
		endpoint:     "https://api.openai.com/v1/chat/completions",
		company:      "OpenAI",
		includeUsage: true,
	}
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

func (m *ChatCompletionsAPI) WithVerbosity(verbosity Verbosity) *ChatCompletionsAPI {
	m.verbosity = verbosity
	return m
}

// WithIncludeUsage sets whether to include stream_options.include_usage in requests.
func (m *ChatCompletionsAPI) WithIncludeUsage(include bool) *ChatCompletionsAPI {
	m.includeUsage = include
	return m
}

// WithClearThink sets the clear_think parameter for Cerebras GLM models.
// When false, thinking from previous turns is preserved across tool-calling loops.
func (m *ChatCompletionsAPI) WithClearThink(clear bool) *ChatCompletionsAPI {
	m.clearThink = &clear
	return m
}

func (m *ChatCompletionsAPI) SetHTTPClient(client *http.Client) {
	m.httpClient = client
}

func (m *ChatCompletionsAPI) Company() string {
	return m.company
}

func (m *ChatCompletionsAPI) Model() string {
	return m.model
}

func (m *ChatCompletionsAPI) SetDebugger(d llms.Debugger) {
	m.debugger = d
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
		"model":    m.model,
		"messages": apiMessages,
		"stream":   true,
	}

	// Include stream_options only when includeUsage is true (not all providers support it)
	if m.includeUsage {
		payload["stream_options"] = map[string]any{"include_usage": true}
	}

	if m.maxCompletionTokens > 0 {
		payload["max_completion_tokens"] = m.maxCompletionTokens
	}

	if m.reasoningEffort != "" {
		payload["reasoning_effort"] = m.reasoningEffort
	}

	if m.verbosity != "" {
		payload["verbosity"] = m.verbosity
	}

	if m.clearThink != nil {
		payload["clear_think"] = *m.clearThink
	}

	if toolbox != nil {
		// Build tools first.
		apiTools := Tools(toolbox)
		// Always include full tools for cacheability; constrain with tool_choice
		payload["tools"] = apiTools

		// Map tools.Choice to Chat Completions tool_choice.
		// Chat Completions now supports an allowed_tools object similar to Responses API.
		choice := toolbox.Choice
		switch choice.Mode {
		case tools.ChoiceAllowOnly:
			if len(choice.AllowedTools) == 0 {
				payload["tool_choice"] = "none"
			} else {
				// Validate that at least one allowed tool exists in apiTools
				exists := false
				for _, n := range choice.AllowedTools {
					for _, t := range apiTools {
						name := ""
						if t.Function != nil {
							name = t.Function.Name
						}
						if t.Custom != nil {
							name = t.Custom.Name
						}
						if name == n {
							exists = true
							break
						}
					}
					if exists {
						break
					}
				}
				if !exists {
					return &ChatCompletionsStream{err: fmt.Errorf("openai chat: no allowed tools found in toolbox")}
				}
				// Build allowed_tools object
				allowed := make([]ChatAllowedTool, 0, len(choice.AllowedTools))
				for _, n := range choice.AllowedTools {
					// Prefer function type; fall back to custom if applicable
					// We don’t try to disambiguate; include function entry, which the model resolves
					allowed = append(allowed, ChatAllowedTool{Type: "function", Function: &ChatAllowedToolFunc{Name: n}})
				}
				payload["tool_choice"] = ChatAllowedToolsChoice{Type: "allowed_tools", Mode: "auto", Tools: allowed}
			}
		case tools.ChoiceRequireOneOf:
			switch len(choice.AllowedTools) {
			case 0:
				payload["tool_choice"] = "none"
			case 1:
				// Force the specific tool by name; validate it exists
				name := choice.AllowedTools[0]
				exists := false
				for _, t := range apiTools {
					if (t.Function != nil && t.Function.Name == name) || (t.Custom != nil && t.Custom.Name == name) {
						exists = true
						break
					}
				}
				if !exists {
					return &ChatCompletionsStream{err: fmt.Errorf("openai chat: required tool %q not found in toolbox", name)}
				}
				payload["tool_choice"] = ChatToolChoice{Type: "function", Function: &ChatToolChoiceFunc{Name: name}}
			default:
				// Multiple allowed: use allowed_tools with mode:required
				exists := false
				for _, n := range choice.AllowedTools {
					for _, t := range apiTools {
						name := ""
						if t.Function != nil {
							name = t.Function.Name
						}
						if t.Custom != nil {
							name = t.Custom.Name
						}
						if name == n {
							exists = true
							break
						}
					}
					if exists {
						break
					}
				}
				if !exists {
					return &ChatCompletionsStream{err: fmt.Errorf("openai chat: none of the required tools are present in toolbox")}
				}
				allowed := make([]ChatAllowedTool, 0, len(choice.AllowedTools))
				for _, n := range choice.AllowedTools {
					allowed = append(allowed, ChatAllowedTool{Type: "function", Function: &ChatAllowedToolFunc{Name: n}})
				}
				payload["tool_choice"] = ChatAllowedToolsChoice{Type: "allowed_tools", Mode: "required", Tools: allowed}
			}
		default:
			payload["tool_choice"] = "auto"
		}
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

	if m.debugger != nil {
		m.debugger.RawRequest(m.endpoint, jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return &ChatCompletionsStream{err: fmt.Errorf("error creating request: %w", err)}
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
				return &ChatCompletionsStream{err: &llms.HTTPError{
					StatusCode: resp.StatusCode,
					Status:     resp.Status,
					ErrorType:  openAIError.Error.Type,
					Message:    openAIError.Error.Message,
				}}
			}
			// Body read okay, but JSON parsing failed or structure mismatch.
			// Just repeat the body up to a limit.
			body := string(bodyBytes)
			if len(body) > 1024 {
				body = body[:1024]
			}
			return &ChatCompletionsStream{err: &llms.HTTPError{
				StatusCode: resp.StatusCode,
				Status:     resp.Status,
				Message:    body,
			}}
		}
		// Default fallback: Read error, empty body, or failed/unexpected JSON parse.
		return &ChatCompletionsStream{err: &llms.HTTPError{
			StatusCode: resp.StatusCode,
			Status:     resp.Status,
		}}
	}

	return &ChatCompletionsStream{ctx: ctx, model: m.model, stream: resp.Body, debugger: m.debugger}
}

type ChatCompletionsStream struct {
	ctx      context.Context
	model    string
	stream   io.Reader
	debugger llms.Debugger
	err      error
	message  llms.Message
	lastText string
	usage    *usage
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

func (s *ChatCompletionsStream) Image() (string, string) {
	// OpenAI's Chat Completions API doesn't generate any images as of this writing.
	return "", ""
}

func (s *ChatCompletionsStream) ToolCall() llms.ToolCall {
	if len(s.message.ToolCalls) == 0 {
		return llms.ToolCall{}
	}
	return s.message.ToolCalls[len(s.message.ToolCalls)-1]
}

func (s *ChatCompletionsStream) Thought() content.Thought {
	// OpenAI API does not currently stream thoughts through Chat Completions API.
	// TODO: Switch to Responses API to support this.
	return content.Thought{}
}

func (s *ChatCompletionsStream) Usage() llms.Usage {
	if s.usage == nil {
		return llms.Usage{}
	}
	return llms.Usage{
		CachedInputTokens: s.usage.PromptTokensDetails.CachedTokens,
		InputTokens:       s.usage.PromptTokens,
		OutputTokens:      s.usage.CompletionTokens,
	}
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
				// If scanning fails (e.g., EOF or error), check for scanner error.
				if err := scanner.Err(); err != nil {
					s.err = fmt.Errorf("error scanning stream: %w", err)
				}
				return // Exit loop on scan failure or EOF
			}

			if s.debugger != nil && strings.TrimSpace(scanner.Text()) != "" {
				s.debugger.RawEvent([]byte(scanner.Text()))
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
						existing := &s.message.ToolCalls[toolDelta.Index]
						if toolDelta.Type != "" {
							if existing.Metadata == nil {
								existing.Metadata = make(map[string]string)
							}
							existing.Metadata["openai:item_type"] = toolDelta.Type
						} else if existing.Metadata == nil {
							switch {
							case toolDelta.Custom != nil && toolDelta.Custom.Input != nil:
								existing.Metadata = map[string]string{"openai:item_type": "custom"}
							case toolDelta.Function != nil:
								existing.Metadata = map[string]string{"openai:item_type": "function"}
							}
						}
						var deltaData []byte
						if toolDelta.Function != nil && toolDelta.Function.Arguments != "" {
							deltaData = []byte(toolDelta.Function.Arguments)
						} else if toolDelta.Custom != nil && toolDelta.Custom.Input != nil {
							// Custom tool input arrives as a JSON string token; treat it as plain text
							deltaData = []byte(*toolDelta.Custom.Input)
						}

						if len(deltaData) > 0 {
							existing.Arguments = append(existing.Arguments, deltaData...)
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
		switch g := t.Grammar().(type) {
		case tools.JSONGrammar:
			if schema := g.Schema(); schema != nil {
				apiTools = append(apiTools, Tool{Type: "function", Function: schema})
			}
		case tools.TextGrammar:
			apiTools = append(apiTools, Tool{Type: "custom", Custom: &CustomToolSchema{
				Name:        t.FuncName(),
				Description: t.Description(),
				Format:      map[string]any{"type": "text"},
			}})
		case tools.LarkGrammar:
			apiTools = append(apiTools, Tool{Type: "custom", Custom: &CustomToolSchema{
				Name:        t.FuncName(),
				Description: t.Description(),
				Format: map[string]any{
					"type": "grammar",
					"grammar": map[string]any{
						"definition": g.Definition,
						"syntax":     "lark",
					},
				},
			}})
		case tools.RegexGrammar:
			apiTools = append(apiTools, Tool{Type: "custom", Custom: &CustomToolSchema{
				Name:        t.FuncName(),
				Description: t.Description(),
				Format: map[string]any{
					"type": "grammar",
					"grammar": map[string]any{
						"definition": g.Definition,
						"syntax":     "regex",
					},
				},
			}})
		default:
			panic(fmt.Sprintf("unsupported grammar type: %T", g))
		}
	}
	return apiTools
}
