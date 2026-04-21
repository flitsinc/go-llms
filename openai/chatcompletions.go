package openai

import (
	"bufio"
	"bytes"
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strconv"
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
	httpClient  *http.Client

	maxCompletionTokens int
	reasoningEffort     Effort
	verbosity           Verbosity

	// When true, include stream_options.include_usage in requests; default true.
	includeUsage bool

	// When true, encode CacheHint items as cache_control on content parts.
	cacheControlPromptHints bool
	// When true, encode assistant Thought items as reasoning_details for replay.
	assistantReasoningReplay bool

	customPayloadValues map[string]any
	customHeaders       map[string]string

	// When set, include prompt_cache_retention in requests that contain a "long"
	// cache hint. This is an OpenAI-specific feature.
	promptCacheRetention string
}

func NewChatCompletionsAPI(accessToken, model string) *ChatCompletionsAPI {
	return &ChatCompletionsAPI{
		accessToken:          accessToken,
		model:                model,
		endpoint:             "https://api.openai.com/v1/chat/completions",
		company:              "OpenAI",
		includeUsage:         true,
		promptCacheRetention: "24h",
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

// WithCacheControlPromptHints encodes CacheHint items as cache_control on
// content parts instead of using prompt_cache_retention.
func (m *ChatCompletionsAPI) WithCacheControlPromptHints() *ChatCompletionsAPI {
	m.cacheControlPromptHints = true
	return m
}

// WithAssistantReasoningReplay encodes assistant Thought items as
// reasoning_details so they can be replayed on OpenAI-compatible providers
// that support preserved reasoning continuity.
func (m *ChatCompletionsAPI) WithAssistantReasoningReplay() *ChatCompletionsAPI {
	m.assistantReasoningReplay = true
	return m
}

// WithPromptCacheRetention enables extended prompt caching with the given
// retention duration (e.g. "24h") when content contains a "long" cache hint.
// This is an OpenAI-specific feature.
func (m *ChatCompletionsAPI) WithPromptCacheRetention(retention string) *ChatCompletionsAPI {
	m.promptCacheRetention = retention
	return m
}

// WithoutPromptCacheRetention disables prompt_cache_retention even when
// content contains a long cache hint.
func (m *ChatCompletionsAPI) WithoutPromptCacheRetention() *ChatCompletionsAPI {
	m.promptCacheRetention = ""
	return m
}

// WithCustomPayloadValue sets a custom key-value pair in the request payload.
// Use this for provider-specific parameters not covered by other methods.
// WARNING: Do not override core fields (stream, model, messages) as this will
// break response parsing or cause unexpected behavior.
func (m *ChatCompletionsAPI) WithCustomPayloadValue(key string, value any) *ChatCompletionsAPI {
	if m.customPayloadValues == nil {
		m.customPayloadValues = make(map[string]any)
	}
	m.customPayloadValues[key] = value
	return m
}

// WithHeader sets an additional HTTP header on requests made by this client.
func (m *ChatCompletionsAPI) WithHeader(key, value string) *ChatCompletionsAPI {
	if m.customHeaders == nil {
		m.customHeaders = make(map[string]string)
	}
	m.customHeaders[key] = value
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

// BuildPayload constructs the request payload without sending it.
// This is exported so wrapper providers (e.g. OpenRouter) can modify the payload
// before calling DoRequest.
func (m *ChatCompletionsAPI) BuildPayload(
	systemPrompt content.Content,
	messages []llms.Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) (map[string]any, error) {
	encodingOptions := chatMessageEncodingOptions{
		cacheControlPromptHints:  m.cacheControlPromptHints,
		assistantReasoningReplay: m.assistantReasoningReplay,
	}

	var apiMessages []Message
	if systemPrompt != nil {
		apiMessages = append(apiMessages, Message{
			Role:    "system",
			Content: ConvertContentWithOptions(systemPrompt, encodingOptions),
		})
	}

	for _, msg := range messages {
		convertedMsgs := MessagesFromLLMWithOptions(msg, encodingOptions)
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

	if !m.cacheControlPromptHints && m.promptCacheRetention != "" && hasLongCacheHint(systemPrompt, messages) {
		payload["prompt_cache_retention"] = m.promptCacheRetention
	}

	for k, v := range m.customPayloadValues {
		payload[k] = v
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
					return nil, fmt.Errorf("openai chat: no allowed tools found in toolbox")
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
					return nil, fmt.Errorf("openai chat: required tool %q not found in toolbox", name)
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
					return nil, fmt.Errorf("openai chat: none of the required tools are present in toolbox")
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

	return payload, nil
}

func (m *ChatCompletionsAPI) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []llms.Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) llms.ProviderStream {
	payload, err := m.BuildPayload(systemPrompt, messages, toolbox, jsonOutputSchema)
	if err != nil {
		return &ChatCompletionsStream{err: err}
	}
	return m.DoRequest(ctx, payload)
}

// DoRequest sends a pre-built payload and returns a streaming response.
// This is exported so wrapper providers (e.g. OpenRouter) can build/modify a
// payload via BuildPayload and then send it.
func (m *ChatCompletionsAPI) DoRequest(ctx context.Context, payload map[string]any) llms.ProviderStream {
	debugger := llms.GetDebugger(ctx)

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return &ChatCompletionsStream{err: fmt.Errorf("error encoding JSON: %w", err)}
	}

	if debugger != nil {
		debugger.RawRequest(m.endpoint, jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return &ChatCompletionsStream{err: fmt.Errorf("error creating request: %w", err)}
	}
	if m.accessToken != "" {
		req.Header.Set("Authorization", fmt.Sprintf("Bearer %s", m.accessToken))
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range m.customHeaders {
		req.Header.Set(k, v)
	}

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

	return &ChatCompletionsStream{ctx: ctx, model: m.model, stream: resp.Body, debugger: debugger}
}

type ChatCompletionsStream struct {
	ctx         context.Context
	model       string
	stream      io.Reader
	debugger    llms.Debugger
	err         error
	message     llms.Message
	lastText    string
	lastThought *content.Thought
	usage       *usage
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

func (s *ChatCompletionsStream) Audio() (string, string) { return "", "" }
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
	if s.lastThought != nil {
		return *s.lastThought
	}
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

func (s *ChatCompletionsStream) emitReasoningDetail(
	rd ReasoningDetail,
	yield func(llms.StreamStatus) bool,
) bool {
	thought, err := s.applyReasoningDetail(rd)
	if err != nil {
		s.err = err
		return false
	}
	if thought == nil {
		return true
	}
	s.lastThought = thought
	return yield(llms.StreamStatusThinking)
}

func (s *ChatCompletionsStream) applyReasoningDetail(rd ReasoningDetail) (*content.Thought, error) {
	aggregate := s.findReasoningThought(rd)
	if aggregate == nil {
		aggregate = &content.Thought{ID: rd.ID}
		s.message.Content = append(s.message.Content, aggregate)
	} else if aggregate.ID == "" && rd.ID != "" {
		aggregate.ID = rd.ID
	}
	s.mergeReasoningMetadata(aggregate, rd)

	switch rd.Type {
	case "reasoning.encrypted":
		if rd.Data == "" {
			return nil, nil
		}
		decodedData, err := base64.StdEncoding.DecodeString(rd.Data)
		if err != nil {
			return nil, fmt.Errorf("error decoding encrypted reasoning data: %w", err)
		}
		aggregate.Encrypted = decodedData
		aggregate.Text = "(Redacted)"
		aggregate.Summary = true
		return &content.Thought{
			ID:        aggregate.ID,
			Text:      aggregate.Text,
			Encrypted: append([]byte(nil), decodedData...),
			Metadata:  cloneThoughtMetadata(aggregate.Metadata),
			Summary:   true,
		}, nil
	case "reasoning.summary":
		if rd.Summary == "" && rd.Signature == "" {
			return nil, nil
		}
		aggregate.Summary = true
		aggregate.Text += rd.Summary
		if rd.Signature != "" {
			aggregate.Signature = rd.Signature
		}
		return &content.Thought{
			ID:        aggregate.ID,
			Text:      rd.Summary,
			Signature: rd.Signature,
			Metadata:  cloneThoughtMetadata(aggregate.Metadata),
			Summary:   true,
		}, nil
	default:
		if rd.Text == "" && rd.Signature == "" {
			return nil, nil
		}
		aggregate.Text += rd.Text
		if rd.Signature != "" {
			aggregate.Signature = rd.Signature
		}
		return &content.Thought{
			ID:        aggregate.ID,
			Text:      rd.Text,
			Signature: rd.Signature,
			Metadata:  cloneThoughtMetadata(aggregate.Metadata),
			Summary:   aggregate.Summary,
		}, nil
	}
}

func (s *ChatCompletionsStream) findReasoningThought(rd ReasoningDetail) *content.Thought {
	var fallback *content.Thought
	for i := len(s.message.Content) - 1; i >= 0; i-- {
		thought, ok := s.message.Content[i].(*content.Thought)
		if !ok {
			continue
		}
		if rd.ID != "" {
			if thought.ID == rd.ID {
				return thought
			}
		}
		if rd.Index != nil {
			if idx, ok := thoughtReasoningIndex(thought); ok && idx == *rd.Index {
				return thought
			}
		}
		if fallback == nil && thoughtMatchesReasoningDetail(thought, rd) {
			fallback = thought
		}
	}
	return fallback
}

func (s *ChatCompletionsStream) mergeReasoningMetadata(thought *content.Thought, rd ReasoningDetail) {
	metadata := reasoningDetailMetadata(rd)
	if len(metadata) == 0 {
		return
	}
	if thought.Metadata == nil {
		thought.Metadata = metadata
		return
	}
	for k, v := range metadata {
		thought.Metadata[k] = v
	}
}

func reasoningDetailMetadata(rd ReasoningDetail) map[string]string {
	var metadata map[string]string
	if rd.Format != "" {
		metadata = map[string]string{"openai:reasoning_format": rd.Format}
	}
	if rd.Index != nil {
		if metadata == nil {
			metadata = make(map[string]string, 1)
		}
		metadata["openai:reasoning_index"] = strconv.Itoa(*rd.Index)
	}
	return metadata
}

func thoughtReasoningIndex(thought *content.Thought) (int, bool) {
	if thought == nil || thought.Metadata == nil {
		return 0, false
	}
	val, ok := thought.Metadata["openai:reasoning_index"]
	if !ok || val == "" {
		return 0, false
	}
	idx, err := strconv.Atoi(val)
	if err != nil {
		return 0, false
	}
	return idx, true
}

func thoughtMatchesReasoningDetail(thought *content.Thought, rd ReasoningDetail) bool {
	if thought == nil {
		return false
	}
	switch rd.Type {
	case "reasoning.encrypted":
		return len(thought.Encrypted) > 0
	case "reasoning.summary":
		return thought.Summary && len(thought.Encrypted) == 0
	default:
		return !thought.Summary && len(thought.Encrypted) == 0
	}
}

func cloneThoughtMetadata(in map[string]string) map[string]string {
	if len(in) == 0 {
		return nil
	}
	out := make(map[string]string, len(in))
	for k, v := range in {
		out[k] = v
	}
	return out
}

func (s *ChatCompletionsStream) Iter() func(yield func(llms.StreamStatus) bool) {
	reader := bufio.NewReader(s.stream)
	var activeToolCallIndex = -1 // Track the index of the tool call being processed
	messageStartYielded := false

	return func(rawYield func(llms.StreamStatus) bool) {
		stopped := false
		yield := func(status llms.StreamStatus) bool {
			if !rawYield(status) {
				stopped = true
				return false
			}
			return true
		}
		defer io.Copy(io.Discard, s.stream)
		defer func() {
			// Ensure ThinkingDone is emitted if the stream ends abnormally
			// (e.g. EOF without [DONE]) while still in thinking state.
			// Don't call yield if the consumer already stopped iterating.
			if s.lastThought != nil {
				s.lastThought = nil
				if !stopped {
					rawYield(llms.StreamStatusThinkingDone)
				}
			}
		}()
		for {
			select {
			case <-s.ctx.Done():
				s.err = s.ctx.Err()
				return
			default:
				// Context OK, keep reading.
			}

			// Read a full logical line using ReadLine to support very long lines.
			var lineBuilder strings.Builder
			for {
				part, isPrefix, err := reader.ReadLine()
				if err != nil {
					if err == io.EOF {
						if lineBuilder.Len() == 0 {
							return
						}
						break
					}
					if s.err == nil {
						s.err = fmt.Errorf("error reading stream: %w", err)
					}
					return
				}
				lineBuilder.Write(part)
				if !isPrefix {
					break
				}
			}

			rawLine := lineBuilder.String()
			if s.debugger != nil && strings.TrimSpace(rawLine) != "" {
				s.debugger.RawEvent([]byte(rawLine))
			}

			line, ok := strings.CutPrefix(rawLine, "data: ")
			if !ok {
				continue
			}
			if line == "[DONE]" {
				// Stream ended. If we were still thinking, emit ThinkingDone.
				if s.lastThought != nil {
					s.lastThought = nil
					if !yield(llms.StreamStatusThinkingDone) {
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
				if s.err == nil {
					s.err = fmt.Errorf("error unmarshalling chunk: %w", err)
				}
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
			if !messageStartYielded {
				messageStartYielded = true
				s.message.ID = chunk.ID
				if !yield(llms.StreamStatusMessageStart) {
					return
				}
			}
			// Handle reasoning/thinking tokens from providers that include them
			// in the OpenAI-compatible streaming format. delta.Reasoning is the
			// legacy plaintext field; reasoning_details carries structured replay data.
			if delta.Reasoning != nil && *delta.Reasoning != "" {
				if !s.emitReasoningDetail(ReasoningDetail{
					Type: "reasoning.text",
					Text: *delta.Reasoning,
				}, yield) {
					return
				}
			}
			for _, rd := range delta.ReasoningDetails {
				if delta.Reasoning != nil && *delta.Reasoning != "" && rd.Text != "" {
					rd.Text = ""
				}
				if !s.emitReasoningDetail(rd, yield) {
					return
				}
			}

			// Content is nullable string in delta
			if delta.Content != nil && *delta.Content != "" {
				// If we were thinking and now got content, emit ThinkingDone.
				if s.lastThought != nil {
					s.lastThought = nil
					if !yield(llms.StreamStatusThinkingDone) {
						return
					}
				}
				s.lastText = *delta.Content
				s.message.Content.Append(s.lastText)
				if !yield(llms.StreamStatusText) {
					return
				}
			}

			// Handle Tool Calls Delta
			if len(delta.ToolCalls) > 0 {
				// If we were thinking and now got tool calls, emit ThinkingDone.
				if s.lastThought != nil {
					s.lastThought = nil
					if !yield(llms.StreamStatusThinkingDone) {
						return
					}
				}
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
			// Check if the overall message is finished
			if chunk.Choices[0].FinishReason != nil {
				switch *chunk.Choices[0].FinishReason {
				case "tool_calls":
					if activeToolCallIndex != -1 {
						if !yield(llms.StreamStatusToolCallReady) {
							return // Abort if yield fails
						}
						activeToolCallIndex = -1 // Reset active tool call
					}
				case "length":
					s.err = fmt.Errorf("%w (finish_reason=%q)", llms.ErrOutputTruncated, *chunk.Choices[0].FinishReason)
					// Do not return here. The stream may still deliver a usage chunk
					// (with empty choices) that populates s.usage. The loop will exit
					// naturally when the stream ends, and s.err is returned via Err().
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
