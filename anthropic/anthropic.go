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

	"github.com/metalim/jsonmap"
	"golang.org/x/oauth2"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

type Model struct {
	apiKey              string
	model               string
	endpoint            string
	company             string
	maxTokens           int
	maxThinkingTokens   int
	adaptiveThinking    bool
	thinkingDisplay     ThinkingDisplay
	effort              Effort
	customPayloadValues map[string]any
	betaFeatures        []string
	httpClient          *http.Client

	// Vertex AI fields
	vertexAI    bool
	tokenSource oauth2.TokenSource
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

// WithVertexAI configures the model to use Google Cloud Vertex AI as the
// endpoint for Anthropic Claude models. Authentication is handled via the
// provided OAuth2 token source (typically from Google Application Default
// Credentials). The model name is embedded in the URL path, and
// anthropic_version is sent in the request body instead of as a header.
func (m *Model) WithVertexAI(ts oauth2.TokenSource, projectID, location string) *Model {
	m.vertexAI = true
	m.tokenSource = ts
	m.company = "Google"
	if location == "global" {
		m.endpoint = fmt.Sprintf(
			"https://aiplatform.googleapis.com/v1/projects/%s/locations/global/publishers/anthropic/models/%s:streamRawPredict",
			projectID, m.model,
		)
	} else {
		m.endpoint = fmt.Sprintf(
			"https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/anthropic/models/%s:streamRawPredict",
			location, projectID, location, m.model,
		)
	}
	return m
}

func (m *Model) WithMaxTokens(maxTokens int) *Model {
	m.maxTokens = maxTokens
	return m
}

func (m *Model) WithThinking(budgetTokens int) *Model {
	m.maxThinkingTokens = budgetTokens
	m.adaptiveThinking = false
	return m
}

// WithAdaptiveThinking enables adaptive thinking mode, where the model
// dynamically decides when and how much to think based on query complexity.
// This is only supported on Claude Opus 4.6+.
func (m *Model) WithAdaptiveThinking() *Model {
	m.adaptiveThinking = true
	return m
}

// WithThinkingDisplay controls whether (and how) the API returns thinking
// content in the response. On Claude Opus 4.7 the default is
// [ThinkingDisplayOmitted] (empty thinking blocks arrive in the stream);
// set to [ThinkingDisplaySummarized] to receive populated thinking content
// so thinking blocks can be replayed back on subsequent turns. Must be
// combined with [Model.WithAdaptiveThinking] or [Model.WithThinking] for
// the display value to be sent.
func (m *Model) WithThinkingDisplay(display ThinkingDisplay) *Model {
	m.thinkingDisplay = display
	return m
}

// WithEffort controls how many tokens Claude uses when responding, trading off
// between response thoroughness and token efficiency. Supported on Claude Opus
// 4.5 and 4.6. For Opus 4.6, this is the recommended way to control thinking
// depth (combine with WithAdaptiveThinking for best results). EffortMax is only
// available on Opus 4.6.
func (m *Model) WithEffort(effort Effort) *Model {
	m.effort = effort
	return m
}

// WithCustomPayloadValue sets a custom key-value pair in the request payload.
// This is useful for setting provider-specific fields that are not directly
// supported by the library (e.g. "speed": "fast" for Anthropic fast mode).
// WARNING: Do not override core fields (stream, model, messages, max_tokens)
// as this will break response parsing or cause unexpected behavior.
func (m *Model) WithCustomPayloadValue(key string, value any) *Model {
	if m.customPayloadValues == nil {
		m.customPayloadValues = make(map[string]any)
	}
	m.customPayloadValues[key] = value
	return m
}

func (m *Model) Company() string {
	return m.company
}

func (m *Model) Model() string {
	return m.model
}

func (m *Model) SetHTTPClient(client *http.Client) {
	m.httpClient = client
}

type schemaContainerKind uint8

const (
	schemaContainerNone schemaContainerKind = iota
	schemaContainerMap
	schemaContainerArray
	schemaContainerDirect
)

func schemaChildContainerKind(key string) schemaContainerKind {
	switch key {
	case "properties", "patternProperties", "dependentSchemas", "$defs", "definitions", "dependencies":
		return schemaContainerMap
	case "anyOf", "allOf", "oneOf", "prefixItems":
		return schemaContainerArray
	case "items", "additionalProperties", "additionalItems", "contains", "propertyNames", "not", "if", "then", "else", "unevaluatedItems", "unevaluatedProperties":
		return schemaContainerDirect
	default:
		return schemaContainerNone
	}
}

// normalizeOutputSchemaForAnthropic returns a deep-normalized schema for Anthropic
// structured outputs without mutating the caller's schema.
func normalizeOutputSchemaForAnthropic(schema *tools.ValueSchema) (any, error) {
	// Round-trip through JSON and decode into jsonmap so object key order is preserved.
	data, err := json.Marshal(schema)
	if err != nil {
		return nil, fmt.Errorf("failed to marshal schema: %w", err)
	}

	decoded := jsonmap.New()
	if err := json.Unmarshal(data, decoded); err != nil {
		return nil, fmt.Errorf("failed to unmarshal schema into ordered map: %w", err)
	}

	normalizeSchemaNode(decoded)
	return decoded, nil
}

func normalizeSchemaNode(node any) {
	switch n := node.(type) {
	case *jsonmap.Map:
		normalizeSchemaObject(n)
	case []any:
		for _, item := range n {
			normalizeSchemaNode(item)
		}
	}
}

func normalizeSchemaObject(node *jsonmap.Map) {
	// Anthropic requires additionalProperties: false on all object schemas.
	isObjectType := false
	if rawType, ok := node.Get("type"); ok {
		isObjectType = schemaTypeIncludesObject(rawType)
	}
	if isObjectType || jsonMapLooksLikeObject(node) {
		node.Set("additionalProperties", false)
	}

	for _, key := range node.Keys() {
		raw, ok := node.Get(key)
		if !ok {
			continue
		}
		switch schemaChildContainerKind(key) {
		case schemaContainerMap:
			normalizeSchemaMapContainer(raw)
		case schemaContainerArray:
			normalizeSchemaArrayContainer(raw)
		case schemaContainerDirect:
			normalizeSchemaNode(raw)
		}
	}
}

func normalizeSchemaMapContainer(raw any) {
	v, ok := raw.(*jsonmap.Map)
	if !ok {
		return
	}
	for _, key := range v.Keys() {
		child, ok := v.Get(key)
		if !ok {
			continue
		}
		normalizeSchemaNode(child)
	}
}

func normalizeSchemaArrayContainer(raw any) {
	switch v := raw.(type) {
	case []any:
		for _, child := range v {
			normalizeSchemaNode(child)
		}
	}
}

func jsonMapLooksLikeObject(node *jsonmap.Map) bool {
	for _, key := range []string{"properties", "patternProperties", "required", "dependencies", "dependentSchemas"} {
		if _, ok := node.Get(key); ok {
			return true
		}
	}
	return false
}

func schemaTypeIncludesObject(raw any) bool {
	switch t := raw.(type) {
	case string:
		return t == "object"
	case []any:
		for _, v := range t {
			if s, ok := v.(string); ok && s == "object" {
				return true
			}
		}
	case []string:
		for _, s := range t {
			if s == "object" {
				return true
			}
		}
	}
	return false
}

func (m *Model) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []llms.Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) llms.ProviderStream {
	debugger := llms.GetDebugger(ctx)

	var apiMessages []message
	for _, msg := range messages {
		apiMessages = append(apiMessages, messageFromLLM(msg))
	}

	maxTokens := m.maxTokens + m.maxThinkingTokens
	if m.adaptiveThinking {
		// With adaptive thinking the model decides how much to think, so we
		// only need the output token budget.
		maxTokens = m.maxTokens
	}

	payload := map[string]any{
		"messages":   apiMessages,
		"stream":     true,
		"max_tokens": maxTokens,
	}

	if m.vertexAI {
		// Vertex AI embeds the model in the URL path and requires
		// anthropic_version in the request body.
		payload["anthropic_version"] = "vertex-2023-10-16"
		// Vertex AI requires beta features as a body parameter rather than
		// an HTTP header. Sending them as headers causes 400 errors for
		// certain betas (e.g. context-1m-2025-08-07).
		if len(m.betaFeatures) > 0 {
			payload["anthropic_beta"] = m.betaFeatures
		}
	} else {
		payload["model"] = m.model
	}

	if systemPrompt != nil {
		payload["system"] = contentFromLLM(systemPrompt)
	}

	outputConfig := map[string]any{}

	if jsonOutputSchema != nil {
		schema, err := normalizeOutputSchemaForAnthropic(jsonOutputSchema)
		if err != nil {
			return &Stream{err: fmt.Errorf("anthropic: failed to normalize JSON output schema: %w", err)}
		}
		outputConfig["format"] = map[string]any{
			"type":   "json_schema",
			"schema": schema,
		}
	}

	if toolbox != nil {
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

	if m.adaptiveThinking {
		thinking := map[string]any{
			"type": "adaptive",
		}
		if m.thinkingDisplay != "" {
			thinking["display"] = string(m.thinkingDisplay)
		}
		payload["thinking"] = thinking
	} else if m.maxThinkingTokens > 0 {
		thinking := map[string]any{
			"type":          "enabled",
			"budget_tokens": m.maxThinkingTokens,
		}
		if m.thinkingDisplay != "" {
			thinking["display"] = string(m.thinkingDisplay)
		}
		payload["thinking"] = thinking
	}

	if m.effort != "" {
		outputConfig["effort"] = m.effort
	}

	for k, v := range m.customPayloadValues {
		payload[k] = v
	}

	if len(outputConfig) > 0 {
		payload["output_config"] = outputConfig
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return &Stream{err: fmt.Errorf("error encoding JSON: %w", err)}
	}

	if debugger != nil {
		debugger.RawRequest(m.endpoint, jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return &Stream{err: fmt.Errorf("error creating request: %w", err)}
	}
	req.Header.Set("Content-Type", "application/json")

	if m.vertexAI {
		// Vertex AI uses OAuth2 Bearer tokens instead of API keys.
		token, err := m.tokenSource.Token()
		if err != nil {
			return &Stream{err: fmt.Errorf("anthropic: failed to get OAuth2 token: %w", err)}
		}
		req.Header.Set("Authorization", "Bearer "+token.AccessToken)
	} else {
		req.Header.Set("X-API-Key", m.apiKey)
		req.Header.Set("anthropic-version", "2023-06-01")
	}

	// Add beta feature headers if any are configured.
	// For Vertex AI, betas are already included in the request body as
	// anthropic_beta, so we only add them as headers for direct Anthropic.
	if !m.vertexAI {
		for _, beta := range m.betaFeatures {
			req.Header.Add("anthropic-beta", beta)
		}
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

	return &Stream{ctx: ctx, model: m.model, stream: resp.Body, debugger: debugger}
}

type Stream struct {
	ctx         context.Context
	model       string
	stream      io.Reader
	err         error
	message     llms.Message
	lastText    string
	lastThought *content.Thought
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
	reader := bufio.NewReader(s.stream)
	return func(yield func(llms.StreamStatus) bool) {
		defer io.Copy(io.Discard, s.stream)
		lastToolCallIndex := -1
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

			// Read a full logical line using ReadLine to support very long lines
			// (e.g. redacted_thinking blocks with large base64 data).
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
					s.err = fmt.Errorf("error reading stream: %w", err)
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
				// Check stop reason
				if event.Delta.StopReason != "" &&
					event.Delta.StopReason != "tool_use" &&
					event.Delta.StopReason != "end_turn" {
					if event.Delta.StopReason == "max_tokens" {
						s.err = fmt.Errorf("%w (stop_reason=%q)", llms.ErrOutputTruncated, event.Delta.StopReason)
					} else {
						s.err = fmt.Errorf("unexpected stop reason: %q", event.Delta.StopReason)
					}
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
