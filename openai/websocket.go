package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"slices"
	"sync"

	"github.com/coder/websocket"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

// WebSocketResponsesAPI implements llms.Provider using the OpenAI Responses API
// over a persistent WebSocket connection. This reduces per-turn overhead in
// tool-heavy workflows by keeping a connection open and using
// previous_response_id to send only incremental input.
type WebSocketResponsesAPI struct {
	responsesConfig

	accessToken string
	endpoint    string
	company     string
	debugger    llms.Debugger

	// WebSocket state
	conn         *websocket.Conn
	externalConn bool
	mu           sync.Mutex

	// Chaining state
	lastResponseID   string
	lastMessageCount int
}

// NewWebSocketResponsesAPI creates a new WebSocket-based provider for the
// OpenAI Responses API. The WebSocket connection is established lazily on the
// first Generate() call. Call Close() when done.
func NewWebSocketResponsesAPI(accessToken, model string) *WebSocketResponsesAPI {
	return &WebSocketResponsesAPI{
		responsesConfig: responsesConfig{
			model:             model,
			temperature:       1.0,
			parallelToolCalls: true,
			store:             true,
			truncation:        "disabled",
		},
		accessToken: accessToken,
		endpoint:    "wss://api.openai.com/v1/responses",
		company:     "OpenAI",
	}
}

// DialResponsesWebSocket connects to the OpenAI Responses API WebSocket
// endpoint. The returned connection can be passed to WithConn().
func DialResponsesWebSocket(ctx context.Context, accessToken string, opts ...DialOption) (*websocket.Conn, error) {
	cfg := dialConfig{endpoint: "wss://api.openai.com/v1/responses"}
	for _, o := range opts {
		o(&cfg)
	}
	conn, _, err := websocket.Dial(ctx, cfg.endpoint, &websocket.DialOptions{
		HTTPHeader: http.Header{
			"Authorization": []string{"Bearer " + accessToken},
		},
	})
	if err != nil {
		return nil, fmt.Errorf("websocket dial: %w", err)
	}
	// Allow large messages (model responses can be large).
	conn.SetReadLimit(64 * 1024 * 1024)
	return conn, nil
}

// DialOption configures DialResponsesWebSocket.
type DialOption func(*dialConfig)

type dialConfig struct {
	endpoint string
}

// WithDialEndpoint overrides the default WebSocket endpoint.
func WithDialEndpoint(endpoint string) DialOption {
	return func(c *dialConfig) { c.endpoint = endpoint }
}

// WithEndpoint sets the WebSocket endpoint and company name.
func (m *WebSocketResponsesAPI) WithEndpoint(endpoint, company string) *WebSocketResponsesAPI {
	m.endpoint = endpoint
	m.company = company
	return m
}

// WithConn sets an externally managed WebSocket connection. The provider will
// not close this connection when Close() is called.
func (m *WebSocketResponsesAPI) WithConn(conn *websocket.Conn) *WebSocketResponsesAPI {
	m.conn = conn
	m.externalConn = true
	return m
}

func (m *WebSocketResponsesAPI) WithMaxOutputTokens(n int) *WebSocketResponsesAPI {
	m.maxOutputTokens = n
	return m
}

// WithContextCompaction enables server-side context compaction (see
// [ContextCompaction]).
func (m *WebSocketResponsesAPI) WithContextCompaction(c ContextCompaction) *WebSocketResponsesAPI {
	m.compaction = &c
	return m
}

func (m *WebSocketResponsesAPI) WithThinking(effort Effort) *WebSocketResponsesAPI {
	m.reasoningEffort = effort
	return m
}

func (m *WebSocketResponsesAPI) WithTemperature(t float64) *WebSocketResponsesAPI {
	m.temperature = t
	return m
}

func (m *WebSocketResponsesAPI) WithTool(tool ResponseTool) *WebSocketResponsesAPI {
	m.specialTools = append(m.specialTools, tool)
	return m
}

func (m *WebSocketResponsesAPI) WithTopP(topP float64) *WebSocketResponsesAPI {
	m.topP = &topP
	return m
}

func (m *WebSocketResponsesAPI) WithTopLogprobs(n int) *WebSocketResponsesAPI {
	m.topLogprobs = n
	return m
}

func (m *WebSocketResponsesAPI) WithParallelToolCalls(parallel bool) *WebSocketResponsesAPI {
	m.parallelToolCalls = parallel
	return m
}

func (m *WebSocketResponsesAPI) WithServiceTier(tier string) *WebSocketResponsesAPI {
	m.serviceTier = tier
	return m
}

func (m *WebSocketResponsesAPI) WithStore(store bool) *WebSocketResponsesAPI {
	m.store = store
	return m
}

func (m *WebSocketResponsesAPI) WithTruncation(truncation string) *WebSocketResponsesAPI {
	m.truncation = truncation
	return m
}

func (m *WebSocketResponsesAPI) WithUser(user string) *WebSocketResponsesAPI {
	m.user = user
	return m
}

func (m *WebSocketResponsesAPI) WithMetadata(metadata map[string]string) *WebSocketResponsesAPI {
	m.metadata = metadata
	return m
}

func (m *WebSocketResponsesAPI) WithPromptCacheKey(key string) *WebSocketResponsesAPI {
	m.promptCacheKey = key
	return m
}

func (m *WebSocketResponsesAPI) WithVerbosity(verbosity Verbosity) *WebSocketResponsesAPI {
	m.verbosity = verbosity
	return m
}

// WithDebugger sets a default debugger for this provider. It is used by
// Warmup and Generate as a fallback when the context does not already carry
// a debugger (via llms.WithDebugger). When used through LLM.WithDebugger,
// the debugger is injected into the context automatically, so calling this
// method is only necessary when invoking Warmup or Generate directly on the
// provider.
func (m *WebSocketResponsesAPI) WithDebugger(d llms.Debugger) *WebSocketResponsesAPI {
	m.debugger = d
	return m
}

func (m *WebSocketResponsesAPI) Company() string { return m.company }
func (m *WebSocketResponsesAPI) Model() string   { return m.model }

func (m *WebSocketResponsesAPI) SetHTTPClient(_ *http.Client) {} // no-op for WebSocket

// Close closes the WebSocket connection. If the connection was provided
// externally via WithConn, this is a no-op.
func (m *WebSocketResponsesAPI) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.conn != nil && !m.externalConn {
		err := m.conn.Close(websocket.StatusNormalClosure, "")
		m.conn = nil
		return err
	}
	return nil
}

// ResetChain clears the chaining state so the next Generate() sends a full
// payload. Use this when starting a new conversation on the same connection.
func (m *WebSocketResponsesAPI) ResetChain() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.lastResponseID = ""
	m.lastMessageCount = 0
}

// Warmup sends a request with generate:false over the WebSocket, which
// pre-loads tools and instructions on the server side. Returns the response ID
// that can be used for faster first turns. The provided context should have a
// timeout to avoid blocking indefinitely.
//
// Warmup must not be called concurrently with Generate or while a stream from
// Generate is being iterated, as they share the same WebSocket connection.
func (m *WebSocketResponsesAPI) Warmup(ctx context.Context, instructions string, toolbox *tools.Toolbox) (string, error) {
	debugger := llms.GetDebugger(ctx)
	if debugger == nil {
		debugger = m.debugger
	}

	m.mu.Lock()
	if err := m.ensureConnected(ctx); err != nil {
		m.mu.Unlock()
		return "", err
	}

	// Build warmup payload with generate:false. We pass nil input here
	// because warmup doesn't send conversation messages; the empty input
	// is set explicitly below. tool_choice may be included alongside
	// tools, which is harmless since generate:false skips generation.
	payload, err := m.buildResponsesPayload(nil, instructions, toolbox, nil)
	if err != nil {
		m.mu.Unlock()
		return "", fmt.Errorf("warmup: payload: %w", err)
	}
	payload["type"] = "response.create"
	payload["input"] = []any{}
	payload["generate"] = false

	jsonData, err := json.Marshal(payload)
	if err != nil {
		m.mu.Unlock()
		return "", fmt.Errorf("warmup: marshal: %w", err)
	}

	if debugger != nil {
		debugger.RawRequest(m.endpoint, jsonData)
	}

	if err := m.conn.Write(ctx, websocket.MessageText, jsonData); err != nil {
		m.mu.Unlock()
		return "", fmt.Errorf("warmup: write: %w", err)
	}

	conn := m.conn
	m.mu.Unlock()

	// Read events until response.completed (mutex released so other calls
	// are not blocked during the network round-trip).
	var responseID string
	for {
		_, data, err := conn.Read(ctx)
		if err != nil {
			return "", fmt.Errorf("warmup: read: %w", err)
		}
		if debugger != nil {
			debugger.RawEvent(data)
		}
		var event ResponseStreamEvent
		if err := json.Unmarshal(data, &event); err != nil {
			return "", fmt.Errorf("warmup: unmarshal: %w", err)
		}
		switch event.Type {
		case "response.created":
			var resp struct {
				ID string `json:"id"`
			}
			if err := json.Unmarshal(event.Response, &resp); err == nil {
				responseID = resp.ID
			}
		case "response.completed":
			if responseID == "" {
				return "", fmt.Errorf("warmup: no response ID received")
			}
			m.mu.Lock()
			m.lastResponseID = responseID
			m.lastMessageCount = 0
			m.mu.Unlock()
			return responseID, nil
		case "error":
			if event.Error != nil {
				return "", fmt.Errorf("warmup: %w", event.Error.err(event.Status))
			}
			return "", fmt.Errorf("warmup: unknown error")
		}
	}
}

// Generate sends a request over the WebSocket and returns a stream of events.
// Callers must fully iterate (or abandon via context cancellation) one stream
// before calling Generate again, as they share the same WebSocket connection.
func (m *WebSocketResponsesAPI) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []llms.Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) llms.ProviderStream {
	debugger := llms.GetDebugger(ctx)
	if debugger == nil {
		debugger = m.debugger
	}

	m.mu.Lock()
	defer m.mu.Unlock()

	if err := m.ensureConnected(ctx); err != nil {
		return newWebSocketStreamError(err)
	}

	// Build instructions from system prompt.
	var instructions string
	var systemInputs []ResponseInput
	if text, ok := systemPrompt.AsString(); ok {
		instructions = text
	} else {
		systemMsg := llms.Message{Role: "system", Content: systemPrompt}
		var err error
		systemInputs, err = convertMessageToInput(systemMsg, inputConversion{})
		if err != nil {
			return newWebSocketStreamError(fmt.Errorf("websocket: failed to convert system message: %w", err))
		}
	}

	// request builds the response.create envelope for msgs, chained to
	// previousResponseID when it is set.
	conversion := newInputConversion(messages, m.endpoint)
	request := func(msgs []llms.Message, previousResponseID string) ([]byte, error) {
		input := slices.Clone(systemInputs)
		for _, msg := range msgs {
			msgInputs, err := convertMessageToInput(msg, conversion)
			if err != nil {
				return nil, fmt.Errorf("websocket: failed to convert message role=%s: %w", msg.Role, err)
			}
			input = append(input, msgInputs...)
		}
		return m.buildRequestEnvelope(input, instructions, previousResponseID, toolbox, jsonOutputSchema)
	}
	send := func(conn *websocket.Conn, data []byte) error {
		if debugger != nil {
			debugger.RawRequest(m.endpoint, data)
		}
		return conn.Write(ctx, websocket.MessageText, data)
	}

	// Chain to the last response when the conversation continues it:
	// incrementally after a turn, or with every message after a warmup.
	var chainedTo string
	unsent := messages
	if m.lastResponseID != "" && m.lastMessageCount > 0 &&
		len(messages) > m.lastMessageCount &&
		messages[m.lastMessageCount].Role == "assistant" {
		chainedTo = m.lastResponseID
		unsent = messages[m.lastMessageCount+1:]
	} else if m.lastResponseID != "" && m.lastMessageCount == 0 {
		chainedTo = m.lastResponseID
	}

	jsonData, err := request(unsent, chainedTo)
	if err != nil {
		return newWebSocketStreamError(err)
	}

	if err := send(m.conn, jsonData); err != nil {
		// If write fails and we manage the connection, try reconnect + retry.
		if m.externalConn {
			return newWebSocketStreamError(fmt.Errorf("websocket: write: %w", err))
		}
		// Close old connection to avoid leak.
		m.conn.Close(websocket.StatusGoingAway, "reconnecting")
		m.conn = nil
		m.lastResponseID = ""
		m.lastMessageCount = 0
		if reconnErr := m.ensureConnected(ctx); reconnErr != nil {
			return newWebSocketStreamError(fmt.Errorf("websocket: reconnect failed: %w", reconnErr))
		}
		// The new connection cannot chain: send the whole conversation.
		chainedTo = ""
		jsonData, err = request(messages, "")
		if err != nil {
			return newWebSocketStreamError(fmt.Errorf("websocket: reconnect: %w", err))
		}
		if err := send(m.conn, jsonData); err != nil {
			return newWebSocketStreamError(fmt.Errorf("websocket: write after reconnect: %w", err))
		}
	}

	conn := m.conn
	msgCount := len(messages)
	stream := &WebSocketStream{
		responsesEventProcessor: responsesEventProcessor{
			debugger:          debugger,
			lastThought:       &content.Thought{},
			reasoningEndpoint: reasoningEndpoint(m.endpoint),
		},
		ctx:  ctx,
		conn: conn,
		onDone: func(responseID string) {
			m.mu.Lock()
			defer m.mu.Unlock()
			if responseID != "" {
				m.lastResponseID = responseID
				m.lastMessageCount = msgCount
			}
		},
	}
	if chainedTo != "" {
		// The server answers previous_response_not_found when it no longer
		// holds the chained response: a response created with store=false
		// lives only in the connection's cache, and a stored one only in the
		// organization that created it. The full conversation depends on
		// neither, so the stream sends it instead.
		stream.resendUnchained = func() error {
			jsonData, err := request(messages, "")
			if err != nil {
				return err
			}
			if err := send(conn, jsonData); err != nil {
				return fmt.Errorf("websocket: write unchained request: %w", err)
			}
			return nil
		}
	}
	return stream
}

// buildRequestEnvelope builds the full response.create JSON envelope from the
// given parameters. It handles text format, tools, tool choice, and wrapping.
func (m *WebSocketResponsesAPI) buildRequestEnvelope(
	input []ResponseInput,
	instructions string,
	previousResponseID string,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) ([]byte, error) {
	payload, err := m.buildResponsesPayload(input, instructions, toolbox, jsonOutputSchema)
	if err != nil {
		return nil, err
	}

	if previousResponseID != "" {
		payload["previous_response_id"] = previousResponseID
	}

	// WebSocket format: all fields at top level alongside "type".
	payload["type"] = "response.create"

	data, err := json.Marshal(payload)
	if err != nil {
		return nil, fmt.Errorf("websocket: marshal: %w", err)
	}
	return data, nil
}

// ensureConnected dials the WebSocket if not already connected. Must be called
// with m.mu held.
func (m *WebSocketResponsesAPI) ensureConnected(ctx context.Context) error {
	if m.conn != nil {
		return nil
	}
	conn, err := DialResponsesWebSocket(ctx, m.accessToken, WithDialEndpoint(m.endpoint))
	if err != nil {
		return err
	}
	m.conn = conn
	return nil
}

// buildResponsesToolsArray converts a toolbox and special tools into the
// JSON-serializable tool array used by the Responses API. Shared by both
// the SSE and WebSocket providers.
func buildResponsesToolsArray(specialTools []ResponseTool, toolbox *tools.Toolbox) ([]any, error) {
	var toolsArr []any
	for _, t := range specialTools {
		toolsArr = append(toolsArr, t)
	}
	if toolbox == nil {
		return toolsArr, nil
	}
	for _, t := range toolbox.All() {
		switch g := t.Grammar().(type) {
		case tools.JSONGrammar:
			if schema := g.Schema(); schema != nil {
				toolsArr = append(toolsArr, FunctionTool{
					Type:        "function",
					Name:        schema.Name,
					Description: schema.Description,
					Parameters:  &schema.Parameters,
					Strict:      true,
				})
			}
		case tools.TextGrammar:
			toolsArr = append(toolsArr, map[string]any{
				"type":        "custom",
				"name":        t.FuncName(),
				"description": t.Description(),
				"format":      map[string]any{"type": "text"},
			})
		case tools.LarkGrammar:
			toolsArr = append(toolsArr, map[string]any{
				"type":        "custom",
				"name":        t.FuncName(),
				"description": t.Description(),
				"format": map[string]any{
					"type":       "grammar",
					"definition": g.Definition,
					"syntax":     "lark",
				},
			})
		case tools.RegexGrammar:
			toolsArr = append(toolsArr, map[string]any{
				"type":        "custom",
				"name":        t.FuncName(),
				"description": t.Description(),
				"format": map[string]any{
					"type":       "grammar",
					"definition": g.Definition,
					"syntax":     "regex",
				},
			})
		default:
			return nil, fmt.Errorf("openai responses: unsupported tool grammar type %T", g)
		}
	}
	return toolsArr, nil
}

// buildToolChoice converts a tools.Choice into the appropriate tool_choice
// value for the Responses API.
func buildToolChoice(choice tools.Choice, toolsArr []any) (any, error) {
	switch choice.Mode {
	case tools.ChoiceAllowOnly:
		if len(choice.AllowedTools) == 0 {
			return "none", nil
		}
		if !anyToolExists(choice.AllowedTools, toolsArr) {
			return nil, fmt.Errorf("openai responses: no allowed tools found in toolbox")
		}
		var entries []any
		for _, n := range choice.AllowedTools {
			entries = append(entries, map[string]any{"type": declaredToolType(n, toolsArr), "name": n})
		}
		return AllowedToolsToolChoice{Type: "allowed_tools", Mode: "auto", Tools: entries}, nil

	case tools.ChoiceRequireOneOf:
		switch len(choice.AllowedTools) {
		case 0:
			return "none", nil
		case 1:
			name := choice.AllowedTools[0]
			if !anyToolExists([]string{name}, toolsArr) {
				return nil, fmt.Errorf("openai responses: required tool %q not found in toolbox", name)
			}
			return map[string]any{"type": declaredToolType(name, toolsArr), "name": name}, nil
		default:
			if !anyToolExists(choice.AllowedTools, toolsArr) {
				return nil, fmt.Errorf("openai responses: none of the required tools are present in toolbox")
			}
			var entries []any
			for _, n := range choice.AllowedTools {
				entries = append(entries, map[string]any{"type": declaredToolType(n, toolsArr), "name": n})
			}
			return AllowedToolsToolChoice{Type: "allowed_tools", Mode: "required", Tools: entries}, nil
		}

	default:
		return "auto", nil
	}
}

// declaredToolType resolves the wire type a named tool is declared with in
// toolsArr, so a forced or allow-listed tool_choice references custom
// (grammar/text) tools as {"type":"custom"} instead of mislabeling them as
// functions, which the API rejects. Unknown names default to "function",
// matching the previous behavior for hosted/special tools.
func declaredToolType(name string, toolsArr []any) string {
	for _, it := range toolsArr {
		switch v := it.(type) {
		case FunctionTool:
			if v.Name == name {
				return "function"
			}
		case map[string]any:
			if n, ok := v["name"].(string); ok && n == name {
				if typ, ok := v["type"].(string); ok && typ != "" {
					return typ
				}
			}
		}
	}
	return "function"
}

// anyToolExists checks whether at least one of the named tools exists in
// toolsArr.
func anyToolExists(names []string, toolsArr []any) bool {
	for _, n := range names {
		for _, it := range toolsArr {
			switch v := it.(type) {
			case FunctionTool:
				if v.Name == n {
					return true
				}
			case map[string]any:
				if name, ok := v["name"].(string); ok && name == n {
					return true
				}
			}
		}
	}
	return false
}
