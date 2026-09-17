// Package typesafe is a go-llms provider for TypeSafe's System One models
// (https://docs.typesafe.ai). A System One model does not generate text: it
// reads a state and answers typed questions with calibrated probabilities.
// The provider maps the go-llms contract onto that shape.
//
//   - The system prompt and messages become the request state, as a JSON
//     object with a "system" string and a "messages" array.
//   - The JSON output schema becomes the questions: one per property, with the
//     property description as the question. See [questionsFromSchema] for the
//     supported property types.
//   - The answers are rendered as the JSON object the schema describes and
//     delivered as one text chunk; the full response with probabilities and
//     confidence stays readable through [Stream.Response].
//
// A boolean property is rendered as true when the model's probability is at
// least 0.5. A number property is read as the probability that its description
// holds: the rendered value is that probability between 0 and 1, not a
// quantity the model counted or estimated. A string property with an enum is
// rendered as the chosen member.
//
// A property listed in the schema's "required" must be answered, or the
// request fails; an unanswered optional property is left out of the rendered
// object. Answers for names the provider never asked about are ignored.
//
// Tools are not supported, and a request without an output schema is
// rejected, because there is nothing to ask.
package typesafe

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"

	"github.com/metalim/jsonmap"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

// DefaultEndpoint is the System One evaluation endpoint.
const DefaultEndpoint = "https://api.typesafe.ai/v1/systemone"

// ErrToolsUnsupported is returned when a toolbox with tools, a tool call, or a
// tool result reaches the provider.
var ErrToolsUnsupported = errors.New("typesafe: tools are not supported by System One models")

// ErrNonTextContent is returned when a message carries an image, audio, or
// video item. The state accepts text and structured JSON; images, audio, and
// video are rejected because the model cannot read them.
var ErrNonTextContent = errors.New("typesafe: System One models accept text and structured JSON only")

type Model struct {
	apiKey     string
	model      string
	endpoint   string
	company    string
	httpClient *http.Client
}

// New creates a provider for the given model, such as "jev-latest" or a
// pinned version like "jev-1.13.0".
func New(apiKey, model string) *Model {
	return &Model{
		apiKey:   apiKey,
		model:    model,
		endpoint: DefaultEndpoint,
		company:  "TypeSafe",
	}
}

// WithEndpoint sets the endpoint (and company name) so compatible endpoints
// can be used.
func (m *Model) WithEndpoint(endpoint, company string) *Model {
	m.endpoint = endpoint
	m.company = company
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

func (m *Model) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []llms.Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) llms.ProviderStream {
	debugger := llms.GetDebugger(ctx)

	if toolbox != nil && len(toolbox.All()) > 0 {
		return &Stream{err: ErrToolsUnsupported}
	}

	state, err := stateFromLLM(systemPrompt, messages)
	if err != nil {
		return &Stream{err: err}
	}

	bindings, err := questionsFromSchema(jsonOutputSchema)
	if err != nil {
		return &Stream{err: err}
	}
	questions := make(map[string]Question, len(bindings))
	for _, b := range bindings {
		questions[b.name] = b.question
	}

	jsonData, err := json.Marshal(request{State: state, Model: m.model, Questions: questions})
	if err != nil {
		return &Stream{err: fmt.Errorf("error encoding JSON: %w", err)}
	}

	if debugger != nil {
		debugger.RawRequest(m.endpoint, jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, m.endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return &Stream{err: fmt.Errorf("error creating request: %w", err)}
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+m.apiKey)

	client := m.httpClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(req)
	if err != nil {
		return &Stream{err: fmt.Errorf("error making request: %w", err)}
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return &Stream{err: fmt.Errorf("error reading response: %w", err)}
	}
	if debugger != nil {
		debugger.RawEvent(body)
	}
	if resp.StatusCode != http.StatusOK {
		return &Stream{err: httpError(resp, body)}
	}

	var response Response
	if err := json.Unmarshal(body, &response); err != nil {
		return &Stream{err: fmt.Errorf("typesafe: error decoding response: %w", err)}
	}

	answerJSON, err := renderAnswers(bindings, response.Answers)
	if err != nil {
		// The request was billed, so keep the response reachable even though
		// no answer could be rendered from it.
		return &Stream{response: &response, err: err}
	}

	return &Stream{
		response: &response,
		text:     string(answerJSON),
		message: llms.Message{
			Role:    "assistant",
			Content: content.FromRawJSON(answerJSON),
		},
	}
}

// httpError maps a non-200 response onto llms.HTTPError. The API documents
// 401, 422, 429, and 529. Only the nested {"error": {...}} shape is read as
// structured fields; any other body reaches the caller as a trimmed snippet in
// Message, and the whole body is always kept in Metadata.Raw.
func httpError(resp *http.Response, body []byte) error {
	httpErr := &llms.HTTPError{
		StatusCode: resp.StatusCode,
		Status:     resp.Status,
		Metadata:   llms.HTTPErrorMetadata{Raw: append(json.RawMessage(nil), body...)},
	}
	var envelope struct {
		Error struct {
			Type    string `json:"type"`
			Code    string `json:"code"`
			Message string `json:"message"`
		} `json:"error"`
	}
	if json.Unmarshal(body, &envelope) == nil && envelope.Error.Message != "" {
		httpErr.ErrorType = envelope.Error.Type
		httpErr.ErrorCode = envelope.Error.Code
		httpErr.Message = envelope.Error.Message
		return httpErr
	}
	httpErr.Message = bodySnippet(body)
	return httpErr
}

// maxErrorSnippet is how many bytes of an unrecognized error body are put in
// the error message. The rest stays in Metadata.Raw.
const maxErrorSnippet = 256

func bodySnippet(body []byte) string {
	if len(body) > maxErrorSnippet {
		body = body[:maxErrorSnippet]
	}
	return strings.TrimSpace(string(body))
}

// stateFromLLM builds the request state from the system prompt and messages.
// Structured content (content.JSON) is embedded as JSON rather than as an
// escaped string so the model sees named fields, which is how the API's own
// guidance says state should be shaped.
func stateFromLLM(systemPrompt content.Content, messages []llms.Message) (*jsonmap.Map, error) {
	state := jsonmap.New()
	if len(systemPrompt) > 0 {
		system, err := stateContent(systemPrompt)
		if err != nil {
			return nil, fmt.Errorf("system prompt: %w", err)
		}
		state.Set("system", system)
	}
	stateMessages := make([]stateMessage, 0, len(messages))
	for i, msg := range messages {
		if msg.Role == "tool" || len(msg.ToolCalls) > 0 {
			return nil, fmt.Errorf("message %d: %w", i, ErrToolsUnsupported)
		}
		body, err := stateContent(msg.Content)
		if err != nil {
			return nil, fmt.Errorf("message %d (role=%s): %w", i, msg.Role, err)
		}
		stateMessages = append(stateMessages, stateMessage{Role: msg.Role, Content: body})
	}
	if len(stateMessages) > 0 {
		state.Set("messages", stateMessages)
	}
	if state.Len() == 0 {
		return nil, errors.New("typesafe: nothing to evaluate; provide a system prompt or a message")
	}
	return state, nil
}

// stateContent converts message content to a state value: a string when every
// item is text, and an array of parts (strings and embedded JSON) when it is
// not. Thoughts and cache hints carry nothing for the model and are skipped;
// content that is only those is an error, because there would be nothing to
// evaluate.
func stateContent(c content.Content) (any, error) {
	var parts []any
	var text strings.Builder
	flushText := func() {
		if text.Len() > 0 {
			parts = append(parts, text.String())
			text.Reset()
		}
	}
	for _, item := range c {
		switch v := item.(type) {
		case *content.Text:
			text.WriteString(v.Text)
		case *content.JSON:
			flushText()
			parts = append(parts, json.RawMessage(v.Data))
		case *content.Thought, *content.CacheHint:
			continue
		default:
			return nil, fmt.Errorf("%w: got %s content", ErrNonTextContent, item.Type())
		}
	}
	if parts == nil {
		if str, ok := c.AsString(); ok {
			return str, nil
		}
		if text.Len() == 0 {
			return nil, errors.New("typesafe: content holds nothing the model can read")
		}
		return text.String(), nil
	}
	flushText()
	return parts, nil
}

// Stream delivers the rendered answer as a single text chunk. The API is not
// streaming; the whole response is known before the first status is yielded.
type Stream struct {
	err      error
	response *Response
	text     string
	message  llms.Message
}

// Response returns the full API response, including per-answer probabilities
// and confidence, or nil when the request failed.
func (s *Stream) Response() *Response {
	return s.response
}

func (s *Stream) Err() error {
	return s.err
}

func (s *Stream) Message() llms.Message {
	return s.message
}

func (s *Stream) Text() string {
	return s.text
}

func (s *Stream) Image() (string, string)  { return "", "" }
func (s *Stream) Audio() (string, string)  { return "", "" }
func (s *Stream) Thought() content.Thought { return content.Thought{} }
func (s *Stream) ToolCall() llms.ToolCall  { return llms.ToolCall{} }

func (s *Stream) Usage() llms.Usage {
	if s.response == nil {
		return llms.Usage{}
	}
	return llms.Usage{
		InputTokens:  s.response.Usage.InputTokens,
		OutputTokens: s.response.Usage.OutputTokens,
	}
}

func (s *Stream) Iter() func(yield func(llms.StreamStatus) bool) {
	return func(yield func(llms.StreamStatus) bool) {
		if s.err != nil {
			return
		}
		if !yield(llms.StreamStatusMessageStart) {
			return
		}
		yield(llms.StreamStatusText)
	}
}
