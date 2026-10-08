// Package typesafe is a go-llms provider for TypeSafe's System One models
// (https://docs.typesafe.ai). A System One model does not generate text: it
// reads a state and answers typed questions with calibrated probabilities.
// The provider maps the go-llms contract onto that shape.
//
//   - The system prompt and messages become the request state, as a JSON
//     object with a "system" string and a "messages" array. Images are lifted
//     out of the content and sent before that object, as the image parts a
//     multimodal decision model reads; see [state].
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

// ErrUnsupportedContent is returned when the system prompt or a message
// carries a content item the state cannot hold: audio, video, a compaction
// checkpoint, or any item type this provider does not know. The state holds
// text, structured JSON, and images.
var ErrUnsupportedContent = errors.New("typesafe: System One models accept text, structured JSON, and images only")

// ErrNothingToEvaluate is returned when the system prompt and messages hold
// nothing the model can read: no text, no JSON, and no image.
var ErrNothingToEvaluate = errors.New("typesafe: nothing to evaluate; provide a system prompt or a message")

// ErrEmptyImageURL is returned when an image item has no URL.
var ErrEmptyImageURL = errors.New("typesafe: image item has no URL")

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

	requestState, err := stateFromLLM(systemPrompt, messages)
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

	jsonData, err := json.Marshal(request{State: requestState, Model: m.model, Questions: questions})
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
//
// Text and JSON keep their place in the conversation. Images do not: the API
// reads an image part only at the top level of the state, not inside a
// message, so every image is lifted out of the content it came with and sent
// before the conversation; see [state]. A system prompt or message that was
// only images contributes nothing to the conversation.
func stateFromLLM(systemPrompt content.Content, messages []llms.Message) (*state, error) {
	s := &state{conversation: jsonmap.New()}
	if len(systemPrompt) > 0 {
		system, err := s.body(systemPrompt)
		if err != nil {
			return nil, fmt.Errorf("system prompt: %w", err)
		}
		if system != nil {
			s.conversation.Set("system", system)
		}
	}
	stateMessages := make([]stateMessage, 0, len(messages))
	for i, msg := range messages {
		if msg.Role == "tool" || len(msg.ToolCalls) > 0 {
			return nil, fmt.Errorf("message %d: %w", i, ErrToolsUnsupported)
		}
		body, err := s.body(msg.Content)
		if err != nil {
			return nil, fmt.Errorf("message %d (role=%s): %w", i, msg.Role, err)
		}
		if body != nil {
			stateMessages = append(stateMessages, stateMessage{Role: msg.Role, Content: body})
		}
	}
	if len(stateMessages) > 0 {
		s.conversation.Set("messages", stateMessages)
	}
	if s.conversation.Len() == 0 && len(s.images) == 0 {
		return nil, ErrNothingToEvaluate
	}
	return s, nil
}

// state is the request state: the conversation object with its "system"
// string and "messages" array, and the images lifted out of them. It is sent
// as the object alone when there are no images, and otherwise as an array of
// the image parts followed by the object (or of the image parts alone when
// the content was only images), because the API reads an image part only at
// the top level of the state.
type state struct {
	images       []imagePart
	conversation *jsonmap.Map
}

func (s *state) MarshalJSON() ([]byte, error) {
	if len(s.images) == 0 {
		return json.Marshal(s.conversation)
	}
	parts := make([]any, 0, len(s.images)+1)
	for _, image := range s.images {
		parts = append(parts, image)
	}
	if s.conversation.Len() > 0 {
		parts = append(parts, s.conversation)
	}
	return json.Marshal(parts)
}

// body converts content to its conversation value, a string when every item
// is text and an array of parts (strings and embedded JSON) when it is not,
// and collects the content's images into the state. Thoughts and cache hints
// carry nothing for the model and are skipped. Content that was only images
// yields a nil value, since the images are what it carried; content with
// nothing the model can read at all is [ErrNothingToEvaluate].
func (s *state) body(c content.Content) (any, error) {
	var parts []any
	var text strings.Builder
	flushText := func() {
		if text.Len() > 0 {
			parts = append(parts, text.String())
			text.Reset()
		}
	}
	images := 0
	for _, item := range c {
		switch v := item.(type) {
		case *content.Text:
			text.WriteString(v.Text)
		case *content.JSON:
			flushText()
			parts = append(parts, json.RawMessage(v.Data))
		case *content.ImageURL:
			if v.URL == "" {
				return nil, ErrEmptyImageURL
			}
			s.images = append(s.images, imagePart{Type: "image_url", ImageURL: imageURLPart{URL: v.URL}})
			images++
		case *content.Thought, *content.CacheHint:
			continue
		default:
			return nil, fmt.Errorf("%w: got %s content", ErrUnsupportedContent, item.Type())
		}
	}
	if parts == nil {
		if text.Len() == 0 {
			if images > 0 {
				return nil, nil
			}
			return nil, fmt.Errorf("%w: content holds nothing the model can read", ErrNothingToEvaluate)
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
