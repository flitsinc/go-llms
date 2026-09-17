package typesafe

import (
	"context"
	"encoding/json"
	"errors"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/metalim/jsonmap"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

func objectSchema(props ...struct {
	name   string
	schema tools.ValueSchema
}) *tools.ValueSchema {
	m := jsonmap.New()
	for _, p := range props {
		m.Set(p.name, p.schema)
	}
	return &tools.ValueSchema{Type: "object", Properties: m}
}

func prop(name string, schema tools.ValueSchema) struct {
	name   string
	schema tools.ValueSchema
} {
	return struct {
		name   string
		schema tools.ValueSchema
	}{name, schema}
}

// reviewSchema is the shape a lint-style caller sends: two Nouls rendered as
// booleans, one rendered as a probability, and one Choice.
func reviewSchema() *tools.ValueSchema {
	return objectSchema(
		prop("exposes_pii", tools.ValueSchema{Type: "boolean", Description: "Does the new code expose personal data under a broad read policy?"}),
		prop("widens_access", tools.ValueSchema{Type: "boolean", Description: "Does the new policy grant broader access than the old one?"}),
		prop("confidence", tools.ValueSchema{Type: "number", Description: "Is the reviewed code clearly unsafe?"}),
		prop("severity", tools.ValueSchema{Type: "string", Enum: []any{"error", "warning", "info"}, Description: "How serious is the worst finding?"}),
	)
}

func collect(t *testing.T, stream llms.ProviderStream) []llms.StreamStatus {
	t.Helper()
	var statuses []llms.StreamStatus
	for status := range stream.Iter() {
		statuses = append(statuses, status)
	}
	return statuses
}

func TestGenerateRoundTrip(t *testing.T) {
	var captured request
	var authorization string
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		authorization = r.Header.Get("Authorization")
		body, err := io.ReadAll(r.Body)
		require.NoError(t, err)
		require.NoError(t, json.Unmarshal(body, &captured))
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(`{
			"model": "jev-1.13.0",
			"answers": {
				"exposes_pii": {"type": "noul", "noul": 0.93},
				"widens_access": {"type": "noul", "noul": 0.12},
				"confidence": {"type": "noul", "noul": 0.81},
				"severity": {"type": "choice", "choice": "error", "probabilities": {"error": 0.7, "warning": 0.2, "info": 0.1}, "confidence": 0.66}
			},
			"usage": {"input_tokens": 312, "output_tokens": 48}
		}`))
	}))
	defer server.Close()

	model := New("secret", "jev-1.13.0").WithEndpoint(server.URL, "Test")
	stream := model.Generate(
		context.Background(),
		content.FromText("You review data model changes."),
		[]llms.Message{
			{Role: "user", Content: content.FromText("const User = MVC.createModel()")},
		},
		nil,
		reviewSchema(),
	)
	require.NoError(t, stream.Err())

	assert.Equal(t, "Bearer secret", authorization)
	assert.Equal(t, "jev-1.13.0", captured.Model)

	// The state names its parts instead of flattening everything to a string.
	stateJSON, err := json.Marshal(captured.State)
	require.NoError(t, err)
	assert.JSONEq(t, `{
		"system": "You review data model changes.",
		"messages": [{"role": "user", "content": "const User = MVC.createModel()"}]
	}`, string(stateJSON))

	// Every property became a question with the description as instructions,
	// and the enum became the choice criteria.
	require.Len(t, captured.Questions, 4)
	assert.Equal(t, Question{Type: "noul", Instructions: "Does the new code expose personal data under a broad read policy?"}, captured.Questions["exposes_pii"])
	assert.Equal(t, "choice", captured.Questions["severity"].Type)
	assert.Equal(t, map[string]any{"error": nil, "warning": nil, "info": nil}, captured.Questions["severity"].Criteria)

	// The answer is delivered as one message-start plus one text chunk holding
	// the object the schema described, in schema order, with each property
	// rendered in its declared type.
	assert.Equal(t, []llms.StreamStatus{llms.StreamStatusMessageStart, llms.StreamStatusText}, collect(t, stream))
	assert.Equal(t, `{"exposes_pii":true,"widens_access":false,"confidence":0.81,"severity":"error"}`, stream.Text())
	assert.Equal(t, "assistant", stream.Message().Role)
	assert.Equal(t, llms.Usage{InputTokens: 312, OutputTokens: 48}, stream.Usage())

	// The distributions behind the rendered answer stay reachable.
	response := stream.(*Stream).Response()
	require.NotNil(t, response)
	assert.InDelta(t, 0.66, *response.Answers["severity"].Confidence, 1e-9)
	assert.InDelta(t, 0.7, response.Answers["severity"].Probabilities["error"], 1e-9)
}

func TestStateEmbedsStructuredContent(t *testing.T) {
	state, err := stateFromLLM(nil, []llms.Message{
		{Role: "user", Content: content.FromRawJSON(json.RawMessage(`{"model":{"fields":[{"name":"email","type":"Email"}]}}`))},
	})
	require.NoError(t, err)
	stateJSON, err := json.Marshal(state)
	require.NoError(t, err)
	// JSON content is embedded as JSON, not as an escaped string, so the model
	// sees named fields.
	assert.JSONEq(t, `{"messages": [{"role": "user", "content": {"model": {"fields": [{"name": "email", "type": "Email"}]}}}]}`, string(stateJSON))
}

func TestStateSkipsThoughtsAndRejectsMedia(t *testing.T) {
	var withThought content.Content
	withThought.Append("hello")
	withThought.AppendThought("private reasoning")
	state, err := stateFromLLM(withThought, nil)
	require.NoError(t, err)
	system, _ := state.Get("system")
	assert.Equal(t, "hello", system)

	_, err = stateFromLLM(content.FromTextAndImage("look", "https://example.com/a.png"), nil)
	assert.ErrorIs(t, err, ErrNonTextContent)

	_, err = stateFromLLM(nil, nil)
	assert.Error(t, err)
}

func TestToolsAreRejected(t *testing.T) {
	model := New("secret", "jev-latest")

	_, err := stateFromLLM(nil, []llms.Message{{Role: "tool", ToolCallID: "call_1", Content: content.FromText("{}")}})
	assert.ErrorIs(t, err, ErrToolsUnsupported)

	_, err = stateFromLLM(nil, []llms.Message{{Role: "assistant", ToolCalls: []llms.ToolCall{{ID: "call_1", Name: "lookup"}}}})
	assert.ErrorIs(t, err, ErrToolsUnsupported)

	// An empty toolbox is what callers without tools pass; it is fine, and
	// the request then fails on the missing schema instead.
	stream := model.Generate(context.Background(), nil, []llms.Message{{Role: "user", Content: content.FromText("x")}}, &tools.Toolbox{}, nil)
	assert.ErrorIs(t, stream.Err(), ErrUnsupportedSchema)

	var toolbox tools.Toolbox
	toolbox.Add(tools.Func("Lookup", "Looks something up", "lookup", func(_ tools.Runner, _ struct{}) tools.Result {
		return tools.Success(nil)
	}))
	stream = model.Generate(context.Background(), nil, []llms.Message{{Role: "user", Content: content.FromText("x")}}, &toolbox, reviewSchema())
	assert.ErrorIs(t, stream.Err(), ErrToolsUnsupported)
	assert.Empty(t, collect(t, stream))
}

func TestQuestionsFromSchemaRejectsWhatTheModelCannotAnswer(t *testing.T) {
	cases := []struct {
		name   string
		schema *tools.ValueSchema
		want   string
	}{
		{"nil schema", nil, "a JSON output schema is required"},
		{"root not object", &tools.ValueSchema{Type: "string"}, "the root must be an object"},
		{"missing description", objectSchema(prop("ok", tools.ValueSchema{Type: "boolean"})), `property "ok": a description is required`},
		{"free string", objectSchema(prop("summary", tools.ValueSchema{Type: "string", Description: "Summarize the change."})), `property "summary": a string property needs an enum`},
		{"nested object", objectSchema(prop("finding", tools.ValueSchema{Type: "object", Description: "The finding."})), `property "finding": type "object"`},
		{"array", objectSchema(prop("findings", tools.ValueSchema{Type: "array", Description: "All findings."})), `property "findings": type "array"`},
		{"integer", objectSchema(prop("count", tools.ValueSchema{Type: "integer", Description: "How many?"})), `property "count": type "integer"`},
		{"non-string enum", objectSchema(prop("level", tools.ValueSchema{Type: "string", Enum: []any{1, 2}, Description: "Which level?"})), "is not a string"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			_, _, err := questionsFromSchema(tc.schema)
			require.ErrorIs(t, err, ErrUnsupportedSchema)
			assert.Contains(t, err.Error(), tc.want)
		})
	}
}

func TestQuestionsFromWireSchema(t *testing.T) {
	// A schema that arrived over HTTP has decoded property values rather than
	// ValueSchema structs; the mapping must read those too.
	var schema tools.ValueSchema
	require.NoError(t, json.Unmarshal([]byte(`{
		"type": "object",
		"properties": {
			"is_urgent": {"type": "boolean", "description": "Is this urgent?"},
			"team": {"type": "string", "enum": ["billing", "technical"], "description": "Which team should handle this?"}
		},
		"required": ["is_urgent", "team"]
	}`), &schema))

	bound, questions, err := questionsFromSchema(&schema)
	require.NoError(t, err)
	assert.Equal(t, 2, bound.Len())
	assert.Equal(t, "noul", questions["is_urgent"].Type)
	assert.Equal(t, "choice", questions["team"].Type)
}

func TestRenderAnswersRejectsMissingOrMistypedAnswers(t *testing.T) {
	bound, _, err := questionsFromSchema(reviewSchema())
	require.NoError(t, err)

	yes := 0.9
	_, err = renderAnswers(bound, map[string]Answer{"exposes_pii": {Type: "noul", Noul: &yes}})
	assert.ErrorContains(t, err, `no answer for "widens_access"`)

	answers := map[string]Answer{
		"exposes_pii":   {Type: "choice", Choice: "yes"},
		"widens_access": {Type: "noul", Noul: &yes},
		"confidence":    {Type: "noul", Noul: &yes},
		"severity":      {Type: "choice", Choice: "error"},
	}
	_, err = renderAnswers(bound, answers)
	assert.ErrorContains(t, err, `answer for "exposes_pii" is not a noul`)
}

func TestHTTPErrorsBecomeHTTPError(t *testing.T) {
	cases := []struct {
		name    string
		status  int
		body    string
		message string
		errType string
	}{
		{"string detail", http.StatusUnprocessableEntity, `{"detail": "questions.x.criteria: field required"}`, "questions.x.criteria: field required", ""},
		{"nested error", http.StatusTooManyRequests, `{"error": {"type": "rate_limit_error", "message": "slow down"}}`, "slow down", "rate_limit_error"},
		{"string error", http.StatusUnauthorized, `{"error": "invalid api key"}`, "invalid api key", ""},
		{"unparseable", 529, `overloaded`, "", ""},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
				w.WriteHeader(tc.status)
				_, _ = w.Write([]byte(tc.body))
			}))
			defer server.Close()

			model := New("secret", "jev-latest").WithEndpoint(server.URL, "Test")
			stream := model.Generate(context.Background(), nil, []llms.Message{{Role: "user", Content: content.FromText("x")}}, nil, reviewSchema())
			var httpErr *llms.HTTPError
			require.True(t, errors.As(stream.Err(), &httpErr), "got %v", stream.Err())
			assert.Equal(t, tc.status, httpErr.StatusCode)
			assert.Equal(t, tc.message, httpErr.Message)
			assert.Equal(t, tc.errType, httpErr.ErrorType)
			assert.Empty(t, collect(t, stream), "a failed stream yields nothing")
		})
	}
}

func TestCancelledContextStopsIteration(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		_, _ = w.Write([]byte(`{"model":"jev-1.13.0","answers":{"ok":{"type":"noul","noul":1}},"usage":{"input_tokens":1,"output_tokens":1}}`))
	}))
	defer server.Close()

	ctx, cancel := context.WithCancel(context.Background())
	model := New("secret", "jev-latest").WithEndpoint(server.URL, "Test")
	schema := objectSchema(prop("ok", tools.ValueSchema{Type: "boolean", Description: "Is it ok?"}))
	stream := model.Generate(ctx, nil, []llms.Message{{Role: "user", Content: content.FromText("x")}}, nil, schema)
	require.NoError(t, stream.Err())

	cancel()
	assert.Empty(t, collect(t, stream))
	assert.ErrorIs(t, stream.Err(), context.Canceled)
}
