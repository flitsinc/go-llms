package llms

import (
	"context"
	"encoding/json"
	"testing"
	"time"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// mockJSONProvider returns a JSON response according to the schema
type mockJSONProvider struct {
	generateCalled   bool
	receivedSchema   *tools.ValueSchema
	receivedMessages []Message
}

func (m *mockJSONProvider) Company() string { return "MockJSON" }
func (m *mockJSONProvider) Model() string   { return "json-model" }
func (m *mockJSONProvider) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) ProviderStream {
	m.generateCalled = true
	m.receivedSchema = jsonOutputSchema
	m.receivedMessages = messages
	return &mockJSONStream{}
}

type mockJSONStream struct{}

func (s *mockJSONStream) Err() error { return nil }
func (s *mockJSONStream) Iter() func(func(StreamStatus) bool) {
	return func(yield func(StreamStatus) bool) {
		yield(StreamStatusText)
	}
}
func (s *mockJSONStream) Message() Message {
	return Message{
		Role:    "assistant",
		Content: content.FromRawJSON(json.RawMessage(`{"foo":"bar"}`)),
	}
}
func (s *mockJSONStream) Text() string             { return "{\"foo\":\"bar\"}" }
func (s *mockJSONStream) Thought() content.Thought { return content.Thought{} }
func (s *mockJSONStream) ToolCall() ToolCall       { return ToolCall{} }
func (s *mockJSONStream) Usage() Usage {
	return Usage{CachedInputTokens: 1, InputTokens: 1, OutputTokens: 1}
}

// Error provider and stream for error propagation test
type errorProvider struct{}

func (e *errorProvider) Company() string { return "err" }
func (e *errorProvider) Model() string   { return "err-model" }
func (e *errorProvider) Generate(ctx context.Context, systemPrompt content.Content, messages []Message, toolbox *tools.Toolbox, jsonOutputSchema *tools.ValueSchema) ProviderStream {
	return &mockJSONStreamWithError{}
}

type mockJSONStreamWithError struct{}

func (s *mockJSONStreamWithError) Err() error { return assert.AnError }
func (s *mockJSONStreamWithError) Iter() func(func(StreamStatus) bool) {
	return func(yield func(StreamStatus) bool) {}
}
func (s *mockJSONStreamWithError) Message() Message         { return Message{} }
func (s *mockJSONStreamWithError) Text() string             { return "" }
func (s *mockJSONStreamWithError) Thought() content.Thought { return content.Thought{} }
func (s *mockJSONStreamWithError) ToolCall() ToolCall       { return ToolCall{} }
func (s *mockJSONStreamWithError) Usage() Usage             { return Usage{} }

func TestLLM_JSONMode_PassesSchemaToProvider(t *testing.T) {
	provider := &mockJSONProvider{}
	llm := New(provider)
	schema := &tools.ValueSchema{
		Type: "object",
		Properties: &map[string]tools.ValueSchema{
			"foo": {Type: "string"},
		},
		Required: []string{"foo"},
	}
	llm.JSONOutputSchema = schema

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Give me foo as JSON")

	assert.True(t, provider.generateCalled, "Provider should be called")
	assert.Equal(t, schema, provider.receivedSchema, "Schema should be passed to provider")
	assert.NoError(t, llm.Err())
	// Should get a text update with JSON string
	require.Len(t, updates, 1)
	text, ok := updates[0].(TextUpdate)
	require.True(t, ok)
	assert.JSONEq(t, `{"foo":"bar"}`, text.Text)
}

func TestLLM_JSONMode_ConflictsWithTools(t *testing.T) {
	provider := &mockJSONProvider{}
	llm := New(provider, testTool)
	llm.JSONOutputSchema = &tools.ValueSchema{Type: "object"}

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Should fail due to conflict")

	assert.ErrorIs(t, llm.Err(), ErrToolsAndJSONOutputConflict)
	assert.Empty(t, updates)
}

func TestLLM_JSONMode_RespectsSchemaInCall(t *testing.T) {
	provider := &mockJSONProvider{}
	llm := New(provider)
	schema := &tools.ValueSchema{
		Type: "object",
		Properties: &map[string]tools.ValueSchema{
			"foo": {Type: "string"},
			"bar": {Type: "number"},
		},
		Required: []string{"foo", "bar"},
	}
	llm.JSONOutputSchema = schema

	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_ = runTestChat(ctx, t, llm, "Give me foo and bar as JSON")

	assert.True(t, provider.generateCalled)
	assert.Equal(t, schema, provider.receivedSchema)
}

func TestLLM_JSONMode_ErrorFromProviderIsPropagated(t *testing.T) {
	llm := New(&errorProvider{})
	llm.JSONOutputSchema = &tools.ValueSchema{Type: "object"}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	_ = runTestChat(ctx, t, llm, "error please")
	assert.Error(t, llm.Err())
}
