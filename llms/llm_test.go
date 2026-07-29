package llms

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"testing"
	"time"

	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/tools"
)

// --- Common Test Tool Definition ---

// Helper to extract JSON data from result content for testing
func extractJSONFromResult(t *testing.T, r tools.Result) json.RawMessage {
	t.Helper()
	require.NotNil(t, r.Content(), "Result content should not be nil")
	require.NotEmpty(t, r.Content(), "Result content should not be empty")
	jsonItem, ok := r.Content()[0].(*content.JSON)
	require.True(t, ok, "First content item should be JSON")
	return jsonItem.Data
}

type TestToolParams struct {
	TestParam string `json:"test_param"`
}

var testTool = tools.Func("Test Tool", "A test tool for testing", "test_tool",
	func(r tools.Runner, p TestToolParams) tools.Result {
		// Use SuccessWithLabel for consistency with other tests
		return tools.SuccessWithLabel("Test Tool Ran", map[string]any{
			"result": fmt.Sprintf("Processed: %s", p.TestParam),
		})
	})

// --- Mock Implementations ---

// mockProvider is a simple mock of the Provider interface for testing
type mockProvider struct {
	generateCalled         bool
	systemPrompt           content.Content
	messages               []Message
	toolbox                *tools.Toolbox
	jsonOutputSchema       *tools.ValueSchema
	toolboxToolsCount      int
	toolCallsToMake        []string // Names of tools to simulate calls for on the *first* Generate call
	processedToolResponses bool     // Tracks if we've seen tool responses in messages
	finalizationExpected   bool
	finalArguments         json.RawMessage
}

func (m *mockProvider) Company() string {
	return "Test Company"
}

func (m *mockProvider) Model() string {
	return "test-model"
}

func (m *mockProvider) SetHTTPClient(_ *http.Client) {}

func (m *mockProvider) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) ProviderStream {
	m.generateCalled = true
	m.systemPrompt = systemPrompt
	m.messages = messages
	m.toolbox = toolbox
	m.jsonOutputSchema = jsonOutputSchema
	if toolbox != nil {
		m.toolboxToolsCount = len(toolbox.All())
	}

	// Check if we've processed tool responses in a previous turn
	m.processedToolResponses = false
	for _, msg := range messages {
		if msg.Role == "tool" {
			m.processedToolResponses = true
			break
		}
	}

	// Determine which tool calls and text to use based on state
	toolCallsToUse := []string{}
	textToGenerate := "This is a test message." // Default for initial call

	if m.processedToolResponses {
		// Generate a response acknowledging the tool results
		textToGenerate = "I've processed the results from the tool."
	} else {
		// This is the initial call, use the predefined tool calls
		toolCallsToUse = m.toolCallsToMake
	}

	return &mockStream{
		provider:             m,
		textToGenerate:       textToGenerate,
		toolCalls:            toolCallsToUse,
		finalizationExpected: m.finalizationExpected && !m.processedToolResponses,
		finalArguments:       m.finalArguments,
	}
}

// mockStream is a simple implementation of ProviderStream for testing
type mockStream struct {
	provider             *mockProvider
	textToGenerate       string
	toolCalls            []string
	message              Message
	finalizationExpected bool
	finalArguments       json.RawMessage
}

func (s *mockStream) Err() error { return nil }

func (s *mockStream) Iter() func(func(StreamStatus) bool) {
	return func(yield func(StreamStatus) bool) {
		// First yield text
		if !yield(StreamStatusText) {
			return
		}

		// Then yield tool calls if any
		for i, toolName := range s.toolCalls {
			uniqueID := fmt.Sprintf("%s-id-%d", toolName, i)
			fullArgsStr := fmt.Sprintf(`{"test_param":"test_value_%s"}`, toolName)
			s.message.ToolCalls = append(s.message.ToolCalls, ToolCall{
				ID:        uniqueID,
				Name:      toolName,
				Arguments: json.RawMessage{}, // Will be set later.
			})

			if !yield(StreamStatusToolCallBegin) {
				return
			}

			// First delta: half of the arguments.
			// The LLM will call stream.ToolCall() which should return the tool call with these partial arguments.
			s.message.ToolCalls[i].Arguments = json.RawMessage(fullArgsStr[:len(fullArgsStr)/2])
			if !yield(StreamStatusToolCallDelta) {
				return
			}

			// Second delta: full arguments.
			// The LLM will call stream.ToolCall() again, which should return the tool call with full arguments.
			s.message.ToolCalls[i].Arguments = json.RawMessage(fullArgsStr)
			if !yield(StreamStatusToolCallDelta) {
				return
			}

			if !yield(StreamStatusToolCallReady) {
				return
			}
		}
	}
}

func (s *mockStream) Message() Message {
	if s.message.Content == nil {
		s.message = Message{
			Role:      "assistant",
			Content:   content.FromText(s.textToGenerate),
			ToolCalls: s.message.ToolCalls,
		}
	}
	return s.message
}

func (s *mockStream) Text() string { return s.textToGenerate }

func (s *mockStream) Audio() (string, string) { return "", "" }
func (s *mockStream) Image() (string, string) { return "", "" }

func (s *mockStream) Thought() content.Thought { return content.Thought{} }

func (s *mockStream) ToolCall() ToolCall {
	if len(s.message.ToolCalls) > 0 {
		return s.message.ToolCalls[len(s.message.ToolCalls)-1]
	}
	return ToolCall{}
}

func (s *mockStream) ToolArgumentFinalization() (json.RawMessage, bool) {
	return s.finalArguments, s.finalizationExpected
}

func (s *mockStream) Usage() Usage {
	return Usage{CachedInputTokens: 10, InputTokens: 20, OutputTokens: 30}
}

// Mock provider that always returns an error stream
type errorMockProvider struct {
	errorMessage string
}

func (m *errorMockProvider) Company() string {
	return "Error Test Company"
}

func (m *errorMockProvider) Model() string {
	return "test-model"
}

func (m *errorMockProvider) SetHTTPClient(_ *http.Client) {}

// Updated signature
func (m *errorMockProvider) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) ProviderStream {
	return &errorMockStream{
		err: fmt.Errorf("provider stream error: %s", m.errorMessage),
	}
}

type errorMockStream struct {
	err error
}

func (s *errorMockStream) Err() error { return s.err }
func (s *errorMockStream) Iter() func(func(StreamStatus) bool) {
	return func(func(StreamStatus) bool) {} // No iteration
}
func (s *errorMockStream) Message() Message         { return Message{} }
func (s *errorMockStream) Text() string             { return "" }
func (s *errorMockStream) Audio() (string, string)  { return "", "" }
func (s *errorMockStream) Image() (string, string)  { return "", "" }
func (s *errorMockStream) Thought() content.Thought { return content.Thought{} }
func (s *errorMockStream) ToolCall() ToolCall       { return ToolCall{} }
func (s *errorMockStream) Usage() Usage             { return Usage{} }

type panicMockProvider struct{}

func (m *panicMockProvider) Company() string {
	return "Panic Test Company"
}

func (m *panicMockProvider) Model() string {
	return "panic-model"
}

func (m *panicMockProvider) SetHTTPClient(_ *http.Client) {}

func (m *panicMockProvider) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) ProviderStream {
	panic("provider exploded")
}

// mockEmptyIDProvider is a provider that returns tool calls with empty IDs
type mockEmptyIDProvider struct{}

func (m *mockEmptyIDProvider) Company() string {
	return "Test Company Empty ID"
}

func (m *mockEmptyIDProvider) Model() string {
	return "test-model"
}

func (m *mockEmptyIDProvider) SetHTTPClient(_ *http.Client) {}

// Updated signature
func (m *mockEmptyIDProvider) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) ProviderStream {
	return &mockEmptyIDStream{}
}

// mockEmptyIDStream is a stream that returns tool calls with empty IDs
type mockEmptyIDStream struct {
	message Message
}

func (s *mockEmptyIDStream) Err() error { return nil }

func (s *mockEmptyIDStream) Iter() func(func(StreamStatus) bool) {
	return func(yield func(StreamStatus) bool) {
		// First yield text
		if !yield(StreamStatusText) {
			return
		}

		// Then yield a tool call with an empty ID
		s.message.ToolCalls = append(s.message.ToolCalls, ToolCall{
			ID:        "", // Empty ID should cause an error
			Name:      "test_tool",
			Arguments: json.RawMessage(`{"test_param":"test_value"}`),
		})

		// The stream should stop after this yields an error in the LLM
		// since the implementation breaks the loop when an error is detected
		if !yield(StreamStatusToolCallBegin) { // Yield begin to trigger the check
			return
		}
		// No need to yield StreamStatusToolCallReady
	}
}

func (s *mockEmptyIDStream) Message() Message {
	if s.message.Content == nil {
		s.message = Message{
			Role:      "assistant",
			Content:   content.FromText("This is a test message."),
			ToolCalls: s.message.ToolCalls,
		}
	}
	return s.message
}

func (s *mockEmptyIDStream) Text() string { return "This is a test message." }

func (s *mockEmptyIDStream) Audio() (string, string) { return "", "" }
func (s *mockEmptyIDStream) Image() (string, string) { return "", "" }

func (s *mockEmptyIDStream) Thought() content.Thought { return content.Thought{} }

func (s *mockEmptyIDStream) ToolCall() ToolCall {
	if len(s.message.ToolCalls) > 0 {
		return s.message.ToolCalls[len(s.message.ToolCalls)-1]
	}
	return ToolCall{}
}

func (s *mockEmptyIDStream) Usage() Usage {
	return Usage{CachedInputTokens: 10, InputTokens: 20, OutputTokens: 30}
}

// mockCancellingProvider creates a stream that will block until context is cancelled (fixed implementation)
type mockCancellingProvider struct{}

func (m *mockCancellingProvider) Company() string { return "Mock Cancelling Provider" }

func (m *mockCancellingProvider) Model() string {
	return "test-model"
}

func (m *mockCancellingProvider) SetHTTPClient(_ *http.Client) {}

// Updated signature
func (m *mockCancellingProvider) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) ProviderStream {
	return &mockCancellingStream{ctx: ctx} // Pass context to the stream
}

// mockCancellingStream is a stream that blocks until context is cancelled (fixed implementation)
type mockCancellingStream struct {
	message Message
	ctx     context.Context // Add context field
}

func (s *mockCancellingStream) Err() error { return nil }

func (s *mockCancellingStream) Iter() func(func(StreamStatus) bool) {
	return func(yield func(StreamStatus) bool) {
		// First yield text
		if !yield(StreamStatusText) {
			return
		}

		// Then block until context is cancelled, respecting the stream's context
		select {
		case <-s.ctx.Done():
			// Context cancelled, stop iterating.
			// The caller (turn) will detect the context error.
			return
		case <-time.After(10 * time.Minute): // Long timeout to ensure context is the trigger
			// This case should not be hit in the test
			panic("mockCancellingStream timed out unexpectedly")
		}
	}
}

func (s *mockCancellingStream) Message() Message {
	if s.message.Content == nil {
		s.message = Message{
			Role:    "assistant",
			Content: content.FromText("This is a test message."),
		}
	}
	return s.message
}

func (s *mockCancellingStream) Text() string             { return "This is a test message." }
func (s *mockCancellingStream) Audio() (string, string)  { return "", "" }
func (s *mockCancellingStream) Image() (string, string)  { return "", "" }
func (s *mockCancellingStream) Thought() content.Thought { return content.Thought{} }
func (s *mockCancellingStream) ToolCall() ToolCall       { return ToolCall{} }
func (s *mockCancellingStream) Usage() Usage             { return Usage{} }

// mockToolWithError always returns an error.
var mockToolWithError = tools.Func("Error Tool", "A tool that always errors", "error_tool",
	func(r tools.Runner, p TestToolParams) tools.Result {
		return tools.Errorf("internal tool error detail")
	})

// --- Helper Functions ---

// setupTestLLM creates a new LLM instance with a mock provider and specified tools.
// It returns the LLM instance and the mock provider for further interaction/verification.
func setupTestLLM(t *testing.T, provider Provider, tools ...tools.Tool) (*LLM, *mockProvider) {
	t.Helper()
	llm := New(provider, tools...)
	// Try to cast provider to *mockProvider if possible for verification
	mockProv, _ := provider.(*mockProvider)
	return llm, mockProv
}

// runTestChat executes llm.ChatWithContext and collects all updates into a slice.
// It handles the context timeout and returns the updates and any error from the LLM.
func runTestChat(ctx context.Context, t *testing.T, llm *LLM, message string) []Update {
	t.Helper()
	var updates []Update

	chatChan := llm.ChatWithContext(ctx, message)

	// Use a select to wait for the chat to finish or the context to be done
	for {
		select {
		case update, ok := <-chatChan:
			if !ok { // Channel closed, chat finished
				return updates
			}
			updates = append(updates, update)
		case <-ctx.Done():
			// Context was cancelled or timed out. This might be expected by the test.
			// Simply return the updates collected so far. The calling test
			// should check llm.Err() if cancellation was unexpected.
			return updates
		}
	}
}

// --- Mocks for Tool Not Found Test ---

// mockAlwaysUnknownToolProvider calls a tool that isn't in the toolbox on every
// turn, i.e. a model that never learns from the "tool not found" results it
// gets back.
type mockAlwaysUnknownToolProvider struct {
	generateCount int
	// lastMessages is the history handed to the most recent Generate call.
	lastMessages []Message
	// realToolEveryOtherTurn makes every second turn call a tool that does
	// exist, so the model looks like it's making progress in between.
	realToolEveryOtherTurn bool
	// maxTurns, when non-zero, is the turn on which the provider stops calling
	// tools so the chat can end on its own.
	maxTurns int
	// truncate ends the stream after the tool call has begun, without ever
	// reaching StreamStatusToolCallReady.
	truncate bool
	// malformedArgs delivers arguments that don't parse, but still finishes the
	// call properly.
	malformedArgs bool
	// danglingKnownCall begins a call to a tool that does exist after the
	// unknown one has finished, and then ends the stream without finishing it.
	danglingKnownCall bool
}

func (m *mockAlwaysUnknownToolProvider) Company() string { return "Test Company" }

func (m *mockAlwaysUnknownToolProvider) Model() string { return "test-model" }

func (m *mockAlwaysUnknownToolProvider) SetHTTPClient(_ *http.Client) {}

func (m *mockAlwaysUnknownToolProvider) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) ProviderStream {
	m.generateCount++
	m.lastMessages = messages
	s := &mockStreamUnknownTool{
		turn:              m.generateCount,
		truncate:          m.truncate,
		malformedArgs:     m.malformedArgs,
		danglingKnownCall: m.danglingKnownCall,
	}
	if m.maxTurns > 0 && m.generateCount >= m.maxTurns {
		s.noToolCall = true
	} else if m.realToolEveryOtherTurn && m.generateCount%2 == 0 {
		s.toolName = "test_tool"
	}
	return s
}

// mockStreamUnknownTool yields text plus a single tool call, by default to a
// tool that doesn't exist.
type mockStreamUnknownTool struct {
	turn              int
	toolName          string
	noToolCall        bool
	truncate          bool
	malformedArgs     bool
	danglingKnownCall bool
	message           Message
}

func (s *mockStreamUnknownTool) Err() error { return nil }

func (s *mockStreamUnknownTool) Iter() func(func(StreamStatus) bool) {
	return func(yield func(StreamStatus) bool) {
		if !yield(StreamStatusText) {
			return
		}
		if s.noToolCall {
			return
		}
		name := s.toolName
		if name == "" {
			name = fmt.Sprintf("tool_does_not_exist_%d", s.turn)
		}
		s.message.ToolCalls = append(s.message.ToolCalls, ToolCall{
			ID:   fmt.Sprintf("unknown-id-%d", s.turn),
			Name: name,
		})
		if !yield(StreamStatusToolCallBegin) {
			return
		}
		args := `{"test_param":"value"}`
		if s.truncate || s.malformedArgs {
			// Cut off mid-value, so the delivered arguments don't parse.
			args = `{"test_param":`
		}
		s.message.ToolCalls[0].Arguments = json.RawMessage(args)
		if !yield(StreamStatusToolCallDelta) {
			return
		}
		if s.truncate {
			// The stream ends mid-call, without error, as a provider cutting the
			// connection short would.
			return
		}
		if !yield(StreamStatusToolCallReady) {
			return
		}
		if s.danglingKnownCall {
			// A second call, to a tool that does exist, which the stream never
			// finishes.
			s.message.ToolCalls = append(s.message.ToolCalls, ToolCall{
				ID:   fmt.Sprintf("known-id-%d", s.turn),
				Name: "test_tool",
			})
			yield(StreamStatusToolCallBegin)
		}
	}
}

func (s *mockStreamUnknownTool) Message() Message {
	if s.message.Content == nil {
		s.message = Message{
			Role:      "assistant",
			Content:   content.FromText(s.Text()),
			ToolCalls: s.message.ToolCalls,
		}
	}
	return s.message
}

func (s *mockStreamUnknownTool) Text() string { return "Trying a tool..." }

func (s *mockStreamUnknownTool) Audio() (string, string) { return "", "" }
func (s *mockStreamUnknownTool) Image() (string, string) { return "", "" }

func (s *mockStreamUnknownTool) Thought() content.Thought { return content.Thought{} }

func (s *mockStreamUnknownTool) ToolCall() ToolCall {
	if len(s.message.ToolCalls) > 0 {
		return s.message.ToolCalls[len(s.message.ToolCalls)-1]
	}
	return ToolCall{}
}

func (s *mockStreamUnknownTool) Usage() Usage { return Usage{} }

// mockToolForStatusTest is a simple tool used for testing status updates path.
var mockToolForStatusTest = tools.Func("Status Tool", "A tool used for status test", "status_tool",
	func(r tools.Runner, p TestToolParams) tools.Result {
		time.Sleep(10 * time.Millisecond) // Simulate work
		return tools.Success(map[string]any{"status": "done"})
	})

// --- Test for ToolCall in Context ---

// toolThatChecksContext retrieves the ToolCall from the context and returns its ID.
var toolThatChecksContext = tools.Func("Context Checker Tool", "Checks for ToolCall in context", "context_checker_tool",
	func(r tools.Runner, p TestToolParams) tools.Result {
		tc, ok := GetToolCall(r.Context())
		if !ok {
			return tools.Errorf("ToolCall not found in context or wrong type")
		}
		return tools.Success(map[string]any{"tool_call_id": tc.ID})
	})
