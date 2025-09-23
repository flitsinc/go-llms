package llms

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"sync"
	"testing"
	"time"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/tools"
	"github.com/stretchr/testify/require"
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
	debugger               Debugger
}

func (m *mockProvider) Company() string {
	return "Test Company"
}

func (m *mockProvider) Model() string {
	return "test-model"
}

func (m *mockProvider) SetDebugger(d Debugger) {
	m.debugger = d
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
		provider:       m,
		textToGenerate: textToGenerate,
		toolCalls:      toolCallsToUse,
	}
}

// mockStream is a simple implementation of ProviderStream for testing
type mockStream struct {
	provider       *mockProvider
	textToGenerate string
	toolCalls      []string
	message        Message
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

func (s *mockStream) Image() (string, string) { return "", "" }

func (s *mockStream) Thought() content.Thought { return content.Thought{} }

func (s *mockStream) ToolCall() ToolCall {
	if len(s.message.ToolCalls) > 0 {
		return s.message.ToolCalls[len(s.message.ToolCalls)-1]
	}
	return ToolCall{}
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

func (m *errorMockProvider) SetDebugger(d Debugger) {}

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
func (s *errorMockStream) Image() (string, string)  { return "", "" }
func (s *errorMockStream) Thought() content.Thought { return content.Thought{} }
func (s *errorMockStream) ToolCall() ToolCall       { return ToolCall{} }
func (s *errorMockStream) Usage() Usage             { return Usage{} }

// mockEmptyIDProvider is a provider that returns tool calls with empty IDs
type mockEmptyIDProvider struct{}

func (m *mockEmptyIDProvider) Company() string {
	return "Test Company Empty ID"
}

func (m *mockEmptyIDProvider) Model() string {
	return "test-model"
}

func (m *mockEmptyIDProvider) SetDebugger(d Debugger) {}

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

func (m *mockCancellingProvider) SetDebugger(d Debugger) {}

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
	var updatesMutex sync.Mutex

	chatChan := llm.ChatWithContext(ctx, message)

	// Use a select to wait for the chat to finish or the context to be done
	for {
		select {
		case update, ok := <-chatChan:
			if !ok { // Channel closed, chat finished
				return updates
			}
			updatesMutex.Lock()
			updates = append(updates, update)
			updatesMutex.Unlock()
		case <-ctx.Done():
			// Context was cancelled or timed out. This might be expected by the test.
			// Simply return the updates collected so far. The calling test
			// should check llm.Err() if cancellation was unexpected.
			return updates
		}
	}
}

// --- Mocks for Tool Not Found Test ---

// mockStreamToolNotFound yields a call to a non-existent tool.
type mockStreamToolNotFound struct {
	message Message // Store message locally
}

func (s *mockStreamToolNotFound) Err() error { return nil }

func (s *mockStreamToolNotFound) Iter() func(func(StreamStatus) bool) {
	return func(yield func(StreamStatus) bool) {
		// Yield text first
		if !yield(StreamStatusText) {
			return
		}
		// Prepare the non-existent tool call in the message
		s.message.ToolCalls = append(s.message.ToolCalls, ToolCall{
			ID:        "not-found-id-1",
			Name:      "tool_does_not_exist", // This tool is not added to the LLM
			Arguments: json.RawMessage(`{}`),
		})
		// Yield begin, which should trigger the error in the turn method
		yield(StreamStatusToolCallBegin)
		// Do not yield Ready, as the error should occur before that
	}
}

func (s *mockStreamToolNotFound) Message() Message {
	if s.message.Content == nil {
		s.message = Message{
			Role:      "assistant",
			Content:   content.FromText("Trying a tool..."),
			ToolCalls: s.message.ToolCalls,
		}
	}
	return s.message
}

func (s *mockStreamToolNotFound) Text() string { return "Trying a tool..." }

func (s *mockStreamToolNotFound) Image() (string, string) { return "", "" }

func (s *mockStreamToolNotFound) Thought() content.Thought { return content.Thought{} }

func (s *mockStreamToolNotFound) ToolCall() ToolCall {
	if len(s.message.ToolCalls) > 0 {
		return s.message.ToolCalls[len(s.message.ToolCalls)-1]
	}
	return ToolCall{}
}

func (s *mockStreamToolNotFound) Usage() Usage { return Usage{} }

// mockProviderToolNotFound returns the mockStreamToolNotFound.
type mockProviderToolNotFound struct {
	mockProvider // Embed basic mockProvider
}

func (m *mockProviderToolNotFound) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) ProviderStream {
	// Manually update embedded mockProvider fields
	m.mockProvider.generateCalled = true
	m.mockProvider.systemPrompt = systemPrompt
	m.mockProvider.messages = messages
	m.mockProvider.toolbox = toolbox
	m.mockProvider.jsonOutputSchema = jsonOutputSchema // Store schema
	return &mockStreamToolNotFound{}
}

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
