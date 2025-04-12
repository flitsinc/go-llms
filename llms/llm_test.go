package llms

import (
	"context"
	"encoding/json"
	"fmt"
	"sync"
	"testing"
	"time"

	"github.com/blixt/go-llms/content"
	"github.com/blixt/go-llms/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// --- Common Test Tool Definition ---

type TestToolParams struct {
	TestParam string `json:"test_param"`
}

var testTool = tools.Func("Test Tool", "A test tool for testing", "test_tool",
	func(r tools.Runner, p TestToolParams) tools.Result {
		return tools.Success("Test tool result", map[string]any{
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
	toolboxToolsCount      int
	toolCallsToMake        []string // Names of tools to simulate calls for on the *first* Generate call
	processedToolResponses bool     // Tracks if we've seen tool responses in messages
}

func (m *mockProvider) Company() string {
	return "Test Company"
}

func (m *mockProvider) Generate(systemPrompt content.Content, messages []Message, toolbox *tools.Toolbox) ProviderStream {
	m.generateCalled = true
	m.systemPrompt = systemPrompt
	m.messages = messages
	m.toolbox = toolbox
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
			s.message.ToolCalls = append(s.message.ToolCalls, ToolCall{
				ID:        uniqueID,
				Name:      toolName,
				Arguments: json.RawMessage(fmt.Sprintf(`{"test_param":"test_value_%s"}`, toolName)),
			})

			if !yield(StreamStatusToolCallBegin) {
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

func (s *mockStream) ToolCall() ToolCall {
	if len(s.message.ToolCalls) > 0 {
		return s.message.ToolCalls[len(s.message.ToolCalls)-1]
	}
	return ToolCall{}
}

func (s *mockStream) CostUSD() float64 { return 0.0001 }

func (s *mockStream) Usage() (inputTokens, outputTokens int) { return 10, 20 }

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

// --- Test Structs for Other Tools ---

// --- Tests ---

// TestChatFlow tests the complete chat flow with a tool call and response.
func TestChatFlow(t *testing.T) {
	// Arrange: Mock provider, LLM with test tool, and system prompt
	mockProv := &mockProvider{
		toolCallsToMake: []string{"test_tool"},
	}
	llm, _ := setupTestLLM(t, mockProv, testTool)
	timeNow := time.Now()
	expectedTimeString := timeNow.Format(time.RFC3339)
	llm.SystemPrompt = func() content.Content {
		return content.Textf("This is a test system prompt. Current time: %s", expectedTimeString)
	}

	// Act: Run the chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: Check for errors and provider calls
	assert.NoError(t, llm.Err(), "No error should occur in the chat flow")
	require.True(t, mockProv.generateCalled, "Provider's Generate method should be called")
	require.NotNil(t, mockProv.systemPrompt, "System prompt should be passed to provider")

	// Assert: System prompt content
	var systemPromptText string
	for _, part := range mockProv.systemPrompt {
		if part.Type() == content.TypeText {
			textPart, ok := part.(*content.Text)
			if ok {
				systemPromptText += textPart.Text
			}
		}
	}
	assert.Contains(t, systemPromptText, "This is a test system prompt")
	assert.Contains(t, systemPromptText, expectedTimeString)

	// Assert: Updates received
	require.Equal(t, 4, len(updates), "Should receive 4 updates")
	textUpdate, ok := updates[0].(TextUpdate)
	require.True(t, ok, "First update should be TextUpdate")
	assert.Equal(t, "This is a test message.", textUpdate.Text)

	toolStartUpdate, ok := updates[1].(ToolStartUpdate)
	require.True(t, ok, "Second update should be ToolStartUpdate")
	assert.Equal(t, "Test Tool", toolStartUpdate.Tool.Label())
	assert.Equal(t, "test_tool", toolStartUpdate.Tool.FuncName())
	assert.Equal(t, "test_tool-id-0", toolStartUpdate.ToolCallID, "ToolCallID should match the ID from the message")

	toolDoneUpdate, ok := updates[2].(ToolDoneUpdate)
	require.True(t, ok, "Third update should be ToolDoneUpdate")
	assert.Equal(t, "Test Tool", toolDoneUpdate.Tool.Label())
	assert.Equal(t, "test_tool-id-0", toolDoneUpdate.ToolCallID, "ToolCallID should match the ID from the message")
	assert.JSONEq(t, `{"result":"Processed: test_value_test_tool"}`, string(toolDoneUpdate.Result.JSON()), "Search should return correct result")

	secondTextUpdate, ok := updates[3].(TextUpdate)
	require.True(t, ok, "Fourth update should be TextUpdate")
	assert.Equal(t, "I've processed the results from the tool.", secondTextUpdate.Text)

	// Assert: Message history
	assert.Equal(t, 4, len(llm.lastSentMessages), "Should have 4 messages in history")
	assert.Equal(t, "user", llm.lastSentMessages[0].Role, "First message should be from user")
	assert.Equal(t, "assistant", llm.lastSentMessages[1].Role, "Second message should be from assistant (initiating tool call)")
	assert.True(t, len(llm.lastSentMessages[1].ToolCalls) > 0, "Second assistant message should contain tool calls")
	assert.Equal(t, "tool", llm.lastSentMessages[2].Role, "Third message should be from tool")
	assert.Equal(t, "assistant", llm.lastSentMessages[3].Role, "Fourth message should be from assistant (final response)")
	assert.True(t, len(llm.lastSentMessages[3].ToolCalls) == 0, "Final assistant message should not contain tool calls")

	// Assert: Cost tracking
	assert.Equal(t, 0.0002, llm.TotalCost(), "Cost should be tracked")
}

// TestErrorHandling tests that errors from the ProviderStream are propagated correctly.
func TestErrorHandling(t *testing.T) {
	// Arrange: Error provider and LLM
	errorProvider := &errorMockProvider{
		errorMessage: "test provider stream error",
	}
	llm := New(errorProvider)

	// Act: Run chat (will error)
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	_ = runTestChat(ctx, t, llm, "Test message")

	// Assert: Check for the expected error
	require.Error(t, llm.Err(), "LLM.Err() should return an error")
	assert.Contains(t, llm.Err().Error(), "test provider stream error", "Error should contain provider's error message")
	assert.Contains(t, llm.Err().Error(), "LLM returned error response", "Error should indicate it came from the LLM layer")
}

// TestContextCancellation tests that context cancellation is properly handled during chat.
func TestContextCancellation(t *testing.T) {
	// Arrange: Standard provider and LLM
	mockProv := &mockProvider{}
	llm, _ := setupTestLLM(t, mockProv)

	// Arrange: Context that cancels immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	// Act: Run chat with cancelled context
	_ = runTestChat(ctx, t, llm, "Test message")

	// Assert: Check for cancellation error
	require.Error(t, llm.Err(), "LLM.Err() should return an error")
	assert.ErrorIs(t, llm.Err(), context.Canceled, "Error should be context.Canceled")
}

// TestMaxTurnsReached tests that WithMaxTurns limits turns and returns ErrMaxTurnsReached.
func TestMaxTurnsReached(t *testing.T) {
	// Arrange: Mock provider forcing tool calls, LLM with MaxTurns(1)
	mockProv := &mockProvider{
		toolCallsToMake: []string{"test_tool"},
	}
	llm, _ := setupTestLLM(t, mockProv, testTool)
	llm.WithMaxTurns(1)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: Updates received (should stop after tool done)
	require.Equal(t, 3, len(updates), "Should receive exactly 3 updates")
	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "First update should be TextUpdate")
	_, ok = updates[1].(ToolStartUpdate)
	require.True(t, ok, "Second update should be ToolStartUpdate")
	_, ok = updates[2].(ToolDoneUpdate)
	require.True(t, ok, "Third update should be ToolDoneUpdate")

	// Assert: Max turns error and turn count
	require.Error(t, llm.Err(), "LLM.Err() should return an error")
	assert.ErrorIs(t, llm.Err(), ErrMaxTurnsReached, "Error should be ErrMaxTurnsReached")
	assert.Equal(t, 1, llm.turns, "Turn count should be 1")
}

// TestMaxTurnsAllowsCompletion tests that an LLM with sufficient max turns completes successfully.
func TestMaxTurnsAllowsCompletion(t *testing.T) {
	// Arrange: Mock provider forcing tool calls, LLM with MaxTurns(2)
	mockProv := &mockProvider{
		toolCallsToMake: []string{"test_tool"},
	}
	llm, _ := setupTestLLM(t, mockProv, testTool)
	llm.WithMaxTurns(2)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: Updates received (should include final text)
	require.Equal(t, 4, len(updates), "Should receive exactly 4 updates")
	finalTextUpdate, ok := updates[3].(TextUpdate)
	require.True(t, ok, "Final update should be TextUpdate")
	assert.Equal(t, "I've processed the results from the tool.", finalTextUpdate.Text)

	// Assert: No error and correct turn count
	assert.NoError(t, llm.Err(), "No error should occur in this test")
	assert.Equal(t, 2, llm.turns, "Turn count should be 2")
}

// TestEmptyToolCallIDError tests that an empty ToolCall ID from the stream causes an error.
func TestEmptyToolCallIDError(t *testing.T) {
	// Arrange: Empty ID provider and LLM
	mockProv := &mockEmptyIDProvider{}
	llm := New(mockProv, testTool)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: Limited updates before error
	require.Equal(t, 1, len(updates), "Should receive exactly 1 update")
	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "First update should be TextUpdate")

	// Assert: Correct error details
	require.Error(t, llm.Err(), "LLM.Err() should return an error")
	assert.Contains(t, llm.Err().Error(), "missing tool call ID", "Error should mention missing tool call ID")
	assert.Contains(t, llm.Err().Error(), "test_tool", "Error should include the tool name")
	assert.Contains(t, llm.Err().Error(), "error streaming", "Error should indicate it came from the streaming part")
}

// TestSuccessfulChatNoError tests that a successful chat returns nil from Err().
func TestSuccessfulChatNoError(t *testing.T) {
	// Arrange: Standard provider and LLM
	mockProv := &mockProvider{}
	llm, _ := setupTestLLM(t, mockProv)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	_ = runTestChat(ctx, t, llm, "Test message")

	// Assert: No error
	assert.NoError(t, llm.Err(), "LLM.Err() should return nil for successful chat")
}

// Mock provider that always returns an error stream
type errorMockProvider struct {
	errorMessage string
}

func (m *errorMockProvider) Company() string {
	return "Error Test Company"
}

func (m *errorMockProvider) Generate(systemPrompt content.Content, messages []Message, toolbox *tools.Toolbox) ProviderStream {
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
func (s *errorMockStream) Message() Message                       { return Message{} }
func (s *errorMockStream) Text() string                           { return "" }
func (s *errorMockStream) ToolCall() ToolCall                     { return ToolCall{} }
func (s *errorMockStream) CostUSD() float64                       { return 0 }
func (s *errorMockStream) Usage() (inputTokens, outputTokens int) { return 0, 0 }

// mockEmptyIDProvider is a provider that returns tool calls with empty IDs
type mockEmptyIDProvider struct{}

func (m *mockEmptyIDProvider) Company() string {
	return "Test Company Empty ID"
}

func (m *mockEmptyIDProvider) Generate(systemPrompt content.Content, messages []Message, toolbox *tools.Toolbox) ProviderStream {
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

func (s *mockEmptyIDStream) ToolCall() ToolCall {
	if len(s.message.ToolCalls) > 0 {
		return s.message.ToolCalls[len(s.message.ToolCalls)-1]
	}
	return ToolCall{}
}

func (s *mockEmptyIDStream) CostUSD() float64 { return 0.0001 }

func (s *mockEmptyIDStream) Usage() (inputTokens, outputTokens int) { return 10, 20 }

// TestAddExternalTools tests adding external tools via schemas.
func TestAddExternalTools(t *testing.T) {
	// Arrange: Mock provider, LLM with regular tools
	mockProv := &mockProvider{}

	// Define tools locally within the test
	type TestCalculatorParams struct {
		Numbers []int  `json:"numbers"`
		Op      string `json:"op"`
	}
	calculator := tools.Func("Calculator", "Perform math operations", "calculator",
		func(r tools.Runner, p TestCalculatorParams) tools.Result {
			var result int
			switch p.Op {
			case "add":
				result = 0
				for _, n := range p.Numbers {
					result += n
				}
			case "multiply":
				result = 1
				for _, n := range p.Numbers {
					result *= n
				}
			}
			return tools.Success("Calculator result", map[string]any{"result": result})
		})

	type TestWeatherParams struct {
		Location string `json:"location"`
	}
	weather := tools.Func("Weather", "Get weather information", "weather",
		func(r tools.Runner, p TestWeatherParams) tools.Result {
			return tools.Success("Weather info", map[string]any{
				"temperature": 72,
				"condition":   "sunny",
			})
		})

	llm, _ := setupTestLLM(t, mockProv, calculator, weather)

	// Arrange: External tool setup
	externalSchemas := []tools.FunctionSchema{
		{Name: "search", Description: "Perform a search", Parameters: tools.ValueSchema{ /*...*/ }},
		{Name: "database", Description: "Query a database", Parameters: tools.ValueSchema{ /*...*/ }},
	}
	externalToolCalls := make(map[string]json.RawMessage)
	handler := func(r tools.Runner, params json.RawMessage) tools.Result {
		tc, ok := GetToolCall(r.Context())
		if !ok {
			return tools.Error("ToolCall not found in context", nil)
		}
		externalToolCalls[tc.Name] = params
		switch tc.Name {
		case "search":
			return tools.Success("Search results", map[string]any{"results": []string{"result1", "result2"}})
		case "database":
			return tools.Success("Database results", map[string]any{"rows": 10})
		default:
			return tools.Error("Unknown tool", nil)
		}
	}

	// Act: Add external tools
	llm.AddExternalTools(externalSchemas, handler)

	// Assert: Toolbox contents
	assert.Equal(t, 4, len(llm.toolbox.All()), "Toolbox should have 4 tools")
	assert.NotNil(t, llm.toolbox.Get("calculator"), "Calculator tool should exist")
	assert.NotNil(t, llm.toolbox.Get("weather"), "Weather tool should exist")
	assert.NotNil(t, llm.toolbox.Get("search"), "Search tool should exist")
	assert.NotNil(t, llm.toolbox.Get("database"), "Database tool should exist")

	// Assert: Tool execution (regular)
	calcParams := json.RawMessage(`{"numbers":[2,3,4],"op":"add"}`)
	calcResult := llm.toolbox.Get("calculator").Run(tools.NopRunner, calcParams)
	assert.NoError(t, calcResult.Error())
	assert.JSONEq(t, `{"result":9}`, string(calcResult.JSON()))
	weatherParams := json.RawMessage(`{"location":"New York"}`)
	weatherResult := llm.toolbox.Get("weather").Run(tools.NopRunner, weatherParams)
	assert.NoError(t, weatherResult.Error())
	assert.JSONEq(t, `{"temperature":72,"condition":"sunny"}`, string(weatherResult.JSON()))

	// Assert: Tool execution (external)
	searchParams := json.RawMessage(`{"query":"test query"}`)
	searchResult := llm.toolbox.Get("search").Run(tools.NopRunner, searchParams)
	assert.NoError(t, searchResult.Error())
	assert.JSONEq(t, `{"results":["result1","result2"]}`, string(searchResult.JSON()))
	assert.JSONEq(t, `{"query":"test query"}`, string(externalToolCalls["search"]), "External tool should receive correct parameters")
	dbParams := json.RawMessage(`{"sql":"SELECT * FROM test"}`)
	dbResult := llm.toolbox.Get("database").Run(tools.NopRunner, dbParams)
	assert.NoError(t, dbResult.Error())
	assert.JSONEq(t, `{"rows":10}`, string(dbResult.JSON()))
	assert.JSONEq(t, `{"sql":"SELECT * FROM test"}`, string(externalToolCalls["database"]), "External tool should receive correct parameters")

	// Assert: Provider receives full toolbox
	llm.provider.Generate(nil, []Message{}, llm.toolbox)
	assert.True(t, mockProv.generateCalled, "Provider's Generate method should be called")
	assert.Equal(t, 4, mockProv.toolboxToolsCount, "All 4 tools should be available to the provider")
}

// TestToolCallIDsFlow tests that tool call IDs flow correctly through the update chain.
func TestToolCallIDsFlow(t *testing.T) {
	// Arrange: Mock provider forcing multiple tool calls
	mockProv := &mockProvider{
		toolCallsToMake: []string{"tool1", "tool2", "tool3"},
	}

	// Define tools locally within the test
	type Tool1Params struct {
		TestParam string `json:"test_param"`
	}
	type Tool2Params struct {
		TestParam string `json:"test_param"`
	}
	type Tool3Params struct {
		TestParam string `json:"test_param"`
	}
	tool1 := tools.Func("Tool 1", "Test tool 1", "tool1", func(r tools.Runner, p Tool1Params) tools.Result {
		return tools.Success("Tool 1 result", map[string]any{"result": "tool1 result"})
	})
	tool2 := tools.Func("Tool 2", "Test tool 2", "tool2", func(r tools.Runner, p Tool2Params) tools.Result {
		return tools.Success("Tool 2 result", map[string]any{"result": "tool2 result"})
	})
	tool3 := tools.Func("Tool 3", "Test tool 3", "tool3", func(r tools.Runner, p Tool3Params) tools.Result {
		return tools.Success("Tool 3 result", map[string]any{"result": "tool3 result"})
	})

	llm, _ := setupTestLLM(t, mockProv, tool1, tool2, tool3)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test multiple tool calls")

	// Assert: No error
	assert.NoError(t, llm.Err(), "No error should occur in this test")

	// Assert: Correct number and type of updates
	require.Equal(t, 8, len(updates), "Should receive exactly 8 updates")
	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "First update should be TextUpdate")
	_, ok = updates[7].(TextUpdate)
	require.True(t, ok, "Final update should be TextUpdate")

	// Assert: Tool call ID propagation and uniqueness
	var lastToolCallID string
	for i := 1; i < 7; i += 2 { // Iterate through start/done pairs
		start, startOK := updates[i].(ToolStartUpdate)
		done, doneOK := updates[i+1].(ToolDoneUpdate)
		require.True(t, startOK, "Update %d should be ToolStartUpdate", i)
		require.True(t, doneOK, "Update %d should be ToolDoneUpdate", i+1)

		expectedToolName := fmt.Sprintf("tool%d", (i/2)+1)
		expectedID := fmt.Sprintf("%s-id-%d", expectedToolName, i/2)

		assert.Equal(t, expectedToolName, start.Tool.FuncName(), "Tool name mismatch in start update %d", i)
		assert.Equal(t, expectedID, start.ToolCallID, "ToolCallID mismatch in start update %d", i)
		assert.Equal(t, expectedToolName, done.Tool.FuncName(), "Tool name mismatch in done update %d", i+1)
		assert.Equal(t, start.ToolCallID, done.ToolCallID, "ToolCallID should match in start/done pair %d", (i/2)+1)

		if i > 1 {
			assert.NotEqual(t, lastToolCallID, start.ToolCallID, "ToolCallIDs should be unique (pair %d vs previous)", (i/2)+1)
		}
		lastToolCallID = start.ToolCallID
	}
}

// TestAddToolWithNilToolbox tests that AddTool properly creates a toolbox when it's nil
func TestAddToolWithNilToolbox(t *testing.T) {
	// Arrange: Mock provider, LLM with no initial tools
	mockProv := &mockProvider{}
	llm, _ := setupTestLLM(t, mockProv)
	assert.Nil(t, llm.toolbox, "Toolbox should be nil initially")

	// Arrange: Tool to add
	simpleTool := tools.Func("Simple Tool", "A simple tool", "simple_tool",
		func(r tools.Runner, p TestToolParams) tools.Result {
			return tools.Success("Simple result", map[string]any{"response": p.TestParam})
		})

	// Act: Add the tool
	llm.AddTool(simpleTool)

	// Assert: Toolbox created and tool added
	assert.NotNil(t, llm.toolbox, "Toolbox should be created")
	require.Equal(t, 1, len(llm.toolbox.All()), "Toolbox should have 1 tool")
	assert.Equal(t, "simple_tool", llm.toolbox.All()[0].FuncName(), "Tool should be added correctly")
}

// TestRunToolCallImageHandling tests the image handling in runToolCall with a fixed implementation
func TestRunToolCallImageHandling(t *testing.T) {
	// Arrange: Image tool
	type ImageToolParams struct {
		ImageURL string `json:"image_url"`
	}
	imageTestTool := tools.Func("Image Tool", "A tool that returns an image", "image_tool",
		func(r tools.Runner, p ImageToolParams) tools.Result {
			return &imageResultWithImages{
				label: "Image result",
				data:  json.RawMessage(`{"result":"Image generated"}`),
				imgs:  []tools.Image{{Name: "test-image", URL: p.ImageURL}},
			}
		})

	// Arrange: Image provider and LLM
	provider := &mockProviderWithImages{}
	llm := New(provider, imageTestTool)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	_ = runTestChat(ctx, t, llm, "Generate a test image")

	// Assert: No error
	assert.NoError(t, llm.Err(), "Should not return an error")

	// Assert: User message with image content exists in history
	var foundImageMessage bool
	for _, msg := range llm.lastSentMessages {
		if msg.Role == "user" {
			for _, part := range msg.Content {
				if part.Type() == content.TypeImageURL {
					foundImageMessage = true
					break
				}
			}
		}
		if foundImageMessage {
			break
		}
	}
	assert.True(t, foundImageMessage, "Should have a user message with image content")
}

// mockProviderWithImages is a fixed implementation for the image test
type mockProviderWithImages struct{}

func (m *mockProviderWithImages) Company() string { return "Mock Image Provider" }

func (m *mockProviderWithImages) Generate(systemPrompt content.Content, messages []Message, toolbox *tools.Toolbox) ProviderStream {
	// Check if we've already processed tool responses in a previous turn
	processedToolResponses := false
	for _, msg := range messages {
		if msg.Role == "tool" {
			processedToolResponses = true
			break
		}
	}

	// If we've processed tool responses, return a stream with just text
	if processedToolResponses {
		return &mockStreamWithImages{
			// No toolCallID needed, Iter will only yield text
		}
	}

	// Otherwise, this is the first turn, make the tool call
	uniqueID := fmt.Sprintf("image-tool-id-%d", time.Now().UnixNano())
	return &mockStreamWithImages{
		toolCallID: uniqueID, // Iter will yield tool call statuses
	}
}

// mockStreamWithImages is a fixed implementation that correctly handles streaming
type mockStreamWithImages struct {
	message      Message
	toolCallMade bool
	toolCallID   string // Will be empty if we shouldn't make a call
}

func (s *mockStreamWithImages) Err() error { return nil }

func (s *mockStreamWithImages) Iter() func(func(StreamStatus) bool) {
	return func(yield func(StreamStatus) bool) {
		// First yield text
		if !yield(StreamStatusText) {
			return
		}

		// Only make the tool call if ID is present and not already made
		if s.toolCallID != "" && !s.toolCallMade {
			// Then yield a tool call to a tool that will return an image
			s.message.ToolCalls = append(s.message.ToolCalls, ToolCall{
				ID:        s.toolCallID,
				Name:      "image_tool",
				Arguments: json.RawMessage(`{"image_url":"https://example.com/test-image.jpg"}`),
			})

			if !yield(StreamStatusToolCallBegin) {
				return
			}

			if !yield(StreamStatusToolCallReady) {
				return
			}

			s.toolCallMade = true
		}
		// Stream ends here naturally
	}
}

func (s *mockStreamWithImages) Message() Message {
	text := "I'll generate an image for you."
	if s.toolCallID == "" { // Adjust text if this is the second turn
		text = "I have processed the image tool result."
	}

	if s.message.Content == nil {
		s.message = Message{
			Role:      "assistant",
			Content:   content.FromText(text),
			ToolCalls: s.message.ToolCalls, // ToolCalls will be empty if toolCallID was empty
		}
	}
	return s.message
}

func (s *mockStreamWithImages) Text() string {
	if s.toolCallID == "" {
		return "I have processed the image tool result."
	}
	return "I'll generate an image for you."
}

func (s *mockStreamWithImages) ToolCall() ToolCall {
	if len(s.message.ToolCalls) > 0 {
		return s.message.ToolCalls[len(s.message.ToolCalls)-1]
	}
	return ToolCall{}
}

func (s *mockStreamWithImages) CostUSD() float64 { return 0.0001 }

func (s *mockStreamWithImages) Usage() (inputTokens, outputTokens int) { return 10, 20 }

// imageResultWithImages is a custom tools.Result implementation that returns images (needed by the test)
type imageResultWithImages struct {
	label string
	data  json.RawMessage
	imgs  []tools.Image
}

func (r *imageResultWithImages) Label() string         { return r.label }
func (r *imageResultWithImages) JSON() json.RawMessage { return r.data }
func (r *imageResultWithImages) Error() error          { return nil }
func (r *imageResultWithImages) Images() []tools.Image { return r.imgs }

// TestTurnContextCancellation tests the context cancellation path in the turn method with a fixed implementation
func TestTurnContextCancellation(t *testing.T) {
	// Create a provider that will block until context is cancelled
	cancelProvider := &mockCancellingProvider{}

	// Create an LLM with the cancelling provider
	llm := New(cancelProvider)

	// Create a context that will be cancelled after a short time
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// Start chat with the context
	startTime := time.Now()
	var updates []Update
	for update := range llm.ChatWithContext(ctx, "Test message") {
		updates = append(updates, update)
	}
	duration := time.Since(startTime)

	// Verify the chat was cancelled due to timeout
	assert.Less(t, duration, 1*time.Second, "Chat should be cancelled quickly")
	assert.Error(t, llm.Err(), "Should return an error when context is cancelled")
	assert.ErrorIs(t, llm.Err(), context.DeadlineExceeded, "Error should be context.DeadlineExceeded")
}

// mockCancellingProvider creates a stream that will block until context is cancelled (fixed implementation)
type mockCancellingProvider struct{}

func (m *mockCancellingProvider) Company() string { return "Mock Cancelling Provider" }

func (m *mockCancellingProvider) Generate(systemPrompt content.Content, messages []Message, toolbox *tools.Toolbox) ProviderStream {
	return &mockCancellingStream{}
}

// mockCancellingStream is a stream that blocks until context is cancelled (fixed implementation)
type mockCancellingStream struct {
	message Message
}

func (s *mockCancellingStream) Err() error { return nil }

func (s *mockCancellingStream) Iter() func(func(StreamStatus) bool) {
	return func(yield func(StreamStatus) bool) {
		// First yield text
		if !yield(StreamStatusText) {
			return
		}

		// Then block forever to force context cancellation
		<-time.After(10 * time.Minute)
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

func (s *mockCancellingStream) Text() string                           { return "This is a test message." }
func (s *mockCancellingStream) ToolCall() ToolCall                     { return ToolCall{} }
func (s *mockCancellingStream) CostUSD() float64                       { return 0 }
func (s *mockCancellingStream) Usage() (inputTokens, outputTokens int) { return 0, 0 }

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

func (s *mockStreamToolNotFound) ToolCall() ToolCall {
	if len(s.message.ToolCalls) > 0 {
		return s.message.ToolCalls[len(s.message.ToolCalls)-1]
	}
	return ToolCall{}
}

func (s *mockStreamToolNotFound) CostUSD() float64                       { return 0 }
func (s *mockStreamToolNotFound) Usage() (inputTokens, outputTokens int) { return 0, 0 }

// mockProviderToolNotFound returns the mockStreamToolNotFound.
type mockProviderToolNotFound struct {
	mockProvider // Embed basic mockProvider
}

func (m *mockProviderToolNotFound) Generate(systemPrompt content.Content, messages []Message, toolbox *tools.Toolbox) ProviderStream {
	m.generateCalled = true
	m.systemPrompt = systemPrompt
	m.messages = messages
	m.toolbox = toolbox
	return &mockStreamToolNotFound{}
}

// TestTurnToolNotFound tests the error path when a called tool is not in the toolbox.
func TestTurnToolNotFound(t *testing.T) {
	// Arrange: Provider that yields a non-existent tool, LLM with *some other* tool
	provider := &mockProviderToolNotFound{}
	llm, _ := setupTestLLM(t, provider, testTool) // LLM has 'test_tool', provider calls 'tool_does_not_exist'

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: Should get text update, then an error
	require.NotEmpty(t, updates, "Should receive at least one update before error")
	_, isText := updates[0].(TextUpdate)
	assert.True(t, isText, "First update should be text")

	require.Error(t, llm.Err(), "LLM should have an error")
	assert.Contains(t, llm.Err().Error(), "tool \"tool_does_not_exist\" not found", "Error message should indicate tool not found")
	assert.Contains(t, llm.Err().Error(), "error streaming", "Error should indicate it came from streaming")
}

// TestChatWrapper tests the simple llms.Chat wrapper function.
func TestChatWrapper(t *testing.T) {
	// Arrange: Use a standard mock provider and LLM
	mockProv := &mockProvider{}
	llm, _ := setupTestLLM(t, mockProv)

	// Act: Run chat using the simple wrapper
	// We don't need a long timeout or context here, just verifying the call.
	// Use a simple channel read to ensure it doesn't block indefinitely.
	chatChan := llm.Chat("Simple chat message")
	select {
	case update, ok := <-chatChan:
		require.True(t, ok, "Should receive at least one update")
		_, isText := update.(TextUpdate)
		assert.True(t, isText, "First update should be text")
		// Drain the rest to prevent goroutine leak
		go func() {
			for range chatChan {
			}
		}()
	case <-time.After(1 * time.Second):
		t.Fatal("llm.Chat did not produce an update within the timeout")
	}

	// Assert: Check provider was called
	assert.NoError(t, llm.Err(), "Simple chat should not produce an error")
	assert.True(t, mockProv.generateCalled, "Provider generate should be called")
	require.Len(t, mockProv.messages, 1, "Provider should receive one message")
	assert.Equal(t, "user", mockProv.messages[0].Role)
	require.Len(t, mockProv.messages[0].Content, 1)
	textPart, ok := mockProv.messages[0].Content[0].(*content.Text)
	require.True(t, ok)
	assert.Equal(t, "Simple chat message", textPart.Text)
}

// TestAlreadyCancelledContext tests behavior when context is cancelled before chat starts.
func TestAlreadyCancelledContext(t *testing.T) {
	// Arrange: Standard provider and LLM
	mockProv := &mockProvider{}
	llm, _ := setupTestLLM(t, mockProv)

	// Arrange: Already cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Act: Call ChatUsingMessages with the cancelled context
	updateChan := llm.ChatUsingMessages(ctx, []Message{{Role: "user", Content: content.FromText("test")}})

	// Assert: Channel should be closed immediately
	select {
	case _, ok := <-updateChan:
		assert.False(t, ok, "Channel should be closed")
	case <-time.After(1 * time.Second):
		t.Fatal("Channel was not closed immediately")
	}

	// Assert: LLM error should be set to context.Canceled
	require.Error(t, llm.Err(), "LLM error should be set")
	assert.ErrorIs(t, llm.Err(), context.Canceled, "Error should be context.Canceled")

	// Assert: Provider Generate should NOT have been called
	assert.False(t, mockProv.generateCalled, "Provider Generate should not be called")
}

// TestNilSystemPrompt tests chat behavior when LLM.SystemPrompt is nil.
func TestNilSystemPrompt(t *testing.T) {
	// Arrange: Provider and LLM, ensure SystemPrompt is nil
	mockProv := &mockProvider{}
	llm, _ := setupTestLLM(t, mockProv)
	llm.SystemPrompt = nil // Explicitly set to nil

	// Act: Run a simple chat
	ctx, cancel := context.WithTimeout(context.Background(), 1*time.Second)
	defer cancel()
	_ = runTestChat(ctx, t, llm, "Test message")

	// Assert: No error and provider received a nil system prompt
	assert.NoError(t, llm.Err())
	assert.True(t, mockProv.generateCalled, "Provider generate should be called")
	assert.Nil(t, mockProv.systemPrompt, "Provider should receive nil system prompt")
}

// mockToolWithError always returns an error.
var mockToolWithError = tools.Func("Error Tool", "A tool that always errors", "error_tool",
	func(r tools.Runner, p TestToolParams) tools.Result {
		return tools.Error("tool failed", fmt.Errorf("internal tool error detail"))
	})

// TestRunToolCallWithError tests the behavior when a called tool returns an error.
func TestRunToolCallWithError(t *testing.T) {
	// Arrange: Provider that calls the erroring tool
	mockProv := &mockProvider{
		toolCallsToMake: []string{"error_tool"},
	}
	llm, _ := setupTestLLM(t, mockProv, mockToolWithError)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: No LLM-level error (tool error shouldn't stop the flow)
	assert.NoError(t, llm.Err(), "LLM should not error just because a tool failed")

	// Assert: Correct updates received (Text, ToolStart, ToolDone, Final Text)
	require.Equal(t, 4, len(updates), "Should receive 4 updates")
	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "Update 0 should be TextUpdate")
	_, ok = updates[1].(ToolStartUpdate)
	require.True(t, ok, "Update 1 should be ToolStartUpdate")
	_, ok = updates[3].(TextUpdate)
	require.True(t, ok, "Update 3 should be TextUpdate")

	// Assert: ToolDoneUpdate contains the error
	doneUpdate, ok := updates[2].(ToolDoneUpdate)
	require.True(t, ok, "Update 2 should be ToolDoneUpdate")
	assert.Equal(t, "error_tool", doneUpdate.Tool.FuncName())
	require.NotNil(t, doneUpdate.Result, "Result should not be nil")
	assert.Error(t, doneUpdate.Result.Error(), "Result error should not be nil")
	assert.Contains(t, doneUpdate.Result.Error().Error(), "internal tool error detail", "Result error should contain tool's internal error")
	assert.Equal(t, "tool failed", doneUpdate.Result.Label(), "Result label should match the error message")
	// Check JSON representation - only contains the detail error
	assert.JSONEq(t, `{"error":"internal tool error detail"}`, string(doneUpdate.Result.JSON()))

	// Assert: Message history includes the tool result message with error
	require.Len(t, llm.lastSentMessages, 4, "Should have 4 messages in history")
	toolResultMessage := llm.lastSentMessages[2]
	assert.Equal(t, "tool", toolResultMessage.Role)
	require.NotNil(t, toolResultMessage.Content)
	require.Len(t, toolResultMessage.Content, 1, "Tool result content should have 1 part")
	jsonPart, ok := toolResultMessage.Content[0].(*content.JSON)
	require.True(t, ok, "Tool result content should be JSON")
	// Check JSON in history - only contains the detail error
	assert.JSONEq(t, `{"error":"internal tool error detail"}`, string(jsonPart.Data))
}

// mockToolForStatusTest is a simple tool used for testing status updates path.
var mockToolForStatusTest = tools.Func("Status Tool", "A tool used for status test", "status_tool",
	func(r tools.Runner, p TestToolParams) tools.Result {
		time.Sleep(10 * time.Millisecond) // Simulate work
		return tools.Success("Status result", map[string]any{"status": "done"})
	})

// TestToolStatusUpdates verifies the basic flow when a tool (potentially with status updates) runs.
// NOTE: This test *does not* verify the reception of ToolStatusUpdate messages,
// as that requires mocking the internal runner callback mechanism.
// It verifies that runToolCall is exercised with a tool call.
func TestToolStatusUpdates(t *testing.T) {
	// Arrange: Provider that calls the status tool
	mockProv := &mockProvider{
		toolCallsToMake: []string{"status_tool"},
	}
	llm, _ := setupTestLLM(t, mockProv, mockToolForStatusTest)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: No LLM-level error
	assert.NoError(t, llm.Err())

	// Assert: Correct basic updates received (Text, ToolStart, ToolDone, Final Text)
	require.Equal(t, 4, len(updates), "Should receive 4 standard updates")
	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "Update 0 should be TextUpdate")
	startUpdate, ok := updates[1].(ToolStartUpdate)
	require.True(t, ok, "Update 1 should be ToolStartUpdate")
	doneUpdate, ok := updates[2].(ToolDoneUpdate)
	require.True(t, ok, "Update 2 should be ToolDoneUpdate")
	_, ok = updates[3].(TextUpdate)
	require.True(t, ok, "Update 3 should be TextUpdate")

	// Assert: ToolDoneUpdate is correct
	assert.Equal(t, "status_tool", doneUpdate.Tool.FuncName())
	assert.Equal(t, startUpdate.ToolCallID, doneUpdate.ToolCallID, "Done ID should match start ID")
	assert.NoError(t, doneUpdate.Result.Error())
	assert.JSONEq(t, `{"status":"done"}`, string(doneUpdate.Result.JSON()))
}

// --- Test for ToolCall in Context ---

// toolThatChecksContext retrieves the ToolCall from the context and returns its ID.
var toolThatChecksContext = tools.Func("Context Checker Tool", "Checks for ToolCall in context", "context_checker_tool",
	func(r tools.Runner, p TestToolParams) tools.Result {
		tc, ok := GetToolCall(r.Context())
		if !ok {
			return tools.Error("Context check failed", fmt.Errorf("ToolCall not found in context or wrong type"))
		}
		return tools.Success("Context check success", map[string]any{"tool_call_id": tc.ID})
	})

// TestToolCallInContext verifies that the ToolCall struct is correctly placed in the context
// and accessible by the tool function via the runner.
func TestToolCallInContext(t *testing.T) {
	// Arrange: Provider that calls the context checking tool
	expectedToolCallID := "context_checker_tool-id-0"
	mockProv := &mockProvider{
		toolCallsToMake: []string{"context_checker_tool"},
	}
	llm, _ := setupTestLLM(t, mockProv, toolThatChecksContext)

	// Act: Run chat
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Test message")

	// Assert: No LLM-level error
	assert.NoError(t, llm.Err())

	// Assert: Correct updates received (Text, ToolStart, ToolDone, Final Text)
	require.Equal(t, 4, len(updates), "Should receive 4 standard updates")
	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "Update 0 should be TextUpdate")
	startUpdate, ok := updates[1].(ToolStartUpdate)
	require.True(t, ok, "Update 1 should be ToolStartUpdate")
	assert.Equal(t, expectedToolCallID, startUpdate.ToolCallID, "ToolCallID in start update should match expected")
	doneUpdate, ok := updates[2].(ToolDoneUpdate)
	require.True(t, ok, "Update 2 should be ToolDoneUpdate")
	_, ok = updates[3].(TextUpdate)
	require.True(t, ok, "Update 3 should be TextUpdate")

	// Assert: ToolDoneUpdate contains the correct ToolCallID extracted from the context
	assert.Equal(t, "context_checker_tool", doneUpdate.Tool.FuncName())
	assert.Equal(t, expectedToolCallID, doneUpdate.ToolCallID, "ToolCallID in done update should match expected")
	require.NoError(t, doneUpdate.Result.Error())
	assert.JSONEq(t, fmt.Sprintf(`{"tool_call_id":%q}`, expectedToolCallID), string(doneUpdate.Result.JSON()))
}
