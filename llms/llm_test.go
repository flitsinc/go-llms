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

// TestCalculatorParams and TestWeatherParams are test structs for our regular tools
type TestCalculatorParams struct {
	Numbers []int  `json:"numbers"`
	Op      string `json:"op"`
}

type TestWeatherParams struct {
	Location string `json:"location"`
}

// TestChatFlow tests the complete chat flow similar to main.go
func TestChatFlow(t *testing.T) {
	// Create a mock provider with tool calls to make
	mockProv := &mockProvider{
		toolCallsToMake: []string{"test_tool"},
	}

	// Create a test tool
	type TestToolParams struct {
		TestParam string `json:"test_param"`
	}

	testTool := tools.Func("Test Tool", "A test tool for testing", "test_tool",
		func(r tools.Runner, p TestToolParams) tools.Result {
			return tools.Success("Test tool result", map[string]any{
				"result": fmt.Sprintf("Processed: %s", p.TestParam),
			})
		})

	// Create an LLM with the test tool
	llm := New(mockProv, testTool)

	// Set a system prompt function
	timeNow := time.Now()
	expectedTimeString := timeNow.Format(time.RFC3339)
	llm.SystemPrompt = func() content.Content {
		return content.Textf("This is a test system prompt. Current time: %s", expectedTimeString)
	}

	// Collect all updates to verify them after the chat
	var updates []Update
	var updatesMutex sync.Mutex

	// Start chat with a test message
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	for update := range llm.ChatWithContext(ctx, "Test message") {
		updatesMutex.Lock()
		updates = append(updates, update)
		updatesMutex.Unlock()
	}

	// Verify provider was called with correct system prompt
	require.True(t, mockProv.generateCalled, "Provider's Generate method should be called")
	require.NotNil(t, mockProv.systemPrompt, "System prompt should be passed to provider")

	// Extract text from the content parts
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

	// Verify updates - we should have:
	// 1. A TextUpdate with the initial assistant message.
	// 2. A ToolStartUpdate for "test_tool".
	// 3. A ToolDoneUpdate with the result from "test_tool".
	// 4. A TextUpdate with the final assistant message after processing the tool result.
	require.Equal(t, 4, len(updates), "Should receive 4 updates")

	// First update should be text
	textUpdate, ok := updates[0].(TextUpdate)
	require.True(t, ok, "First update should be TextUpdate")
	assert.Equal(t, "This is a test message.", textUpdate.Text)

	// Second update should be tool start
	toolStartUpdate, ok := updates[1].(ToolStartUpdate)
	require.True(t, ok, "Second update should be ToolStartUpdate")
	assert.Equal(t, "Test Tool", toolStartUpdate.Tool.Label())
	assert.Equal(t, "test_tool", toolStartUpdate.Tool.FuncName())

	// Third update should be tool done
	toolDoneUpdate, ok := updates[2].(ToolDoneUpdate)
	require.True(t, ok, "Third update should be ToolDoneUpdate")
	assert.Equal(t, "Test Tool", toolDoneUpdate.Tool.Label())
	assert.JSONEq(t, `{"result":"Processed: test_value_test_tool"}`, string(toolDoneUpdate.Result.JSON()))

	// Fourth update should be the final text response after tool processing
	secondTextUpdate, ok := updates[3].(TextUpdate)
	require.True(t, ok, "Fourth update should be TextUpdate")
	assert.Equal(t, "I've processed the results from the tool.", secondTextUpdate.Text)

	// Verify message history was updated correctly
	// Expected: user -> assistant (with tool call) -> tool (response) -> assistant (final response)
	assert.Equal(t, 4, len(llm.lastSentMessages), "Should have 4 messages in history")
	assert.Equal(t, "user", llm.lastSentMessages[0].Role, "First message should be from user")
	assert.Equal(t, "assistant", llm.lastSentMessages[1].Role, "Second message should be from assistant (initiating tool call)")
	assert.True(t, len(llm.lastSentMessages[1].ToolCalls) > 0, "Second assistant message should contain tool calls")
	assert.Equal(t, "tool", llm.lastSentMessages[2].Role, "Third message should be from tool")
	assert.Equal(t, "assistant", llm.lastSentMessages[3].Role, "Fourth message should be from assistant (final response)")
	assert.True(t, len(llm.lastSentMessages[3].ToolCalls) == 0, "Final assistant message should not contain tool calls")

	// Verify cost was tracked
	assert.Equal(t, 0.0002, llm.TotalCost(), "Cost should be tracked")
}

// TestAddExternalTools tests that the AddExternalTools function properly adds external tools
// alongside regular tools to the LLM's toolbox
func TestAddExternalTools(t *testing.T) {
	// Create a mock provider
	mockProv := &mockProvider{}

	// Create an LLM with two regular tools
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

	weather := tools.Func("Weather", "Get weather information", "weather",
		func(r tools.Runner, p TestWeatherParams) tools.Result {
			return tools.Success("Weather info", map[string]any{
				"temperature": 72,
				"condition":   "sunny",
			})
		})

	llm := New(mockProv, calculator, weather)

	// Create external tool schemas
	externalSchemas := []tools.FunctionSchema{
		{
			Name:        "search",
			Description: "Search for information",
			Parameters: tools.ValueSchema{
				Type: "object",
				Properties: &map[string]tools.ValueSchema{
					"query": {Type: "string"},
				},
				Required: []string{"query"},
			},
		},
		{
			Name:        "database",
			Description: "Query a database",
			Parameters: tools.ValueSchema{
				Type: "object",
				Properties: &map[string]tools.ValueSchema{
					"sql": {Type: "string"},
				},
				Required: []string{"sql"},
			},
		},
	}

	// Variables to track which external tools were called
	externalToolCalls := make(map[string]json.RawMessage)

	// Add external tools to the LLM
	llm.AddExternalTools(externalSchemas, func(r tools.Runner, funcName string, params json.RawMessage) tools.Result {
		// Store the tool call for verification
		externalToolCalls[funcName] = params

		switch funcName {
		case "search":
			return tools.Success("Search results", map[string]any{"results": []string{"result1", "result2"}})
		case "database":
			return tools.Success("Database results", map[string]any{"rows": 10})
		default:
			return tools.Error("Unknown tool", nil)
		}
	})

	// Verify the toolbox has all four tools (2 regular + 2 external)
	assert.Equal(t, 4, len(llm.toolbox.All()), "Toolbox should have 4 tools")

	// Verify each tool exists in the toolbox
	assert.NotNil(t, llm.toolbox.Get("calculator"), "Calculator tool should exist")
	assert.NotNil(t, llm.toolbox.Get("weather"), "Weather tool should exist")
	assert.NotNil(t, llm.toolbox.Get("search"), "Search tool should exist")
	assert.NotNil(t, llm.toolbox.Get("database"), "Database tool should exist")

	// Verify the tools can be called
	calcParams := json.RawMessage(`{"numbers":[2,3,4],"op":"add"}`)
	calcResult := llm.toolbox.Get("calculator").Run(tools.NopRunner, calcParams)
	assert.NoError(t, calcResult.Error(), "Calculator should not return error")
	assert.JSONEq(t, `{"result":9}`, string(calcResult.JSON()), "Calculator should return correct result")

	weatherParams := json.RawMessage(`{"location":"New York"}`)
	weatherResult := llm.toolbox.Get("weather").Run(tools.NopRunner, weatherParams)
	assert.NoError(t, weatherResult.Error(), "Weather should not return error")
	assert.JSONEq(t, `{"temperature":72,"condition":"sunny"}`, string(weatherResult.JSON()), "Weather should return correct result")

	searchParams := json.RawMessage(`{"query":"test query"}`)
	searchResult := llm.toolbox.Get("search").Run(tools.NopRunner, searchParams)
	assert.NoError(t, searchResult.Error(), "Search should not return error")
	assert.JSONEq(t, `{"results":["result1","result2"]}`, string(searchResult.JSON()), "Search should return correct result")
	assert.JSONEq(t, `{"query":"test query"}`, string(externalToolCalls["search"]), "External tool should receive correct parameters")

	dbParams := json.RawMessage(`{"sql":"SELECT * FROM test"}`)
	dbResult := llm.toolbox.Get("database").Run(tools.NopRunner, dbParams)
	assert.NoError(t, dbResult.Error(), "Database should not return error")
	assert.JSONEq(t, `{"rows":10}`, string(dbResult.JSON()), "Database should return correct result")
	assert.JSONEq(t, `{"sql":"SELECT * FROM test"}`, string(externalToolCalls["database"]), "External tool should receive correct parameters")

	// Directly call Generate to verify the toolbox is passed correctly
	llm.provider.Generate(nil, []Message{}, llm.toolbox)
	assert.True(t, mockProv.generateCalled, "Provider's Generate method should be called")
	assert.Equal(t, 4, mockProv.toolboxToolsCount, "All 4 tools should be available to the provider")
}

// TestErrorHandling tests error handling in the Chat flow
func TestErrorHandling(t *testing.T) {
	// Create a mock provider that will return an error
	errorProvider := &errorMockProvider{
		errorMessage: "test error",
	}

	// Create an LLM with the error provider
	llm := New(errorProvider)

	// Start chat and collect updates
	var updates []Update
	for update := range llm.Chat("Test message") {
		updates = append(updates, update)
	}

	// Verify we received an error update
	require.Equal(t, 1, len(updates), "Should receive 1 update")
	errorUpdate, ok := updates[0].(ErrorUpdate)
	require.True(t, ok, "Update should be ErrorUpdate")
	assert.Contains(t, errorUpdate.Error.Error(), "test error")

	// Verify the LLM's Err() method also returns the error
	require.Error(t, llm.Err(), "LLM.Err() should return an error")
	assert.Contains(t, llm.Err().Error(), "test error", "LLM.Err() should contain the error message")
}

// Mock provider that always returns an error
type errorMockProvider struct {
	errorMessage string
}

func (m *errorMockProvider) Company() string {
	return "Error Test Company"
}

func (m *errorMockProvider) Generate(systemPrompt content.Content, messages []Message, toolbox *tools.Toolbox) ProviderStream {
	return &errorMockStream{
		err: fmt.Errorf("error: %s", m.errorMessage),
	}
}

type errorMockStream struct {
	err error
}

func (s *errorMockStream) Err() error { return s.err }
func (s *errorMockStream) Iter() func(func(StreamStatus) bool) {
	return func(func(StreamStatus) bool) {}
}
func (s *errorMockStream) Message() Message                       { return Message{} }
func (s *errorMockStream) Text() string                           { return "" }
func (s *errorMockStream) ToolCall() ToolCall                     { return ToolCall{} }
func (s *errorMockStream) CostUSD() float64                       { return 0 }
func (s *errorMockStream) Usage() (inputTokens, outputTokens int) { return 0, 0 }

// TestCancellation tests that context cancellation is properly handled
func TestCancellation(t *testing.T) {
	// Create a mock provider with a long running operation
	mockProv := &mockProvider{}

	// Create an LLM with the provider
	llm := New(mockProv)

	// Create a context that will be cancelled immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	// Start chat and collect updates
	var updates []Update
	for update := range llm.ChatWithContext(ctx, "Test message") {
		updates = append(updates, update)
	}

	// Verify we received a cancellation error
	require.Equal(t, 1, len(updates), "Should receive 1 update")
	errorUpdate, ok := updates[0].(ErrorUpdate)
	require.True(t, ok, "Update should be ErrorUpdate")
	assert.Contains(t, errorUpdate.Error.Error(), "context canceled")

	// Verify the LLM's Err() method also returns the cancellation error
	require.Error(t, llm.Err(), "LLM.Err() should return an error")
	assert.Contains(t, llm.Err().Error(), "context canceled", "LLM.Err() should contain the cancellation error")
}

// TestMaxTurns tests that the WithMaxTurns functionality properly limits the number of turns
func TestMaxTurns(t *testing.T) {
	// Create a mock provider that always makes a tool call to force multiple turns
	mockProv := &mockProvider{
		toolCallsToMake: []string{"test_tool"}, // Will only call this tool on first turn
	}

	// Create a test tool
	type TestToolParams struct {
		TestParam string `json:"test_param"`
	}

	testTool := tools.Func("Test Tool", "A test tool for testing", "test_tool",
		func(r tools.Runner, p TestToolParams) tools.Result {
			return tools.Success("Test tool result", map[string]any{
				"result": fmt.Sprintf("Processed: %s", p.TestParam),
			})
		})

	// Create an LLM with the test tool and max 1 turn
	llm := New(mockProv, testTool)
	llm.WithMaxTurns(1)

	// Start chat and collect updates
	var updates []Update
	for update := range llm.Chat("Test message") {
		updates = append(updates, update)
	}

	// We should get:
	// 1. A TextUpdate with the initial assistant message
	// 2. A ToolStartUpdate for "test_tool"
	// 3. A ToolDoneUpdate with the result
	// 4. An ErrorUpdate with ErrMaxTurnsReached (since max turns = 1)

	// Verify we have the right number of updates
	require.Equal(t, 4, len(updates), "Should receive exactly 4 updates")

	// Verify the first three updates are the expected types
	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "First update should be TextUpdate")

	_, ok = updates[1].(ToolStartUpdate)
	require.True(t, ok, "Second update should be ToolStartUpdate")

	_, ok = updates[2].(ToolDoneUpdate)
	require.True(t, ok, "Third update should be ToolDoneUpdate")

	// The last update should be an error update with ErrMaxTurnsReached
	errorUpdate, ok := updates[3].(ErrorUpdate)
	require.True(t, ok, "Fourth update should be ErrorUpdate")
	assert.ErrorIs(t, errorUpdate.Error, ErrMaxTurnsReached, "Error should be ErrMaxTurnsReached")

	// Verify the LLM's Err() method also returns the max turns error
	require.Error(t, llm.Err(), "LLM.Err() should return an error")
	assert.ErrorIs(t, llm.Err(), ErrMaxTurnsReached, "LLM.Err() should return ErrMaxTurnsReached")

	// Verify turn count
	assert.Equal(t, 1, llm.turns, "Turn count should be 1")
}

// TestMaxTurnsComplete tests that an LLM with sufficient max turns completes successfully
func TestMaxTurnsComplete(t *testing.T) {
	// Create a mock provider that makes a tool call on first turn
	mockProv := &mockProvider{
		toolCallsToMake: []string{"test_tool"},
	}

	// Create a test tool
	type TestToolParams struct {
		TestParam string `json:"test_param"`
	}

	testTool := tools.Func("Test Tool", "A test tool for testing", "test_tool",
		func(r tools.Runner, p TestToolParams) tools.Result {
			return tools.Success("Test tool result", map[string]any{
				"result": fmt.Sprintf("Processed: %s", p.TestParam),
			})
		})

	// Create an LLM with the test tool and max 2 turns (which should be sufficient to complete)
	llm := New(mockProv, testTool)
	llm.WithMaxTurns(2)

	// Start chat and collect updates
	var updates []Update
	for update := range llm.Chat("Test message") {
		updates = append(updates, update)
	}

	// We should get:
	// 1. A TextUpdate with the initial assistant message
	// 2. A ToolStartUpdate for "test_tool"
	// 3. A ToolDoneUpdate with the result from "test_tool"
	// 4. A TextUpdate with the final response after processing tool result

	// Verify we have exactly 4 updates (no error)
	require.Equal(t, 4, len(updates), "Should receive exactly 4 updates")

	// None of the updates should be an error
	for i, update := range updates {
		_, ok := update.(ErrorUpdate)
		assert.False(t, ok, "Update %d should not be an error update", i)
	}

	// Verify the last update is a text update (the final response)
	finalTextUpdate, ok := updates[3].(TextUpdate)
	require.True(t, ok, "Final update should be TextUpdate")
	assert.Equal(t, "I've processed the results from the tool.", finalTextUpdate.Text)

	// Verify turn count is 2
	assert.Equal(t, 2, llm.turns, "Turn count should be 2")
}
