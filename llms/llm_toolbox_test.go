package llms

import (
	"context"
	"encoding/json"
	"fmt"
	"testing"
	"time"

	"github.com/flitsinc/go-llms/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestAddExternalTools tests adding external tools via schemas.
func TestAddExternalTools(t *testing.T) {
	// Arrange: Mock provider, LLM with regular tools
	mockProv := &mockProvider{
		// Configure mock to call the external tools
		toolCallsToMake: []string{"search", "database"},
	}

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
			return tools.Success(map[string]any{"result": result})
		})

	type TestWeatherParams struct {
		Location string `json:"location"`
	}
	weather := tools.Func("Weather", "Get weather information", "weather",
		func(r tools.Runner, p TestWeatherParams) tools.Result {
			return tools.Success(map[string]any{
				"temperature": 72,
				"condition":   "sunny",
			})
		})

	llm, prov := setupTestLLM(t, mockProv, calculator, weather)
	require.NotNil(t, prov, "Cast to *mockProvider should succeed")

	// Arrange: External tool setup
	externalSchemas := []tools.FunctionSchema{
		{Name: "search", Description: "Perform a search", Parameters: tools.ValueSchema{ /*...*/ }},
		{Name: "database", Description: "Query a database", Parameters: tools.ValueSchema{ /*...*/ }},
	}
	externalToolCalls := make(map[string]json.RawMessage)
	handler := func(r tools.Runner, params json.RawMessage) tools.Result {
		tc, ok := GetToolCall(r.Context())
		if !ok {
			// This path should ideally not be hit in the refactored test, but handle defensively.
			// Since we now panic on nil errors, we must return a real error.
			return tools.ErrorWithLabel("ToolCall not found in context", fmt.Errorf("context missing tool call"))
		}
		externalToolCalls[tc.Name] = params
		switch tc.Name {
		case "search":
			// Simulate receiving specific args for assertion
			assert.JSONEq(t, `{"test_param":"test_value_search"}`, string(params))
			return tools.Success(map[string]any{"results": []string{"result1", "result2"}})
		case "database":
			// Simulate receiving specific args for assertion
			assert.JSONEq(t, `{"test_param":"test_value_database"}`, string(params))
			return tools.Success(map[string]any{"rows": 10})
		default:
			return tools.ErrorWithLabel("Unknown external tool", fmt.Errorf("unknown tool: %s", tc.Name))
		}
	}

	// Act: Add external tools
	llm.AddExternalTools(externalSchemas, handler)

	// Assert: Verify external tools have non-nil and correct schemas
	searchTool := llm.toolbox.Get("search")
	require.NotNil(t, searchTool, "Search tool should exist")
	searchSchema := searchTool.Schema()
	require.NotNil(t, searchSchema, "Schema for external tool 'search' should not be nil")
	assert.Equal(t, "search", searchSchema.Name)
	assert.Equal(t, "Perform a search", searchSchema.Description)

	dbTool := llm.toolbox.Get("database")
	require.NotNil(t, dbTool, "Database tool should exist")
	dbSchema := dbTool.Schema()
	require.NotNil(t, dbSchema, "Schema for external tool 'database' should not be nil")
	assert.Equal(t, "database", dbSchema.Name)
	assert.Equal(t, "Query a database", dbSchema.Description)

	// Assert: Toolbox contents (verify all tools are present)
	assert.Equal(t, 4, len(llm.toolbox.All()), "Toolbox should have 4 tools")
	assert.NotNil(t, llm.toolbox.Get("calculator"), "Calculator tool should exist")
	assert.NotNil(t, llm.toolbox.Get("weather"), "Weather tool should exist")
	assert.NotNil(t, llm.toolbox.Get("search"), "Search tool should exist")
	assert.NotNil(t, llm.toolbox.Get("database"), "Database tool should exist")

	// Act: Run chat to trigger the mock tool calls
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	updates := runTestChat(ctx, t, llm, "Run search and database tools")

	// Assert: No LLM error during the flow
	assert.NoError(t, llm.Err(), "Chat flow should complete without LLM error")

	// Assert: Updates received for the full flow (Text -> Search Tool Sequence -> Database Tool Sequence -> Text)
	// Expected: Text, Start(S), 2xDelta(S), Done(S), Start(DB), 2xDelta(DB), Done(DB), Text(Final)
	// 1 + (1+2+1) + (1+2+1) + 1 = 1 + 4 + 4 + 1 = 10
	require.Equal(t, 10, len(updates), "Should receive 10 updates for the full flow")

	_, ok := updates[0].(TextUpdate)
	require.True(t, ok, "Update 0 should be TextUpdate")

	// Search tool: TS at 1, TD at 4
	searchStart, ok := updates[1].(ToolStartUpdate)
	require.True(t, ok, "Update 1 should be ToolStartUpdate (search)")
	assert.Equal(t, "search", searchStart.Tool.FuncName())
	assert.Equal(t, "search-id-0", searchStart.ToolCallID)

	searchDone, ok := updates[4].(ToolDoneUpdate) // Was updates[2]
	require.True(t, ok, "Update 4 should be ToolDoneUpdate (search)")
	assert.Equal(t, "search", searchDone.Tool.FuncName())
	assert.Equal(t, searchStart.ToolCallID, searchDone.ToolCallID)
	require.NoError(t, searchDone.Result.Error())
	resultJSON := extractJSONFromResult(t, searchDone.Result)
	assert.JSONEq(t, `{"results":["result1","result2"]}`, string(resultJSON))

	// Database tool: TS at 5, TD at 8
	dbStart, ok := updates[5].(ToolStartUpdate) // Was updates[3]
	require.True(t, ok, "Update 5 should be ToolStartUpdate (database)")
	assert.Equal(t, "database", dbStart.Tool.FuncName())
	assert.Equal(t, "database-id-1", dbStart.ToolCallID)

	dbDone, ok := updates[8].(ToolDoneUpdate) // Was updates[4]
	require.True(t, ok, "Update 8 should be ToolDoneUpdate (database)")
	assert.Equal(t, "database", dbDone.Tool.FuncName())
	assert.Equal(t, dbStart.ToolCallID, dbDone.ToolCallID)
	require.NoError(t, dbDone.Result.Error())
	resultJSON = extractJSONFromResult(t, dbDone.Result)
	assert.JSONEq(t, `{"rows":10}`, string(resultJSON))

	finalText, ok := updates[9].(TextUpdate) // Was updates[5]
	require.True(t, ok, "Update 9 should be TextUpdate (final)")
	assert.Equal(t, "I've processed the results from the tool.", finalText.Text)

	// Assert: External tool handler received correct parameters (verified via externalToolCalls map)
	assert.Contains(t, externalToolCalls, "search", "Search tool handler should have been called")
	assert.Contains(t, externalToolCalls, "database", "Database tool handler should have been called")
	// The JSONEq assertions for params are now inside the handler

	// Assert: Provider receives full toolbox (this was checked implicitly by Generate call in runTestChat)
	assert.True(t, prov.generateCalled, "Provider's Generate method should be called")
	assert.Equal(t, 4, prov.toolboxToolsCount, "All 4 tools should be available to the provider in the first Generate call")
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
			return tools.Success(map[string]any{"response": p.TestParam})
		})

	// Act: Add the tool
	llm.AddTool(simpleTool)

	// Assert: Toolbox created and tool added
	assert.NotNil(t, llm.toolbox, "Toolbox should be created")
	require.Equal(t, 1, len(llm.toolbox.All()), "Toolbox should have 1 tool")
	assert.Equal(t, "simple_tool", llm.toolbox.All()[0].FuncName(), "Tool should be added correctly")
}
