package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

// TestOpenAIE2E tests the request generation logic of the Generate function
// by using a mock HTTP server, verifying request payloads and headers.
func TestOpenAIE2E(t *testing.T) {
	const mockAPIKey = "test-api-key"
	const mockModel = "gpt-4-test"
	const mockCompany = "OpenAI" // Default, can be overridden by WithEndpoint

	// --- Tool Definition for Toolbox Case ---
	type searchParams struct {
		Query string `json:"query" description:"Search query"`
	}
	webSearchTool := tools.Func(
		"Web Search",       // Label (Human readable)
		"Searches the web", // Description
		"web_search",       // FuncName (API level)
		func(r tools.Runner, p searchParams) tools.Result {
			// Mock handler - not actually called in this test, but needed for definition
			return tools.Success(nil)
		},
	)
	// --- End Tool Definition ---

	testCases := []struct {
		name                  string
		systemPrompt          content.Content
		messages              []llms.Message
		toolbox               *tools.Toolbox
		jsonOutputSchema      *tools.ValueSchema
		maxCompletionTokens   int
		reasoningEffort       Effort
		disableStreamOptions  bool
		customEndpoint        string // For testing WithEndpoint
		customEndpointCompany string // For testing WithEndpoint
		// verifyRequest is called after the server handler has sent its response.
		verifyRequest func(t *testing.T, headers http.Header, body map[string]any)
		// customResponse allows custom SSE responses from the mock server.
		// If nil, a default minimal SSE response is sent.
		customResponse func(t *testing.T, w http.ResponseWriter)
		// expectedStreamStatuses verifies the stream status sequence.
		expectedStreamStatuses []llms.StreamStatus
		// verifyStreamOutput verifies the stream output.
		verifyStreamOutput func(t *testing.T, collectedStatuses []llms.StreamStatus, finalToolCall llms.ToolCall, finalText string, stream llms.ProviderStream)
	}{
		{
			name: "Basic text generation",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Hello")},
			},
			maxCompletionTokens: 50,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				assert.Equal(t, mockModel, body["model"])
				messages, ok := body["messages"].([]interface{})
				require.True(t, ok)
				require.Len(t, messages, 1)

				userMsg, ok := messages[0].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "user", userMsg["role"])
				verifyOpenAIContent(t, content.FromText("Hello"), userMsg["content"])

				assert.True(t, body["stream"].(bool))
				streamOpts, ok := body["stream_options"].(map[string]interface{})
				require.True(t, ok)
				assert.True(t, streamOpts["include_usage"].(bool))
				assert.Equal(t, float64(50), body["max_completion_tokens"])
				assert.Nil(t, body["tools"], "Tools should be absent")
				assert.Nil(t, body["response_format"], "Response format should be absent")
			},
			customResponse: func(t *testing.T, w http.ResponseWriter) {
				textFrag1 := "This is "
				textFrag2 := "a streamed "
				textFrag3 := "response."
				responses := []string{
					fmt.Sprintf(`data: {"id": "chatcmpl-text-1", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}`, mockModel),
					fmt.Sprintf(`data: {"id": "chatcmpl-text-2", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"content": "%s"}, "finish_reason": null}]}`, mockModel, escapeJSONStringValue(textFrag1)),
					fmt.Sprintf(`data: {"id": "chatcmpl-text-3", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"content": "%s"}, "finish_reason": null}]}`, mockModel, escapeJSONStringValue(textFrag2)),
					fmt.Sprintf(`data: {"id": "chatcmpl-text-4", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"content": "%s"}, "finish_reason": null}]}`, mockModel, escapeJSONStringValue(textFrag3)),
					fmt.Sprintf(`data: {"id": "chatcmpl-text-5", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}`, mockModel),
					`data: [DONE]`,
				}
				for _, res := range responses {
					_, err := fmt.Fprintln(w, res)
					require.NoError(t, err)
					if flusher, ok := w.(http.Flusher); ok {
						flusher.Flush()
					}
				}
			},
			expectedStreamStatuses: []llms.StreamStatus{
				llms.StreamStatusText,
				llms.StreamStatusText,
				llms.StreamStatusText,
			},
			verifyStreamOutput: func(t *testing.T, collectedStatuses []llms.StreamStatus, finalToolCall llms.ToolCall, finalText string, stream llms.ProviderStream) {
				assert.Equal(t, "This is a streamed response.", finalText, "Final text output mismatch")
				assert.Empty(t, finalToolCall.ID, "No tool call should be present in basic text generation")
			},
		},
		{
			name: "With stream options disabled",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Hello")},
			},
			disableStreamOptions: true,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				// stream_options should be omitted entirely
				assert.Nil(t, body["stream_options"]) 
			},
		},
		{
			name:         "With system prompt",
			systemPrompt: content.FromText("Be a helpful assistant."),
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Hi there")},
			},
			maxCompletionTokens: 100,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				assert.Equal(t, mockModel, body["model"])
				messages, ok := body["messages"].([]interface{})
				require.True(t, ok)
				require.Len(t, messages, 2, "Should have system and user messages")

				systemMsg, ok := messages[0].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "system", systemMsg["role"])
				verifyOpenAIContent(t, content.FromText("Be a helpful assistant."), systemMsg["content"])

				userMsg, ok := messages[1].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "user", userMsg["role"])
				verifyOpenAIContent(t, content.FromText("Hi there"), userMsg["content"])

				assert.Equal(t, float64(100), body["max_completion_tokens"])
			},
		},
		{
			name: "With toolbox",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Search for a fun query.")},
			},
			toolbox:             tools.Box(webSearchTool), // webSearchTool is defined above
			maxCompletionTokens: 150,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				assert.NotNil(t, body["tools"], "Tools should be present")
				toolsList, ok := body["tools"].([]interface{})
				require.True(t, ok)
				require.Len(t, toolsList, 1)
				toolDef, ok := toolsList[0].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "function", toolDef["type"])
				fn, ok := toolDef["function"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "web_search", fn["name"])
				assert.Equal(t, "Searches the web", fn["description"])
				params, ok := fn["parameters"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "object", params["type"])
				props, ok := params["properties"].(map[string]interface{})
				require.True(t, ok)
				queryProp, ok := props["query"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "string", queryProp["type"])
				assert.Equal(t, "Search query", queryProp["description"])
				required, ok := params["required"].([]interface{})
				require.True(t, ok)
				assert.Contains(t, required, "query")
				assert.Equal(t, float64(150), body["max_completion_tokens"])
			},
		},
		{
			name: "With JSON mode (schema output)",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Output some structured data.")},
			},
			jsonOutputSchema: &tools.ValueSchema{
				Type: "object",
				Properties: &map[string]tools.ValueSchema{
					"name": {Type: "string", Description: "The name"},
					"age":  {Type: "integer", Description: "The age"},
				},
				Required: []string{"name", "age"},
			},
			maxCompletionTokens: 200,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				assert.NotNil(t, body["response_format"], "Response format should be present for JSON mode")
				responseFormat, ok := body["response_format"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "json_schema", responseFormat["type"])

				jsonSchema, ok := responseFormat["json_schema"].(map[string]interface{})
				require.True(t, ok, "json_schema field missing or not a map")
				assert.Equal(t, "structured_output", jsonSchema["name"], "Default schema name mismatch")
				assert.True(t, jsonSchema["strict"].(bool), "Strict mode should be true by default")

				schemaDef, ok := jsonSchema["schema"].(map[string]interface{})
				require.True(t, ok, "schema field in json_schema missing or not a map")
				assert.Equal(t, "object", schemaDef["type"])
				props, ok := schemaDef["properties"].(map[string]interface{})
				require.True(t, ok)
				nameProp, ok := props["name"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "string", nameProp["type"])
				assert.Equal(t, "The name", nameProp["description"])
				ageProp, ok := props["age"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "integer", ageProp["type"])
				assert.Equal(t, "The age", ageProp["description"])
				required, ok := schemaDef["required"].([]interface{})
				require.True(t, ok)
				assert.Contains(t, required, "name")
				assert.Contains(t, required, "age")
				assert.Equal(t, float64(200), body["max_completion_tokens"])
			},
		},
		{
			name: "Assistant message with tool_call in history",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("User query leading to tool use")},
				{
					Role:    "assistant",
					Content: content.Content{}, // Assistant might have no text part, just tool calls
					ToolCalls: []llms.ToolCall{
						{
							ID:        "call_123",
							Name:      "web_search",
							Arguments: json.RawMessage(`{"query": "openai tool call format"}`),
						},
					},
				},
				{
					Role:       "tool",
					ToolCallID: "call_123",
					Content:    content.FromText("Search results about OpenAI tool calls."),
				},
			},
			toolbox:             tools.Box(webSearchTool), // Toolbox is needed for the provider to know about web_search
			maxCompletionTokens: 75,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				assert.Equal(t, mockModel, body["model"])
				messages, ok := body["messages"].([]interface{})
				require.True(t, ok, "messages should be an array")
				require.Len(t, messages, 3, "Should have user, assistant, and tool messages")

				// User Message (Index 0)
				userMsg, ok := messages[0].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "user", userMsg["role"])
				verifyOpenAIContent(t, content.FromText("User query leading to tool use"), userMsg["content"])

				// Assistant Message with Tool Call (Index 1)
				assistantMsg, ok := messages[1].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "assistant", assistantMsg["role"])
				// Assistant content can be nil or an empty string if only tool_calls are present.
				// verifyOpenAIContent handles nil/empty string for empty content.Content{}
				verifyOpenAIContent(t, content.Content{}, assistantMsg["content"])

				require.NotNil(t, assistantMsg["tool_calls"], "Assistant message should have tool_calls")
				toolCalls, ok := assistantMsg["tool_calls"].([]interface{})
				require.True(t, ok)
				require.Len(t, toolCalls, 1)
				actualToolCall, ok := toolCalls[0].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "call_123", actualToolCall["id"])
				assert.Equal(t, "function", actualToolCall["type"])
				fnCall, ok := actualToolCall["function"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "web_search", fnCall["name"])
				assert.JSONEq(t, `{"query": "openai tool call format"}`, fnCall["arguments"].(string))

				// Tool Message (Index 2)
				toolMsg, ok := messages[2].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "tool", toolMsg["role"])
				assert.Equal(t, "call_123", toolMsg["tool_call_id"])
				verifyOpenAIContent(t, content.FromText("Search results about OpenAI tool calls."), toolMsg["content"])

				assert.Equal(t, float64(75), body["max_completion_tokens"])
				assert.NotNil(t, body["tools"], "Top-level 'tools' definition should be present")
			},
		},
		{
			name: "With reasoning effort",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Think hard about this.")},
			},
			reasoningEffort: EffortHigh,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				assert.Equal(t, string(EffortHigh), body["reasoning_effort"].(string))
			},
		},
		{
			name: "With custom endpoint",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Testing custom endpoint.")},
			},
			customEndpointCompany: "CustomLLMProvider",
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				// The main verification is that the client was configured to use this endpoint,
				// and the mock server (which client.endpoint was redirected to) received the call.
				// The `client.Company()` method would also reflect `CustomLLMProvider`.
				assert.Equal(t, mockModel, body["model"]) // Model is still the one set in New()
			},
		},
		{
			name: "Complex Stream: Sequential Tool Calls and Interspersed Text",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Perform two web searches with commentary.")},
			},
			toolbox:             tools.Box(webSearchTool),
			maxCompletionTokens: 350,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				assert.NotNil(t, body["tools"], "Tools should be present")
				toolsList, ok := body["tools"].([]interface{})
				require.True(t, ok)
				assert.Len(t, toolsList, 1)
				toolDef, ok := toolsList[0].(map[string]interface{})
				require.True(t, ok)
				functionDef, ok := toolDef["function"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "web_search", functionDef["name"])
			},
			customResponse: func(t *testing.T, w http.ResponseWriter) {
				arg1Frag1 := `{"query":"first`
				arg1Frag2 := `_query_openai"}`
				text1 := "Okay, I've done the first search."
				arg2Frag1 := `{"query":"second_`
				arg2Frag2 := `query_openai"}`
				text2 := "And I've completed the second search as well."

				responses := []string{
					fmt.Sprintf(`data: {"id": "chatcmpl-test-multi-1", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}`, mockModel),
					fmt.Sprintf(`data: {"id": "chatcmpl-test-multi-2", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "id": "tool_call_id_001", "type": "function", "function": {"name": "web_search", "arguments": "%s"}}]}, "finish_reason": null}]}`, mockModel, escapeJSONStringValue(arg1Frag1)),
					fmt.Sprintf(`data: {"id": "chatcmpl-test-multi-3", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 0, "function": {"arguments": "%s"}}]}, "finish_reason": null}]}`, mockModel, escapeJSONStringValue(arg1Frag2)),
					fmt.Sprintf(`data: {"id": "chatcmpl-test-multi-4", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}`, mockModel),
					fmt.Sprintf(`data: {"id": "chatcmpl-test-multi-5", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"content": "%s"}, "finish_reason": null}]}`, mockModel, escapeJSONStringValue(text1)),
					fmt.Sprintf(`data: {"id": "chatcmpl-test-multi-6", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "id": "tool_call_id_002", "type": "function", "function": {"name": "web_search", "arguments": "%s"}}]}, "finish_reason": null}]}`, mockModel, escapeJSONStringValue(arg2Frag1)),
					fmt.Sprintf(`data: {"id": "chatcmpl-test-multi-7", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"tool_calls": [{"index": 1, "function": {"arguments": "%s"}}]}, "finish_reason": null}]}`, mockModel, escapeJSONStringValue(arg2Frag2)),
					fmt.Sprintf(`data: {"id": "chatcmpl-test-multi-8", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]}`, mockModel),
					fmt.Sprintf(`data: {"id": "chatcmpl-test-multi-9", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"content": "%s"}, "finish_reason": null}]}`, mockModel, escapeJSONStringValue(text2)),
					fmt.Sprintf(`data: {"id": "chatcmpl-test-multi-10", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 20, "completion_tokens": 80, "total_tokens": 100}}`, mockModel),
					`data: [DONE]`,
				}
				for _, res := range responses {
					_, err := fmt.Fprintln(w, res)
					require.NoError(t, err)
					if flusher, ok := w.(http.Flusher); ok {
						flusher.Flush()
					}
				}
			},
			expectedStreamStatuses: []llms.StreamStatus{
				llms.StreamStatusToolCallBegin,
				llms.StreamStatusToolCallDelta,
				llms.StreamStatusToolCallReady,
				llms.StreamStatusText,
				llms.StreamStatusToolCallBegin,
				llms.StreamStatusToolCallDelta,
				llms.StreamStatusToolCallReady,
				llms.StreamStatusText,
			},
			verifyStreamOutput: func(t *testing.T, collectedStatuses []llms.StreamStatus, finalToolCall llms.ToolCall, finalText string, stream llms.ProviderStream) {
				fullMessage := stream.Message()
				require.Len(t, fullMessage.ToolCalls, 2, "Should have two tool calls in the final message")

				assert.Equal(t, "tool_call_id_001", fullMessage.ToolCalls[0].ID, "Tool Call 1 ID mismatch")
				assert.Equal(t, "web_search", fullMessage.ToolCalls[0].Name, "Tool Call 1 Name mismatch")
				expectedArgs1 := `{"query":"first_query_openai"}`
				assert.JSONEq(t, expectedArgs1, string(fullMessage.ToolCalls[0].Arguments), "Tool Call 1 arguments mismatch")

				assert.Equal(t, "tool_call_id_002", fullMessage.ToolCalls[1].ID, "Tool Call 2 ID mismatch")
				assert.Equal(t, "web_search", fullMessage.ToolCalls[1].Name, "Tool Call 2 Name mismatch")
				expectedArgs2 := `{"query":"second_query_openai"}`
				assert.JSONEq(t, expectedArgs2, string(fullMessage.ToolCalls[1].Arguments), "Tool Call 2 arguments mismatch")

				expectedFullText := "Okay, I've done the first search.And I've completed the second search as well."
				assert.Equal(t, expectedFullText, finalText, "Final accumulated text mismatch")
			},
		},
		{
			name: "JSON Mode Streaming Output",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Output complex JSON data with streaming.")},
			},
			jsonOutputSchema: &tools.ValueSchema{
				Type: "object",
				Properties: &map[string]tools.ValueSchema{
					"name":      {Type: "string", Description: "Person's name"},
					"age":       {Type: "integer", Description: "Person's age"},
					"isStudent": {Type: "boolean", Description: "Is the person a student"},
					"courses": {
						Type:  "array",
						Items: &tools.ValueSchema{Type: "string"},
					},
				},
				Required: []string{"name", "age", "isStudent"},
			},
			maxCompletionTokens: 300,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				assert.NotNil(t, body["response_format"], "Response format should be present for JSON mode")
				responseFormat, ok := body["response_format"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "json_schema", responseFormat["type"])
				jsonSchema, ok := responseFormat["json_schema"].(map[string]interface{})
				require.True(t, ok, "json_schema field missing or not a map")
				assert.Equal(t, "structured_output", jsonSchema["name"], "Default schema name mismatch")
				schemaDef, ok := jsonSchema["schema"].(map[string]interface{})
				require.True(t, ok, "schema field in json_schema missing or not a map")
				assert.Equal(t, "object", schemaDef["type"])
			},
			customResponse: func(t *testing.T, w http.ResponseWriter) {
				jsonFrag1 := `{"name":"Alice", "age":30,`
				jsonFrag2 := ` "isStudent":true, "courses":["Math",`
				jsonFrag3 := ` "Physics"]}`

				responses := []string{
					fmt.Sprintf(`data: {"id": "chatcmpl-json-1", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}`, mockModel),
					fmt.Sprintf(`data: {"id": "chatcmpl-json-2", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"content": "%s"}, "finish_reason": null}]}`, mockModel, escapeJSONStringValue(jsonFrag1)),
					fmt.Sprintf(`data: {"id": "chatcmpl-json-3", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"content": "%s"}, "finish_reason": null}]}`, mockModel, escapeJSONStringValue(jsonFrag2)),
					fmt.Sprintf(`data: {"id": "chatcmpl-json-4", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {"content": "%s"}, "finish_reason": null}]}`, mockModel, escapeJSONStringValue(jsonFrag3)),
					fmt.Sprintf(`data: {"id": "chatcmpl-json-5", "object": "chat.completion.chunk", "created": 1677652288, "model": "%s", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 50, "total_tokens": 60}}`, mockModel),
					`data: [DONE]`,
				}
				for _, res := range responses {
					_, err := fmt.Fprintln(w, res)
					require.NoError(t, err)
					if flusher, ok := w.(http.Flusher); ok {
						flusher.Flush()
					}
				}
			},
			expectedStreamStatuses: []llms.StreamStatus{
				llms.StreamStatusText,
				llms.StreamStatusText,
				llms.StreamStatusText,
			},
			verifyStreamOutput: func(t *testing.T, collectedStatuses []llms.StreamStatus, finalToolCall llms.ToolCall, finalText string, stream llms.ProviderStream) {
				expectedJSON := `{"name":"Alice", "age":30, "isStudent":true, "courses":["Math", "Physics"]}`
				assert.JSONEq(t, expectedJSON, finalText, "Assembled JSON output mismatch")
				assert.Empty(t, finalToolCall.ID, "No tool call should be present in JSON mode stream output")
			},
		},
	}

	for _, tc := range testCases {
		tc := tc // Capture range variable
		t.Run(tc.name, func(t *testing.T) {
			var capturedHeaders http.Header
			var capturedBody map[string]any
			requestHandled := make(chan struct{})
			verificationRan := false

			mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				defer close(requestHandled) // Signal that handler has finished

				// Capture request details
				capturedHeaders = r.Header.Clone()
				bodyBytes, err := io.ReadAll(r.Body)
				if err != nil {
					t.Errorf("Failed to read request body in mock server: %v", err)
					http.Error(w, "Failed to read request body", http.StatusInternalServerError)
					return
				}
				var tempBody map[string]any
				if err := json.Unmarshal(bodyBytes, &tempBody); err != nil {
					// Log the body for debugging if unmarshal fails
					t.Logf("Request body: %s", string(bodyBytes))
					t.Errorf("Failed to unmarshal request body in mock server: %s", err)
					http.Error(w, "Failed to unmarshal request body", http.StatusBadRequest)
					return
				}
				capturedBody = tempBody

				// Standard headers
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)

				// Send mock response
				if tc.customResponse != nil {
					tc.customResponse(t, w)
				} else {
					// Default minimal valid SSE response
					if _, err := fmt.Fprintln(w, `data: {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1677652288, "model": "`+mockModel+`", "choices": [{"index": 0, "delta": {"role": "assistant"}, "finish_reason": null}]}`); err != nil {
						t.Logf("Error writing mock response (message_start): %v", err)
					}
					if _, err := fmt.Fprintln(w, `data: {"id": "chatcmpl-test", "object": "chat.completion.chunk", "created": 1677652288, "model": "`+mockModel+`", "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}], "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}}`); err != nil {
						t.Logf("Error writing mock response (message_stop/usage): %v", err)
					}
					if _, err := fmt.Fprintln(w, `data: [DONE]`); err != nil {
						t.Logf("Error writing mock response (DONE): %v", err)
					}
				}
			}))
			defer mockServer.Close()

			client := NewChatCompletionsAPI(mockAPIKey, mockModel)

			// Special handling for the customEndpoint test case to use the mockServer.URL
			if tc.name == "With custom endpoint" {
				tc.customEndpoint = mockServer.URL // Ensure it hits our mock server
			}

			// Determine the endpoint and company for the client
			endpointToUse := mockServer.URL // Default to the mock server for all tests
			companyToUse := mockCompany     // Default company

			if tc.customEndpoint != "" {
				endpointToUse = tc.customEndpoint
				if tc.customEndpointCompany != "" {
					companyToUse = tc.customEndpointCompany
				} else {
					companyToUse = tc.customEndpointCompany // if empty, it's empty
				}
			}
			client = client.WithEndpoint(endpointToUse, companyToUse)

			// Verify company if it was custom set for this test case
			if tc.name == "With custom endpoint" && tc.customEndpointCompany != "" {
				assert.Equal(t, tc.customEndpointCompany, client.Company(), "Client company not updated by WithEndpoint")
			}

			// Apply other configurations
			if tc.maxCompletionTokens > 0 {
				client = client.WithMaxCompletionTokens(tc.maxCompletionTokens)
			}
			if tc.reasoningEffort != "" {
				client = client.WithThinking(tc.reasoningEffort)
			}
		if tc.disableStreamOptions {
			client = client.WithStreamOptionsDisabled(true)
		}

			stream := client.Generate(context.Background(), tc.systemPrompt, tc.messages, tc.toolbox, tc.jsonOutputSchema)

			var collectedStatuses []llms.StreamStatus
			var finalToolCall llms.ToolCall
			var finalTextBuilder strings.Builder

			for status := range stream.Iter() {
				if tc.verifyStreamOutput != nil || len(tc.expectedStreamStatuses) > 0 {
					collectedStatuses = append(collectedStatuses, status)
					currentCall := stream.ToolCall()
					switch status {
					case llms.StreamStatusToolCallBegin:
						finalToolCall.ID = currentCall.ID
						finalToolCall.Name = currentCall.Name
						// Arguments will be accumulated by the stream provider
						// and available in currentCall.Arguments or fully
						// available when StreamStatusToolCallReady is emitted.
					case llms.StreamStatusToolCallDelta:
						// Stream provider handles argument accumulation.
						// finalToolCall.Arguments will be updated implicitly
						// when currentCall is assigned at Ready.
					case llms.StreamStatusToolCallReady:
						finalToolCall = currentCall // Capture the fully formed tool call
					case llms.StreamStatusText:
						finalTextBuilder.WriteString(stream.Text())
					}
				}
			}

			finalText := finalTextBuilder.String()

			// Wait for the server handler to finish processing the request and sending the response
			<-requestHandled

			require.NoError(t, stream.Err(), "Stream iteration failed")

			// Perform request verification now that the server has responded
			if tc.verifyRequest != nil {
				// Common header checks
				authHeader := capturedHeaders.Get("Authorization")
				assert.Equal(t, "Bearer "+mockAPIKey, authHeader, "Authorization header mismatch")
				contentTypeHeader := capturedHeaders.Get("Content-Type")
				assert.Equal(t, "application/json", contentTypeHeader, "Content-Type header mismatch")

				tc.verifyRequest(t, capturedHeaders, capturedBody)
				verificationRan = true
			}

			// Ensure that if verification logic was defined, it actually ran.
			if tc.verifyRequest != nil {
				assert.True(t, verificationRan, "Request verification logic was defined but did not run.")
			}

			// Perform stream output verification if defined
			if len(tc.expectedStreamStatuses) > 0 {
				assert.Equal(t, tc.expectedStreamStatuses, collectedStatuses, "Stream status sequence mismatch")
			}
			if tc.verifyStreamOutput != nil {
				tc.verifyStreamOutput(t, collectedStatuses, finalToolCall, finalText, stream)
			}
		})
	}
}

// Helper to convert content.Content to the expected OpenAI message content format for verification
func verifyOpenAIContent(t *testing.T, expectedContent content.Content, actualContentAny any) {
	if len(expectedContent) == 0 {
		if actualContentAny == nil { // OpenAI might omit content if it's truly empty and no tool calls
			return
		}
		// If not nil, it might be an empty string for single text, or empty array for multi-part
		if actualStr, ok := actualContentAny.(string); ok {
			assert.Empty(t, actualStr, "Expected empty content, but got non-empty string")
			return
		}
		if actualArr, ok := actualContentAny.([]interface{}); ok {
			assert.Empty(t, actualArr, "Expected empty content, but got non-empty array")
			return
		}
		// Fall through to detailed check if it's not obviously empty in a recognized format
	}

	// OpenAI sends single text content as a string, multi-part or image as an array.
	if len(expectedContent) == 1 {
		item := expectedContent[0]
		if textItem, ok := item.(*content.Text); ok {
			actualText, isStr := actualContentAny.(string)
			require.True(t, isStr, "Expected single text content to be a string, got %T", actualContentAny)
			assert.Equal(t, textItem.Text, actualText)
			return
		}
		// If not a single text item, it falls into the array representation
	}

	actualContentList, ok := actualContentAny.([]interface{})
	require.True(t, ok, "Expected content to be an array of parts, got %T", actualContentAny)
	require.Len(t, actualContentList, len(expectedContent), "Content parts count mismatch")

	for i, expectedItem := range expectedContent {
		actualPart, ok := actualContentList[i].(map[string]interface{})
		require.True(t, ok, "Actual content part %d is not a map", i)

		typeVal, ok := actualPart["type"].(string)
		require.True(t, ok, "Content part %d 'type' is not a string", i)

		switch v := expectedItem.(type) {
		case *content.Text:
			assert.Equal(t, "text", typeVal, "Part %d type mismatch for Text", i)
			textVal, ok := actualPart["text"].(string)
			require.True(t, ok, "Part %d 'text' is not a string for Text", i)
			assert.Equal(t, v.Text, textVal, "Part %d text content mismatch", i)
		case *content.ImageURL:
			assert.Equal(t, "image_url", typeVal, "Part %d type mismatch for ImageURL", i)
			imgURLVal, ok := actualPart["image_url"].(map[string]interface{})
			require.True(t, ok, "Part %d 'image_url' is not a map for ImageURL", i)
			assert.Equal(t, v.URL, imgURLVal["url"], "Part %d image URL mismatch", i)
			// OpenAI adds a default "detail": "auto"
			assert.Equal(t, "auto", imgURLVal["detail"], "Part %d image detail mismatch")
		case *content.JSON: // JSON content is serialized as text for OpenAI
			assert.Equal(t, "text", typeVal, "Part %d type mismatch for JSON (expected text)", i)
			textVal, ok := actualPart["text"].(string)
			require.True(t, ok, "Part %d 'text' is not a string for JSON content", i)
			assert.JSONEq(t, string(v.Data), textVal, "Part %d JSON content mismatch")
		default:
			t.Fatalf("Unhandled expected content item type: %T for part %d", expectedItem, i)
		}
	}
}

// escapeJSONStringValue prepares a string to be safely embedded as a JSON string value.
// It marshals the string (which adds quotes and escapes contents) and then removes the outer quotes.
func escapeJSONStringValue(s string) string {
	b, err := json.Marshal(s)
	if err != nil {
		// This should ideally not happen for a simple string.
		// Panic might be too harsh for a test helper, but indicates a serious issue.
		panic(fmt.Sprintf("escapeJSONStringValue: failed to marshal string: %v", err))
	}
	// Remove the leading and trailing quotes added by json.Marshal
	if len(b) < 2 || b[0] != '"' || b[len(b)-1] != '"' {
		// This case should not happen if json.Marshal worked correctly for a string.
		panic(fmt.Sprintf("escapeJSONStringValue: marshaled string has unexpected format: %s", string(b)))
	}
	return string(b[1 : len(b)-1])
}
