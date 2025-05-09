package anthropic

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestAnthropicE2E tests the request generation logic of the Generate function
// by using a mock HTTP server, verifying request payloads and headers.
func TestAnthropicE2E(t *testing.T) {
	const mockAPIKey = "test-api-key"
	const mockModel = "claude-test-1"
	const mockCompany = "MockCompany"

	// --- Tool Definition for Toolbox Case ---
	type searchParams struct {
		Query string `json:"query" description:"Search query"`
	}
	webSearchTool := tools.Func(
		"Web Search", // Label (Human readable)
		"Searches the web",
		"web_search", // FuncName (API level)
		func(r tools.Runner, p searchParams) tools.Result {
			// Mock handler - not actually called in this test, but needed for definition
			return tools.Success(nil)
		},
	)
	// --- End Tool Definition ---

	testCases := []struct {
		name             string
		systemPrompt     content.Content
		messages         []llms.Message
		toolbox          *tools.Toolbox
		jsonOutputSchema *tools.ValueSchema
		maxTokens        int
		// verifyRequest is called after the server handler has sent its response.
		verifyRequest func(t *testing.T, headers http.Header, body map[string]any)
		// customResponse allows custom SSE responses from the mock server.
		// If nil, a default minimal SSE response is sent.
		customResponse func(t *testing.T, w http.ResponseWriter)
	}{
		{
			name: "Basic text generation",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Hello")},
			},
			maxTokens: 50,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				assert.Equal(t, mockModel, body["model"])
				assert.Len(t, body["messages"].([]interface{}), 1)
				assert.True(t, body["stream"].(bool))
				assert.Equal(t, float64(50), body["max_tokens"])
				assert.Nil(t, body["system"], "System prompt should be absent")
				assert.Nil(t, body["tools"], "Tools should be absent")
				assert.Nil(t, body["tool_choice"], "Tool choice should be absent")
			},
		},
		{
			name:         "With system prompt",
			systemPrompt: content.FromText("Be helpful."),
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Hi")},
			},
			maxTokens: 100,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				assert.NotNil(t, body["system"], "System prompt should be present")
				sysPrompt, ok := body["system"].(string)
				require.True(t, ok, "System prompt should marshal as string for single text item")
				assert.Equal(t, "Be helpful.", sysPrompt)
				assert.Equal(t, float64(100), body["max_tokens"])
			},
		},
		{
			name: "With toolbox",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Search for cats")},
			},
			toolbox:   tools.Box(webSearchTool),
			maxTokens: 150,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				assert.NotNil(t, body["tools"], "Tools should be present")
				toolsList, ok := body["tools"].([]interface{})
				require.True(t, ok)
				assert.Len(t, toolsList, 1)
				toolDef, ok := toolsList[0].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "web_search", toolDef["name"])
				// Verify the schema generated from the searchParams struct
				inputSchema, ok := toolDef["input_schema"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "object", inputSchema["type"])
				props, ok := inputSchema["properties"].(map[string]interface{})
				require.True(t, ok)
				queryProp, ok := props["query"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "string", queryProp["type"])
				assert.Equal(t, "Search query", queryProp["description"])
				required, ok := inputSchema["required"].([]interface{})
				require.True(t, ok)
				assert.Contains(t, required, "query")
				// Check tool choice
				assert.NotNil(t, body["tool_choice"], "Tool choice should be present")
				toolChoice, ok := body["tool_choice"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "auto", toolChoice["type"])
				assert.Equal(t, float64(150), body["max_tokens"])
			},
		},
		{
			name: "With JSON mode",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Output JSON")},
			},
			// Manually construct the ValueSchema for JSON mode
			jsonOutputSchema: &tools.ValueSchema{
				Type: "object",
				Properties: &map[string]tools.ValueSchema{
					"data": {Type: "string", Description: "The data"},
				},
				Required: []string{"data"},
			},
			maxTokens: 200,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				assert.NotNil(t, body["tools"], "Tools should be present for JSON mode")
				toolsList, ok := body["tools"].([]interface{})
				require.True(t, ok)
				assert.Len(t, toolsList, 1)
				toolDef, ok := toolsList[0].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, jsonModeToolName, toolDef["name"])
				// Verify the input schema matches the manually constructed one
				inputSchema, ok := toolDef["input_schema"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "object", inputSchema["type"])
				props, ok := inputSchema["properties"].(map[string]interface{})
				require.True(t, ok)
				dataProp, ok := props["data"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "string", dataProp["type"])
				assert.Equal(t, "The data", dataProp["description"])
				required, ok := inputSchema["required"].([]interface{})
				require.True(t, ok)
				assert.Contains(t, required, "data")
				// Check tool choice
				assert.NotNil(t, body["tool_choice"], "Tool choice should be present for JSON mode")
				toolChoice, ok := body["tool_choice"].(map[string]interface{})
				require.True(t, ok)
				assert.Equal(t, "tool", toolChoice["type"])
				assert.Equal(t, jsonModeToolName, toolChoice["name"])
				assert.Equal(t, float64(200), body["max_tokens"])
			},
		},
		{
			name: "Assistant message with only tool_call included in history",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Initial user query for web search.")},
				{ // Assistant's response - only tool call, no text content
					Role:    "assistant",
					Content: content.Content{}, // Explicitly empty content
					ToolCalls: []llms.ToolCall{
						{
							ID:        "test_tool_id_search_001",
							Name:      "web_search", // Matches webSearchTool defined above
							Arguments: json.RawMessage(`{"query": "anthropic tool use serialization"}`),
						},
					},
				},
				{ // Tool result message
					Role:       "tool",
					ToolCallID: "test_tool_id_search_001",
					Content:    content.FromText("Result from web_search about serialization."),
				},
			},
			toolbox:   tools.Box(webSearchTool), // Toolbox is needed for the provider to expect/process tools
			maxTokens: 70,
			verifyRequest: func(t *testing.T, headers http.Header, body map[string]any) {
				assert.Equal(t, mockModel, body["model"])
				messages, ok := body["messages"].([]interface{})
				require.True(t, ok, "messages should be an array")
				require.Len(t, messages, 3, "Should have user, assistant, and tool_result messages in history")

				// --- Verify User Message (Index 0) ---
				userMsg, ok := messages[0].(map[string]any)
				require.True(t, ok, "User message should be a map")
				assert.Equal(t, "user", userMsg["role"])
				var userContentText string
				if userContentStr, isStr := userMsg["content"].(string); isStr {
					userContentText = userContentStr
				} else if userContentArr, isArr := userMsg["content"].([]interface{}); isArr && len(userContentArr) > 0 {
					if textItem, itemOk := userContentArr[0].(map[string]any); itemOk {
						if textVal, textStrOk := textItem["text"].(string); textStrOk {
							userContentText = textVal
						}
					}
				}
				assert.Equal(t, "Initial user query for web search.", userContentText)

				// --- Key verification: Assistant Message (Index 1) ---
				assistantMsg, ok := messages[1].(map[string]any)
				require.True(t, ok, "Assistant message should be a map")
				assert.Equal(t, "assistant", assistantMsg["role"])
				contentList, ok := assistantMsg["content"].([]interface{})
				assert.True(t, ok, "Assistant message 'content' should be an array, not a string or null, when tool_calls are present.")

				foundToolUse := false
				// Loop is safe: if !ok, contentList is nil, loop doesn't run, foundToolUse remains false.
				for _, item := range contentList {
					contentItem, itemOk := item.(map[string]any)
					require.True(t, itemOk, "Content item should be a map")
					itemType, typeOk := contentItem["type"].(string)
					require.True(t, typeOk, "Content item should have a 'type' field")

					if itemType == "tool_use" {
						foundToolUse = true
						assert.Equal(t, "test_tool_id_search_001", contentItem["id"], "Tool call ID mismatch")
						assert.Equal(t, "web_search", contentItem["name"], "Tool call name mismatch")
						inputArgs, inputOk := contentItem["input"].(map[string]interface{})
						require.True(t, inputOk, "Tool call 'input' should be a map")
						assert.Equal(t, "anthropic tool use serialization", inputArgs["query"], "Tool call query argument mismatch")
						break
					}
				}
				assert.True(t, foundToolUse, "Assistant message 'content' MUST include a 'tool_use' block because ToolCalls were specified on the llms.Message")

				// --- Verify Tool Result Message (Index 2) ---
				toolResultMsg, ok := messages[2].(map[string]any)
				require.True(t, ok, "Tool result message should be a map")
				assert.Equal(t, "user", toolResultMsg["role"], "Tool result message role should be transformed to 'user' for Anthropic")

				toolResultContentList, ok := toolResultMsg["content"].([]interface{})
				require.True(t, ok, "Tool result 'content' should be an array")
				require.Len(t, toolResultContentList, 1, "Tool result content list should have one item")

				toolResultItem, ok := toolResultContentList[0].(map[string]any)
				require.True(t, ok, "Tool result content item should be a map")
				assert.Equal(t, "tool_result", toolResultItem["type"], "Tool result item type mismatch")
				assert.Equal(t, "test_tool_id_search_001", toolResultItem["tool_use_id"], "Tool result item tool_use_id mismatch")

				var toolResultText string
				if toolResultActualContent, contentOk := toolResultItem["content"].(string); contentOk {
					toolResultText = toolResultActualContent
				} else if toolResultActualContentArr, arrOk := toolResultItem["content"].([]interface{}); arrOk && len(toolResultActualContentArr) > 0 {
					if textItem, itemOk := toolResultActualContentArr[0].(map[string]any); itemOk {
						if textVal, textStrOk := textItem["text"].(string); textStrOk {
							toolResultText = textVal
						}
					}
				}
				assert.Equal(t, "Result from web_search about serialization.", toolResultText, "Tool result text content mismatch")

				assert.Equal(t, float64(70), body["max_tokens"])
				assert.NotNil(t, body["tools"], "Top-level 'tools' definition should be present in the request if toolbox was used")
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
					t.Errorf("Failed to unmarshal request body in mock server: %s, body: %s", err, string(bodyBytes))
					http.Error(w, "Failed to unmarshal request body", http.StatusBadRequest)
					return
				}
				capturedBody = tempBody

				// Standard headers (checked by performRequestVerification if needed, but good for server to set content type)
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)

				// Send mock response
				if tc.customResponse != nil {
					tc.customResponse(t, w)
				} else {
					// Default minimal valid SSE response
					if _, err := fmt.Fprintln(w, `data: {"type": "message_start", "message": {"role": "assistant"}}`); err != nil {
						t.Logf("Error writing mock response: %v", err)
					}
					if _, err := fmt.Fprintln(w, `data: {"type": "message_stop"}`); err != nil {
						t.Logf("Error writing mock response: %v", err)
					}
				}
			}))
			defer mockServer.Close()

			client := New(mockAPIKey, mockModel).
				WithEndpoint(mockServer.URL+"/v1/messages", mockCompany).
				WithMaxTokens(tc.maxTokens)

			stream := client.Generate(context.Background(), tc.systemPrompt, tc.messages, tc.toolbox, tc.jsonOutputSchema)

			iter := stream.Iter()
			iter(func(status llms.StreamStatus) bool { return true }) // Consume the stream

			// Wait for the server handler to finish processing the request and sending the response
			<-requestHandled

			require.NoError(t, stream.Err(), "Stream iteration failed")

			// Perform request verification now that the server has responded
			if tc.verifyRequest != nil {
				tc.verifyRequest(t, capturedHeaders, capturedBody)
				verificationRan = true
			}

			// Ensure that if verification logic was defined, it actually ran.
			if tc.verifyRequest != nil {
				assert.True(t, verificationRan, "Request verification logic was defined but did not run.")
			}
		})
	}
}
