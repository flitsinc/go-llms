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

// TestAnthropicE2EGenerate tests the request generation logic of the Generate function
// by using a mock HTTP server, verifying request payloads and headers.
func TestAnthropicE2EGenerate(t *testing.T) {
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
		// Verification function to check the request received by the mock server
		verifyRequest func(t *testing.T, r *http.Request, body map[string]any)
	}{
		{
			name: "Basic text generation",
			messages: []llms.Message{
				{Role: "user", Content: content.FromText("Hello")},
			},
			maxTokens: 50,
			verifyRequest: func(t *testing.T, r *http.Request, body map[string]any) {
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
			verifyRequest: func(t *testing.T, r *http.Request, body map[string]any) {
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
			toolbox:   tools.Box(webSearchTool), // Use tools.Box with the defined tool
			maxTokens: 150,
			verifyRequest: func(t *testing.T, r *http.Request, body map[string]any) {
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
			verifyRequest: func(t *testing.T, r *http.Request, body map[string]any) {
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
	}

	for _, tc := range testCases {
		tc := tc // Capture range variable
		t.Run(tc.name, func(t *testing.T) {
			requestVerified := false
			mockServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
				assert.Equal(t, "POST", r.Method)
				assert.Equal(t, "/v1/messages", r.URL.Path)
				assert.Equal(t, "application/json", r.Header.Get("Content-Type"))
				assert.Equal(t, mockAPIKey, r.Header.Get("X-API-Key"))
				assert.Equal(t, "2023-06-01", r.Header.Get("anthropic-version"))

				bodyBytes, err := io.ReadAll(r.Body)
				require.NoError(t, err)
				var requestBody map[string]any
				err = json.Unmarshal(bodyBytes, &requestBody)
				require.NoError(t, err, "Failed to unmarshal request body: %s", string(bodyBytes))

				if tc.verifyRequest != nil {
					tc.verifyRequest(t, r, requestBody)
				}
				requestVerified = true // Mark that the handler ran and verified

				// Send minimal valid SSE response to prevent blocking
				w.Header().Set("Content-Type", "text/event-stream")
				w.WriteHeader(http.StatusOK)
				// Use the existing sseEvent helper if available, otherwise define locally or skip body
				if _, err := fmt.Fprintln(w, "data: {\"type\": \"message_start\", \"message\": {\"role\": \"assistant\"}}"); err != nil {
					t.Logf("Error writing response: %v", err)
				}
				if _, err := fmt.Fprintln(w, "data: {\"type\": \"message_stop\"}"); err != nil {
					t.Logf("Error writing response: %v", err)
				}

			}))
			defer mockServer.Close()

			client := New(mockAPIKey, mockModel).
				WithEndpoint(mockServer.URL+"/v1/messages", mockCompany).
				WithMaxTokens(tc.maxTokens)

			stream := client.Generate(context.Background(), tc.systemPrompt, tc.messages, tc.toolbox, tc.jsonOutputSchema)

			// Iterate to ensure the stream interaction happens and checks for client-side errors
			iter := stream.Iter()
			iter(func(status llms.StreamStatus) bool { return true }) // Just consume

			require.NoError(t, stream.Err(), "Stream iteration failed")
			assert.True(t, requestVerified, "Mock server handler did not run or verify the request")
		})
	}
}
