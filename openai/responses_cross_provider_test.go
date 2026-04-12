package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

// TestCrossProviderToolCallReplay_MockServer verifies that when an assistant message
// contains tool calls from Anthropic (toolu_01* IDs, no openai:item_id metadata),
// the Responses API path generates synthetic item IDs and produces a valid request payload.
//
// This simulates the exact production scenario: the editor agent's smart mode was switched
// from claude-4-opus-medium to gpt-5.4-concise, and in-flight durable workflows had
// Claude tool call history that needed to be replayed through OpenAI.
func TestCrossProviderToolCallReplay_MockServer(t *testing.T) {
	type readModuleParams struct {
		ModuleName string `json:"module_name" description:"Name of the module to read"`
	}
	readModuleTool := tools.Func(
		"Read Module",
		"Reads a module's source code",
		"readModule",
		func(r tools.Runner, p readModuleParams) tools.Result {
			return tools.Success(nil)
		},
	)

	var capturedPayload map[string]any

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("failed to read request body: %v", err)
		}
		if err := json.Unmarshal(body, &capturedPayload); err != nil {
			t.Fatalf("failed to parse request body: %v", err)
		}

		// Send a minimal valid SSE response
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		responses := []string{
			`data: {"type":"response.created","response":{"id":"resp_test","object":"response","status":"in_progress","output":[]}}`,
			`data: {"type":"response.output_item.added","output_index":0,"item":{"type":"message","id":"msg_test","role":"assistant","content":[]}}`,
			`data: {"type":"response.content_part.added","output_index":0,"content_index":0,"part":{"type":"output_text","text":""}}`,
			`data: {"type":"response.output_text.delta","output_index":0,"content_index":0,"delta":"Done."}`,
			`data: {"type":"response.output_text.done","output_index":0,"content_index":0,"text":"Done."}`,
			`data: {"type":"response.content_part.done","output_index":0,"content_index":0,"part":{"type":"output_text","text":"Done."}}`,
			`data: {"type":"response.output_item.done","output_index":0,"item":{"type":"message","id":"msg_test","role":"assistant","content":[{"type":"output_text","text":"Done."}]}}`,
			`data: {"type":"response.completed","response":{"id":"resp_test","object":"response","status":"completed","output":[{"type":"message","id":"msg_test","role":"assistant","content":[{"type":"output_text","text":"Done."}]}],"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`,
		}
		for _, line := range responses {
			fmt.Fprintln(w, line)
		}
		if flusher, ok := w.(http.Flusher); ok {
			flusher.Flush()
		}
	}))
	defer server.Close()

	// Build conversation history that simulates the production scenario:
	// 1. User asks to edit a module
	// 2. Assistant (previously Claude) responded with a readModule tool call (toolu_01* ID, NO openai metadata)
	// 3. Tool result was returned
	// 4. Now we're sending this history to OpenAI for the next turn
	messages := []llms.Message{
		{
			Role:    "user",
			Content: content.FromText("Remove the auth requirement from Views.Studio and Components.UploadCard"),
		},
		{
			Role:    "assistant",
			Content: content.Content{&content.Text{Text: "I'll read the modules to understand the current auth setup."}},
			ToolCalls: []llms.ToolCall{
				{
					ID:        "toolu_01GRVUqdmDwoW27uDQJ378yi",
					Name:      "readModule",
					Arguments: json.RawMessage(`{"module_name":"Views.Studio"}`),
					// NOTE: No Metadata - this is the key part. Claude doesn't set openai:item_id
				},
			},
		},
		{
			Role:       "tool",
			ToolCallID: "toolu_01GRVUqdmDwoW27uDQJ378yi",
			Content:    content.FromText(`export default function Studio() { return <div>Studio</div>; }`),
		},
	}

	api := NewResponsesAPI("test-key", "gpt-5.4-concise").
		WithEndpoint(server.URL, "OpenAI").
		WithMaxOutputTokens(100)

	stream := api.Generate(
		context.Background(),
		content.FromText("You are an editor agent."),
		messages,
		tools.Box(readModuleTool),
		nil,
	)

	// Drain the stream
	for status := range stream.Iter() {
		_ = status
	}
	if stream.Err() != nil {
		t.Fatalf("stream error: %v", stream.Err())
	}

	// Verify the captured payload
	inputItems, ok := capturedPayload["input"].([]any)
	if !ok {
		t.Fatalf("expected input array in payload, got %T", capturedPayload["input"])
	}

	// Find the function_call item in the input
	var foundFunctionCall bool
	var foundFunctionCallOutput bool
	for _, item := range inputItems {
		itemMap, ok := item.(map[string]any)
		if !ok {
			continue
		}
		switch itemMap["type"] {
		case "function_call":
			foundFunctionCall = true
			// Verify the synthetic item_id was generated
			id, _ := itemMap["id"].(string)
			if id != "fc_synthetic_toolu_01GRVUqdmDwoW27uDQJ378yi" {
				t.Errorf("expected synthetic item ID 'fc_synthetic_toolu_01GRVUqdmDwoW27uDQJ378yi', got %q", id)
			}
			// Verify the original call_id is preserved
			callID, _ := itemMap["call_id"].(string)
			if callID != "toolu_01GRVUqdmDwoW27uDQJ378yi" {
				t.Errorf("expected call_id 'toolu_01GRVUqdmDwoW27uDQJ378yi', got %q", callID)
			}
			// Verify the tool name is preserved
			name, _ := itemMap["name"].(string)
			if name != "readModule" {
				t.Errorf("expected name 'readModule', got %q", name)
			}
		case "function_call_output":
			foundFunctionCallOutput = true
			callID, _ := itemMap["call_id"].(string)
			if callID != "toolu_01GRVUqdmDwoW27uDQJ378yi" {
				t.Errorf("expected function_call_output call_id 'toolu_01GRVUqdmDwoW27uDQJ378yi', got %q", callID)
			}
		}
	}

	if !foundFunctionCall {
		t.Error("did not find function_call item in payload — conversion failed silently")
	}
	if !foundFunctionCallOutput {
		t.Error("did not find function_call_output item in payload")
	}

	t.Logf("Payload structure verified: synthetic ID correctly generated for cross-provider tool call")
}

// TestCrossProviderToolCallReplay_LiveAPI sends a cross-provider conversation to the
// real OpenAI Responses API to verify that synthetic item IDs are accepted.
// This test requires OPENAI_API_KEY to be set and is skipped otherwise.
func TestCrossProviderToolCallReplay_LiveAPI(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY not set, skipping live API test")
	}

	type readModuleParams struct {
		ModuleName string `json:"module_name" description:"Name of the module to read"`
	}
	readModuleTool := tools.Func(
		"Read Module",
		"Reads a module's source code",
		"readModule",
		func(r tools.Runner, p readModuleParams) tools.Result {
			return tools.Success(nil)
		},
	)

	// Build the same cross-provider conversation: Claude tool calls replayed to OpenAI
	messages := []llms.Message{
		{
			Role:    "user",
			Content: content.FromText("Read the Home module."),
		},
		{
			Role:    "assistant",
			Content: content.Content{&content.Text{Text: "I'll read the Home module for you."}},
			ToolCalls: []llms.ToolCall{
				{
					ID:        "toolu_01ABC123def456",
					Name:      "readModule",
					Arguments: json.RawMessage(`{"module_name":"Views.Home"}`),
					// No openai:item_id metadata — simulating a Claude tool call
				},
			},
		},
		{
			Role:       "tool",
			ToolCallID: "toolu_01ABC123def456",
			Content:    content.FromText(`export default function Home() { return <div>Hello World</div>; }`),
		},
	}

	api := NewResponsesAPI(apiKey, "gpt-4.1-nano").
		WithMaxOutputTokens(50).
		WithStore(false)

	stream := api.Generate(
		context.Background(),
		content.FromText("You are a helpful assistant. Just say 'OK' and nothing else."),
		messages,
		tools.Box(readModuleTool),
		nil,
	)

	var gotText bool
	for status := range stream.Iter() {
		if status == llms.StreamStatusText {
			gotText = true
		}
	}

	if err := stream.Err(); err != nil {
		t.Fatalf("OpenAI Responses API rejected synthetic item IDs: %v", err)
	}

	if !gotText {
		// Even if no text came through, the fact that we didn't get an error means
		// OpenAI accepted the synthetic IDs
		t.Log("No text output, but API accepted the request without error")
	} else {
		t.Logf("OpenAI accepted synthetic item IDs and responded with: %s", stream.Text())
	}

	t.Log("PASS: OpenAI Responses API accepts synthetic item IDs for cross-provider tool call replay")
}
