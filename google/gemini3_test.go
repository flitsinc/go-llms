package google

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
)

func TestGemini3Config(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}

		genConfig, ok := payload["generationConfig"].(map[string]any)
		if !ok {
			t.Fatal("generationConfig missing or invalid")
		}

		// Check Thinking Level
		thinkingConfig, ok := genConfig["thinkingConfig"].(map[string]any)
		if !ok {
			t.Fatal("thinkingConfig missing or invalid")
		}
		if level, ok := thinkingConfig["thinkingLevel"].(string); !ok || level != string(ThinkingLevelHigh) {
			t.Errorf("expected thinkingLevel 'HIGH', got %v", thinkingConfig["thinkingLevel"])
		}
		if include, ok := thinkingConfig["includeThoughts"].(bool); !ok || !include {
			t.Errorf("expected includeThoughts true, got %v", thinkingConfig["includeThoughts"])
		}

		// Check Media Resolution
		if res, ok := genConfig["mediaResolution"].(string); !ok || res != string(MediaResolutionHigh) {
			t.Errorf("expected mediaResolution 'MEDIA_RESOLUTION_HIGH', got %v", genConfig["mediaResolution"])
		}

		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"candidates": [{"content": {"parts": [{"text": "done"}]}}]}`))
	}))
	defer server.Close()

	client := server.Client()
	model := New("gemini-3-pro-preview").
		WithGeminiAPI("fake-key").
		WithThinkingLevel(ThinkingLevelHigh).
		WithMediaResolution(MediaResolutionHigh)
	model.SetHTTPClient(client)

	// Override endpoint to point to test server
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("hello")},
	}, nil, nil)

	if err := stream.Err(); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}
}

func TestGemini3ThoughtSignature(t *testing.T) {
	// Test that thought signature is captured from stream and sent back in requests

	// 1. Test Receiving Signature
	streamResp := `data: {"candidates": [{"content": {"parts": [{"functionCall": {"name": "test_tool", "args": {}}, "thoughtSignature": "sig-123"}]}}]}`

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(streamResp))
	}))
	defer server.Close()

	client := server.Client()
	model := New("gemini-3-pro-preview").WithGeminiAPI("fake-key")
	model.SetHTTPClient(client)
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("call tool")},
	}, nil, nil)

	var toolCall llms.ToolCall
	stream.Iter()(func(status llms.StreamStatus) bool {
		if status == llms.StreamStatusToolCallReady {
			toolCall = stream.ToolCall()
		}
		return true
	})

	if toolCall.Name != "test_tool" {
		t.Errorf("Expected tool name 'test_tool', got %q", toolCall.Name)
	}
	if sig, ok := toolCall.Metadata["google:thought_signature"]; !ok || sig != "sig-123" {
		t.Errorf("Expected thought signature 'sig-123' in metadata, got %v", toolCall.Metadata)
	}

	// 2. Test Sending Signature
	// We'll create a new server to verify the request payload contains the signature
	server2 := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}

		contents := payload["contents"].([]interface{})

		// Find the model message (assistant) which contains the function call and signature
		var modelMsg map[string]interface{}
		for _, item := range contents {
			msg := item.(map[string]interface{})
			if role, ok := msg["role"].(string); ok && role == "model" {
				modelMsg = msg
				break
			}
		}

		if modelMsg == nil {
			t.Fatal("Could not find model message in payload")
		}

		lastMsg := modelMsg

		// parts can be a single object or an array due to custom unmarshaling logic in the library,
		// but here we receive what the library marshaled.
		// The library marshals parts as an array (slice) unless it's length 1 where it marshals just the object.
		// Let's check how it was marshaled.
		var toolPart map[string]interface{}
		if partsArr, ok := lastMsg["parts"].([]interface{}); ok {
			toolPart = partsArr[0].(map[string]interface{})
		} else if partsObj, ok := lastMsg["parts"].(map[string]interface{}); ok {
			toolPart = partsObj
		} else {
			t.Fatalf("parts is neither array nor object: %T", lastMsg["parts"])
		}

		if sig, ok := toolPart["thoughtSignature"].(string); !ok || sig != "sig-123" {
			t.Errorf("Expected sent thoughtSignature 'sig-123', got %v", toolPart["thoughtSignature"])
		}

		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`{"candidates": [{"content": {"parts": [{"text": "response"}]}}]}`))
	}))
	defer server2.Close()

	model.endpoint = server2.URL

	// Construct a history that includes the tool call with signature
	// Note: messagesFromLLM maps 'assistant' role to 'model' role for Google
	history := []llms.Message{
		{Role: "user", Content: content.FromText("call tool")},
		{
			Role:      "assistant",
			ToolCalls: []llms.ToolCall{toolCall}, // This toolCall has the metadata from step 1
		},
		{
			Role:       "tool",
			ToolCallID: toolCall.ID,
			Content:    content.FromText("result"),
		},
	}

	stream2 := model.Generate(ctx, nil, history, nil, nil)
	if err := stream2.Err(); err != nil {
		t.Fatalf("Generate 2 failed: %v", err)
	}
}
