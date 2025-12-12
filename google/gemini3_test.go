package google

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"reflect"
	"strings"
	"testing"

	"github.com/metalim/jsonmap"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
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
			Role:         "tool",
			ToolCallID:   toolCall.ID,
			ToolCallName: toolCall.Name,
			Content:      content.FromText("result"),
		},
	}

	stream2 := model.Generate(ctx, nil, history, nil, nil)
	if err := stream2.Err(); err != nil {
		t.Fatalf("Generate 2 failed: %v", err)
	}
}

func TestGemini3AggregatesFunctionResponses(t *testing.T) {
	toolCalls := []llms.ToolCall{
		{ID: "t1", Name: "a", Arguments: json.RawMessage(`{}`)},
		{ID: "t2", Name: "b", Arguments: json.RawMessage(`{}`)},
		{ID: "t3", Name: "c", Arguments: json.RawMessage(`{}`)},
	}
	history := []llms.Message{
		{Role: "assistant", ToolCalls: toolCalls},
		{Role: "tool", ToolCallID: "t1", ToolCallName: "a", Content: content.FromRawJSON(json.RawMessage(`{"ok":1}`))},
		{Role: "tool", ToolCallID: "t2", ToolCallName: "b", Content: content.FromRawJSON(json.RawMessage(`{"ok":2}`))},
		{Role: "tool", ToolCallID: "t3", ToolCallName: "c", Content: content.FromRawJSON(json.RawMessage(`{"ok":3}`))},
	}

	payloadCh := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		payloadCh <- payload

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"candidates": [{"content": {"parts": [{"text": "done"}]}}]}` + "\n\n"))
	}))
	defer server.Close()

	model := New("gemini-3-pro-preview").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, history, nil, nil)
	if err := stream.Err(); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	payload := <-payloadCh

	contents, ok := payload["contents"].([]any)
	if !ok {
		t.Fatalf("expected contents array, got %T", payload["contents"])
	}

	var functionMsg map[string]any
	functionCount := 0
	for _, raw := range contents {
		msg, ok := raw.(map[string]any)
		if !ok {
			continue
		}
		if role, _ := msg["role"].(string); role == "function" {
			functionMsg = msg
			functionCount++
		}
	}

	if functionCount != 1 {
		t.Fatalf("expected exactly one function message, got %d", functionCount)
	}
	if functionMsg == nil {
		t.Fatalf("function message missing")
	}

	var parts []any
	switch p := functionMsg["parts"].(type) {
	case []any:
		parts = p
	case map[string]any:
		parts = []any{p}
	default:
		t.Fatalf("unexpected parts type %T", functionMsg["parts"])
	}

	if len(parts) != 3 {
		t.Fatalf("expected 3 function response parts, got %d", len(parts))
	}

	names := map[string]bool{}
	for idx, partRaw := range parts {
		part, ok := partRaw.(map[string]any)
		if !ok {
			t.Fatalf("part %d is not an object", idx)
		}
		fr, ok := part["functionResponse"].(map[string]any)
		if !ok {
			t.Fatalf("part %d missing functionResponse", idx)
		}
		name, _ := fr["name"].(string)
		if name == "" {
			t.Fatalf("part %d missing functionResponse.name", idx)
		}
		names[name] = true
	}

	for _, expected := range []string{"a", "b", "c"} {
		if !names[expected] {
			t.Fatalf("missing functionResponse for %q", expected)
		}
	}
}

func TestStreamingToolArguments_PayloadConfig(t *testing.T) {
	// Test that WithStreamingToolArguments(true) adds the flag to the request payload

	payloadCh := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload map[string]any
		if err := json.NewDecoder(r.Body).Decode(&payload); err != nil {
			t.Fatalf("failed to decode request body: %v", err)
		}
		payloadCh <- payload

		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = w.Write([]byte(`data: {"candidates": [{"content": {"parts": [{"text": "done"}]}}]}` + "\n\n"))
	}))
	defer server.Close()

	// Build a toolbox so toolConfig is generated
	schema := tools.FunctionSchema{Name: "test_func", Description: "Test", Parameters: tools.ValueSchema{Type: "object"}}
	tb := tools.Box(
		tools.External("Test", &schema, func(r tools.Runner, params json.RawMessage) tools.Result {
			return tools.SuccessFromString("ok")
		}),
	)

	model := New("gemini-3-pro-preview").
		WithGeminiAPI("fake-key").
		WithStreamingToolArguments(true)
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("call the tool")},
	}, tb, nil)

	if err := stream.Err(); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	payload := <-payloadCh

	// Check that streamFunctionCallArguments is set in functionCallingConfig
	toolConfig, ok := payload["toolConfig"].(map[string]any)
	if !ok {
		t.Fatal("toolConfig missing from payload")
	}
	funcConfig, ok := toolConfig["functionCallingConfig"].(map[string]any)
	if !ok {
		t.Fatal("functionCallingConfig missing from toolConfig")
	}
	streamArg, ok := funcConfig["streamFunctionCallArguments"].(bool)
	if !ok || !streamArg {
		t.Errorf("expected streamFunctionCallArguments to be true, got %v", funcConfig["streamFunctionCallArguments"])
	}
}

func TestStreamingToolArguments_StableID(t *testing.T) {
	// Test that when the backend sends multiple chunks with the same functionCall.id,
	// we reuse the same tool call entry and emit Begin once, multiple Deltas

	// Simulate streaming: two chunks with the same function call ID, then finish
	streamResp := strings.Join([]string{
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"id": "fc-123", "name": "search", "args": {"q": "hel"}}}]}}]}`,
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"id": "fc-123", "name": "search", "args": {"q": "hello"}}}]}}]}`,
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"id": "fc-123", "name": "search", "args": {"q": "hello world"}}}]}, "finishReason": "STOP"}]}`,
	}, "\n")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(streamResp))
	}))
	defer server.Close()

	model := New("gemini-3-pro-preview").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("search for something")},
	}, nil, nil)

	var beginCount, readyCount int
	var deltaArgs []string
	var finalToolCall llms.ToolCall

	stream.Iter()(func(status llms.StreamStatus) bool {
		switch status {
		case llms.StreamStatusToolCallBegin:
			beginCount++
		case llms.StreamStatusToolCallDelta:
			deltaArgs = append(deltaArgs, string(stream.ToolCall().Arguments))
		case llms.StreamStatusToolCallReady:
			readyCount++
			finalToolCall = stream.ToolCall()
		}
		return true
	})

	if err := stream.Err(); err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	// Should have exactly 1 Begin, 3 Deltas (one per chunk), 1 Ready
	if beginCount != 1 {
		t.Errorf("Expected 1 ToolCallBegin, got %d", beginCount)
	}
	expectedDeltas := []string{
		`{"q": "hel"`,
		`{"q": "hello"`,
		`{"q": "hello world"`,
		`{"q": "hello world"}`,
	}
	if len(deltaArgs) != len(expectedDeltas) {
		t.Fatalf("Delta count mismatch: got %d, want %d", len(deltaArgs), len(expectedDeltas))
	}
	for i := range deltaArgs {
		if deltaArgs[i] != expectedDeltas[i] {
			t.Fatalf("Delta %d mismatch: got %q, want %q", i, deltaArgs[i], expectedDeltas[i])
		}
	}
	if readyCount != 1 {
		t.Errorf("Expected 1 ToolCallReady, got %d", readyCount)
	}

	// Final tool call should have the server-provided ID and final arguments
	if finalToolCall.ID != "fc-123" {
		t.Errorf("Expected tool call ID 'fc-123', got %q", finalToolCall.ID)
	}
	if finalToolCall.Name != "search" {
		t.Errorf("Expected tool call name 'search', got %q", finalToolCall.Name)
	}
	// Final args should be the last update
	var args map[string]string
	if err := json.Unmarshal(finalToolCall.Arguments, &args); err != nil {
		t.Fatalf("Failed to unmarshal final arguments: %v", err)
	}
	if args["q"] != "hello world" {
		t.Errorf("Expected final args q='hello world', got %q", args["q"])
	}

	// Should only have 1 tool call in the message
	msg := stream.Message()
	if len(msg.ToolCalls) != 1 {
		t.Errorf("Expected 1 tool call in message, got %d", len(msg.ToolCalls))
	}
}

func TestStreamingToolArguments_OneShotWithoutFinishReason(t *testing.T) {
	// Test that one-shot tool calls (without finishReason) still emit ToolCallReady at EOF
	// This is the backward-compatible behavior for non-streaming scenarios

	streamResp := `data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "get_time", "args": {}}}]}}]}`

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(streamResp))
	}))
	defer server.Close()

	model := New("gemini-2.0-flash").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("what time is it?")},
	}, nil, nil)

	var beginCount, readyCount int
	var deltaArgs []string
	var toolCall llms.ToolCall

	stream.Iter()(func(status llms.StreamStatus) bool {
		switch status {
		case llms.StreamStatusToolCallBegin:
			beginCount++
		case llms.StreamStatusToolCallDelta:
			deltaArgs = append(deltaArgs, string(stream.ToolCall().Arguments))
		case llms.StreamStatusToolCallReady:
			readyCount++
			toolCall = stream.ToolCall()
		}
		return true
	})

	if err := stream.Err(); err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	// Should emit Begin, Delta, Ready even without finishReason
	if beginCount != 1 {
		t.Errorf("Expected 1 ToolCallBegin, got %d", beginCount)
	}
	expectedDeltas := []string{
		`{`,
		`{}`,
	}
	if !reflect.DeepEqual(deltaArgs, expectedDeltas) {
		t.Fatalf("Delta arguments mismatch\n got: %v\nwant: %v", deltaArgs, expectedDeltas)
	}
	if readyCount != 1 {
		t.Errorf("Expected 1 ToolCallReady, got %d", readyCount)
	}

	if toolCall.Name != "get_time" {
		t.Errorf("Expected tool call name 'get_time', got %q", toolCall.Name)
	}
}

func TestStreamingToolArguments_MultipleToolCalls(t *testing.T) {
	// Test handling multiple different tool calls in the same response

	streamResp := strings.Join([]string{
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"id": "fc-1", "name": "get_weather", "args": {"city": "NYC"}}}]}}]}`,
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"id": "fc-2", "name": "get_time", "args": {"tz": "EST"}}}]}}]}`,
		`data: {"candidates": [{"content": {"role": "model", "parts": []}, "finishReason": "STOP"}]}`,
	}, "\n")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(streamResp))
	}))
	defer server.Close()

	model := New("gemini-3-pro-preview").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("weather and time please")},
	}, nil, nil)

	var beginCount, readyCount int

	stream.Iter()(func(status llms.StreamStatus) bool {
		switch status {
		case llms.StreamStatusToolCallBegin:
			beginCount++
		case llms.StreamStatusToolCallReady:
			readyCount++
		}
		return true
	})

	if err := stream.Err(); err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	// Should have 2 Begins (one per tool call)
	if beginCount != 2 {
		t.Errorf("Expected 2 ToolCallBegin, got %d", beginCount)
	}

	// Should have 2 Ready (emitted at EOF for pending tool calls)
	if readyCount != 2 {
		t.Errorf("Expected 2 ToolCallReady, got %d", readyCount)
	}

	// Should have 2 tool calls in the message
	msg := stream.Message()
	if len(msg.ToolCalls) != 2 {
		t.Fatalf("Expected 2 tool calls in message, got %d", len(msg.ToolCalls))
	}

	// Verify both tool calls are present with correct IDs
	ids := make(map[string]bool)
	names := make(map[string]bool)
	for _, tc := range msg.ToolCalls {
		ids[tc.ID] = true
		names[tc.Name] = true
	}
	if !ids["fc-1"] || !ids["fc-2"] {
		t.Errorf("Expected tool call IDs fc-1 and fc-2, got %v", ids)
	}
	if !names["get_weather"] || !names["get_time"] {
		t.Errorf("Expected tool names get_weather and get_time, got %v", names)
	}
}

func TestApplyPartialArgsJSON(t *testing.T) {
	// Test the applyPartialArgsJSON helper function
	// Updated to use pointer-based types per official Google docs

	strPtr := func(s string) *string { return &s }
	numPtr := func(n float64) *float64 { return &n }
	boolPtr := func(b bool) *bool { return &b }

	t.Run("Append to existing string field", func(t *testing.T) {
		current := json.RawMessage(`{"content": "Hello "}`)
		patches := []partialFunctionArgument{
			{JSONPath: "$.content", StringValue: strPtr("World")},
		}

		result, err := applyPartialArgsJSON(current, patches)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		var parsed map[string]string
		if err := json.Unmarshal(result, &parsed); err != nil {
			t.Fatalf("Failed to parse result: %v", err)
		}
		if parsed["content"] != "Hello World" {
			t.Errorf("Expected 'Hello World', got %q", parsed["content"])
		}
	})

	t.Run("Set new field", func(t *testing.T) {
		current := json.RawMessage(`{}`)
		patches := []partialFunctionArgument{
			{JSONPath: ".title", StringValue: strPtr("New Title")},
		}

		result, err := applyPartialArgsJSON(current, patches)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		var parsed map[string]string
		if err := json.Unmarshal(result, &parsed); err != nil {
			t.Fatalf("Failed to parse result: %v", err)
		}
		if parsed["title"] != "New Title" {
			t.Errorf("Expected 'New Title', got %q", parsed["title"])
		}
	})

	t.Run("Multiple patches with string append", func(t *testing.T) {
		current := json.RawMessage(`{"a": "1"}`)
		patches := []partialFunctionArgument{
			{JSONPath: "a", StringValue: strPtr("2")},
			{JSONPath: "b", StringValue: strPtr("3")},
		}

		result, err := applyPartialArgsJSON(current, patches)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		var parsed map[string]string
		if err := json.Unmarshal(result, &parsed); err != nil {
			t.Fatalf("Failed to parse result: %v", err)
		}
		if parsed["a"] != "12" {
			t.Errorf("Expected 'a' to be '12' (appended), got %q", parsed["a"])
		}
		if parsed["b"] != "3" {
			t.Errorf("Expected 'b' to be '3', got %q", parsed["b"])
		}
	})

	t.Run("Number value", func(t *testing.T) {
		current := json.RawMessage(`{}`)
		patches := []partialFunctionArgument{
			{JSONPath: "$.brightness", NumberValue: numPtr(50)},
		}

		result, err := applyPartialArgsJSON(current, patches)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		var parsed map[string]float64
		if err := json.Unmarshal(result, &parsed); err != nil {
			t.Fatalf("Failed to parse result: %v", err)
		}
		if parsed["brightness"] != 50 {
			t.Errorf("Expected brightness=50, got %v", parsed["brightness"])
		}
	})

	t.Run("Bool value", func(t *testing.T) {
		current := json.RawMessage(`{}`)
		patches := []partialFunctionArgument{
			{JSONPath: "enabled", BoolValue: boolPtr(true)},
		}

		result, err := applyPartialArgsJSON(current, patches)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		var parsed map[string]bool
		if err := json.Unmarshal(result, &parsed); err != nil {
			t.Fatalf("Failed to parse result: %v", err)
		}
		if !parsed["enabled"] {
			t.Error("Expected enabled=true")
		}
	})

	t.Run("Null value", func(t *testing.T) {
		current := json.RawMessage(`{"field": "old"}`)
		patches := []partialFunctionArgument{
			{JSONPath: "field", NullValue: boolPtr(true)},
		}

		result, err := applyPartialArgsJSON(current, patches)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		var parsed map[string]any
		if err := json.Unmarshal(result, &parsed); err != nil {
			t.Fatalf("Failed to parse result: %v", err)
		}
		if parsed["field"] != nil {
			t.Errorf("Expected field=null, got %v", parsed["field"])
		}
	})

	t.Run("Empty patches returns original", func(t *testing.T) {
		current := json.RawMessage(`{"key": "value"}`)

		result, err := applyPartialArgsJSON(current, nil)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		if string(result) != string(current) {
			t.Errorf("Expected unchanged result, got %s", string(result))
		}
	})

	t.Run("Ignores complex JSON paths", func(t *testing.T) {
		current := json.RawMessage(`{"simple": "value"}`)
		patches := []partialFunctionArgument{
			{JSONPath: "$.nested.path", StringValue: strPtr("ignored")},
			{JSONPath: "$.items[0]", StringValue: strPtr("also ignored")},
		}

		result, err := applyPartialArgsJSON(current, patches)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		var parsed map[string]any
		if err := json.Unmarshal(result, &parsed); err != nil {
			t.Fatalf("Failed to parse result: %v", err)
		}
		// Complex paths should be ignored, original value preserved
		if parsed["simple"] != "value" {
			t.Errorf("Expected 'simple' to be 'value', got %v", parsed["simple"])
		}
		// No new fields should be added for complex paths
		if _, ok := parsed["nested"]; ok {
			t.Error("Did not expect 'nested' field to be added")
		}
	})

	t.Run("Patch with no value is skipped", func(t *testing.T) {
		// Per docs, a patch with only jsonPath (no value) can signal end of string streaming
		current := json.RawMessage(`{"location": "New Delhi"}`)
		patches := []partialFunctionArgument{
			{JSONPath: "$.location"}, // No value - should be skipped
		}

		result, err := applyPartialArgsJSON(current, patches)
		if err != nil {
			t.Fatalf("Unexpected error: %v", err)
		}

		var parsed map[string]string
		if err := json.Unmarshal(result, &parsed); err != nil {
			t.Fatalf("Failed to parse result: %v", err)
		}
		// Original value should be preserved
		if parsed["location"] != "New Delhi" {
			t.Errorf("Expected location='New Delhi', got %q", parsed["location"])
		}
	})
}

func TestStreamingToolArguments_EmptyFunctionCallEndsCall(t *testing.T) {
	// Per official Google docs, an empty functionCall {} signals end of current function call

	streamResp := strings.Join([]string{
		// First function call starts
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "get_weather"}}]}}]}`,
		// Partial args for first call
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"partialArgs": [{"jsonPath": "$.location", "stringValue": "NYC"}]}}]}}]}`,
		// Empty functionCall signals end of first call
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {}}]}}]}`,
	}, "\n")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(streamResp))
	}))
	defer server.Close()

	model := New("gemini-3-pro-preview").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("weather please")},
	}, nil, nil)

	var beginCount, deltaCount, readyCount int
	var finalToolCall llms.ToolCall

	stream.Iter()(func(status llms.StreamStatus) bool {
		switch status {
		case llms.StreamStatusToolCallBegin:
			beginCount++
		case llms.StreamStatusToolCallDelta:
			deltaCount++
		case llms.StreamStatusToolCallReady:
			readyCount++
			finalToolCall = stream.ToolCall()
		}
		return true
	})

	if err := stream.Err(); err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	// Should have 1 Begin, 1 Delta (for the partial args), 1 Ready (from empty functionCall)
	if beginCount != 1 {
		t.Errorf("Expected 1 ToolCallBegin, got %d", beginCount)
	}
	if deltaCount != 2 {
		t.Errorf("Expected 2 ToolCallDelta, got %d", deltaCount)
	}
	if readyCount != 1 {
		t.Errorf("Expected 1 ToolCallReady, got %d", readyCount)
	}

	if finalToolCall.Name != "get_weather" {
		t.Errorf("Expected tool name 'get_weather', got %q", finalToolCall.Name)
	}
}

// getWeatherFunctionDeclaration provides a test schema matching Google's examples
func getWeatherFunctionDeclaration() tools.FunctionSchema {
	props := jsonmap.New()
	props.Set("location", tools.ValueSchema{
		Type:        "string",
		Description: "The location to get the weather for",
	})
	props.Set("country", tools.ValueSchema{
		Type:        "string",
		Description: "The country to get the weather for",
	})
	props.Set("unit", tools.ValueSchema{
		Type:        "string",
		Description: "Temperature unit (C or F)",
	})
	props.Set("purpose", tools.ValueSchema{
		Type:        "string",
		Description: "Describes the purpose of asking the weather",
	})
	return tools.FunctionSchema{
		Name:        "get_current_weather",
		Description: "Get the current weather in a city",
		Parameters: tools.ValueSchema{
			Type:       "object",
			Properties: props,
			Required:   []string{"location", "unit", "country"},
		},
	}
}

// TestStreamingWithFunctionDeclarations_WithoutHistory tests streaming function calls
// with tools configured but no conversation history (from Python genai package)
func TestStreamingWithFunctionDeclarations_WithoutHistory(t *testing.T) {
	// Simulates a streaming response with function call parts
	streamResp := strings.Join([]string{
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "get_current_weather", "willContinue": true}}]}}]}`,
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"partialArgs": [{"jsonPath": "$.location", "stringValue": "boston"}], "willContinue": true}}]}}]}`,
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"partialArgs": [{"jsonPath": "$.unit", "stringValue": "C"}], "willContinue": true}}]}}]}`,
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"partialArgs": [{"jsonPath": "$.country", "stringValue": "US"}]}}]}}]}`,
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {}}]}}]}`,
	}, "\n")

	payloadCh := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)
		payloadCh <- payload

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(streamResp))
	}))
	defer server.Close()

	schema := getWeatherFunctionDeclaration()
	tb := tools.Box(
		tools.External("Weather", &schema, func(r tools.Runner, params json.RawMessage) tools.Result {
			return tools.SuccessFromString("ok")
		}),
	)

	model := New("gemini-2.5-pro").
		WithGeminiAPI("fake-key").
		WithStreamingToolArguments(true)
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("get the current weather in boston in celsius, the country should be US")},
	}, tb, nil)

	// Verify the request payload has streamFunctionCallArguments
	payload := <-payloadCh
	toolConfig := payload["toolConfig"].(map[string]any)
	funcConfig := toolConfig["functionCallingConfig"].(map[string]any)
	if streamArg, ok := funcConfig["streamFunctionCallArguments"].(bool); !ok || !streamArg {
		t.Errorf("expected streamFunctionCallArguments=true in request")
	}

	// Consume stream and verify structure
	var chunks int
	stream.Iter()(func(status llms.StreamStatus) bool {
		chunks++
		return true
	})

	if err := stream.Err(); err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	msg := stream.Message()
	if msg.Role != "model" {
		t.Errorf("Expected role 'model', got %q", msg.Role)
	}
	if len(msg.ToolCalls) != 1 {
		t.Errorf("Expected 1 tool call, got %d", len(msg.ToolCalls))
	}
	if msg.ToolCalls[0].Name != "get_current_weather" {
		t.Errorf("Expected tool name 'get_current_weather', got %q", msg.ToolCalls[0].Name)
	}
}

// TestStreamingWithFunctionDeclarations_WithHistory tests streaming function calls
// with previous conversation history containing function calls (from Python genai package)
func TestStreamingWithFunctionDeclarations_WithHistory(t *testing.T) {
	// Build history that matches previous_generate_content_history from Python test:
	// - User message
	// - Model message with functionCall(name, will_continue=true)
	// - Model message with functionCall(partial_args with null_value, will_continue=false)
	//
	// In our Go library, this history would be reconstructed from previous stream events.
	// We test that sending such history back to the API works correctly.

	payloadCh := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)
		payloadCh <- payload

		w.Header().Set("Content-Type", "text/event-stream")
		// Response continues with text after function call history
		_, _ = w.Write([]byte(`data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "Based on my analysis..."}]}}]}` + "\n"))
	}))
	defer server.Close()

	schema := getWeatherFunctionDeclaration()
	tb := tools.Box(
		tools.External("Weather", &schema, func(r tools.Runner, params json.RawMessage) tools.Result {
			return tools.SuccessFromString("ok")
		}),
	)

	// Construct history with streaming function call parts
	// This simulates what we would have captured from a previous streaming session
	history := []llms.Message{
		{
			Role:    "user",
			Content: content.FromText("get the current weather in boston in celsius, the country is U.S."),
		},
		{
			// First model message: function call started (will_continue=true)
			Role: "assistant",
			ToolCalls: []llms.ToolCall{
				{
					ID:        "fc-weather-1",
					Name:      "get_current_weather",
					Arguments: json.RawMessage(`{"location":"boston","unit":"C","country":null}`),
				},
			},
		},
		{
			// Tool response
			Role:         "tool",
			ToolCallID:   "fc-weather-1",
			ToolCallName: "get_current_weather",
			Content:      content.FromRawJSON(json.RawMessage(`{"temperature": 21, "unit": "C"}`)),
		},
	}

	model := New("gemini-2.5-pro").
		WithGeminiAPI("fake-key").
		WithStreamingToolArguments(true)
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, history, tb, nil)

	if err := stream.Err(); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	// Verify the payload contains the history
	payload := <-payloadCh
	contents, ok := payload["contents"].([]any)
	if !ok {
		t.Fatal("contents missing from payload")
	}

	// Should have: user, model (with functionCall), function (with response)
	if len(contents) < 3 {
		t.Errorf("Expected at least 3 content items in history, got %d", len(contents))
	}

	// Consume stream
	stream.Iter()(func(status llms.StreamStatus) bool {
		return true
	})

	if err := stream.Err(); err != nil {
		t.Fatalf("Stream error: %v", err)
	}
}

// TestStreamingWithFunctionDeclarations_WithResponse tests full round-trip:
// streaming function call → function response → continuation (from Python genai package)
func TestStreamingWithFunctionDeclarations_WithResponse(t *testing.T) {
	// This test simulates the full flow:
	// 1. First call: Get streaming function call
	// 2. Capture the function call content
	// 3. Add function response
	// 4. Second call: Continue conversation with response

	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		callCount++

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		if callCount == 1 {
			// First call: return streaming function call
			streamResp := strings.Join([]string{
				`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "get_current_weather", "willContinue": true}}]}}]}`,
				`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"partialArgs": [{"jsonPath": "$.location", "stringValue": "boston"}], "willContinue": true}}]}}]}`,
				`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"partialArgs": [{"jsonPath": "$.unit", "stringValue": "C"}]}}]}}]}`,
				`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {}}]}}]}`,
			}, "\n")
			w.Write([]byte(streamResp))
		} else {
			// Second call: return text response after function result
			w.Write([]byte(`data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "The weather in Boston is currently 21°C."}]}}]}` + "\n"))
		}
	}))
	defer server.Close()

	schema := getWeatherFunctionDeclaration()
	tb := tools.Box(
		tools.External("Weather", &schema, func(r tools.Runner, params json.RawMessage) tools.Result {
			return tools.SuccessFromString("ok")
		}),
	)

	model := New("gemini-2.5-pro").
		WithGeminiAPI("fake-key").
		WithStreamingToolArguments(true)
	model.endpoint = server.URL

	ctx := context.Background()

	// First call: get the function call
	stream1 := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("get the current weather in boston in celsius")},
	}, tb, nil)

	var toolCall llms.ToolCall
	stream1.Iter()(func(status llms.StreamStatus) bool {
		if status == llms.StreamStatusToolCallReady {
			toolCall = stream1.ToolCall()
		}
		return true
	})

	if err := stream1.Err(); err != nil {
		t.Fatalf("First stream error: %v", err)
	}

	if toolCall.Name != "get_current_weather" {
		t.Fatalf("Expected tool call 'get_current_weather', got %q", toolCall.Name)
	}

	// Build history with the function call and response
	history := []llms.Message{
		{Role: "user", Content: content.FromText("get the current weather in boston in celsius")},
		{
			Role:      "assistant",
			ToolCalls: []llms.ToolCall{toolCall},
		},
		{
			Role:         "tool",
			ToolCallID:   toolCall.ID,
			ToolCallName: toolCall.Name,
			Content:      content.FromRawJSON(json.RawMessage(`{"temperature": 21, "unit": "C"}`)),
		},
	}

	// Second call: continue with function response
	stream2 := model.Generate(ctx, nil, history, tb, nil)

	var textContent string
	stream2.Iter()(func(status llms.StreamStatus) bool {
		if status == llms.StreamStatusText {
			textContent += stream2.Text()
		}
		return true
	})

	if err := stream2.Err(); err != nil {
		t.Fatalf("Second stream error: %v", err)
	}

	if !strings.Contains(textContent, "21") {
		t.Errorf("Expected response to contain temperature, got %q", textContent)
	}

	if callCount != 2 {
		t.Errorf("Expected 2 API calls, got %d", callCount)
	}
}

// TestStreamingWithPartialArgsNullValue tests nullValue handling in partialArgs
// (from Python genai package)
func TestStreamingWithPartialArgsNullValue(t *testing.T) {
	// Test that null_value in partialArgs is handled correctly
	// From Python: types.PartialArg(json_path='$.country', null_value="NULL_VALUE")

	streamResp := strings.Join([]string{
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "get_current_weather", "willContinue": true}}]}}]}`,
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"partialArgs": [{"jsonPath": "$.location", "stringValue": "boston"}], "willContinue": true}}]}}]}`,
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"partialArgs": [{"jsonPath": "$.country", "nullValue": true}], "willContinue": false}}]}}]}`,
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {}}]}}]}`,
	}, "\n")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(streamResp))
	}))
	defer server.Close()

	model := New("gemini-2.5-pro").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("weather in boston")},
	}, nil, nil)

	var toolCall llms.ToolCall
	stream.Iter()(func(status llms.StreamStatus) bool {
		if status == llms.StreamStatusToolCallReady {
			toolCall = stream.ToolCall()
		}
		return true
	})

	if err := stream.Err(); err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	// Parse the arguments and verify null value
	var args map[string]any
	if err := json.Unmarshal(toolCall.Arguments, &args); err != nil {
		t.Fatalf("Failed to parse arguments: %v", err)
	}

	// country should be null
	if args["country"] != nil {
		t.Errorf("Expected country=null, got %v", args["country"])
	}
	// location should be set
	if args["location"] != "boston" {
		t.Errorf("Expected location='boston', got %v", args["location"])
	}
}

// TestStreamingWithWillContinueFlag tests willContinue flag handling (from Python genai package)
func TestStreamingWithWillContinueFlag(t *testing.T) {
	// Test proper handling of willContinue at functionCall level
	// willContinue=true means more chunks coming
	// willContinue=false or absent means function call is complete

	streamResp := strings.Join([]string{
		// First chunk: name with willContinue=true
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "get_current_weather", "willContinue": true}}]}}]}`,
		// Second chunk: partialArgs with willContinue=true
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"partialArgs": [{"jsonPath": "$.location", "stringValue": "NYC", "willContinue": true}], "willContinue": true}}]}}]}`,
		// Third chunk: more string content for same arg
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"partialArgs": [{"jsonPath": "$.location", "stringValue": ""}], "willContinue": true}}]}}]}`,
		// Fourth chunk: willContinue=false signals end
		`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"willContinue": false}}]}}]}`,
	}, "\n")

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(streamResp))
	}))
	defer server.Close()

	model := New("gemini-2.5-pro").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("weather")},
	}, nil, nil)

	var beginCount, deltaCount, readyCount int
	stream.Iter()(func(status llms.StreamStatus) bool {
		switch status {
		case llms.StreamStatusToolCallBegin:
			beginCount++
		case llms.StreamStatusToolCallDelta:
			deltaCount++
		case llms.StreamStatusToolCallReady:
			readyCount++
		}
		return true
	})

	if err := stream.Err(); err != nil {
		t.Fatalf("Stream error: %v", err)
	}

	// Should have exactly 1 Begin (first chunk with name)
	if beginCount != 1 {
		t.Errorf("Expected 1 ToolCallBegin, got %d", beginCount)
	}

	// Should have 3 Deltas (chunks 2 and 3 have partialArgs, finalize adds closing brace)
	if deltaCount != 3 {
		t.Errorf("Expected 3 ToolCallDelta, got %d", deltaCount)
	}

	// Should have 1 Ready (when willContinue=false)
	if readyCount != 1 {
		t.Errorf("Expected 1 ToolCallReady, got %d", readyCount)
	}
}

// TestStreamingMultiTurnConversation tests multi-turn conversation with streaming
// function calls (from Python genai package)
func TestStreamingMultiTurnConversation(t *testing.T) {
	turnCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		turnCount++

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		switch turnCount {
		case 1:
			// First turn: function call for boston
			streamResp := strings.Join([]string{
				`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "get_current_weather"}}]}}]}`,
				`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"partialArgs": [{"jsonPath": "$.location", "stringValue": "boston"}]}}]}}]}`,
				`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {}}]}}]}`,
			}, "\n")
			w.Write([]byte(streamResp))
		case 2:
			// Response to function call
			w.Write([]byte(`data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "Boston is 21C."}]}}]}` + "\n"))
		case 3:
			// Second turn: function call for new brunswick
			streamResp := strings.Join([]string{
				`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"name": "get_current_weather"}}]}}]}`,
				`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {"partialArgs": [{"jsonPath": "$.location", "stringValue": "new brunswick"}]}}]}}]}`,
				`data: {"candidates": [{"content": {"role": "model", "parts": [{"functionCall": {}}]}}]}`,
			}, "\n")
			w.Write([]byte(streamResp))
		case 4:
			// Response to second function call
			w.Write([]byte(`data: {"candidates": [{"content": {"role": "model", "parts": [{"text": "New Brunswick is 18C."}]}}]}` + "\n"))
		}
	}))
	defer server.Close()

	schema := getWeatherFunctionDeclaration()
	tb := tools.Box(
		tools.External("Weather", &schema, func(r tools.Runner, params json.RawMessage) tools.Result {
			return tools.SuccessFromString("ok")
		}),
	)

	model := New("gemini-2.5-pro").
		WithGeminiAPI("fake-key").
		WithStreamingToolArguments(true)
	model.endpoint = server.URL

	ctx := context.Background()

	// Simulate multi-turn conversation like the Python test
	messages := []string{
		"get the current weather in boston in celsius",
		"get the current weather in new brunswick in celsius",
	}

	var history []llms.Message

	for i, userMsg := range messages {
		// Add user message
		history = append(history, llms.Message{
			Role:    "user",
			Content: content.FromText(userMsg),
		})

		// Generate
		stream := model.Generate(ctx, nil, history, tb, nil)

		var toolCall llms.ToolCall
		var hasToolCall bool
		stream.Iter()(func(status llms.StreamStatus) bool {
			if status == llms.StreamStatusToolCallReady {
				toolCall = stream.ToolCall()
				hasToolCall = true
			}
			return true
		})

		if err := stream.Err(); err != nil {
			t.Fatalf("Turn %d stream error: %v", i+1, err)
		}

		if hasToolCall {
			// Add assistant message with tool call
			history = append(history, llms.Message{
				Role:      "assistant",
				ToolCalls: []llms.ToolCall{toolCall},
			})

			// Add tool response
			history = append(history, llms.Message{
				Role:         "tool",
				ToolCallID:   toolCall.ID,
				ToolCallName: toolCall.Name,
				Content:      content.FromRawJSON(json.RawMessage(`{"temperature": 21, "unit": "C"}`)),
			})

			// Get model's response to tool result
			stream2 := model.Generate(ctx, nil, history, tb, nil)
			var textResp string
			stream2.Iter()(func(status llms.StreamStatus) bool {
				if status == llms.StreamStatusText {
					textResp += stream2.Text()
				}
				return true
			})

			if err := stream2.Err(); err != nil {
				t.Fatalf("Turn %d response error: %v", i+1, err)
			}

			// Add model's text response to history
			history = append(history, llms.Message{
				Role:    "assistant",
				Content: content.FromText(textResp),
			})
		}
	}

	// Should have had 4 API calls total (2 function calls + 2 responses)
	if turnCount != 4 {
		t.Errorf("Expected 4 API calls for multi-turn, got %d", turnCount)
	}

	// History should have accumulated messages
	// 2 user + 2 assistant(tool) + 2 tool + 2 assistant(text) = 8
	if len(history) != 8 {
		t.Errorf("Expected 8 messages in history, got %d", len(history))
	}
}

// This test simulates streaming partialArgs and verifies the emitted tool deltas
// remain a valid JSON prefix as fields and string segments arrive.
func TestGeminiToolDeltasArePrefixSafe(t *testing.T) {
	t.Parallel()

	stream := newStreamingArgsBuilder()
	toolCallDeltaSentBytes := 0

	// Chunk 1: platform, willContinue=true (string stays open)
	patch1 := partialFunctionArgument{
		JSONPath:     "$.platform",
		StringValue:  strPtr("web"),
		WillContinue: true,
	}
	_ = stream.applyPartialArg(patch1)
	delta1 := string(stream.buf.Bytes()[toolCallDeltaSentBytes:])
	toolCallDeltaSentBytes = stream.buf.Len()

	// Chunk 2: supplementalContext, final chunk (willContinue=false)
	patch2 := partialFunctionArgument{
		JSONPath:     "$.supplementalContext",
		StringValue:  strPtr("User wants cozy"),
		WillContinue: false,
	}
	_ = stream.applyPartialArg(patch2)
	delta2 := string(stream.buf.Bytes()[toolCallDeltaSentBytes:])
	toolCallDeltaSentBytes = stream.buf.Len()

	// Finalize (closes any open strings and the object) and emit tail delta
	stream.finalize()
	delta3 := string(stream.buf.Bytes()[toolCallDeltaSentBytes:])

	combined := delta1 + delta2 + delta3
	var parsed map[string]any
	if err := json.Unmarshal([]byte(combined), &parsed); err != nil {
		t.Fatalf("streamed tool args should be valid JSON; got %q with error: %v", combined, err)
	}

	if parsed["platform"] != "web" || parsed["supplementalContext"] != "User wants cozy" {
		t.Fatalf("unexpected parsed content: %#v", parsed)
	}
}

// This test mirrors the docs examples: numberValue, streamed string chunks,
// valueless partialArg for continued streaming, and parallel calls.
func TestGeminiDocStreamingExamples(t *testing.T) {
	t.Parallel()

	// Single call: brightness number, colorTemperature string with empty continuation
	builder := newStreamingArgsBuilder()
	_ = builder.applyPartialArg(partialFunctionArgument{
		JSONPath:    "$.brightness",
		NumberValue: floatPtr(50),
	})
	_ = builder.applyPartialArg(partialFunctionArgument{
		JSONPath:     "$.colorTemperature",
		StringValue:  strPtr("warm"),
		WillContinue: true,
	})
	_ = builder.applyPartialArg(partialFunctionArgument{
		JSONPath: "$.colorTemperature",
	})
	builder.finalize()

	var parsed map[string]any
	if err := json.Unmarshal(builder.buf.Bytes(), &parsed); err != nil {
		t.Fatalf("single-call example JSON invalid: %v", err)
	}
	if parsed["brightness"] != float64(50) || parsed["colorTemperature"] != "warm" {
		t.Fatalf("single-call example unexpected: %#v", parsed)
	}

	// Parallel calls: two builders interleaved
	call1 := newStreamingArgsBuilder()
	call2 := newStreamingArgsBuilder()

	_ = call1.applyPartialArg(partialFunctionArgument{
		JSONPath:     "$.location",
		StringValue:  strPtr("New Delhi"),
		WillContinue: true,
	})
	_ = call1.applyPartialArg(partialFunctionArgument{
		JSONPath: "$.location",
	})
	call1.finalize()

	if err := json.Unmarshal(call1.buf.Bytes(), &parsed); err != nil {
		t.Fatalf("call1 JSON invalid: %v", err)
	}
	if parsed["location"] != "New Delhi" {
		t.Fatalf("call1 location mismatch: %#v", parsed)
	}

	_ = call2.applyPartialArg(partialFunctionArgument{
		JSONPath:     "$.location",
		StringValue:  strPtr("San Francisco"),
		WillContinue: true,
	})
	_ = call2.applyPartialArg(partialFunctionArgument{
		JSONPath: "$.location",
	})
	call2.finalize()

	if err := json.Unmarshal(call2.buf.Bytes(), &parsed); err != nil {
		t.Fatalf("call2 JSON invalid: %v", err)
	}
	if parsed["location"] != "San Francisco" {
		t.Fatalf("call2 location mismatch: %#v", parsed)
	}
}

func strPtr(s string) *string     { return &s }
func floatPtr(f float64) *float64 { return &f }

// TestSanitizeSchemaForGemini_StripsExclusiveMinimum tests that the exclusiveMinimum
// keyword is stripped from tool schemas sent to Gemini. This keyword is generated
// by Zod's .positive() validation but is not supported by Gemini's API.
func TestSanitizeSchemaForGemini_StripsExclusiveMinimum(t *testing.T) {
	// Create a schema with exclusiveMinimum by storing it as a *jsonmap.Map
	// (simulating how it would arrive from a TypeScript frontend via jsonmap)
	props := jsonmap.New()

	// Create a property with exclusiveMinimum (unsupported by Gemini)
	countProp := jsonmap.New()
	countProp.Set("type", "integer")
	countProp.Set("description", "A positive count")
	countProp.Set("exclusiveMinimum", float64(0)) // Unsupported field
	props.Set("count", countProp)

	schema := tools.FunctionSchema{
		Name:        "test_func",
		Description: "Test function",
		Parameters: tools.ValueSchema{
			Type:       "object",
			Properties: props,
			Required:   []string{"count"},
		},
	}

	payloadCh := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)
		payloadCh <- payload

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`data: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}` + "\n"))
	}))
	defer server.Close()

	tb := tools.Box(
		tools.External("Test", &schema, func(r tools.Runner, params json.RawMessage) tools.Result {
			return tools.SuccessFromString("ok")
		}),
	)

	model := New("gemini-2.0-flash").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("test")},
	}, tb, nil)

	if err := stream.Err(); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	payload := <-payloadCh

	// Navigate to the function declaration parameters
	toolsPayload := payload["tools"].(map[string]any)
	declarations := toolsPayload["functionDeclarations"].([]any)
	decl := declarations[0].(map[string]any)
	params := decl["parameters"].(map[string]any)
	properties := params["properties"].(map[string]any)
	countSchema := properties["count"].(map[string]any)

	// Verify exclusiveMinimum was stripped
	if _, exists := countSchema["exclusiveMinimum"]; exists {
		t.Error("exclusiveMinimum should have been stripped from schema")
	}

	// Verify supported fields are preserved
	if countSchema["type"] != "integer" {
		t.Errorf("Expected type 'integer', got %v", countSchema["type"])
	}
	if countSchema["description"] != "A positive count" {
		t.Errorf("Expected description 'A positive count', got %v", countSchema["description"])
	}
}

// TestSanitizeSchemaForGemini_StripsConstAndPattern tests that const and pattern
// keywords are stripped from tool schemas. These are generated by z.literal()
// and z.string().regex() but not supported by Gemini.
func TestSanitizeSchemaForGemini_StripsConstAndPattern(t *testing.T) {
	props := jsonmap.New()

	// Property with const (from z.literal())
	typeProp := jsonmap.New()
	typeProp.Set("type", "null")
	typeProp.Set("const", nil) // Unsupported field
	props.Set("nullField", typeProp)

	// Property with pattern, minLength, maxLength (from z.string().regex())
	emailProp := jsonmap.New()
	emailProp.Set("type", "string")
	emailProp.Set("pattern", "^[a-z]+@[a-z]+\\.[a-z]+$") // Unsupported
	emailProp.Set("minLength", float64(5))               // Unsupported
	emailProp.Set("maxLength", float64(100))             // Unsupported
	props.Set("email", emailProp)

	schema := tools.FunctionSchema{
		Name:        "test_func",
		Description: "Test function",
		Parameters: tools.ValueSchema{
			Type:       "object",
			Properties: props,
		},
	}

	payloadCh := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)
		payloadCh <- payload

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`data: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}` + "\n"))
	}))
	defer server.Close()

	tb := tools.Box(
		tools.External("Test", &schema, func(r tools.Runner, params json.RawMessage) tools.Result {
			return tools.SuccessFromString("ok")
		}),
	)

	model := New("gemini-2.0-flash").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("test")},
	}, tb, nil)

	if err := stream.Err(); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	payload := <-payloadCh

	toolsPayload := payload["tools"].(map[string]any)
	declarations := toolsPayload["functionDeclarations"].([]any)
	decl := declarations[0].(map[string]any)
	params := decl["parameters"].(map[string]any)
	properties := params["properties"].(map[string]any)

	// Check nullField - const should be stripped
	nullSchema := properties["nullField"].(map[string]any)
	if _, exists := nullSchema["const"]; exists {
		t.Error("const should have been stripped from nullField schema")
	}
	if nullSchema["type"] != "null" {
		t.Errorf("nullField type should be preserved as 'null', got %v", nullSchema["type"])
	}

	// Check email - pattern, minLength, maxLength should be stripped
	emailSchema := properties["email"].(map[string]any)
	for _, field := range []string{"pattern", "minLength", "maxLength"} {
		if _, exists := emailSchema[field]; exists {
			t.Errorf("%s should have been stripped from email schema", field)
		}
	}
	if emailSchema["type"] != "string" {
		t.Errorf("email type should be preserved as 'string', got %v", emailSchema["type"])
	}
}

// TestSanitizeSchemaForGemini_NestedProperties tests that nested properties
// stored as *jsonmap.Map are also sanitized recursively.
func TestSanitizeSchemaForGemini_NestedProperties(t *testing.T) {
	// Create nested object with unsupported fields
	innerProps := jsonmap.New()
	valueProp := jsonmap.New()
	valueProp.Set("type", "number")
	valueProp.Set("exclusiveMinimum", float64(0))
	valueProp.Set("exclusiveMaximum", float64(100))
	innerProps.Set("value", valueProp)

	// Create outer property as a nested object
	outerProp := jsonmap.New()
	outerProp.Set("type", "object")
	outerProp.Set("properties", innerProps)

	props := jsonmap.New()
	props.Set("nested", outerProp)

	schema := tools.FunctionSchema{
		Name:        "test_func",
		Description: "Test function",
		Parameters: tools.ValueSchema{
			Type:       "object",
			Properties: props,
		},
	}

	payloadCh := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)
		payloadCh <- payload

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`data: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}` + "\n"))
	}))
	defer server.Close()

	tb := tools.Box(
		tools.External("Test", &schema, func(r tools.Runner, params json.RawMessage) tools.Result {
			return tools.SuccessFromString("ok")
		}),
	)

	model := New("gemini-2.0-flash").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("test")},
	}, tb, nil)

	if err := stream.Err(); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	payload := <-payloadCh

	toolsPayload := payload["tools"].(map[string]any)
	declarations := toolsPayload["functionDeclarations"].([]any)
	decl := declarations[0].(map[string]any)
	params := decl["parameters"].(map[string]any)
	properties := params["properties"].(map[string]any)
	nestedSchema := properties["nested"].(map[string]any)
	nestedProps := nestedSchema["properties"].(map[string]any)
	valueSchema := nestedProps["value"].(map[string]any)

	// Verify unsupported fields were stripped from nested property
	for _, field := range []string{"exclusiveMinimum", "exclusiveMaximum"} {
		if _, exists := valueSchema[field]; exists {
			t.Errorf("%s should have been stripped from nested value schema", field)
		}
	}

	// Verify supported fields are preserved
	if valueSchema["type"] != "number" {
		t.Errorf("nested value type should be 'number', got %v", valueSchema["type"])
	}
}

// TestSanitizeSchemaForGemini_AnyOfWithUnsupportedFields tests that AnyOf schemas
// with unsupported fields are also sanitized.
func TestSanitizeSchemaForGemini_AnyOfWithUnsupportedFields(t *testing.T) {
	props := jsonmap.New()

	// Create an anyOf with unsupported fields in one of the schemas
	anyOfProp := jsonmap.New()
	anyOf := []map[string]any{
		{
			"type": "string",
		},
		{
			"type":  "integer",
			"const": float64(42), // Unsupported field in anyOf schema
		},
		{
			"type":             "number",
			"exclusiveMinimum": float64(0), // Unsupported field
		},
	}
	anyOfProp.Set("anyOf", anyOf)
	props.Set("value", anyOfProp)

	schema := tools.FunctionSchema{
		Name:        "test_func",
		Description: "Test function",
		Parameters: tools.ValueSchema{
			Type:       "object",
			Properties: props,
		},
	}

	payloadCh := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)
		payloadCh <- payload

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`data: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}` + "\n"))
	}))
	defer server.Close()

	tb := tools.Box(
		tools.External("Test", &schema, func(r tools.Runner, params json.RawMessage) tools.Result {
			return tools.SuccessFromString("ok")
		}),
	)

	model := New("gemini-2.0-flash").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("test")},
	}, tb, nil)

	if err := stream.Err(); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	payload := <-payloadCh

	toolsPayload := payload["tools"].(map[string]any)
	declarations := toolsPayload["functionDeclarations"].([]any)
	decl := declarations[0].(map[string]any)
	params := decl["parameters"].(map[string]any)
	properties := params["properties"].(map[string]any)
	valueSchema := properties["value"].(map[string]any)
	anyOfSchemas := valueSchema["anyOf"].([]any)

	// Check each anyOf schema for unsupported fields
	for i, schema := range anyOfSchemas {
		schemaMap := schema.(map[string]any)
		for _, field := range []string{"const", "exclusiveMinimum"} {
			if _, exists := schemaMap[field]; exists {
				t.Errorf("anyOf[%d]: %s should have been stripped", i, field)
			}
		}
	}

	// Verify types are preserved
	if anyOfSchemas[0].(map[string]any)["type"] != "string" {
		t.Error("anyOf[0] type should be 'string'")
	}
	if anyOfSchemas[1].(map[string]any)["type"] != "integer" {
		t.Error("anyOf[1] type should be 'integer'")
	}
	if anyOfSchemas[2].(map[string]any)["type"] != "number" {
		t.Error("anyOf[2] type should be 'number'")
	}
}

// TestSanitizeSchemaForGemini_ArrayItemsWithUnsupportedFields tests that array
// Items schemas with unsupported fields are sanitized.
func TestSanitizeSchemaForGemini_ArrayItemsWithUnsupportedFields(t *testing.T) {
	props := jsonmap.New()

	// Create an array property with items that have unsupported fields
	itemsProp := jsonmap.New()
	itemsProp.Set("type", "integer")
	itemsProp.Set("minimum", float64(1))          // Supported by some but not Gemini
	itemsProp.Set("exclusiveMinimum", float64(0)) // Unsupported

	arrayProp := jsonmap.New()
	arrayProp.Set("type", "array")
	arrayProp.Set("items", itemsProp)
	props.Set("numbers", arrayProp)

	schema := tools.FunctionSchema{
		Name:        "test_func",
		Description: "Test function",
		Parameters: tools.ValueSchema{
			Type:       "object",
			Properties: props,
		},
	}

	payloadCh := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)
		payloadCh <- payload

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`data: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}` + "\n"))
	}))
	defer server.Close()

	tb := tools.Box(
		tools.External("Test", &schema, func(r tools.Runner, params json.RawMessage) tools.Result {
			return tools.SuccessFromString("ok")
		}),
	)

	model := New("gemini-2.0-flash").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("test")},
	}, tb, nil)

	if err := stream.Err(); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	payload := <-payloadCh

	toolsPayload := payload["tools"].(map[string]any)
	declarations := toolsPayload["functionDeclarations"].([]any)
	decl := declarations[0].(map[string]any)
	params := decl["parameters"].(map[string]any)
	properties := params["properties"].(map[string]any)
	arraySchema := properties["numbers"].(map[string]any)
	itemsSchema := arraySchema["items"].(map[string]any)

	// Verify unsupported fields were stripped from items
	for _, field := range []string{"minimum", "exclusiveMinimum"} {
		if _, exists := itemsSchema[field]; exists {
			t.Errorf("items: %s should have been stripped", field)
		}
	}

	// Verify type is preserved
	if itemsSchema["type"] != "integer" {
		t.Errorf("items type should be 'integer', got %v", itemsSchema["type"])
	}
}

// TestSanitizeSchemaForGemini_PreservesValueSchemaType tests that properties
// already stored as tools.ValueSchema are handled correctly.
func TestSanitizeSchemaForGemini_PreservesValueSchemaType(t *testing.T) {
	props := jsonmap.New()

	// Set a property as tools.ValueSchema (not *jsonmap.Map)
	props.Set("name", tools.ValueSchema{
		Type:        "string",
		Description: "The user's name",
	})

	// Also add one as *jsonmap.Map for comparison
	ageProp := jsonmap.New()
	ageProp.Set("type", "integer")
	ageProp.Set("description", "The user's age")
	props.Set("age", ageProp)

	schema := tools.FunctionSchema{
		Name:        "test_func",
		Description: "Test function",
		Parameters: tools.ValueSchema{
			Type:       "object",
			Properties: props,
			Required:   []string{"name", "age"},
		},
	}

	payloadCh := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)
		payloadCh <- payload

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`data: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}` + "\n"))
	}))
	defer server.Close()

	tb := tools.Box(
		tools.External("Test", &schema, func(r tools.Runner, params json.RawMessage) tools.Result {
			return tools.SuccessFromString("ok")
		}),
	)

	model := New("gemini-2.0-flash").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("test")},
	}, tb, nil)

	if err := stream.Err(); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	payload := <-payloadCh

	toolsPayload := payload["tools"].(map[string]any)
	declarations := toolsPayload["functionDeclarations"].([]any)
	decl := declarations[0].(map[string]any)
	params := decl["parameters"].(map[string]any)
	properties := params["properties"].(map[string]any)

	// Verify both properties are present and correct
	nameSchema := properties["name"].(map[string]any)
	if nameSchema["type"] != "string" {
		t.Errorf("name type should be 'string', got %v", nameSchema["type"])
	}
	if nameSchema["description"] != "The user's name" {
		t.Errorf("name description mismatch, got %v", nameSchema["description"])
	}

	ageSchema := properties["age"].(map[string]any)
	if ageSchema["type"] != "integer" {
		t.Errorf("age type should be 'integer', got %v", ageSchema["type"])
	}
	if ageSchema["description"] != "The user's age" {
		t.Errorf("age description mismatch, got %v", ageSchema["description"])
	}

	// Verify required is preserved
	required := params["required"].([]any)
	if len(required) != 2 {
		t.Errorf("expected 2 required fields, got %d", len(required))
	}
}

// TestSanitizeSchemaForGemini_MapStringAny tests that properties stored as
// map[string]any (not *jsonmap.Map) are also sanitized. This can happen when
// schemas come from certain sources or after JSON round-tripping.
func TestSanitizeSchemaForGemini_MapStringAny(t *testing.T) {
	props := jsonmap.New()

	// Set a property directly as map[string]any (not *jsonmap.Map)
	countProp := map[string]any{
		"type":             "integer",
		"description":      "A positive count",
		"exclusiveMinimum": float64(0), // Unsupported field
	}
	props.Set("count", countProp)

	schema := tools.FunctionSchema{
		Name:        "test_func",
		Description: "Test function",
		Parameters: tools.ValueSchema{
			Type:       "object",
			Properties: props,
			Required:   []string{"count"},
		},
	}

	payloadCh := make(chan map[string]any, 1)
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		defer r.Body.Close()
		var payload map[string]any
		json.NewDecoder(r.Body).Decode(&payload)
		payloadCh <- payload

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)
		w.Write([]byte(`data: {"candidates": [{"content": {"parts": [{"text": "ok"}]}}]}` + "\n"))
	}))
	defer server.Close()

	tb := tools.Box(
		tools.External("Test", &schema, func(r tools.Runner, params json.RawMessage) tools.Result {
			return tools.SuccessFromString("ok")
		}),
	)

	model := New("gemini-2.0-flash").WithGeminiAPI("fake-key")
	model.endpoint = server.URL

	ctx := context.Background()
	stream := model.Generate(ctx, nil, []llms.Message{
		{Role: "user", Content: content.FromText("test")},
	}, tb, nil)

	if err := stream.Err(); err != nil {
		t.Fatalf("Generate failed: %v", err)
	}

	payload := <-payloadCh

	toolsPayload := payload["tools"].(map[string]any)
	declarations := toolsPayload["functionDeclarations"].([]any)
	decl := declarations[0].(map[string]any)
	params := decl["parameters"].(map[string]any)
	properties := params["properties"].(map[string]any)
	countSchema := properties["count"].(map[string]any)

	// Verify exclusiveMinimum was stripped
	if _, exists := countSchema["exclusiveMinimum"]; exists {
		t.Error("exclusiveMinimum should have been stripped from map[string]any schema")
	}

	// Verify supported fields are preserved
	if countSchema["type"] != "integer" {
		t.Errorf("Expected type 'integer', got %v", countSchema["type"])
	}
	if countSchema["description"] != "A positive count" {
		t.Errorf("Expected description 'A positive count', got %v", countSchema["description"])
	}
}
