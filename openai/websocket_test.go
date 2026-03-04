package openai

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/coder/websocket"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
)

// newTestWSServer creates an httptest server that accepts a WebSocket upgrade
// and sends the provided SSE-style events as individual WebSocket messages.
func newTestWSServer(t *testing.T, events []string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			t.Logf("websocket accept error: %v", err)
			return
		}
		defer conn.CloseNow()

		// Read the request message (response.create envelope).
		_, _, err = conn.Read(r.Context())
		if err != nil {
			t.Logf("websocket read request: %v", err)
			return
		}

		// Send each event as a separate text message.
		for _, ev := range events {
			if err := conn.Write(r.Context(), websocket.MessageText, []byte(ev)); err != nil {
				return
			}
		}
	}))
}

func wsEndpoint(server *httptest.Server) string {
	return "ws" + strings.TrimPrefix(server.URL, "http")
}

func TestWebSocketStream_BasicText(t *testing.T) {
	events := []string{
		`{"type":"response.created","response":{"id":"resp_1"}}`,
		`{"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}`,
		`{"type":"response.output_text.delta","delta":"Hello"}`,
		`{"type":"response.output_text.delta","delta":" world"}`,
		`{"type":"response.completed","response":{"usage":{"input_tokens":10,"output_tokens":5,"total_tokens":15}}}`,
	}

	server := newTestWSServer(t, events)
	defer server.Close()

	provider := NewWebSocketResponsesAPI("test-token", "gpt-5").
		WithEndpoint(wsEndpoint(server), "Test")
	defer provider.Close()

	stream := provider.Generate(
		context.Background(),
		content.FromText("You are helpful"),
		[]llms.Message{{Role: "user", Content: content.FromText("Hi")}},
		nil, nil,
	)

	var texts []string
	for status := range stream.Iter() {
		if status == llms.StreamStatusText {
			texts = append(texts, stream.Text())
		}
	}

	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	joined := strings.Join(texts, "")
	if joined != "Hello world" {
		t.Fatalf("expected 'Hello world', got %q", joined)
	}

	usage := stream.Usage()
	if usage.InputTokens != 10 || usage.OutputTokens != 5 {
		t.Fatalf("unexpected usage: %+v", usage)
	}
}

func TestWebSocketStream_ToolCallStreaming(t *testing.T) {
	events := []string{
		`{"type":"response.created","response":{"id":"resp_tc"}}`,
		`{"type":"response.output_item.added","item":{"type":"function_call","id":"fc_1","name":"get_weather","call_id":"call_1","arguments":""}}`,
		`{"type":"response.function_call_arguments.delta","delta":"{\"city\":"}`,
		`{"type":"response.function_call_arguments.delta","delta":"\"NYC\"}"}`,
		`{"type":"response.function_call_arguments.done","arguments":"{\"city\":\"NYC\"}"}`,
		`{"type":"response.completed","response":{"usage":{"input_tokens":20,"output_tokens":10,"total_tokens":30}}}`,
	}

	server := newTestWSServer(t, events)
	defer server.Close()

	provider := NewWebSocketResponsesAPI("test-token", "gpt-5").
		WithEndpoint(wsEndpoint(server), "Test")
	defer provider.Close()

	stream := provider.Generate(
		context.Background(),
		content.FromText("You are helpful"),
		[]llms.Message{{Role: "user", Content: content.FromText("Weather?")}},
		nil, nil,
	)

	var beginCount, deltaCount, readyCount int
	for status := range stream.Iter() {
		switch status {
		case llms.StreamStatusToolCallBegin:
			beginCount++
		case llms.StreamStatusToolCallDelta:
			deltaCount++
		case llms.StreamStatusToolCallReady:
			readyCount++
		}
	}

	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	if beginCount != 1 {
		t.Fatalf("expected 1 tool call begin, got %d", beginCount)
	}
	if deltaCount != 2 {
		t.Fatalf("expected 2 tool call deltas, got %d", deltaCount)
	}
	if readyCount != 1 {
		t.Fatalf("expected 1 tool call ready, got %d", readyCount)
	}

	tc := stream.ToolCall()
	if tc.Name != "get_weather" {
		t.Fatalf("expected tool name 'get_weather', got %q", tc.Name)
	}
	if string(tc.Arguments) != `{"city":"NYC"}` {
		t.Fatalf("expected arguments '{\"city\":\"NYC\"}', got %q", string(tc.Arguments))
	}
}

func TestWebSocketStream_ChainingWithPreviousResponseID(t *testing.T) {
	// Channel to collect requests from the server handler.
	reqCh := make(chan json.RawMessage, 10)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.CloseNow()

		for {
			_, data, err := conn.Read(r.Context())
			if err != nil {
				return
			}
			reqCh <- json.RawMessage(append([]byte{}, data...))

			events := []string{
				`{"type":"response.created","response":{"id":"resp_1"}}`,
				`{"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}`,
				`{"type":"response.output_text.delta","delta":"ok"}`,
				`{"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":2,"total_tokens":7}}}`,
			}
			for _, ev := range events {
				if err := conn.Write(r.Context(), websocket.MessageText, []byte(ev)); err != nil {
					return
				}
			}
		}
	}))
	defer server.Close()

	endpoint := "ws" + strings.TrimPrefix(server.URL, "http")
	provider := NewWebSocketResponsesAPI("test-token", "gpt-5").
		WithEndpoint(endpoint, "Test")
	defer provider.Close()

	// First turn: user message.
	messages1 := []llms.Message{
		{Role: "user", Content: content.FromText("Hi")},
	}
	stream1 := provider.Generate(context.Background(), content.FromText("sys"), messages1, nil, nil)
	for range stream1.Iter() {
	}
	if err := stream1.Err(); err != nil {
		t.Fatalf("turn 1 error: %v", err)
	}

	// Second turn: add assistant + tool result messages (simulating tool loop).
	messages2 := []llms.Message{
		{Role: "user", Content: content.FromText("Hi")},
		{Role: "assistant", Content: content.FromText("ok"), ID: "msg_1"},
		{Role: "user", Content: content.FromText("Thanks")},
	}
	stream2 := provider.Generate(context.Background(), content.FromText("sys"), messages2, nil, nil)
	for range stream2.Iter() {
	}
	if err := stream2.Err(); err != nil {
		t.Fatalf("turn 2 error: %v", err)
	}

	// Collect requests.
	var receivedRequests []json.RawMessage
	for len(reqCh) > 0 {
		receivedRequests = append(receivedRequests, <-reqCh)
	}

	if len(receivedRequests) < 2 {
		t.Fatalf("expected 2 requests, got %d", len(receivedRequests))
	}

	var req2 struct {
		Type     string `json:"type"`
		Response struct {
			PreviousResponseID string          `json:"previous_response_id"`
			Input              json.RawMessage `json:"input"`
		} `json:"response"`
	}
	if err := json.Unmarshal(receivedRequests[1], &req2); err != nil {
		t.Fatalf("unmarshal second request: %v", err)
	}
	if req2.Response.PreviousResponseID != "resp_1" {
		t.Fatalf("expected previous_response_id 'resp_1', got %q", req2.Response.PreviousResponseID)
	}
}

func TestWebSocketStream_ReasoningSummary(t *testing.T) {
	events := []string{
		`{"type":"response.created","response":{"id":"resp_r"}}`,
		`{"type":"response.output_item.added","item":{"type":"reasoning","id":"r_1"}}`,
		`{"type":"response.reasoning_summary_text.delta","delta":"Thinking...","item_id":"r_1"}`,
		`{"type":"response.reasoning_summary_text.done","text":"Thinking...","item_id":"r_1"}`,
		`{"type":"response.output_item.done","item":{"type":"reasoning","id":"r_1","summary":[{"type":"summary_text","text":"Thinking..."}]}}`,
		`{"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}`,
		`{"type":"response.output_text.delta","delta":"Done"}`,
		`{"type":"response.completed","response":{"usage":{"input_tokens":5,"output_tokens":3,"total_tokens":8}}}`,
	}

	server := newTestWSServer(t, events)
	defer server.Close()

	provider := NewWebSocketResponsesAPI("test-token", "gpt-5").
		WithEndpoint(wsEndpoint(server), "Test")
	defer provider.Close()

	stream := provider.Generate(
		context.Background(),
		content.FromText("sys"),
		[]llms.Message{{Role: "user", Content: content.FromText("think")}},
		nil, nil,
	)

	var thinkingCount, doneCount int
	for status := range stream.Iter() {
		switch status {
		case llms.StreamStatusThinking:
			thinkingCount++
		case llms.StreamStatusThinkingDone:
			doneCount++
		}
	}

	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	if thinkingCount < 1 {
		t.Fatalf("expected at least 1 thinking event, got %d", thinkingCount)
	}
	if doneCount != 1 {
		t.Fatalf("expected 1 thinking done event, got %d", doneCount)
	}
}

func TestWebSocketStream_ErrorEvent(t *testing.T) {
	events := []string{
		`{"type":"error","error":{"code":"invalid_request","message":"bad request"}}`,
	}

	server := newTestWSServer(t, events)
	defer server.Close()

	provider := NewWebSocketResponsesAPI("test-token", "gpt-5").
		WithEndpoint(wsEndpoint(server), "Test")
	defer provider.Close()

	stream := provider.Generate(
		context.Background(),
		content.FromText("sys"),
		[]llms.Message{{Role: "user", Content: content.FromText("hi")}},
		nil, nil,
	)

	for range stream.Iter() {
	}

	if err := stream.Err(); err == nil {
		t.Fatal("expected error from error event")
	} else if !strings.Contains(err.Error(), "bad request") {
		t.Fatalf("expected 'bad request' in error, got: %v", err)
	}
}

func TestWebSocketStream_ResponseIDCaptured(t *testing.T) {
	events := []string{
		`{"type":"response.created","response":{"id":"resp_capture"}}`,
		`{"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}`,
		`{"type":"response.output_text.delta","delta":"hello"}`,
		`{"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`,
	}

	server := newTestWSServer(t, events)
	defer server.Close()

	provider := NewWebSocketResponsesAPI("test-token", "gpt-5").
		WithEndpoint(wsEndpoint(server), "Test")
	defer provider.Close()

	stream := provider.Generate(
		context.Background(),
		content.FromText("sys"),
		[]llms.Message{{Role: "user", Content: content.FromText("hi")}},
		nil, nil,
	)

	for range stream.Iter() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("stream error: %v", err)
	}

	// After consuming the stream, the provider should have captured the response ID.
	if provider.lastResponseID != "resp_capture" {
		t.Fatalf("expected lastResponseID 'resp_capture', got %q", provider.lastResponseID)
	}
}

func TestWebSocketStream_ResetChain(t *testing.T) {
	provider := NewWebSocketResponsesAPI("test-token", "gpt-5")
	provider.lastResponseID = "resp_old"
	provider.lastMessageCount = 5

	provider.ResetChain()

	if provider.lastResponseID != "" {
		t.Fatalf("expected empty lastResponseID after reset, got %q", provider.lastResponseID)
	}
	if provider.lastMessageCount != 0 {
		t.Fatalf("expected 0 lastMessageCount after reset, got %d", provider.lastMessageCount)
	}
}

func TestWebSocketStream_ContextCancellation(t *testing.T) {
	// Server that sends one event then blocks forever.
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.CloseNow()

		_, _, _ = conn.Read(r.Context())

		_ = conn.Write(r.Context(), websocket.MessageText,
			[]byte(`{"type":"response.created","response":{"id":"resp_hang"}}`))
		_ = conn.Write(r.Context(), websocket.MessageText,
			[]byte(`{"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}`))
		_ = conn.Write(r.Context(), websocket.MessageText,
			[]byte(`{"type":"response.output_text.delta","delta":"partial"}`))

		// Keep the connection open by reading (will fail when client closes).
		for {
			_, _, err := conn.Read(r.Context())
			if err != nil {
				return
			}
		}
	}))
	defer server.Close()

	provider := NewWebSocketResponsesAPI("test-token", "gpt-5").
		WithEndpoint(wsEndpoint(server), "Test")
	defer provider.Close()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	stream := provider.Generate(
		ctx,
		content.FromText("sys"),
		[]llms.Message{{Role: "user", Content: content.FromText("hi")}},
		nil, nil,
	)

	// Iterate and cancel after we get the first event.
	done := make(chan error, 1)
	go func() {
		for range stream.Iter() {
			cancel()
		}
		done <- stream.Err()
	}()

	err := <-done
	if err == nil {
		t.Fatal("expected context cancellation error")
	}
}

func TestWebSocketStream_ErrorStreamIterNoPanic(t *testing.T) {
	// Calling Iter() on an error stream (from a failed Generate) should not
	// panic, even without checking Err() first.
	stream := newWebSocketStreamError(fmt.Errorf("test error"))
	for range stream.Iter() {
		t.Fatal("should not yield any statuses")
	}
	if stream.Err() == nil || stream.Err().Error() != "test error" {
		t.Fatalf("expected 'test error', got %v", stream.Err())
	}
}

func TestWebSocketStream_WarmupChainsToFirstGenerate(t *testing.T) {
	reqCh := make(chan json.RawMessage, 10)

	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.CloseNow()

		for {
			_, data, err := conn.Read(r.Context())
			if err != nil {
				return
			}
			reqCh <- json.RawMessage(append([]byte{}, data...))

			// Check if this is a warmup (generate:false) or regular request.
			var envelope struct {
				Response struct {
					Generate *bool `json:"generate"`
				} `json:"response"`
			}
			_ = json.Unmarshal(data, &envelope)

			var respID string
			if envelope.Response.Generate != nil && !*envelope.Response.Generate {
				respID = "resp_warmup"
			} else {
				respID = "resp_gen"
			}

			events := []string{
				`{"type":"response.created","response":{"id":"` + respID + `"}}`,
				`{"type":"response.output_item.added","item":{"type":"message","id":"msg_1"}}`,
				`{"type":"response.output_text.delta","delta":"ok"}`,
				`{"type":"response.completed","response":{"usage":{"input_tokens":1,"output_tokens":1,"total_tokens":2}}}`,
			}
			for _, ev := range events {
				if err := conn.Write(r.Context(), websocket.MessageText, []byte(ev)); err != nil {
					return
				}
			}
		}
	}))
	defer server.Close()

	provider := NewWebSocketResponsesAPI("test-token", "gpt-5").
		WithEndpoint(wsEndpoint(server), "Test")
	defer provider.Close()

	// Warmup first.
	respID, err := provider.Warmup(context.Background(), "sys", nil)
	if err != nil {
		t.Fatalf("warmup error: %v", err)
	}
	if respID != "resp_warmup" {
		t.Fatalf("expected warmup resp ID 'resp_warmup', got %q", respID)
	}

	// First Generate should chain off the warmup response ID.
	stream := provider.Generate(
		context.Background(),
		content.FromText("sys"),
		[]llms.Message{{Role: "user", Content: content.FromText("Hi")}},
		nil, nil,
	)
	for range stream.Iter() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("generate error: %v", err)
	}

	// Drain the request channel and check the second request (Generate).
	var requests []json.RawMessage
	for len(reqCh) > 0 {
		requests = append(requests, <-reqCh)
	}

	if len(requests) < 2 {
		t.Fatalf("expected at least 2 requests (warmup + generate), got %d", len(requests))
	}

	var genReq struct {
		Response struct {
			PreviousResponseID string `json:"previous_response_id"`
		} `json:"response"`
	}
	if err := json.Unmarshal(requests[1], &genReq); err != nil {
		t.Fatalf("unmarshal generate request: %v", err)
	}
	if genReq.Response.PreviousResponseID != "resp_warmup" {
		t.Fatalf("expected previous_response_id 'resp_warmup', got %q", genReq.Response.PreviousResponseID)
	}
}

func TestWebSocketProvider_CompanyAndModel(t *testing.T) {
	p := NewWebSocketResponsesAPI("token", "gpt-5")
	if p.Company() != "OpenAI" {
		t.Fatalf("expected company 'OpenAI', got %q", p.Company())
	}
	if p.Model() != "gpt-5" {
		t.Fatalf("expected model 'gpt-5', got %q", p.Model())
	}
}

func TestWebSocketStream_Warmup(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		conn, err := websocket.Accept(w, r, nil)
		if err != nil {
			return
		}
		defer conn.CloseNow()

		// Read warmup request.
		_, data, err := conn.Read(r.Context())
		if err != nil {
			return
		}

		// Verify it has generate:false.
		var req map[string]any
		if err := json.Unmarshal(data, &req); err != nil {
			t.Logf("unmarshal warmup: %v", err)
			return
		}
		resp, _ := req["response"].(map[string]any)
		if gen, ok := resp["generate"].(bool); !ok || gen {
			t.Logf("expected generate:false in warmup request")
		}

		events := []string{
			`{"type":"response.created","response":{"id":"resp_warmup"}}`,
			`{"type":"response.completed","response":{"usage":{"input_tokens":0,"output_tokens":0,"total_tokens":0}}}`,
		}
		for _, ev := range events {
			if err := conn.Write(r.Context(), websocket.MessageText, []byte(ev)); err != nil {
				return
			}
		}
	}))
	defer server.Close()

	provider := NewWebSocketResponsesAPI("test-token", "gpt-5").
		WithEndpoint(wsEndpoint(server), "Test")
	defer provider.Close()

	respID, err := provider.Warmup(context.Background(), "You are helpful", nil)
	if err != nil {
		t.Fatalf("warmup error: %v", err)
	}
	if respID != "resp_warmup" {
		t.Fatalf("expected respID 'resp_warmup', got %q", respID)
	}
}
