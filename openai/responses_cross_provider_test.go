package openai

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

type crossProviderReadParams struct {
	ModuleName string `json:"module_name" description:"Module name"`
}

func crossProviderReplayMessages() []llms.Message {
	return []llms.Message{
		{Role: "user", Content: content.FromText("Inspect the current module.")},
		{
			Role: "assistant",
			Content: content.Content{
				&content.Thought{Text: "I need to inspect the module first.", Signature: "foreign-provider-signature"},
				&content.Text{Text: "I'll inspect the module first."},
			},
			ToolCalls: []llms.ToolCall{
				{
					ID:        "toolu_01foreign",
					Name:      "readModule",
					Arguments: json.RawMessage(`{"module_name":"Main"}`),
					Metadata:  map[string]string{"openai:item_type": "function"},
				},
			},
		},
		{
			Role:       "tool",
			ToolCallID: "toolu_01foreign",
			Content:    content.FromText(`{"code":"export default function Main() {}"}`),
		},
		{Role: "user", Content: content.FromText("Reply with OK.")},
	}
}

func crossProviderToolbox() *tools.Toolbox {
	readModule := tools.Func(
		"Read Module",
		"Read a module",
		"readModule",
		func(r tools.Runner, p crossProviderReadParams) tools.Result {
			return tools.Success(map[string]any{"code": "export default function Main() {}"})
		},
	)
	return tools.Box(readModule)
}

func TestResponsesAPI_CrossProviderToolCallReplay(t *testing.T) {
	var payload map[string]any
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, err := io.ReadAll(r.Body)
		if err != nil {
			t.Fatalf("read request: %v", err)
		}
		if err := json.Unmarshal(body, &payload); err != nil {
			t.Fatalf("decode request: %v", err)
		}
		w.Header().Set("Content-Type", "text/event-stream")
		_, _ = io.WriteString(w, "data: {\"type\":\"response.completed\"}\n\n")
	}))
	defer server.Close()

	stream := NewResponsesAPI("test-key", "gpt-5.6-luna").
		WithEndpoint(server.URL, "OpenAI").
		Generate(
			context.Background(),
			content.FromText("You are a concise assistant."),
			crossProviderReplayMessages(),
			crossProviderToolbox(),
			nil,
		)
	for range stream.Iter() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("cross-provider replay failed: %v", err)
	}

	inputs, ok := payload["input"].([]any)
	if !ok {
		t.Fatalf("expected input array, got %T", payload["input"])
	}
	var functionCall map[string]any
	var functionOutput map[string]any
	for _, input := range inputs {
		item, ok := input.(map[string]any)
		if !ok {
			continue
		}
		switch item["type"] {
		case "function_call":
			functionCall = item
		case "function_call_output":
			functionOutput = item
		}
	}
	if functionCall == nil || functionOutput == nil {
		t.Fatalf("expected function call and output in payload: %#v", inputs)
	}
	if _, exists := functionCall["id"]; exists {
		t.Fatalf("foreign function call must not fabricate a Responses item ID: %#v", functionCall)
	}
	if functionCall["call_id"] != "toolu_01foreign" {
		t.Fatalf("unexpected function call ID: %#v", functionCall)
	}
	if functionOutput["call_id"] != "toolu_01foreign" {
		t.Fatalf("tool output lost call linkage: %#v", functionOutput)
	}
}

func TestResponsesAPI_CrossProviderToolCallReplayLive(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY is not set")
	}

	stream := NewResponsesAPI(apiKey, "gpt-5.6-luna").
		WithMaxOutputTokens(128).
		WithStore(false).
		Generate(
			context.Background(),
			content.FromText("Reply with exactly OK."),
			crossProviderReplayMessages(),
			crossProviderToolbox(),
			nil,
		)
	for range stream.Iter() {
	}
	if err := stream.Err(); err != nil {
		t.Fatalf("live GPT-5.6 cross-provider replay failed: %v", err)
	}
}
