package main

import (
	"context"
	"os"
	"strings"
	"testing"
	"time"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/openai"
	"github.com/flitsinc/go-llms/openrouter"
)

// These integration tests make real API calls and verify the fix for the
// gpt-5.5-concise message ID format bug. They require OPENAI_API_KEY and
// OPENROUTER_API_KEY environment variables.

// streamToCompletion drains a provider stream, collecting text and returning
// the final message. It fails the test on stream errors.
func streamToCompletion(t *testing.T, stream llms.ProviderStream) (string, llms.Message) {
	t.Helper()
	var text strings.Builder
	for yield := range stream.Iter() {
		switch yield {
		case llms.StreamStatusText:
			text.WriteString(stream.Text())
		}
	}
	require.NoError(t, stream.Err(), "stream should complete without error")
	return text.String(), stream.Message()
}

// TestIntegration_OpenRouterToOpenAIResponses reproduces the exact production
// bug: OpenRouter (Chat Completions) → OpenAI Responses API multi-turn.
//
// Before the fix, Chat Completions set Message.ID = chunk.ID (e.g. "gen-..."),
// which the Responses API rejected with: Invalid 'input[N].id': 'gen-XXXXX'.
func TestIntegration_OpenRouterToOpenAIResponses(t *testing.T) {
	openrouterKey := os.Getenv("OPENROUTER_API_KEY")
	openaiKey := os.Getenv("OPENAI_API_KEY")
	if openrouterKey == "" || openaiKey == "" {
		t.Skip("OPENROUTER_API_KEY and OPENAI_API_KEY required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 90*time.Second)
	defer cancel()

	systemPrompt := content.FromText("You are a helpful assistant. Keep responses to one sentence.")

	// ---- Turn 1: OpenRouter (Chat Completions) ----
	orProvider := openrouter.New(openrouterKey, "anthropic/claude-sonnet-4")
	userMsg1 := llms.Message{
		Role:    "user",
		Content: content.FromText("What is 2+2? Answer in one word."),
	}

	stream1 := orProvider.Generate(ctx, systemPrompt, []llms.Message{userMsg1}, nil, nil)
	text1, assistantMsg1 := streamToCompletion(t, stream1)

	require.NotEmpty(t, text1, "OpenRouter should produce text")
	assert.Equal(t, "assistant", assistantMsg1.Role)
	assert.Empty(t, assistantMsg1.ID,
		"Chat Completions must NOT propagate chunk ID as Message.ID — got %q", assistantMsg1.ID)

	t.Logf("Turn 1 (OpenRouter): text=%q, message.ID=%q", text1, assistantMsg1.ID)

	// ---- Turn 2: OpenAI Responses API with turn 1 history ----
	// This is the step that failed in production: the "gen-..." ID from
	// OpenRouter was replayed and rejected by the Responses API.
	responsesProvider := openai.NewResponsesAPI(openaiKey, "gpt-4.1-mini")
	userMsg2 := llms.Message{
		Role:    "user",
		Content: content.FromText("Now what is 3+3? Answer in one word."),
	}
	turn2Messages := []llms.Message{userMsg1, assistantMsg1, userMsg2}

	stream2 := responsesProvider.Generate(ctx, systemPrompt, turn2Messages, nil, nil)
	text2, assistantMsg2 := streamToCompletion(t, stream2)

	require.NotEmpty(t, text2, "Responses API should produce text")
	assert.True(t, strings.HasPrefix(assistantMsg2.ID, "msg_"),
		"Responses API should assign a msg_ prefixed ID, got: %q", assistantMsg2.ID)

	t.Logf("Turn 2 (Responses API): text=%q, message.ID=%q", text2, assistantMsg2.ID)
}

// TestIntegration_OpenAIChatCompletionsMultiTurn verifies pure OpenAI Chat
// Completions multi-turn works after the fix.
func TestIntegration_OpenAIChatCompletionsMultiTurn(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	provider := openai.New(apiKey, "gpt-4.1-mini")
	systemPrompt := content.FromText("You are a helpful assistant. Keep responses to one sentence.")

	// Turn 1
	userMsg1 := llms.Message{
		Role:    "user",
		Content: content.FromText("What is the capital of France? One word."),
	}
	stream1 := provider.Generate(ctx, systemPrompt, []llms.Message{userMsg1}, nil, nil)
	text1, msg1 := streamToCompletion(t, stream1)

	require.NotEmpty(t, text1)
	assert.Empty(t, msg1.ID, "Chat Completions should not propagate chunk ID — got %q", msg1.ID)
	t.Logf("Turn 1: text=%q, message.ID=%q", text1, msg1.ID)

	// Turn 2 (multi-turn)
	userMsg2 := llms.Message{
		Role:    "user",
		Content: content.FromText("And the capital of Germany? One word."),
	}
	stream2 := provider.Generate(ctx, systemPrompt, []llms.Message{userMsg1, msg1, userMsg2}, nil, nil)
	text2, msg2 := streamToCompletion(t, stream2)

	require.NotEmpty(t, text2)
	assert.Empty(t, msg2.ID, "Chat Completions turn 2 should also have empty ID — got %q", msg2.ID)
	t.Logf("Turn 2: text=%q, message.ID=%q", text2, msg2.ID)
}

// TestIntegration_OpenAIResponsesMultiTurn verifies the Responses API path
// correctly assigns msg_ IDs across multiple turns.
func TestIntegration_OpenAIResponsesMultiTurn(t *testing.T) {
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		t.Skip("OPENAI_API_KEY required")
	}

	ctx, cancel := context.WithTimeout(context.Background(), 60*time.Second)
	defer cancel()

	provider := openai.NewResponsesAPI(apiKey, "gpt-4.1-mini")
	systemPrompt := content.FromText("You are a helpful assistant. Keep responses to one sentence.")

	// Turn 1
	userMsg1 := llms.Message{
		Role:    "user",
		Content: content.FromText("What is 5+5? Answer in one word."),
	}
	stream1 := provider.Generate(ctx, systemPrompt, []llms.Message{userMsg1}, nil, nil)
	text1, msg1 := streamToCompletion(t, stream1)

	require.NotEmpty(t, text1)
	assert.True(t, strings.HasPrefix(msg1.ID, "msg_"),
		"Responses API turn 1 should have msg_ ID, got: %q", msg1.ID)
	t.Logf("Turn 1: text=%q, message.ID=%q", text1, msg1.ID)

	// Turn 2
	userMsg2 := llms.Message{
		Role:    "user",
		Content: content.FromText("Now what is 10+10? Answer in one word."),
	}
	stream2 := provider.Generate(ctx, systemPrompt, []llms.Message{userMsg1, msg1, userMsg2}, nil, nil)
	text2, msg2 := streamToCompletion(t, stream2)

	require.NotEmpty(t, text2)
	assert.True(t, strings.HasPrefix(msg2.ID, "msg_"),
		"Responses API turn 2 should have msg_ ID, got: %q", msg2.ID)
	t.Logf("Turn 2: text=%q, message.ID=%q", text2, msg2.ID)
}
