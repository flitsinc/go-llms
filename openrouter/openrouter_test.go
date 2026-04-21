package openrouter

import (
	"encoding/json"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestNew_BuildPayload_UsesOpenRouterEncoding(t *testing.T) {
	p := New("", "anthropic/claude-sonnet-4")

	payload, err := p.BuildPayload(
		content.Content{
			&content.Text{Text: "Cache me"},
			&content.CacheHint{Duration: "long"},
		},
		[]llms.Message{{
			Role: "assistant",
			Content: content.Content{
				&content.Thought{
					ID:       "t1",
					Text:     "first",
					Metadata: map[string]string{"openai:reasoning_format": "anthropic-claude-v1", "openai:reasoning_index": "0"},
				},
				&content.Thought{
					ID:        "t2",
					Text:      "(Redacted)",
					Encrypted: []byte("secret"),
					Summary:   true,
					Metadata:  map[string]string{"openai:reasoning_format": "anthropic-claude-v1", "openai:reasoning_index": "1"},
				},
				&content.Thought{
					ID:        "t3",
					Text:      "third",
					Signature: "sig-3",
					Metadata:  map[string]string{"openai:reasoning_format": "anthropic-claude-v1", "openai:reasoning_index": "2"},
				},
				&content.Text{Text: "Need tool output"},
			},
		}},
		nil,
		nil,
	)
	require.NoError(t, err)

	encoded, err := json.Marshal(payload)
	require.NoError(t, err)

	var raw map[string]any
	require.NoError(t, json.Unmarshal(encoded, &raw))

	_, hasPromptCacheRetention := raw["prompt_cache_retention"]
	assert.False(t, hasPromptCacheRetention)

	messages, ok := raw["messages"].([]any)
	require.True(t, ok)
	require.Len(t, messages, 2)

	systemMessage := messages[0].(map[string]any)
	systemContent := systemMessage["content"].([]any)
	firstPart := systemContent[0].(map[string]any)
	assert.Equal(t, map[string]any{"type": "ephemeral"}, firstPart["cache_control"])

	assistantMessage := messages[1].(map[string]any)
	reasoningDetails := assistantMessage["reasoning_details"].([]any)
	require.Len(t, reasoningDetails, 3)

	first := reasoningDetails[0].(map[string]any)
	assert.Equal(t, "reasoning.text", first["type"])
	assert.Equal(t, "first", first["text"])
	assert.Equal(t, "t1", first["id"])

	second := reasoningDetails[1].(map[string]any)
	assert.Equal(t, "reasoning.encrypted", second["type"])
	assert.Equal(t, "t2", second["id"])
	assert.NotEmpty(t, second["data"])

	third := reasoningDetails[2].(map[string]any)
	assert.Equal(t, "reasoning.text", third["type"])
	assert.Equal(t, "third", third["text"])
	assert.Equal(t, "sig-3", third["signature"])
	assert.Equal(t, "t3", third["id"])
}

func TestNewWithReasoning_AddsReasoningPayload(t *testing.T) {
	p := NewWithReasoning("", "anthropic/claude-sonnet-4", Reasoning{Effort: "medium"})

	payload, err := p.BuildPayload(nil, nil, nil, nil)
	require.NoError(t, err)

	encoded, err := json.Marshal(payload)
	require.NoError(t, err)

	var raw map[string]any
	require.NoError(t, json.Unmarshal(encoded, &raw))

	reasoning, ok := raw["reasoning"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "medium", reasoning["effort"])
}
