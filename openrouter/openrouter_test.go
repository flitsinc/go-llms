package openrouter

import (
	"encoding/json"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestReasoningDetailsFromContent_PreservesOrderAndMetadata(t *testing.T) {
	details := reasoningDetailsFromContent(content.Content{
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
	})

	require.Len(t, details, 3)
	require.NotNil(t, details[0].Index)
	require.NotNil(t, details[1].Index)
	require.NotNil(t, details[2].Index)

	assert.Equal(t, "reasoning.text", details[0].Type)
	assert.Equal(t, "t1", details[0].ID)
	assert.Equal(t, "first", details[0].Text)
	assert.Equal(t, "anthropic-claude-v1", details[0].Format)
	assert.Equal(t, 0, *details[0].Index)

	assert.Equal(t, "reasoning.encrypted", details[1].Type)
	assert.Equal(t, "t2", details[1].ID)
	assert.NotEmpty(t, details[1].Data)
	assert.Equal(t, "anthropic-claude-v1", details[1].Format)
	assert.Equal(t, 1, *details[1].Index)

	assert.Equal(t, "reasoning.text", details[2].Type)
	assert.Equal(t, "t3", details[2].ID)
	assert.Equal(t, "third", details[2].Text)
	assert.Equal(t, "sig-3", details[2].Signature)
	assert.Equal(t, "anthropic-claude-v1", details[2].Format)
	assert.Equal(t, 2, *details[2].Index)
}

func TestProviderGenerate_UsesOpenRouterFields(t *testing.T) {
	p := NewWithReasoning("", "anthropic/claude-sonnet-4", Reasoning{Effort: "medium"})

	payload, err := p.buildPayload(
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

	reasoning, ok := raw["reasoning"].(map[string]any)
	require.True(t, ok)
	assert.Equal(t, "medium", reasoning["effort"])

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

func TestReasoningDetailsFromContent_SummaryThoughtsBecomeReasoningSummary(t *testing.T) {
	details := reasoningDetailsFromContent(content.Content{
		&content.Thought{
			ID:       "summary_1",
			Text:     "summary text",
			Summary:  true,
			Metadata: map[string]string{"openai:reasoning_index": "3"},
		},
	})

	require.Len(t, details, 1)
	assert.Equal(t, "reasoning.summary", details[0].Type)
	assert.Equal(t, "summary_1", details[0].ID)
	assert.Equal(t, "summary text", details[0].Summary)
	require.NotNil(t, details[0].Index)
	assert.Equal(t, 3, *details[0].Index)
}
