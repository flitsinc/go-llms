package anthropic

import (
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

func TestToolResultCacheControl(t *testing.T) {
	for _, tc := range []struct {
		name       string
		content    content.Content
		wantTTL    string
		wantBlocks int
	}{
		{
			name:       "text",
			content:    content.Content{&content.Text{Text: "result"}, &content.CacheHint{Duration: "short"}},
			wantTTL:    "5m",
			wantBlocks: 1,
		},
		{
			name: "mixed content with multiple hints",
			content: content.Content{
				&content.JSON{Data: []byte(`{"ok":true}`)},
				&content.CacheHint{Duration: "short"},
				&content.ImageURL{URL: "data:image/png;base64,aW1hZ2U="},
				&content.CacheHint{Duration: "long"},
			},
			wantTTL:    "1h",
			wantBlocks: 2,
		},
		{
			name:    "empty text",
			content: content.Content{&content.Text{Text: " "}, &content.CacheHint{Duration: "short"}},
			wantTTL: "5m",
		},
		{
			name:    "hint only",
			content: content.Content{&content.CacheHint{Duration: "long"}},
			wantTTL: "1h",
		},
	} {
		t.Run(tc.name, func(t *testing.T) {
			original := append(content.Content(nil), tc.content...)
			msg, err := messageFromLLM(llms.Message{Role: "tool", ToolCallID: "tool_1", Content: tc.content, IsError: true})
			require.NoError(t, err)
			assert.Equal(t, "user", msg.Role)
			require.Len(t, msg.Content, 1)
			result := msg.Content[0]
			assert.Equal(t, "tool_result", result.Type)
			assert.Equal(t, "tool_1", result.ToolUseID)
			assert.True(t, result.IsError)
			assert.Equal(t, &cacheControl{Type: "ephemeral", TTL: tc.wantTTL}, result.CacheControl)
			require.Len(t, result.Content, tc.wantBlocks)
			for _, block := range result.Content {
				assert.Nil(t, block.CacheControl, "tool result sub-content is not cacheable")
			}
			assert.Equal(t, original, tc.content, "serialization must not change the caller's history")
		})
	}
}

func TestToolResultWithoutCacheHint(t *testing.T) {
	msg, err := messageFromLLM(llms.Message{Role: "tool", ToolCallID: "tool_1", Content: content.FromText("result")})
	require.NoError(t, err)
	require.Len(t, msg.Content, 1)
	assert.Nil(t, msg.Content[0].CacheControl)
	require.Len(t, msg.Content[0].Content, 1)
	assert.Equal(t, "result", msg.Content[0].Content[0].Text)
}

func TestUserCacheHintsRemainOnIndividualBlocks(t *testing.T) {
	msg, err := messageFromLLM(llms.Message{Role: "user", Content: content.Content{
		&content.Text{Text: "static"}, &content.CacheHint{Duration: "long"},
		&content.Text{Text: "dynamic"}, &content.CacheHint{Duration: "short"},
	}})
	require.NoError(t, err)
	require.Len(t, msg.Content, 2)
	assert.Equal(t, &cacheControl{Type: "ephemeral", TTL: "1h"}, msg.Content[0].CacheControl)
	assert.Equal(t, &cacheControl{Type: "ephemeral", TTL: "5m"}, msg.Content[1].CacheControl)
}
