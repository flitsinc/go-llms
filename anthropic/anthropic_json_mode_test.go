package anthropic

import (
	"context"
	"strings"
	"testing"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// This file contains tests for JSON mode in the Anthropic provider.
// With the native output_config.format API, JSON output arrives as regular
// text_delta events — no special stream-level handling is needed.
// These tests verify that text_delta streaming works correctly for JSON content.

func TestAnthropicJSONModeStream(t *testing.T) {
	t.Run("Single Text Delta", func(t *testing.T) {
		var streamContent strings.Builder
		jsonDelta := `{"foo": 123}`

		streamContent.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_start", Index: 0, ContentBlock: &contentBlock{Type: "text"}}))
		streamContent.WriteString(sseEvent(streamEvent{
			Type:  "content_block_delta",
			Index: 0,
			Delta: delta{Type: "text_delta", Text: jsonDelta},
		}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 0}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

		stream := newTestAnthropicStream(context.Background(), "claude-3-haiku", streamContent.String())

		var yielded []string
		iter := stream.Iter()
		iter(func(status llms.StreamStatus) bool {
			if status == llms.StreamStatusText {
				yielded = append(yielded, stream.Text())
			}
			return true
		})

		require.NoError(t, stream.Err())
		assert.Equal(t, []string{jsonDelta}, yielded)
		msg := stream.Message()
		require.Len(t, msg.Content, 1)
		assert.Equal(t, jsonDelta, msg.Content[0].(*content.Text).Text)
	})

	t.Run("Multi Text Delta", func(t *testing.T) {
		var streamContent strings.Builder
		delta1 := `{"foo": `
		delta2 := `123, "bar": "baz"}`
		finalJSON := delta1 + delta2

		streamContent.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_start", Index: 0, ContentBlock: &contentBlock{Type: "text"}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 0, Delta: delta{Type: "text_delta", Text: delta1}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 0, Delta: delta{Type: "text_delta", Text: delta2}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 0}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

		stream := newTestAnthropicStream(context.Background(), "claude-3-haiku", streamContent.String())

		var yielded []string
		iter := stream.Iter()
		iter(func(status llms.StreamStatus) bool {
			if status == llms.StreamStatusText {
				yielded = append(yielded, stream.Text())
			}
			return true
		})

		require.NoError(t, stream.Err())
		assert.Equal(t, []string{delta1, delta2}, yielded)
		msg := stream.Message()
		require.Len(t, msg.Content, 1)
		assert.Equal(t, finalJSON, msg.Content[0].(*content.Text).Text)
	})

	t.Run("Error Handling", func(t *testing.T) {
		var streamContent strings.Builder
		errMsg := "JSON mode error!"
		errType := "json_mode_error"

		streamContent.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
		streamContent.WriteString(sseEvent(streamEvent{
			Type: "error",
			Error: &errorInfo{
				Type:    errType,
				Message: errMsg,
			},
		}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

		stream := newTestAnthropicStream(context.Background(), "claude-3-haiku", streamContent.String())

		var yielded []string
		iter := stream.Iter()
		iter(func(status llms.StreamStatus) bool {
			if status == llms.StreamStatusText {
				yielded = append(yielded, stream.Text())
			}
			return true
		})

		require.Error(t, stream.Err())
		assert.Contains(t, stream.Err().Error(), errMsg)
		assert.Contains(t, stream.Err().Error(), errType)
		assert.Empty(t, yielded)
	})
}
