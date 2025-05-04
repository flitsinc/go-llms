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

// This file contains tests for the simulated JSON mode in the Anthropic provider.
// It focuses on the special handling of JSON mode tool blocks in the streaming API.

func TestAnthropicJSONModeStream(t *testing.T) {
	t.Run("Single Delta JSON Mode", func(t *testing.T) {
		var streamContent strings.Builder
		jsonDelta := `{"foo": 123}`

		streamContent.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
		streamContent.WriteString(sseEvent(streamEvent{
			Type:  "content_block_start",
			Index: 0,
			ContentBlock: &contentBlock{
				Type:  "tool_use",
				Name:  jsonModeToolName,
				ID:    "json_01",
				Input: nil, // Should be ignored in JSON mode
			},
		}))
		streamContent.WriteString(sseEvent(streamEvent{
			Type:  "content_block_delta",
			Index: 0,
			Delta: delta{
				Type:        "input_json_delta",
				PartialJSON: jsonDelta,
			},
		}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 0}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

		stream := newTestAnthropicStream(context.Background(), "claude-3-haiku", streamContent.String())
		stream.isJSONMode = true // Simulate JSON mode

		var yielded []string
		iter := stream.Iter()
		iter(func(status llms.StreamStatus) bool {
			if status == llms.StreamStatusText {
				yielded = append(yielded, stream.Text())
			}
			return true
		})

		require.NoError(t, stream.Err())
		assert.Equal(t, []string{jsonDelta}, yielded, "Should yield the JSON delta as text in JSON mode")
		// The message content should accumulate the JSON as text
		msg := stream.Message()
		require.Len(t, msg.Content, 1)
		assert.Equal(t, jsonDelta, msg.Content[0].(*content.Text).Text)
	})

	t.Run("Multi Delta JSON Mode", func(t *testing.T) {
		var streamContent strings.Builder
		delta1 := `{"foo": `
		delta2 := `123, "bar": "baz"}`
		finalJSON := delta1 + delta2

		streamContent.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
		streamContent.WriteString(sseEvent(streamEvent{
			Type:  "content_block_start",
			Index: 0,
			ContentBlock: &contentBlock{
				Type:  "tool_use",
				Name:  jsonModeToolName,
				ID:    "json_02",
				Input: nil,
			},
		}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 0, Delta: delta{Type: "input_json_delta", PartialJSON: delta1}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 0, Delta: delta{Type: "input_json_delta", PartialJSON: delta2}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 0}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

		stream := newTestAnthropicStream(context.Background(), "claude-3-haiku", streamContent.String())
		stream.isJSONMode = true

		var yielded []string
		iter := stream.Iter()
		iter(func(status llms.StreamStatus) bool {
			if status == llms.StreamStatusText {
				yielded = append(yielded, stream.Text())
			}
			return true
		})

		require.NoError(t, stream.Err())
		assert.Equal(t, []string{delta1, delta2}, yielded, "Should yield each JSON delta as text in JSON mode")
		msg := stream.Message()
		require.Len(t, msg.Content, 1)
		assert.Equal(t, finalJSON, msg.Content[0].(*content.Text).Text)
	})

	t.Run("Text and JSON Mode Interleaved", func(t *testing.T) {
		var streamContent strings.Builder
		text1 := "Here is your data: "
		jsonDelta := `{"foo": 1}`

		streamContent.WriteString(sseEvent(streamEvent{Type: "message_start", Message: &messageEvent{Role: "assistant"}}))
		// Text block
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_start", Index: 0, ContentBlock: &contentBlock{Type: "text"}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 0, Delta: delta{Type: "text_delta", Text: text1}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 0}))
		// JSON mode tool block
		streamContent.WriteString(sseEvent(streamEvent{
			Type:  "content_block_start",
			Index: 1,
			ContentBlock: &contentBlock{
				Type:  "tool_use",
				Name:  jsonModeToolName,
				ID:    "json_03",
				Input: nil,
			},
		}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_delta", Index: 1, Delta: delta{Type: "input_json_delta", PartialJSON: jsonDelta}}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "content_block_stop", Index: 1}))
		streamContent.WriteString(sseEvent(streamEvent{Type: "message_stop"}))

		stream := newTestAnthropicStream(context.Background(), "claude-3-haiku", streamContent.String())
		stream.isJSONMode = true

		var yielded []string
		iter := stream.Iter()
		iter(func(status llms.StreamStatus) bool {
			if status == llms.StreamStatusText {
				yielded = append(yielded, stream.Text())
			}
			return true
		})

		require.NoError(t, stream.Err())
		assert.Equal(t, []string{text1, jsonDelta}, yielded, "Should yield text delta and then JSON delta")
		msg := stream.Message()
		require.Len(t, msg.Content, 1, "Expected merged content due to current Append behavior")
		assert.Equal(t, text1+jsonDelta, msg.Content[0].(*content.Text).Text, "Expected merged content string")
	})

	t.Run("JSON Mode Error Handling", func(t *testing.T) {
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
		stream.isJSONMode = true

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
		assert.Empty(t, yielded, "No text should be yielded if error occurs before any content")
	})
}
