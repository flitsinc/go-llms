package openai

import (
	"context"
	"encoding/json"
	"fmt"

	"github.com/coder/websocket"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
)

// WebSocketStream implements llms.ProviderStream for WebSocket-based streaming.
type WebSocketStream struct {
	responsesEventProcessor // shared event processing
	ctx                     context.Context
	conn                    *websocket.Conn
	onDone                  func(responseID string)
}

func newWebSocketStreamError(err error) *WebSocketStream {
	return &WebSocketStream{
		responsesEventProcessor: responsesEventProcessor{err: err},
		ctx:                     context.Background(),
	}
}

func (s *WebSocketStream) Err() error            { return s.err }
func (s *WebSocketStream) Message() llms.Message { return s.message }
func (s *WebSocketStream) Text() string          { return s.lastText }

func (s *WebSocketStream) Image() (string, string) {
	return s.lastImage.URL, s.lastImage.MIME
}

func (s *WebSocketStream) ToolCall() llms.ToolCall {
	if len(s.message.ToolCalls) == 0 {
		return llms.ToolCall{}
	}
	return s.message.ToolCalls[len(s.message.ToolCalls)-1]
}

func (s *WebSocketStream) Thought() content.Thought {
	if s.lastThought != nil {
		return *s.lastThought
	}
	return content.Thought{}
}

func (s *WebSocketStream) Usage() llms.Usage {
	if s.usage == nil {
		return llms.Usage{}
	}
	return llms.Usage{
		CachedInputTokens: s.usage.InputTokensDetails.CachedTokens,
		InputTokens:       s.usage.InputTokens,
		OutputTokens:      s.usage.OutputTokens,
	}
}

func (s *WebSocketStream) Iter() func(yield func(llms.StreamStatus) bool) {
	return func(yield func(llms.StreamStatus) bool) {
		if s.err != nil {
			return
		}
		for {
			select {
			case <-s.ctx.Done():
				s.err = s.ctx.Err()
				return
			default:
			}

			_, data, err := s.conn.Read(s.ctx)
			if err != nil {
				s.err = fmt.Errorf("websocket read: %w", err)
				return
			}

			if s.debugger != nil {
				s.debugger.RawEvent(data)
			}

			var event ResponseStreamEvent
			if err := json.Unmarshal(data, &event); err != nil {
				s.err = fmt.Errorf("websocket unmarshal: %w", err)
				return
			}

			if s.processEvent(event, data, yield) {
				if s.onDone != nil && s.err == nil {
					s.onDone(s.responseID)
				}
				return
			}
		}
	}
}
