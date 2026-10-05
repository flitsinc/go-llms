package openai

import (
	"context"
	"encoding/json"
	"errors"
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
	// resendUnchained sends the whole conversation without
	// previous_response_id. It is set when the request chained to an earlier
	// response and is used at most once, when the server no longer holds
	// that response.
	resendUnchained func() error
}

var (
	_ llms.SearchStream                   = (*WebSocketStream)(nil)
	_ llms.CompactionStream               = (*WebSocketStream)(nil)
	_ llms.ToolArgumentFinalizationStream = (*WebSocketStream)(nil)
)

func newWebSocketStreamError(err error) *WebSocketStream {
	return &WebSocketStream{
		responsesEventProcessor: responsesEventProcessor{err: err},
		ctx:                     context.Background(),
	}
}

func (s *WebSocketStream) Err() error            { return s.err }
func (s *WebSocketStream) Message() llms.Message { return s.message }
func (s *WebSocketStream) Text() string          { return s.lastText }

func (s *WebSocketStream) Audio() (string, string) { return "", "" }
func (s *WebSocketStream) Image() (string, string) {
	return s.lastImage.URL, s.lastImage.MIME
}

func (s *WebSocketStream) ToolCall() llms.ToolCall {
	if len(s.message.ToolCalls) == 0 {
		return llms.ToolCall{}
	}
	return s.message.ToolCalls[len(s.message.ToolCalls)-1]
}

// ToolArgumentFinalization returns the independent provider-final argument
// snapshot for the active function call, when the protocol expects one.
func (s *WebSocketStream) ToolArgumentFinalization() (json.RawMessage, bool) {
	if s.argumentFinalization == nil {
		return nil, false
	}
	return s.argumentFinalization.arguments, true
}

func (s *WebSocketStream) Thought() content.Thought {
	if s.lastThought != nil {
		return *s.lastThought
	}
	return content.Thought{}
}

func (s *WebSocketStream) Search() llms.SearchActivity {
	return s.lastSearch
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
				if s.retryUnchained() {
					continue
				}
				if s.onDone != nil && s.err == nil {
					s.onDone(s.responseID)
				}
				return
			}
		}
	}
}

// retryUnchained resends the request without previous_response_id when the
// server refused it because it no longer holds the chained response, and
// reports whether the stream continues with the resent request. The refusal
// arrives before the response is created, so nothing has been yielded yet.
func (s *WebSocketStream) retryUnchained() bool {
	if s.resendUnchained == nil || s.responseID != "" {
		return false
	}
	var httpErr *llms.HTTPError
	if !errors.As(s.err, &httpErr) || httpErr.ErrorCode != "previous_response_not_found" {
		return false
	}
	resend := s.resendUnchained
	s.resendUnchained = nil
	if err := resend(); err != nil {
		s.err = err
		return false
	}
	s.err = nil
	return true
}
