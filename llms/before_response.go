package llms

import (
	"sync"

	"github.com/flitsinc/go-llms/content"
)

// BeforeResponseState allows callers to inspect and mutate outbound messages
// right before the provider request is made.
type BeforeResponseState interface {
	// Turn returns the 1-based turn number for the upcoming provider call.
	Turn() int
	// Messages returns a cloned snapshot of the current outbound messages.
	Messages() []Message
	// Prepend inserts messages at the beginning of outbound messages.
	Prepend(messages ...Message)
	// Append adds messages at the end of outbound messages.
	Append(messages ...Message)
	// Replace replaces all outbound messages with the provided slice.
	Replace(messages ...Message)
}

type beforeResponseState struct {
	mu           sync.Mutex
	turnNumber   int
	systemPrompt content.Content
	messages     []Message
	frozen       bool
}

func newBeforeResponseState(turnNumber int, systemPrompt content.Content, messages []Message) *beforeResponseState {
	return &beforeResponseState{
		turnNumber:   turnNumber,
		systemPrompt: cloneContent(systemPrompt),
		messages:     cloneMessages(messages),
	}
}

func (s *beforeResponseState) Turn() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.turnNumber
}

func (s *beforeResponseState) Messages() []Message {
	s.mu.Lock()
	defer s.mu.Unlock()
	return cloneMessages(s.messages)
}

func (s *beforeResponseState) Prepend(messages ...Message) {
	if len(messages) == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.frozen {
		return
	}
	next := cloneMessages(messages)
	s.messages = append(next, s.messages...)
}

func (s *beforeResponseState) Append(messages ...Message) {
	if len(messages) == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.frozen {
		return
	}
	s.messages = append(s.messages, cloneMessages(messages)...)
}

func (s *beforeResponseState) Replace(messages ...Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.frozen {
		return
	}
	s.messages = cloneMessages(messages)
}

func (s *beforeResponseState) freeze() (content.Content, []Message) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.frozen = true
	return cloneContent(s.systemPrompt), cloneMessages(s.messages)
}

func cloneMessages(messages []Message) []Message {
	if len(messages) == 0 {
		return nil
	}
	out := make([]Message, len(messages))
	for i := range messages {
		out[i] = cloneMessage(messages[i])
	}
	return out
}

func cloneMessage(m Message) Message {
	clone := m
	clone.Content = cloneContent(m.Content)
	if len(m.ToolCalls) > 0 {
		clone.ToolCalls = make([]ToolCall, len(m.ToolCalls))
		for i := range m.ToolCalls {
			tc := m.ToolCalls[i]
			clone.ToolCalls[i] = ToolCall{
				ID:        tc.ID,
				Name:      tc.Name,
				Arguments: append([]byte(nil), tc.Arguments...),
				Metadata:  cloneMetadata(tc.Metadata),
			}
		}
	} else {
		clone.ToolCalls = nil
	}
	return clone
}

func cloneContent(c content.Content) content.Content {
	if len(c) == 0 {
		return nil
	}
	out := make(content.Content, 0, len(c))
	for _, item := range c {
		switch v := item.(type) {
		case *content.Text:
			out = append(out, &content.Text{Text: v.Text})
		case *content.ImageURL:
			out = append(out, &content.ImageURL{URL: v.URL, MimeType: v.MimeType})
		case *content.JSON:
			out = append(out, &content.JSON{Data: append([]byte(nil), v.Data...)})
		case *content.Thought:
			out = append(out, &content.Thought{
				ID:        v.ID,
				Text:      v.Text,
				Encrypted: append([]byte(nil), v.Encrypted...),
				Signature: v.Signature,
				Summary:   v.Summary,
			})
		case *content.CacheHint:
			out = append(out, &content.CacheHint{Duration: v.Duration})
		default:
			out = append(out, item)
		}
	}
	return out
}
