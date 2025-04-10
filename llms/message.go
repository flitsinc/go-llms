package llms

import (
	"github.com/blixt/go-llms/content"
)

type Message struct {
	// Role can be "system", "user", "assistant", or "tool".
	Role string `json:"role"`
	// Name can be used to identify different identities within the same role.
	Name string `json:"name,omitempty"`
	// Content is the message content.
	Content content.Content `json:"content"`
	// ToolCalls is the list of tool calls that this message is part of.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID is the ID of the tool call that this message is part of.
	ToolCallID string `json:"tool_call_id,omitempty"`
}
