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
	// ToolCalls represents the list of tools that an assistant message is invoking.
	// This field is used when the message is from an assistant (Role="assistant") that is calling tools.
	// Each ToolCall contains an ID, name of the tool being called, and arguments to pass to the tool.
	ToolCalls []ToolCall `json:"tool_calls,omitempty"`
	// ToolCallID identifies which tool call a message is responding to.
	// This field is used when the message is a tool response (Role="tool") that is responding to a previous tool call.
	// It should match the ID of the original ToolCall that this message is responding to.
	ToolCallID string `json:"tool_call_id,omitempty"`
}
