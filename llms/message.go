package llms

import (
	"encoding/json"
	"fmt"

	"github.com/flitsinc/go-llms/content"
)

type Message struct {
	// Role can be "system", "user", "assistant", or "tool".
	Role string `json:"role"`
	// ID is an optional identifier for the message when provided by a provider API.
	ID string `json:"id,omitempty"`
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
	// ToolCallName is the function name corresponding to the ToolCallID when available.
	// This is primarily used by providers (e.g., Gemini) that require function responses to reference the function name.
	ToolCallName string `json:"tool_call_name,omitempty"`
}

// UnmarshalJSON implements the json.Unmarshaler interface for Message. It
// handles the case where the 'content' field might be a simple string instead
// of the expected array of content items.
func (m *Message) UnmarshalJSON(data []byte) error {
	// Use an alias type to avoid infinite recursion when calling json.Unmarshal
	type MessageAlias Message
	var aux struct {
		MessageAlias
		Content json.RawMessage `json:"content"`
	}

	if err := json.Unmarshal(data, &aux); err != nil {
		return err
	}

	// Assign all fields except Content
	*m = Message(aux.MessageAlias)

	// Explicitly check for JSON null
	if string(aux.Content) == "null" {
		m.Content = nil // Ensure content is nil for null input
		return nil
	}

	// Check if the raw content is a JSON string
	var contentStr string
	if err := json.Unmarshal(aux.Content, &contentStr); err == nil {
		// If it is a string, treat it as text content
		m.Content = content.FromText(contentStr)
		return nil
	}

	// If it's not null or a simple string, try unmarshalling as the standard Content array
	if err := json.Unmarshal(aux.Content, &m.Content); err != nil {
		// If it fails here, it's an actual error with the array format
		return fmt.Errorf("failed to unmarshal content field as array: %w", err)
	}

	return nil
}
