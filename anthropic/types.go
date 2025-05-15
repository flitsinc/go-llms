package anthropic

import (
	"encoding/json"

	"github.com/flitsinc/go-llms/tools"
)

type Tool struct {
	Name        string            `json:"name"`
	Description string            `json:"description"`
	InputSchema tools.ValueSchema `json:"input_schema"`
}

type message struct {
	Role    string      `json:"role"`    // Either "user" or "assistant"
	Content contentList `json:"content"` // The content of the message (text, images, tool_use blocks)
	ID      string      `json:"id,omitempty"`
	Usage   *usage      `json:"usage,omitempty"` // Token usage statistics
}

type contentList []contentItem

// MarshalJSON implements custom JSON marshaling for contentList.
// If the list contains exactly one text item, it marshals as a string.
// Otherwise, it marshals as a normal JSON array.
func (cl contentList) MarshalJSON() ([]byte, error) {
	if len(cl) == 1 && cl[0].Type == "text" {
		return json.Marshal(cl[0].Text)
	}
	// Marshal as regular array for all other cases
	return json.Marshal([]contentItem(cl))
}

// contentItem represents a single content block in a message
type contentItem struct {
	Type string `json:"type"` // Type of content: "text", "image", "tool_use", "tool_result", "thinking"
	Text string `json:"text,omitempty"`

	// Source of an image.
	Source *source `json:"source,omitempty"` // Contains image data in base64 format

	// Tool use from assistant messages.

	ID    string          `json:"id,omitempty"`    // Unique ID for the tool_use block
	Name  string          `json:"name,omitempty"`  // Name of the tool being called
	Input json.RawMessage `json:"input,omitempty"` // Arguments passed to the tool

	// Tool results.

	ToolUseID string      `json:"tool_use_id,omitempty"` // ID of the tool_use block this result responds to
	Content   contentList `json:"content,omitempty"`     // Content of the tool result

	// Citations supporting the text block

	Citations []citation `json:"citations,omitempty"` // Source citations for the text

	// Thinking content from extended thinking feature

	Thinking  string `json:"thinking,omitempty"`  // Claude's internal reasoning process
	Signature string `json:"signature,omitempty"` // Cryptographic signature for thinking verification

	// Redacted thinking content

	Data string `json:"data,omitempty"` // Used for redacted thinking content
}

// source represents the source of an image
type source struct {
	Type      string `json:"type"`       // Always "base64" for Anthropic
	MediaType string `json:"media_type"` // MIME type of the image (e.g., "image/jpeg")
	Data      string `json:"data"`       // Base64-encoded image data
}

// streamEvent represents an event in the streaming response
type streamEvent struct {
	Type         string        `json:"type"`                    // Event type: "message_start", "content_block_start", etc.
	Message      *messageEvent `json:"message,omitempty"`       // Used in message_start events
	Index        int           `json:"index,omitempty"`         // Position of content block in the content array
	ContentBlock *contentBlock `json:"content_block,omitempty"` // Used in content_block_start events
	Delta        delta         `json:"delta,omitempty"`         // Used in content_block_delta events
	Error        *errorInfo    `json:"error,omitempty"`         // Error information if type is "error"
}

// messageEvent contains message metadata for message_start events
type messageEvent struct {
	ID    string `json:"id"`              // Unique message ID
	Role  string `json:"role"`            // Either "user" or "assistant"
	Usage *usage `json:"usage,omitempty"` // Token usage statistics
}

// contentBlock represents the initial state of a content block
type contentBlock struct {
	Type  string          `json:"type"`            // Type of content block: "text", "tool_use", "thinking"
	Text  string          `json:"text,omitempty"`  // Initial text content (typically empty)
	ID    string          `json:"id,omitempty"`    // Unique ID for the content block (used for tool_use blocks)
	Name  string          `json:"name,omitempty"`  // For tool_use blocks, name of the tool being called
	Input json.RawMessage `json:"input,omitempty"` // Arguments passed to the tool
	// Fields for thinking blocks
	Thinking  string `json:"thinking,omitempty"`  // Initial thinking content if provided directly
	Signature string `json:"signature,omitempty"` // Initial signature if provided directly
	Data      string `json:"data,omitempty"`      // For redacted_thinking, the base64 encoded data
}

// delta represents incremental updates in content_block_delta events
type delta struct {
	Type         string `json:"type"`                    // Type of delta: "text_delta", "input_json_delta", "thinking_delta", etc.
	PartialJSON  string `json:"partial_json,omitempty"`  // For tool_use blocks, fragments of JSON for the input field
	Text         string `json:"text,omitempty"`          // Text fragment for text content blocks
	Thinking     string `json:"thinking,omitempty"`      // Thinking fragment for thinking content blocks
	Signature    string `json:"signature,omitempty"`     // Used in signature_delta events to verify thinking content
	Usage        *usage `json:"usage,omitempty"`         // Token usage updates
	StopReason   string `json:"stop_reason,omitempty"`   // Reason for stopping: "end_turn", "tool_use", etc.
	StopSequence string `json:"stop_sequence,omitempty"` // Custom stop sequence if that caused the stop
}

type usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// citation represents a source citation in a text content block
type citation struct {
	// The type of citation will depend on the type of document being cited
	PageLocation         *pageLocation         `json:"page_location,omitempty"`          // For PDF document citations
	CharLocation         *charLocation         `json:"char_location,omitempty"`          // For text document citations
	ContentBlockLocation *contentBlockLocation `json:"content_block_location,omitempty"` // For citations to other content blocks
}

// pageLocation identifies a location in a PDF document
type pageLocation struct {
	PageNumber int `json:"page_number"` // Page number in the PDF
}

// charLocation identifies a span of characters in a text document
type charLocation struct {
	StartChar int `json:"start_char"` // Start character index
	EndChar   int `json:"end_char"`   // End character index
}

// contentBlockLocation identifies a content block
type contentBlockLocation struct {
	BlockID string `json:"block_id"` // ID of the content block being cited
}

// errorInfo contains error details in error events
type errorInfo struct {
	Type    string `json:"type"`    // Error type (e.g., "overloaded_error")
	Message string `json:"message"` // Human-readable error message
}
