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
}

type contentList []contentItem

// MarshalJSON implements custom JSON marshaling for contentList.
// If the list contains exactly one text item, it marshals as a string.
// Otherwise, it marshals as a normal JSON array.
func (cl contentList) MarshalJSON() ([]byte, error) {
	// Marshal as regular array for all other cases
	return json.Marshal([]contentItem(cl))
}

// contentItem represents a single content block in a message
type contentItem struct {
	// Type of content. One of:
	// - "text"
	// - "image"
	// - "tool_use"
	// - "tool_result"
	// - "thinking"
	// - "redacted_thinking"
	// - "document"
	// - "server_tool_use"
	// - "web_search_tool_result"
	// - "code_execution_tool_result"
	// - "mcp_tool_use"
	// - "mcp_tool_result"
	// - "container_upload"
	Type string `json:"type"`

	// Thinking related fields:

	Thinking  string `json:"thinking,omitempty"`  // Claude's internal reasoning process (required for "thinking" type)
	Signature string `json:"signature,omitempty"` // Cryptographic signature for thinking verification (required for "thinking" type)
	Data      string `json:"data,omitempty"`      // Base64-encoded redacted thinking content (required for "redacted_thinking" type)

	// Text content.
	Text string `json:"text,omitempty"`

	// Fields for tool use by the assistant:

	ID    string          `json:"id,omitempty"`    // Unique ID for the tool_use block
	Name  string          `json:"name,omitempty"`  // Name of the tool being called
	Input json.RawMessage `json:"input,omitempty"` // Arguments passed to the tool

	// Tool results:

	ToolUseID string      `json:"tool_use_id,omitempty"` // ID of the tool_use block this result responds to
	Content   contentList `json:"content,omitempty"`     // Content of the tool result
	IsError   bool        `json:"is_error,omitempty"`    // Whether the tool result is an error

	// Cache control for content up to this point.
	CacheControl *cacheControl `json:"cache_control,omitempty"`

	// Document-specific fields:

	Context string `json:"context,omitempty"` // Additional context for document content
	Title   string `json:"title,omitempty"`   // Title of the document (max 500 chars)

	// Name of the MCP server.
	ServerName string `json:"server_name,omitempty"`

	// File identifier for container uploads.
	FileID string `json:"file_id,omitempty"`

	// Source of an image or document. Contains image data (base64/URL/file) or document data.
	Source *source `json:"source,omitempty"`

	// Citations for text content.
	Citations []citation `json:"citations,omitempty"`
}

// source represents the source of an image or document.
type source struct {
	Type      string `json:"type"`                 // "base64", "url", "file", or "content" for documents
	MediaType string `json:"media_type,omitempty"` // MIME type of the image (e.g., "image/jpeg")
	Data      string `json:"data,omitempty"`       // Base64-encoded image data
	URL       string `json:"url,omitempty"`        // URL of the image
	FileID    string `json:"file_id,omitempty"`    // File ID for uploaded files
	Content   string `json:"content,omitempty"`    // Content for document sources
}

// cacheControl represents cache control settings for a content block.
type cacheControl struct {
	Type string `json:"type"`          // Cache control type, currently only "ephemeral"
	TTL  string `json:"ttl,omitempty"` // Time-to-live: "5m" (5 minutes) or "1h" (1 hour), defaults to "5m"
}

// streamEvent represents an event in the streaming response.
type streamEvent struct {
	Type         string        `json:"type"`                    // Event type: "message_start", "content_block_start", etc.
	Index        int           `json:"index,omitempty"`         // Position of content block in the content array
	Delta        delta         `json:"delta"`                   // Used in "message_delta" and "content_block_delta" events
	Message      *messageEvent `json:"message,omitempty"`       // Used in "message_start" events
	ContentBlock *contentBlock `json:"content_block,omitempty"` // Used in "content_block_start" events
	Error        *errorInfo    `json:"error,omitempty"`         // Error information if type is "error"
	Usage        *usage        `json:"usage,omitempty"`         // Token usage updates (only "message_start" and final "message_delta" events)
}

// messageEvent contains message metadata for message_start events.
type messageEvent struct {
	ID    string `json:"id"`              // Unique message ID
	Role  string `json:"role"`            // Either "user" or "assistant"
	Usage *usage `json:"usage,omitempty"` // Token usage statistics
}

// contentBlock represents the initial state of a content block
type contentBlock struct {
	Type  string          `json:"type"`            // Type of content block: "text", "tool_use", "thinking", "redacted_thinking"
	Text  string          `json:"text,omitempty"`  // Initial text content (typically empty)
	ID    string          `json:"id,omitempty"`    // Unique ID for the content block (used for tool_use blocks)
	Name  string          `json:"name,omitempty"`  // For tool_use blocks, name of the tool being called
	Input json.RawMessage `json:"input,omitempty"` // Arguments passed to the tool
	// Fields for thinking blocks
	Thinking  string `json:"thinking,omitempty"`  // Initial thinking content (for "thinking" type)
	Signature string `json:"signature,omitempty"` // Initial signature (for "thinking" type)
	Data      string `json:"data,omitempty"`      // Base64-encoded data (for "redacted_thinking" type)
}

// delta represents incremental updates in "content_block_delta" and "message_delta" events
type delta struct {
	// content_block_delta

	Type        string `json:"type,omitempty"`         // Type of delta: "text_delta", "input_json_delta", "thinking_delta", "signature_delta"
	Text        string `json:"text,omitempty"`         // Text fragment for text content blocks
	PartialJSON string `json:"partial_json,omitempty"` // For tool_use blocks, fragments of JSON for the input field
	Thinking    string `json:"thinking,omitempty"`     // Thinking fragment for thinking content blocks
	Signature   string `json:"signature,omitempty"`    // Used in signature_delta events to verify thinking content

	// message_delta

	StopReason   string  `json:"stop_reason,omitempty"`   // Reason for stopping: "end_turn", "tool_use", etc.
	StopSequence *string `json:"stop_sequence,omitempty"` // Custom stop sequence if that caused the stop
}

type usage struct {
	InputTokens              *int           `json:"input_tokens,omitempty"`
	OutputTokens             *int           `json:"output_tokens,omitempty"`
	CacheCreationInputTokens *int           `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     *int           `json:"cache_read_input_tokens,omitempty"`
	ServerToolUse            *serverToolUse `json:"server_tool_use,omitempty"`
}

// serverToolUse represents server tool usage statistics
type serverToolUse struct {
	WebSearchRequests int `json:"web_search_requests,omitempty"`
}

// citation represents a source citation in a text content block
type citation struct {
	// Common fields for all citation types
	Type      string `json:"type"`       // Citation type: "char_location", "page_location", "content_block_location", or "web_search_result_location"
	CitedText string `json:"cited_text"` // The text being cited

	// Document citation fields (for char_location, page_location, content_block_location)
	DocumentIndex int     `json:"document_index,omitempty"` // Index of the document being cited
	DocumentTitle *string `json:"document_title,omitempty"` // Title of the document (max 255 chars)

	// Character location fields
	StartCharIndex int `json:"start_char_index,omitempty"` // Start character index (for char_location)
	EndCharIndex   int `json:"end_char_index,omitempty"`   // End character index (for char_location)

	// Page location fields
	StartPageNumber int `json:"start_page_number,omitempty"` // Start page number (for page_location)
	EndPageNumber   int `json:"end_page_number,omitempty"`   // End page number (for page_location)

	// Content block location fields
	StartBlockIndex int `json:"start_block_index,omitempty"` // Start block index (for content_block_location)
	EndBlockIndex   int `json:"end_block_index,omitempty"`   // End block index (for content_block_location)

	// Web search result location fields
	EncryptedIndex string  `json:"encrypted_index,omitempty"` // Encrypted index for web search results
	Title          *string `json:"title,omitempty"`           // Title of the web search result (max 512 chars)
	URL            string  `json:"url,omitempty"`             // URL of the web search result (max 2048 chars)
}

// errorInfo contains error details in error events
type errorInfo struct {
	Type    string `json:"type"`    // Error type (e.g., "overloaded_error")
	Message string `json:"message"` // Human-readable error message
}
