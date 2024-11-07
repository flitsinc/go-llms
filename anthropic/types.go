package anthropic

import (
	"encoding/json"

	"github.com/blixt/go-llms/tools"
)

type Tool struct {
	Name        string            `json:"name"`
	Description string            `json:"description"`
	InputSchema tools.ValueSchema `json:"input_schema"`
}

type message struct {
	Role    string      `json:"role"`
	Content contentList `json:"content"`
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

type contentItem struct {
	Type  string `json:"type"`
	Text  string `json:"text,omitempty"`
	Image *image `json:"image,omitempty"`

	// Tool use from assistant messages.

	ID    string          `json:"id,omitempty"`
	Name  string          `json:"name,omitempty"`
	Input json.RawMessage `json:"input,omitempty"`

	// Tool results.

	ToolUseID string      `json:"tool_use_id,omitempty"`
	Content   contentList `json:"content,omitempty"`
}

type image struct {
	Type   string `json:"type"`
	Source source `json:"source"`
}

type source struct {
	Type      string `json:"type"`
	MediaType string `json:"media_type"`
	Data      string `json:"data"`
}

type streamEvent struct {
	Type         string        `json:"type"`
	Message      *messageEvent `json:"message,omitempty"`
	Index        int           `json:"index,omitempty"`
	ContentBlock *contentBlock `json:"content_block,omitempty"`
	Delta        delta         `json:"delta,omitempty"`
}

type messageEvent struct {
	ID    string `json:"id"`
	Role  string `json:"role"`
	Usage *usage `json:"usage,omitempty"`
}

type contentBlock struct {
	Type string `json:"type"`
	Text string `json:"text,omitempty"`
	ID   string `json:"id,omitempty"`
	Name string `json:"name,omitempty"`
}

type delta struct {
	Type        string `json:"type"`
	PartialJSON string `json:"partial_json,omitempty"`
	Text        string `json:"text,omitempty"`
	Usage       *usage `json:"usage,omitempty"`
	StopReason  string `json:"stop_reason,omitempty"`
}

type usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}
