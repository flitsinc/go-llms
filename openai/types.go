package openai

import (
	"encoding/json"
	"fmt"

	"github.com/blixt/go-llms/content"
	"github.com/blixt/go-llms/llms"
	"github.com/blixt/go-llms/tools"
)

type Tool struct {
	Type     string               `json:"type"`
	Function tools.FunctionSchema `json:"function"`
}

type imageURL struct {
	URL string `json:"url"`
}

type contentItem struct {
	Type     string    `json:"type"`
	Text     *string   `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type contentList []contentItem

func convertContent(c content.Content) (cl contentList) {
	for _, item := range c {
		var ci contentItem
		switch v := item.(type) {
		case *content.Text:
			ci.Type = "text"
			text := v.Text
			ci.Text = &text
		case *content.ImageURL:
			ci.Type = "image_url"
			ci.ImageURL = &imageURL{URL: v.URL}
		case *content.JSON:
			ci.Type = "text"
			text := string(v.Data)
			ci.Text = &text
		default:
			panic(fmt.Sprintf("unhandled content item type %T", item))
		}
		cl = append(cl, ci)
	}
	return cl
}

func (cl contentList) MarshalJSON() ([]byte, error) {
	// Marshal into a simple string when the only content is one text item.
	if len(cl) == 1 && cl[0].Type == "text" {
		return json.Marshal(cl[0].Text)
	}
	// Otherwise, directly marshal the content slice.
	return json.Marshal([]contentItem(cl))
}

func (cl *contentList) UnmarshalJSON(data []byte) error {
	// Try to unmarshal data as a JSON string first.
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		*cl = contentList{
			{
				Type: "text",
				Text: &text,
			},
		}
		return nil
	}
	// If that failed, unmarshal it as an array of content items.
	var value []contentItem
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	*cl = contentList(value)
	return nil
}

type message struct {
	// Role can be "system", "user", "assistant", or "tool".
	Role string `json:"role"`
	// Name can be used to identify different identities within the same role.
	Name string `json:"name,omitempty"`
	// Content is the message content.
	Content contentList `json:"content"`
	// ToolCalls is the list of tool calls that this message is part of.
	ToolCalls []toolCall `json:"tool_calls,omitempty"`
	// ToolCallID is the ID of the tool call that this message is part of.
	ToolCallID string `json:"tool_call_id,omitempty"`
}

func messageFromLLM(m llms.Message) message {
	toolCalls := make([]toolCall, len(m.ToolCalls))
	for i, tc := range m.ToolCalls {
		toolCalls[i] = toolCall{
			ID:       tc.ID,
			Type:     "function",
			Function: toolCallFunction{Name: tc.Name, Arguments: string(tc.Arguments)},
		}
	}
	return message{
		Role:       m.Role,
		Name:       m.Name,
		Content:    convertContent(m.Content),
		ToolCalls:  toolCalls,
		ToolCallID: m.ToolCallID,
	}
}

type toolCallFunction struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

type toolCall struct {
	ID       string           `json:"id"`
	Type     string           `json:"type"`
	Function toolCallFunction `json:"function"`
}

func (t toolCall) ToLLM() llms.ToolCall {
	return llms.ToolCall{
		ID:        t.ID,
		Name:      t.Function.Name,
		Arguments: json.RawMessage(t.Function.Arguments),
	}
}

type toolCallDelta struct {
	toolCall
	Index int `json:"index"`
}

type chatCompletionDelta struct {
	Role      string          `json:"role"`
	Content   string          `json:"content"`
	ToolCalls []toolCallDelta `json:"tool_calls"`
}

type chatCompletionChoice struct {
	Index        int                 `json:"index"`
	Delta        chatCompletionDelta `json:"delta"`
	FinishReason string              `json:"finish_reason"`
}

type chatCompletionChunk struct {
	ID                string                 `json:"id"`
	Object            string                 `json:"object"`
	Created           int64                  `json:"created"`
	Model             string                 `json:"model"`
	SystemFingerprint string                 `json:"system_fingerprint"`
	Choices           []chatCompletionChoice `json:"choices"`
	Usage             *usage                 `json:"usage"`
}

type usage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	TotalTokens      int `json:"total_tokens"`
}
