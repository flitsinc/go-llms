package openai

import (
	"encoding/json"
	"fmt"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

// responseFormat specifies the format the model must output.
type responseFormat struct {
	Type       string                `json:"type"`                  // e.g., "text", "json_object", "json_schema"
	JSONSchema *jsonSchemaDefinition `json:"json_schema,omitempty"` // Used when type is "json_schema"
}

// jsonSchemaDefinition defines the schema details for JSON schema response format.
type jsonSchemaDefinition struct {
	Name   string             `json:"name"`             // A name for the schema
	Schema *tools.ValueSchema `json:"schema"`           // The actual JSON schema
	Strict bool               `json:"strict,omitempty"` // Whether to strictly enforce the schema
}

type Tool struct {
	Type     string               `json:"type"`
	Function tools.FunctionSchema `json:"function"`
}

type imageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type contentPart struct {
	Type     string    `json:"type"`
	Text     *string   `json:"text,omitempty"`
	ImageURL *imageURL `json:"image_url,omitempty"`
}

type contentList []contentPart

func convertContent(c content.Content) contentList {
	cl := make(contentList, 0, len(c))
	for _, item := range c {
		var cp contentPart
		switch v := item.(type) {
		case *content.Text:
			cp.Type = "text"
			text := v.Text
			cp.Text = &text
		case *content.ImageURL:
			cp.Type = "image_url"
			cp.ImageURL = &imageURL{
				URL:    v.URL,
				Detail: "auto",
			}
		case *content.JSON:
			cp.Type = "text"
			text := string(v.Data)
			cp.Text = &text
		case *content.Thought:
			// OpenAI does not expect thinking tokens as input; ignore.
		case *content.CacheHint:
			// OpenAI has implicit caching; ignore.
		default:
			panic(fmt.Sprintf("unhandled content item type %T", item))
		}
		cl = append(cl, cp)
	}
	return cl
}

func (cl contentList) MarshalJSON() ([]byte, error) {
	if len(cl) == 1 && cl[0].Type == "text" && cl[0].Text != nil {
		return json.Marshal(*cl[0].Text)
	}
	return json.Marshal([]contentPart(cl))
}

func (cl *contentList) UnmarshalJSON(data []byte) error {
	var text string
	if err := json.Unmarshal(data, &text); err == nil {
		*cl = contentList{{Type: "text", Text: &text}}
		return nil
	}
	var value []contentPart
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	*cl = contentList(value)
	return nil
}

type message struct {
	Role       string      `json:"role"`
	Content    contentList `json:"content"`
	ToolCalls  []toolCall  `json:"tool_calls,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
}

// messagesFromLLM converts an llms.Message to the OpenAI API message format.
// It may return multiple messages if the input is a tool result with auxiliary content.
func messagesFromLLM(m llms.Message) []message {
	if m.Role == "tool" {
		var messagesToReturn []message
		var primaryResultString string
		var secondaryContent content.Content

		if len(m.Content) > 0 {
			firstItem := m.Content[0]
			switch v := firstItem.(type) {
			case *content.Text:
				primaryResultString = v.Text
			case *content.JSON:
				primaryResultString = string(v.Data)
			case *content.ImageURL:
				primaryResultString = v.URL
			default:
				primaryResultString = ""
			}

			if len(m.Content) > 1 {
				secondaryContent = m.Content[1:]
			}
		} else {
			primaryResultString = ""
		}

		primaryMessage := message{
			Role:       "tool",
			Content:    contentList{{Type: "text", Text: &primaryResultString}},
			ToolCallID: m.ToolCallID,
		}
		messagesToReturn = append(messagesToReturn, primaryMessage)

		if len(secondaryContent) > 0 {
			secondaryAPIContent := convertContent(secondaryContent)
			if len(secondaryAPIContent) > 0 {
				secondaryMessage := message{
					Role:    "user",
					Content: secondaryAPIContent,
				}
				messagesToReturn = append(messagesToReturn, secondaryMessage)
			}
		}
		return messagesToReturn
	}

	apiRole := m.Role
	apiContent := convertContent(m.Content)

	if len(apiContent) == 0 && len(m.ToolCalls) == 0 {
		return []message{}
	}

	msg := message{
		Role:    apiRole,
		Content: apiContent,
	}

	if m.Role == "assistant" && len(m.ToolCalls) > 0 {
		msg.ToolCalls = make([]toolCall, len(m.ToolCalls))
		for i, tc := range m.ToolCalls {
			args := string(tc.Arguments)
			if !json.Valid([]byte(args)) {
				args = "{}"
			}
			msg.ToolCalls[i] = toolCall{
				ID:   tc.ID,
				Type: "function",
				Function: toolCallFunction{
					Name:      tc.Name,
					Arguments: args,
				},
			}
		}
	}

	return []message{msg}
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
	Index    int              `json:"index"`
	ID       string           `json:"id,omitempty"`
	Type     string           `json:"type,omitempty"`
	Function toolCallFunction `json:"function,omitempty"`
}

func (t toolCallDelta) ToLLM() llms.ToolCall {
	return llms.ToolCall{
		ID:        t.ID,
		Name:      t.Function.Name,
		Arguments: json.RawMessage(t.Function.Arguments),
	}
}

type chatCompletionDelta struct {
	Role      string          `json:"role,omitempty"`
	Content   *string         `json:"content,omitempty"`
	ToolCalls []toolCallDelta `json:"tool_calls,omitempty"`
}

type chatCompletionChoice struct {
	Index        int                 `json:"index"`
	Delta        chatCompletionDelta `json:"delta"`
	FinishReason *string             `json:"finish_reason"`
	Logprobs     interface{}         `json:"logprobs"`
}

type chatCompletionChunk struct {
	ID                string                 `json:"id"`
	Object            string                 `json:"object"`
	Created           int64                  `json:"created"`
	Model             string                 `json:"model"`
	SystemFingerprint string                 `json:"system_fingerprint,omitempty"`
	Choices           []chatCompletionChoice `json:"choices"`
	Usage             *usage                 `json:"usage,omitempty"`
}

type usage struct {
	PromptTokens            int                     `json:"prompt_tokens"`
	CompletionTokens        int                     `json:"completion_tokens"`
	TotalTokens             int                     `json:"total_tokens"`
	PromptTokensDetails     promptTokensDetails     `json:"prompt_tokens_details"`
	CompletionTokensDetails completionTokensDetails `json:"completion_tokens_details"`
}

type promptTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
	AudioTokens  int `json:"audio_tokens"`
}

type completionTokensDetails struct {
	ReasoningTokens          int `json:"reasoning_tokens"`
	AudioTokens              int `json:"audio_tokens"`
	AcceptedPredictionTokens int `json:"accepted_prediction_tokens"`
	RejectedPredictionTokens int `json:"rejected_prediction_tokens"`
}
