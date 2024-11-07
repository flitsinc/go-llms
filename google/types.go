package google

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/blixt/go-llms/content"
	"github.com/blixt/go-llms/llms"
)

type errorResponse struct {
	Error struct {
		Code    int    `json:"code"`
		Message string `json:"message"`
		Status  string `json:"status"`
	} `json:"error"`
}

type inlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"`
}

type fileData struct {
	MimeType string `json:"mimeType"`
	FileURI  string `json:"fileUri"`
}

type functionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args,omitempty"`
}

type functionResponse struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type videoOffset struct {
	Seconds int `json:"seconds"`
	Nanos   int `json:"nanos"`
}

type videoMetadata struct {
	StartOffset videoOffset `json:"startOffset"`
	EndOffset   videoOffset `json:"endOffset"`
}

type part struct {
	Text             *string           `json:"text,omitempty"`
	InlineData       *inlineData       `json:"inlineData,omitempty"`
	FileData         *fileData         `json:"fileData,omitempty"`
	FunctionCall     *functionCall     `json:"functionCall,omitempty"`
	FunctionResponse *functionResponse `json:"functionResponse,omitempty"`
	VideoMetadata    *videoMetadata    `json:"videoMetadata,omitempty"`
}

type parts []part

func createFunctionResponse(name string, c content.Content) *functionResponse {
	if len(c) == 1 {
		if jc, ok := c[0].(*content.JSON); ok {
			return &functionResponse{Name: name, Response: jc.Data}
		}
	}
	var response []any
	for _, item := range c {
		switch v := item.(type) {
		case *content.JSON:
			response = append(response, v.Data)
		case *content.Text:
			response = append(response, v.Text)
		default:
			panic(fmt.Sprintf("unhandled content item type %T", item))
		}
	}
	data, err := json.Marshal(response)
	if err != nil {
		// TODO: Return a normal error.
		panic(fmt.Sprintf("failed to marshal function response: %v", err))
	}
	return &functionResponse{Name: name, Response: data}
}

func convertContent(c content.Content) (p parts) {
	for _, item := range c {
		var pp part
		switch v := item.(type) {
		case *content.Text:
			text := v.Text
			pp.Text = &text
		case *content.ImageURL:
			if dataValue, found := strings.CutPrefix(v.URL, "data:"); found {
				mimeType, data, found := strings.Cut(dataValue, ";base64,")
				if !found {
					panic(fmt.Sprintf("unsupported data URI format %q", v.URL))
				}
				pp.InlineData = &inlineData{mimeType, data}
			} else {
				// TODO: We are missing MIME type here.
				pp.FileData = &fileData{FileURI: v.URL}
			}
		case *content.JSON:
			text := string(v.Data)
			pp.Text = &text
		default:
			panic(fmt.Sprintf("unhandled content item type %T", item))
		}
		p = append(p, pp)
	}
	return p
}

func (p parts) MarshalJSON() ([]byte, error) {
	// If there's just one part, don't wrap it in an array.
	if len(p) == 1 {
		return json.Marshal(p[0])
	}
	// Otherwise, directly marshal the parts slice.
	return json.Marshal([]part(p))
}

func (p *parts) UnmarshalJSON(data []byte) error {
	// Try to unmarshal data as a single part first.
	var pp part
	if err := json.Unmarshal(data, &pp); err == nil {
		*p = parts{pp}
		return nil
	}
	// If that failed, unmarshal it as an array of parts.
	var value []part
	if err := json.Unmarshal(data, &value); err != nil {
		return err
	}
	*p = parts(value)
	return nil
}

type message struct {
	Role  string `json:"role"`
	Parts parts  `json:"parts"`
}

func messageFromLLM(m llms.Message) message {
	var role string
	switch m.Role {
	case "assistant":
		role = "model"
	case "tool":
		role = "user"
	default:
		role = m.Role
	}
	if m.ToolCallID != "" {
		return message{
			Role: role,
			Parts: parts{
				{FunctionResponse: createFunctionResponse(m.ToolCallID, m.Content)},
			},
		}
	}
	return message{
		Role:  role,
		Parts: convertContent(m.Content),
	}
}

type usageMetadata struct {
	PromptTokenCount     int `json:"promptTokenCount"`
	CandidatesTokenCount int `json:"candidatesTokenCount"`
	TotalTokenCount      int `json:"totalTokenCount"`
}

type streamingResponse struct {
	Candidates    []candidate    `json:"candidates"`
	UsageMetadata *usageMetadata `json:"usageMetadata,omitempty"`
}

type candidate struct {
	Content       candidateContent `json:"content"`
	SafetyRatings []safetyRating   `json:"safetyRatings,omitempty"`
	FinishReason  string           `json:"finishReason,omitempty"`
}

type candidateContent struct {
	Role  string `json:"role"`
	Parts parts  `json:"parts"`
}

type safetyRating struct {
	Category         string  `json:"category"`
	Probability      string  `json:"probability"`
	ProbabilityScore float64 `json:"probabilityScore"`
	Severity         string  `json:"severity"`
	SeverityScore    float64 `json:"severityScore"`
}
