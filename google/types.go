package google

import (
	"encoding/json"
	"fmt"
	"strings"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
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

// messagesFromLLM converts an llms.Message to the Google API message format.
// It may return multiple messages if the input is a tool result with auxiliary content.
func messagesFromLLM(m llms.Message) []message {
	if m.Role == "tool" {
		var messagesToReturn []message
		var primaryResultJSON json.RawMessage
		var secondaryContent content.Content

		// Extract primary result (must be JSON) and potential secondary content.
		if len(m.Content) > 0 {
			if jsonItem, ok := m.Content[0].(*content.JSON); ok {
				primaryResultJSON = jsonItem.Data
				if len(m.Content) > 1 {
					secondaryContent = m.Content[1:]
				}
			} else {
				// If the first part isn't JSON, create an error response and treat all original content as secondary.
				errorData, _ := json.Marshal(map[string]string{"error": "Primary tool result must be JSON for Google Gemini"})
				primaryResultJSON = errorData
				secondaryContent = m.Content // All original content becomes secondary
			}
		} else {
			// Handle empty result content - send empty JSON object.
			primaryResultJSON = json.RawMessage("{}")
		}

		// Create the primary tool/function response message.
		// Google expects the functionResponse.Response to contain {"name": toolCallID, "content": actual_result}
		responseWrapperJSON, err := json.Marshal(map[string]any{
			"name":    m.ToolCallID,      // Use the original ToolCallID here
			"content": primaryResultJSON, // Note: primaryResultJSON is already marshaled JSON
		})
		if err != nil {
			// Handle marshaling error for the wrapper, maybe return an error message
			panic(fmt.Sprintf("failed to marshal google function response wrapper: %v", err))
		}

		primaryMessage := message{
			Role: "function", // Google uses "function" role for function responses.
			Parts: parts{
				{
					FunctionResponse: &functionResponse{
						Name:     m.ToolCallID, // Associate with the call ID
						Response: responseWrapperJSON,
					},
				},
			},
		}
		messagesToReturn = append(messagesToReturn, primaryMessage)

		// Handle additional content items (if any) by creating a secondary user message.
		if len(secondaryContent) > 0 {
			secondaryParts := convertContent(secondaryContent)
			if len(secondaryParts) > 0 { // Only add if there are convertible parts
				secondaryMessage := message{
					Role:  "user", // Faked user message for additional content
					Parts: secondaryParts,
				}
				messagesToReturn = append(messagesToReturn, secondaryMessage)
			}
		}
		return messagesToReturn
	}

	// Handle regular messages (user, model/assistant)
	apiRole := m.Role
	if apiRole == "assistant" {
		apiRole = "model" // Google uses "model" for assistant role
	}

	apiParts := convertContent(m.Content)

	// Add function calls if the message is from the assistant/model
	if m.Role == "assistant" {
		for _, toolCall := range m.ToolCalls {
			// Ensure toolCall.Arguments is not nil before trying to use it
			args := json.RawMessage("{}") // Default to empty JSON object if nil
			if toolCall.Arguments != nil {
				args = toolCall.Arguments
			}
			apiParts = append(apiParts, part{
				FunctionCall: &functionCall{
					Name: toolCall.Name,
					Args: args,
				},
			})
		}
	}

	// Only return a message if it has parts
	if len(apiParts) == 0 {
		return []message{}
	}

	return []message{{
		Role:  apiRole,
		Parts: apiParts,
	}}
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
