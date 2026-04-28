package openai

import (
	"bytes"
	"encoding/json"
	"net/http"

	"github.com/flitsinc/go-llms/llms"
)

type openAIErrorResponse struct {
	Error struct {
		Code     json.RawMessage     `json:"code"`
		Message  string              `json:"message"`
		Type     string              `json:"type"`
		Metadata openAIErrorMetadata `json:"metadata"`
	} `json:"error"`
}

type openAIErrorMetadata struct {
	ProviderName string          `json:"provider_name"`
	Raw          json.RawMessage `json:"raw"`
}

func parseHTTPError(resp *http.Response, bodyBytes []byte) (*llms.HTTPError, bool) {
	var openAIError openAIErrorResponse
	if jsonErr := json.Unmarshal(bodyBytes, &openAIError); jsonErr != nil || openAIError.Error.Message == "" {
		return nil, false
	}

	return &llms.HTTPError{
		StatusCode: resp.StatusCode,
		Status:     resp.Status,
		ErrorCode:  rawJSONScalarString(openAIError.Error.Code),
		ErrorType:  openAIError.Error.Type,
		Message:    openAIError.Error.Message,
		Metadata:   parseHTTPErrorMetadata(openAIError.Error.Metadata),
	}, true
}

func parseHTTPErrorMetadata(metadata openAIErrorMetadata) llms.HTTPErrorMetadata {
	raw := normalizeRawError(metadata.Raw)
	httpMetadata := llms.HTTPErrorMetadata{
		ProviderName: metadata.ProviderName,
		Raw:          raw,
	}
	populateRawErrorMetadata(raw, &httpMetadata)
	return httpMetadata
}

func normalizeRawError(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 {
		return nil
	}

	var rawString string
	if json.Unmarshal(raw, &rawString) == nil && json.Valid([]byte(rawString)) {
		return append(json.RawMessage(nil), []byte(rawString)...)
	}

	return append(json.RawMessage(nil), raw...)
}

func populateRawErrorMetadata(raw json.RawMessage, metadata *llms.HTTPErrorMetadata) {
	if len(raw) == 0 {
		return
	}

	// Gateways return raw upstream errors as plain strings, nested error objects, or flat objects.
	var rawMessage string
	if json.Unmarshal(raw, &rawMessage) == nil {
		metadata.RawErrorMessage = rawMessage
		return
	}

	var rawError struct {
		Code       json.RawMessage `json:"code"`
		Message    string          `json:"message"`
		Type       string          `json:"type"`
		StatusCode int             `json:"status_code"`
		Error      struct {
			Code       json.RawMessage `json:"code"`
			Message    string          `json:"message"`
			Type       string          `json:"type"`
			StatusCode int             `json:"status_code"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &rawError) != nil {
		return
	}

	if rawError.Error.Message != "" || rawError.Error.Type != "" || len(rawError.Error.Code) > 0 {
		metadata.RawErrorCode = rawJSONScalarString(rawError.Error.Code)
		metadata.RawErrorType = rawError.Error.Type
		metadata.RawErrorMessage = rawError.Error.Message
		metadata.RawErrorStatusCode = rawError.Error.StatusCode
		return
	}

	metadata.RawErrorCode = rawJSONScalarString(rawError.Code)
	metadata.RawErrorType = rawError.Type
	metadata.RawErrorMessage = rawError.Message
	metadata.RawErrorStatusCode = rawError.StatusCode
}

func rawJSONScalarString(raw json.RawMessage) string {
	if len(raw) == 0 || bytes.Equal(raw, []byte("null")) {
		return ""
	}

	var stringValue string
	if json.Unmarshal(raw, &stringValue) == nil {
		return stringValue
	}

	decoder := json.NewDecoder(bytes.NewReader(raw))
	decoder.UseNumber()
	var numberValue json.Number
	if decoder.Decode(&numberValue) == nil {
		return numberValue.String()
	}

	return ""
}
