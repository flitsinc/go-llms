package openai

import (
	"bytes"
	"encoding/json"
	"net/http"
	"strconv"

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

	// "status" is left untyped because upstreams disagree on its shape: most
	// gateways put an HTTP code there, Google puts a symbolic name such as
	// "INVALID_ARGUMENT". Decoding it as an int made json.Unmarshal fail on
	// every Google error, and the failure discarded the message it had already
	// decoded — so Google's actual complaint never reached callers or traces.
	var rawError struct {
		Code       json.RawMessage `json:"code"`
		Message    string          `json:"message"`
		Type       string          `json:"type"`
		Status     json.RawMessage `json:"status"`
		StatusCode json.RawMessage `json:"status_code"`
		Error      struct {
			Code       json.RawMessage `json:"code"`
			Message    string          `json:"message"`
			Type       string          `json:"type"`
			Status     json.RawMessage `json:"status"`
			StatusCode json.RawMessage `json:"status_code"`
		} `json:"error"`
	}
	if json.Unmarshal(raw, &rawError) != nil {
		return
	}

	nestedStatusCode, nestedStatusName := rawErrorStatus(rawError.Error.StatusCode, rawError.Error.Status)
	nestedType := rawError.Error.Type
	if nestedType == "" {
		nestedType = nestedStatusName
	}
	if rawError.Error.Message != "" || nestedType != "" || rawJSONScalarString(rawError.Error.Code) != "" || nestedStatusCode != 0 {
		metadata.RawErrorCode = rawJSONScalarString(rawError.Error.Code)
		metadata.RawErrorType = nestedType
		metadata.RawErrorMessage = rawError.Error.Message
		metadata.RawErrorStatusCode = nestedStatusCode
		return
	}

	statusCode, statusName := rawErrorStatus(rawError.StatusCode, rawError.Status)
	errorType := rawError.Type
	if errorType == "" {
		errorType = statusName
	}
	metadata.RawErrorCode = rawJSONScalarString(rawError.Code)
	metadata.RawErrorType = errorType
	metadata.RawErrorMessage = rawError.Message
	metadata.RawErrorStatusCode = statusCode
}

// rawErrorStatus separates the two things upstreams put in "status" and
// "status_code": a numeric HTTP code, and a symbolic name like
// "INVALID_ARGUMENT". Either field can hold either shape, so both are read.
func rawErrorStatus(statusCode, status json.RawMessage) (code int, name string) {
	for _, candidate := range []json.RawMessage{statusCode, status} {
		text := rawJSONScalarString(candidate)
		if text == "" {
			continue
		}
		if parsed, err := strconv.Atoi(text); err == nil {
			if code == 0 {
				code = parsed
			}
			continue
		}
		if name == "" {
			name = text
		}
	}
	return code, name
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
