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

	var fields map[string]json.RawMessage
	if json.Unmarshal(raw, &fields) != nil {
		return
	}

	upstream, ok := decodeUpstreamError(fields["error"])
	if !ok {
		upstream, _ = decodeUpstreamError(raw)
	}

	metadata.RawErrorCode = upstream.code
	metadata.RawErrorType = upstream.errorType
	metadata.RawErrorMessage = upstream.message
	metadata.RawErrorStatusCode = upstream.statusCode
}

type upstreamError struct {
	code       string
	message    string
	errorType  string
	statusCode int
}

// decodeUpstreamError reads one level of a gateway's raw upstream error.
//
// The fields are decoded one at a time rather than through a single struct
// because this body comes straight from whichever provider the gateway called,
// and its shape is not ours to assume. Decoding it as a struct meant one
// unexpected type failed the whole unmarshal and discarded every other field
// with it — which is how Google's errors lost the message that explained them.
//
// Reports whether anything was found, so the caller can fall back from the
// nested shape to the flat one.
func decodeUpstreamError(raw json.RawMessage) (upstreamError, bool) {
	var fields map[string]json.RawMessage
	if len(raw) == 0 || json.Unmarshal(raw, &fields) != nil {
		return upstreamError{}, false
	}

	upstream := upstreamError{
		code:      rawJSONScalarString(fields["code"]),
		message:   rawJSONScalarString(fields["message"]),
		errorType: rawJSONScalarString(fields["type"]),
	}
	upstream.statusCode, _ = strconv.Atoi(rawJSONScalarString(fields["status_code"]))

	// "status" is the one genuinely ambiguous key: an HTTP code on most
	// gateways, and a canonical name such as "INVALID_ARGUMENT" on Google,
	// where it is the error's type rather than its code. Route it by what it
	// holds instead of forcing one reading on both.
	if status := rawJSONScalarString(fields["status"]); status != "" {
		if code, err := strconv.Atoi(status); err == nil {
			if upstream.statusCode == 0 {
				upstream.statusCode = code
			}
		} else if upstream.errorType == "" {
			upstream.errorType = status
		}
	}

	return upstream, upstream.code != "" || upstream.message != "" || upstream.errorType != "" || upstream.statusCode != 0
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
