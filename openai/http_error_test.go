package openai

import (
	"encoding/json"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
)

// Verbatim body from OpenRouter, captured 2026-08-19 against
// google/gemini-3.7-flash. Note "status": "INVALID_ARGUMENT": Google puts a
// symbolic name where every other gateway puts an HTTP code.
const googleViaOpenRouterErrorBody = `{"error":{"message":"Provider returned error","code":400,"metadata":{"raw":"{\n  \"error\": {\n    \"code\": 400,\n    \"message\": \"* GenerateContentRequest.tools[0].function_declarations[0].parameters.properties[s].enum[1]: cannot be empty\\n\",\n    \"status\": \"INVALID_ARGUMENT\"\n  }\n}\n","provider_name":"Google AI Studio","is_byok":false,"provider_error_code":"400","previous_errors":[{"code":400,"message":"Provider returned error","provider_name":"Google","raw":"{\n  \"error\": {\n    \"code\": 400,\n    \"message\": \"* GenerateContentRequest.tools[0].function_declarations[0].parameters.properties[s].enum[1]: cannot be empty\\n\",\n    \"status\": \"INVALID_ARGUMENT\"\n  }\n}\n"}]}},"user_id":"org_x"}`

// The gateway's own message is always "Provider returned error", so the raw
// upstream message is the only thing that says what was actually wrong. Losing
// it turned a one-line schema complaint into an unexplained 400.
func TestParseHTTPError_GoogleSymbolicStatus(t *testing.T) {
	resp := &http.Response{StatusCode: 400, Status: "400 Bad Request"}

	httpErr, ok := parseHTTPError(resp, []byte(googleViaOpenRouterErrorBody))
	if !ok {
		t.Fatal("expected the error body to parse")
	}

	const wantMessage = "* GenerateContentRequest.tools[0].function_declarations[0].parameters.properties[s].enum[1]: cannot be empty\n"
	if got := httpErr.Metadata.RawErrorMessage; got != wantMessage {
		t.Errorf("RawErrorMessage = %q, want %q", got, wantMessage)
	}
	if got := httpErr.Metadata.RawErrorType; got != "INVALID_ARGUMENT" {
		t.Errorf("RawErrorType = %q, want the symbolic status", got)
	}
	if got := httpErr.Metadata.RawErrorCode; got != "400" {
		t.Errorf("RawErrorCode = %q, want %q", got, "400")
	}
	if got := httpErr.Metadata.ProviderName; got != "Google AI Studio" {
		t.Errorf("ProviderName = %q", got)
	}
}

func TestDecodeUpstreamError_StatusIsRoutedByWhatItHolds(t *testing.T) {
	for name, tc := range map[string]struct {
		body           string
		wantStatusCode int
		wantType       string
	}{
		"symbolic status becomes the type":  {body: `{"status":"INVALID_ARGUMENT"}`, wantType: "INVALID_ARGUMENT"},
		"numeric status becomes the code":   {body: `{"status":503}`, wantStatusCode: 503},
		"numeric status sent as a string":   {body: `{"status":"503"}`, wantStatusCode: 503},
		"status_code owns the code":         {body: `{"status_code":429,"status":"RESOURCE_EXHAUSTED"}`, wantStatusCode: 429, wantType: "RESOURCE_EXHAUSTED"},
		"an explicit type is not displaced": {body: `{"type":"rate_limit","status":"RESOURCE_EXHAUSTED"}`, wantType: "rate_limit"},
	} {
		t.Run(name, func(t *testing.T) {
			upstream := decodeUpstreamError([]byte(tc.body))
			if upstream.statusCode != tc.wantStatusCode || upstream.errorType != tc.wantType {
				t.Errorf("statusCode/type = (%d, %q), want (%d, %q)",
					upstream.statusCode, upstream.errorType, tc.wantStatusCode, tc.wantType)
			}
		})
	}
}

// The reason fields are decoded one at a time: a shape we did not expect in one
// of them used to discard the whole object, message included.
func TestDecodeUpstreamError_OneOddFieldDoesNotCostTheOthers(t *testing.T) {
	upstream := decodeUpstreamError([]byte(`{"code":{"unexpected":"object"},"message":"the real reason","status_code":400}`))
	if upstream.message != "the real reason" {
		t.Errorf("message = %q, want it preserved alongside the odd field", upstream.message)
	}
	if upstream.statusCode != 400 {
		t.Errorf("statusCode = %d, want 400", upstream.statusCode)
	}
}

// An empty nested "error" object must not shadow fields sitting flat on the
// raw body itself.
func TestParseHTTPErrorMetadata_FlatRawErrorWithNullNestedCode(t *testing.T) {
	metadata := parseHTTPErrorMetadata(openAIErrorMetadata{
		Raw: json.RawMessage(`{"message": "flat message", "type": "flat_type", "error": {"code": null}}`),
	})

	assert.Empty(t, metadata.RawErrorCode)
	assert.Equal(t, "flat_type", metadata.RawErrorType)
	assert.Equal(t, "flat message", metadata.RawErrorMessage)
}

func TestParseHTTPErrorMetadata_NestedStatusCodeOnly(t *testing.T) {
	metadata := parseHTTPErrorMetadata(openAIErrorMetadata{
		Raw: json.RawMessage(`{"error": {"status_code": 429}}`),
	})

	assert.Equal(t, 429, metadata.RawErrorStatusCode)
}

func TestParseHTTPErrorMetadata_NestedStatusOnly(t *testing.T) {
	metadata := parseHTTPErrorMetadata(openAIErrorMetadata{
		Raw: json.RawMessage(`{"error": {"status": 429}}`),
	})

	assert.Equal(t, 429, metadata.RawErrorStatusCode)
}

// OpenRouter's normalized error_type is the field it documents for switching on
// error categories; the envelope "type" is often absent. provider_code is the
// upstream provider's own code and lands in RawErrorCode.
func TestParseHTTPError_NormalizedErrorTypeAndProviderCode(t *testing.T) {
	body := `{"error":{"code":413,"message":"Provider returned error","metadata":{"error_type":"payload_too_large","provider_code":"request_too_large","provider_name":"Claude Platform on AWS","is_byok":false}},"user_id":"org_x"}`
	resp := &http.Response{StatusCode: 413, Status: "413 Payload Too Large"}

	httpErr, ok := parseHTTPError(resp, []byte(body))
	if !ok {
		t.Fatal("expected the error body to parse")
	}

	assert.Equal(t, "payload_too_large", httpErr.ErrorType)
	assert.Equal(t, "request_too_large", httpErr.Metadata.RawErrorCode)
	assert.Equal(t, "Claude Platform on AWS", httpErr.Metadata.ProviderName)
	assert.True(t, httpErr.IsRequestTooLarge())
}

// An explicit envelope "type" wins over metadata.error_type.
func TestParseHTTPError_EnvelopeTypeWins(t *testing.T) {
	body := `{"error":{"code":400,"type":"invalid_request_error","message":"Provider returned error","metadata":{"error_type":"context_length_exceeded"}}}`
	resp := &http.Response{StatusCode: 400, Status: "400 Bad Request"}

	httpErr, ok := parseHTTPError(resp, []byte(body))
	if !ok {
		t.Fatal("expected the error body to parse")
	}

	assert.Equal(t, "invalid_request_error", httpErr.ErrorType)
}

// Verbatim body OpenAI returned (2026-10-05) for a reasoning item replayed by
// an ID the key's organization does not hold.
func TestParseHTTPError_ParamAndReplayRejected(t *testing.T) {
	body := `{"error":{"message":"Item with id 'rs_0645db6af0c67af0006ac3bacda14887d1b16c5fea754e94ff' not found.","type":"invalid_request_error","param":"input","code":null}}`
	resp := &http.Response{StatusCode: 404, Status: "404 Not Found"}

	httpErr, ok := parseHTTPError(resp, []byte(body))
	if !ok {
		t.Fatal("expected the error body to parse")
	}

	assert.Equal(t, "input", httpErr.Param)
	assert.Equal(t, "", httpErr.ErrorCode)
	assert.True(t, httpErr.IsReplayRejected())
}

// A null param, as OpenAI sends for an undecryptable blob, parses as empty.
func TestParseHTTPError_NullParam(t *testing.T) {
	body := `{"error":{"message":"The encrypted content gAAA...xxxx could not be verified. Reason: Encrypted content could not be decrypted or parsed.","type":"invalid_request_error","param":null,"code":"invalid_encrypted_content"}}`
	resp := &http.Response{StatusCode: 400, Status: "400 Bad Request"}

	httpErr, ok := parseHTTPError(resp, []byte(body))
	if !ok {
		t.Fatal("expected the error body to parse")
	}

	assert.Equal(t, "", httpErr.Param)
	assert.Equal(t, "invalid_encrypted_content", httpErr.ErrorCode)
	assert.True(t, httpErr.IsReplayRejected())
}
