package openai

import (
	"net/http"
	"testing"
)

// Verbatim body from OpenRouter, captured 2026-08-19 against
// google/gemini-3.7-flash. Note "status": "INVALID_ARGUMENT" — Google puts a
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
			upstream, ok := decodeUpstreamError([]byte(tc.body))
			if !ok {
				t.Fatalf("decodeUpstreamError(%s) found nothing", tc.body)
			}
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
	upstream, ok := decodeUpstreamError([]byte(`{"code":{"unexpected":"object"},"message":"the real reason","status_code":400}`))
	if !ok {
		t.Fatal("expected the object to decode")
	}
	if upstream.message != "the real reason" {
		t.Errorf("message = %q, want it preserved alongside the odd field", upstream.message)
	}
	if upstream.statusCode != 400 {
		t.Errorf("statusCode = %d, want 400", upstream.statusCode)
	}
}
