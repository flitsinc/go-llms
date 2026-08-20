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

func TestRawErrorStatus(t *testing.T) {
	for name, tc := range map[string]struct {
		statusCode, status string
		wantCode           int
		wantName           string
	}{
		"symbolic status only":      {status: `"INVALID_ARGUMENT"`, wantName: "INVALID_ARGUMENT"},
		"numeric status only":       {status: `503`, wantCode: 503},
		"numeric status as string":  {status: `"503"`, wantCode: 503},
		"status_code wins the code": {statusCode: `429`, status: `"RESOURCE_EXHAUSTED"`, wantCode: 429, wantName: "RESOURCE_EXHAUSTED"},
		"absent":                    {},
	} {
		t.Run(name, func(t *testing.T) {
			code, statusName := rawErrorStatus(rawMessageOrNil(tc.statusCode), rawMessageOrNil(tc.status))
			if code != tc.wantCode || statusName != tc.wantName {
				t.Errorf("rawErrorStatus() = (%d, %q), want (%d, %q)", code, statusName, tc.wantCode, tc.wantName)
			}
		})
	}
}

func rawMessageOrNil(raw string) []byte {
	if raw == "" {
		return nil
	}
	return []byte(raw)
}
