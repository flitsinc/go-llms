package llms

import (
	"testing"
)

func TestIsRequestTooLarge(t *testing.T) {
	tests := []struct {
		name string
		err  HTTPError
		want bool
	}{
		{
			name: "HTTP 413",
			err:  HTTPError{StatusCode: 413, Status: "413 Request Entity Too Large"},
			want: true,
		},
		{
			name: "upstream 413 via gateway",
			err: HTTPError{
				StatusCode: 400,
				Metadata:   HTTPErrorMetadata{RawErrorStatusCode: 413},
			},
			want: true,
		},
		{
			name: "OpenAI context_length_exceeded code",
			err: HTTPError{
				StatusCode: 400,
				ErrorCode:  "context_length_exceeded",
				Message:    "This model's maximum context length is 128000 tokens.",
			},
			want: true,
		},
		{
			name: "upstream context_length_exceeded via gateway",
			err: HTTPError{
				StatusCode: 400,
				ErrorCode:  "400",
				Metadata:   HTTPErrorMetadata{RawErrorCode: "context_length_exceeded"},
			},
			want: true,
		},
		{
			name: "Anthropic prompt too long via OpenRouter metadata",
			err: HTTPError{
				StatusCode: 400,
				ErrorCode:  "400",
				Message:    "Provider returned error",
				Metadata: HTTPErrorMetadata{
					ProviderName:    "Anthropic",
					RawErrorType:    "invalid_request_error",
					RawErrorMessage: "prompt is too long: 1203058 tokens > 1000000 maximum",
				},
			},
			want: true,
		},
		{
			name: "Anthropic prompt too long in top-level message",
			err: HTTPError{
				StatusCode: 400,
				ErrorCode:  "400",
				Message:    "prompt is too long: 202812 tokens > 200000 maximum",
			},
			want: true,
		},
		{
			name: "OpenRouter maximum context length in top-level message",
			err: HTTPError{
				StatusCode: 400,
				Message:    "This endpoint's maximum context length is 200000 tokens. However, you requested about 200146 tokens.",
			},
			want: true,
		},
		{
			name: "unrelated 400 error",
			err: HTTPError{
				StatusCode: 400,
				ErrorCode:  "400",
				ErrorType:  "invalid_request_error",
				Message:    "Provider returned error",
				Metadata: HTTPErrorMetadata{
					RawErrorType:    "invalid_request_error",
					RawErrorMessage: "messages.56.content.1: each tool_use must have a single result",
				},
			},
			want: false,
		},
		{
			name: "rate limit error",
			err: HTTPError{
				StatusCode: 429,
				ErrorType:  "rate_limit_error",
				Message:    "Rate limit exceeded",
			},
			want: false,
		},
		{
			name: "generic 500 error",
			err: HTTPError{
				StatusCode: 500,
				Message:    "Internal server error",
			},
			want: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.IsRequestTooLarge(); got != tt.want {
				t.Errorf("IsRequestTooLarge() = %v, want %v", got, tt.want)
			}
		})
	}
}
