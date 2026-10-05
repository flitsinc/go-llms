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
			name: "OpenRouter payload_too_large error_type",
			err: HTTPError{
				StatusCode: 413,
				ErrorCode:  "413",
				ErrorType:  "payload_too_large",
			},
			want: true,
		},
		{
			name: "OpenRouter payload_too_large error_type without HTTP 413",
			err: HTTPError{
				ErrorType: "payload_too_large",
			},
			want: true,
		},
		{
			name: "Anthropic request_too_large type",
			err: HTTPError{
				StatusCode: 400,
				ErrorType:  "request_too_large",
			},
			want: true,
		},
		{
			name: "Anthropic request_too_large via upstream raw error",
			err: HTTPError{
				StatusCode: 400,
				Message:    "Provider returned error",
				Metadata: HTTPErrorMetadata{
					ProviderName: "Claude Platform on AWS",
					RawErrorType: "request_too_large",
				},
			},
			want: true,
		},
		{
			name: "Anthropic request_too_large via upstream provider code",
			err: HTTPError{
				StatusCode: 400,
				Metadata:   HTTPErrorMetadata{RawErrorCode: "request_too_large"},
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

// The messages are the ones the providers returned live (2026-10-04).
func TestIsCompactionRejected(t *testing.T) {
	tests := []struct {
		name string
		err  HTTPError
		want bool
	}{
		{
			name: "OpenAI cannot decrypt the checkpoint",
			err: HTTPError{
				StatusCode: 400,
				ErrorCode:  "invalid_encrypted_content",
				ErrorType:  "invalid_request_error",
				Message:    "The encrypted content for item cmp_0000000000000000 could not be verified. Reason: Encrypted content could not be decrypted or parsed.",
			},
			want: true,
		},
		{
			name: "upstream invalid_encrypted_content via gateway",
			err: HTTPError{
				StatusCode: 400,
				ErrorCode:  "400",
				Metadata:   HTTPErrorMetadata{RawErrorCode: "invalid_encrypted_content"},
			},
			want: true,
		},
		{
			name: "upstream that does not know the block",
			err: HTTPError{
				StatusCode: 400,
				ErrorType:  "invalid_request_error",
				Message:    "messages.1.content.0: Input tag 'compaction' found using 'type' does not match any of the expected tags: 'text', 'thinking', 'tool_use'",
			},
			want: true,
		},
		{
			// The compaction setting, not the checkpoint: dropping the
			// checkpoint cannot fix the request.
			name: "Anthropic model without the compaction strategy",
			err: HTTPError{
				StatusCode: 400,
				ErrorType:  "invalid_request_error",
				Message:    "'claude-haiku-4-5-20251001' does not support the 'compact_20260112' context management strategy.",
			},
			want: false,
		},
		{
			name: "Anthropic rejected compaction setting",
			err: HTTPError{
				StatusCode: 400,
				ErrorType:  "invalid_request_error",
				Message:    "context_management.edits.0.trigger.value: Input should be greater than or equal to 50000",
			},
			want: false,
		},
		{
			name: "Anthropic empty compaction block",
			err: HTTPError{
				StatusCode: 400,
				ErrorType:  "invalid_request_error",
				Message:    "messages.1.content.0.compaction.content: content cannot be empty",
			},
			want: true,
		},
		{
			name: "Anthropic rejection relayed in gateway metadata",
			err: HTTPError{
				StatusCode: 400,
				ErrorCode:  "400",
				Message:    "Provider returned error",
				Metadata: HTTPErrorMetadata{
					ProviderName:    "Anthropic",
					RawErrorType:    "invalid_request_error",
					RawErrorMessage: "messages.1.content.0: `compaction` blocks require a `compact_20260112` strategy in `context_management.edits`.",
				},
			},
			want: true,
		},
		{
			name: "unrelated invalid request",
			err: HTTPError{
				StatusCode: 400,
				ErrorType:  "invalid_request_error",
				Message:    "messages.1.content.0.tool_use.input: Input should be a valid dictionary",
			},
			want: false,
		},
		{
			name: "compaction mentioned outside a 400",
			err: HTTPError{
				StatusCode: 500,
				ErrorType:  "api_error",
				Message:    "compaction failed",
			},
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := tt.err.IsCompactionRejected(); got != tt.want {
				t.Errorf("IsCompactionRejected() = %v, want %v", got, tt.want)
			}
		})
	}
}
