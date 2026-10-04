package llms

import (
	"encoding/json"
	"errors"
	"fmt"
	"strings"
)

// ErrOutputTruncated is returned when a model's output is cut short because it
// hit the max output token limit (e.g. finish_reason="length" in OpenAI,
// stop_reason="max_tokens" in Anthropic).
var ErrOutputTruncated = errors.New("output truncated: model reached max output token limit")

// HTTPError represents an HTTP error response from an LLM provider.
type HTTPError struct {
	StatusCode int               // HTTP status code (e.g., 429, 503, 500)
	Status     string            // Full status text (e.g., "429 Too Many Requests")
	ErrorCode  string            // Provider-specific error code from the response body
	ErrorType  string            // Provider-specific error type (e.g., "rate_limit_error")
	Message    string            // Human-readable error message
	Metadata   HTTPErrorMetadata // Optional upstream-provider diagnostics
}

// HTTPErrorMetadata contains upstream-provider diagnostics returned through a gateway.
type HTTPErrorMetadata struct {
	ProviderName       string          // Upstream provider name
	Raw                json.RawMessage // Raw upstream error payload
	RawErrorCode       string          // Upstream provider-specific error code
	RawErrorType       string          // Upstream provider-specific error type
	RawErrorMessage    string          // Upstream provider error message
	RawErrorStatusCode int             // Upstream provider HTTP status code
}

func (e *HTTPError) Error() string {
	if e.ErrorType != "" && e.Message != "" {
		return fmt.Sprintf("%s: %s: %s", e.Status, e.ErrorType, e.Message)
	}
	if e.Message != "" {
		return fmt.Sprintf("%s: %s", e.Status, e.Message)
	}
	return e.Status
}

// IsRequestTooLarge reports whether the error indicates that the request
// exceeded the model's context window or payload size limit.
//
// It checks structured fields first (HTTP 413, OpenAI's "context_length_exceeded"
// error code) and falls back to message-based detection for providers like
// Anthropic that do not expose a structured error code for this condition.
func (e *HTTPError) IsRequestTooLarge() bool {
	if e.StatusCode == 413 || e.Metadata.RawErrorStatusCode == 413 {
		return true
	}
	if isRequestTooLargeCode(e.ErrorCode) || isRequestTooLargeCode(e.ErrorType) ||
		isRequestTooLargeCode(e.Metadata.RawErrorCode) || isRequestTooLargeCode(e.Metadata.RawErrorType) {
		return true
	}
	return isRequestTooLargeMessage(e.Message) || isRequestTooLargeMessage(e.Metadata.RawErrorMessage)
}

// isRequestTooLargeCode matches the structured codes providers and gateways
// use for oversized requests: OpenRouter's normalized error_type values
// "context_length_exceeded" (HTTP 400) and "payload_too_large" (HTTP 413), and
// Anthropic's native "request_too_large" error type.
func isRequestTooLargeCode(code string) bool {
	switch code {
	case "context_length_exceeded", "payload_too_large", "request_too_large":
		return true
	}
	return false
}

func isRequestTooLargeMessage(msg string) bool {
	return strings.Contains(msg, "prompt is too long") ||
		strings.Contains(msg, "maximum context length")
}

// IsCompactionRejected reports whether the provider refused a replayed native
// context compaction checkpoint (a content.Compaction in the request history):
// OpenAI could not decrypt it, for example after a change of organization, or
// Anthropic does not accept the block on this model. The checkpoint stays
// unusable however often the request is repeated, so the caller should drop
// it and send the history it summarized instead. The answer is only
// meaningful for a request that replayed a checkpoint.
//
// OpenAI reports this with the structured "invalid_encrypted_content" code.
// Anthropic (directly and through OpenRouter's Messages endpoint) returns a
// plain invalid_request_error, so this is a compatibility adapter that falls
// back to the error message, as IsRequestTooLarge does.
func (e *HTTPError) IsCompactionRejected() bool {
	if e.ErrorCode == "invalid_encrypted_content" || e.Metadata.RawErrorCode == "invalid_encrypted_content" {
		return true
	}
	if e.StatusCode != 400 {
		return false
	}
	return (e.ErrorType == "invalid_request_error" && isCompactionRejectedMessage(e.Message)) ||
		(e.Metadata.RawErrorType == "invalid_request_error" && isCompactionRejectedMessage(e.Metadata.RawErrorMessage))
}

// isCompactionRejectedMessage matches Anthropic's messages for a compaction
// block it will not take back, which name either the block ("compaction") or
// the strategy ("compact_20260112"), e.g. "'claude-haiku-4-5-20251001' does
// not support the 'compact_20260112' context management strategy." or
// "messages.1.content.0.compaction.content: content cannot be empty".
func isCompactionRejectedMessage(msg string) bool {
	return strings.Contains(msg, "compaction") || strings.Contains(msg, "compact_20")
}
