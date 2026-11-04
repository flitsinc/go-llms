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
	Param      string            // Request parameter the error points at, when the provider names one (e.g., "input")
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

// IsReplayRejected reports whether the provider refused provider-bound items
// the request replayed from earlier responses: a native context compaction
// checkpoint (content.Compaction), a reasoning item's encrypted content, or an
// item it resolves by ID. Such items belong to the account that produced
// them, so they stop working when the API key moves to another OpenAI
// organization, and they stay unusable however often the request is
// repeated. The caller should stop replaying them: drop the checkpoints and
// thoughts and send the history they stood for. The answer is only meaningful
// for a request that replayed such items.
//
// It matches:
//   - OpenAI's "invalid_encrypted_content" code, for a checkpoint or reasoning
//     content it cannot decrypt (directly or relayed by a gateway);
//   - OpenAI's answer to a replayed item ID it cannot find: HTTP 404, an
//     "invalid_request_error" pointing at the "input" parameter;
//   - Anthropic's refusal of a compaction block (see compactionBlockRejected).
//
// It does not cover a rejected compaction setting, such as a model that does
// not support the compaction strategy: dropping the checkpoint cannot fix
// that request.
func (e *HTTPError) IsReplayRejected() bool {
	if e.ErrorCode == "invalid_encrypted_content" || e.Metadata.RawErrorCode == "invalid_encrypted_content" {
		return true
	}
	if e.StatusCode == 404 && e.ErrorType == "invalid_request_error" && e.Param == "input" {
		return true
	}
	return e.compactionBlockRejected()
}

// compactionBlockRejected reports whether Anthropic (directly or through
// OpenRouter's Messages endpoint) refused a replayed compaction block. It
// returns a plain invalid_request_error, so this is a compatibility adapter
// that falls back to the error message, as IsRequestTooLarge does.
func (e *HTTPError) compactionBlockRejected() bool {
	if e.StatusCode != 400 {
		return false
	}
	return (e.ErrorType == "invalid_request_error" && isCompactionBlockRejectedMessage(e.Message)) ||
		(e.Metadata.RawErrorType == "invalid_request_error" && isCompactionBlockRejectedMessage(e.Metadata.RawErrorMessage))
}

// isCompactionBlockRejectedMessage matches Anthropic's messages that point at
// a compaction content block, e.g.
// "messages.1.content.0.compaction.content: content cannot be empty",
// "messages.1.content.0: `compaction` blocks require a `compact_20260112`
// strategy in `context_management.edits`." or, on an upstream that does not
// know the block, "Input tag 'compaction' found using 'type' does not match
// any of the expected tags". Messages about the compaction setting (such as
// "does not support the 'compact_20260112' context management strategy") do
// not name the block and are left out.
func isCompactionBlockRejectedMessage(msg string) bool {
	return strings.Contains(msg, ".compaction.") ||
		strings.Contains(msg, "`compaction` block") ||
		strings.Contains(msg, "tag 'compaction'")
}

// IsTooManyImages reports whether the provider refused the request for the
// images it carries: more images than the model accepts in one request, or
// more images than the many-image threshold with one larger than the stricter
// dimension limit that applies above it. Sending fewer images clears both.
// Providers whose limits go-llms knows refuse such requests before sending
// them with an [*ImageLimitError]; this covers the requests that reach the
// provider anyway, such as a model missing from a provider's table.
//
// The providers answer with a plain 400 invalid_request_error (or
// INVALID_ARGUMENT on Vertex AI Gemini) and no structured code, so this is a
// compatibility adapter that falls back to the error message, as
// IsRequestTooLarge does (see isTooManyImagesMessage).
func (e *HTTPError) IsTooManyImages() bool {
	if e.StatusCode != 400 && e.Metadata.RawErrorStatusCode != 400 {
		return false
	}
	return isTooManyImagesMessage(e.Message) || isTooManyImagesMessage(e.Metadata.RawErrorMessage)
}

// isTooManyImagesMessage matches the refusals observed on 2026-11-04 by
// sending one image over each limit, directly and relayed through OpenRouter:
//   - Anthropic, Amazon Bedrock and Microsoft Foundry for a model with a
//     200k-token context window: "Too much media: 0 document pages + 101
//     images > 100";
//   - Anthropic for other models, and Claude on Vertex AI: "too many images
//     and documents: 601 + 0 > 600";
//   - every Claude upstream above 20 images: "messages.0.content.0.image.
//     source.base64.data: At least one of the image dimensions exceed max
//     allowed size for many-image requests: 3000 pixels";
//   - Gemini on Vertex AI: "Unable to submit request because it has 3001
//     image files but the model only supports up to 3000."
func isTooManyImagesMessage(msg string) bool {
	return strings.HasPrefix(msg, "Too much media: ") ||
		strings.HasPrefix(msg, "too many images and documents: ") ||
		strings.Contains(msg, "max allowed size for many-image requests") ||
		strings.Contains(msg, " image files but the model only supports up to ")
}
