package llms

import "fmt"

// HTTPError represents an HTTP error response from an LLM provider.
type HTTPError struct {
	StatusCode int    // HTTP status code (e.g., 429, 503, 500)
	Status     string // Full status text (e.g., "429 Too Many Requests")
	ErrorType  string // Provider-specific error type (e.g., "rate_limit_error")
	Message    string // Human-readable error message
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
