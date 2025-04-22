package tools

import (
	"encoding/json"
	"fmt"

	"github.com/flitsinc/go-llms/content"
)

// Result defines the outcome of a tool execution.
type Result interface {
	// Label returns a short single line description of the entire tool run.
	Label() string
	// Content returns the structured content of the result.
	Content() content.Content
	// Error returns the error that occurred during the tool run, if any.
	Error() error
}

type result struct {
	label   string
	content content.Content
	err     error
}

func (r *result) Label() string {
	return r.label
}

func (r *result) Content() content.Content {
	return r.content
}

func (r *result) Error() error {
	return r.err
}

func Error(err error) Result {
	return ErrorWithLabel("", err)
}

func Errorf(format string, args ...any) Result {
	return ErrorWithLabel("", fmt.Errorf(format, args...))
}

func ErrorWithLabel(label string, err error) Result {
	if err == nil {
		panic("tools: cannot create error result with nil error")
	}
	// Create a JSON structure for the error message.
	errorJSON, _ := json.Marshal(map[string]string{"error": err.Error()})
	c := content.FromRawJSON(errorJSON)
	if label == "" {
		label = fmt.Sprintf("Error: %s", err)
	}
	return &result{label, c, err}
}

// Success creates a result by marshaling the value to JSON content. It attempts
// to generate a label automatically from the value if it implements
// fmt.Stringer.
func Success(value any) Result {
	label := "Success"
	if stringer, ok := value.(fmt.Stringer); ok {
		label = stringer.String()
		if len(label) > 80 { // Keep auto-labels reasonably short
			label = label[:77] + "..."
		}
	}
	return SuccessWithLabel(label, value)
}

// SuccessWithLabel creates a result with an explicit label by marshaling the
// value to JSON content.
func SuccessWithLabel(label string, value any) Result {
	c, err := content.FromAny(value)
	if err != nil {
		return ErrorWithLabel(fmt.Sprintf("Error (%s)", label), fmt.Errorf("failed to marshal success result to JSON: %w", err))
	}
	return SuccessWithContent(label, c)
}

// SuccessFromString creates a result containing the string wrapped in
// {"output": ...} JSON content. It generates a label automatically from the
// string content.
func SuccessFromString(output string) Result {
	label := output
	if len(label) > 80 {
		label = label[:77] + "..."
	}
	return SuccessFromStringWithLabel(label, output)
}

// Successf creates a result containing the formatted string wrapped in
// {"output": ...} JSON content. It generates a label automatically from the
// string content.
func Successf(format string, args ...any) Result {
	output := fmt.Sprintf(format, args...)
	return SuccessFromString(output)
}

// SuccessFromStringWithLabel creates a result with an explicit label,
// containing the string wrapped in {"output": ...} JSON content.
func SuccessFromStringWithLabel(label string, output string) Result {
	return SuccessWithLabel(label, map[string]string{"output": output})
}

// SuccessWithContent creates a result with an explicit label and
// pre-constructed content. This is the escape hatch for advanced use cases,
// like including images.
func SuccessWithContent(label string, content content.Content) Result {
	if label == "" {
		label = "Success"
	}
	return &result{label: label, content: content, err: nil}
}
