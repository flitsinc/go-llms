package llms

import (
	"context"
	"fmt"
	"net/http"
	"regexp"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/tools"
)

// Usage represents token usage information from an LLM response.
type Usage struct {
	CachedInputTokens        int // Number of cached input tokens (cache reads)
	CacheCreationInputTokens int // Number of input tokens used to create cache (cache writes)
	InputTokens              int // Number of input tokens
	OutputTokens             int // Number of output tokens
}

func (u *Usage) Add(other Usage) {
	u.CachedInputTokens += other.CachedInputTokens
	u.CacheCreationInputTokens += other.CacheCreationInputTokens
	u.InputTokens += other.InputTokens
	u.OutputTokens += other.OutputTokens
}

type ProviderStream interface {
	Err() error
	Iter() func(yield func(StreamStatus) bool)
	Message() Message
	Text() string
	Image() (string, string)
	Thought() content.Thought
	ToolCall() ToolCall
	Usage() Usage
}

// Debugger is used to debug communication with the LLM.
type Debugger interface {
	RawRequest(endpoint string, data []byte)
	RawEvent([]byte)
}

type stdOutDebugger struct{}

func (d *stdOutDebugger) RawRequest(endpoint string, data []byte) {
	// Remove Google API keys from the URL before logging it.
	fmt.Printf("\033[1;90m%s\033[0m\n", regexp.MustCompile(`([&?]key)=[^&]*`).ReplaceAllString(endpoint, "$1=…"))
	fmt.Printf("-> \033[2;34m%s\033[0m\n", string(data))
}

func (d *stdOutDebugger) RawEvent(data []byte) {
	fmt.Printf("<- \033[2;32m%s\033[0m\n", string(data))
}

var StdOutDebugger = &stdOutDebugger{}

type Provider interface {
	Company() string
	Model() string
	SetDebugger(d Debugger)
	SetHTTPClient(client *http.Client)
	// Generate takes a system prompt, message history, and optional toolbox,
	// returning a stream for the LLM's response. The provided context should
	// be respected for cancellation.
	Generate(
		ctx context.Context,
		systemPrompt content.Content,
		messages []Message,
		toolbox *tools.Toolbox,
		jsonOutputSchema *tools.ValueSchema,
	) ProviderStream
}
