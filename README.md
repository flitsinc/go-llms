# go-llms

A powerful and flexible Go library for interacting with Large Language Models (LLMs) with built-in support for function calling and streaming responses. Currently supports Anthropic, Google, and OpenAI providers.

## Features

- 🔄 Streaming responses for real-time interaction
- 🛠 Built-in function calling support with a flexible tool system
- 🔌 Extensible provider system (currently supports Anthropic, Google, and OpenAI)
- 📦 Simple and intuitive API
- 🔍 Debug mode for development
- 💰 Cost tracking for API usage

## Requirements

- Go 1.23.2 or later

## Installation

```bash
go get github.com/blixt/go-llms
```

## Quick Start

Here's a simple example that creates an LLM instance and has a conversation with it:

```go
package main

import (
    "fmt"
    "os"

    "github.com/blixt/go-llms/anthropic"
    "github.com/blixt/go-llms/content"
    "github.com/blixt/go-llms/llms"
    "github.com/blixt/go-llms/tools"
)

func main() {
    // Create a new LLM instance with Anthropic's Claude
    llm := llms.New(
        anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-3-5-sonnet-latest"),
    )

    // Optional: Set a system prompt
    llm.SystemPrompt = func() content.Content {
        return content.FromText("You are a helpful assistant.")
    }

    // Start a chat conversation
    for update := range llm.Chat("What's the capital of France?") {
        switch update := update.(type) {
        case llms.ErrorUpdate:
            panic(update.Error)
        case llms.TextUpdate:
            fmt.Print(update.Text)
        }
    }
}
```

## Advanced Usage with Tools

Here's an example showing how to use tools (function calling):

```go
package main

import (
    "fmt"
    "os"

    "github.com/blixt/go-llms/anthropic"
    "github.com/blixt/go-llms/llms"
    "github.com/blixt/go-llms/tools"
)

// Define tool parameters
type CommandParams struct {
    Command string `json:"command" description:"The shell command to run"`
}

// Create a shell command tool
var RunCommand = tools.Func[CommandParams](
    "Run Command",
    "Run a shell command and return the output",
    "run_command",
    func(r tools.Runner, p CommandParams) tools.Result {
        return tools.Success(p.Command, map[string]any{
            "output": "Command output would go here",
        })
    },
)

func main() {
    // Create LLM with tools
    llm := llms.New(
        anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-3-5-sonnet-latest"),
        RunCommand,
    )

    // Chat with tool usage
    for update := range llm.Chat("List files in the current directory") {
        switch update := update.(type) {
        case llms.ErrorUpdate:
            fmt.Println("Error:", update.Error)
            return
        case llms.TextUpdate:
            fmt.Print(update.Text)
        case llms.ToolStartUpdate:
            fmt.Printf("(Using tool: %s)\n", update.Tool.Label())
        case llms.ToolDoneUpdate:
            fmt.Printf("(Tool result: %s)\n", update.Result.Label())
        }
    }
}
```

## Provider Support

The library currently supports:

- Anthropic (Claude models)
- Google (Gemini API and Vertex AI)
- OpenAI (GPT models)

Each provider can be initialized with their respective configuration:

```go
// Anthropic
llm := llms.New(anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-3-5-sonnet-latest"))

// Google Gemini
llm := llms.New(google.New("gemini-pro").WithGeminiAPI(os.Getenv("GOOGLE_API_KEY")))

// Google Vertex AI
llm := llms.New(google.New("gemini-pro").WithVertexAI(accessToken, projectID, region))

// OpenAI
llm := llms.New(openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-4"))
```

You can easily implement new providers by implementing the `Provider` interface:

```go
type Provider interface {
    Company() string
    Generate(systemPrompt content.Content, messages []Message, toolbox *tools.Toolbox) ProviderStream
}
```

## Debug Mode

Enable debug mode to write detailed interaction logs to `debug.yaml`:

```go
llm.SetDebug(true)
```

The debug file includes:

- Received messages
- Tool results
- Sent messages
- System prompts
- Available tools

## Cost Tracking

Track the cost of your LLM interactions:

```go
cost := llm.TotalCost() // Returns the total cost in USD
```

## License

MIT License - See LICENSE file for details.
