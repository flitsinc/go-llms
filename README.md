# go-llms

A powerful and flexible Go library for interacting with Large Language Models (LLMs) with built-in support for function calling and streaming responses. Currently supports Anthropic, Google, and OpenAI providers.

## Features

- 🔄 Streaming responses for real-time interaction
- 🛠 Built-in function calling support with a flexible tool system
- 🔌 Extensible provider system (currently supports Anthropic, Google, and OpenAI)
- 📦 Simple and intuitive API
- 🔍 Debug mode for development
- 💰 Cost tracking for API usage

### On the roadmap

- [ ] 🗃️ Prompt caching
- [ ] 🧠 Reasoning
- [ ] ☎️ Realtime streaming (WebRTC)
- [ ] 🖼️ Image output

## Requirements

- Go 1.24.2 or later

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

    "github.com/blixt/go-llms/content"
    "github.com/blixt/go-llms/llms"
    "github.com/blixt/go-llms/openai"
    "github.com/blixt/go-llms/tools"
)

func main() {
    // Create a new LLM instance with OpenAI's o3-mini model
    llm := llms.New(
        openai.New(os.Getenv("OPENAI_API_KEY"), "o3-mini"),
    )

    // Optional: Set a system prompt
    llm.SystemPrompt = func() content.Content {
        return content.FromText("You are a helpful assistant.")
    }

    // Start a chat conversation
    for update := range llm.Chat("What's the capital of France?") {
        switch update := update.(type) {
        case llms.TextUpdate:
            fmt.Print(update.Text)
        }
    }
    
    // Check for errors after the chat completes
    if err := llm.Err(); err != nil {
        panic(err)
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
        return tools.SuccessWithLabel(p.Command, map[string]any{
            "output": fmt.Sprintf("Simulated output for: %s", p.Command),
        })
    },
)

func main() {
    // Create a new LLM instance using Anthropic's Claude with tools
    llm := llms.New(
        anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-3-7-sonnet-latest"),
        RunCommand,
    )

    // Chat with tool usage
    for update := range llm.Chat("List files in the current directory") {
        switch update := update.(type) {
        case llms.TextUpdate:
            fmt.Print(update.Text)
        case llms.ToolStartUpdate:
            fmt.Printf("(Using tool: %s)\n", update.Tool.Label()) // Shows "Run Command"
        case llms.ToolDoneUpdate:
            // Shows the label specific to this execution, e.g., "ls -l"
            fmt.Printf("(Tool result: %s)\n", update.Result.Label()) 
        }
    }
    
    // Check for errors after the chat completes
    if err := llm.Err(); err != nil {
        panic(err)
    }
}
```

## External Tools

Sometimes, you might have a set of predefined tool schemas (perhaps from an external source or another system) that you want the LLM to be able to use. `AddExternalTools` allows you to provide these schemas along with a single handler function.

This is useful when the logic for handling multiple tools is centralized, or when you need to dynamically add tools based on external definitions.

The handler function receives the `tools.Runner` and the raw JSON parameters for the called tool. You can use `llms.GetToolCall(r.Context())` within the handler to retrieve the specific `ToolCall` instance, which includes the function name (`tc.Name`) and unique call ID (`tc.ID`), allowing you to dispatch to the correct logic.

```go
// Example external tool schemas (could come from a config file, API, etc.)
var externalToolSchemas = []tools.FunctionSchema{
    {
        Name: "get_stock_price",
        /* ... */
    },
    {
        Name: "get_weather",
        /* ... */
    },
}

func main() {
    llm := llms.New(anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-3-7-sonnet-latest"))

    // Add external tools and their handler
    llm.AddExternalTools(externalToolSchemas, handleExternalTool)

    // Now the LLM can use "get_stock_price" and "get_weather"
    for update := range llm.Chat("What's the weather in London?") {
        switch update := update.(type) {
        case llms.TextUpdate:
            fmt.Print(update.Text)
        case llms.ToolStartUpdate:
            fmt.Printf("(Using tool: %s)\n", update.Tool.Label())
        case llms.ToolDoneUpdate:
            fmt.Printf("(Tool result: %s - %s)\n", update.Tool.Label(), update.Result.Label())
        }
    }
    if err := llm.Err(); err != nil {
        panic(err)
    }
}

// Single handler that forwards external tool calls.
func handleExternalTool(r tools.Runner, params json.RawMessage) tools.Result {
    // Get the specific tool call details from the context
    toolCall, ok := llms.GetToolCall(r.Context())
    if !ok {
        return tools.Errorf("Could not get tool call details from context")
    }

    // Typically, you would now:
    // 1. Construct a request to your external API endpoint (e.g., using http.Client).
    targetURL := fmt.Sprintf("https://api.example.com/tool?name=%s", toolCall.Name)
    req, err := http.NewRequestWithContext(r.Context(), "POST", targetURL, bytes.NewReader(params))
    // ... set headers, handle error ...

    // 2. Execute the request.
    resp, err := httpClient.Do(req)
    // ... handle error ...

    // 3. Process the response.
    defer resp.Body.Close()
    if resp.StatusCode != http.StatusOK {
        // Use Errorf or ErrorWithLabel for tool errors
        bodyBytes, _ := io.ReadAll(resp.Body)
        return tools.Errorf("External tool API failed (%s): %s", resp.Status, string(bodyBytes))
    }
    bodyBytes, err := io.ReadAll(resp.Body)
    // ... handle read error ...

    // 4. Return the result based on the response body.
    return tools.Success(json.RawMessage(bodyBytes))
}
```

## Provider Support

The library currently supports:

- Anthropic (Claude models)
- Google (Gemini API and Vertex AI)
- OpenAI (GPT/O models)

Each provider can be initialized with their respective configuration:

```go
// Anthropic
llm := llms.New(anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-3-7-sonnet-latest"))

// Google Gemini
llm := llms.New(google.New("gemini-pro").WithGeminiAPI(os.Getenv("GOOGLE_API_KEY")))

// Google Vertex AI
llm := llms.New(google.New("gemini-pro").WithVertexAI(accessToken, projectID, region))

// OpenAI
llm := llms.New(openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-4o"))
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
llms.New(openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-4o")).
    WithDebug()
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
