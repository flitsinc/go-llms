# go-llms

A powerful and flexible Go library for interacting with Large Language Models (LLMs) with built-in support for function calling and streaming responses. Currently supports Anthropic, Google, and OpenAI compatible providers.

## Features

- Supports Anthropic, Google (Gemini + Vertex), and OpenAI (Chat Completions + Responses)
- Also supports using custom endpoints compatible with any of the APIs above
- Streaming responses (including thinking) for real-time interaction
- Built-in tool calling with Go generics to generate JSON schemas automatically
- Structured output (JSON schema based) for model response
- OpenAI only: Custom grammars like Lark / Regex for structured output
- Prompt cache hints
- Image inputs
- Usage tracking

### On the roadmap

- [ ] Text diffusion model support (Inception, Google)
- [ ] Realtime streaming (WebRTC)
- [ ] Image output

## Installation

```bash
go get github.com/flitsinc/go-llms
```

## Quick Start

Here’s a simple example that creates an LLM instance and has a conversation with it:

```go
package main

import (
    "fmt"
    "os"

    "github.com/flitsinc/go-llms/content"
    "github.com/flitsinc/go-llms/llms"
    "github.com/flitsinc/go-llms/openai"
    "github.com/flitsinc/go-llms/tools"
)

func main() {
    // Create a new LLM instance with OpenAI's o4-mini model
    llm := llms.New(
        openai.NewResponsesAPI(os.Getenv("OPENAI_API_KEY"), "o4-mini"),
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

Here’s an example showing how to use tools (function calling):

```go
package main

import (
    "fmt"
    "os"

    "github.com/flitsinc/go-llms/anthropic"
    "github.com/flitsinc/go-llms/llms"
    "github.com/flitsinc/go-llms/tools"
)

// Define tool parameters
type CommandParams struct {
    Command string `json:"command" description:"The shell command to run"`
}

// Create a shell command tool
var RunCommand = tools.Func(
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
        anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-20250514"),
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
    llm := llms.New(anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-20250514"))

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
    // 1. Construct a request to your external API endpoint (or send it to a browser client)
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

## Grammar-Based Tools (OpenAI Only)

OpenAI supports custom tools that can enforce specific input formats using grammars. This allows you to constrain the model's output to follow precise patterns, which is useful for structured data extraction, validation, or parsing tasks.

The library supports three grammar types:

- **Lark Grammar**: For complex parsing using [Lark parser syntax](https://lark-parser.readthedocs.io/)
- **Regex Grammar**: For pattern matching using regular expressions
- **Text Grammar**: For free-form text input

Here are examples of grammar-based tools:

```go
package main

import (
    "fmt"
    "os"
    "regexp"
    "strconv"
    "strings"

    "github.com/flitsinc/go-llms/llms"
    "github.com/flitsinc/go-llms/openai"
    "github.com/flitsinc/go-llms/tools"
)

// Example 1: Lark Grammar for Math Expressions
var mathGrammar = tools.Lark(`
start: expr
expr: term (SP ADD SP term)* -> add
    | term
term: factor (SP MUL SP factor)* -> mul
    | factor
factor: INT
SP: " "
ADD: "+"
MUL: "*"
%import common.INT
`)

var MathTool = tools.FuncGrammar(
    mathGrammar,
    "Math Calculator",
    "Evaluate simple math expressions with addition and multiplication",
    "calculate_math",
    func(r tools.Runner, expression string) tools.Result {
        // Simple evaluation for demo (in practice, use a proper parser)
        expression = strings.ReplaceAll(expression, " ", "")
        
        // Handle simple cases for demonstration
        if strings.Contains(expression, "+") {
            parts := strings.Split(expression, "+")
            if len(parts) == 2 {
                a, err1 := strconv.Atoi(parts[0])
                b, err2 := strconv.Atoi(parts[1])
                if err1 == nil && err2 == nil {
                    result := a + b
                    return tools.SuccessWithLabel(expression, map[string]any{
                        "result": result,
                        "expression": expression,
                    })
                }
            }
        }
        
        // Fallback for single numbers
        if num, err := strconv.Atoi(expression); err == nil {
            return tools.SuccessWithLabel(expression, map[string]any{
                "result": num,
                "expression": expression,
            })
        }
        
        return tools.ErrorWithLabel("Invalid expression", fmt.Errorf("could not parse: %s", expression))
    },
)

// Example 2: Regex Grammar for Email Validation
var emailGrammar = tools.Regex(`^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$`)

var EmailValidator = tools.FuncGrammar(
    emailGrammar,
    "Email Validator",
    "Validate and extract information from email addresses",
    "validate_email",
    func(r tools.Runner, email string) tools.Result {
        emailRegex := regexp.MustCompile(`^([a-zA-Z0-9._%+-]+)@([a-zA-Z0-9.-]+\.[a-zA-Z]{2,})$`)
        matches := emailRegex.FindStringSubmatch(email)
        
        if len(matches) != 3 {
            return tools.ErrorWithLabel("Invalid email", fmt.Errorf("email format is invalid: %s", email))
        }
        
        username := matches[1]
        domain := matches[2]
        
        return tools.SuccessWithLabel(email, map[string]any{
            "email": email,
            "username": username,
            "domain": domain,
            "is_valid": true,
        })
    },
)

func main() {
    // Create LLM with grammar-based tools (OpenAI only)
    llm := llms.New(
        openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-5"),
        MathTool,
        EmailValidator,
    )

    // The model can now use structured tools with grammar constraints
    for update := range llm.Chat("Calculate 15 + 27 and validate the email user@example.com") {
        switch update := update.(type) {
        case llms.TextUpdate:
            fmt.Print(update.Text)
        case llms.ToolStartUpdate:
            fmt.Printf("(Using %s: ", update.Tool.Label())
        case llms.ToolDoneUpdate:
            fmt.Printf("%s)\n", update.Result.Label())
        }
    }
    
    if err := llm.Err(); err != nil {
        panic(err)
    }
}
```

**Note**: Grammar-based tools are currently only supported by OpenAI's API. Other providers currently do not support grammar-based tools.

## Provider Support

The library currently supports:

- Anthropic
- Google (Gemini API and Vertex AI)
- OpenAI and all compatible providers (you can customize the endpoint)
- OpenAI’s newer Responses API

Each provider can be initialized with their respective configuration:

```go
// Anthropic
llm := llms.New(anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-20250514"))

// Google Gemini
llm := llms.New(google.New("gemini-2.5-flash").WithGeminiAPI(os.Getenv("GOOGLE_API_KEY")))

// Google Vertex AI
ts, err := googleoauth.DefaultTokenSource(ctx, "https://www.googleapis.com/auth/cloud-platform")
llm := llms.New(google.New("gemini-2.5-flash").WithVertexAI(ts, projectID, "global"))

// OpenAI (Responses API)
llm := llms.New(openai.NewResponsesAPI(os.Getenv("OPENAI_API_KEY"), "o4-mini"))

// OpenAI (Chat Completions API)
llm := llms.New(openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-4.1"))

// OpenAI-compatible endpoint (e.g., xAI)
// You can use the OpenAI provider with compatible APIs by configuring the endpoint.
llm := llms.New(
    openai.New(os.Getenv("XAI_API_KEY"), "grok-3-latest").
        WithEndpoint("https://api.x.ai/v1/chat/completions", "xAI"),
)
```

You can easily implement new providers by implementing the `Provider` interface:

```go
type Provider interface {
    Company() string
    Model() string
    // Generate takes a system prompt, message history, optional toolbox, and an optional JSON output schema.
    // The provider should respect the context for cancellation during its operations.
    Generate(
        ctx context.Context,
        systemPrompt content.Content,
        messages []Message,
        toolbox *tools.Toolbox,
        jsonOutputSchema *tools.ValueSchema,
    ) ProviderStream
}

type ProviderStream interface {
    Err() error
    Iter() func(yield func(StreamStatus) bool)
    Message() Message
    Text() string
    Thought() content.Thought
    ToolCall() ToolCall
    Usage() Usage
}
```

## Usage Tracking

Track the usage of your LLM interactions:

```go
usage := llm.TotalUsage
fmt.Printf("Cached Input Tokens: %d, Input Tokens: %d, Output Tokens: %d\n",
    usage.CachedInputTokens, usage.InputTokens, usage.OutputTokens)
```

As patterns emerge between providers with regards to cache tokens, speculative tokens, etc. these will be added too.

## When to use this?

When you want to make providers easily swappable and a simplified API that focuses on hekoing you implement the most common types of agentic flows.

Since each LLM provider has its own quirks, especially around reasoning, streaming, and tool calling, we’ve done our best to smooth those over, but expect some differences still.

### Provider quirks

#### `additionalProperties` forbidden and required depending on provider

Google doesn’t allow the `additionalProperties` field for JSON schemas (probably a bug), while OpenAI’s new Responses API requires it for tool calls! It’s also commonly required for models with strict JSON outputs since it helps with speculative decoding.

Because of this, we strip out the `additionalProperties` field before sending it to Google, so it shouldn’t be a problem for you, just keep it in mind.

#### Anthropic doesn’t stream partial property values by default

The streaming API of Anthropic only sends complete string values when streaming tool calls, so if you have a tool call like `edit_file` which produces very long fields nothing will update until that field has completely finished generating.

To fix this, use [fine-grained tool streaming](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/fine-grained-tool-streaming) which is currently in beta, by calling `.WithBeta("fine-grained-tool-streaming-2025-05-14")` on the Anthropic provider instance.

## License

MIT License - See LICENSE file for details.
