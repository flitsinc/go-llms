package llms_test

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"time"

	"github.com/blixt/go-llms/anthropic"
	"github.com/blixt/go-llms/content"
	"github.com/blixt/go-llms/google"
	"github.com/blixt/go-llms/llms"
	"github.com/blixt/go-llms/openai"
	"github.com/blixt/go-llms/tools"
)

// This example demonstrates basic chat functionality using Anthropic.
func ExampleLLM_Chat() {
	// Note: Requires ANTHROPIC_API_KEY environment variable to be set.
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Println("ANTHROPIC_API_KEY environment variable not set.")
		// Skip example if key is not set.
		return
	}

	// Create a new LLM instance with Anthropic's Claude Sonnet model
	llm := llms.New(
		anthropic.New(apiKey, "claude-3-7-sonnet-latest"),
	)

	// Optional: Set a system prompt
	llm.SystemPrompt = func() content.Content {
		return content.FromText("You are a helpful assistant.")
	}

	fmt.Println("User: What's the capital of France?")
	fmt.Print("Assistant: ")

	// Start a chat conversation
	for update := range llm.Chat("What's the capital of France?") {
		switch update := update.(type) {
		case llms.TextUpdate:
			fmt.Print(update.Text) // Simulating streaming output
		}
	}
	fmt.Println() // Add a newline after the stream

	// Check for errors after the chat completes
	if err := llm.Err(); err != nil {
		log.Fatalf("Chat failed: %v", err)
	}

	/*
		Example Interaction:

		User: What's the capital of France?
		Assistant: The capital of France is Paris.
	*/
}

// This example demonstrates chatting with context using Google Gemini.
func ExampleLLM_ChatWithContext() {
	// Note: Requires GOOGLE_API_KEY environment variable to be set.
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		fmt.Println("GOOGLE_API_KEY environment variable not set.")
		// Skip example if key is not set.
		return
	}

	// Create a new LLM instance with Google's Gemini Flash model
	llm := llms.New(
		google.New("gemini-2.5-flash-preview-04-17").WithGeminiAPI(apiKey),
	)

	// Create a context with a timeout
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	fmt.Println("User: Tell me a short story.")
	fmt.Print("Assistant: ")

	// Start a chat conversation with context
	for update := range llm.ChatWithContext(ctx, "Tell me a short story.") {
		switch update := update.(type) {
		case llms.TextUpdate:
			fmt.Print(update.Text)
		}
	}
	fmt.Println()

	// Check for errors (including context cancellation)
	if err := llm.Err(); err != nil {
		// Note: Check err against context.DeadlineExceeded, context.Canceled, etc.
		log.Printf("Chat finished with error: %v", err)
	}

	/*
		Example Interaction:

		User: Tell me a short story.
		Assistant: Once upon a time, in a land filled with rolling green hills, lived a curious rabbit named Pip. Pip loved exploring... (output may vary)
	*/
}

// This example demonstrates using tools (function calling) with OpenAI.
func ExampleLLM_Chat_withTools() {
	// Note: Requires OPENAI_API_KEY environment variable to be set.
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("OPENAI_API_KEY environment variable not set.")
		// Skip example if key is not set.
		return
	}

	// Define tool parameters struct within the example
	type CommandParams struct {
		Command string `json:"command" description:"The shell command to run"`
	}

	// Create a shell command tool (simulated execution) within the example
	RunCommand := tools.Func(
		"Run Command", // Label for the tool type
		"Run a shell command and return the output", // Description
		"run_command", // Name used by the LLM
		func(r tools.Runner, p CommandParams) tools.Result {
			// We use SuccessWithLabel to provide a dynamic label for this specific execution.
			return tools.SuccessWithLabel(fmt.Sprintf("Executed '%s'", p.Command), map[string]any{
				"output": fmt.Sprintf("Simulated output for: %s", p.Command),
			})
		},
	)

	// Create an LLM instance with the RunCommand tool using OpenAI
	llm := llms.New(
		openai.New(apiKey, "gpt-4o-mini"), // Use a model known for tool use
		RunCommand,                        // Register the tool
	)

	fmt.Println("User: List files in the current directory.")
	fmt.Print("Assistant:\n")

	// Start a chat conversation that might involve tools
	for update := range llm.Chat("List files in the current directory using the run_command tool.") {
		switch update := update.(type) {
		case llms.TextUpdate:
			fmt.Print(update.Text)
		case llms.ToolStartUpdate:
			// Shows the generic label for the tool type being started
			fmt.Printf("(System: Using tool: %s)\n", update.Tool.Label()) // e.g., "Run Command"
		case llms.ToolStatusUpdate:
			// You can optionally report status updates from the tool runner
			fmt.Printf("(System: Tool status: %s - %s)\n", update.Tool.Label(), update.Status)
		case llms.ToolDoneUpdate:
			// Shows the potentially dynamic label returned by the tool result.
			fmt.Printf("(System: Tool result: %s)\n", update.Result.Label())
		}
	}
	fmt.Println() // Add a newline after the stream

	if err := llm.Err(); err != nil {
		log.Fatalf("Chat failed: %v", err)
	}

	/*
		Example Interaction (output depends heavily on model and tool execution):

		User: List files in the current directory.
		Assistant:
		(System: Using tool: Run Command)
		(System: Tool result: Executed 'ls -l')
		Okay, I have simulated running the command. The output is: Simulated output for: ls -l
	*/
}

// This example demonstrates enabling debug mode with Google Gemini.
func ExampleLLM_WithDebug() {
	// Note: Requires GOOGLE_API_KEY environment variable to be set.
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		fmt.Println("GOOGLE_API_KEY environment variable not set.")
		// Skip example if key is not set.
		return
	}

	// Create LLM with Google Gemini and enable debug mode
	llm := llms.New(
		google.New("gemini-2.5-flash-preview-04-17").WithGeminiAPI(apiKey),
	).WithDebug() // Enable debug logging to debug.yaml

	// Subsequent calls to llm.Chat() will write detailed logs.
	fmt.Println("Debug mode enabled. Interactions will be logged to debug.yaml.")

	// Perform a simple chat to generate some debug output
	for update := range llm.Chat("Hello!") {
		switch update := update.(type) {
		case llms.TextUpdate:
			fmt.Print(update.Text)
		}
	}
	fmt.Println()

	if err := llm.Err(); err != nil {
		log.Printf("Chat failed: %v", err)
	}

	/*
		Example Interaction:

		Debug mode enabled. Interactions will be logged to debug.yaml.
		Hello there! How can I help you today?
	*/
}

// This example demonstrates setting a maximum number of LLM turns with Anthropic.
func ExampleLLM_WithMaxTurns() {
	// Note: Requires ANTHROPIC_API_KEY environment variable to be set.
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Println("ANTHROPIC_API_KEY environment variable not set.")
		// Skip example if key is not set.
		return
	}

	// Create LLM with Anthropic Claude Sonnet and limit to 1 turn
	llm := llms.New(
		anthropic.New(apiKey, "claude-3-7-sonnet-latest"),
	).WithMaxTurns(1)

	// Perform a simple chat
	for update := range llm.Chat("Why is the sky blue?") {
		switch update := update.(type) {
		case llms.TextUpdate:
			fmt.Print(update.Text)
		}
	}
	fmt.Println()

	// If the conversation required more turns (e.g., complex tool use),
	// llm.Err() might return llms.ErrMaxTurnsReached.
	if err := llm.Err(); err != nil {
		if err == llms.ErrMaxTurnsReached {
			fmt.Println("Max turns reached as expected.")
		} else {
			log.Printf("Chat failed with unexpected error: %v", err)
		}
	} else {
		fmt.Println("Chat completed within max turns.")
	}

	/*
		Example Interaction (output may vary):

		The sky appears blue due to a phenomenon called Rayleigh scattering...
		Chat completed within max turns.
	*/
}

// This example demonstrates adding and using external tools with OpenAI.
func ExampleLLM_AddExternalTools() {
	// Note: Requires OPENAI_API_KEY environment variable to be set.
	apiKey := os.Getenv("OPENAI_API_KEY")
	if apiKey == "" {
		fmt.Println("OPENAI_API_KEY environment variable not set.")
		// Skip example if key is not set.
		return
	}

	// Define external tool schemas within the example
	externalToolSchemas := []tools.FunctionSchema{
		{
			Name:        "get_weather",
			Description: "Get the current weather for a location",
			Parameters: tools.ValueSchema{
				Type: "object",
				Properties: &map[string]tools.ValueSchema{
					"location": {
						Type:        "string",
						Description: "The city and state, e.g. San Francisco, CA",
					},
				},
				Required: []string{"location"},
			},
		},
	}

	// Define single handler for external tools within the example
	handleExternalTool := func(r tools.Runner, params json.RawMessage) tools.Result {
		// Get the specific tool call details (Name, ID) from the context
		toolCall, ok := llms.GetToolCall(r.Context())
		if !ok {
			return tools.Errorf("Could not get tool call details from context")
		}

		// Decode parameters based on the tool name
		switch toolCall.Name {
		case "get_weather":
			var weatherParams struct {
				Location string `json:"location"`
			}
			if err := json.Unmarshal(params, &weatherParams); err != nil {
				return tools.Errorf("Invalid parameters for get_weather: %v", err)
			}
			// Simulate calling an external weather API and return data with a dynamic label.
			return tools.SuccessWithLabel(fmt.Sprintf("Weather for %s", weatherParams.Location), map[string]any{
				"location":    weatherParams.Location,
				"temperature": "70F",
				"condition":   "Sunny",
			})
		default:
			return tools.Errorf("Unknown external tool: %s", toolCall.Name)
		}
	}

	// Create an LLM instance with OpenAI (without tools initially)
	llm := llms.New(openai.New(apiKey, "gpt-4o-mini"))

	// Add external tools and their single handler
	llm.AddExternalTools(externalToolSchemas, handleExternalTool)

	fmt.Println("User: What's the weather in London?")
	fmt.Print("Assistant:\n")

	// Start a chat using the externally defined tool
	for update := range llm.Chat("What's the weather in London?") {
		switch update := update.(type) {
		case llms.TextUpdate:
			fmt.Print(update.Text)
		case llms.ToolStartUpdate:
			// Note: The Tool.Label() for external tools defaults to the Name.
			fmt.Printf("(System: Using tool: %s)\n", update.Tool.Label()) // Shows "get_weather"
		case llms.ToolDoneUpdate:
			// Shows the potentially dynamic label returned by the tool result.
			fmt.Printf("(System: Tool result: %s)\n", update.Result.Label())
		}
	}
	fmt.Println()

	if err := llm.Err(); err != nil {
		log.Fatalf("Chat failed: %v", err)
	}

	/*
		Example Interaction (output depends heavily on model and tool execution):

		User: What's the weather in London?
		Assistant:
		(System: Using tool: get_weather)
		(System: Tool result: Weather for London)
		The weather in London is currently Sunny with a temperature of 70F.
	*/
}
