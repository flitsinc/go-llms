package main

import (
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/joho/godotenv"

	"github.com/flitsinc/go-llms/anthropic"
	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/google"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/openai"
	"github.com/flitsinc/go-llms/tools"
)

func init() {
	// Put your API keys in .env and this will load them.
	godotenv.Overload()
}

func main() {
	// Check command-line arguments
	if len(os.Args) < 2 {
		printUsage()
		return
	}

	provider := os.Args[1]
	var llmProvider llms.Provider

	switch provider {
	case "openai", "openai-responses":
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			fmt.Println("Error: OPENAI_API_KEY environment variable is not set")
			return
		}
		if provider == "openai-responses" {
			llmProvider = openai.NewResponsesAPI(apiKey, "o4-mini")
		} else {
			llmProvider = openai.New(apiKey, "o4-mini")
		}
	case "anthropic":
		apiKey := os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			fmt.Println("Error: ANTHROPIC_API_KEY environment variable is not set")
			return
		}
		llmProvider = anthropic.New(apiKey, "claude-sonnet-4-20250514").WithBeta("extended-cache-ttl-2025-04-11")
	case "google":
		apiKey := os.Getenv("GEMINI_API_KEY")
		if apiKey == "" {
			fmt.Println("Error: GEMINI_API_KEY environment variable is not set")
			return
		}
		llmProvider = google.New("gemini-2.5-flash").WithGeminiAPI(apiKey)
	default:
		printUsage()
		return
	}

	llm := llms.New(llmProvider, RunShellCmd)

	// System prompt is dynamic so it can always be up-to-date.
	llm.SystemPrompt = func() content.Content {
		return content.Content{
			&content.Text{Text: "You're a helpful bot of few words. If at first you don't succeed, try again."},
			&content.CacheHint{Duration: "long"},
			&content.Text{Text: fmt.Sprintf(" The time is %s.", time.Now().Format(time.RFC1123))},
		}
	}

	// Chat returns a channel of updates.
	for update := range llm.Chat("Give me a random number between 1 and 100! Then tell me a poem about it.") {
		switch update := update.(type) {
		case llms.TextUpdate:
			// Received for each chunk of text from the LLM.
			fmt.Print(update.Text)
		case llms.ToolStartUpdate:
			// Received the moment the LLM streams that it intends to use a tool.
			fmt.Printf("(%s: ", update.Tool.Label())
		case llms.ToolDoneUpdate:
			// Received after the LLM finished sending arguments and the tool ran.
			fmt.Printf("%s)\n", update.Result.Label())
		}
	}

	// Check for errors at the end of the chat
	if err := llm.Err(); err != nil {
		panic(err)
	}

	fmt.Println()
}

func printUsage() {
	fmt.Println("Usage: go run main.go <provider>")
	fmt.Println()
	fmt.Println("Supported providers:")
	fmt.Println("  openai           - Uses OpenAI's o4-mini (requires OPENAI_API_KEY)")
	fmt.Println("  openai-responses - Uses OpenAI's Responses API with o4-mini (requires OPENAI_API_KEY)")
	fmt.Println("  anthropic        - Uses Anthropic's Claude Sonnet 4 (requires ANTHROPIC_API_KEY)")
	fmt.Println("  google           - Uses Google's Gemini 2.5 Flash (requires GEMINI_API_KEY)")
	fmt.Println()
	fmt.Println("Environment variables can be set directly or loaded from a .env file.")
	fmt.Println()
	fmt.Println("Example:")
	fmt.Println("  OPENAI_API_KEY=your-key go run main.go openai")
}

// How to define a tool:

type RunShellCmdParams struct {
	Command string `json:"command" description:"The shell command to run"`
}

var RunShellCmd = tools.Func(
	"Run shell command",
	"Run a shell command on the user's computer and return the output",
	"run_shell_cmd",
	func(r tools.Runner, p RunShellCmdParams) tools.Result {
		// Run the shell command and capture the output or error.
		cmd := exec.CommandContext(r.Context(), "sh", "-c", p.Command)
		output, err := cmd.CombinedOutput() // Combines both STDOUT and STDERR
		if err != nil {
			return tools.ErrorWithLabel(fmt.Sprintf("%s \033[31m(%d)\033[0m", p.Command, cmd.ProcessState.ExitCode()), fmt.Errorf("%w: %s", err, output))
		}
		return tools.SuccessWithLabel(p.Command, map[string]any{"output": string(output)})
	})
