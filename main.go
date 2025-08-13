package main

import (
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/joho/godotenv"
	"github.com/maja42/goval"

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
			llmProvider = openai.NewResponsesAPI(apiKey, "gpt-5").
				WithThinking(openai.EffortLow).
				WithVerbosity(openai.VerbosityLow)
		} else {
			llmProvider = openai.New(apiKey, "gpt-5").
				WithThinking(openai.EffortLow).
				WithVerbosity(openai.VerbosityLow)
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
	case "groq":
		apiKey := os.Getenv("GROQ_API_KEY")
		if apiKey == "" {
			fmt.Println("Error: GROQ_API_KEY environment variable is not set")
			return
		}
		llmProvider = openai.New(apiKey, "moonshotai/kimi-k2-instruct").WithEndpoint("https://api.groq.com/openai/v1/chat/completions", "Groq")
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

	var prevUpdate llms.UpdateType

	// llm.Chat returns a channel of updates.
	for update := range llm.Chat("List the files in the current directory. Then tell me a poem about it.") {
		// Output formatting: Add two newlines before new update types.
		if t := update.Type(); prevUpdate != "" && t != prevUpdate && (t == llms.UpdateTypeText || t == llms.UpdateTypeThinking || t == llms.UpdateTypeToolStart) {
			fmt.Println()
			fmt.Println()
		}

		// Handle the update.
		switch update := update.(type) {
		case llms.ThinkingUpdate:
			// Show a thinking bubble and dim the text for thinking blocks.
			if prevUpdate != llms.UpdateTypeThinking {
				fmt.Print("\033[2m💭 ")
			}
			fmt.Print(update.Text)
		case llms.ThinkingDoneUpdate:
			// End of thinking; restore color and break line.
			fmt.Print("\033[0m")
		case llms.TextUpdate:
			// Print each chunk of text from the LLM as they come in.
			fmt.Print(update.Text)
		case llms.ToolStartUpdate:
			// Print the tool name when the LLM streams that it intends to use a tool.
			fmt.Printf("(%s: ", update.Tool.Label())
		case llms.ToolDoneUpdate:
			// Print the tool result when the LLM finished sending arguments and the tool ran.
			fmt.Printf("%s)", update.Result.Label())
		}
		prevUpdate = update.Type()
	}

	// Check for errors at the end of the chat
	if err := llm.Err(); err != nil {
		panic(err)
	}

	fmt.Println()
	fmt.Println()
	fmt.Printf("Usage: %+v\n", llm.TotalUsage)
}

func printUsage() {
	fmt.Println("Usage: go run main.go <provider>")
	fmt.Println()
	fmt.Println("Supported providers:")
	fmt.Println("  openai           - Uses OpenAI's gpt-5 (requires OPENAI_API_KEY)")
	fmt.Println("  openai-responses - Uses OpenAI's Responses API with gpt-5 (requires OPENAI_API_KEY)")
	fmt.Println("  anthropic        - Uses Anthropic's Claude Sonnet 4 (requires ANTHROPIC_API_KEY)")
	fmt.Println("  google           - Uses Google's Gemini 2.5 Flash (requires GEMINI_API_KEY)")
	fmt.Println("  groq             - Uses kimi-k2-instruct (requires GROQ_API_KEY)")
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

// Example of a custom tool that uses a Lark grammar (OpenAI only!)

var mathExpr = tools.Lark(`
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

var DoMath = tools.FuncGrammar(
	mathExpr,
	"Do some math",
	"Evaluate a math expression and return the result",
	"do_math",
	func(r tools.Runner, expr string) tools.Result {
		// Evaluate the math expression and return the result.
		eval := goval.NewEvaluator()
		result, err := eval.Evaluate(expr, nil, nil)
		if err != nil {
			return tools.ErrorWithLabel("Math evaluation failed", err)
		}
		return tools.SuccessWithLabel(expr, result)
	},
)
