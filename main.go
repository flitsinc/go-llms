package main

import (
	"context"
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
	"github.com/flitsinc/go-llms/openrouter"
	"github.com/flitsinc/go-llms/tools"
)

func init() {
	// Put your API keys in .env and this will load them.
	godotenv.Overload()
}

// dim prints text in faded/dim terminal style.
func dim(format string, a ...any) { fmt.Printf("\033[2m"+format+"\033[0m", a...) }

func main() {
	// Check command-line arguments
	if len(os.Args) < 2 {
		printUsage()
		return
	}

	provider := os.Args[1]
	var llmProvider llms.Provider

	var cleanup func()
	switch provider {
	case "openai", "openai-responses", "openai-ws", "openai-ws-warmup":
		apiKey := os.Getenv("OPENAI_API_KEY")
		if apiKey == "" {
			fmt.Println("Error: OPENAI_API_KEY environment variable is not set")
			return
		}
		switch provider {
		case "openai-responses":
			llmProvider = openai.NewResponsesAPI(apiKey, "gpt-5.4").
				WithThinking(openai.EffortLow).
				WithVerbosity(openai.VerbosityLow)
		case "openai-ws", "openai-ws-warmup":
			wsProvider := openai.NewWebSocketResponsesAPI(apiKey, "gpt-5.4").
				WithThinking(openai.EffortLow).
				WithVerbosity(openai.VerbosityLow)
			llmProvider = wsProvider
			cleanup = func() { wsProvider.Close() }
		default:
			llmProvider = openai.New(apiKey, "gpt-5.4").
				WithThinking(openai.EffortLow).
				WithVerbosity(openai.VerbosityLow)
		}
	case "anthropic":
		apiKey := os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			fmt.Println("Error: ANTHROPIC_API_KEY environment variable is not set")
			return
		}
		llmProvider = anthropic.New(apiKey, "claude-sonnet-4-6").WithBeta("extended-cache-ttl-2025-04-11")
	case "google":
		apiKey := os.Getenv("GEMINI_API_KEY")
		if apiKey == "" {
			fmt.Println("Error: GEMINI_API_KEY environment variable is not set")
			return
		}
		llmProvider = google.New("gemini-3-flash-preview").
			WithThinkingLevel(google.ThinkingLevelLow).
			WithGeminiAPI(apiKey)
	case "groq":
		apiKey := os.Getenv("GROQ_API_KEY")
		if apiKey == "" {
			fmt.Println("Error: GROQ_API_KEY environment variable is not set")
			return
		}
		llmProvider = openai.New(apiKey, "moonshotai/kimi-k2-instruct").WithEndpoint("https://api.groq.com/openai/v1/chat/completions", "Groq")
	case "openrouter":
		apiKey := os.Getenv("OPENROUTER_API_KEY")
		if apiKey == "" {
			fmt.Println("Error: OPENROUTER_API_KEY environment variable is not set")
			return
		}
		model := "anthropic/claude-sonnet-4"
		if len(os.Args) > 2 {
			model = os.Args[2]
		}
		llmProvider = openrouter.New(apiKey, model)
	default:
		printUsage()
		return
	}
	if cleanup != nil {
		defer cleanup()
	}

	llm := llms.New(llmProvider, RunShellCmd)

	systemPrompt := "You're a helpful bot of few words. If at first you don't succeed, try again."

	// System prompt is dynamic so it can always be up-to-date.
	llm.SystemPrompt = func() content.Content {
		return content.Content{
			&content.Text{Text: systemPrompt},
			&content.CacheHint{Duration: "long"},
			&content.Text{Text: fmt.Sprintf(" The time is %s.", time.Now().Format(time.RFC1123))},
		}
	}

	// For WebSocket with warmup, connect + pre-load tools/instructions.
	// This is timed separately since it happens before the chat starts.
	if provider == "openai-ws-warmup" {
		if wsProvider, ok := llmProvider.(*openai.WebSocketResponsesAPI); ok {
			warmupStart := time.Now()
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
			_, err := wsProvider.Warmup(ctx, systemPrompt, llm.Toolbox())
			cancel()
			if err != nil {
				fmt.Printf("Warmup error: %v\n", err)
				return
			}
			dim("[connect+warmup %s]\n", time.Since(warmupStart).Round(time.Millisecond))
		}
	}

	var prevUpdate llms.UpdateType
	chatStart := time.Now()
	turnStart := chatStart
	// lastActivity tracks when the last "boundary" occurred for TTFT:
	// starts at chatStart, updated after each tool completes. This way
	// TTFT measures from request-sent (turn 1) or tool-done (turn 2+)
	// to first content, not from MessageStartUpdate which is already
	// part of the response stream.
	lastActivity := chatStart
	turn := 0
	turnFirstToken := false

	// Prompt designed to force at least 2 tool turns: one for listing files,
	// one for counting lines, then a final text response.
	prompt := "First, list the files in the current directory. Second, count the total lines of Go code (find . -name '*.go' | xargs wc -l). Finally, summarize what you found in 2 sentences."

	// llm.Chat returns a channel of updates.
	for update := range llm.Chat(prompt) {
		// Output formatting: Add two newlines before new update types.
		if t := update.Type(); prevUpdate != "" && t != prevUpdate && (t == llms.UpdateTypeText || t == llms.UpdateTypeThinking || t == llms.UpdateTypeToolStart) {
			fmt.Println()
			fmt.Println()
		}

		// Handle the update.
		switch update := update.(type) {
		case llms.MessageStartUpdate:
			if turn > 0 {
				dim(" [turn %d: %s]", turn, time.Since(turnStart).Round(time.Millisecond))
				fmt.Println()
			}
			turn++
			turnStart = time.Now()
			turnFirstToken = false
		case llms.ThinkingUpdate:
			if !turnFirstToken {
				turnFirstToken = true
				dim("[TTFT %s] ", time.Since(lastActivity).Round(time.Millisecond))
			}
			if prevUpdate != llms.UpdateTypeThinking {
				fmt.Print("\033[2m💭 ")
			}
			fmt.Print(update.Text)
		case llms.ThinkingDoneUpdate:
			fmt.Print("\033[0m")
		case llms.TextUpdate:
			if !turnFirstToken {
				turnFirstToken = true
				dim("[TTFT %s] ", time.Since(lastActivity).Round(time.Millisecond))
			}
			fmt.Print(update.Text)
		case llms.ToolStartUpdate:
			if !turnFirstToken {
				turnFirstToken = true
				dim("[TTFT %s] ", time.Since(lastActivity).Round(time.Millisecond))
			}
			fmt.Printf("(%s: ", update.Tool.Label())
		case llms.ToolDoneUpdate:
			fmt.Printf("%s)", update.Result.Label())
			lastActivity = time.Now()
		}
		prevUpdate = update.Type()
	}

	// Check for errors at the end of the chat
	if err := llm.Err(); err != nil {
		panic(err)
	}

	totalDur := time.Since(chatStart).Round(time.Millisecond)
	fmt.Println()
	fmt.Println()
	dim("Total: %s | Turns: %d | Usage: %+v\n", totalDur, max(turn, 1), llm.TotalUsage)
}

func printUsage() {
	fmt.Println("Usage: go run main.go <provider>")
	fmt.Println()
	fmt.Println("Supported providers:")
	fmt.Println("  openai            - Uses OpenAI Chat Completions with gpt-5.4 (requires OPENAI_API_KEY)")
	fmt.Println("  openai-responses  - Uses OpenAI Responses API with gpt-5.4 (requires OPENAI_API_KEY)")
	fmt.Println("  openai-ws         - Uses OpenAI Responses API over WebSocket (requires OPENAI_API_KEY)")
	fmt.Println("  openai-ws-warmup  - Same as openai-ws but with Warmup pre-loading (requires OPENAI_API_KEY)")
	fmt.Println("  anthropic         - Uses Anthropic's Claude Sonnet 4.6 (requires ANTHROPIC_API_KEY)")
	fmt.Println("  google            - Uses Google's Gemini 3 Flash (requires GEMINI_API_KEY)")
	fmt.Println("  groq              - Uses kimi-k2-instruct (requires GROQ_API_KEY)")
	fmt.Println("  openrouter        - Uses OpenRouter with any model (requires OPENROUTER_API_KEY)")
	fmt.Println("                      Optional: pass model as second arg (default: anthropic/claude-sonnet-4)")
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
