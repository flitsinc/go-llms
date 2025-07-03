package main

import (
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/joho/godotenv"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/google"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

func init() {
	// Put your API keys in .env and this will load them.
	godotenv.Overload()
}

func main() {
	llm := llms.New(
		// openai.New(os.Getenv("OPENAI_API_KEY"), "o4-mini"),
		// anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-20250514"),
		google.New("gemini-2.5-flash").WithGeminiAPI(os.Getenv("GEMINI_API_KEY")),
		RunShellCmd,
	)

	// System prompt is dynamic so it can always be up-to-date.
	llm.SystemPrompt = func() content.Content {
		return content.Textf("The time is %s. You're a helpful bot of few words. If at first you don't succeed, try again.", time.Now().Format(time.RFC1123))
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
