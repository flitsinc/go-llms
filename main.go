package main

import (
	"fmt"
	"os"
	"os/exec"
	"time"

	"github.com/joho/godotenv"

	"github.com/blixt/go-llms/anthropic"
	"github.com/blixt/go-llms/content"
	"github.com/blixt/go-llms/llms"
	"github.com/blixt/go-llms/tools"
)

func init() {
	// Put your API keys in .env and this will load them.
	godotenv.Overload()
}

func main() {
	llm := llms.New(
		// openai.New(os.Getenv("OPENAI_API_KEY"), "gpt-4o"),
		anthropic.New(os.Getenv("ANTHROPIC_API_KEY"), "claude-3-5-sonnet-latest"),
		RunShellCmd,
	)

	// System prompt is dynamic so it can always be up-to-date.
	llm.SystemPrompt = func() content.Content {
		return content.Textf("The time is %s. You're a helpful bot of few words. If at first you don't succeed, try again.", time.Now().Format(time.RFC1123))
	}

	// Chat returns a channel of updates.
	for update := range llm.Chat("Give me a random number between 1 and 100!") {
		switch update := update.(type) {
		case llms.ErrorUpdate:
			panic(update.Error)
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
		cmd := exec.Command("sh", "-c", p.Command)
		output, err := cmd.CombinedOutput() // Combines both STDOUT and STDERR
		if err != nil {
			return tools.Error(fmt.Sprintf("%s \033[31m(%d)\033[0m", p.Command, cmd.ProcessState.ExitCode()), fmt.Errorf("%w: %s", err, output))
		}
		return tools.Success(p.Command, map[string]any{"output": string(output)})
	})
