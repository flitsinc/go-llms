package llms

import (
	"encoding/json"
	"fmt"
	"os"
	"slices"

	"sigs.k8s.io/yaml"

	"github.com/blixt/go-llms/content"
	"github.com/blixt/go-llms/tools"
)

type LLM struct {
	provider Provider
	messages []Message
	toolbox  *tools.Toolbox

	totalCost float64
	debug     bool

	// SystemPrompt should return the system prompt for the LLM. It's a function
	// to allow the system prompt to dynamically change throughout a single
	// conversation.
	SystemPrompt func() content.Content
}

func New(provider Provider, allTools ...tools.Tool) *LLM {
	var toolbox *tools.Toolbox
	if len(allTools) > 0 {
		toolbox = tools.Box(allTools...)
	}
	return &LLM{
		provider: provider,
		toolbox:  toolbox,
	}
}

// Chat sends a text message to the LLM and immediately returns a channel over
// which updates will come in. The LLM will use the tools available and keep
// generating more messages until it's done using tools.
func (l *LLM) Chat(message string) <-chan Update {
	return l.ChatUsingContent(content.FromText(message))
}

// ChatUsingContent sends a message (which can contain images) to the LLM and
// immediately returns a channel over which updates will come in. The LLM will
// use the tools available and keep generating more messages until it's done
// using tools.
func (l *LLM) ChatUsingContent(message content.Content) <-chan Update {
	l.messages = append(l.messages, Message{
		Role:    "user",
		Content: message,
	})

	// Send off the user's message to the LLM, and keep asking the LLM for more
	// responses for as long as it's making tool calls.
	updateChan := make(chan Update)
	go func() {
		defer close(updateChan)
		for {
			shouldContinue, err := l.step(updateChan)
			if err != nil {
				updateChan <- ErrorUpdate{Error: err}
				return
			}
			if !shouldContinue {
				return
			}
		}
	}()

	return updateChan
}

func (l *LLM) AddTool(t tools.Tool) {
	if l.toolbox == nil {
		l.toolbox = tools.Box(t)
	} else {
		l.toolbox.Add(t)
	}
}

func (l *LLM) TotalCost() float64 {
	return l.totalCost
}

func (l *LLM) SetDebug(enabled bool) {
	l.debug = enabled
}

func (l *LLM) step(updateChan chan<- Update) (bool, error) {
	var systemPrompt content.Content
	if l.SystemPrompt != nil {
		systemPrompt = l.SystemPrompt()
	}

	// This will hold results from tool calls, to be sent back to the LLM.
	var toolMessages []Message

	stream := l.provider.Generate(systemPrompt, l.messages, l.toolbox)
	if err := stream.Err(); err != nil {
		return false, fmt.Errorf("LLM returned error response: %w", err)
	}

	// Write the entire message history to the file debug.yaml. The function is
	// deferred so that we get data even if a panic occurs.
	defer func() {
		if !l.debug {
			return
		}
		var toolsSchema []*tools.FunctionSchema
		if l.toolbox != nil {
			for _, tool := range l.toolbox.All() {
				toolsSchema = append(toolsSchema, tool.Schema())
			}
		}
		debugData := map[string]any{
			// Prefixed with numbers so the keys remain in this order.
			"1_receivedMessage": stream.Message(),
			"2_toolResults":     toolMessages,
			"3_sentMessages":    l.messages,
			"4_systemPrompt":    systemPrompt,
			"5_availableTools":  toolsSchema,
		}
		if debugYAML, err := yaml.Marshal(debugData); err == nil {
			os.WriteFile("debug.yaml", debugYAML, 0644)
		}
	}()

	// A future version of Go will allow us to use range on the function below:
	// for status := range stream.Iter() {
	stream.Iter()(func(status StreamStatus) bool {
		switch status {
		case StreamStatusText:
			updateChan <- TextUpdate{Text: stream.Text()}
		case StreamStatusToolCallBegin:
			tool := l.toolbox.Get(stream.ToolCall().Name)
			if tool == nil {
				// TODO: This should be handled more gracefully.
				panic(fmt.Sprintf("tool %q not found", stream.ToolCall().Name))
			}
			updateChan <- ToolStartUpdate{Tool: tool}
		case StreamStatusToolCallReady:
			messages := l.runToolCall(l.toolbox, stream.ToolCall(), updateChan)
			toolMessages = append(toolMessages, messages...)
		}
		return true
	})

	if err := stream.Err(); err != nil {
		return false, fmt.Errorf("error streaming: %w", err)
	}

	// Add the fully assembled message plus tool call results to the message history.
	l.messages = append(l.messages, stream.Message())
	// Role "tool" must always come first.
	slices.SortStableFunc(toolMessages, func(a, b Message) int {
		if a.Role == "tool" && b.Role != "tool" {
			return -1
		}
		if a.Role != "tool" && b.Role == "tool" {
			return 1
		}
		return 0
	})
	l.messages = append(l.messages, toolMessages...)

	l.totalCost += stream.CostUSD()

	// Return true if there were tool calls, since the LLM should look at the results.
	return len(toolMessages) > 0, nil
}

func (l *LLM) runToolCall(toolbox *tools.Toolbox, toolCall ToolCall, updateChan chan<- Update) []Message {
	if toolCall.ID != "" {
		// As a sanity check, make sure we don't try to run the same tool call twice.
		for _, message := range l.messages {
			if message.ToolCallID == toolCall.ID {
				fmt.Printf("\ntool call %q (%s) has already been run\n", toolCall.ID, toolCall.Name)
			}
		}
	}

	t := toolbox.Get(toolCall.Name)
	runner := tools.NewRunner(toolbox, func(status string) {
		updateChan <- ToolStatusUpdate{Status: status, Tool: t}
	})

	result := toolbox.Run(runner, toolCall.Name, json.RawMessage(toolCall.Arguments))
	updateChan <- ToolDoneUpdate{Result: result, Tool: t}

	callID := toolCall.ID
	if callID == "" {
		callID = toolCall.Name
	}

	messages := []Message{
		{
			Role:       "tool",
			Content:    content.FromRawJSON(result.JSON()),
			ToolCallID: callID,
		},
	}

	if images := result.Images(); len(images) > 0 {
		// "tool" messages can't actually contain image content. So we need to
		// fake an assistant message instead.
		message := Message{
			Role: "user",
			// TODO: Support more than one image name.
			Content: content.Textf("Here is %s. This is an automated message, not actually from the user.", images[0].Name),
		}
		for _, image := range images {
			message.Content.AddImage(image.URL)
		}
		messages = append(messages, message)
	}

	return messages
}
