package llms

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"slices"

	"sigs.k8s.io/yaml"

	"github.com/blixt/go-llms/content"
	"github.com/blixt/go-llms/tools"
)

// LLM represents the interface to an LLM provider, maintaining state between
// individual calls, for example when tool calling is being performed. Note that
// this is NOT thread safe for this reason.
type LLM struct {
	provider         Provider
	lastSentMessages []Message
	toolbox          *tools.Toolbox

	totalCost float64
	debug     bool

	// SystemPrompt should return the system prompt for the LLM. It's a function
	// to allow the system prompt to dynamically change throughout a single
	// conversation.
	SystemPrompt func() content.Content
}

// New creates a new LLM instance with the specified provider and optional
// tools. The provider handles communication with the actual LLM service. If
// tools are provided, they will be available for the LLM to use during
// conversations.
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
	return l.ChatWithContext(context.Background(), message)
}

// ChatWithContext sends a text message to the LLM and immediately returns a
// channel over which updates will come in. The LLM will use the tools available
// and keep generating more messages until it's done using tools. The provided
// context can be used to pass values to tools, set deadlines, cancel, etc.
func (l *LLM) ChatWithContext(ctx context.Context, message string) <-chan Update {
	return l.ChatUsingContent(ctx, content.FromText(message))
}

// ChatUsingContent sends a message (which can contain images) to the LLM and
// immediately returns a channel over which updates will come in. The LLM will
// use the tools available and keep generating more messages until it's done
// using tools. The provided context can be used to pass values to tools, set
// deadlines, cancel, etc.
func (l *LLM) ChatUsingContent(ctx context.Context, message content.Content) <-chan Update {
	return l.ChatUsingMessages(ctx, append(l.lastSentMessages, Message{
		Role:    "user",
		Content: message,
	}))
}

// ChatUsingMessages sends a message history to the LLM and immediately returns
// a channel over which updates will come in. The LLM will use the tools
// available and keep generating more messages until it's done using tools. The
// provided context can be used to pass values to tools, set deadlines, cancel,
// etc.
func (l *LLM) ChatUsingMessages(ctx context.Context, messages []Message) <-chan Update {
	l.lastSentMessages = messages
	// Send off the user's message to the LLM, and keep asking the LLM for more
	// responses for as long as it's making tool calls.
	updateChan := make(chan Update)
	go func() {
		defer close(updateChan)
		for {
			select {
			case <-ctx.Done():
				updateChan <- ErrorUpdate{Error: ctx.Err()}
				return
			default:
				shouldContinue, err := l.step(ctx, updateChan)
				if err != nil {
					updateChan <- ErrorUpdate{Error: err}
					return
				}
				if !shouldContinue {
					return
				}
			}
		}
	}()

	return updateChan
}

// AddTool adds a new tool to the LLM's toolbox. If the toolbox doesn't exist
// yet, it will be created. Tools allow the LLM to perform actions beyond just
// generating text, such as fetching data, running calculations, or interacting
// with external systems.
func (l *LLM) AddTool(t tools.Tool) {
	if l.toolbox == nil {
		l.toolbox = tools.Box(t)
	} else {
		l.toolbox.Add(t)
	}
}

// TotalCost returns the accumulated cost in USD of all LLM calls made through
// this instance. This helps track usage and expenses when working with
// commercial LLM providers.
func (l *LLM) TotalCost() float64 {
	return l.totalCost
}

// SetDebug enables or disables debug mode. When debug mode is enabled, the LLM
// will write detailed information about each interaction to a debug.yaml file,
// including the message history, tool calls, and other relevant data. This is
// useful for troubleshooting and understanding the LLM's behavior.
func (l *LLM) SetDebug(enabled bool) {
	l.debug = enabled
}

func (l *LLM) step(ctx context.Context, updateChan chan<- Update) (bool, error) {
	var systemPrompt content.Content
	if l.SystemPrompt != nil {
		systemPrompt = l.SystemPrompt()
	}

	// This will hold results from tool calls, to be sent back to the LLM.
	var toolMessages []Message

	stream := l.provider.Generate(systemPrompt, l.lastSentMessages, l.toolbox)
	if err := stream.Err(); err != nil {
		return false, fmt.Errorf("LLM returned error response: %w", err)
	}

	if l.debug {
		// Write the entire message history to the file debug.yaml. The function
		// is deferred so that we get data even if a panic occurs.
		defer func() {
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
				"3_sentMessages":    l.lastSentMessages,
				"4_systemPrompt":    systemPrompt,
				"5_availableTools":  toolsSchema,
			}
			if debugYAML, err := yaml.Marshal(debugData); err == nil {
				os.WriteFile("debug.yaml", debugYAML, 0644)
			}
		}()
	}

	done := make(chan bool)
	var streamErr error

	go func() {
	loop:
		for status := range stream.Iter() {
			select {
			case <-ctx.Done():
				streamErr = ctx.Err()
				break loop
			default:
				switch status {
				case StreamStatusText:
					updateChan <- TextUpdate{Text: stream.Text()}
				case StreamStatusToolCallBegin:
					tool := l.toolbox.Get(stream.ToolCall().Name)
					if tool == nil {
						streamErr = fmt.Errorf("tool %q not found", stream.ToolCall().Name)
						break loop
					}
					updateChan <- ToolStartUpdate{Tool: tool}
				case StreamStatusToolCallReady:
					messages := l.runToolCall(ctx, l.toolbox, stream.ToolCall(), updateChan)
					toolMessages = append(toolMessages, messages...)
				}
			}
		}
		done <- true
	}()

	select {
	case <-ctx.Done():
		return false, ctx.Err()
	case <-done:
		if streamErr != nil {
			return false, fmt.Errorf("error streaming: %w", streamErr)
		}
	}

	// Add the fully assembled message plus tool call results to the message history.
	l.lastSentMessages = append(l.lastSentMessages, stream.Message())
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
	l.lastSentMessages = append(l.lastSentMessages, toolMessages...)

	l.totalCost += stream.CostUSD()

	// Return true if there were tool calls, since the LLM should look at the results.
	return len(toolMessages) > 0, nil
}

func (l *LLM) runToolCall(ctx context.Context, toolbox *tools.Toolbox, toolCall ToolCall, updateChan chan<- Update) []Message {
	if toolCall.ID != "" {
		// As a sanity check, make sure we don't try to run the same tool call twice.
		for _, message := range l.lastSentMessages {
			if message.ToolCallID == toolCall.ID {
				fmt.Printf("\ntool call %q (%s) has already been run\n", toolCall.ID, toolCall.Name)
			}
		}
	}

	t := toolbox.Get(toolCall.Name)
	runner := tools.NewRunner(ctx, toolbox, func(status string) {
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
