package llms

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"slices"

	"sigs.k8s.io/yaml"

	"github.com/blixt/go-llms/content"
	"github.com/blixt/go-llms/tools"
)

var (
	ErrMaxTurnsReached = errors.New("max turns reached")
)

// LLM represents the interface to an LLM provider, maintaining state between
// individual calls, for example when tool calling is being performed. Note that
// this is NOT thread safe for this reason.
type LLM struct {
	provider Provider
	toolbox  *tools.Toolbox

	turns, maxTurns  int
	lastSentMessages []Message

	totalCost float64
	debug     bool
	err       error // Last error encountered during operation

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
	// Reset error state for new chat
	l.err = nil

	updateChan := make(chan Update)

	// Check if context is already cancelled before starting goroutine
	if err := ctx.Err(); err != nil {
		l.err = err
		close(updateChan) // Close channel immediately
		return updateChan // Return the closed channel
	}

	// Send off the user's message to the LLM, and keep asking the LLM for more
	// responses for as long as it's making tool calls.
	go func() {
		defer close(updateChan)
		for {
			select {
			case <-ctx.Done():
				l.err = ctx.Err()
				if l.err == nil {
					l.err = context.Canceled
				}
				// Just close the channel and let the caller check Err()
				return
			default:
				shouldContinue, err := l.turn(ctx, updateChan)
				if err != nil {
					l.err = err
					// Just close the channel and let the caller check Err()
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

// AddExternalTools adds one or more external tools to the LLM's toolbox. Unlike
// regular tools, external tools are usually forwarded to some other code
// (sometimes over the network) and handled there, before a result is produced.
// For this reason, a list of tool definitions can be provided, and then the
// tool's raw JSON parameters are passed into the handler. The handler can use
// `GetToolCall(r.Context())` to retrieve the `ToolCall` object, which includes
// the function name (`Name`) and the unique `ID` for the specific call.
func (l *LLM) AddExternalTools(schemas []tools.FunctionSchema, handler func(r tools.Runner, params json.RawMessage) tools.Result) {
	for i := range schemas {
		schema := schemas[i] // Create a new variable for the loop
		l.AddTool(tools.External(schema.Name, &schema, handler))
	}
}

// AddTool adds a new tool to the LLM's toolbox. If the toolbox doesn't exist
// yet, it will be created. Tools allow the LLM to perform actions beyond just
// generating text, such as fetching data, running calculations, or interacting
// with external systems.
func (l *LLM) AddTool(t tools.Tool) {
	if t == nil {
		panic("attempted to add a nil tool to the LLM toolbox")
	}
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

// WithDebug enables debug mode. When debug mode is enabled, the LLM will write
// detailed information about each interaction to a debug.yaml file, including
// the message history, tool calls, and other relevant data. This is useful for
// troubleshooting and understanding the LLM's behavior.
func (l *LLM) WithDebug() *LLM {
	l.debug = true
	return l
}

// WithMaxTurns sets the maximum number of turns the LLM will make. This is
// useful to prevent infinite loops or excessive usage. A value of 0 means no
// limit. A value of 1 means the LLM will only ever do one API call, and so on.
func (l *LLM) WithMaxTurns(maxTurns int) *LLM {
	l.maxTurns = maxTurns
	return l
}

// Err returns the last error encountered during LLM operation. This is useful
// for checking errors after a Chat loop completes. Returns nil if no error
// occurred.
func (l *LLM) Err() error {
	return l.err
}

func (l *LLM) turn(ctx context.Context, updateChan chan<- Update) (bool, error) {
	if l.maxTurns > 0 && l.turns >= l.maxTurns {
		return false, ErrMaxTurnsReached
	}
	l.turns++

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
					toolCall := stream.ToolCall()
					if toolCall.ID == "" {
						streamErr = fmt.Errorf("missing tool call ID for tool %q", toolCall.Name)
						break loop
					}

					tool := l.toolbox.Get(toolCall.Name)
					if tool == nil {
						streamErr = fmt.Errorf("tool %q not found", toolCall.Name)
						break loop
					}
					updateChan <- ToolStartUpdate{
						ToolCallID: toolCall.ID,
						Tool:       tool,
					}
				case StreamStatusToolCallReady:
					// TODO: We may want to support parallel tool calls, which
					// means the results would need to be collected later (and
					// maybe out of sequence).
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
	if toolCall.ID == "" {
		panic(fmt.Sprintf("tool call (%s) is missing an ID", toolCall.Name))
	}

	// As a sanity check, make sure we don't try to run the same tool call twice.
	for _, message := range l.lastSentMessages {
		if message.ToolCallID == toolCall.ID {
			panic(fmt.Sprintf("tool call %q (%s) has already been run", toolCall.ID, toolCall.Name))
		}
	}

	t := toolbox.Get(toolCall.Name)
	// Create a new context with the ToolCall value
	ctxWithValue := context.WithValue(ctx, ToolCallContextKey, toolCall)
	runner := tools.NewRunner(ctxWithValue, toolbox, func(status string) {
		// TODO: Add tests to verify that ToolStatusUpdate messages are correctly
		//       received by the caller through the update channel. This might
		//       require a more sophisticated mock runner or test setup.
		updateChan <- ToolStatusUpdate{
			ToolCallID: toolCall.ID,
			Status:     status,
			Tool:       t,
		}
	})

	result := toolbox.Run(runner, toolCall.Name, json.RawMessage(toolCall.Arguments))
	updateChan <- ToolDoneUpdate{
		ToolCallID: toolCall.ID,
		Result:     result,
		Tool:       t,
	}

	messages := []Message{
		{
			Role:       "tool",
			Content:    content.FromRawJSON(result.JSON()),
			ToolCallID: toolCall.ID,
		},
	}

	if images := result.Images(); len(images) > 0 {
		// TODO: Revisit this image handling logic. Faking a user message feels
		//       like a workaround. Explore if the Message/Content system can be
		//       extended to support images directly within tool results, or find
		//       a cleaner mechanism to associate images with their originating
		//       tool call without polluting the user message history synthetically.
		// "tool" messages can't actually contain image content. So we need to
		// fake a user message instead.
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
