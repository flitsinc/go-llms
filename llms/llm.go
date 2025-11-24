package llms

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"slices"
	"time"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/tools"
)

var (
	ErrMaxTurnsReached            = errors.New("max turns reached")
	ErrToolsAndJSONOutputConflict = errors.New("cannot specify both tools and a JSON output schema")
)

func cloneMetadata(src map[string]string) map[string]string {
	if len(src) == 0 {
		return nil
	}
	clone := make(map[string]string, len(src))
	for k, v := range src {
		clone[k] = v
	}
	return clone
}

// LLM represents the interface to an LLM provider, maintaining state between
// individual calls, for example when tool calling is being performed. Note that
// this is NOT thread safe for this reason.
type LLM struct {
	provider Provider
	toolbox  *tools.Toolbox

	turns, maxTurns  int
	lastSentMessages []Message

	err error // Last error encountered during operation

	// SystemPrompt should return the system prompt for the LLM. It's a function
	// to allow the system prompt to dynamically change throughout a single
	// conversation.
	SystemPrompt func() content.Content

	// JSONOutputSchema specifies a schema the LLM must conform its output to.
	// If set, the LLM output will be JSON conforming to this schema.
	// Cannot be used simultaneously with tools.
	JSONOutputSchema *tools.ValueSchema

	// TotalUsage tracks the sum of the numbers returned by each turn in the LLM.
	TotalUsage Usage

	// TrackTTFT is a function that will be called with the time it took for the
	// LLM to generate the first token of the turn.
	TrackTTFT func(context.Context, time.Duration)

	// TrackUsage is a function that will be called with the token usage
	// information after each LLM turn completes.
	TrackUsage func(ctx context.Context, usage Usage, success bool)
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

	// Launch a goroutine to manage the chat turns and stream processing.
	// This goroutine owns the updateChan and ensures it's closed on exit.
	go func() {
		defer close(updateChan)
		for {
			select {
			case <-ctx.Done():
				l.err = ctx.Err()
				if l.err == nil {
					l.err = context.Canceled
				}
				// Exit goroutine, defer close(updateChan) will run.
				return
			default:
				shouldContinue, err := l.turn(ctx, updateChan)
				if err != nil {
					l.err = err
					// Exit goroutine on error, defer close(updateChan) will run.
					return
				}
				if !shouldContinue {
					// Normal completion (e.g., no tool calls), exit goroutine.
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

func (l *LLM) SetToolChoice(choice tools.Choice) *LLM {
	if l.toolbox == nil {
		l.toolbox = tools.Box()
	}
	l.toolbox.Choice = choice
	return l
}

func (l *LLM) String() string {
	return fmt.Sprintf("%s (%s)", l.provider.Model(), l.provider.Company())
}

func (l *LLM) WithDebugger(d Debugger) *LLM {
	l.provider.SetDebugger(d)
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

	turnStart := time.Now()

	// Check for conflicting configuration: Tools and JSONOutputSchema
	hasTools := l.toolbox != nil && len(l.toolbox.All()) > 0
	if l.JSONOutputSchema != nil && hasTools {
		return false, ErrToolsAndJSONOutputConflict
	}

	var systemPrompt content.Content
	if l.SystemPrompt != nil {
		systemPrompt = l.SystemPrompt()
	}

	// This will hold results from tool calls, to be sent back to the LLM.
	var toolMessages []Message

	stream := l.provider.Generate(ctx, systemPrompt, l.lastSentMessages, l.toolbox, l.JSONOutputSchema)
	if err := stream.Err(); err != nil {
		return false, fmt.Errorf("LLM returned error response: %w", err)
	}

	trackTTFT := l.TrackTTFT
	shouldReportTTFT := trackTTFT != nil

	var success bool
	// Report usage if tracking is enabled.
	trackUsage := l.TrackUsage
	defer func() {
		usage := stream.Usage()
		l.TotalUsage.Add(usage)
		if trackUsage != nil {
			trackUsage(ctx, usage, success)
		}
	}()

	// Tracks how many bytes of the tool call arguments we sent so far in
	// deltas. We probably want to move this into the responsibility of each
	// provider so we can reduce allocations.
	var toolCallDeltaSentBytes int

	for status := range stream.Iter() {
		// For now assume the first event we get on the stream is the first token.
		if shouldReportTTFT {
			shouldReportTTFT = false
			ttft := time.Since(turnStart)
			trackTTFT(ctx, ttft)
		}
		// Check context at the beginning of each iteration.
		// This ensures we react promptly if cancellation happens *between* stream events.
		select {
		case <-ctx.Done():
			// Propagate cancellation error immediately
			return false, ctx.Err()
		default:
			// Context OK, process status
		}
		switch status {
		case StreamStatusMessageStart:
			msg := stream.Message()
			if msg.ID != "" {
				updateChan <- MessageStartUpdate{MessageID: msg.ID}
			}

		case StreamStatusText:
			updateChan <- TextUpdate{stream.Text()}

		case StreamStatusImage:
			url, mime := stream.Image()
			updateChan <- ImageUpdate{URL: url, MimeType: mime}

		case StreamStatusThinking:
			updateChan <- ThinkingUpdate{stream.Thought()}

		case StreamStatusThinkingDone:
			updateChan <- ThinkingDoneUpdate{}

		case StreamStatusToolCallBegin:
			toolCall := stream.ToolCall()
			if toolCall.ID == "" {
				return false, fmt.Errorf("missing tool call ID for tool %q", toolCall.Name)
			}
			tool := l.toolbox.Get(toolCall.Name)
			if tool == nil {
				return false, fmt.Errorf("tool %q not found", toolCall.Name)
			}
			toolCallDeltaSentBytes = 0
			updateChan <- ToolStartUpdate{ToolCallID: toolCall.ID, Tool: tool}

		case StreamStatusToolCallDelta:
			toolCall := stream.ToolCall()
			if argLen := len(toolCall.Arguments); argLen > toolCallDeltaSentBytes {
				// Only send the new part of the arguments.
				updateChan <- ToolDeltaUpdate{toolCall.ID, toolCall.Arguments[toolCallDeltaSentBytes:]}
				toolCallDeltaSentBytes = argLen
			}

		case StreamStatusToolCallReady:
			// TODO: We want to support parallel tool calls, which means results
			// would need to be collected later (and maybe out of order).
			toolCall := stream.ToolCall()
			// Usually there shouldn't be any more changes to arguments but we
			// have to make sure all arguments are sent before we run the tool.
			if argLen := len(toolCall.Arguments); argLen > toolCallDeltaSentBytes {
				// Only send the new part of the arguments.
				updateChan <- ToolDeltaUpdate{toolCall.ID, toolCall.Arguments[toolCallDeltaSentBytes:]}
				toolCallDeltaSentBytes = argLen
			}
			toolMessage := l.runToolCall(ctx, l.toolbox, toolCall, updateChan)
			toolMessages = append(toolMessages, toolMessage)
		}
	}
	// Check stream error after iterating
	if streamErr := stream.Err(); streamErr != nil {
		return false, fmt.Errorf("error iterating stream: %w", streamErr)
	}
	// Also check if the context was cancelled *during* stream iteration,
	// even if the iterator itself didn't return an error.
	if ctx.Err() != nil {
		return false, ctx.Err()
	}
	success = true

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

	// Return true if there were tool calls, since the LLM should look at the results.
	return len(toolMessages) > 0, nil
}

func (l *LLM) runToolCall(ctx context.Context, toolbox *tools.Toolbox, toolCall ToolCall, updateChan chan<- Update) Message {
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
		select {
		case <-ctx.Done(): // Don't send if already cancelled
		default:
			updateChan <- ToolStatusUpdate{toolCall.ID, status, t}
		}
	})

	result := toolbox.Run(runner, toolCall.Name, json.RawMessage(toolCall.Arguments))
	select {
	case <-ctx.Done(): // Don't send if already cancelled
	default:
		// TODO: If we ever expose a "tool starting to run" update, the Metadata can be sent there instead.
		updateChan <- ToolDoneUpdate{ToolCallID: toolCall.ID, Result: result, Tool: t, Metadata: cloneMetadata(toolCall.Metadata)}
	}

	return Message{
		Role:         "tool",
		Content:      result.Content(),
		ToolCallID:   toolCall.ID,
		ToolCallName: toolCall.Name,
	}
}
