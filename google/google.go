package google

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"time"

	"golang.org/x/oauth2"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
	"github.com/flitsinc/go-llms/tools"
)

// clearAdditionalProperties recursively sets AdditionalProperties to nil
// in all ValueSchemas to make them compatible with Google's API.
func clearAdditionalProperties(schema *tools.ValueSchema) {
	schema.AdditionalProperties = nil

	if schema.Items != nil {
		clearAdditionalProperties(schema.Items)
	}

	if schema.Properties != nil {
		for _, k := range schema.Properties.Keys() {
			raw, ok := schema.Properties.Get(k)
			if !ok {
				continue
			}
			v, ok := raw.(tools.ValueSchema)
			if !ok {
				if rm, rok := raw.(json.RawMessage); rok {
					if err := json.Unmarshal(rm, &v); err != nil {
						continue
					}
				} else {
					continue
				}
			}
			clearAdditionalProperties(&v)
			schema.Properties.Set(k, v)
		}
	}

	for i := range schema.AnyOf {
		clearAdditionalProperties(&schema.AnyOf[i])
	}
}

type Model struct {
	tokenSource     oauth2.TokenSource
	model           string
	endpoint        string
	maxOutputTokens int
	temperature     float64
	topK            int
	topP            float64
	includeThoughts bool
	thinkingBudget  int
	thinkingLevel   ThinkingLevel
	mediaResolution MediaResolution
	debugger        llms.Debugger
	modalities      []string
	httpClient      *http.Client

	// streamFunctionCallArguments enables streaming of function call arguments
	// on Vertex AI Gemini 3+ models. When true, the backend sends partial argument
	// chunks via partialArgs with a stable functionCall.id across chunks.
	// Note: This is only supported on Vertex AI; the Gemini Developer API ignores this.
	streamFunctionCallArguments bool
}

func New(model string) *Model {
	return &Model{
		model:       model,
		temperature: math.NaN(),
		topP:        math.NaN(),
	}
}

func (m *Model) WithGeminiAPI(apiKey string) *Model {
	m.tokenSource = nil
	m.endpoint = fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:streamGenerateContent?alt=sse&key=%s", m.model, apiKey)
	return m
}

// WithVertexAI configures the model to use an OAuth2 token source
// for authenticating to the Vertex AI API. This is the recommended approach
// for production environments.
func (m *Model) WithVertexAI(ts oauth2.TokenSource, projectID, location string) *Model {
	m.tokenSource = ts
	if location == "global" {
		m.endpoint = fmt.Sprintf("https://aiplatform.googleapis.com/v1/projects/%s/locations/global/publishers/google/models/%s:streamGenerateContent?alt=sse", projectID, m.model)
	} else {
		m.endpoint = fmt.Sprintf("https://%s-aiplatform.googleapis.com/v1/projects/%s/locations/%s/publishers/google/models/%s:streamGenerateContent?alt=sse", location, projectID, location, m.model)
	}
	return m
}

// WithVertexAIAccessToken configures the model to use a static access token for
// authenticating to the Vertex AI API.
func (m *Model) WithVertexAIAccessToken(accessToken, projectID, location string) *Model {
	return m.WithVertexAI(
		oauth2.StaticTokenSource(&oauth2.Token{AccessToken: accessToken}),
		projectID,
		location,
	)
}

func (m *Model) WithMaxOutputTokens(maxOutputTokens int) *Model {
	m.maxOutputTokens = maxOutputTokens
	return m
}

func (m *Model) WithTemperature(temperature float64) *Model {
	m.temperature = temperature
	return m
}

func (m *Model) WithTopK(topK int) *Model {
	m.topK = topK
	return m
}

func (m *Model) WithTopP(topP float64) *Model {
	m.topP = topP
	return m
}

func (m *Model) WithThinking(budgetTokens int) *Model {
	m.includeThoughts = budgetTokens > 0
	m.thinkingBudget = budgetTokens
	return m
}

// WithThinkingLevel sets the thinking level for the model (e.g. "high", "low").
// This corresponds to the "Thinking Levels" feature in Gemini 3.
func (m *Model) WithThinkingLevel(level ThinkingLevel) *Model {
	m.thinkingLevel = level
	// If level is set, we likely imply including thoughts, but let's rely on
	// WithThinking to enable the boolean flag or we can enable it here too.
	// Usually enabling thinking level implies thinking is on.
	if level != "" {
		m.includeThoughts = true
	}
	return m
}

// WithMediaResolution sets the media resolution for the model
// output/processing. Valid values might include "MEDIA_RESOLUTION_UNSPECIFIED",
// "MEDIA_RESOLUTION_LOW", "MEDIA_RESOLUTION_MEDIUM", "MEDIA_RESOLUTION_HIGH".
func (m *Model) WithMediaResolution(resolution MediaResolution) *Model {
	m.mediaResolution = resolution
	return m
}

// WithModalities configures the model to use specific modalities for the
// response. Usually not necessary.
//
// Some valid values are: "TEXT", "IMAGE", "AUDIO"
func (m *Model) WithModalities(modalities ...string) *Model {
	m.modalities = modalities
	return m
}

// WithStreamingToolArguments enables streaming of function call arguments on
// Vertex AI Gemini 3+ models. When enabled, the backend sends partial argument
// chunks with a stable functionCall.id, allowing the stream to emit ToolCallBegin
// once, then multiple ToolCallDelta events as arguments stream in, and finally
// ToolCallReady when complete.
//
// Note: This feature is only supported on Vertex AI. The Gemini Developer API
// (WithGeminiAPI) does not support partialArgs and will ignore this setting.
func (m *Model) WithStreamingToolArguments(enabled bool) *Model {
	m.streamFunctionCallArguments = enabled
	return m
}

func (m *Model) SetHTTPClient(client *http.Client) {
	m.httpClient = client
}

func (m *Model) Company() string {
	return "Google"
}

func (m *Model) Model() string {
	return m.model
}

func (m *Model) SetDebugger(d llms.Debugger) {
	m.debugger = d
}

func (m *Model) Generate(
	ctx context.Context,
	systemPrompt content.Content,
	messages []llms.Message,
	toolbox *tools.Toolbox,
	jsonOutputSchema *tools.ValueSchema,
) llms.ProviderStream {
	if m.endpoint == "" {
		return &Stream{err: fmt.Errorf("must call either WithVertexAI(…) or WithGenerativeLanguageAPI(…) first")}
	}

	var apiMessages []message
	var pendingFunctionMsg *message
	var deferredAfterFunction []message

	flushPending := func() {
		if pendingFunctionMsg != nil {
			apiMessages = append(apiMessages, *pendingFunctionMsg)
			pendingFunctionMsg = nil
		}
		if len(deferredAfterFunction) > 0 {
			apiMessages = append(apiMessages, deferredAfterFunction...)
			deferredAfterFunction = nil
		}
	}

	for _, msg := range messages {
		convertedMsgs, err := messagesFromLLM(msg)
		if err != nil {
			return &Stream{err: fmt.Errorf("failed to convert message for Google: %w", err)}
		}
		if msg.Role == "tool" {
			if len(convertedMsgs) > 0 && convertedMsgs[0].Role == "function" {
				if pendingFunctionMsg == nil {
					pendingFunctionMsg = &message{Role: "function"}
				}
				pendingFunctionMsg.Parts = append(pendingFunctionMsg.Parts, convertedMsgs[0].Parts...)
				if len(convertedMsgs) > 1 {
					deferredAfterFunction = append(deferredAfterFunction, convertedMsgs[1:]...)
				}
				continue
			}
		}

		flushPending()
		apiMessages = append(apiMessages, convertedMsgs...)
	}
	flushPending()

	payload := map[string]any{
		"contents": apiMessages,
		"safetySettings": []map[string]any{
			{
				"category":  "HARM_CATEGORY_HATE_SPEECH",
				"threshold": "BLOCK_ONLY_HIGH",
			},
			{
				"category":  "HARM_CATEGORY_DANGEROUS_CONTENT",
				"threshold": "BLOCK_ONLY_HIGH",
			},
			{
				"category":  "HARM_CATEGORY_SEXUALLY_EXPLICIT",
				"threshold": "BLOCK_ONLY_HIGH",
			},
			{
				"category":  "HARM_CATEGORY_HARASSMENT",
				"threshold": "BLOCK_ONLY_HIGH",
			},
			{
				"category":  "HARM_CATEGORY_CIVIC_INTEGRITY",
				"threshold": "BLOCK_ONLY_HIGH",
			},
		},
	}

	generationConfig := map[string]any{}
	if m.maxOutputTokens > 0 {
		generationConfig["maxOutputTokens"] = m.maxOutputTokens
	}
	if !math.IsNaN(m.temperature) {
		generationConfig["temperature"] = m.temperature
	}
	if !math.IsNaN(m.topP) {
		generationConfig["topP"] = m.topP
	}
	if m.topK > 0 {
		generationConfig["topK"] = m.topK
	}

	if jsonOutputSchema != nil {
		generationConfig["responseMimeType"] = "application/json"
		generationConfig["responseSchema"] = jsonOutputSchema
	}

	if m.includeThoughts {
		thinkingConfig := map[string]any{
			"includeThoughts": true,
		}
		if m.thinkingBudget > 0 {
			thinkingConfig["thinkingBudget"] = m.thinkingBudget
		}
		// Gemini 3 Thinking Levels
		if m.thinkingLevel != "" {
			thinkingConfig["thinkingLevel"] = m.thinkingLevel
		}
		generationConfig["thinkingConfig"] = thinkingConfig
	}

	if m.mediaResolution != "" {
		generationConfig["mediaResolution"] = m.mediaResolution
	}

	if len(m.modalities) > 0 {
		generationConfig["responseModalities"] = m.modalities
	}

	if len(generationConfig) > 0 {
		payload["generationConfig"] = generationConfig
	}

	if systemPrompt != nil {
		payload["systemInstruction"] = map[string]any{
			"parts": convertContent(systemPrompt),
		}
	}

	if toolbox != nil {
		allTools := toolbox.All()
		// Build declarations from all tools (do not filter; we'll restrict via toolConfig for cacheability)
		declarations := make([]tools.FunctionSchema, len(allTools))
		for i, tool := range allTools {
			// Google supports only function-style tools; JSON grammar is fine.
			switch g := tool.Grammar().(type) {
			case tools.JSONGrammar:
				schema := *g.Schema()
				clearAdditionalProperties(&schema.Parameters)
				declarations[i] = schema
			default:
				panic(fmt.Sprintf("unsupported grammar type: %T", g))
			}
		}
		payload["tools"] = map[string]any{
			"functionDeclarations": declarations,
		}

		// Map Choice to Google's tool configuration (allowedFunctionNames)
		// We keep all functionDeclarations above for cacheability, and rely on
		// toolConfig.functionCallingConfig.allowedFunctionNames to constrain use.
		choice := toolbox.Choice
		functionCallingConfig := map[string]any{}

		switch choice.Mode {
		case tools.ChoiceAllowOnly:
			if len(choice.AllowedTools) == 0 {
				functionCallingConfig["mode"] = "NONE"
			} else {
				functionCallingConfig["mode"] = "AUTO"
				functionCallingConfig["allowedFunctionNames"] = choice.AllowedTools
			}
		case tools.ChoiceRequireOneOf:
			if len(choice.AllowedTools) == 0 {
				functionCallingConfig["mode"] = "NONE"
			} else {
				functionCallingConfig["mode"] = "ANY"
				functionCallingConfig["allowedFunctionNames"] = choice.AllowedTools
			}
		default:
			functionCallingConfig["mode"] = "AUTO"
		}

		// Enable streaming function call arguments when configured.
		// This is supported on Vertex AI Gemini 3+ models and causes the backend
		// to send partialArgs with a stable functionCall.id across chunks.
		if m.streamFunctionCallArguments {
			functionCallingConfig["streamFunctionCallArguments"] = true
		}

		payload["toolConfig"] = map[string]any{
			"functionCallingConfig": functionCallingConfig,
		}
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return &Stream{err: fmt.Errorf("error encoding JSON: %w", err)}
	}

	if m.debugger != nil {
		m.debugger.RawRequest(m.endpoint, jsonData)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", m.endpoint, bytes.NewReader(jsonData))
	if err != nil {
		return &Stream{err: fmt.Errorf("error creating request: %w", err)}
	}
	if m.tokenSource != nil {
		token, err := m.tokenSource.Token()
		if err != nil {
			return &Stream{err: fmt.Errorf("error getting token from source: %w", err)}
		}
		req.Header.Set("Authorization", "Bearer "+token.AccessToken)
	}
	req.Header.Set("Content-Type", "application/json")

	client := m.httpClient
	if client == nil {
		client = http.DefaultClient
	}

	resp, err := client.Do(req)
	if err != nil {
		return &Stream{err: fmt.Errorf("error making request: %w", err)}
	}
	if resp.StatusCode != http.StatusOK {
		defer resp.Body.Close()
		bodyBytes, readErr := io.ReadAll(resp.Body)

		if readErr == nil && len(bodyBytes) > 0 {
			var errResp errorResponse // Assumes this struct matches Google's { "error": { ... } } format
			if jsonErr := json.Unmarshal(bodyBytes, &errResp); jsonErr == nil && errResp.Error.Message != "" {
				// Successfully parsed the Google error format
				return &Stream{err: fmt.Errorf("%s: %s", resp.Status, errResp.Error.Message)}
			}
			// Body read okay, but JSON parsing failed or structure mismatch.
			// Fall through to return status only.
		}
		// Default fallback: Read error, empty body, or failed/unexpected JSON parse.
		return &Stream{err: fmt.Errorf("%s", resp.Status)}
	}
	return &Stream{
		ctx:            ctx,
		model:          m.model,
		stream:         resp.Body,
		debugger:       m.debugger,
		toolCallsByID:  make(map[string]int),
		toolArgsByID:   make(map[string]json.RawMessage),
		toolCallsReady: make(map[string]bool),
	}
}

type Stream struct {
	ctx         context.Context
	model       string
	stream      io.Reader
	err         error
	message     llms.Message
	lastText    string
	lastThought *content.Thought
	usage       *usageMetadata
	debugger    llms.Debugger
	lastImage   struct{ URL, MIME string }

	// Tool call tracking for streaming function call arguments.
	// Maps functionCall.ID to the index in message.ToolCalls.
	toolCallsByID map[string]int
	// Tracks the latest arguments JSON snapshot for each tool call ID.
	toolArgsByID map[string]json.RawMessage
	// Tracks tool calls that have had ToolCallReady emitted (by ID).
	toolCallsReady map[string]bool
}

func (s *Stream) Err() error {
	return s.err
}

func (s *Stream) Message() llms.Message {
	return s.message
}

func (s *Stream) Text() string {
	return s.lastText
}

func (s *Stream) Image() (string, string) {
	return s.lastImage.URL, s.lastImage.MIME
}

func (s *Stream) ToolCall() llms.ToolCall {
	if len(s.message.ToolCalls) == 0 {
		return llms.ToolCall{}
	}
	return s.message.ToolCalls[len(s.message.ToolCalls)-1]
}

func (s *Stream) Thought() content.Thought {
	return *s.lastThought
}

func (s *Stream) Usage() llms.Usage {
	if s.usage == nil {
		return llms.Usage{}
	}
	// TODO: Report cache context tokens.
	return llms.Usage{
		CachedInputTokens: 0,
		InputTokens:       s.usage.PromptTokenCount,
		OutputTokens:      s.usage.CandidatesTokenCount,
	}
}

func (s *Stream) Iter() func(yield func(llms.StreamStatus) bool) {
	reader := bufio.NewReader(s.stream)
	return func(yield func(llms.StreamStatus) bool) {
		defer io.Copy(io.Discard, s.stream)
		// Track whether the last yielded event was a thinking update, so we can emit ThinkingDone
		lastEventWasThinking := false
		messageStartYielded := false

		// emitPendingToolCallsReady emits ToolCallReady for any tool calls that
		// haven't been marked ready yet. This handles the case where the stream
		// ends without a finishReason (common in one-shot scenarios).
		emitPendingToolCallsReady := func() bool {
			for callID := range s.toolCallsByID {
				if !s.toolCallsReady[callID] {
					s.toolCallsReady[callID] = true
					if !yield(llms.StreamStatusToolCallReady) {
						return false
					}
				}
			}
			return true
		}

		for {
			select {
			case <-s.ctx.Done():
				s.err = s.ctx.Err()
				return
			default:
			}

			// Read a full logical line using ReadLine to support very long lines
			var lineBuilder strings.Builder
			for {
				part, isPrefix, err := reader.ReadLine()
				if err != nil {
					if err == io.EOF {
						// If we have accumulated partial data, process it before returning
						if lineBuilder.Len() == 0 {
							// Stream ended; emit Ready for any pending tool calls
							emitPendingToolCallsReady()
							return
						}
						break
					}
					s.err = fmt.Errorf("error reading stream: %w", err)
					return
				}
				lineBuilder.Write(part)
				if !isPrefix {
					break
				}
			}

			rawLine := lineBuilder.String()

			if s.debugger != nil && strings.TrimSpace(rawLine) != "" {
				s.debugger.RawEvent([]byte(rawLine))
			}

			line, ok := strings.CutPrefix(rawLine, "data: ")
			if !ok {
				continue
			}

			var chunk streamingResponse
			if err := json.Unmarshal([]byte(line), &chunk); err != nil {
				s.err = fmt.Errorf("error unmarshalling chunk: %w", err)
				return
			}
			if chunk.UsageMetadata != nil {
				s.usage = chunk.UsageMetadata
			}
			if len(chunk.Candidates) < 1 {
				continue
			}
			delta := chunk.Candidates[0].Content
			if delta.Role != "" {
				s.message.Role = delta.Role
			}
			// Emit message_start once before any content is yielded. Google does not
			// currently provide a message ID, but downstream expects the event.
			if !messageStartYielded {
				messageStartYielded = true
				if !yield(llms.StreamStatusMessageStart) {
					return
				}
			}
			for _, p := range delta.Parts {
				if p.Text != nil && *p.Text != "" {
					if p.Thought {
						// Note: Google only gives us summaries.
						s.message.Content.SetThoughtSummary(*p.Text, p.ThoughtSignature)
						s.lastThought = &content.Thought{
							Text:      *p.Text,
							Signature: p.ThoughtSignature,
							Summary:   true,
						}
						if !yield(llms.StreamStatusThinking) {
							return
						}
						lastEventWasThinking = true
					} else {
						// Before yielding text, if we were previously thinking, signal done
						if lastEventWasThinking {
							if !yield(llms.StreamStatusThinkingDone) {
								return
							}
							lastEventWasThinking = false
						}
						s.lastText = *p.Text
						s.message.Content.Append(s.lastText)
						if !yield(llms.StreamStatusText) {
							return
						}
					}
				}
				// Emit images from inline/file data parts as discrete image events.
				if p.InlineData != nil {
					if p.InlineData.MimeType != "" && p.InlineData.Data != "" {
						uri := content.BuildDataURI(p.InlineData.MimeType, p.InlineData.Data)
						s.message.Content = append(s.message.Content, &content.ImageURL{URL: uri, MimeType: p.InlineData.MimeType})
						s.lastImage.URL = uri
						s.lastImage.MIME = p.InlineData.MimeType
						if !yield(llms.StreamStatusImage) {
							return
						}
					}
				}
				if p.FileData != nil {
					url := p.FileData.FileURI
					mime := p.FileData.MimeType
					if mime == "" {
						mime = content.ExtractMIMETypeFromURIOrURL(url)
					}
					if url != "" {
						s.message.Content = append(s.message.Content, &content.ImageURL{URL: url, MimeType: mime})
						s.lastImage.URL = url
						s.lastImage.MIME = mime
						if !yield(llms.StreamStatusImage) {
							return
						}
					}
				}
				if p.FunctionCall != nil {
					// Before yielding tool events, if we were previously thinking, signal done
					if lastEventWasThinking {
						if !yield(llms.StreamStatusThinkingDone) {
							return
						}
						lastEventWasThinking = false
					}

					fc := p.FunctionCall

					// Per official Google docs for streaming function call arguments:
					// - A chunk with `name` field starts a new function call
					// - Chunks with `partialArgs` continue the current call
					// - An empty functionCall {} signals the end of the current call
					// - `willContinue` at functionCall level indicates if more chunks are expected
					//
					// We support two tracking modes:
					// 1. ID-based: when fc.ID is provided (may be available in some API versions)
					// 2. Index-based: track "current active call" by message.ToolCalls index

					// Detect empty functionCall {} which signals end of current function call
					isEmptyFunctionCall := fc.Name == "" && fc.Args == nil && len(fc.PartialArgs) == 0 &&
						(fc.WillContinue == nil || !*fc.WillContinue)

					if isEmptyFunctionCall {
						// Mark all pending tool calls as ready
						for callID := range s.toolCallsByID {
							if !s.toolCallsReady[callID] {
								s.toolCallsReady[callID] = true
								if !yield(llms.StreamStatusToolCallReady) {
									return
								}
							}
						}
						continue
					}

					// Determine call ID for tracking
					callID := fc.ID
					if callID == "" {
						// No ID provided - use name-based tracking for streaming scenarios.
						// Per docs, `name` appears on first chunk of each function call.
						if fc.Name != "" {
							// New function call starting - generate a unique ID
							callID = fmt.Sprintf("%s-%d", fc.Name, time.Now().UnixNano())
						} else if len(s.message.ToolCalls) > 0 {
							// Continuation chunk (no name, has partialArgs) - use last call's ID
							callID = s.message.ToolCalls[len(s.message.ToolCalls)-1].ID
						} else {
							// Edge case: partialArgs without any prior call - skip
							continue
						}
					}

					// Get or create the llms.ToolCall entry for this ID.
					idx, exists := s.toolCallsByID[callID]
					if !exists {
						// First time we've seen this function call.
						metadata := make(map[string]string)
						if p.ThoughtSignature != "" {
							metadata["google:thought_signature"] = p.ThoughtSignature
						}

						// Initial arguments: may be nil if we're only getting partialArgs at first.
						args := fc.Args
						if args == nil {
							args = json.RawMessage("{}")
						}

						s.message.ToolCalls = append(s.message.ToolCalls, llms.ToolCall{
							ID:        callID,
							Name:      fc.Name,
							Arguments: args,
							Metadata:  metadata,
						})
						idx = len(s.message.ToolCalls) - 1
						s.toolCallsByID[callID] = idx
						s.toolArgsByID[callID] = args

						// First chunk: signal begin
						if !yield(llms.StreamStatusToolCallBegin) {
							return
						}
					}

					// Update arguments snapshot based on args and/or partialArgs.
					currentArgs := s.toolArgsByID[callID]

					// If the backend sends full args (fc.Args), prefer that as the latest snapshot.
					if len(fc.Args) > 0 {
						currentArgs = fc.Args
					}

					// Apply partialArgs if present (streaming argument updates).
					if len(fc.PartialArgs) > 0 {
						merged, err := applyPartialArgsJSON(currentArgs, fc.PartialArgs)
						if err == nil {
							currentArgs = merged
						}
						// On error, we continue with whatever we have - partial args are best-effort.
					}

					// Persist the updated snapshot
					s.toolArgsByID[callID] = currentArgs
					s.message.ToolCalls[idx].Arguments = currentArgs

					// Emit a delta for this chunk (only if we have content to report)
					if len(fc.Args) > 0 || len(fc.PartialArgs) > 0 {
						if !yield(llms.StreamStatusToolCallDelta) {
							return
						}
					}

					// Check if this is the final chunk for the function call.
					// Per docs: willContinue=false or absent means this is the final chunk.
					// Also check finishReason for non-streaming scenarios.
					isFinalChunk := (fc.WillContinue != nil && !*fc.WillContinue) ||
						chunk.Candidates[0].FinishReason != ""
					if isFinalChunk && !s.toolCallsReady[callID] {
						s.toolCallsReady[callID] = true
						if !yield(llms.StreamStatusToolCallReady) {
							return
						}
					}
				}
				// If the candidate reports a finish reason and we were thinking just before,
				// signal that thinking has completed.
				if chunk.Candidates[0].FinishReason != "" {
					if lastEventWasThinking {
						if !yield(llms.StreamStatusThinkingDone) {
							return
						}
						lastEventWasThinking = false
					}
				}
			}
		}
	}
}
