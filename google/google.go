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
	"regexp"
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
		for k := range *schema.Properties {
			v := (*schema.Properties)[k]
			clearAdditionalProperties(&v)
			(*schema.Properties)[k] = v
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
	debug           bool
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
	m.includeThoughts = true
	m.thinkingBudget = budgetTokens
	return m
}

func (m *Model) WithDebug() *Model {
	m.debug = true
	return m
}

func (m *Model) Company() string {
	return "Google"
}

func (m *Model) Model() string {
	return m.model
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
	for _, msg := range messages {
		convertedMsgs := messagesFromLLM(msg)
		apiMessages = append(apiMessages, convertedMsgs...)
	}

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
		generationConfig["thinkingConfig"] = thinkingConfig
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
		switch choice.Mode {
		case tools.ChoiceAllowOnly:
			if len(choice.AllowedTools) == 0 {
				payload["toolConfig"] = map[string]any{
					"functionCallingConfig": map[string]any{
						"mode": "NONE",
					},
				}
			} else {
				payload["toolConfig"] = map[string]any{
					"functionCallingConfig": map[string]any{
						"mode":                 "AUTO",
						"allowedFunctionNames": choice.AllowedTools,
					},
				}
			}
		case tools.ChoiceRequireOneOf:
			if len(choice.AllowedTools) == 0 {
				payload["toolConfig"] = map[string]any{
					"functionCallingConfig": map[string]any{
						"mode": "NONE",
					},
				}
			} else {
				payload["toolConfig"] = map[string]any{
					"functionCallingConfig": map[string]any{
						"mode":                 "ANY",
						"allowedFunctionNames": choice.AllowedTools,
					},
				}
			}
		default:
			payload["toolConfig"] = map[string]any{
				"functionCallingConfig": map[string]any{
					"mode": "AUTO",
				},
			}
		}
	}

	jsonData, err := json.Marshal(payload)
	if err != nil {
		return &Stream{err: fmt.Errorf("error encoding JSON: %w", err)}
	}

	if m.debug {
		fmt.Printf("\033[1;90m%s\033[0m\n", regexp.MustCompile(`([&?]key)=[^&]*`).ReplaceAllString(m.endpoint, "$1=…"))
		fmt.Printf("-> \033[2;34m%s\033[0m\n", string(jsonData))
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

	resp, err := http.DefaultClient.Do(req)
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
	return &Stream{ctx: ctx, model: m.model, stream: resp.Body, debug: m.debug}
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
	debug       bool
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
	scanner := bufio.NewScanner(s.stream)
	return func(yield func(llms.StreamStatus) bool) {
		defer io.Copy(io.Discard, s.stream)
		// Track whether the last yielded event was a thinking update, so we can emit ThinkingDone
		lastEventWasThinking := false
		for {
			select {
			case <-s.ctx.Done():
				s.err = s.ctx.Err()
				return
			default:
				// Context OK, keep scanning.
			}
			if !scanner.Scan() {
				if err := scanner.Err(); err != nil {
					s.err = fmt.Errorf("error scanning stream: %w", err)
				}
				return
			}

			if s.debug && strings.TrimSpace(scanner.Text()) != "" {
				fmt.Printf("<- \033[2;32m%s\033[0m\n", scanner.Text())
			}

			line, ok := strings.CutPrefix(scanner.Text(), "data: ")
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
				if p.FunctionCall != nil {
					// Before yielding tool events, if we were previously thinking, signal done
					if lastEventWasThinking {
						if !yield(llms.StreamStatusThinkingDone) {
							return
						}
						lastEventWasThinking = false
					}
					// Note: Gemini's streaming API doesn't have partial tool calls.
					// Google doesn't provide tool call IDs, so we generate our own
					// using a combination of function name and a timestamp to ensure uniqueness
					uniqueID := fmt.Sprintf("%s-%d", p.FunctionCall.Name, time.Now().UnixNano())
					s.message.ToolCalls = append(s.message.ToolCalls, llms.ToolCall{
						ID:        uniqueID,
						Name:      p.FunctionCall.Name,
						Arguments: p.FunctionCall.Args,
					})
					if !yield(llms.StreamStatusToolCallBegin) {
						return
					}
					if !yield(llms.StreamStatusToolCallDelta) {
						return
					}
					if !yield(llms.StreamStatusToolCallReady) {
						return
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
