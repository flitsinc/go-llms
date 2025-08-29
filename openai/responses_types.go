package openai

import (
	"encoding/json"

	"github.com/flitsinc/go-llms/tools"
)

// ResponseInput is an interface for all input types
type ResponseInput interface {
	responseInput()
}

// InputContent is an interface for content types within messages
type InputContent interface {
	inputContent()
}

// ResponseItem is a generic interface for response items
type ResponseItem interface {
	responseItem()
}

// ResponseTool is an interface for all tool types
type ResponseTool interface {
	responseTool()
}

// TextInput implements ResponseInput for simple text input
type TextInput string

func (TextInput) responseInput() {}

// InputMessage implements ResponseInput for message input
type InputMessage struct {
	Type    string         `json:"type"` // "message"
	Role    string         `json:"role"` // "user", "assistant", "system", "developer"
	Content []InputContent `json:"content,omitempty"`
	Status  string         `json:"status,omitempty"` // "in_progress", "completed", "incomplete"
	ID      string         `json:"id,omitempty"`
}

func (InputMessage) responseInput() {}
func (InputMessage) responseItem()  {}

// InputText implements InputContent for text content
type InputText struct {
	Type string `json:"type"` // "input_text"
	Text string `json:"text"`
}

func (InputText) inputContent() {}

// InputImage implements InputContent for image content
type InputImage struct {
	Type     string `json:"type"` // "input_image"
	ImageURL string `json:"image_url,omitempty"`
	FileID   string `json:"file_id,omitempty"`
	Detail   string `json:"detail"` // "high", "low", "auto"
}

func (InputImage) inputContent() {}

// InputFile implements InputContent for file content
type InputFile struct {
	Type     string `json:"type"` // "input_file"
	FileID   string `json:"file_id,omitempty"`
	FileData string `json:"file_data,omitempty"`
	Filename string `json:"filename,omitempty"`
}

func (InputFile) inputContent() {}

// OutputMessage implements ResponseItem for output messages
type OutputMessage struct {
	Type    string          `json:"type"` // "message"
	ID      string          `json:"id"`
	Role    string          `json:"role"`   // "assistant"
	Status  string          `json:"status"` // "in_progress", "completed", "incomplete"
	Content []OutputContent `json:"content"`
}

func (OutputMessage) responseItem() {}

func (OutputMessage) responseInput() {}

// OutputContent is an interface for output content types
type OutputContent interface {
	outputContent()
}

// OutputText implements OutputContent for text output
type OutputText struct {
	Type        string       `json:"type"` // "output_text"
	Text        string       `json:"text"`
	Annotations []Annotation `json:"annotations"`
	Logprobs    []Logprob    `json:"logprobs,omitempty"`
}

func (OutputText) outputContent() {}

// Refusal implements OutputContent for refusals
type Refusal struct {
	Type    string `json:"type"` // "refusal"
	Refusal string `json:"refusal"`
}

func (Refusal) outputContent() {}

// Annotation is an interface for different annotation types
type Annotation interface {
	annotation()
}

// FileCitation implements Annotation
type FileCitation struct {
	Type     string `json:"type"` // "file_citation"
	FileID   string `json:"file_id"`
	Filename string `json:"filename"`
	Index    int    `json:"index"`
}

func (FileCitation) annotation() {}

// URLCitation implements Annotation
type URLCitation struct {
	Type       string `json:"type"` // "url_citation"
	URL        string `json:"url"`
	Title      string `json:"title"`
	StartIndex int    `json:"start_index"`
	EndIndex   int    `json:"end_index"`
}

func (URLCitation) annotation() {}

// ContainerFileCitation implements Annotation
type ContainerFileCitation struct {
	Type        string `json:"type"` // "container_file_citation"
	ContainerID string `json:"container_id"`
	FileID      string `json:"file_id"`
	Filename    string `json:"filename"`
	StartIndex  int    `json:"start_index"`
	EndIndex    int    `json:"end_index"`
}

func (ContainerFileCitation) annotation() {}

// FilePath implements Annotation
type FilePath struct {
	Type   string `json:"type"` // "file_path"
	FileID string `json:"file_id"`
	Index  int    `json:"index"`
}

func (FilePath) annotation() {}

// Logprob represents log probability information
type Logprob struct {
	Token       string       `json:"token"`
	Logprob     float64      `json:"logprob"`
	Bytes       []int        `json:"bytes"`
	TopLogprobs []TopLogprob `json:"top_logprobs"`
}

// TopLogprob represents top log probabilities
type TopLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes"`
}

// FunctionCall implements ResponseItem for function calls
type FunctionCall struct {
	Type      string `json:"type"` // "function_call"
	ID        string `json:"id,omitempty"`
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
	CallID    string `json:"call_id"`
	Status    string `json:"status,omitempty"` // "in_progress", "completed", "incomplete"
}

func (FunctionCall) responseItem()  {}
func (FunctionCall) responseInput() {}

// CustomToolCall implements ResponseItem for custom tools (grammar/text tools)
type CustomToolCall struct {
	Type   string `json:"type"` // "custom_tool_call"
	ID     string `json:"id,omitempty"`
	Name   string `json:"name"`
	Input  string `json:"input"`
	CallID string `json:"call_id"`
	Status string `json:"status,omitempty"` // "in_progress", "completed", "incomplete"
}

func (CustomToolCall) responseItem()  {}
func (CustomToolCall) responseInput() {}

// FunctionCallOutput implements ResponseInput and ResponseItem
type FunctionCallOutput struct {
	Type   string `json:"type"` // "function_call_output"
	ID     string `json:"id,omitempty"`
	CallID string `json:"call_id"`
	Output string `json:"output"`
	Status string `json:"status,omitempty"` // "in_progress", "completed", "incomplete"
}

func (FunctionCallOutput) responseInput() {}
func (FunctionCallOutput) responseItem()  {}

// Reasoning implements ResponseItem for reasoning
type Reasoning struct {
	Type             string             `json:"type"` // "reasoning"
	ID               string             `json:"id"`
	Summary          []ReasoningSummary `json:"summary"`
	Status           string             `json:"status,omitempty"` // "in_progress", "completed", "incomplete"
	EncryptedContent string             `json:"encrypted_content,omitempty"`
}

func (Reasoning) responseItem()  {}
func (Reasoning) responseInput() {}

// ReasoningSummary represents reasoning summary content
type ReasoningSummary struct {
	Type string `json:"type"` // "summary_text"
	Text string `json:"text"`
}

// FileSearchCall implements ResponseItem
type FileSearchCall struct {
	Type    string             `json:"type"` // "file_search_call"
	ID      string             `json:"id"`
	Queries []string           `json:"queries"`
	Status  string             `json:"status"` // "in_progress", "searching", "incomplete", "failed"
	Results []FileSearchResult `json:"results,omitempty"`
}

func (FileSearchCall) responseItem() {}

// FileSearchResult represents a file search result
type FileSearchResult struct {
	FileID     string            `json:"file_id,omitempty"`
	Filename   string            `json:"filename,omitempty"`
	Score      float64           `json:"score,omitempty"`
	Text       string            `json:"text,omitempty"`
	Attributes map[string]string `json:"attributes,omitempty"`
}

// WebSearchCall implements ResponseItem
type WebSearchCall struct {
	Type   string          `json:"type"` // "web_search_call"
	ID     string          `json:"id"`
	Status string          `json:"status"`
	Action WebSearchAction `json:"action"`
}

func (WebSearchCall) responseItem() {}

// WebSearchAction is an interface for web search actions
type WebSearchAction interface {
	webSearchAction()
}

// SearchAction implements WebSearchAction
type SearchAction struct {
	Type  string `json:"type"` // "search"
	Query string `json:"query"`
}

func (SearchAction) webSearchAction() {}

// OpenPageAction implements WebSearchAction
type OpenPageAction struct {
	Type string `json:"type"` // "open_page"
	URL  string `json:"url"`
}

func (OpenPageAction) webSearchAction() {}

// FindAction implements WebSearchAction
type FindAction struct {
	Type    string `json:"type"` // "find"
	Pattern string `json:"pattern"`
	URL     string `json:"url"`
}

func (FindAction) webSearchAction() {}

// ComputerCall implements ResponseItem
type ComputerCall struct {
	Type                string         `json:"type"` // "computer_call"
	ID                  string         `json:"id"`
	CallID              string         `json:"call_id"`
	Action              ComputerAction `json:"action"`
	Status              string         `json:"status"`
	PendingSafetyChecks []SafetyCheck  `json:"pending_safety_checks"`
}

func (ComputerCall) responseItem() {}

// ComputerAction is an interface for computer actions
type ComputerAction interface {
	computerAction()
}

// ClickAction implements ComputerAction
type ClickAction struct {
	Type   string `json:"type"` // "click"
	X      int    `json:"x"`
	Y      int    `json:"y"`
	Button string `json:"button"` // "left", "right", "wheel", "back", "forward"
}

func (ClickAction) computerAction() {}

// ScreenshotAction implements ComputerAction
type ScreenshotAction struct {
	Type string `json:"type"` // "screenshot"
}

func (ScreenshotAction) computerAction() {}

// TypeAction implements ComputerAction
type TypeAction struct {
	Type string `json:"type"` // "type"
	Text string `json:"text"`
}

func (TypeAction) computerAction() {}

// ComputerCallOutput implements ResponseItem
type ComputerCallOutput struct {
	Type                     string         `json:"type"` // "computer_call_output"
	ID                       string         `json:"id,omitempty"`
	CallID                   string         `json:"call_id"`
	Output                   ComputerOutput `json:"output"`
	Status                   string         `json:"status,omitempty"`
	AcknowledgedSafetyChecks []SafetyCheck  `json:"acknowledged_safety_checks,omitempty"`
}

func (ComputerCallOutput) responseItem() {}

// ComputerOutput represents computer call output
type ComputerOutput struct {
	Type     string `json:"type"` // "computer_screenshot"
	FileID   string `json:"file_id,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

// SafetyCheck represents a safety check
type SafetyCheck struct {
	ID      string `json:"id"`
	Code    string `json:"code,omitempty"`
	Message string `json:"message,omitempty"`
}

// ImageGenerationCall implements ResponseItem
type ImageGenerationCall struct {
	Type   string  `json:"type"` // "image_generation_call"
	ID     string  `json:"id"`
	Status string  `json:"status,omitempty"`
	Result *string `json:"result,omitempty"` // base64 encoded image
}

func (ImageGenerationCall) responseItem() {}

// CodeInterpreterCall implements ResponseItem
type CodeInterpreterCall struct {
	Type        string                  `json:"type"` // "code_interpreter_call"
	ID          string                  `json:"id"`
	ContainerID string                  `json:"container_id"`
	Code        *string                 `json:"code"`
	Status      string                  `json:"status"`
	Outputs     []CodeInterpreterOutput `json:"outputs"`
}

func (CodeInterpreterCall) responseItem() {}

// CodeInterpreterOutput is an interface for code interpreter outputs
type CodeInterpreterOutput interface {
	codeInterpreterOutput()
}

// CodeInterpreterLogs implements CodeInterpreterOutput
type CodeInterpreterLogs struct {
	Type string `json:"type"` // "logs"
	Logs string `json:"logs"`
}

func (CodeInterpreterLogs) codeInterpreterOutput() {}

// CodeInterpreterImage implements CodeInterpreterOutput
type CodeInterpreterImage struct {
	Type string `json:"type"` // "image"
	URL  string `json:"url"`
}

func (CodeInterpreterImage) codeInterpreterOutput() {}

// LocalShellCall implements ResponseItem
type LocalShellCall struct {
	Type   string           `json:"type"` // "local_shell_call"
	ID     string           `json:"id"`
	CallID string           `json:"call_id"`
	Action LocalShellAction `json:"action"`
	Status string           `json:"status"`
}

func (LocalShellCall) responseItem() {}

// LocalShellAction represents a local shell action
type LocalShellAction struct {
	Type             string            `json:"type"` // "exec"
	Command          []string          `json:"command"`
	Env              map[string]string `json:"env"`
	TimeoutMs        *int              `json:"timeout_ms,omitempty"`
	User             *string           `json:"user,omitempty"`
	WorkingDirectory *string           `json:"working_directory,omitempty"`
}

// LocalShellCallOutput implements ResponseItem
type LocalShellCallOutput struct {
	Type   string  `json:"type"` // "local_shell_call_output"
	ID     string  `json:"id"`
	Output string  `json:"output"`
	Status *string `json:"status,omitempty"`
}

func (LocalShellCallOutput) responseItem() {}

// MCPListTools implements ResponseItem
type MCPListTools struct {
	Type        string    `json:"type"` // "mcp_list_tools"
	ID          string    `json:"id"`
	ServerLabel string    `json:"server_label"`
	Tools       []MCPTool `json:"tools"`
	Error       *string   `json:"error,omitempty"`
}

func (MCPListTools) responseItem() {}

// MCPTool represents an MCP tool
type MCPTool struct {
	Name        string         `json:"name"`
	InputSchema map[string]any `json:"input_schema"`
	Description *string        `json:"description,omitempty"`
	Annotations map[string]any `json:"annotations,omitempty"`
}

// MCPApprovalRequest implements ResponseItem
type MCPApprovalRequest struct {
	Type        string `json:"type"` // "mcp_approval_request"
	ID          string `json:"id"`
	Name        string `json:"name"`
	Arguments   string `json:"arguments"`
	ServerLabel string `json:"server_label"`
}

func (MCPApprovalRequest) responseItem() {}

// MCPApprovalResponse implements ResponseItem
type MCPApprovalResponse struct {
	Type              string  `json:"type"` // "mcp_approval_response"
	ID                *string `json:"id,omitempty"`
	ApprovalRequestID string  `json:"approval_request_id"`
	Approve           bool    `json:"approve"`
	Reason            *string `json:"reason,omitempty"`
}

func (MCPApprovalResponse) responseItem() {}

// MCPCall implements ResponseItem
type MCPCall struct {
	Type        string  `json:"type"` // "mcp_call"
	ID          string  `json:"id"`
	Name        string  `json:"name"`
	Arguments   string  `json:"arguments"`
	ServerLabel string  `json:"server_label"`
	Output      *string `json:"output,omitempty"`
	Error       *string `json:"error,omitempty"`
}

func (MCPCall) responseItem() {}

// ItemReference implements ResponseItem
type ItemReference struct {
	Type string `json:"type,omitempty"` // "item_reference"
	ID   string `json:"id"`
}

func (ItemReference) responseItem() {}

// Tool types

// FunctionTool implements ResponseTool
type FunctionTool struct {
	Type        string             `json:"type"` // "function"
	Name        string             `json:"name"`
	Description string             `json:"description,omitempty"`
	Parameters  *tools.ValueSchema `json:"parameters"`
	Strict      bool               `json:"strict"`
}

func (FunctionTool) responseTool() {}

// FileSearchTool implements ResponseTool
type FileSearchTool struct {
	Type           string            `json:"type"` // "file_search"
	VectorStoreIDs []string          `json:"vector_store_ids"`
	Filters        *FileSearchFilter `json:"filters,omitempty"`
	MaxNumResults  *int              `json:"max_num_results,omitempty"`
	RankingOptions *RankingOptions   `json:"ranking_options,omitempty"`
}

func (FileSearchTool) responseTool() {}

// FileSearchFilter is an interface for file search filters
type FileSearchFilter interface {
	fileSearchFilter()
}

// ComparisonFilter implements FileSearchFilter
type ComparisonFilter struct {
	Type  string `json:"type"` // "eq", "ne", "gt", "gte", "lt", "lte"
	Key   string `json:"key"`
	Value any    `json:"value"` // string, number, or boolean
}

func (ComparisonFilter) fileSearchFilter() {}

// CompoundFilter implements FileSearchFilter
type CompoundFilter struct {
	Type    string             `json:"type"` // "and", "or"
	Filters []FileSearchFilter `json:"filters"`
}

func (CompoundFilter) fileSearchFilter() {}

// RankingOptions represents ranking options for file search
type RankingOptions struct {
	Ranker         string  `json:"ranker,omitempty"`
	ScoreThreshold float64 `json:"score_threshold,omitempty"`
}

// WebSearchTool implements ResponseTool
type WebSearchTool struct {
	Type              string        `json:"type"`                          // "web_search_preview" or "web_search_preview_2025_03_11"
	SearchContextSize string        `json:"search_context_size,omitempty"` // "low", "medium", "high"
	UserLocation      *UserLocation `json:"user_location,omitempty"`
}

func (WebSearchTool) responseTool() {}

// UserLocation represents user location for web search
type UserLocation struct {
	Type     string `json:"type"` // "approximate"
	City     string `json:"city,omitempty"`
	Country  string `json:"country,omitempty"`
	Region   string `json:"region,omitempty"`
	Timezone string `json:"timezone,omitempty"`
}

// ComputerUseTool implements ResponseTool
type ComputerUseTool struct {
	Type          string `json:"type"` // "computer_use_preview"
	DisplayWidth  int    `json:"display_width"`
	DisplayHeight int    `json:"display_height"`
	Environment   string `json:"environment"`
}

func (ComputerUseTool) responseTool() {}

// MCPToolConfig implements ResponseTool
type MCPToolConfig struct {
	Type            string            `json:"type"` // "mcp"
	ServerLabel     string            `json:"server_label"`
	ServerURL       string            `json:"server_url"`
	Headers         map[string]string `json:"headers,omitempty"`
	AllowedTools    any               `json:"allowed_tools,omitempty"`    // array of strings or filter object
	RequireApproval any               `json:"require_approval,omitempty"` // string ("always", "never") or filter object
}

func (MCPToolConfig) responseTool() {}

// CodeInterpreterTool implements ResponseTool
type CodeInterpreterTool struct {
	Type      string `json:"type"`      // "code_interpreter"
	Container any    `json:"container"` // string (container ID) or CodeInterpreterContainerAuto
}

func (CodeInterpreterTool) responseTool() {}

// CodeInterpreterContainerAuto represents auto container configuration
type CodeInterpreterContainerAuto struct {
	Type    string   `json:"type"` // "auto"
	FileIDs []string `json:"file_ids,omitempty"`
}

// ImageGenerationTool implements ResponseTool
type ImageGenerationTool struct {
	Type              string          `json:"type"`                         // "image_generation"
	Model             string          `json:"model,omitempty"`              // default: "gpt-image-1"
	Quality           string          `json:"quality,omitempty"`            // "low", "medium", "high", "auto"
	Size              string          `json:"size,omitempty"`               // "1024x1024", "1024x1536", "1536x1024", "auto"
	OutputFormat      string          `json:"output_format,omitempty"`      // "png", "webp", "jpeg"
	OutputCompression int             `json:"output_compression,omitempty"` // default: 100
	Background        string          `json:"background,omitempty"`         // "transparent", "opaque", "auto"
	Moderation        string          `json:"moderation,omitempty"`         // default: "auto"
	PartialImages     int             `json:"partial_images,omitempty"`     // 0-3, default: 0
	InputImageMask    *InputImageMask `json:"input_image_mask,omitempty"`
}

func (ImageGenerationTool) responseTool() {}

// InputImageMask represents an image mask for inpainting
type InputImageMask struct {
	FileID   string `json:"file_id,omitempty"`
	ImageURL string `json:"image_url,omitempty"`
}

// LocalShellTool implements ResponseTool
type LocalShellTool struct {
	Type string `json:"type"` // "local_shell"
}

func (LocalShellTool) responseTool() {}

// Response format types

// TextResponseFormat represents the text response format configuration.
type TextResponseFormat struct {
	Type        string             `json:"type"`                  // "text", "json_object", "json_schema"
	Name        string             `json:"name,omitempty"`        // only for Type == "json_schema"
	Schema      *tools.ValueSchema `json:"schema,omitempty"`      // only for Type == "json_schema"
	Description string             `json:"description,omitempty"` // only for Type == "json_schema"
	Strict      bool               `json:"strict,omitempty"`      // only for Type == "json_schema"
}

// Tool choice types

// ToolChoice can be a string or an object (keep for backwards compatibility where needed)
type ToolChoice any

// ResponsesToolChoice is a strongly-typed struct for forcing tool use in the Responses API.
// For function tools, the shape is: {"type":"function","name":"tool_name"}
// For custom tools, the shape is: {"type":"custom","name":"tool_name"}
// For hosted tools: {"type":"file_search"} etc.
// For MCP tools: {"type":"mcp","server_label":"...","name":"..."}
// When constraining to an allow-list, use AllowedToolsToolChoice below.

// AllowedToolsToolChoice represents the object to constrain allowed tools.
// Example:
//
//	{
//	  "type": "allowed_tools",
//	  "mode": "auto", // or "required"
//	  "tools": [
//	    {"type": "function", "name": "get_weather"},
//	    {"type": "custom", "name": "my_custom"}
//	  ]
//	}
type AllowedToolsToolChoice struct {
	Type  string `json:"type"`  // "allowed_tools"
	Mode  string `json:"mode"`  // "auto" | "required"
	Tools []any  `json:"tools"` // entries like {type:function,name:...}, {type:custom,name:...}, hosted/mcp, etc.
}

// ResponsesToolChoice is a strongly-typed struct for forcing tool use in the Responses API.
// For function tools, the shape is: {"type":"function","function":{"name":"tool_name"}}
type ResponsesToolChoice struct {
	Type     string                       `json:"type"`
	Function *ResponsesToolChoiceFunction `json:"function,omitempty"`
}

type ResponsesToolChoiceFunction struct {
	Name string `json:"name"`
}

// HostedToolChoice represents a hosted tool choice
type HostedToolChoice struct {
	Type string `json:"type"` // file_search, web_search_preview, computer_use_preview, code_interpreter, image_generation
}

// FunctionToolChoice represents a function tool choice
type FunctionToolChoice struct {
	Type string `json:"type"` // "function"
	Name string `json:"name"`
}

// MCPToolChoice represents an MCP tool choice
type MCPToolChoice struct {
	Type        string  `json:"type"` // "mcp"
	ServerLabel string  `json:"server_label"`
	Name        *string `json:"name,omitempty"`
}

// Reasoning configuration

// ReasoningConfig represents reasoning configuration
type ReasoningConfig struct {
	Effort          string  `json:"effort,omitempty"`           // "low", "medium", "high"
	Summary         *string `json:"summary,omitempty"`          // "auto", "concise", "detailed"
	GenerateSummary *string `json:"generate_summary,omitempty"` // deprecated
}

// responsesUsage represents token usage information
// Note: This is different from ChatCompletions API usage type (different field names)
type responsesUsage struct {
	InputTokens         int                 `json:"input_tokens"`
	OutputTokens        int                 `json:"output_tokens"`
	TotalTokens         int                 `json:"total_tokens"`
	InputTokensDetails  InputTokensDetails  `json:"input_tokens_details"`
	OutputTokensDetails OutputTokensDetails `json:"output_tokens_details"`
}

// InputTokensDetails represents input token details
// Note: Similar to ChatCompletions API but with fewer fields
type InputTokensDetails struct {
	CachedTokens int `json:"cached_tokens"`
}

// OutputTokensDetails represents output token details
// Note: Similar to ChatCompletions API but with fewer fields
type OutputTokensDetails struct {
	ReasoningTokens int `json:"reasoning_tokens"`
}

// Streaming types

// ResponseStreamEvent represents a streaming event from the Responses API.
type ResponseStreamEvent struct {
	Type           string `json:"type"`
	SequenceNumber int    `json:"sequence_number"`

	// response.created, response.in_progress, response.completed, response.failed, response.incomplete, response.queued
	Response json.RawMessage `json:"response,omitempty"`

	// response.output_item.added/done
	Item        json.RawMessage `json:"item,omitempty"`
	OutputIndex *int            `json:"output_index,omitempty"`

	// content_part events
	ContentIndex *int            `json:"content_index,omitempty"`
	ItemID       string          `json:"item_id,omitempty"`
	Part         json.RawMessage `json:"part,omitempty"`

	// delta events
	Delta json.RawMessage `json:"delta,omitempty"`

	// .done events
	Arguments string `json:"arguments,omitempty"` // function_call_arguments.done
	Text      string `json:"text,omitempty"`      // output_text.done
	Refusal   string `json:"refusal,omitempty"`   // refusal.done
	Code      string `json:"code,omitempty"`      // code_interpreter_call_code.done

	// error event
	Error *StreamError `json:"error,omitempty"`

	// usage event
	Usage *responsesUsage `json:"usage,omitempty"`

	// other fields for specific events
	Annotation      json.RawMessage `json:"annotation,omitempty"`
	AnnotationIndex *int            `json:"annotation_index,omitempty"`
}

// StreamError represents an error object in the stream.
type StreamError struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}
