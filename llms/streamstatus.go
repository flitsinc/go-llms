package llms

type StreamStatus int

const (
	// StreamStatusNone means either the stream hasn't started, or it has finished.
	StreamStatusNone StreamStatus = iota
	// StreamStatusText means the stream produced more text content.
	StreamStatusText
	// StreamStatusToolCallBegin means the stream started a tool call. The name of the function is available, but not the arguments.
	StreamStatusToolCallBegin
	// StreamStatusToolCallData means the stream is streaming the arguments for a tool call.
	StreamStatusToolCallData
	// StreamStatusToolCallReady means the stream finished streaming the arguments for a tool call.
	StreamStatusToolCallReady

	// TODO: Add thinking related stream statuses?
)
