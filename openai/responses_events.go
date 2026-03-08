package openai

import (
	"encoding/json"
	"fmt"
	"os"
	"strings"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
)

// responsesEventProcessor contains the shared state and logic for processing
// Responses API streaming events. It is embedded by both ResponsesStream (SSE)
// and WebSocketStream (WebSocket) to avoid duplicating event-handling code.
type responsesEventProcessor struct {
	message        llms.Message
	lastText       string
	lastImage      struct{ URL, MIME string }
	usage          *responsesUsage
	lastThought    *content.Thought
	activeToolCall *llms.ToolCall
	responseID     string
	debugger       llms.Debugger
	err            error
}

// processEvent handles a single parsed ResponseStreamEvent.
// It calls yield() for each StreamStatus produced (some events produce multiple).
// Returns done=true when the stream should end (response.completed or error).
func (p *responsesEventProcessor) processEvent(
	event ResponseStreamEvent,
	rawJSON []byte,
	yield func(llms.StreamStatus) bool,
) (done bool) {
	switch event.Type {
	case "response.created":
		p.message.Role = "assistant"
		var resp struct {
			ID string `json:"id"`
		}
		if err := json.Unmarshal(event.Response, &resp); err == nil {
			p.responseID = resp.ID
		}

	case "response.output_item.added":
		var item struct {
			Type string `json:"type"`
			ID   string `json:"id"`
		}
		if err := json.Unmarshal(event.Item, &item); err == nil {
			switch item.Type {
			case "message":
				p.message.ID = item.ID
				if !yield(llms.StreamStatusMessageStart) {
					return true
				}
			case "function_call":
				if p.activeToolCall != nil {
					if !yield(llms.StreamStatusToolCallReady) {
						return true
					}
				}
				var fc FunctionCall
				if err := json.Unmarshal(event.Item, &fc); err == nil {
					metadata := map[string]string{
						"openai:item_id":   fc.ID,
						"openai:item_type": item.Type,
					}
					llmToolCall := llms.ToolCall{
						ID:        fc.CallID,
						Name:      fc.Name,
						Arguments: json.RawMessage{},
						Metadata:  metadata,
					}
					p.message.ToolCalls = append(p.message.ToolCalls, llmToolCall)
					p.activeToolCall = &p.message.ToolCalls[len(p.message.ToolCalls)-1]
					if !yield(llms.StreamStatusToolCallBegin) {
						return true
					}
				}
			case "custom_tool_call":
				if p.activeToolCall != nil {
					if !yield(llms.StreamStatusToolCallReady) {
						return true
					}
				}
				var ctc struct {
					Type   string `json:"type"`
					ID     string `json:"id"`
					Name   string `json:"name"`
					CallID string `json:"call_id"`
					Input  string `json:"input"`
				}
				if err := json.Unmarshal(event.Item, &ctc); err != nil {
					ctc.ID = item.ID
					ctc.Name = "custom"
				}
				metadata := map[string]string{
					"openai:item_id":   ctc.ID,
					"openai:item_type": item.Type,
				}
				llmToolCall := llms.ToolCall{
					ID:        ctc.CallID,
					Name:      ctc.Name,
					Arguments: json.RawMessage(ctc.Input),
					Metadata:  metadata,
				}
				p.message.ToolCalls = append(p.message.ToolCalls, llmToolCall)
				p.activeToolCall = &p.message.ToolCalls[len(p.message.ToolCalls)-1]
				if !yield(llms.StreamStatusToolCallBegin) {
					return true
				}
			case "reasoning":
				p.message.Content.AppendThoughtWithID(item.ID, "", true)
				p.lastThought = &content.Thought{ID: item.ID, Summary: true}
				if !yield(llms.StreamStatusThinking) {
					return true
				}
			}
		}

	case "response.output_text.delta":
		var delta struct {
			Delta string `json:"delta"`
		}
		if err := json.Unmarshal(rawJSON, &delta); err == nil {
			p.lastText = delta.Delta
			p.message.Content.Append(p.lastText)
			if p.lastText != "" {
				if !yield(llms.StreamStatusText) {
					return true
				}
			}
		}

	case "response.function_call_arguments.delta":
		if p.activeToolCall != nil {
			var delta struct {
				Delta string `json:"delta"`
			}
			if err := json.Unmarshal(rawJSON, &delta); err == nil {
				p.activeToolCall.Arguments = append(p.activeToolCall.Arguments, []byte(delta.Delta)...)
				if !yield(llms.StreamStatusToolCallDelta) {
					return true
				}
			}
		}

	case "response.function_call_arguments.done":
		if p.activeToolCall != nil {
			if !yield(llms.StreamStatusToolCallReady) {
				return true
			}
			p.activeToolCall = nil
		}

	case "response.custom_tool_call_input.delta":
		if p.activeToolCall != nil {
			var delta struct {
				Delta string `json:"delta"`
			}
			if err := json.Unmarshal(rawJSON, &delta); err == nil {
				p.activeToolCall.Arguments = append(p.activeToolCall.Arguments, []byte(delta.Delta)...)
				if !yield(llms.StreamStatusToolCallDelta) {
					return true
				}
			}
		}

	case "response.custom_tool_call_input.done":
		if p.activeToolCall != nil {
			if !yield(llms.StreamStatusToolCallReady) {
				return true
			}
			p.activeToolCall = nil
		}

	case "response.reasoning_summary_part.added":
		var partEvent struct {
			Part struct {
				Type string `json:"type"`
				Text string `json:"text"`
			} `json:"part"`
			ItemID string `json:"item_id"`
		}
		if err := json.Unmarshal(rawJSON, &partEvent); err == nil && partEvent.Part.Text != "" {
			p.message.Content.AppendThoughtWithID(partEvent.ItemID, partEvent.Part.Text, true)
			p.lastThought = &content.Thought{ID: partEvent.ItemID, Text: partEvent.Part.Text, Summary: true}
			if !yield(llms.StreamStatusThinking) {
				return true
			}
		}

	case "response.reasoning_summary_part.done":
		// Complete text already streamed via delta events.

	case "response.reasoning_summary_text.delta":
		var deltaEvent struct {
			Delta  string `json:"delta"`
			ItemID string `json:"item_id"`
		}
		if err := json.Unmarshal(rawJSON, &deltaEvent); err == nil && deltaEvent.Delta != "" {
			p.message.Content.AppendThoughtWithID(deltaEvent.ItemID, deltaEvent.Delta, true)
			p.lastThought = &content.Thought{ID: deltaEvent.ItemID, Text: deltaEvent.Delta, Summary: true}
			if !yield(llms.StreamStatusThinking) {
				return true
			}
		}

	case "response.reasoning_summary_text.done":
		var doneEvent struct {
			Text   string `json:"text"`
			ItemID string `json:"item_id"`
		}
		if err := json.Unmarshal(rawJSON, &doneEvent); err == nil {
			thought := p.message.Content.AppendThoughtWithID(doneEvent.ItemID, "", true)
			if doneEvent.Text != "" {
				thought.Text = doneEvent.Text
			}
			thought.Summary = true
			p.lastThought = nil
			if !yield(llms.StreamStatusThinkingDone) {
				return true
			}
		}

	case "response.output_item.done":
		var itemHdr struct {
			Type   string `json:"type"`
			Status string `json:"status"`
		}
		if err := json.Unmarshal(event.Item, &itemHdr); err == nil {
			switch itemHdr.Type {
			case "image_generation_call":
				if itemHdr.Status != "completed" {
					break
				}
				var img struct {
					ID            string `json:"id"`
					Background    string `json:"background"`
					OutputFormat  string `json:"output_format"`
					Quality       string `json:"quality"`
					Result        string `json:"result"`
					RevisedPrompt string `json:"revised_prompt"`
					Size          string `json:"size"`
				}
				err := json.Unmarshal(event.Item, &img)
				if err != nil {
					p.err = fmt.Errorf("failed to parse image generation result: %w", err)
					return true
				}
				// Debug: log the image generation result details
				fmt.Fprintf(os.Stderr, "[go-llms] image_generation_call done: id=%s status=%s output_format=%s background=%s quality=%s size=%s result_len=%d revised_prompt=%q\n",
					img.ID, itemHdr.Status, img.OutputFormat, img.Background, img.Quality, img.Size, len(img.Result), img.RevisedPrompt)
				if img.Result == "" {
					break
				}
				mime := "image/png"
				switch img.OutputFormat {
				case "png":
					mime = "image/png"
				case "jpeg", "jpg":
					mime = "image/jpeg"
				case "webp":
					mime = "image/webp"
				case "gif":
					mime = "image/gif"
				}
				dataURI := content.BuildDataURI(mime, img.Result)
				p.lastImage.URL = dataURI
				p.lastImage.MIME = mime
				p.message.Content = append(p.message.Content, &content.ImageURL{URL: dataURI, MimeType: mime})
				if !yield(llms.StreamStatusImage) {
					return true
				}
				if event.Usage != nil {
					p.usage = event.Usage
				}
			case "reasoning":
				var reasoningItem Reasoning
				if err := json.Unmarshal(event.Item, &reasoningItem); err == nil {
					var summaryBuilder strings.Builder
					for _, part := range reasoningItem.Summary {
						if part.Text == "" {
							continue
						}
						if summaryBuilder.Len() > 0 {
							summaryBuilder.WriteString("\n")
						}
						summaryBuilder.WriteString(part.Text)
					}
					thought := p.message.Content.AppendThoughtWithID(reasoningItem.ID, "", true)
					if summaryBuilder.Len() > 0 {
						thought.Text = summaryBuilder.String()
					}
					if p.lastThought != nil {
						p.lastThought = nil
						if !yield(llms.StreamStatusThinkingDone) {
							return true
						}
					}
				}
			}
		}

	case "response.image_generation_call.completed":
		itemStr := string(event.Item)
		truncLen := len(itemStr)
		if truncLen > 200 {
			truncLen = 200
		}
		fmt.Fprintf(os.Stderr, "[go-llms] image_generation_call.completed event received: %s\n", itemStr[:truncLen])

	case "response.image_generation_call.in_progress",
		"response.image_generation_call.generating",
		"response.image_generation_call.partial_image":
		// Ignored for now.

	case "response.completed":
		if event.Response != nil {
			var response struct {
				Usage *responsesUsage `json:"usage"`
			}
			if err := json.Unmarshal(event.Response, &response); err == nil && response.Usage != nil {
				p.usage = response.Usage
			}
		}

		if p.activeToolCall != nil {
			if !yield(llms.StreamStatusToolCallReady) {
				return true
			}
			p.activeToolCall = nil
		}
		return true

	case "error":
		if event.Error != nil {
			p.err = fmt.Errorf("stream error (%s): %s", event.Error.Code, event.Error.Message)
		}
		return true
	}

	return false
}
