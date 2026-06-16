package llms

import (
	"encoding/json"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/tools"
)

type UpdateType string

const (
	UpdateTypeToolStart    UpdateType = "tool_start"
	UpdateTypeToolDelta    UpdateType = "tool_delta"
	UpdateTypeToolStatus   UpdateType = "tool_status"
	UpdateTypeToolDone     UpdateType = "tool_done"
	UpdateTypeText         UpdateType = "text"
	UpdateTypeImage        UpdateType = "image"
	UpdateTypeAudio        UpdateType = "audio"
	UpdateTypeThinking     UpdateType = "thinking"
	UpdateTypeThinkingDone UpdateType = "thinking_done"
	UpdateTypeMessageStart UpdateType = "message_start"
	UpdateTypeSearch       UpdateType = "search"
)

type Update interface {
	Type() UpdateType
}

type ToolStartUpdate struct {
	ToolCallID string
	Tool       tools.Tool
}

func (u ToolStartUpdate) Type() UpdateType {
	return UpdateTypeToolStart
}

type ToolDeltaUpdate struct {
	ToolCallID string
	Delta      json.RawMessage
}

func (u ToolDeltaUpdate) Type() UpdateType {
	return UpdateTypeToolDelta
}

type ToolStatusUpdate struct {
	ToolCallID string
	Status     string
	Tool       tools.Tool
}

func (u ToolStatusUpdate) Type() UpdateType {
	return UpdateTypeToolStatus
}

type ToolDoneUpdate struct {
	ToolCallID string
	Result     tools.Result
	Tool       tools.Tool
	// Metadata carries provider-specific fields that should be forwarded unchanged.
	Metadata map[string]string
}

func (u ToolDoneUpdate) Type() UpdateType {
	return UpdateTypeToolDone
}

type TextUpdate struct {
	Text string
}

func (u TextUpdate) Type() UpdateType {
	return UpdateTypeText
}

type ImageUpdate struct {
	URL      string
	MimeType string
	// Metadata holds provider-specific metadata that should be forwarded unchanged.
	Metadata map[string]string
}

func (u ImageUpdate) Type() UpdateType {
	return UpdateTypeImage
}

type AudioUpdate struct {
	URL      string
	MimeType string
	// Metadata holds provider-specific metadata that should be forwarded unchanged.
	Metadata map[string]string
}

func (u AudioUpdate) Type() UpdateType {
	return UpdateTypeAudio
}

type ThinkingUpdate struct {
	content.Thought
}

func (u ThinkingUpdate) Type() UpdateType {
	return UpdateTypeThinking
}

type ThinkingDoneUpdate struct{}

func (u ThinkingDoneUpdate) Type() UpdateType {
	return UpdateTypeThinkingDone
}

type MessageStartUpdate struct {
	MessageID string
}

func (u MessageStartUpdate) Type() UpdateType {
	return UpdateTypeMessageStart
}

// SearchSource is one result a provider-run search drew on, when the provider reports them.
type SearchSource struct {
	Title string
	URL   string
}

// SearchActivity describes a single provider-run search the model performed (e.g. xAI's
// web_search / x_search Agent Tools). It is informational: these searches run server-side, so
// the caller never executes anything, but surfacing the query lets a UI show what was looked up.
type SearchActivity struct {
	// Source is the surface searched, e.g. "web" or "x".
	Source string
	// Query is the search query the model issued.
	Query string
	// ResultCount is how many results the search returned, or 0 when the provider omits it.
	ResultCount int
	// Sources are the result sources when the provider includes them, otherwise empty.
	Sources []SearchSource
}

type SearchUpdate struct {
	SearchActivity
}

func (u SearchUpdate) Type() UpdateType {
	return UpdateTypeSearch
}
