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
