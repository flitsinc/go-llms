package llms

import (
	"github.com/flitsinc/go-llms/tools"
)

type UpdateType string

const (
	UpdateTypeToolStart  UpdateType = "tool_start"
	UpdateTypeToolStatus UpdateType = "tool_status"
	UpdateTypeToolDone   UpdateType = "tool_done"
	UpdateTypeText       UpdateType = "text"
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
