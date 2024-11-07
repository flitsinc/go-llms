package llms

import (
	"encoding/json"
)

type ToolCall struct {
	ID        string
	Name      string
	Arguments json.RawMessage
}
