package tools

import (
	"encoding/json"
	"fmt"
)

type Toolbox struct {
	// Tools preserves insertion order of tool names.
	tools []Tool
	// Choice controls tool selection policy for providers.
	Choice Choice
}

// Box returns a new Toolbox containing the given tools.
func Box(tools ...Tool) *Toolbox {
	t := &Toolbox{
		tools: make([]Tool, 0, len(tools)),
	}
	for _, tool := range tools {
		t.Add(tool)
	}
	return t
}

// Add adds a tool to the toolbox.
func (t *Toolbox) Add(tool Tool) {
	funcName := tool.FuncName()
	for _, existing := range t.tools {
		if existing.FuncName() == funcName {
			panic(fmt.Sprintf("tool %q already exists", funcName))
		}
	}
	t.tools = append(t.tools, tool)
}

func (t *Toolbox) All() []Tool {
	// Be nil-safe so callers can iterate even when the toolbox hasn't been configured.
	tools := []Tool{}
	if t == nil {
		return tools
	}
	return append(tools, t.tools...)
}

// Get returns the tool with the given function name.
func (t *Toolbox) Get(funcName string) Tool {
	// Be nil-safe so code paths that receive unexpected tool calls don't panic.
	if t == nil {
		return nil
	}
	for _, tool := range t.tools {
		if tool.FuncName() == funcName {
			return tool
		}
	}
	return nil
}

// Run runs the tool with the given name and parameters, which should be provided as a JSON string.
func (t *Toolbox) Run(r Runner, funcName string, params json.RawMessage) Result {
	tool := t.Get(funcName)
	if tool == nil {
		err := fmt.Errorf("tool %q not found", funcName)
		return Error(err)
	}
	return tool.Run(r, params)
}
