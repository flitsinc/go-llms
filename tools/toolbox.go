package tools

import (
	"encoding/json"
	"fmt"
)

type Toolbox struct {
	tools map[string]Tool
}

// Box returns a new Toolbox containing the given tools.
func Box(tools ...Tool) *Toolbox {
	t := &Toolbox{
		tools: make(map[string]Tool),
	}
	for _, tool := range tools {
		t.Add(tool)
	}
	return t
}

// Add adds a tool to the toolbox.
func (t *Toolbox) Add(tool Tool) {
	funcName := tool.FuncName()
	if _, ok := t.tools[funcName]; ok {
		panic(fmt.Sprintf("tool %q already exists", funcName))
	}
	t.tools[funcName] = tool
}

func (t *Toolbox) All() []Tool {
	tools := []Tool{}
	for _, tool := range t.tools {
		tools = append(tools, tool)
	}
	return tools
}

// Get returns the tool with the given function name.
func (t *Toolbox) Get(funcName string) Tool {
	return t.tools[funcName]
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
