package tools

import (
	"encoding/json"
	"fmt"
)

// NotFoundError is the error carried by the result of a tool call naming a
// function that isn't in the toolbox. Consumers can use errors.As to tell this
// apart from an error the tool itself returned, e.g. to decide that the turn is
// recoverable and the model should be given a chance to pick another tool.
type NotFoundError struct {
	FuncName string
}

func (e *NotFoundError) Error() string {
	return fmt.Sprintf("tool %q not found", e.FuncName)
}

// unknownTool stands in for a function name the model called that isn't in the
// toolbox. It runs nothing; its result is the same "tool not found" error
// Toolbox.Run produces for the same name.
type unknownTool struct {
	funcName string
	grammar  Grammar
}

// Unknown returns a placeholder Tool for a function name that isn't in the
// toolbox. It exists so a model calling a tool that doesn't exist (a
// hallucinated name, or a real tool that was removed while the conversation
// history still mentions it) is an in-band tool error the model can recover
// from, rather than something that aborts the turn.
//
// Consumers that act on a tool before it produces a result — displaying it,
// spawning an executor, starting a workflow — must check IsUnknown first and do
// nothing for those: there is no tool to run and the result is already known.
func Unknown(funcName string) Tool {
	return &unknownTool{
		funcName: funcName,
		// A no-parameter schema, so consumers can call Grammar().Schema()
		// uniformly. The schema is never sent to a provider; an unknown tool is
		// by definition not in the toolbox.
		grammar: NewJSONGrammarWithSchema(&FunctionSchema{
			Name:        funcName,
			Description: "This tool does not exist.",
			Parameters:  ValueSchema{Type: "object"},
		}, true /*skipValidation*/),
	}
}

// IsUnknown reports whether the tool is a placeholder returned by Unknown, i.e.
// the model called a function name that doesn't exist.
func IsUnknown(t Tool) bool {
	_, ok := t.(*unknownTool)
	return ok
}

func (t *unknownTool) Label() string { return fmt.Sprintf("Unknown tool %q", t.funcName) }

func (t *unknownTool) Description() string { return "This tool does not exist." }

func (t *unknownTool) FuncName() string { return t.funcName }

func (t *unknownTool) Grammar() Grammar { return t.grammar }

func (t *unknownTool) Run(r Runner, params json.RawMessage) Result {
	return Error(&NotFoundError{FuncName: t.funcName})
}
