package tools

import (
	"encoding/json"
	"fmt"
	"reflect"
	"strings"
	"sync"
)

type Tool interface {
	// Label returns a nice human readable title for the tool.
	Label() string
	// Description returns the description of the tool.
	Description() string
	// FuncName returns the function name for the tool.
	FuncName() string
	// Run runs the tool with the provided parameters.
	Run(r Runner, params json.RawMessage) Result
	// Grammar returns the grammar configuration for this tool.
	Grammar() Grammar
}

var jsonRawMessageType = reflect.TypeOf(json.RawMessage{})

// Func returns a tool for a function implementation with the given name and description.
func Func[Params any](label, description, funcName string, fn func(r Runner, params Params) Result) Tool {
	var zeroParams Params
	schemaType := reflect.TypeOf(zeroParams)
	if schemaType.Kind() != reflect.Struct && schemaType != jsonRawMessageType {
		panic("Params must be a struct or json.RawMessage")
	}
	// Create JSON grammar up front so we can avoid runtime type assertions.
	jg := NewJSONGrammar(funcName, description, schemaType)
	t := &tool{
		label:       label,
		description: description,
		grammar:     jg,
		funcName:    funcName,
		fn: func(r Runner, params json.RawMessage) Result {
			// Validate against the known JSON grammar when applicable.
			if !jg.SkipValidation() {
				params = omitOptionalNulls(jg.Schema().Parameters, params)
				if err := validateJSON(jg.Schema(), params); err != nil {
					return ErrorWithLabel("LLM misbehaved", fmt.Errorf("validation error for %s: %w", funcName, err))
				}
			}
			var p Params
			if err := json.Unmarshal(params, &p); err != nil {
				return ErrorWithLabel("LLM misbehaved", fmt.Errorf("unmarshal error for %s: %w", funcName, err))
			}
			return fn(r, p)
		},
	}
	return t
}

// External returns a tool where the schema is provided explicitly, and the
// handler function receives raw JSON parameters. This is suitable for external
// tools where schema generation via reflection is not possible or desired.
func External(label string, schema *FunctionSchema, fn func(r Runner, params json.RawMessage) Result) Tool {
	return externalTool("External", label, schema, nil, fn)
}

// ExternalDeclared returns an External tool that also keeps declaration, the
// JSON function declaration (name, description, parameters and any other
// member) that schema was decoded from. Its grammar is a DeclaredJSONGrammar,
// so a provider that can show the model the caller's own declaration sends it
// instead of re-encoding schema (openai.ChatCompletionsAPI.WithDeclaredTools):
// the same JSON value, members in the declared order, compacted, with HTML
// characters escaped. Decoding imposes ValueSchema's member order on schema
// and drops the keywords ValueSchema has no field for, so the re-encoded
// schema differs from the declaration.
//
// An empty or null declaration is none, and the tool is the one External
// returns. Any other declaration must be a JSON object whose "name" member
// (matched case-sensitively, empty when absent) equals schema.Name, the name
// tool choice and the toolbox go by; ExternalDeclared panics otherwise, as it
// does for a nil schema.
func ExternalDeclared(label string, schema *FunctionSchema, declaration json.RawMessage, fn func(r Runner, params json.RawMessage) Result) Tool {
	return externalTool("ExternalDeclared", label, schema, declaration, fn)
}

// externalTool builds the tool External and ExternalDeclared return, and
// names constructor in its panics. A nil declaration declares nothing.
func externalTool(constructor, label string, schema *FunctionSchema, declaration json.RawMessage, fn func(r Runner, params json.RawMessage) Result) Tool {
	if schema == nil {
		panic(constructor + " requires a non-nil schema")
	}
	grammar := NewJSONGrammarWithSchema(schema, true /*skipValidation*/)
	if declares(constructor, schema.Name, declaration) {
		grammar = &declaredJSONGrammar{JSONGrammar: grammar, declaration: declaration}
	}
	return &tool{
		label:       label,
		description: schema.Description, // Use description from schema
		funcName:    schema.Name,        // Use name from schema
		fn:          fn,                 // Handler directly accepts raw JSON
		grammar:     grammar,
	}
}

// declares reports whether declaration declares a function: false when it is
// empty or null, true when it is a JSON object whose "name" member is name,
// and a panic naming constructor otherwise. It reads the "name" member
// case-sensitively, as encoding/json/v2 and vLLM's request models do
// (decoding into a struct with encoding/json would match "Name" too).
func declares(constructor, name string, declaration json.RawMessage) bool {
	if len(declaration) == 0 {
		return false
	}
	var members map[string]json.RawMessage
	if err := json.Unmarshal(declaration, &members); err != nil {
		panic(fmt.Sprintf("%s requires a declaration that is a JSON object: %v", constructor, err))
	}
	if members == nil {
		return false // null
	}
	var declared string
	if raw, ok := members["name"]; ok {
		if err := json.Unmarshal(raw, &declared); err != nil {
			panic(fmt.Sprintf("%s requires a declaration whose name is a string: %v", constructor, err))
		}
	}
	if declared != name {
		panic(fmt.Sprintf("%s requires a declaration named %q, not %q", constructor, name, declared))
	}
	return true
}

type tool struct {
	label, description, funcName string

	fn func(r Runner, params json.RawMessage) Result

	// If non-nil, this is a grammar-based tool (custom tool for providers that support it).
	grammar Grammar
}

func (t *tool) Label() string {
	return t.label
}

func (t *tool) Description() string {
	return t.description
}

func (t *tool) FuncName() string {
	return t.funcName
}

func (t *tool) Run(r Runner, params json.RawMessage) Result { return t.fn(r, params) }

func (t *tool) Grammar() Grammar { return t.grammar }

// Grammar specifies input format for grammar-based tools.
// Providers that support custom tools can expose these to the model.
type Grammar interface{ isGrammar() }

// TextGrammar accepts arbitrary free-form text.
type TextGrammar struct{}

func (TextGrammar) isGrammar() {}

// LarkGrammar enforces a Lark grammar definition.
type LarkGrammar struct{ Definition string }

func (LarkGrammar) isGrammar() {}

// RegexGrammar enforces a regex grammar definition.
type RegexGrammar struct{ Definition string }

func (RegexGrammar) isGrammar() {}

// JSONGrammar implements Grammar for JSON-schema-based tools so that JSON
// tools follow the same grammar pathway as other grammars while preserving
// the existing public API behavior.
type JSONGrammar interface {
	Grammar
	Schema() *FunctionSchema
	// SkipValidation returns true when the tool should bypass JSON validation
	// (e.g., External tools that accept arbitrary json.RawMessage parameters).
	SkipValidation() bool
}

type jsonGrammarImpl struct {
	name        string
	description string

	// If provided, use this schema directly (External tools).
	preset *FunctionSchema

	// For reflection-based schema generation.
	schemaOnce sync.Once
	schema     *FunctionSchema
	schemaType reflect.Type

	// Whether to skip JSON validation for params.
	skipValidation bool
}

func (*jsonGrammarImpl) isGrammar() {}

func (g *jsonGrammarImpl) Schema() *FunctionSchema {
	if g.preset != nil {
		return g.preset
	}
	g.schemaOnce.Do(func() {
		schema := generateSchema(g.name, g.description, g.schemaType)
		g.schema = &schema
	})
	return g.schema
}

func (g *jsonGrammarImpl) SkipValidation() bool { return g.skipValidation }

// DeclaredJSONGrammar is the JSONGrammar of a tool its caller declared in
// JSON (ExternalDeclared).
type DeclaredJSONGrammar interface {
	JSONGrammar
	// Declaration returns the function declaration Schema was decoded from.
	Declaration() json.RawMessage
}

type declaredJSONGrammar struct {
	JSONGrammar
	declaration json.RawMessage
}

func (g *declaredJSONGrammar) Declaration() json.RawMessage { return g.declaration }

// NewJSONGrammar constructs a JSON grammar that will lazily generate a schema
// using reflection over the provided params type.
func NewJSONGrammar(name, description string, schemaType reflect.Type) JSONGrammar {
	return &jsonGrammarImpl{
		name:           name,
		description:    description,
		schemaType:     schemaType,
		skipValidation: schemaType == jsonRawMessageType,
	}
}

// NewJSONGrammarWithSchema constructs a JSON grammar around an explicit schema.
func NewJSONGrammarWithSchema(schema *FunctionSchema, skipValidation bool) JSONGrammar {
	return &jsonGrammarImpl{
		name:           schema.Name,
		description:    schema.Description,
		preset:         schema,
		skipValidation: skipValidation,
	}
}

// Text returns a Grammar that accepts unconstrained free-form text.
func Text() Grammar { return TextGrammar{} }

// Lark returns a Grammar that enforces a Lark grammar.
func Lark(definition string) Grammar { return LarkGrammar{Definition: strings.TrimSpace(definition)} }

// Regex returns a Grammar that enforces a regex grammar.
func Regex(definition string) Grammar { return RegexGrammar{Definition: strings.TrimSpace(definition)} }

// FuncGrammar registers a grammar-based tool that receives a single string input.
// The input is derived from the provider's streaming arguments. If the raw
// parameters are a JSON string, it's unmarshaled; otherwise, the raw bytes are
// interpreted as a plain string.
func FuncGrammar(grammar Grammar, label, description, funcName string, fn func(r Runner, input string) Result) Tool {
	if grammar == nil {
		panic("FuncGrammar requires a non-nil grammar (use tools.Text(), tools.Lark(), or tools.Regex())")
	}
	t := &tool{
		label:       label,
		description: description,
		funcName:    funcName,
		grammar:     grammar,
	}
	t.fn = func(r Runner, params json.RawMessage) Result {
		// Providers for grammar tools must supply plain text (not JSON-wrapped) input.
		// We pass it through as-is to the tool implementation.
		input := string(params)
		return fn(r, input)
	}
	return t
}
