package tools

var (
	// NopRunner is a runner that does nothing extra.
	NopRunner = NewRunner(nil, func(status string) {})
)

type Runner interface {
	Toolbox() *Toolbox
	Report(status string)
}

type runner struct {
	toolbox *Toolbox
	report  func(status string)
}

// NewRunner returns a new Runner. Tools run with this Runner will report status
// updates to the provided function.
func NewRunner(toolbox *Toolbox, report func(status string)) Runner {
	return &runner{toolbox: toolbox, report: report}
}

func (r *runner) Toolbox() *Toolbox {
	return r.toolbox
}

func (r *runner) Report(status string) {
	r.report(status)
}
