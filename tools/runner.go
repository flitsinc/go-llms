package tools

import (
	"context"
)

var (
	// NopRunner is a runner that does nothing extra.
	NopRunner = NewRunner(context.Background(), nil, func(status string) {})
)

type Runner interface {
	Context() context.Context
	Toolbox() *Toolbox
	Report(status string)
}

type runner struct {
	ctx     context.Context
	toolbox *Toolbox
	report  func(status string)
}

// NewRunner returns a new Runner. Tools run with this Runner will report status
// updates to the provided function.
func NewRunner(ctx context.Context, toolbox *Toolbox, report func(status string)) Runner {
	return &runner{ctx: ctx, toolbox: toolbox, report: report}
}

func (r *runner) Context() context.Context {
	return r.ctx
}

func (r *runner) Toolbox() *Toolbox {
	return r.toolbox
}

func (r *runner) Report(status string) {
	r.report(status)
}
