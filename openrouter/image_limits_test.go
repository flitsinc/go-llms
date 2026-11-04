package openrouter

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/flitsinc/go-llms/llms"
)

func TestImageLimitsFor(t *testing.T) {
	claude := func(maxImages int) llms.ImageLimits {
		return llms.ImageLimits{MaxImages: maxImages, MaxDimension: 8000, ManyImagesThreshold: 20, ManyImagesMaxDimension: 3000}
	}
	tests := []struct {
		model string
		want  llms.ImageLimits
	}{
		{"anthropic/claude-haiku-4.5", claude(100)},
		{"anthropic/claude-opus-4.6", claude(600)},
		{"openai/gpt-5.5", llms.ImageLimits{MaxImages: 1500}},
		{"google/gemini-3-flash-preview", llms.ImageLimits{MaxImages: 3000}},
		{"google/gemini-3-pro-image-preview", llms.ImageLimits{MaxImages: 14}},
		{"moonshotai/kimi-k2", llms.ImageLimits{}},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.want, ImageLimitsFor(tt.model), tt.model)
	}
}
