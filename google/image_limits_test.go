package google

import (
	"testing"

	"github.com/stretchr/testify/assert"

	"github.com/flitsinc/go-llms/llms"
)

func TestImageLimitsFor(t *testing.T) {
	tests := []struct {
		model     string
		vertexAI  bool
		maxImages int
	}{
		{"gemini-3-flash-preview", false, 3600},
		{"gemini-3-pro-image-preview", false, 3600},
		{"gemini-3-flash-preview", true, 3000},
		{"gemini-2.5-pro", true, 3000},
		{"gemini-2.5-flash-image", true, 3},
		{"gemini-3-pro-image-preview", true, 14},
		{"gemini-3.1-flash-image-preview", true, 14},
	}
	for _, tt := range tests {
		assert.Equal(t, tt.maxImages, ImageLimitsFor(tt.model, tt.vertexAI).MaxImages, "%s vertexAI=%v", tt.model, tt.vertexAI)
	}
}

func TestModelImageLimits(t *testing.T) {
	assert.Equal(t, 3000, New("gemini-3-flash-preview").WithVertexAIAccessToken("token", "project", "global").imageLimits().MaxImages)
	assert.Equal(t, 3000, New("gemini-3-flash-preview").WithVertexAIAccessToken("token", "project", "us-east5").imageLimits().MaxImages)
	assert.Equal(t, 3600, New("gemini-3-flash-preview").WithGeminiAPI("key").imageLimits().MaxImages)
	assert.Equal(t, 10, New("gemini-3-flash-preview").WithGeminiAPI("key").WithImageLimits(llms.ImageLimits{MaxImages: 10}).imageLimits().MaxImages)
}
