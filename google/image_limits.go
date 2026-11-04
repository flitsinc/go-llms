package google

import (
	"net/url"
	"strings"

	"github.com/flitsinc/go-llms/llms"
)

const (
	// maxImagesGeminiAPI is the image count limit of the Gemini Developer
	// API: "Gemini models support a maximum of 3,600 image files per request"
	// in https://ai.google.dev/gemini-api/docs/image-understanding (read
	// 2026-11-04).
	maxImagesGeminiAPI = 3600
	// maxImagesVertexAI is the image count limit of Gemini chat models on
	// Vertex AI ("Maximum images per prompt: 3,000" in
	// https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/image-understanding,
	// read 2026-11-04), confirmed on 2026-11-04 by the refusal "Unable to
	// submit request because it has 3001 image files but the model only
	// supports up to 3000."
	maxImagesVertexAI = 3000
)

// vertexAIImageModelLimits are the Vertex AI models that accept fewer images
// than maxImagesVertexAI, by model ID prefix, from the same page.
var vertexAIImageModelLimits = []struct {
	prefix    string
	maxImages int
}{
	{"gemini-2.5-flash-image", 3},
	{"gemini-3-pro-image", 14},
	{"gemini-3.1-flash-image", 14},
	{"gemini-3.1-flash-lite-image", 14},
}

// ImageLimitsFor returns the image limits of model on Vertex AI (vertexAI
// true) or the Gemini Developer API. Neither documents a per-image size or
// dimension limit that a request can break: inline data counts toward the
// request size, which is left to the API.
func ImageLimitsFor(model string, vertexAI bool) llms.ImageLimits {
	if !vertexAI {
		return llms.ImageLimits{MaxImages: maxImagesGeminiAPI}
	}
	for _, m := range vertexAIImageModelLimits {
		if strings.HasPrefix(model, m.prefix) {
			return llms.ImageLimits{MaxImages: m.maxImages}
		}
	}
	return llms.ImageLimits{MaxImages: maxImagesVertexAI}
}

// WithImageLimits overrides the image limits a request is checked against
// before it is sent: each nonzero field replaces the limit [ImageLimitsFor]
// would apply, for a model go-llms does not know, and fields set by earlier
// calls stay. A request that breaks the result fails with an [*llms.ImageLimitError].
func (m *Model) WithImageLimits(overrides llms.ImageLimits) *Model {
	m.imageLimitOverrides = m.imageLimitOverrides.WithOverrides(overrides)
	return m
}

// imageLimits returns the limits Generate checks a request against.
func (m *Model) imageLimits() llms.ImageLimits {
	return ImageLimitsFor(m.model, isVertexAIEndpoint(m.endpoint)).WithOverrides(m.imageLimitOverrides)
}

func isVertexAIEndpoint(endpoint string) bool {
	u, err := url.Parse(endpoint)
	return err == nil && strings.HasSuffix(u.Hostname(), "aiplatform.googleapis.com")
}
