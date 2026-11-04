package anthropic

import (
	"strings"

	"github.com/flitsinc/go-llms/llms"
)

// Image limits of the Claude API, from "Image limits and costs" in
// https://platform.claude.com/docs/en/build-with-claude/vision (read
// 2026-11-04). Each value was also confirmed on 2026-11-04 by sending one
// image over the limit to Anthropic, Amazon Bedrock and Vertex AI.
const (
	// maxImagesSmallContext is the image count limit of models with a
	// 200k-token context window ("Too much media: 0 document pages + 101
	// images > 100"). It stays 100 for Claude Sonnet 4.5 with the
	// context-1m beta.
	maxImagesSmallContext = 100
	// maxImages is the image count limit of every other model ("too many
	// images and documents: 601 + 0 > 600").
	maxImages = 600
	// maxImageDimension is the largest width or height of any image.
	maxImageDimension = 8000
	// manyImagesThreshold and manyImagesMaxDimension: a request with more
	// than 20 images may not carry an image wider or taller than 3000 px. The
	// docs give 3000 px only as the size that is safe everywhere; the
	// refusal names 3000 px on Anthropic, Bedrock and Vertex AI alike.
	manyImagesThreshold    = 20
	manyImagesMaxDimension = 3000
	// maxImageBytes is the largest base64 image data on the Claude API and on
	// Vertex AI ("image exceeds 10 MB maximum: 10503224 bytes > 10485760
	// bytes"). Amazon Bedrock allows 5 MiB, and gateways may route to it, so
	// the limit is only applied to requests sent to Anthropic or Vertex AI.
	maxImageBytes = 10 << 20
)

// smallContextModels are the Claude models with a 200k-token context window,
// as Anthropic names them after normalizeModelID. Every model since Claude
// Opus 4.6 and Claude Sonnet 4.6 has a 1M-token window, so the list is closed:
// a model missing from it gets the larger limit.
// See https://platform.claude.com/docs/en/about-claude/models/overview.
var smallContextModels = []string{
	"claude-3-",
	"claude-haiku-4-5",
	"claude-opus-4-5",
	"claude-opus-4-1",
	"claude-opus-4-0",
	"claude-opus-4-2025",
	"claude-sonnet-4-5",
	"claude-sonnet-4-0",
	"claude-sonnet-4-2025",
}

// smallContextModelsExact are the 200k-token models whose bare ID would also
// prefix newer models (claude-opus-4 against claude-opus-4-6).
var smallContextModelsExact = []string{
	"claude-opus-4",
	"claude-sonnet-4",
}

// ImageLimitsFor returns the image limits Claude applies to a request for
// model on any platform, except the per-image size limit, which depends on
// the platform (see [Model.WithImageLimits]). model may be in any of the forms
// providers use: "claude-haiku-4-5-20251001", Vertex AI's
// "claude-haiku-4-5@20251001", Bedrock's
// "us.anthropic.claude-haiku-4-5-20251001-v1:0" or OpenRouter's
// "anthropic/claude-haiku-4.5". It returns no limits for a model that is not
// Claude, such as another vendor's model behind an Anthropic-compatible
// endpoint.
func ImageLimitsFor(model string) llms.ImageLimits {
	if !strings.HasPrefix(normalizeModelID(model), "claude-") {
		return llms.ImageLimits{}
	}
	limits := llms.ImageLimits{
		MaxImages:              maxImages,
		MaxDimension:           maxImageDimension,
		ManyImagesThreshold:    manyImagesThreshold,
		ManyImagesMaxDimension: manyImagesMaxDimension,
	}
	if hasSmallContextWindow(model) {
		limits.MaxImages = maxImagesSmallContext
	}
	return limits
}

// WithImageLimits overrides the image limits a request is checked against
// before it is sent: each nonzero field replaces the limit [ImageLimitsFor]
// (and the platform) would apply, for a model or platform go-llms does not
// know, and fields set by earlier calls stay. A request that breaks the result
// fails with an [*llms.ImageLimitError].
func (m *Model) WithImageLimits(overrides llms.ImageLimits) *Model {
	m.imageLimitOverrides = m.imageLimitOverrides.WithOverrides(overrides)
	return m
}

// imageLimits returns the limits Generate checks a request against. The
// per-image size limit is only known for Anthropic's own endpoint and Vertex
// AI; an Anthropic-compatible endpoint may route to Amazon Bedrock, which
// allows less, so it is left to the endpoint.
func (m *Model) imageLimits() llms.ImageLimits {
	limits := ImageLimitsFor(m.model)
	if m.vertexAI || m.endpoint == defaultEndpoint {
		limits.MaxImageBytes = maxImageBytes
	}
	return limits.WithOverrides(m.imageLimitOverrides)
}

func hasSmallContextWindow(model string) bool {
	id := normalizeModelID(model)
	for _, exact := range smallContextModelsExact {
		if id == exact {
			return true
		}
	}
	for _, prefix := range smallContextModels {
		if strings.HasPrefix(id, prefix) {
			return true
		}
	}
	return false
}

// normalizeModelID reduces the platform forms of a model ID to Anthropic's
// own: it drops everything before "claude-" (a gateway or Bedrock prefix),
// the "@" or "-v1:0" version suffixes of Vertex AI and Bedrock, and writes
// OpenRouter's version dots as dashes.
func normalizeModelID(model string) string {
	id := strings.ToLower(model)
	if i := strings.Index(id, "claude-"); i >= 0 {
		id = id[i:]
	}
	if i := strings.IndexAny(id, "@:"); i >= 0 {
		id = id[:i]
	}
	return strings.ReplaceAll(id, ".", "-")
}
