package openai

import (
	"net/url"

	"github.com/flitsinc/go-llms/llms"
)

// maxImages is the image count limit of the OpenAI API: "Up to 1,500 images
// per request" under "Image input requirements" in
// https://developers.openai.com/api/docs/guides/images-vision (read
// 2026-11-04). OpenAI's per-image limit is a patch count that depends on the
// model and detail level, not a size or dimension, so it is left to the API.
const maxImages = 1500

// ImageLimits returns the image limits of the OpenAI API, which apply to
// every OpenAI model. Other OpenAI-compatible endpoints have their own.
func ImageLimits() llms.ImageLimits {
	return llms.ImageLimits{MaxImages: maxImages}
}

// imageLimits returns the limits a request to endpoint is checked against:
// OpenAI's own for OpenAI's API and none for another endpoint, because
// go-llms cannot know what an OpenAI-compatible endpoint accepts, overridden
// by what the caller set with WithImageLimits.
func imageLimits(endpoint string, overrides llms.ImageLimits) llms.ImageLimits {
	var known llms.ImageLimits
	if u, err := url.Parse(endpoint); err == nil && u.Hostname() == "api.openai.com" {
		known = ImageLimits()
	}
	return known.WithOverrides(overrides)
}

// WithImageLimits overrides the image limits a request is checked against
// before it is sent: each nonzero field replaces OpenAI's limit, or supplies
// one for another endpoint, and fields set by earlier calls stay. A request
// that breaks the result fails with an [*llms.ImageLimitError].
func (m *ChatCompletionsAPI) WithImageLimits(overrides llms.ImageLimits) *ChatCompletionsAPI {
	m.imageLimitOverrides = m.imageLimitOverrides.WithOverrides(overrides)
	return m
}

// WithImageLimits overrides the image limits a request is checked against
// before it is sent: each nonzero field replaces OpenAI's limit, or supplies
// one for another endpoint, and fields set by earlier calls stay. A request
// that breaks the result fails with an [*llms.ImageLimitError].
func (m *ResponsesAPI) WithImageLimits(overrides llms.ImageLimits) *ResponsesAPI {
	m.imageLimitOverrides = m.imageLimitOverrides.WithOverrides(overrides)
	return m
}

// WithImageLimits overrides the image limits a request is checked against
// before it is sent: each nonzero field replaces OpenAI's limit, or supplies
// one for another endpoint, and fields set by earlier calls stay. A request
// that breaks the result fails with an [*llms.ImageLimitError].
func (m *WebSocketResponsesAPI) WithImageLimits(overrides llms.ImageLimits) *WebSocketResponsesAPI {
	m.imageLimitOverrides = m.imageLimitOverrides.WithOverrides(overrides)
	return m
}
