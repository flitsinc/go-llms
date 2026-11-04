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

// imageLimitConfig holds a provider's image limit options. Without them, a
// request to OpenAI's own API is checked against [ImageLimits] and a request
// to another endpoint is not checked, because go-llms cannot know what an
// OpenAI-compatible endpoint accepts.
type imageLimitConfig struct {
	limits    llms.ImageLimits
	limitsSet bool
	maxImages int
}

func (c imageLimitConfig) resolve(endpoint string) llms.ImageLimits {
	limits := c.limits
	if !c.limitsSet && isOpenAIEndpoint(endpoint) {
		limits = ImageLimits()
	}
	if c.maxImages > 0 {
		limits.MaxImages = c.maxImages
	}
	return limits
}

func isOpenAIEndpoint(endpoint string) bool {
	u, err := url.Parse(endpoint)
	return err == nil && u.Hostname() == "api.openai.com"
}

// WithImageLimits sets the image limits a request is checked against before it
// is sent, replacing OpenAI's own for an endpoint that serves other models. A
// request that breaks them fails with an [*llms.ImageLimitError].
func (m *ChatCompletionsAPI) WithImageLimits(limits llms.ImageLimits) *ChatCompletionsAPI {
	m.images.limits = limits
	m.images.limitsSet = true
	return m
}

// WithMaxImages sets the most images a request may carry, keeping the other
// image limits. A request with more fails with an [*llms.ImageLimitError].
func (m *ChatCompletionsAPI) WithMaxImages(n int) *ChatCompletionsAPI {
	m.images.maxImages = n
	return m
}

// WithImageLimits sets the image limits a request is checked against before it
// is sent, replacing OpenAI's own for an endpoint that serves other models. A
// request that breaks them fails with an [*llms.ImageLimitError].
func (m *ResponsesAPI) WithImageLimits(limits llms.ImageLimits) *ResponsesAPI {
	m.images.limits = limits
	m.images.limitsSet = true
	return m
}

// WithMaxImages sets the most images a request may carry, keeping the other
// image limits. A request with more fails with an [*llms.ImageLimitError].
func (m *ResponsesAPI) WithMaxImages(n int) *ResponsesAPI {
	m.images.maxImages = n
	return m
}

// WithImageLimits sets the image limits a request is checked against before it
// is sent, replacing OpenAI's own for an endpoint that serves other models. A
// request that breaks them fails with an [*llms.ImageLimitError].
func (m *WebSocketResponsesAPI) WithImageLimits(limits llms.ImageLimits) *WebSocketResponsesAPI {
	m.images.limits = limits
	m.images.limitsSet = true
	return m
}

// WithMaxImages sets the most images a request may carry, keeping the other
// image limits. A request with more fails with an [*llms.ImageLimitError].
func (m *WebSocketResponsesAPI) WithMaxImages(n int) *WebSocketResponsesAPI {
	m.images.maxImages = n
	return m
}
