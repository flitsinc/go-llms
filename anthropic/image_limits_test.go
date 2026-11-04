package anthropic

import (
	"context"
	"errors"
	"net/http"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
)

func TestImageLimitsFor(t *testing.T) {
	tests := []struct {
		model     string
		maxImages int
	}{
		{"claude-haiku-4-5-20251001", 100},
		{"claude-haiku-4-5@20251001", 100},
		{"us.anthropic.claude-haiku-4-5-20251001-v1:0", 100},
		{"anthropic/claude-haiku-4.5", 100},
		{"claude-sonnet-4-5", 100},
		{"anthropic/claude-sonnet-4", 100},
		{"claude-opus-4-20250514", 100},
		{"claude-opus-4@20250514", 100},
		{"claude-opus-4-1-20250805", 100},
		{"claude-opus-4-5-20251101", 100},
		{"claude-3-7-sonnet-20250219", 100},
		{"claude-opus-4-6", 600},
		{"anthropic/claude-sonnet-4.6", 600},
		{"claude-opus-4-7", 600},
		{"claude-sonnet-5-5", 600},
		{"anthropic/claude-opus-5.5", 600},
		{"claude-haiku-5-5", 600},
	}
	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			assert.Equal(t, llms.ImageLimits{
				MaxImages:              tt.maxImages,
				MaxDimension:           8000,
				ManyImagesThreshold:    20,
				ManyImagesMaxDimension: 3000,
			}, ImageLimitsFor(tt.model))
		})
	}

	t.Run("other vendors behind an Anthropic-compatible endpoint", func(t *testing.T) {
		assert.Equal(t, llms.ImageLimits{}, ImageLimitsFor("moonshotai/kimi-k2"))
	})
}

// failingTransport fails the test if a request reaches the network.
type failingTransport struct{ t *testing.T }

func (f failingTransport) RoundTrip(*http.Request) (*http.Response, error) {
	f.t.Error("request was sent")
	return nil, errors.New("request was sent")
}

func screenshotTurns(n int, image string) []llms.Message {
	messages := []llms.Message{{Role: "user", Content: content.FromText("Take screenshots.")}}
	for i := range n {
		id := "call_" + strings.Repeat("x", i+1)
		messages = append(messages,
			llms.Message{Role: "assistant", ToolCalls: []llms.ToolCall{{ID: id, Name: "screenshot", Arguments: []byte("{}")}}},
			llms.Message{Role: "tool", ToolCallID: id, Content: content.Content{&content.ImageURL{URL: image}}},
		)
	}
	return messages
}

func TestGenerateRefusesImageLimitsBeforeSending(t *testing.T) {
	const tiny = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
	oversized := "data:image/png;base64," + strings.Repeat("A", maxImageBytes+4)

	tests := []struct {
		name     string
		model    *Model
		messages []llms.Message
		reason   llms.ImageLimitReason
		limit    int
	}{
		{
			name:     "tool result images past the 200k-token model limit",
			model:    New("key", "claude-haiku-4-5"),
			messages: screenshotTurns(101, tiny),
			reason:   llms.ImageLimitTooMany,
			limit:    100,
		},
		{
			name:     "caller-supplied count limit",
			model:    New("key", "claude-opus-4-6").WithImageLimits(llms.ImageLimits{MaxImages: 3}),
			messages: screenshotTurns(4, tiny),
			reason:   llms.ImageLimitTooMany,
			limit:    3,
		},
		{
			name:     "image over the Claude API size limit",
			model:    New("key", "claude-opus-4-6"),
			messages: screenshotTurns(1, oversized),
			reason:   llms.ImageLimitTooLarge,
			limit:    maxImageBytes,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tt.model.SetHTTPClient(&http.Client{Transport: failingTransport{t}})
			stream := tt.model.Generate(context.Background(), nil, tt.messages, nil, nil)
			var limitErr *llms.ImageLimitError
			require.True(t, errors.As(stream.Err(), &limitErr), "got %v", stream.Err())
			assert.Equal(t, tt.reason, limitErr.Reason)
			assert.Equal(t, tt.limit, limitErr.Limit)
			// The first image is the first tool result: message 2, part 0.
			assert.Equal(t, 2, limitErr.Images[0].Message)
			assert.Equal(t, 0, limitErr.Images[0].Part)
			assert.Same(t, tt.messages[2].Content[0], limitErr.Images[0].Image)
		})
	}
}

func TestModelImageLimitsByPlatform(t *testing.T) {
	assert.Equal(t, maxImageBytes, New("key", "claude-opus-4-6").imageLimits().MaxImageBytes)
	assert.Equal(t, maxImageBytes, New("", "claude-opus-4-6").WithVertexAI(nil, "project", "us-east5").imageLimits().MaxImageBytes)
	assert.Zero(t, New("key", "anthropic/claude-opus-4.6").WithEndpoint("https://openrouter.ai/api/v1/messages", "OpenRouter").imageLimits().MaxImageBytes)
	// The platform follows the endpoint the model ends up with, whatever the
	// order of the options.
	assert.Equal(t, maxImageBytes, New("key", "claude-opus-4-6").WithEndpoint("https://gateway.example", "Gateway").
		WithVertexAI(nil, "project", "global").imageLimits().MaxImageBytes)

	limits := New("key", "claude-future-1").WithImageLimits(llms.ImageLimits{MaxImages: 50, ManyImagesMaxDimension: 2000}).imageLimits()
	assert.Equal(t, llms.ImageLimits{
		MaxImages:              50,
		MaxImageBytes:          maxImageBytes,
		MaxDimension:           8000,
		ManyImagesThreshold:    20,
		ManyImagesMaxDimension: 2000,
	}, limits, "a caller overrides the count and pixel limits of a model the table does not know")
}

func TestGenerateLeavesImageSizeToCustomEndpoints(t *testing.T) {
	var sent bool
	m := New("key", "anthropic/claude-opus-4.6").WithEndpoint("https://gateway.example/v1/messages", "Gateway")
	m.SetHTTPClient(&http.Client{Transport: roundTripFunc(func(*http.Request) (*http.Response, error) {
		sent = true
		return nil, errors.New("offline")
	})})
	oversized := "data:image/png;base64," + strings.Repeat("A", maxImageBytes+4)
	stream := m.Generate(context.Background(), nil, screenshotTurns(1, oversized), nil, nil)
	require.Error(t, stream.Err())
	assert.True(t, sent, "the gateway decides the per-image size limit")
}

type roundTripFunc func(*http.Request) (*http.Response, error)

func (f roundTripFunc) RoundTrip(r *http.Request) (*http.Response, error) { return f(r) }
