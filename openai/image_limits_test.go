package openai

import (
	"context"
	"errors"
	"net/http"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
)

func TestImageLimitConfigResolve(t *testing.T) {
	tests := []struct {
		name     string
		config   imageLimitConfig
		endpoint string
		want     llms.ImageLimits
	}{
		{"OpenAI's own API", imageLimitConfig{}, "https://api.openai.com/v1/responses", llms.ImageLimits{MaxImages: 1500}},
		{"OpenAI's WebSocket API", imageLimitConfig{}, "wss://api.openai.com/v1/responses", llms.ImageLimits{MaxImages: 1500}},
		{"another compatible endpoint", imageLimitConfig{}, "https://api.groq.com/openai/v1/chat/completions", llms.ImageLimits{}},
		{"caller count on another endpoint", imageLimitConfig{maxImages: 5}, "https://api.x.ai/v1/responses", llms.ImageLimits{MaxImages: 5}},
		{
			"caller limits replace OpenAI's",
			imageLimitConfig{limits: llms.ImageLimits{MaxImages: 100, MaxDimension: 8000}, limitsSet: true},
			"https://api.openai.com/v1/chat/completions",
			llms.ImageLimits{MaxImages: 100, MaxDimension: 8000},
		},
		{
			"caller count over caller limits",
			imageLimitConfig{limits: llms.ImageLimits{MaxImages: 100, MaxDimension: 8000}, limitsSet: true, maxImages: 7},
			"https://openrouter.ai/api/v1/chat/completions",
			llms.ImageLimits{MaxImages: 7, MaxDimension: 8000},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			assert.Equal(t, tt.want, tt.config.resolve(tt.endpoint))
		})
	}
}

type imageLimitFailingTransport struct{ t *testing.T }

func (f imageLimitFailingTransport) RoundTrip(*http.Request) (*http.Response, error) {
	f.t.Error("request was sent")
	return nil, errors.New("request was sent")
}

func TestGenerateRefusesTooManyImagesBeforeSending(t *testing.T) {
	const tiny = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ/pLvAAAAAElFTkSuQmCC"
	messages := []llms.Message{
		{Role: "user", Content: content.Content{&content.ImageURL{URL: tiny}}},
		{Role: "assistant", ToolCalls: []llms.ToolCall{{ID: "c1", Name: "screenshot", Arguments: []byte("{}")}}},
		{Role: "tool", ToolCallID: "c1", Content: content.Content{&content.ImageURL{URL: tiny}}},
	}
	chat := NewChatCompletionsAPI("key", "gpt-5.5").WithMaxImages(1)
	chat.SetHTTPClient(&http.Client{Transport: imageLimitFailingTransport{t}})
	responses := NewResponsesAPI("key", "gpt-5.5").WithMaxImages(1)
	responses.SetHTTPClient(&http.Client{Transport: imageLimitFailingTransport{t}})
	// No connection is dialled: the check runs before the WebSocket connects.
	websocket := NewWebSocketResponsesAPI("key", "gpt-5.5").WithMaxImages(1)

	for name, provider := range map[string]llms.Provider{"chat completions": chat, "responses": responses, "websocket": websocket} {
		t.Run(name, func(t *testing.T) {
			stream := provider.Generate(context.Background(), nil, messages, nil, nil)
			var limitErr *llms.ImageLimitError
			require.True(t, errors.As(stream.Err(), &limitErr), "got %v", stream.Err())
			assert.Equal(t, llms.ImageLimitTooMany, limitErr.Reason)
			assert.Equal(t, 1, limitErr.Limit)
			assert.Equal(t, 2, limitErr.Count)
			assert.Equal(t, []llms.ImageRef{{Message: 0, Part: 0}, {Message: 2, Part: 0}}, limitErr.Images)
		})
	}
}
