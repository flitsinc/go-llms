package llms

import (
	"bytes"
	"encoding/base64"
	"errors"
	"image"
	"image/png"
	"strings"
	"testing"

	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"

	"github.com/flitsinc/go-llms/content"
)

// pngDataURI returns a data URI of a w x h PNG.
func pngDataURI(t *testing.T, w, h int) string {
	t.Helper()
	var buf bytes.Buffer
	require.NoError(t, png.Encode(&buf, image.NewGray(image.Rect(0, 0, w, h))))
	return "data:image/png;base64," + base64.StdEncoding.EncodeToString(buf.Bytes())
}

func imageItem(url string) *content.ImageURL { return &content.ImageURL{URL: url} }

// userImages returns a user message holding n copies of url.
func userImages(n int, url string) Message {
	msg := Message{Role: "user"}
	for range n {
		msg.Content = append(msg.Content, imageItem(url))
	}
	return msg
}

func TestCheckImageLimits(t *testing.T) {
	small := pngDataURI(t, 2, 2)
	wide := pngDataURI(t, 3001, 1)
	huge := pngDataURI(t, 8001, 1)
	claude := ImageLimits{
		MaxImages:              100,
		MaxImageBytes:          10 << 20,
		MaxDimension:           8000,
		ManyImagesThreshold:    20,
		ManyImagesMaxDimension: 3000,
	}

	tests := []struct {
		name     string
		limits   ImageLimits
		messages []Message
		// want is nil when the request passes.
		want *ImageLimitError
	}{
		{
			name:     "no limits known",
			limits:   ImageLimits{},
			messages: []Message{userImages(1000, huge)},
		},
		{
			name:     "at the count limit",
			limits:   claude,
			messages: []Message{userImages(100, small)},
		},
		{
			name:   "tool result images count with earlier turns",
			limits: ImageLimits{MaxImages: 2},
			messages: []Message{
				{Role: "user", Content: content.Content{&content.Text{Text: "look"}, imageItem(small)}},
				{Role: "assistant", ToolCalls: []ToolCall{{ID: "c1", Name: "screenshot"}}},
				{Role: "tool", ToolCallID: "c1", Content: content.Content{&content.Text{Text: "shot"}, imageItem(small), imageItem(small)}},
			},
			want: &ImageLimitError{
				Reason: ImageLimitTooMany,
				Limit:  2,
				Count:  3,
				Images: []ImageRef{{Message: 0, Part: 1}, {Message: 2, Part: 1}, {Message: 2, Part: 2}},
			},
		},
		{
			name:   "inline image over the byte limit",
			limits: ImageLimits{MaxImages: 10, MaxImageBytes: 1000},
			messages: []Message{
				{Role: "user", Content: content.Content{
					imageItem(small),
					imageItem("data:image/png;base64," + strings.Repeat("A", 1001)),
					imageItem("https://example.com/" + strings.Repeat("a", 200) + ".png"),
				}},
			},
			want: &ImageLimitError{
				Reason: ImageLimitTooLarge,
				Limit:  1000,
				Count:  3,
				Images: []ImageRef{{Message: 0, Part: 1}},
			},
		},
		{
			name:     "image over the dimension limit",
			limits:   claude,
			messages: []Message{{Role: "user", Content: content.Content{imageItem(small), imageItem(huge)}}},
			want: &ImageLimitError{
				Reason: ImageLimitDimensions,
				Limit:  8000,
				Count:  2,
				Images: []ImageRef{{Message: 0, Part: 1}},
			},
		},
		{
			name:     "wide image at the many-image threshold",
			limits:   claude,
			messages: []Message{userImages(19, small), {Role: "tool", Content: content.Content{imageItem(wide)}}},
		},
		{
			name:   "wide tool result image above the many-image threshold",
			limits: claude,
			messages: []Message{
				userImages(20, small),
				{Role: "tool", Content: content.Content{&content.Text{Text: "shot"}, imageItem(wide)}},
			},
			want: &ImageLimitError{
				Reason: ImageLimitManyImageDimensions,
				Limit:  3000,
				Count:  21,
				Images: []ImageRef{{Message: 1, Part: 1}},
			},
		},
		{
			name:   "images given by URL or in unknown formats are not measured",
			limits: claude,
			messages: []Message{
				userImages(20, small),
				{Role: "user", Content: content.Content{
					imageItem("https://example.com/wide.png"),
					imageItem("data:image/png;base64,bm90IGFuIGltYWdl"),
				}},
			},
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			err := CheckImageLimits("Anthropic", "claude-test", tt.limits, tt.messages)
			if tt.want == nil {
				require.NoError(t, err)
				return
			}
			var limitErr *ImageLimitError
			require.True(t, errors.As(err, &limitErr), "got %v", err)
			tt.want.Provider = "Anthropic"
			tt.want.Model = "claude-test"
			tt.want.Limits = tt.limits
			assert.Equal(t, tt.want, limitErr)
			assert.NotEmpty(t, limitErr.Error())
		})
	}
}
