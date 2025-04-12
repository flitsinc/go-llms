package tools

import (
	"encoding/base64"
	"encoding/json"
	"fmt"
	"image"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/image/draw"
)

type Result interface {
	// Label returns a short single line description of the entire tool run.
	Label() string
	// JSON returns the JSON representation of the result.
	JSON() json.RawMessage
	// Error returns the error that occurred during the tool run, if any.
	Error() error
	// Images returns a slice of base64 encoded images.
	Images() []Image
}

type result struct {
	label   string
	content json.RawMessage
	err     error
	images  []Image
}

func (r *result) Label() string {
	return r.label
}

func (r *result) JSON() json.RawMessage {
	return r.content
}

func (r *result) Error() error {
	return r.err
}

func (r *result) Images() []Image {
	return r.images
}

func Error(err error) Result {
	return ErrorWithLabel("", err)
}

func ErrorWithLabel(label string, err error) Result {
	if err == nil {
		panic("tools: cannot create error result with nil error")
	}
	content := json.RawMessage(fmt.Sprintf("{\"error\": %q}", err.Error()))
	if label == "" {
		label = fmt.Sprintf("Error: %s", err)
	}
	return &result{label, content, err, nil}
}

func Success(content any) Result {
	var label string
	if stringer, ok := content.(fmt.Stringer); ok {
		label = stringer.String()
		if len(label) > 50 {
			label = label[:50]
		}
	}
	return SuccessWithLabel(label, content)
}

func SuccessWithLabel(label string, content any) Result {
	jsonContent, err := json.Marshal(content)
	if err != nil {
		return ErrorWithLabel(label, err)
	}
	if label == "" {
		label = "success"
	}
	return &result{label, jsonContent, nil, nil}
}

type ResultBuilder struct {
	images []Image
}

type Image struct {
	Name, URL string
}

func (b *ResultBuilder) AddImage(path string, highQuality bool) error {
	file, err := os.Open(path)
	if err != nil {
		return err
	}
	defer file.Close()

	img, format, err := image.Decode(file)
	if err != nil {
		return err
	}

	// Check image dimensions and resize if necessary.
	var maxDim int
	if highQuality {
		maxDim = 2048
	} else {
		maxDim = 512
	}

	bounds := img.Bounds()
	width, height := bounds.Dx(), bounds.Dy()
	if width > maxDim || height > maxDim {
		var newWidth, newHeight int
		if width > height {
			newWidth = maxDim
			newHeight = (height * maxDim) / width
		} else {
			newHeight = maxDim
			newWidth = (width * maxDim) / height
		}

		resizedImg := image.NewRGBA(image.Rect(0, 0, newWidth, newHeight))
		draw.CatmullRom.Scale(resizedImg, resizedImg.Bounds(), img, bounds, draw.Over, nil)
		img = resizedImg
	}

	// Turn the image data into a base64 string.
	var encodedImage strings.Builder
	encoder := base64.NewEncoder(base64.StdEncoding, &encodedImage)
	defer encoder.Close()

	var mimeType string
	switch format {
	case "jpeg":
		err = jpeg.Encode(encoder, img, nil)
		mimeType = "image/jpeg"
	case "png":
		err = png.Encode(encoder, img)
		mimeType = "image/png"
	default:
		return fmt.Errorf("unsupported image format: %s", format)
	}
	if err != nil {
		return err
	}

	dataURI := fmt.Sprintf("data:%s;base64,%s", mimeType, encodedImage.String())
	return b.AddImageURL(filepath.Base(path), dataURI)
}

func (b *ResultBuilder) AddImageURL(name, dataURI string) error {
	image := Image{
		Name: name,
		URL:  dataURI,
	}
	b.images = append(b.images, image)
	return nil
}

func (b *ResultBuilder) Error(err error) Result {
	return b.ErrorWithLabel("", err)
}

func (b *ResultBuilder) ErrorWithLabel(label string, err error) Result {
	if err == nil {
		panic("tools: cannot create error result with nil error in ResultBuilder")
	}
	content := json.RawMessage(fmt.Sprintf("{\"error\": %q}", err.Error()))
	if label == "" {
		label = fmt.Sprintf("Error: %s", err)
	}
	return &result{label, content, err, b.images}
}

func (b *ResultBuilder) Success(content any) Result {
	var label string
	if stringer, ok := content.(fmt.Stringer); ok {
		label = stringer.String()
		if len(label) > 50 {
			label = label[:50]
		}
	}
	return b.SuccessWithLabel(label, content)
}
func (b *ResultBuilder) SuccessWithLabel(label string, content any) Result {
	jsonContent, err := json.Marshal(content)
	if err != nil {
		return b.ErrorWithLabel(label, err)
	}
	if label == "" {
		label = "Success"
	}
	return &result{label, jsonContent, nil, b.images}
}
