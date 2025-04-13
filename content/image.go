package content

import (
	"encoding/base64"
	"fmt"
	"image"
	_ "image/gif"
	"image/jpeg"
	"image/png"
	"os"
	"path/filepath"
	"strings"

	"golang.org/x/image/draw"
	_ "golang.org/x/image/webp"
)

// ImageToDataURI reads an image from the given path, resizes it if necessary
// based on the quality setting, encodes it as base64, and returns its filename
// and a data URI string.
func ImageToDataURI(path string, highQuality bool) (name, dataURI string, err error) {
	file, err := os.Open(path)
	if err != nil {
		return "", "", fmt.Errorf("failed to open image file: %w", err)
	}
	defer file.Close()

	img, format, err := image.Decode(file)
	if err != nil {
		return "", "", fmt.Errorf("failed to decode image: %w", err)
	}

	// Check image dimensions and resize if necessary.
	var maxDim int
	if highQuality {
		// Common max dimension for high quality in models like GPT-4 Vision
		maxDim = 2048
	} else {
		// Common max dimension for low quality / faster processing
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
		// Use a high-quality scaler
		draw.CatmullRom.Scale(resizedImg, resizedImg.Bounds(), img, bounds, draw.Over, nil)
		img = resizedImg // Replace original with resized image
	}

	// Encode the image data into a base64 string.
	var encodedImage strings.Builder
	encoder := base64.NewEncoder(base64.StdEncoding, &encodedImage)

	var mimeType string
	switch format {
	case "jpeg":
		err = jpeg.Encode(encoder, img, &jpeg.Options{Quality: 90}) // Use good quality for JPEG
		mimeType = "image/jpeg"
	case "png":
		err = png.Encode(encoder, img)
		mimeType = "image/png"
	case "gif":
		// Note: Encoding animated GIFs is complex; this will encode the first frame.
		// Consider using a dedicated GIF library if animation needs preservation.
		err = jpeg.Encode(encoder, img, &jpeg.Options{Quality: 90}) // Encode as JPEG for simplicity
		mimeType = "image/jpeg"                                     // Pretend it's JPEG
	case "webp":
		// Go's standard library doesn't have a webp encoder. Need external library or convert.
		// Encode as PNG as a fallback.
		err = png.Encode(encoder, img)
		mimeType = "image/png"
	default:
		// Fallback: attempt to encode as PNG
		err = png.Encode(encoder, img)
		mimeType = "image/png"
		if err != nil {
			return "", "", fmt.Errorf("unsupported image format %q and failed fallback to PNG: %w", format, err)
		}
	}
	// Close the encoder *after* encoding attempts.
	closeErr := encoder.Close()
	if err != nil {
		return "", "", fmt.Errorf("failed to encode image as %q: %w", mimeType, err)
	}
	if closeErr != nil {
		// This might happen if the writer (strings.Builder) fails
		return "", "", fmt.Errorf("failed to close image encoder: %w", closeErr)
	}

	dataURI = fmt.Sprintf("data:%s;base64,%s", mimeType, encodedImage.String())
	name = filepath.Base(path)
	return name, dataURI, nil
}
