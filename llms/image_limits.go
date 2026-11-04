package llms

import (
	"encoding/base64"
	"fmt"
	"image"
	"strings"

	// Register the decoders DecodeConfig needs to read the dimensions of the
	// formats providers accept (PNG, JPEG, GIF, WebP).
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"

	_ "golang.org/x/image/webp"

	"github.com/flitsinc/go-llms/content"
)

// ImageLimits describes the images a provider accepts in one request. A zero
// field means the provider has no such limit, or that go-llms does not know
// it, and is not checked. Every [content.ImageURL] in the request counts,
// wherever it sits: images in earlier turns and images in tool results count
// the same as images in the newest user message, because providers count
// every image block they receive.
type ImageLimits struct {
	// MaxImages is the most images one request may carry.
	MaxImages int
	// MaxImageBytes is the largest inline (data URI) image the provider
	// accepts, measured as the length of its base64 data.
	MaxImageBytes int
	// MaxDimension is the largest width or height, in pixels, of any image.
	MaxDimension int
	// ManyImagesThreshold and ManyImagesMaxDimension describe a stricter
	// dimension limit that applies to every image once a request carries more
	// than ManyImagesThreshold images.
	ManyImagesThreshold    int
	ManyImagesMaxDimension int
}

// WithOverrides returns l with every nonzero field of overrides in place of
// its own. Providers resolve the limits they check this way: their own table
// for the model, overridden by what the caller set with WithImageLimits, so a
// caller can supply or correct any limit for a model the table does not know.
func (l ImageLimits) WithOverrides(overrides ImageLimits) ImageLimits {
	override := func(limit *int, value int) {
		if value != 0 {
			*limit = value
		}
	}
	override(&l.MaxImages, overrides.MaxImages)
	override(&l.MaxImageBytes, overrides.MaxImageBytes)
	override(&l.MaxDimension, overrides.MaxDimension)
	override(&l.ManyImagesThreshold, overrides.ManyImagesThreshold)
	override(&l.ManyImagesMaxDimension, overrides.ManyImagesMaxDimension)
	return l
}

// ImageLimitReason names the limit a request broke.
type ImageLimitReason string

const (
	// ImageLimitTooMany means the request carries more images than
	// ImageLimits.MaxImages.
	ImageLimitTooMany ImageLimitReason = "too_many_images"
	// ImageLimitTooLarge means an inline image is larger than
	// ImageLimits.MaxImageBytes.
	ImageLimitTooLarge ImageLimitReason = "image_too_large"
	// ImageLimitDimensions means an image is wider or taller than
	// ImageLimits.MaxDimension.
	ImageLimitDimensions ImageLimitReason = "image_dimensions"
	// ImageLimitManyImageDimensions means the request carries more than
	// ImageLimits.ManyImagesThreshold images and an image is wider or taller
	// than ImageLimits.ManyImagesMaxDimension. Sending fewer images or smaller
	// images clears it.
	ImageLimitManyImageDimensions ImageLimitReason = "many_image_dimensions"
)

// ImageRef points at one image in the messages passed to
// [Provider.Generate]: Content[Part] of messages[Message]. A tool result's
// images are parts of its "tool" message, even where the provider nests them
// inside a tool result block on the wire.
type ImageRef struct {
	Message int
	Part    int
}

// ImageLimitError is returned, before anything is sent, when a request breaks
// the provider's image limits. go-llms only reports the problem; it never
// drops or resizes images, so the caller decides which images to remove.
type ImageLimitError struct {
	// Provider is the company serving the request (Provider.Company).
	Provider string
	// Model is the model the request was for.
	Model string
	// Reason names the limit the request broke.
	Reason ImageLimitReason
	// Limit is the value of the broken limit: images for
	// [ImageLimitTooMany], base64 bytes for [ImageLimitTooLarge], pixels for
	// [ImageLimitDimensions] and [ImageLimitManyImageDimensions].
	Limit int
	// Count is the number of images in the request.
	Count int
	// Limits are all the limits the request was checked against, so a
	// caller can tell how many images clear [ImageLimitManyImageDimensions]
	// (Limits.ManyImagesThreshold) as well as [ImageLimitTooMany].
	Limits ImageLimits
	// Images are the offending images, oldest first. For [ImageLimitTooMany]
	// that is every image in the request, since any of them can go; for the
	// other reasons it is each image that is too large.
	Images []ImageRef
}

// MaxImagesThatFit returns how many images a request may carry to clear the
// refusal, and false when sending fewer images cannot clear it because one
// image is too large on its own. For [ImageLimitTooMany] it is the count
// limit; for [ImageLimitManyImageDimensions] it is the many-image threshold
// (or the count limit, if lower), at or under which the stricter dimension
// limit no longer applies.
func (e *ImageLimitError) MaxImagesThatFit() (int, bool) {
	switch e.Reason {
	case ImageLimitTooMany:
		return e.Limits.MaxImages, true
	case ImageLimitManyImageDimensions:
		n := e.Limits.ManyImagesThreshold
		if e.Limits.MaxImages > 0 && e.Limits.MaxImages < n {
			n = e.Limits.MaxImages
		}
		return n, true
	}
	return 0, false
}

func (e *ImageLimitError) Error() string {
	switch e.Reason {
	case ImageLimitTooMany:
		return fmt.Sprintf("%s: request has %d images, %s accepts at most %d", e.Provider, e.Count, e.Model, e.Limit)
	case ImageLimitTooLarge:
		return fmt.Sprintf("%s: %d images exceed %s's limit of %d bytes per image", e.Provider, len(e.Images), e.Model, e.Limit)
	case ImageLimitDimensions:
		return fmt.Sprintf("%s: %d images exceed %s's limit of %d pixels per side", e.Provider, len(e.Images), e.Model, e.Limit)
	case ImageLimitManyImageDimensions:
		return fmt.Sprintf("%s: request has %d images and %d of them exceed %s's limit of %d pixels per side for requests with many images", e.Provider, e.Count, len(e.Images), e.Model, e.Limit)
	}
	return fmt.Sprintf("%s: request breaks %s's image limits (%s)", e.Provider, e.Model, e.Reason)
}

// CheckImageLimits returns an [*ImageLimitError] when messages break limits,
// and nil otherwise. Providers call it at the start of Generate so a request
// the provider would refuse is never sent. Dimensions are read from the
// headers of inline images only; an image given by URL, or one whose format
// cannot be decoded, is not checked against the dimension limits.
func CheckImageLimits(provider, model string, limits ImageLimits, messages []Message) error {
	if limits == (ImageLimits{}) {
		return nil
	}
	var images []ImageRef
	var urls []*content.ImageURL
	for i, msg := range messages {
		for j, item := range msg.Content {
			if img, ok := item.(*content.ImageURL); ok {
				images = append(images, ImageRef{Message: i, Part: j})
				urls = append(urls, img)
			}
		}
	}
	if len(images) == 0 {
		return nil
	}
	fail := func(reason ImageLimitReason, limit int, refs []ImageRef) error {
		return &ImageLimitError{
			Provider: provider,
			Model:    model,
			Reason:   reason,
			Limit:    limit,
			Count:    len(images),
			Limits:   limits,
			Images:   refs,
		}
	}
	if limits.MaxImages > 0 && len(images) > limits.MaxImages {
		return fail(ImageLimitTooMany, limits.MaxImages, images)
	}

	if limits.MaxImageBytes > 0 {
		var tooLarge []ImageRef
		for i, img := range urls {
			if data, ok := inlineImageData(img.URL); ok && len(data) > limits.MaxImageBytes {
				tooLarge = append(tooLarge, images[i])
			}
		}
		if len(tooLarge) > 0 {
			return fail(ImageLimitTooLarge, limits.MaxImageBytes, tooLarge)
		}
	}

	manyImages := limits.ManyImagesThreshold > 0 && limits.ManyImagesMaxDimension > 0 &&
		len(images) > limits.ManyImagesThreshold
	if limits.MaxDimension == 0 && !manyImages {
		return nil
	}
	sides := make([]int, len(urls))
	for i, img := range urls {
		sides[i] = inlineImageLongestSide(img.URL)
	}
	if limits.MaxDimension > 0 {
		if refs := imagesLongerThan(images, sides, limits.MaxDimension); len(refs) > 0 {
			return fail(ImageLimitDimensions, limits.MaxDimension, refs)
		}
	}
	if manyImages {
		if refs := imagesLongerThan(images, sides, limits.ManyImagesMaxDimension); len(refs) > 0 {
			return fail(ImageLimitManyImageDimensions, limits.ManyImagesMaxDimension, refs)
		}
	}
	return nil
}

func imagesLongerThan(images []ImageRef, sides []int, limit int) []ImageRef {
	var refs []ImageRef
	for i, side := range sides {
		if side > limit {
			refs = append(refs, images[i])
		}
	}
	return refs
}

// inlineImageData returns the base64 data of a "data:<mime>;base64,<data>"
// URI, and false for any other URL.
func inlineImageData(url string) (string, bool) {
	rest, ok := strings.CutPrefix(url, "data:")
	if !ok {
		return "", false
	}
	_, data, ok := strings.Cut(rest, ";base64,")
	return data, ok
}

// inlineImageLongestSide returns the larger of an inline image's width and
// height in pixels, read from its header without decoding the pixels, or 0
// when the image is not inline or its format is unknown.
func inlineImageLongestSide(url string) int {
	data, ok := inlineImageData(url)
	if !ok {
		return 0
	}
	cfg, _, err := image.DecodeConfig(base64.NewDecoder(base64.StdEncoding, strings.NewReader(data)))
	if err != nil {
		return 0
	}
	return max(cfg.Width, cfg.Height)
}
