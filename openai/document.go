package openai

import (
	"cmp"
	"mime"

	"github.com/flitsinc/go-llms/content"
)

// documentFilename is the name a document part carries. OpenAI refuses a
// file sent as data without one, so a document that names none is called
// "document" plus an extension of its type: the data URI's type, else the
// declared one, as Gemini reads it. mime.ExtensionsByType sorts what the
// system knows, so application/pdf is always ".pdf" (Go's own table has it),
// while a rarer type can come out with any of its extensions.
func documentFilename(doc *content.DocumentURL) string {
	if doc.Filename != "" {
		return doc.Filename
	}
	mimeType := cmp.Or(content.ExtractMIMETypeFromURIOrURL(doc.URL), doc.MimeType)
	if extensions, err := mime.ExtensionsByType(mimeType); err == nil && len(extensions) > 0 {
		return "document" + extensions[0]
	}
	return "document"
}
