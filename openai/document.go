package openai

import (
	"mime"

	"github.com/flitsinc/go-llms/content"
)

// documentFilename is the name a document part carries. OpenAI refuses a
// file sent as data without one, so a document that names none is called
// "document", with the extension of its type when it has one.
func documentFilename(doc *content.DocumentURL) string {
	if doc.Filename != "" {
		return doc.Filename
	}
	mimeType := doc.MimeType
	if parsed, _, ok := content.ParseDataURI(doc.URL); ok && mimeType == "" {
		mimeType = parsed
	}
	if mimeType == "" {
		mimeType = content.GuessMIMETypeFromURL(doc.URL)
	}
	if extensions, err := mime.ExtensionsByType(mimeType); err == nil && len(extensions) > 0 {
		return "document" + extensions[0]
	}
	return "document"
}
