package openai

import (
	"cmp"
	"mime"

	"github.com/flitsinc/go-llms/content"
)

// documentFilename is the name a document part carries. OpenAI refuses a
// file sent as data without one, and reads the file's type from its name, so
// a document that names none is called "document" plus an extension of its
// type. The type is the one Gemini reads too: a data URI's own type, else the
// declared one, else a guess from the URL's path. mime.ExtensionsByType sorts
// what the system knows, so application/pdf is always ".pdf" (Go's own table
// has it), while a rarer type can come out with any of its extensions.
func documentFilename(doc *content.DocumentURL) string {
	if doc.Filename != "" {
		return doc.Filename
	}
	mimeType, _, isData := content.ParseDataURI(doc.URL)
	if !isData {
		mimeType = cmp.Or(doc.MimeType, content.GuessMIMETypeFromURL(doc.URL))
	}
	if extensions, err := mime.ExtensionsByType(mimeType); err == nil && len(extensions) > 0 {
		return "document" + extensions[0]
	}
	return "document"
}
