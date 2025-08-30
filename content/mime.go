package content

import (
    "mime"
    "net/url"
    "path"
    "strings"
)

// GuessMIMETypeFromURL attempts to infer a MIME type from a URL's path
// extension using the local MIME database. Falls back to a small set of
// common image/audio/video extensions when necessary. Returns an empty
// string if no guess can be made.
func GuessMIMETypeFromURL(rawURL string) string {
    u, err := url.Parse(rawURL)
    if err != nil {
        return ""
    }
    ext := strings.ToLower(path.Ext(u.Path))
    if ext == "" {
        return ""
    }
    if t := mime.TypeByExtension(ext); t != "" {
        if i := strings.IndexByte(t, ';'); i >= 0 {
            t = t[:i]
        }
        return t
    }
    switch ext {
    case ".jpg", ".jpeg":
        return "image/jpeg"
    case ".png":
        return "image/png"
    case ".gif":
        return "image/gif"
    case ".webp":
        return "image/webp"
    case ".svg":
        return "image/svg+xml"
    case ".bmp":
        return "image/bmp"
    case ".ico":
        return "image/x-icon"
    case ".tif", ".tiff":
        return "image/tiff"
    case ".heic":
        return "image/heic"
    case ".avif":
        return "image/avif"
    case ".mp4":
        return "video/mp4"
    case ".mov":
        return "video/quicktime"
    case ".m4v":
        return "video/x-m4v"
    case ".webm":
        return "video/webm"
    case ".mp3":
        return "audio/mpeg"
    case ".wav":
        return "audio/wav"
    case ".ogg":
        return "audio/ogg"
    default:
        return ""
    }
}

// ParseDataURI parses a data URI and returns its MIME type and base64 data.
// It returns ok=false if the provided string is not a data URI or does not
// match the expected `data:<mime>;base64,<payload>` shape.
func ParseDataURI(uri string) (mimeType string, base64Data string, ok bool) {
    value, found := strings.CutPrefix(uri, "data:")
    if !found {
        return "", "", false
    }
    mimeType, rest, found := strings.Cut(value, ";base64,")
    if !found {
        return "", "", false
    }
    return mimeType, rest, true
}

// BuildDataURI constructs a data URI from the given MIME type and base64 data.
func BuildDataURI(mimeType, base64Data string) string {
    return "data:" + mimeType + ";base64," + base64Data
}

// ExtractMIMETypeFromURIOrURL tries to obtain a MIME type from either a
// data URI (preferred) or a regular URL path extension as a fallback.
func ExtractMIMETypeFromURIOrURL(s string) string {
    if mt, _, ok := ParseDataURI(s); ok {
        return mt
    }
    return GuessMIMETypeFromURL(s)
}

// IsImageMIME returns true if the provided MIME type looks like an image type.
func IsImageMIME(m string) bool {
    return strings.HasPrefix(strings.ToLower(m), "image/")
}


