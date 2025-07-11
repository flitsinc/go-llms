package content

import (
	"encoding/json"
	"fmt"
)

type Type string

const (
	TypeText      Type = "text"
	TypeImageURL  Type = "image_url"
	TypeJSON      Type = "json"
	TypeThought   Type = "thought"
	TypeCacheHint Type = "cache_hint"
)

type Item interface {
	Type() Type
}

type Text struct {
	Text string `json:"text"`
}

func (t *Text) Type() Type {
	return TypeText
}

type ImageURL struct {
	URL string `json:"image_url"`
}

func (iu *ImageURL) Type() Type {
	return TypeImageURL
}

type JSON struct {
	Data json.RawMessage `json:"data"`
}

func (j *JSON) Type() Type {
	return TypeJSON
}

type Thought struct {
	ID        string `json:"id,omitempty"`
	Text      string `json:"text,omitempty"`
	Encrypted []byte `json:"encrypted,omitempty"`
	Signature string `json:"signature,omitempty"`
	// Summary is true if the thought is a complete summary of the thinking
	// session, as opposed to the actual thinking stream.
	Summary bool `json:"summary"`
}

func (t *Thought) Type() Type {
	return TypeThought
}

type CacheHint struct {
	// Duration: "short", "long"
	Duration string `json:"duration,omitempty"`
}

func (c *CacheHint) Type() Type {
	return TypeCacheHint
}

type Content []Item

// FromAny marshals the given value to JSON and returns a new JSON content item
// with the marshalled JSON data.
func FromAny(value any) (Content, error) {
	data, err := json.Marshal(value)
	if err != nil {
		return nil, err
	}
	return FromRawJSON(data), nil
}

// FromRawJSON returns a new JSON content item with the given raw JSON data.
func FromRawJSON(data json.RawMessage) Content {
	return Content{
		&JSON{Data: data},
	}
}

// FromText returns a new content item with the given text.
func FromText(text string) Content {
	return Content{
		&Text{Text: text},
	}
}

// Textf returns a new content item with the provided formatted text.
func Textf(format string, args ...any) Content {
	return FromText(fmt.Sprintf(format, args...))
}

// FromTextAndImage returns a new content item with the given text and image URL.
func FromTextAndImage(text, imageURL string) Content {
	return Content{
		&Text{Text: text},
		&ImageURL{URL: imageURL},
	}
}

// AddImage adds an image URL to the content.
func (c *Content) AddImage(imageURL string) {
	*c = append(*c, &ImageURL{URL: imageURL})
}

// Append adds the text to the last content item if it's a text item, otherwise
// it adds a new text item to the end of the list.
func (c *Content) Append(text string) {
	if l := len(*c); l > 0 {
		if tc, ok := (*c)[l-1].(*Text); ok {
			tc.Text += text
			return
		}
	}
	*c = append(*c, &Text{Text: text})
}

// AppendThought adds the given text to the last content item if it's a thought,
// otherwise it adds a new thought item to the end of the list.
func (c *Content) AppendThought(text string) {
	if l := len(*c); l > 0 {
		if tc, ok := (*c)[l-1].(*Thought); ok && len(tc.Encrypted) == 0 {
			tc.Text += text
			return
		}
	}
	*c = append(*c, &Thought{Text: text})
}

// AppendThoughtWithID finds an existing thought with the given ID and appends text to it,
// or creates a new thought with the given ID if none exists. Returns the thought that was
// updated or created. This is useful for streaming APIs that provide thought IDs upfront.
func (c *Content) AppendThoughtWithID(id, text string, summary bool) *Thought {
	// Look for an existing thought with this ID
	for i := len(*c) - 1; i >= 0; i-- {
		if thought, ok := (*c)[i].(*Thought); ok && thought.ID == id {
			thought.Text += text
			return thought
		}
	}

	// No existing thought with this ID, create a new one
	thought := &Thought{
		ID:      id,
		Text:    text,
		Summary: summary,
	}
	*c = append(*c, thought)
	return thought
}

// SetThoughtSummary will replace the last content item if it's a thought,
// otherwise it adds a new thought item to the end of the list.
func (c *Content) SetThoughtSummary(text, signature string) {
	if l := len(*c); l > 0 {
		if tc, ok := (*c)[l-1].(*Thought); ok && len(tc.Encrypted) == 0 {
			tc.Text = text
			tc.Signature = signature
			tc.Summary = true
			return
		}
	}
	*c = append(*c, &Thought{Text: text, Signature: signature, Summary: true})
}

// SetThoughtSignature sets the signature for the last thought item. Panics if
// there is no thought item.
func (c *Content) SetThoughtSignature(signature string) {
	if len(*c) == 0 {
		panic("tried to sign thought before any content")
	}
	lastThought, ok := (*c)[len(*c)-1].(*Thought)
	if !ok {
		panic("tried to sign thought before any thinking content")
	}
	lastThought.Signature = signature
}

// MarshalJSON implements the json.Marshaler interface for Content.
func (c Content) MarshalJSON() ([]byte, error) {
	items := make([]map[string]any, len(c))
	for i, item := range c {
		// First marshal the concrete item
		itemData, err := json.Marshal(item)
		if err != nil {
			return nil, fmt.Errorf("failed to marshal item: %w", err)
		}

		// Unmarshal into a map to work with it
		var itemMap map[string]any
		if err := json.Unmarshal(itemData, &itemMap); err != nil {
			return nil, fmt.Errorf("failed to process item: %w", err)
		}

		// Add the type field
		itemMap["type"] = item.Type()
		items[i] = itemMap
	}

	return json.Marshal(items)
}

// UnmarshalJSON implements the json.Unmarshaler interface for Content.
func (c *Content) UnmarshalJSON(data []byte) error {
	// Unmarshal as an array of items with type info
	var items []json.RawMessage
	if err := json.Unmarshal(data, &items); err != nil {
		return err
	}

	result := make(Content, 0, len(items))
	for _, itemData := range items {
		// Extract just the type field first
		var typeContainer struct {
			Type Type `json:"type"`
		}
		if err := json.Unmarshal(itemData, &typeContainer); err != nil {
			return fmt.Errorf("failed to extract item type: %w", err)
		}

		// Create and unmarshal the appropriate concrete type
		var item Item
		switch typeContainer.Type {
		case TypeText:
			item = &Text{}
		case TypeImageURL:
			item = &ImageURL{}
		case TypeJSON:
			item = &JSON{}
		case TypeThought:
			item = &Thought{}
		case TypeCacheHint:
			item = &CacheHint{}
		default:
			return fmt.Errorf("unknown content item type: %q", typeContainer.Type)
		}

		// Unmarshal the full data into the concrete type
		if err := json.Unmarshal(itemData, item); err != nil {
			return fmt.Errorf("failed to unmarshal %q item: %w", typeContainer.Type, err)
		}

		result = append(result, item)
	}

	*c = result
	return nil
}

// AsString returns the text content and true if the Content contains exactly
// one Text item. Otherwise, it returns an empty string and false.
func (c Content) AsString() (string, bool) {
	if len(c) == 1 {
		if textItem, ok := c[0].(*Text); ok {
			return textItem.Text, true
		}
	}
	return "", false
}
