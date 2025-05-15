package content

import (
	"encoding/json"
	"fmt"
)

type Type string

const (
	TypeText     Type = "text"
	TypeImageURL Type = "imageURL"
	TypeJSON     Type = "json"
	TypeThought  Type = "thought"
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
	Text      string `json:"text,omitempty"`
	Encrypted []byte `json:"encrypted,omitempty"`
	Signature string `json:"signature,omitempty"`
}

func (t *Thought) Type() Type {
	return TypeThought
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
