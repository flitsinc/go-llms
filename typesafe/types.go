package typesafe

// Question is one typed question in a System One request. Type is "noul" for
// a yes/no probability and "choice" for a selection; Instructions is the
// question itself, taken from the schema property's description. Criteria
// names the options of a choice, each mapped to an optional description of
// when to pick it; the provider sends the enum members with no description.
type Question struct {
	Type         string             `json:"type"`
	Instructions string             `json:"instructions"`
	Criteria     map[string]*string `json:"criteria,omitempty"`
}

// Answer is the model's answer to one question. Which fields are populated
// depends on Type: a "noul" answer carries Noul, the probability that the
// question's instructions hold; a "choice" answer carries Choice, plus
// Probabilities over the offered options and Confidence.
type Answer struct {
	Type          string             `json:"type"`
	Noul          *float64           `json:"noul,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Probabilities map[string]float64 `json:"probabilities,omitempty"`
	Confidence    *float64           `json:"confidence,omitempty"`
}

// Usage is the token usage reported by the API. Output tokens are reported
// but not billed.
type Usage struct {
	InputTokens  int `json:"input_tokens"`
	OutputTokens int `json:"output_tokens"`
}

// Response is the full System One response, kept on the stream so callers
// that need the probability distributions behind the JSON answer can read
// them through [Stream.Response].
type Response struct {
	Model   string            `json:"model"`
	Answers map[string]Answer `json:"answers"`
	Usage   Usage             `json:"usage"`
}

type request struct {
	State     any                 `json:"state"`
	Model     string              `json:"model"`
	Questions map[string]Question `json:"questions"`
}

type stateMessage struct {
	Role    string `json:"role"`
	Content any    `json:"content"`
}
