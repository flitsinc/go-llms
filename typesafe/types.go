package typesafe

// Question is one typed question in a System One request. Instructions and
// criteria accept strings or JSON structure, as the TypeSafe API does.
type Question struct {
	Type         string `json:"type"`
	Instructions any    `json:"instructions"`
	Criteria     any    `json:"criteria,omitempty"`
}

// Answer is the model's answer to one question. Which fields are populated
// depends on Type: a "noul" answer carries Noul; a "choice" answer carries
// Choice, Probabilities, and Confidence; a "score" answer carries Score,
// Legend, Probabilities, and Confidence.
type Answer struct {
	Type          string             `json:"type"`
	Noul          *float64           `json:"noul,omitempty"`
	Choice        string             `json:"choice,omitempty"`
	Score         *float64           `json:"score,omitempty"`
	Legend        map[string]string  `json:"legend,omitempty"`
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
