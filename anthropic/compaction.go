package anthropic

import (
	"encoding/json"
	"fmt"
	"slices"

	"github.com/flitsinc/go-llms/content"
	"github.com/flitsinc/go-llms/llms"
)

const (
	// compactionBeta is the beta for server-side threshold compaction
	// (context_management edit "compact_20260112"). Requests that replay a
	// compaction block need it too.
	compactionBeta = "compact-2026-01-12"
	// compactionEditType is the context_management edit for threshold compaction.
	compactionEditType = "compact_20260112"
	// minCompactionTriggerInputTokens is the smallest trigger the API accepts.
	minCompactionTriggerInputTokens = 50_000
)

// ContextCompaction configures Anthropic server-side context compaction:
// once a request's input reaches TriggerInputTokens, the API summarizes the
// conversation into a "compaction" block at the start of the response and
// continues from that summary.
// https://platform.claude.com/docs/en/build-with-claude/compaction-threshold
type ContextCompaction struct {
	// TriggerInputTokens is the input size that triggers compaction. The API
	// requires at least 50,000; zero uses the API default (150,000).
	TriggerInputTokens int
	// Instructions replaces the default summarization prompt when non-empty.
	Instructions string
}

// WithContextCompaction enables server-side context compaction. The returned
// compaction block is surfaced as a content.Compaction in the assistant
// message and must be replayed verbatim in place of the history it covers.
func (m *Model) WithContextCompaction(c ContextCompaction) *Model {
	m.contextCompaction = &c
	return m
}

// contextManagement is the request's context_management parameter.
func (c *ContextCompaction) contextManagement() (map[string]any, error) {
	edit := map[string]any{"type": compactionEditType}
	if c.TriggerInputTokens != 0 {
		if c.TriggerInputTokens < minCompactionTriggerInputTokens {
			return nil, fmt.Errorf("compaction trigger %d is below the API minimum of %d input tokens",
				c.TriggerInputTokens, minCompactionTriggerInputTokens)
		}
		edit["trigger"] = map[string]any{"type": "input_tokens", "value": c.TriggerInputTokens}
	}
	if c.Instructions != "" {
		edit["instructions"] = c.Instructions
	}
	return map[string]any{"edits": []any{edit}}, nil
}

// requestBetaFeatures adds the compaction beta to the model's betas when the
// request configures compaction or replays a compaction block, without
// mutating the model's own list.
func (m *Model) requestBetaFeatures(messages []llms.Message) []string {
	if slices.Contains(m.betaFeatures, compactionBeta) {
		return m.betaFeatures
	}
	if m.contextCompaction == nil && !messagesContainCompaction(messages) {
		return m.betaFeatures
	}
	return append(slices.Clone(m.betaFeatures), compactionBeta)
}

func messagesContainCompaction(messages []llms.Message) bool {
	for _, msg := range messages {
		for _, item := range msg.Content {
			if _, ok := item.(*content.Compaction); ok {
				return true
			}
		}
	}
	return false
}

// compactionBlock is the wire form of a "compaction" block. The API rejects
// fields it did not return on these blocks, so it is encoded on its own
// rather than through contentItem's general field set.
type compactionBlock struct {
	// Content is the summary; null when summarization failed.
	Content *string
	// EncryptedContent is set by endpoints that relay an encrypted form of
	// the summary (OpenRouter's Messages API) and must be replayed with it.
	EncryptedContent string
	Signature        string
}

// compactionContentItem converts a stored compaction for replay.
func compactionContentItem(c *content.Compaction) (contentItem, error) {
	if c.Provider != content.CompactionProviderAnthropic {
		return contentItem{}, fmt.Errorf("anthropic: cannot replay %q compaction: %w", c.Provider, content.ErrForeignCompaction)
	}
	summary := c.Text
	return contentItem{
		Type:       "compaction",
		compaction: &compactionBlock{Content: &summary, EncryptedContent: c.Encrypted, Signature: c.Signature},
	}, nil
}

// MarshalJSON encodes compaction blocks with exactly the fields the API
// returned; every other block uses the default encoding.
func (ci contentItem) MarshalJSON() ([]byte, error) {
	if c := ci.compaction; c != nil {
		return json.Marshal(struct {
			Type             string        `json:"type"`
			Content          *string       `json:"content"`
			EncryptedContent string        `json:"encrypted_content,omitempty"`
			Signature        string        `json:"signature,omitempty"`
			CacheControl     *cacheControl `json:"cache_control,omitempty"`
		}{"compaction", c.Content, c.EncryptedContent, c.Signature, ci.CacheControl})
	}
	type plainContentItem contentItem
	return json.Marshal(plainContentItem(ci))
}

// decodeCompactionString decodes a compaction block's string field, which is
// absent or null when the provider has nothing to report.
func decodeCompactionString(raw json.RawMessage) (string, error) {
	if len(raw) == 0 {
		return "", nil
	}
	var value *string
	if err := json.Unmarshal(raw, &value); err != nil {
		return "", err
	}
	if value == nil {
		return "", nil
	}
	return *value, nil
}

// startCompaction begins a streamed compaction block. On-demand compaction
// delivers the whole block here; threshold compaction delivers the summary
// in a compaction_delta instead.
func (s *Stream) startCompaction(block contentBlock) error {
	text, err := decodeCompactionString(block.Content)
	if err != nil {
		return fmt.Errorf("compaction content: %w", err)
	}
	encrypted, err := decodeCompactionString(block.EncryptedContent)
	if err != nil {
		return fmt.Errorf("compaction encrypted_content: %w", err)
	}
	s.pendingCompaction = &content.Compaction{
		Provider:  content.CompactionProviderAnthropic,
		Text:      text,
		Encrypted: encrypted,
		Signature: block.Signature,
	}
	return nil
}

func (s *Stream) appendCompactionDelta(d delta) error {
	if s.pendingCompaction == nil {
		return nil
	}
	text, err := decodeCompactionString(d.Content)
	if err != nil {
		return fmt.Errorf("compaction_delta content: %w", err)
	}
	encrypted, err := decodeCompactionString(d.EncryptedContent)
	if err != nil {
		return fmt.Errorf("compaction_delta encrypted_content: %w", err)
	}
	s.pendingCompaction.Text += text
	s.pendingCompaction.Encrypted += encrypted
	return nil
}

// finishCompaction completes the streamed compaction block and reports
// whether it produced a checkpoint. A block with no summary (content: null)
// means summarization failed: the request was answered from the full
// history, so there is nothing to replay.
func (s *Stream) finishCompaction() bool {
	compaction := s.pendingCompaction
	s.pendingCompaction = nil
	if compaction == nil || (compaction.Text == "" && compaction.Encrypted == "") {
		return false
	}
	s.message.Content = append(s.message.Content, compaction)
	s.lastCompaction = *compaction
	return true
}

// recordIterations applies the per-iteration usage that compaction adds.
// The top-level usage covers only the non-compaction iterations, so the
// compaction iterations are added to bill the request in full, and the final
// iteration is the context the request ended with.
// https://platform.claude.com/docs/en/build-with-claude/compaction-threshold#understanding-usage
func (s *Stream) recordIterations(iterations []usageIteration) {
	var compaction llms.Usage
	nonCompaction := 0
	for _, it := range iterations {
		if it.Type == "compaction" {
			compaction.Add(it.toUsage())
		} else {
			nonCompaction++
		}
	}
	s.compactionUsage = compaction
	if len(iterations) == 0 {
		return
	}
	final := iterations[len(iterations)-1]
	usage := final.toUsage()
	// Iterations may omit the cache buckets. With a single non-compaction
	// iteration the top-level usage covers exactly that iteration, so its
	// buckets stand in; otherwise they are unknown and stay zero.
	if final.Type != "compaction" && nonCompaction == 1 {
		if final.CacheReadInputTokens == nil {
			usage.CachedInputTokens = s.usage.CachedInputTokens
		}
		if final.CacheCreationInputTokens == nil {
			usage.CacheCreationInputTokens = s.usage.CacheCreationInputTokens
		}
	}
	s.finalIterationUsage = &usage
}

func (it usageIteration) toUsage() llms.Usage {
	return llms.Usage{
		CachedInputTokens:        derefInt(it.CacheReadInputTokens),
		CacheCreationInputTokens: derefInt(it.CacheCreationInputTokens),
		InputTokens:              it.InputTokens,
		OutputTokens:             it.OutputTokens,
	}
}

func derefInt(v *int) int {
	if v == nil {
		return 0
	}
	return *v
}
