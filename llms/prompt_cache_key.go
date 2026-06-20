package llms

import "context"

var promptCacheKeyContextKey = &contextKey{"prompt-cache-key"}

// WithPromptCacheKey returns a copy of ctx carrying the given prompt cache key.
// Providers that support prompt caching (e.g. OpenAI) will read this value
// from the context and include it in the API request.
func WithPromptCacheKey(ctx context.Context, key string) context.Context {
	return context.WithValue(ctx, promptCacheKeyContextKey, key)
}

// GetPromptCacheKey returns the prompt cache key stored in ctx, or "" if none
// is set.
func GetPromptCacheKey(ctx context.Context) string {
	s, _ := ctx.Value(promptCacheKeyContextKey).(string)
	return s
}
