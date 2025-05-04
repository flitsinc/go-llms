package llms

import (
	"context"
	"testing"
	"time"

	"github.com/flitsinc/go-llms/content"
	"github.com/stretchr/testify/assert"
	"github.com/stretchr/testify/require"
)

// TestContextCancellation tests that context cancellation is properly handled during chat.
func TestContextCancellation(t *testing.T) {
	// Arrange: Standard provider and LLM
	mockProv := &mockProvider{}
	llm, _ := setupTestLLM(t, mockProv)

	// Arrange: Context that cancels immediately
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	// Act: Run chat with cancelled context
	_ = runTestChat(ctx, t, llm, "Test message")

	// Assert: Check for cancellation error
	require.Error(t, llm.Err(), "LLM.Err() should return an error")
	assert.ErrorIs(t, llm.Err(), context.Canceled, "Error should be context.Canceled")
}

// TestAlreadyCancelledContext tests behavior when context is cancelled before chat starts.
func TestAlreadyCancelledContext(t *testing.T) {
	// Arrange: Standard provider and LLM
	mockProv := &mockProvider{}
	llm, _ := setupTestLLM(t, mockProv)

	// Arrange: Already cancelled context
	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	// Act: Call ChatUsingMessages with the cancelled context
	updateChan := llm.ChatUsingMessages(ctx, []Message{{Role: "user", Content: content.FromText("test")}})

	// Assert: Channel should be closed immediately
	select {
	case _, ok := <-updateChan:
		assert.False(t, ok, "Channel should be closed")
	case <-time.After(1 * time.Second):
		t.Fatal("Channel was not closed immediately")
	}

	// Assert: LLM error should be set to context.Canceled
	require.Error(t, llm.Err(), "LLM error should be set")
	assert.ErrorIs(t, llm.Err(), context.Canceled, "Error should be context.Canceled")

	// Assert: Provider Generate should NOT have been called
	assert.False(t, mockProv.generateCalled, "Provider Generate should not be called")
}

// TestTurnContextCancellation tests the context cancellation path in the turn method with a fixed implementation
func TestTurnContextCancellation(t *testing.T) {
	// Create a provider that will block until context is cancelled
	cancelProvider := &mockCancellingProvider{}

	// Create an LLM with the cancelling provider
	llm := New(cancelProvider)

	// Create a context that will be cancelled after a short time
	ctx, cancel := context.WithTimeout(context.Background(), 100*time.Millisecond)
	defer cancel()

	// Start chat with the context
	startTime := time.Now()
	for update := range llm.ChatWithContext(ctx, "Test message") {
		// This test doesn't care about the updates.
		_ = update
	}
	duration := time.Since(startTime)

	// Verify the chat was cancelled due to timeout
	assert.Less(t, duration, 1*time.Second, "Chat should be cancelled quickly")
	assert.Error(t, llm.Err(), "Should return an error when context is cancelled")
	assert.ErrorIs(t, llm.Err(), context.DeadlineExceeded, "Error should be context.DeadlineExceeded")
}
