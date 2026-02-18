// Package pipeline implements source-agnostic message processing.
//
// The pipeline consumes messages from any Source, applies PII scrubbing,
// and transmits sanitized data to the hive mind server. It knows nothing
// about protocols — it just processes the universal Message format.
package pipeline

import (
	"context"

	"github.com/shshwtsuthar/recall/pipes/scrubber"
	"github.com/shshwtsuthar/recall/pipes/transmitter"
	"github.com/shshwtsuthar/recall/source"
)

// Config holds pipeline configuration.
type Config struct {
	// ServerURL is the hive mind ingest endpoint.
	// Example: "https://hivemind.yourdomain.com/ingest"
	ServerURL string

	// EnvSecrets is a map of environment variable name → value.
	// Any occurrence of these values in messages will be scrubbed.
	// Example: {"DATABASE_URL": "postgres://...", "API_KEY": "sk-..."}
	EnvSecrets map[string]string
}

// Run consumes messages from a source, scrubs them, and transmits to the server.
// It blocks until the source completes or ctx is cancelled.
//
// Architecture:
//  1. Creates a buffered message channel (prevents source blocking)
//  2. Spawns source.Run() in a goroutine (produces messages)
//  3. Consumes messages in a loop, applying scrubbing and transmission
//  4. Drains remaining messages after source signals completion
//  5. Returns the source's error status
//
// The pipeline never blocks the source. If transmission is slow, messages
// queue in the channel buffer. Transmitter.Send() is async (fire-and-forget),
// so transmission latency never stalls message processing.
func Run(ctx context.Context, src source.Source, config Config) error {
	// Buffered channel prevents source from blocking if pipeline is busy.
	// 100 messages is generous for typical ACP traffic (1-5 messages/sec).
	messages := make(chan source.Message, 100)

	// Create transmitter with source name (used in server payloads).
	tx := transmitter.New(config.ServerURL, src.Name())

	// Start source in background goroutine.
	// The source owns the channel and will close it when done.
	sourceErr := make(chan error, 1)
	go func() {
		sourceErr <- src.Run(ctx, messages)
	}()

	// Consume messages until source completes or context is cancelled.
	for {
		select {
		case <-ctx.Done():
			// Context cancelled (e.g., SIGINT). Stop immediately.
			return ctx.Err()

		case err := <-sourceErr:
			// Source completed (success or failure).
			// Drain any remaining messages in the channel before exiting.
			for msg := range messages {
				processMessage(msg, tx, config.EnvSecrets)
			}
			return err

		case msg, ok := <-messages:
			if !ok {
				// Channel closed by source — it's done producing.
				// Wait for the source error and return it.
				return <-sourceErr
			}
			processMessage(msg, tx, config.EnvSecrets)
		}
	}
}

// processMessage applies the scrubbing pipeline and transmits the result.
// This function is called for every message, regardless of source type.
func processMessage(msg source.Message, tx *transmitter.Client, envSecrets map[string]string) {
	// Apply the full scrubbing pipeline.
	scrubbed := scrubber.Scrub(msg.Raw)
	if len(envSecrets) > 0 {
		scrubbed = scrubber.ScrubEnvVars(scrubbed, envSecrets)
	}

	// Transmit scrubbed message (async, fire-and-forget).
	// If transmission fails, transmitter logs to stderr but never blocks us.
	tx.Send(msg.Direction, msg.SessionID, scrubbed)

	// Note: We DO NOT forward messages here. That's the source's job.
	// Sources handle forwarding because they know the destination (stdio, file, etc.).
	// The pipeline only processes — it never performs I/O except transmission.
}
