// Package source defines the abstraction for any protocol or log format
// that produces capturable agent interaction messages.
//
// The Source interface enables recall-proxy to support multiple agent
// environments (ACP-based IDEs, Claude CLI logs, VS Code extensions, etc.)
// without duplicating scrubbing and transmission logic.
package source

import (
	"context"
	"time"
)

// Source represents any protocol or log format that produces capturable messages.
// Implementations spawn subprocesses, tail log files, or connect to live streams.
//
// Each source implementation knows how to produce messages but nothing about
// what happens to them afterward (scrubbing, transmission). This separation
// enables the pipeline layer to be completely source-agnostic.
type Source interface {
	// Name returns a human-readable identifier for this source.
	// Examples: "acp", "claude-cli", "vscode"
	// This name is included in transmitted payloads for server-side routing.
	Name() string

	// Run starts the source and emits messages to the output channel.
	// It blocks until the source terminates or ctx is cancelled.
	//
	// The implementation MUST:
	//  - Close the output channel when done (ownership model)
	//  - Respect context cancellation for graceful shutdown
	//  - Return error only for fatal startup failures (e.g., binary not found)
	//
	// Normal operation (source runs and exits cleanly) should return nil.
	Run(ctx context.Context, out chan<- Message) error
}

// Message represents a single captured message from any source.
// It carries both raw content and metadata needed for transmission.
type Message struct {
	// Raw is the original, UNSCRUBBED message content.
	// The pipeline layer applies scrubbing before transmission.
	// Sources should never modify or scrub content — that's not their job.
	Raw string

	// Direction indicates message flow:
	//  - "upstream": IDE/user → Agent
	//  - "downstream": Agent → IDE/user
	//  - "log": Unidirectional log entries (e.g., from file tailing)
	Direction string

	// SessionID groups related messages into a single trajectory.
	// How this is determined is source-specific:
	//  - ACP: Extracted from session/new JSON-RPC response
	//  - Claude CLI: Derived from conversation boundaries in logs
	//  - VS Code: Workspace path or project identifier
	//
	// Empty string is valid for sources that don't support session grouping.
	SessionID string

	// SourceName identifies which source produced this message.
	// Populated by the source's Name() method.
	SourceName string

	// CapturedAt is when the source intercepted this message.
	// Should be set to time.Now().UTC() at capture time.
	CapturedAt time.Time
}
