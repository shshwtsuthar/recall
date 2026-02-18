// Package transmitter handles sending scrubbed messages to the main server.
//
// Design constraints:
//   - Transmission must NEVER stall the message pipeline. If the server is slow,
//     unreachable, or returns an error, the user's session is unaffected.
//   - Each call to Send() is non-blocking. It spawns a goroutine internally.
//   - Failed transmissions are logged to stderr (visible in IDE dev consoles)
//     but are otherwise silently dropped.
package transmitter

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"
)

// Payload is the JSON body sent to the main server for each message.
type Payload struct {
	// Direction indicates message flow.
	// "upstream" = IDE/User to Agent (user prompts, context)
	// "downstream" = Agent to IDE/User (responses, tool calls, thoughts)
	// "log" = Unidirectional log entries (e.g., from file tailing)
	Direction string `json:"direction"`

	// Raw is the scrubbed message content as it was captured exactly.
	Raw string `json:"raw"`

	// SessionID groups all messages from one session into a trajectory.
	// Source-specific: ACP derives from session/new, others use different strategies.
	SessionID string `json:"session_id"`

	// SourceName identifies which source type produced this message.
	// Examples: "acp", "claude-cli", "vscode"
	// Useful for the adapter layer to know which protocol parser to use.
	SourceName string `json:"source_name"`

	// CapturedAt is the RFC3339 timestamp when this message was intercepted.
	CapturedAt string `json:"captured_at"`
}

// Client is a configured transmitter. Create one at startup and reuse it.
// It is safe to call Send() from multiple goroutines concurrently.
type Client struct {
	serverURL  string
	sourceName string
	httpClient *http.Client
}

// New creates a transmitter Client.
//
//   - serverURL: the full URL of your hive mind ingest endpoint,
//     e.g. "https://hivemind.yourdomain.com/ingest"
//   - sourceName: the name of the source producing messages, e.g. "acp", "claude-cli"
func New(serverURL, sourceName string) *Client {
	return &Client{
		serverURL:  serverURL,
		sourceName: sourceName,
		httpClient: &http.Client{
			// Hard timeout: if the server doesn't respond in 5 seconds, drop it.
			// The pipeline cannot wait longer than this in the worst case where Send
			// is called synchronously — but see Send() below, it's async.
			Timeout: 5 * time.Second,
		},
	}
}

// Send queues a scrubbed message for transmission to the server.
// It returns immediately — transmission happens in a background goroutine.
// The message pipeline is never blocked by network latency or server errors.
func (c *Client) Send(direction, sessionID, raw string) {
	payload := Payload{
		Direction:  direction,
		Raw:        raw,
		SessionID:  sessionID,
		SourceName: c.sourceName,
		CapturedAt: time.Now().UTC().Format(time.RFC3339Nano),
	}

	// Fire and forget. The goroutine owns the payload; no shared state.
	go func() {
		if err := c.send(payload); err != nil {
			// Log to stderr. This surfaces in IDE dev consoles.
			// We never write to stdout — that may be reserved for forwarding.
			fmt.Fprintf(os.Stderr, "[recall/transmit] error: %v\n", err)
		}
	}()
}

// send performs the actual HTTP POST. Called inside a goroutine by Send().
func (c *Client) send(payload Payload) error {
	body, err := json.Marshal(payload)
	if err != nil {
		return fmt.Errorf("marshal payload: %w", err)
	}

	resp, err := c.httpClient.Post(c.serverURL, "application/json", bytes.NewReader(body))
	if err != nil {
		return fmt.Errorf("http post: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("server returned %d", resp.StatusCode)
	}

	return nil
}
