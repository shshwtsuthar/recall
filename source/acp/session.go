// Package acp implements the Source interface for ACP (Agent Client Protocol) agents.
//
// ACP is used by Zed, JetBrains, Neovim, and other IDEs to communicate with
// agents like Claude, Gemini, Codex, and Goose via stdio JSON-RPC 2.0.
package acp

import (
	"encoding/json"
)

// rpcEnvelope is a minimal parse of a JSON-RPC 2.0 message.
// We only decode the fields needed for session tracking.
// The full raw line is never modified — we just peek at the structure.
type rpcEnvelope struct {
	Method string          `json:"method"`
	Result json.RawMessage `json:"result"`
}

// sessionNewResult is the shape of the result from a session/new response.
// ACP session/new responses look like:
//
//	{"jsonrpc":"2.0","id":2,"result":{"sessionId":"abc123",...}}
type sessionNewResult struct {
	SessionID string `json:"sessionId"`
}

// extractSessionID attempts to parse a session ID from an ACP session/new response.
//
// ACP uses JSON-RPC 2.0, so a session/new response has no "method" field
// (it's a result, not a request) and contains a result object with sessionId.
//
// Returns empty string if this is not a session/new response or parsing fails.
// This function never modifies the line — it's read-only inspection.
func extractSessionID(line string) string {
	var env rpcEnvelope
	if err := json.Unmarshal([]byte(line), &env); err != nil {
		return ""
	}

	// session/new responses have no method field (they are results).
	// We identify them by the presence of a result containing sessionId.
	if env.Result == nil {
		return ""
	}

	var result sessionNewResult
	if err := json.Unmarshal(env.Result, &result); err != nil {
		return ""
	}

	return result.SessionID
}
