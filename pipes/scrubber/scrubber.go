// Package scrubber provides a fast, regex-based first pass for removing
// obvious PII from ACP messages before they leave the client machine.
//
// This is intentionally simple and deterministic. It catches the high-confidence,
// structured secrets that should never leave the machine under any circumstance:
// API keys, tokens, credentials, and environment-specific paths.
//
// More sophisticated NLP-based scrubbing (names, addresses, context-dependent PII)
// is handled server-side by the Presidio pipeline. These two passes are complementary.
package scrubber

import (
	"regexp"
	"strings"
)

// rule pairs a compiled regex with the placeholder it replaces matches with.
type rule struct {
	pattern     *regexp.Regexp
	placeholder string
}

// rules is the ordered list of scrubbing patterns applied to every line.
// Order matters: more specific patterns (e.g. provider-specific keys) run
// before generic fallbacks so placeholders are maximally informative.
var rules = []rule{

	// -------------------------------------------------------------------------
	// Provider-specific API keys — highly structured, zero false positives
	// -------------------------------------------------------------------------

	// Anthropic: sk-ant-api03-... (95 base64url chars after prefix)
	{
		regexp.MustCompile(`sk-ant-[a-zA-Z0-9\-_]{20,}`),
		"<ANTHROPIC_API_KEY>",
	},
	// OpenAI: sk-... and sk-proj-...
	{
		regexp.MustCompile(`sk-proj-[a-zA-Z0-9\-_]{20,}`),
		"<OPENAI_PROJECT_KEY>",
	},
	{
		regexp.MustCompile(`\bsk-[a-zA-Z0-9]{32,}\b`),
		"<OPENAI_API_KEY>",
	},
	// Google AI / Gemini: AIza...
	{
		regexp.MustCompile(`AIza[0-9A-Za-z\-_]{35}`),
		"<GOOGLE_API_KEY>",
	},
	// GitHub tokens
	{
		regexp.MustCompile(`ghp_[a-zA-Z0-9]{36}`),
		"<GITHUB_PAT>",
	},
	{
		regexp.MustCompile(`gho_[a-zA-Z0-9]{36}`),
		"<GITHUB_OAUTH_TOKEN>",
	},
	{
		regexp.MustCompile(`ghs_[a-zA-Z0-9]{36}`),
		"<GITHUB_APP_TOKEN>",
	},
	// AWS access key IDs (always start with AKIA or ASIA)
	{
		regexp.MustCompile(`\b(?:AKIA|ASIA)[0-9A-Z]{16}\b`),
		"<AWS_ACCESS_KEY_ID>",
	},
	// AWS secret access keys (40 base64 chars, commonly after "aws_secret")
	// We match on context-word + value to reduce false positives
	{
		regexp.MustCompile(`(?i)aws[_\-]?secret[_\-]?(?:access[_\-]?)?key["'\s:=]+([A-Za-z0-9/+=]{40})`),
		"<AWS_SECRET_ACCESS_KEY>",
	},
	// Stripe keys
	{
		regexp.MustCompile(`\b(?:sk|pk|rk)_(?:live|test)_[a-zA-Z0-9]{24,}\b`),
		"<STRIPE_KEY>",
	},
	// Slack tokens
	{
		regexp.MustCompile(`xox[baprs]-[0-9A-Za-z\-]{10,}`),
		"<SLACK_TOKEN>",
	},
	// npm tokens
	{
		regexp.MustCompile(`npm_[a-zA-Z0-9]{36}`),
		"<NPM_TOKEN>",
	},
	// HuggingFace tokens
	{
		regexp.MustCompile(`hf_[a-zA-Z0-9]{34,}`),
		"<HUGGINGFACE_TOKEN>",
	},

	// -------------------------------------------------------------------------
	// Generic credential patterns (context-word + value)
	// These catch ad-hoc tokens, passwords, secrets in config/env files
	// -------------------------------------------------------------------------

	// Matches: password="abc123", secret: 'xyz', token = "...", api_key="..."
	{
		regexp.MustCompile(`(?i)(?:password|passwd|secret|token|api[_\-]?key|auth[_\-]?key|access[_\-]?key|private[_\-]?key|client[_\-]?secret)["'\s]*[:=]["'\s]*([^\s"',}\]]{8,})`),
		`<REDACTED_CREDENTIAL>`,
	},

	// -------------------------------------------------------------------------
	// Structured token formats
	// -------------------------------------------------------------------------

	// JWT tokens (three base64url segments separated by dots)
	{
		regexp.MustCompile(`eyJ[a-zA-Z0-9\-_]+\.eyJ[a-zA-Z0-9\-_]+\.[a-zA-Z0-9\-_]+`),
		"<JWT_TOKEN>",
	},
	// Bearer tokens in Authorization headers
	{
		regexp.MustCompile(`(?i)bearer\s+[a-zA-Z0-9\-_\.]+`),
		"Bearer <REDACTED_TOKEN>",
	},
	// Basic auth in URLs: https://user:pass@host
	{
		regexp.MustCompile(`[a-zA-Z][a-zA-Z0-9+\-.]*://[^:@\s]+:[^@\s]+@`),
		"<REDACTED_AUTH_URL>@",
	},

	// -------------------------------------------------------------------------
	// Email addresses
	// -------------------------------------------------------------------------
	{
		regexp.MustCompile(`\b[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}\b`),
		"<EMAIL>",
	},

	// -------------------------------------------------------------------------
	// Environment-specific filesystem paths
	// These prevent absolute paths from leaking usernames and machine structure.
	// We replace only the user-specific prefix, preserving the relative path
	// for context, which preserves trajectory usefulness for the hive mind.
	// -------------------------------------------------------------------------

	// macOS/Linux home dirs: /Users/john/project/file.go → $HOME/project/file.go
	{
		regexp.MustCompile(`/(?:Users|home)/[a-zA-Z0-9._\-]+`),
		"$HOME",
	},
	// Windows home dirs: C:\Users\john → %USERPROFILE%
	{
		regexp.MustCompile(`(?i)[A-Za-z]:\\Users\\[a-zA-Z0-9._\-]+`),
		"%USERPROFILE%",
	},

	// -------------------------------------------------------------------------
	// Private IP ranges & localhost with ports
	// These can fingerprint internal network topology.
	// We preserve the port since it's often semantically meaningful (e.g. :5432
	// signals postgres, :6379 signals redis) — we just strip the IP.
	// -------------------------------------------------------------------------
	{
		regexp.MustCompile(`\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}|192\.168\.\d{1,3}\.\d{1,3})\b`),
		"<PRIVATE_IP>",
	},
}

// Scrub applies all PII scrubbing rules to a single line of ACP JSON text.
// It returns the sanitized line. The original line is never modified.
// This function is safe to call from multiple goroutines concurrently.
func Scrub(line string) string {
	for _, r := range rules {
		line = r.pattern.ReplaceAllString(line, r.placeholder)
	}
	return line
}

// ScrubEnvVars scrubs values of any known secret environment variable names
// found anywhere in the line. This is a second pass specifically for patterns
// like `os.Getenv("MY_SECRET")` results appearing in trajectories.
//
// envSecrets is the set of env var names the user has declared as sensitive.
// In practice this is populated from a config file at proxy startup.
func ScrubEnvVars(line string, envSecrets map[string]string) string {
	for name, value := range envSecrets {
		if value == "" || len(value) < 4 {
			// Don't scrub empty or trivially short values — too many false positives.
			continue
		}
		line = strings.ReplaceAll(line, value, "<ENV:"+name+">")
	}
	return line
}
