// recall-proxy: a transparent interceptor for agent interactions.
//
// It sits between IDEs and agents, scrubbing PII from every message and
// transmitting sanitized trajectories to the hive mind server while passing
// all traffic through unmodified.
//
// Supports multiple source types:
//   - acp: ACP-compatible IDEs (Zed, JetBrains, Neovim) with agents (claude, gemini, codex, goose)
//   - claude-cli: Claude Code CLI log files (future)
//   - vscode: VS Code extension integration (future)
//
// Usage:
//
//	recall-proxy --source acp --agent claude -- --experimental-acp
//	recall-proxy --agent claude -- --experimental-acp  (--source defaults to acp)
//
// The "--" separator marks the start of arguments passed directly to the agent.
//
// Environment variables:
//
//	RECALL_SERVER   The ingest endpoint URL (required)
//	                  e.g. https://recall.yourdomain.com/ingest
//	RECALL_SECRETS  Comma-separated list of env var names whose values
//	                  should be scrubbed from all messages.
//	                  e.g. DATABASE_URL,INTERNAL_API_KEY,GITHUB_TOKEN
package main

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"strings"
	"syscall"

	"github.com/shshwtsuthar/recall/pipeline"
	"github.com/shshwtsuthar/recall/source"
	"github.com/shshwtsuthar/recall/source/acp"
)

func main() {
	cfg, err := parseConfig()
	if err != nil {
		fmt.Fprintf(os.Stderr, "[recall] config error: %v\n", err)
		os.Exit(1)
	}

	// Resolve the values of the declared secret env vars.
	// We look up the actual values at startup so the scrubber can
	// replace them if they appear verbatim in any message.
	envSecrets := resolveEnvSecrets(cfg.secretVarNames)

	// Create source based on source type.
	var src source.Source
	switch cfg.sourceType {
	case "acp":
		src = acp.New(acp.Config{
			AgentArgs: cfg.agentArgs,
		})
	// FUTURE: Additional source types
	// case "claude-cli":
	//     src = claudecli.New(claudecli.Config{LogDir: cfg.logDir})
	// case "vscode":
	//     src = vscode.New(vscode.Config{WebSocketPort: cfg.wsPort})
	default:
		fmt.Fprintf(os.Stderr, "[recall] unknown source type: %s\n", cfg.sourceType)
		os.Exit(1)
	}

	fmt.Fprintf(os.Stderr, "[recall] proxy starting — source: %s | server: %s\n",
		src.Name(), cfg.serverURL)

	// Setup context with signal handling for graceful shutdown.
	// When user presses Ctrl+C (SIGINT) or sends SIGTERM, we cancel the
	// context which propagates to the source (killing subprocess) and
	// pipeline (draining messages and exiting cleanly).
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	sigChan := make(chan os.Signal, 1)
	signal.Notify(sigChan, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		<-sigChan
		fmt.Fprintf(os.Stderr, "[recall] shutting down gracefully...\n")
		cancel()
	}()

	// Run the pipeline with the selected source.
	pipelineConfig := pipeline.Config{
		ServerURL:  cfg.serverURL,
		EnvSecrets: envSecrets,
	}

	if err := pipeline.Run(ctx, src, pipelineConfig); err != nil && err != context.Canceled {
		fmt.Fprintf(os.Stderr, "[recall] proxy exited with error: %v\n", err)
		os.Exit(1)
	}
}

// config holds everything the proxy needs to start.
type config struct {
	sourceType     string   // "acp", "claude-cli", "vscode"
	agentArgs      []string // for ACP source: the agent binary + its arguments
	serverURL      string   // hive mind ingest endpoint
	secretVarNames []string // names of env vars whose values should be scrubbed
}

// parseConfig reads configuration from CLI flags and environment variables.
//
// We deliberately keep configuration minimal and env-var-driven.
// This proxy is spawned by the IDE on demand — it must start instantly
// and without requiring a config file to be in a specific location.
func parseConfig() (config, error) {
	var cfg config
	cfg.sourceType = "acp" // Default for backward compatibility

	// Parse flags manually to avoid pulling in flag package complexity.
	// The structure is:
	//   recall-proxy [--source <type>] [--agent <binary>] [-- <agent-args...>]
	args := os.Args[1:]

	for i := 0; i < len(args); i++ {
		switch args[i] {
		case "--source":
			if i+1 >= len(args) {
				return cfg, fmt.Errorf("--source requires a value")
			}
			i++
			cfg.sourceType = args[i]

		case "--agent":
			if i+1 >= len(args) {
				return cfg, fmt.Errorf("--agent requires a value")
			}
			i++
			// Build agentArgs starting with the binary name.
			cfg.agentArgs = []string{args[i]}

		case "--":
			// Everything after -- is passed to the agent.
			cfg.agentArgs = append(cfg.agentArgs, args[i+1:]...)
			i = len(args) // stop the loop
		}
	}

	// Validate source-specific requirements.
	if cfg.sourceType == "acp" && len(cfg.agentArgs) == 0 {
		return cfg, fmt.Errorf("acp source requires --agent. Usage: recall-proxy --source acp --agent <binary> [-- <args>]")
	}

	// Server URL from environment.
	cfg.serverURL = os.Getenv("RECALL_SERVER")
	if cfg.serverURL == "" {
		return cfg, fmt.Errorf("RECALL_SERVER environment variable is required")
	}

	// Secret var names from environment.
	if raw := os.Getenv("RECALL_SECRETS"); raw != "" {
		for _, name := range strings.Split(raw, ",") {
			name = strings.TrimSpace(name)
			if name != "" {
				cfg.secretVarNames = append(cfg.secretVarNames, name)
			}
		}
	}

	return cfg, nil
}

// resolveEnvSecrets takes a list of environment variable names and returns
// a map of name → current value. Only variables with non-empty values are included.
// The map is passed to the scrubber to replace any literal occurrences of these
// values in messages with typed placeholders.
func resolveEnvSecrets(names []string) map[string]string {
	secrets := make(map[string]string, len(names))
	for _, name := range names {
		if value := os.Getenv(name); value != "" {
			secrets[name] = value
		}
	}
	return secrets
}
