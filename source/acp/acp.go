package acp

import (
	"bufio"
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"sync"
	"time"

	"github.com/shshwtsuthar/recall/source"
)

// Config holds ACP-specific configuration.
type Config struct {
	// AgentArgs is the agent binary and its arguments.
	// Example: ["claude", "--experimental-acp"]
	AgentArgs []string
}

// Source implements the source.Source interface for ACP agents.
// It spawns the real agent as a subprocess, wires bidirectional stdio pipes,
// and emits messages while forwarding original content transparently.
type Source struct {
	config Config
}

// New creates an ACP source with the given configuration.
func New(config Config) *Source {
	return &Source{config: config}
}

// Name returns the identifier for this source type.
func (s *Source) Name() string {
	return "acp"
}

// Run spawns the ACP agent subprocess and intercepts bidirectional stdio traffic.
//
// Architecture:
//  1. Spawns agent as subprocess with exec.CommandContext (respects ctx cancellation)
//  2. Wires stdin/stdout pipes (stderr passes through to os.Stderr)
//  3. Launches two goroutines:
//     - Upstream: os.Stdin → emit Message → agent stdin
//     - Downstream: agent stdout → extract sessionID → emit Message → os.Stdout
//  4. Waits for both goroutines and subprocess to complete
//  5. Closes the output channel (ownership model)
//
// The IDE and agent see unmodified ACP traffic — they are completely unaware
// of the proxy's presence. We just observe and emit messages for the pipeline.
func (s *Source) Run(ctx context.Context, out chan<- source.Message) error {
	if len(s.config.AgentArgs) == 0 {
		return fmt.Errorf("no agent command specified")
	}

	agentBinary := s.config.AgentArgs[0]
	agentCmdArgs := s.config.AgentArgs[1:]

	// Spawn the real agent as a subprocess with context for cancellation.
	cmd := exec.CommandContext(ctx, agentBinary, agentCmdArgs...)

	// Wire up the agent's stdin and stdout.
	// cmd.Stderr is passed through directly — agent error output goes straight
	// to our stderr, which is visible in the IDE's dev console.
	agentStdin, err := cmd.StdinPipe()
	if err != nil {
		return fmt.Errorf("create agent stdin pipe: %w", err)
	}
	agentStdout, err := cmd.StdoutPipe()
	if err != nil {
		return fmt.Errorf("create agent stdout pipe: %w", err)
	}
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		return fmt.Errorf("start agent %q: %w", agentBinary, err)
	}

	// Session tracking: sessionID groups all messages in one conversation.
	// It is updated when we see a session/new response from the agent.
	// Protected by mutex since both goroutines may read it.
	var (
		sessionID string
		sessionMu sync.RWMutex
	)

	getSessionID := func() string {
		sessionMu.RLock()
		defer sessionMu.RUnlock()
		return sessionID
	}

	setSessionID := func(id string) {
		sessionMu.Lock()
		defer sessionMu.Unlock()
		sessionID = id
	}

	// done is closed when either pipe goroutine finishes, signaling the other
	// to stop. This prevents goroutine leaks if one side closes early.
	done := make(chan struct{})
	closeOnce := sync.Once{}
	signalDone := func() { closeOnce.Do(func() { close(done) }) }

	var wg sync.WaitGroup
	wg.Add(2)

	// -------------------------------------------------------------------------
	// Goroutine A: UPSTREAM — IDE → proxy → agent
	// Reads from os.Stdin (written by IDE), emits messages, forwards to agent.
	// -------------------------------------------------------------------------
	go func() {
		defer wg.Done()
		defer signalDone()
		defer agentStdin.Close()

		scanner := newScanner(os.Stdin)
		for scanner.Scan() {
			select {
			case <-done:
				return
			case <-ctx.Done():
				return
			default:
			}

			line := scanner.Text()

			// Emit message for pipeline processing.
			out <- source.Message{
				Raw:        line,
				Direction:  "upstream",
				SessionID:  getSessionID(),
				SourceName: s.Name(),
				CapturedAt: time.Now().UTC(),
			}

			// Forward ORIGINAL (unmodified) line to agent.
			// The agent must see real ACP messages.
			fmt.Fprintln(agentStdin, line)
		}

		if err := scanner.Err(); err != nil && err != io.EOF {
			fmt.Fprintf(os.Stderr, "[recall/acp] upstream read error: %v\n", err)
		}
	}()

	// -------------------------------------------------------------------------
	// Goroutine B: DOWNSTREAM — agent → proxy → IDE
	// Reads from agent stdout, extracts session ID, emits messages, forwards to IDE.
	// -------------------------------------------------------------------------
	go func() {
		defer wg.Done()
		defer signalDone()

		scanner := newScanner(agentStdout)
		for scanner.Scan() {
			select {
			case <-done:
				return
			case <-ctx.Done():
				return
			default:
			}

			line := scanner.Text()

			// Attempt to extract session ID from session/new responses.
			// This is best-effort — we never modify the line based on this.
			if id := extractSessionID(line); id != "" {
				setSessionID(id)
				fmt.Fprintf(os.Stderr, "[recall/acp] session started: %s\n", id)
			}

			// Emit message for pipeline processing.
			out <- source.Message{
				Raw:        line,
				Direction:  "downstream",
				SessionID:  getSessionID(),
				SourceName: s.Name(),
				CapturedAt: time.Now().UTC(),
			}

			// Forward ORIGINAL (unmodified) line to IDE.
			// The IDE must see real ACP messages.
			fmt.Fprintln(os.Stdout, line)
		}

		if err := scanner.Err(); err != nil && err != io.EOF {
			fmt.Fprintf(os.Stderr, "[recall/acp] downstream read error: %v\n", err)
		}
	}()

	// Wait for both pipe goroutines to finish.
	wg.Wait()

	// CRITICAL: Source owns the channel lifecycle. We must close it.
	close(out)

	// Wait for the agent process to exit and return its status.
	return cmd.Wait()
}

// newScanner creates a bufio.Scanner with a generous buffer.
//
// ACP messages can be large — a single message may contain the full content
// of a file the agent read. The default 64KB buffer is too small.
// We allocate 4MB to handle even very large file reads without dropping data.
func newScanner(r io.Reader) *bufio.Scanner {
	scanner := bufio.NewScanner(r)
	scanner.Buffer(make([]byte, 4*1024*1024), 4*1024*1024)
	return scanner
}
