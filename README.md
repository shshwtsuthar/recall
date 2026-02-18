# Recall

`recall-proxy` is a transparent proxy for agent sessions (currently only implemented for ACP).

It sits between the editor (for example Zed) and an ACP-compatible agent binary, then:

1. Forwards traffic unchanged so the editor and agent keep working normally.
2. Captures upstream/downstream ACP messages.
3. Scrubs sensitive data.
4. Sends the scrubbed payloads to your ingest server.

So all in all: user sends data to the agent, proxy recieves it, forwards it to the agent. Agent receives the data, outputs something, the output is forwarded to the proxy which in turn forwards it to the user. The proxy acts like unobstructing middleware capturing raw data.

## Current Scope

- ACP is (as of now) the most widely accepted protocol for Agent to Agent communication. Even though this protocol is now deprecated and merged with the A2A protocol, multiple other agent applications are `acp` only.
- It captures only sessions launched through `recall-proxy`.
- No global daemon/hook: each editor/agent integration must explicitly use this proxy command.
- Gemini is the textbook example of ACP communication given by Zed's documentation. This makes the proxy practical and ridiculously easy to set up.

## Build

```bash
go build .
```

Or use the existing binary in this repo: `./recall`.

## Configuration

Environment variables:

- `RECALL_SERVER` (required): full ingest endpoint URL.
  - Example: `http://127.0.0.1:8080/ingest`
- `RECALL_SECRETS` (optional): comma-separated env var names whose values should be redacted.
  - Example: `DATABASE_URL,GITHUB_TOKEN,OPENAI_API_KEY`

CLI shape:

```bash
./recall-proxy --source acp --agent <agent-binary> -- <agent-args...>
```

Notes:

- `--source` defaults to `acp`.
- `--` separates proxy flags from arguments passed to the real agent.

## Run Against a Local Server (Verification)

This is the easiest way to confirm transmission works.

### 1) Start a local ingest server

```py
from http.server import BaseHTTPRequestHandler, HTTPServer

class H(BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(n).decode("utf-8", errors="replace")
        print("\n--- REQUEST ---")
        print("Path:", self.path)
        print("Body:", body)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

HTTPServer(("127.0.0.1", 8080), H).serve_forever()
```

Running this will start a server on 8080. This is the ingest server where the raw conversation from the agent and the user will be sent to.

### 2) Start recall-proxy

```bash
RECALL_SERVER=http://127.0.0.1:8080/ingest ./recall-proxy --source acp --agent cat
```

`cat` is used only for loopback testing.

### 3) Send one ACP line

In the terminal window where the Go binary was ran, type:

```json
{"jsonrpc":"2.0","id":1,"method":"ping"}
```

### 4) Confirm capture

In the window where the Python program is running, you should be able to see `POST /ingest` bodies containing fields such as:

- `direction` (`upstream` / `downstream`)
- `raw` (scrubbed text)
- `session_id`
- `source_name` (`acp`)
- `captured_at`

If you see those requests, transmission is working.

## Integrating with an Editor (Zed Example)

The key requirement is: configure the editor's ACP agent command to run `recall` instead of running the agent binary directly.

Conceptually:

- Before:
  - `<agent-binary> --experimental-acp ...`
- After:
  - `recall --source acp --agent <agent-binary> -- --experimental-acp ...`

In Zed, Gemini is a textbook example of an agent who supports communication via the ACP procotol. Hence, we will configure Zed to use it.

In your `settings.json`, you will have a block:

```json
{
  "agent_servers": {
    "claude": {
      "favorite_models": ["sonnet"],
    },
```

Here, you need to add a block that will launch Recall:

```json
    "recall-gemini": {
      "type": "custom",
      "command": "/home/shashwat/Documents/Projects/recall/recall",
      "args": [
        "--source",
        "acp",
        "--agent",
        "gemini",
        "--",
        "--experimental-acp",
      ],
      "env": {
        "RECALL_SERVER": "http://127.0.0.1:8080/ingest",
      },
    },
```

So now, your config should look like:

```json
// Zed settings
//
// For information on how to configure Zed, see the Zed
// documentation: https://zed.dev/docs/configuring-zed
//
// To see all of Zed's default settings without changing your
// custom settings, run `zed: open default settings` from the
// command palette (cmd-shift-p / ctrl-shift-p)
{
  "agent_servers": {
    "claude": {
      "favorite_models": ["sonnet"],
    },
    "recall-gemini": {
      "type": "custom",
      "command": "/home/shashwat/Documents/Projects/recall/recall",
      "args": [
        "--source",
        "acp",
        "--agent",
        "gemini",
        "--",
        "--experimental-acp",
      ],
      "env": {
        "RECALL_SERVER": "http://127.0.0.1:8080/ingest",
      },
    },
  },
  "ui_font_size": 16,
  "buffer_font_size": 15,
  "theme": {
    "mode": "system",
    "light": "One Light",
    "dark": "One Dark",
  },
}
```

## Multiple Editors / Windows

`recall-proxy` is per launched session/process.

- If Zed is configured to use `recall-proxy`, Zed traffic is captured.
- If VS Code is not configured to use `recall-proxy`, VS Code traffic is not captured.
- To capture both, configure both integrations to launch through `recall-proxy` (typically separate proxy processes).

## Scrubbing Behavior (Client-Side)

Before transmission, the proxy scrubs common sensitive patterns, including:

- API keys/tokens (Anthropic, OpenAI, Google, GitHub, AWS, Stripe, Slack, npm, HuggingFace)
- Generic `password/secret/token/api_key` assignments
- JWT and Bearer tokens
- Emails
- User-home path prefixes
- Private IPs
- Explicit secret env var values from `RECALL_SECRETS`

Traffic forwarded between editor and agent remains unmodified.

## Troubleshooting

- Error: `RECALL_SERVER environment variable is required`
  - Set `RECALL_SERVER` before running.
- Error: `acp source requires --agent`
  - Pass `--agent <binary>`.
- No requests on local server:
  - Ensure local server is listening on the exact host/port in `RECALL_SERVER`.
  - Ensure editor is actually launching `recall-proxy` (not agent directly).
  - Send a test message to verify end-to-end path first.
