# LongChat Proxy

[![CI](https://github.com/newtonlorenz/longchat-proxy/actions/workflows/ci.yml/badge.svg)](https://github.com/newtonlorenz/longchat-proxy/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A transparent rolling-context proxy for OpenAI-compatible local LLM servers.**

LongChat Proxy lets an ordinary chat client keep a conversation going after it outgrows a
model's context window. It summarises older turns with a small helper model, preserves the
newest turns verbatim, and forwards a bounded request to your main model. The client does not
need a plugin or special API.

It works with clients that support a custom OpenAI base URL and servers such as LM Studio,
llama.cpp, Ollama's OpenAI endpoint, LocalAI, vLLM and compatible gateways.

```text
Phone, browser or desktop client
              │ OpenAI chat/completions
              ▼
       ┌──────────────────┐
       │  LongChat Proxy  │
       │                  │
       │ recent turns ───────────────┐
       │ older turns ──┐             │
       └───────────────┼─────────────┘
                       ▼             ▼
                small summariser   main model
                       │             ▲
                       └─ memory ─────┘
```

## Why this exists

Most local chat clients send the entire conversation on every request. Eventually one of two
things happens:

1. the server rejects the request because it is larger than the context window, or
2. the client/server silently discards the oldest turns.

Simply truncating history loses names, decisions, preferences and unfinished work. LongChat
Proxy instead maintains a rolling memory while keeping the recent conversational texture
unchanged.

## What it does

- Transparently proxies OpenAI-compatible endpoints; only `/v1/chat/completions` is modified.
- Reserves room for the answer before measuring the input budget.
- Summarises old turns in bounded batches, so the helper model cannot overflow either.
- Keeps a configurable recent window verbatim.
- Merges existing system/developer instructions with rolling memory into one system message.
- Preserves tool calls, tool results and text/image attachment metadata during accounting.
- Supports streaming responses without waiting for the full answer.
- Reuses summaries when the client sends the same conversation prefix again.
- Supports private in-memory state, persistent SQLite state, or stateless operation.
- Accepts an optional exact Hugging Face `tokenizer.json`; otherwise uses a conservative estimate.
- Can compact before the physical limit to trade context detail for latency and prompt-cache reuse.
- Can enforce provider-specific main-request fields and remove configured volatile client boilerplate.
- Hides the internal summariser from `/v1/models` by default.
- Exposes safe health, readiness and authenticated status endpoints.
- Fails explicitly if compaction is required but cannot be completed safely.

## What it does not do

- It does not increase the model's actual context window.
- It cannot guarantee perfect recall. Summaries are deliberately lossy.
- It cannot recover messages already removed by the client before they reach the proxy.
- It does not make a model smarter or add retrieval over external documents.
- It is not a substitute for TLS, a VPN or network access control.

## Quick start

Requirements: Python 3.11+ and a running OpenAI-compatible model server.

```bash
git clone https://github.com/newtonlorenz/longchat-proxy.git
cd longchat-proxy
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
cp .env.example .env
```

Edit `.env`:

```dotenv
LONGCHAT_UPSTREAM_BASE_URL=http://127.0.0.1:1234
LONGCHAT_CONTEXT_TOKENS=8192
LONGCHAT_MAX_OUTPUT_TOKENS=2048
LONGCHAT_RECENT_TOKENS=1600

# Optional but strongly recommended: a small, separately loaded helper model.
LONGCHAT_SUMMARY_MODEL=qwen3.5-summary
LONGCHAT_SUMMARY_CONTEXT_TOKENS=8192
LONGCHAT_SUMMARY_TOKENS=400
```

Validate the setup, then start it:

```bash
longchat-proxy doctor
longchat-proxy serve
```

Point your client at:

```text
http://127.0.0.1:1235/v1
```

Use the main model's normal model ID. Do **not** select the summary model.

## LM Studio example

Load two models in LM Studio:

- your main model at its useful context size, such as 8K or 16K;
- a fast 1B–4B instruct model as the summariser.

Use one parallel slot for each if this is a single-user machine. A practical 32GB Apple Silicon
configuration is a 4-bit 20B–30B main model plus a 4-bit 1B–3B helper.

Some Qwen templates support an explicit thinking switch. Disable thinking for the summariser so
it spends its small output budget on memory rather than hidden reasoning:

```dotenv
LONGCHAT_SUMMARY_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":false}}
```

Then configure LongChat:

```dotenv
LONGCHAT_UPSTREAM_BASE_URL=http://127.0.0.1:1234
LONGCHAT_SUMMARY_MODEL=qwen3.5-summary
LONGCHAT_CONTEXT_TOKENS=8192
LONGCHAT_MAX_OUTPUT_TOKENS=2048
LONGCHAT_RECENT_TOKENS=1600
LONGCHAT_SUMMARY_TOKENS=400
LONGCHAT_STATE_MODE=sqlite
```

The main request remains under the client's control by default. LongChat caps its output allowance
but does not otherwise enable or disable main-model thinking unless you configure an override.

To enforce non-thinking mode even when a client sends conflicting options, configure a main-request
override:

```dotenv
LONGCHAT_REQUEST_EXTRA_BODY={"chat_template_kwargs":{"enable_thinking":false}}
```

Some clients inject a changing timestamp and tool schema into every system prompt, which defeats
prompt caching. If you do not use those client tools, the following optional controls keep the
stable parts of the prompt while removing that boilerplate:

```dotenv
LONGCHAT_DROP_TOOLS=true
LONGCHAT_STRIP_SYSTEM_TIMESTAMP=true
LONGCHAT_DROP_SYSTEM_SECTIONS=Tools
```

## Ollama example

Ollama exposes an OpenAI-compatible endpoint on port `11434`:

```dotenv
LONGCHAT_UPSTREAM_BASE_URL=http://127.0.0.1:11434
LONGCHAT_SUMMARY_MODEL=qwen3.5:2b
LONGCHAT_CONTEXT_TOKENS=8192
```

Make sure both the main model and summary model have already been pulled. Model names must match
the IDs returned by `GET /v1/models`.

## Phone and LAN access

LongChat binds to loopback by default. For a phone, a private VPN such as Tailscale is the safest
simple option:

```dotenv
LONGCHAT_HOST=100.x.y.z
LONGCHAT_CLIENT_API_KEY=generate-a-long-random-value
```

The proxy refuses to start an unauthenticated non-loopback listener unless you explicitly set
`LONGCHAT_ALLOW_UNAUTHENTICATED_REMOTE=true`.

Configure the phone client with:

```text
Base URL: http://100.x.y.z:1235/v1
API key:  the LONGCHAT_CLIENT_API_KEY value
Model ID: your main model ID
```

If you expose the service beyond a private VPN, put it behind an HTTPS reverse proxy and apply
network-level access controls. Do not put the raw HTTP listener on the public internet.

## How compaction works

For every chat request LongChat:

1. caps the requested output at `LONGCHAT_MAX_OUTPUT_TOKENS`;
2. subtracts output and safety reserves from the main context window;
3. counts the system prompt, tools, messages and image reserves;
4. forwards the request unchanged if it fits;
5. finds a cached summary matching an exact prefix, if one exists;
6. keeps approximately `LONGCHAT_RECENT_TOKENS` of recent turns verbatim;
7. summarises older complete exchanges in helper-model-sized batches;
8. combines original instructions and rolling memory into one system message;
9. verifies that the final request fits before forwarding it.

Responses include diagnostic headers:

```text
X-LongChat-Compacted: true
X-LongChat-Cache-Hit: true
X-LongChat-Estimated-Input-Tokens: 1834
X-LongChat-Summary-Calls: 0
```

Clients may send `X-LongChat-Conversation-ID` to make state lookup more specific. It is optional;
exact-prefix matching works with unmodified clients.

## Token counting

The default estimator is intentionally conservative and accounts for UTF-8 text, punctuation,
message framing, tools and configurable image reserves. For the tightest budgets, install the
tokenizer extra and point LongChat at the main model's `tokenizer.json`:

```bash
pip install -e '.[tokenizers]'
```

```dotenv
LONGCHAT_TOKENIZER_PATH=/path/to/model/tokenizer.json
```

Even with an exact text tokenizer, image token use and chat-template framing vary by runtime, so
`LONGCHAT_SAFETY_TOKENS` remains important.

## State and privacy

`LONGCHAT_STATE_MODE` supports:

| Mode | Behaviour |
|---|---|
| `memory` | Default. Summaries disappear when the process stops. |
| `sqlite` | Persists rolling summaries across restarts. |
| `off` | Never stores summaries; old turns may be re-summarised repeatedly. |

Only prefix hashes and generated summaries are stored—not complete transcripts. Summaries can
still contain private information. Protect the SQLite file with normal OS account and disk
encryption controls. Records are bounded by count and age.

## Configuration

All settings use the `LONGCHAT_` prefix.

| Variable | Default | Purpose |
|---|---:|---|
| `HOST` | `127.0.0.1` | Listener address. |
| `PORT` | `1235` | Listener port. |
| `UPSTREAM_BASE_URL` | `http://127.0.0.1:1234` | Main OpenAI-compatible server. |
| `UPSTREAM_API_KEY` | unset | Credential sent to the upstream server. |
| `CLIENT_API_KEY` | unset | Bearer key required from clients. |
| `ALLOW_UNAUTHENTICATED_REMOTE` | `false` | Explicitly allow an unsafe remote listener. |
| `CONTEXT_TOKENS` | `8192` | Main model's loaded context size. |
| `MAX_OUTPUT_TOKENS` | `2048` | Hard response-token cap and reserved allowance. |
| `RECENT_TOKENS` | `1600` | Approximate recent verbatim window. |
| `COMPACTION_TRIGGER_TOKENS` | `0` | Optional earlier input threshold; zero uses the physical budget. |
| `SUMMARY_TOKENS` | `400` | Maximum helper-model summary output. |
| `SAFETY_TOKENS` | `384` | Extra margin for templates/counting differences. |
| `MIN_MESSAGES` | `6` | Minimum turn count before compaction is allowed. |
| `IMAGE_TOKEN_RESERVE` | `1200` | Estimated cost per image/video item. |
| `SUMMARY_MODEL` | request model | Dedicated helper model ID. |
| `SUMMARY_BASE_URL` | upstream URL | Optional separate helper server. |
| `SUMMARY_API_KEY` | upstream key | Optional separate helper credential. |
| `SUMMARY_CONTEXT_TOKENS` | `8192` | Helper model's loaded context size. |
| `SUMMARY_TEMPERATURE` | `0.2` | Helper sampling temperature. |
| `SUMMARY_EXTRA_BODY` | `{}` | Provider-specific JSON merged into helper requests. |
| `REQUEST_EXTRA_BODY` | `{}` | Enforced provider-specific JSON merged into main requests. |
| `DROP_TOOLS` | `false` | Remove tool definitions and choices from main requests. |
| `STRIP_SYSTEM_TIMESTAMP` | `false` | Remove changing `Current date and time:` instruction lines. |
| `DROP_SYSTEM_SECTIONS` | unset | Comma-separated Markdown headings removed from leading instructions. |
| `SYSTEM_APPEND` | unset | Stable guidance appended to the first instruction message. |
| `TOKENIZER_PATH` | unset | Optional Hugging Face `tokenizer.json`. |
| `ESTIMATOR_MULTIPLIER` | `1.10` | Conservatism applied to estimated counts. |
| `STATE_MODE` | `memory` | `memory`, `sqlite`, or `off`. |
| `STATE_PATH` | user data directory | SQLite database location. |
| `STATE_MAX_RECORDS` | `500` | Maximum cached summaries. |
| `STATE_TTL_DAYS` | `30` | Persistent-record lifetime. |
| `REQUEST_MAX_BYTES` | `16777216` | Maximum request body size. |
| `UPSTREAM_TIMEOUT_SECONDS` | `600` | Main-model request timeout. |
| `SUMMARY_TIMEOUT_SECONDS` | `300` | Helper-model request timeout. |
| `RATE_LIMIT_RPM` | `0` | Per-IP request limit; zero disables it. |
| `CORS_ORIGINS` | unset | Comma-separated browser origins allowed to call the proxy. |
| `HIDE_SUMMARY_MODEL` | `true` | Remove the helper from `/v1/models`. |
| `ON_COMPACTION_ERROR` | `error` | `error` safely stops; `passthrough` forwards oversized input. |
| `LOG_LEVEL` | `INFO` | Python log level. |

Run `longchat-proxy print-config` to inspect the effective configuration with secrets redacted.
The upstream and summary base URLs may be written with or without a trailing `/v1`.

## Operational endpoints

| Endpoint | Authentication | Purpose |
|---|---|---|
| `GET /longchat/health` | none | Process liveness only. |
| `GET /longchat/ready` | client key | Checks upstream `/v1/models`. |
| `GET /longchat/status` | client key | Redacted config, counters and state size. |

## Docker

```bash
cp .env.example .env
# Set LONGCHAT_CLIENT_API_KEY in .env because Docker listens outside loopback.
docker compose up --build
```

The included Compose file uses `host.docker.internal` to reach a model server running on the host.
Linux users may need the included host-gateway mapping or a different upstream address.

## Running as a service

- macOS launchd template: [`examples/launchd/ai.longchat-proxy.plist`](examples/launchd/ai.longchat-proxy.plist)
- Linux systemd unit: [`examples/systemd/longchat-proxy.service`](examples/systemd/longchat-proxy.service)

Both templates keep secrets in a separate environment file rather than embedding them in the
service definition.

## Failure behaviour

LongChat defaults to failing closed. If a request needs compaction and the helper is unavailable,
it returns a `503 compaction_error` rather than silently dropping history or forwarding a request
known to be oversized. Set `LONGCHAT_ON_COMPACTION_ERROR=passthrough` only if your upstream has its
own trustworthy overflow strategy.

## Limitations

- Summary quality depends on the helper model and prompt-following ability.
- Clients that pre-truncate their own history prevent LongChat from seeing the lost turns.
- Multimodal content is budgeted and represented as attachment metadata; images are not sent to
  the summariser by default.
- Exact provider compatibility varies. Use `SUMMARY_EXTRA_BODY` for template-specific controls.
- One process is recommended. In-memory locks and counters are not shared across Uvicorn workers.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,tokenizers]'
ruff check .
pytest
```

See [CONTRIBUTING.md](CONTRIBUTING.md) and [SECURITY.md](SECURITY.md).

## License

MIT. See [LICENSE](LICENSE).
