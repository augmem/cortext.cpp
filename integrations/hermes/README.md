# augmem.hermes

Cortext memory provider for [Hermes Agent](https://hermes-agent.nousresearch.com/).

| | |
|---|---|
| **PyPI / import** | `augmem.hermes` |
| **Hermes provider id** | `cortext` |
| **CLI** | `augmem-hermes` |

Local **tri-modal** adaptive memory (text / image / audio) with Focus /
Sensitivity / Stability knobs, backed by the Cortext engine and SQLite.

**Silent by design:** the Hermes agent/LLM never sees Cortext tools, never
gets a Cortext system prompt, and never has to “use memory.” Ingest and
recall run only through Hermes lifecycle hooks. Prefetch injects unlabeled
prior context before each model call.

<!-- installation commands matching integrations/hermes/pyproject.toml and integrations/hermes/src/augmem/hermes/cli.py -->
## Install

Install into the **same Python environment as `hermes`**. The installer adds
the provider under `$HERMES_HOME/plugins/memory/cortext` and selects it as the
active memory provider automatically:

```bash
python -m pip install augmem.hermes
augmem-hermes install
```

From this monorepo (editable, uses the local Python bindings):

```bash
cd integrations/hermes
uv sync --extra dev
uv run augmem-hermes install
```

Or install into an environment that already has Hermes:

```bash
uv pip install -e integrations/hermes
augmem-hermes install
```

Enable the provider:

```bash
hermes config set memory.provider cortext
# or
hermes memory setup   # select "cortext"
```

Pass `--no-enable` to `augmem-hermes install` when you want to install the
files without changing Hermes's active memory provider.

Hermes's current `hermes plugins install` command installs Git repositories,
not PyPI package names. For this package, use the PyPI install above; a future
Hermes PyPI-registry feature would be needed for
`hermes plugins install augmem.hermes`.

Requirements:

- Python 3.10+
- `augmem.cortext` (declared dependency; path-sourced in this monorepo)
- Hermes Agent with memory-provider plugins enabled

## Layout

```text
integrations/hermes/
  pyproject.toml
  src/augmem/hermes/
    provider.py          # MemoryProvider implementation
    cli.py               # augmem-hermes install|status
    plugin/              # Hermes plugin tree (symlinked on install)
      __init__.py
      plugin.yaml
  tests/
```

`augmem-hermes install` links `src/augmem/hermes/plugin` to:

```text
$HERMES_HOME/plugins/memory/cortext
```

Default `HERMES_HOME` is `~/.hermes`.

<!-- configuration keys from integrations/hermes/src/augmem/hermes/provider.py -->
## Config

Non-secret settings live in `$HERMES_HOME/cortext.json` (written by
`hermes memory setup` / `save_config`):

```json
{
  "db_path": "$HERMES_HOME/cortext.sqlite",
  "focus": 0.55,
  "sensitivity": 0.5,
  "stability": 0.65,
  "top_k": 6,
  "seam_pre_tool": true,
  "auto_sync_turns": true,
  "auto_consolidate": true
}
```

| Key | Meaning |
|---|---|
| `db_path` | Cortext SQLite path (`$HERMES_HOME` expanded) |
| `focus` | Retrieval selectivity (F) |
| `sensitivity` | Responsiveness to new/surprising input (S) |
| `stability` | Durability / write-rate bias (T) |
| `top_k` | Memories injected per prefetch/search |
| `seam_pre_tool` | Block a proposed tool action only when Cortext returns a live interrupt with relevant prior context |
| `auto_sync_turns` | Persist each completed turn |
| `auto_consolidate` | Run consolidation on session end / when recommended |

## Agent surface

| Surface | Behavior |
|---|---|
| Tools | **None** (`get_tool_schemas() == []`) |
| System prompt | **Empty** (no product mention) |
| Prefetch text | Unlabeled bullet list of prior facts (no “Cortext” header) |
| CLI / install | Operator-only (`augmem-hermes`); not visible to the model |

<!-- lifecycle seams from integrations/hermes/src/augmem/hermes/provider.py -->
## Lifecycle (every seam)

Cortext runs at **four turn seams** by default (`seam_user`, `seam_pre_llm`,
`seam_pre_tool`, `seam_post_llm` — all `true`):

```text
 user message
     │
     ▼
 [USER SEAM]  on_turn_start → ingest user text (async)
     │         also on_memory_write / on_delegation / on_pre_compress
     ▼
 [PRE-LLM]    prefetch → recall packet injected before every model call
     │         (including tool-loop follow-up calls)
     ▼
   LLM
     │
     ▼
 [ACTION]     pre_tool_call → Natural action signal; block only on a live
     │         interrupt with relevant retrieved context
     ▼
   tool executes
     │
     ▼
 [POST-LLM]   sync_turn → ingest assistant (+ tool results), async
     │
     ▼
 session end → consolidate / flush
```

| Seam | Hermes hook | Cortext action |
|---|---|---|
| After user interaction | `on_turn_start` | Store user message |
| After user interaction | `on_memory_write` | Mirror MEMORY.md / USER.md |
| After user interaction | `on_delegation` | Store subagent task/result |
| After user interaction | `on_pre_compress` | Snapshot before context drop |
| Before every LLM call | `prefetch` | Inject recalled memories |
| Before next turn | `queue_prefetch` | Warm cache (async) |
| Before tool execution | `pre_tool_call` | Process proposed action with `Retention.NATURAL`; block only on an interrupt with prior context |
| After LLM turn | `sync_turn` | Store assistant + tool calls/results + media |

### Multimodal inputs

Cortext embeds **text, images (screenshots), and audio** in one space:

| Input | How it arrives | Cortext path |
|---|---|---|
| Chat text | user / assistant content | `process_text` |
| Screenshot / paste | OpenAI `image_url` parts, data URLs, file paths | `process_image_with_media` |
| Tool call | assistant `tool_calls` | compact JSON text signal |
| Tool result | `role=tool` content (text or image) | text + image/audio as present |
| WAV / voice | data URL or `.wav` path | `process_audio_with_media` |

Image decode uses Pillow (declared dependency). Long image edges are capped at
2048px for low latency. Media can be disabled with `"ingest_media": false`.
Remote `http(s)` media URLs are **not** fetched by default (SSRF safety); set
`"fetch_remote_media": true` only when you trust conversation content.

User text ingested at `on_turn_start` is **not** written again in `sync_turn`
(deduped by content hash). Non-primary contexts (`cron`, `subagent`, `flush`)
skip durable writes so system prompts do not pollute the store.

## Provenance (`source_id`)

Every stored signal gets an opaque Cortext `source_id` with **user vs agent**
role first:

```text
hermes/{user|agent}/{actor_id}/{session_id}/...
```

| Event | Example `source_id` |
|---|---|
| User message (turn 3) | `hermes/user/alice/sess-1/turn/3` |
| Agent reply (turn 3) | `hermes/agent/hermes/sess-1/turn/3` |
| Prefetch / search probe | `hermes/agent/hermes/sess-1/prefetch` |
| Explicit remember (agent tool) | `hermes/agent/hermes/sess-1/remember` |
| Explicit remember (`role=user`) | `hermes/user/alice/sess-1/remember` |
| Built-in MEMORY.md mirror | `hermes/user/alice/sess-1/builtin/memory` |
| Tool result in agent loop | `hermes/agent/hermes/sess-1/turn/3/tool/shell` |

`actor_id` comes from Hermes when available (`user_id` / `agent_identity`),
otherwise defaults to `user` / `agent`. Cortext treats `source_id` as opaque
provenance (same-source grouping); it does not parse these paths.

## Develop

```bash
cd integrations/hermes
uv sync --extra dev
uv run pytest
```

Unit tests use a fake engine and do not require a native Cortext build.

### Live model verification

Proves silent multi-seam memory with a **real Cortext engine** and a **live
chat model** (Bailey fact → new session prefetch → model answer).

```bash
# 1) Build native Cortext (repo root)
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext

# 2) Point bindings at the library (macOS example)
export CORTEXT_LIBRARY_PATH="$PWD/../../build/ffi-release/libcortext.dylib"
# Linux: .../libcortext.so

# 3) Live credentials
export CORTEXT_HERMES_LIVE=1
export OPENAI_API_KEY=sk-...
# optional:
# export OPENAI_BASE_URL=https://api.openai.com/v1
# export OPENAI_MODEL=gpt-4o-mini

# 4) Run
cd integrations/hermes
uv run augmem-hermes live-e2e --force
# or pytest:
uv run pytest -m live -s
```

What it checks (designed so chat history cannot cheat):

1. **Session A** stores a fact via seams, then **fully shuts down** the engine  
2. **Session B** opens a **new** provider on the same SQLite file only  
3. Cold-start **prefetch** must contain the fact (no Session A messages)  
4. **Control** live call: probe only → model must *not* know the secret  
5. **Treatment** live call: probe + cold-start prefetch only → must use memory  
6. Treatment must **beat** control; provider stays invisible  

Exit codes: `0` pass, `1` fail, `2` skipped (missing deps/flags).

## Notes

- Provider store paths use `Retention.DURABLE`; prefetch/search uses
  `Retention.EPHEMERAL` so recall does not create a new memory. The action
  gate uses `Retention.NATURAL` so the engine's boundary policy can retain an
  action trace while returning an immediate interrupt decision. Queries are
  labeled with `hermes/*` source ids so they remain distinguishable from
  explicit remembers.
- Coding-repo retrieval (find symbol X in the tree) is out of scope; pair with a
  code-index MCP for that. Cortext is for long-horizon personal/project memory.
