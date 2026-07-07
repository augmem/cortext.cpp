# augmem.cortext

Python bindings for Cortext, the on-device memory engine behind augmem. Use it
to turn text, audio, and image signals into persistent memories, then retrieve
relevant context for an app, agent, notebook, or LLM prompt.

The wheel is designed to work out of the box: it includes the native Cortext
library for the common desktop/server platforms and the default AIST GGUF
embedding asset. No separate model download is required for a normal install.

## Install

```bash
pip install augmem.cortext
```

```python
import augmem.cortext as cortext

print(cortext.version())
```

Supported release wheel platforms:

- Linux `x86_64` and `aarch64`
- macOS `x86_64` and Apple Silicon
- Windows `x86_64` and `aarch64`

## Quickstart

```python
import augmem.cortext as cortext

cfg = cortext.Config(
    focus=0.55,       # retrieval selectivity
    sensitivity=0.50, # responsiveness to new/surprising signals
    stability=0.65,   # preference for durable, stable context
)

with cortext.Cortext("memory.sqlite", config=cfg) as memory:
    memory.process_text(
        "The garage door code is 8841.",
        source_id="user/profile",
        include_embedding=False,
    )
    memory.process_text(
        "Mina's vet appointment is July 12 at 4pm.",
        source_id="chat/morning",
        include_embedding=False,
    )

    ctx = memory.process_text(
        "We are leaving soon. What should I remember about the garage?",
        source_id="chat/assistant",
        include_embedding=False,
    )

    for item in ctx["retrieved_memory"]:
        print(item["text"], item["rel"])

    if ctx["consolidation_recommended"]:
        memory.consolidate()

    memory.flush()
```

Use `":memory:"` instead of a file path for a temporary engine that disappears
when the process exits.

## Using It With an LLM

The basic loop is:

1. Store user-approved or application-observed context with `process_text`.
2. On later turns, call `process_text` and read `ctx["retrieved_memory"]`.
3. Insert those memory snippets into your prompt.
4. Call `consolidate()` when `consolidation_recommended` is true.

Example prompt assembly:

```python
ctx = memory.process_text(
    user_message,
    source_id=f"conversation/{conversation_id}",
    include_embedding=False,
)

memories = "\n".join(
    f"- {m['text']}" for m in ctx["retrieved_memory"][:6]
)

prompt = f"""Relevant memory:
{memories or "- none"}

User:
{user_message}
"""
```

Important: `process_text` is durable in the current Python API. It can retrieve
memory, but it also writes the input signal into the configured store. Use it
for turns you are willing to remember. Use `embed_text` for embedding-only work
that must not mutate memory.

## API Shape

```python
memory = cortext.Cortext(
    db_path="memory.sqlite",
    config=cortext.Config(),
    library_path=None,  # optional native library override
)
```

Core methods:

- `process_text(text, source_id, include_embedding=True) -> dict`
- `process_audio(pcm, source_id, include_embedding=True) -> dict`
- `process_image(data, width, height, channels, source_id, include_embedding=True) -> dict`
- `embed_text(text) -> list[float]`
- `embed_audio(pcm) -> list[float]`
- `embed_image(data, width, height, channels) -> list[float]`
- `consolidate() -> dict`
- `flush()`, `reset()`, `close()`

Each `process_*` method also has a `*_json` variant that returns the raw JSON
string from the native API.

## Returned Context

`process_*` returns a dictionary with the memory packet and diagnostics for the
current signal. The most commonly used fields are:

- `retrieved_memory`: long-term memories selected for this signal.
- `working_memory`: short-term active context.
- `should_interrupt`, `interrupt_aborted`, `at_boundary`: realtime behavior
  flags.
- `consolidation_recommended`, `consolidation_required`: maintenance hints.
- `output`: scores, storage decisions, filter status, and operation timings.
- `encode_ms`, `process_ms`, `hydrate_ms`, `total_ms`: latency breakdown.
- `embedding`, `embedding_dimension`: included only when requested.

Memory entries include `text`, `source_id`, `timestamp`, `modality`,
`mimetype`, `rel`, usage counts, metric scores, and soft-anchor metadata.

For prompt injection, pass `include_embedding=False`; embeddings are large and
usually not needed in the response packet.

## Audio and Image

Audio input is 16 kHz mono float PCM:

```python
pcm = [0.0] * 16000
ctx = memory.process_audio(pcm, "mic/main", include_embedding=False)
```

Image input is row-major RGB or RGBA bytes:

```python
rgb = bytes([0, 0, 0] * 64 * 64)
ctx = memory.process_image(rgb, 64, 64, 3, "camera/main")
```

Use `process_audio_with_media` or `process_image_with_media` when you want to
store original media bytes alongside the canonical signal:

```python
media = cortext.Media(data=jpeg_bytes, mimetype="image/jpeg")
ctx = memory.process_image_with_media(rgb, 64, 64, 3, "camera/main", media)
```

## Runtime Assets

Normal wheels include:

- `libcortext.so`, `libcortext.dylib`, or `cortext.dll` for supported
  platforms.
- The default AIST GGUF embedding asset and tokenizer vocabulary.

Override paths only when you are doing local development or testing a custom
asset:

- `CORTEXT_LIBRARY_PATH=/path/to/libcortext.so`
- `CORTEXT_AIST_MODEL_PATH=/path/to/AIST-87M_q8_0.gguf`

Library loading order is `library_path=`, `CORTEXT_LIBRARY_PATH`, bundled
library, local checkout build outputs, then the system library search path.

## Troubleshooting

- `Could not locate the Cortext shared library`: install a supported wheel, set
  `CORTEXT_LIBRARY_PATH`, or pass `library_path=...`.
- Engine creation fails: check write permission for `db_path`; for local builds,
  check `CORTEXT_AIST_MODEL_PATH`.
- Very large context dictionaries: call `process_text(...,
  include_embedding=False)`.
- Need a clean temporary run: use `cortext.Cortext(":memory:")`.
- Native failure details: call `cortext.last_error()` immediately after the
  exception.

## Build From Source

For local development without bundled artifacts:

```bash
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
export CORTEXT_LIBRARY_PATH="$PWD/build/ffi-release/libcortext.so"
```

Build a release wheel with bundled native libraries and the AIST asset:

```bash
python scripts/build_python_package.py --zig /path/to/zig
```

The generated wheel is written to `bindings/python/dist/`.
