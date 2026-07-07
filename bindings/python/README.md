# augmem.cortext

Python bindings for Cortext, the on-device memory engine behind augmem. Use it
to persist text, audio, and image signals, then retrieve relevant context for
agents, applications, notebooks, and LLM prompts.

The PyPI wheels ship cross-platform native Cortext libraries for supported
desktop/server platforms. They do not embed the large AIST GGUF model. On first
engine creation, the wrapper uses `CORTEXT_AIST_MODEL_PATH` if set, then any
bundled/local model already present, otherwise it downloads the default AIST
GGUF model into the user cache and verifies its checksum before use.

## Install

```bash
pip install augmem.cortext
```

```python
import augmem.cortext as cortext

print(cortext.version())
```

Python 3.10+ is required. Release wheels target Linux, macOS, and Windows on
`x86_64`/`aarch64` where native artifacts are published.

## Quickstart

```python
import augmem.cortext as cortext

cfg = cortext.Config(
    focus=0.55,       # retrieval selectivity
    sensitivity=0.50, # responsiveness to new or surprising input
    stability=0.65,   # preference for durable, stable context
)

with cortext.Cortext("memory.sqlite", config=cfg) as memory:
    memory.process_text(
        "The garage door code is 8841.",
        source_id="user/profile",
        include_embedding=False,
    )

    ctx = memory.process_text(
        "We are leaving soon. What should I remember about the garage?",
        source_id="chat/assistant",
        include_embedding=False,
    )

    for item in ctx.get("retrieved_memory", []):
        print(item.get("text"), item.get("rel"))

    if ctx.get("consolidation_recommended"):
        memory.consolidate()

    memory.flush()
```

Use `":memory:"` for a temporary engine. Use a file path when memories should
survive process restarts.

## Chat Completions Memory Loop

The normal loop is simple:

1. Call `process_text` for turns or observations you are willing to remember.
2. On later turns, read `ctx["retrieved_memory"]`.
3. Pass those snippets as context messages to Chat Completions.
4. Call `consolidate()` when Cortext recommends it.

Install the OpenAI SDK and set `OPENAI_API_KEY`:

```bash
pip install openai
```

```python
from openai import OpenAI
import augmem.cortext as cortext
import os

client = OpenAI()
memory = cortext.Cortext("memory.sqlite")

def answer(conversation_id: str, user_message: str) -> str:
    ctx = memory.process_text(
        user_message,
        source_id=f"conversation/{conversation_id}",
        include_embedding=False,
    )

    memories = "\n".join(
        f"- {m.get('text', '')}"
        for m in ctx.get("retrieved_memory", [])[:6]
        if m.get("text")
    )

    completion = client.chat.completions.create(
        model=os.environ.get("OPENAI_MODEL", "gpt-5-mini"),
        messages=[
            {
                "role": "developer",
                "content": (
                    "Use the supplied Cortext memories when they are relevant. "
                    "Ignore them when they are not relevant."
                ),
            },
            {
                "role": "developer",
                "content": f"Cortext retrieved memory:\n{memories or '- none'}",
            },
            {"role": "user", "content": user_message},
        ],
    )

    if ctx.get("consolidation_recommended"):
        memory.consolidate()

    return completion.choices[0].message.content or ""
```

Durable-write warning: `process_text`, `process_audio`, and `process_image`
retrieve context and also write the input signal to the configured store. Do
not use them as read-only queries for content that should not be remembered.
Use `embed_text`, `embed_audio`, or `embed_image` for embedding-only work.

## Returned Context

`process_*` returns a dictionary parsed from the native context packet. Common
fields:

- `retrieved_memory`: long-term memories selected for the current signal.
- `working_memory`: short-term active context.
- `should_interrupt`, `interrupt_aborted`, `at_boundary`: realtime behavior
  flags.
- `consolidation_recommended`, `consolidation_required`: maintenance hints.
- `output`: scores, storage decisions, filter status, and operation timings.
- `encode_ms`, `process_ms`, `hydrate_ms`, `total_ms`: latency breakdown.
- `embedding`, `embedding_dimension`: present only when requested.

Memory entries commonly include `text`, `source_id`, `timestamp`, `modality`,
`mimetype`, `rel`, usage counts, scores, and soft-anchor metadata. For prompt
assembly, pass `include_embedding=False`; embeddings are large and rarely
needed in the returned packet.

## Audio and Image

Audio input is 16 kHz mono float PCM:

```python
pcm = [0.0] * 16000
ctx = memory.process_audio(pcm, "mic/main", include_embedding=False)
```

Image input is row-major RGB or RGBA bytes:

```python
rgb = bytes([0, 0, 0] * 64 * 64)
ctx = memory.process_image(rgb, 64, 64, 3, "camera/main", include_embedding=False)
```

Use media variants when you want to store original bytes next to the canonical
signal:

```python
media = cortext.Media(data=jpeg_bytes, mimetype="image/jpeg")
ctx = memory.process_image_with_media(
    rgb, 64, 64, 3, "camera/main", media, include_embedding=False
)
```

## Runtime Assets and Model Cache

The wheel contains the native Cortext shared library for supported platforms.
The AIST GGUF model is intentionally not embedded in registry packages because
of PyPI size constraints.

Model resolution on engine creation:

1. `CORTEXT_AIST_MODEL_PATH=/path/to/AIST-87M_q8_0.gguf`
2. A bundled or checkout-local model, if one exists.
3. Download the default AIST GGUF model to the user cache and verify its
   checksum before loading it.

The first run may need network access and enough cache space for the model
(roughly 135-142 MiB, depending on quantization). Later runs reuse the verified
cache. Set `CORTEXT_MODEL_CACHE_DIR` to control the cache root, or set
`CORTEXT_AIST_MODEL_PATH` for offline deployments and pinned model files.

Native library override:

- `CORTEXT_LIBRARY_PATH=/path/to/libcortext.so`
- or `cortext.Cortext(..., library_path="/path/to/libcortext.so")`

## API Shape

```python
memory = cortext.Cortext(
    db_path="memory.sqlite",
    config=cortext.Config(),
    library_path=None,
)
```

Core methods:

- `process_text(text, source_id, include_embedding=True) -> dict`
- `process_audio(pcm, source_id, include_embedding=True) -> dict`
- `process_image(data, width, height, channels, source_id, include_embedding=True) -> dict`
- `process_audio_with_media(...) -> dict`
- `process_image_with_media(...) -> dict`
- `embed_text(text) -> list[float]`
- `embed_audio(pcm) -> list[float]`
- `embed_image(data, width, height, channels) -> list[float]`
- `consolidate() -> dict`
- `flush()`, `reset()`, `close()`

Each `process_*`, `embed_*`, and `consolidate` method has a JSON variant that
returns the raw native JSON string.

## Troubleshooting

- First engine creation is slow: the model may be downloading and verifying.
- Model download fails: check network access, cache write permission, or set
  `CORTEXT_AIST_MODEL_PATH` to a local GGUF file.
- Checksum failure: remove the partially downloaded cache file and retry.
- Native library cannot be found: install a supported wheel, set
  `CORTEXT_LIBRARY_PATH`, or pass `library_path=...`.
- Very large context objects: call `process_text(..., include_embedding=False)`.
- Need a clean temporary run: use `cortext.Cortext(":memory:")`.
- Native failure details: call `cortext.last_error()` immediately after the
  exception.

## Build From Source

For local development from a repository checkout:

```bash
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
export CORTEXT_LIBRARY_PATH="$PWD/build/ffi-release/libcortext.so"
export CORTEXT_AIST_MODEL_PATH="$PWD/models/AIST-87M-GGUF/AIST-87M_q8_0.gguf"
```

Build a release wheel with native libraries:

```bash
python scripts/build_python_package.py --zig /path/to/zig --skip-models
```

The wheel is written to `bindings/python/dist/`. Registry wheels should include
native libraries but leave the large AIST model to the runtime cache path above.
