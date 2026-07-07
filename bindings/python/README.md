# augmem.cortext

Python bindings for the Cortext memory engine used by augmem. The package loads
the native Cortext shared library with `ctypes` and exposes a small Python API
for processing text, audio, and image signals into memory-aware context.

## Install

```bash
pip install augmem.cortext
```

The import name is:

```python
import augmem.cortext as cortext
```

## Supported Wheels

Release wheels are intended to include the native Cortext library for:

- Linux: `x86_64`, `aarch64`
- macOS: `x86_64`, `aarch64`
- Windows: `x86_64`, `aarch64`

If your platform is not covered, build the native library locally and set
`CORTEXT_LIBRARY_PATH` to the resulting `libcortext.so`, `libcortext.dylib`, or
`cortext.dll`.

## Quickstart

```python
import augmem.cortext as cortext

cfg = cortext.Config(
    focus=0.55,
    sensitivity=0.5,
    stability=0.65,
)

with cortext.Cortext(
    db_path="memory.sqlite",
    models_dir="models",
    config=cfg,
) as engine:
    first = engine.process_text(
        "Bailey is preparing for a product demo on Friday.",
        "chat/main",
        include_embedding=False,
    )

    second = engine.process_text(
        "What should I remember about Bailey's week?",
        "chat/main",
        include_embedding=False,
    )

    for memory in second["retrieved_memory"]:
        print(memory["text"])

    if second["consolidation_recommended"]:
        engine.consolidate()

    engine.flush()
```

Use `db_path=":memory:"` for a session-local engine that does not persist after
the process exits. Use a file path when you want memory to survive restarts.

## No-Storage Calls

`process_text`, `process_audio`, and `process_image` process the signal through
the memory pipeline and may write to the configured store.

For embedding-only work, use the `embed_*` calls. They return vectors and do not
mutate memory state:

```python
with cortext.Cortext(db_path=":memory:", models_dir="models") as engine:
    embedding = engine.embed_text("embed this without storing it")
    print(len(embedding))
```

The Python binding does not currently expose the C++ `Retention::Ephemeral`
processing option. If you need a fully ephemeral processing session, create the
engine with `db_path=":memory:"`.

## Configuration

`Config` exposes Cortext's three main knobs plus feature toggles:

- `focus`: raises selectivity and favors higher-confidence context.
- `sensitivity`: controls how readily new or surprising signals trigger memory
  behavior.
- `stability`: favors longer-lived context and more stable continuity.

All three knobs are floats and default to `0.5`.

Additional booleans default to the values in `Config`: `affect_interrupt`,
`affect_retrieval`, `reinforcement_enabled`, `procedural_enabled`,
`sequential_edges_enabled`, `signal_filter_audio_enabled`,
`signal_filter_image_enabled`, and `signal_filter_text_enabled`.

## Returned Context

`process_*` methods return decoded JSON dictionaries. The `process_*_json`
methods return the raw JSON string.

At a high level, a context contains:

- `working_memory`: active short-term context slots.
- `retrieved_memory`: long-term memories selected for the current signal.
- `should_interrupt`, `interrupt_aborted`, `at_boundary`: realtime behavior
  flags.
- `consolidation_recommended`, `consolidation_required`: consolidation hints.
- `output`: scores, decisions, stored IDs, filter status, and operation timings.
- `encode_ms`, `process_ms`, `hydrate_ms`, `total_ms`: timing fields.
- `embedding` and `embedding_dimension` when `include_embedding=True`.

Memory entries include fields such as `text`, `source_id`, `timestamp`,
`modality`, `mimetype`, usage counts, metric scores, and `soft_anchors`.

Pass `include_embedding=False` to `process_*` when you do not need the embedding
in the returned dictionary.

## Audio, Image, and Media

Audio input is 16 kHz mono float32 PCM:

```python
pcm = [0.0] * 16000
ctx = engine.process_audio(pcm, "mic/main", include_embedding=False)
```

Image input is row-major RGB or RGBA bytes with explicit dimensions:

```python
ctx = engine.process_image(rgb_bytes, width, height, 3, "camera/main")
```

`process_audio_with_media` and `process_image_with_media` can store original
media bytes alongside the canonical processing input. Pass `media_mimetype`
whenever media bytes are provided, or use `cortext.Media(data, mimetype)`.

## Models and Native Runtime

`models_dir` points to local model assets used by the native engine. In a source
checkout this is usually `models`. In an application, ship or mount the model
directory you want Cortext to use and pass that path to `Cortext(...)`.

Library loading order is:

1. `library_path=` passed to `Cortext`.
2. `CORTEXT_LIBRARY_PATH`.
3. The bundled wheel native library for the current platform.
4. Local development build locations in a Cortext source checkout.
5. The system library search path.

## Local Source Build

For development from a source checkout without bundled native libraries:

```bash
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
export CORTEXT_LIBRARY_PATH="$PWD/build/ffi-release/libcortext.so"
```

On macOS the library is `libcortext.dylib`; on Windows it is `cortext.dll`.

To build the release wheel with bundled native libraries:

```bash
python scripts/build_python_package.py
```

Set `ZIG=/path/to/zig` or pass `--zig /path/to/zig` when Zig is not on `PATH`.

## Troubleshooting

- `Could not locate the Cortext shared library`: install a wheel for a supported
  platform, pass `library_path=...`, or set `CORTEXT_LIBRARY_PATH`.
- Engine creation fails: check that `models_dir` points to valid model assets
  and that the process can read/write `db_path`.
- Windows DLL load failures: make sure dependent DLLs are next to `cortext.dll`
  or on the DLL search path.
- Large responses: call `process_text(..., include_embedding=False)` or the
  equivalent audio/image method to omit the returned embedding.
- Need caller-owned storage: implement `DBProvider` and optionally
  `ObjectStoreProvider`, then pass them as `store=` and `object_store=`.

Use `cortext.version()` to confirm the loaded native library version and
`cortext.last_error()` when debugging native failures.
