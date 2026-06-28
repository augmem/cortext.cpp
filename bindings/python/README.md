# cortext Python Binding

This package loads the native Cortext shared library with `ctypes`.

## Build

From the repository root:

```bash
cmake --preset ffi-release
cmake --build --preset ffi-release --target cortext
```

Set `CORTEXT_LIBRARY_PATH=/path/to/libcortext.so` when loading a non-default
library.

## Use

```bash
PYTHONPATH=bindings/python python3 - <<'PY'
import cortext

with cortext.Cortext(db_path=":memory:", models_dir="models") as engine:
    print(cortext.version())
    print(engine.process_text("Bailey likes tennis balls.", "chat/main"))
    print(len(engine.embed_text("embed without storing")))
    engine.consolidate()
    engine.flush()
PY
```

## API

- `Config`: `focus`, `sensitivity`, `stability`,
  `affect_interrupt`, `affect_retrieval`, `reinforcement_enabled`,
  `procedural_enabled`, `sequential_edges_enabled`,
  `signal_filter_audio_enabled`, `signal_filter_image_enabled`,
  `signal_filter_text_enabled`.
- `Cortext.process_text`, `process_audio`, `process_image`: return decoded JSON
  context dictionaries.
- `Cortext.process_*_json`: return raw JSON strings.
- `Cortext.embed_text`, `embed_audio`, `embed_image`: return embedding lists and
  do not mutate memory state.
- `Cortext.consolidate`, `flush`, `reset`, `close`.
- `DBProvider` and `ObjectStoreProvider`: callback interfaces for caller-owned
  storage.

Audio input is 16 kHz mono float32 PCM. Image input is row-major RGB/RGBA bytes
with explicit dimensions.
