# FFI glue (engine-owned)

This directory holds **optional** foreign-function glue compiled with the
engine. It is not a language package.

| Path | Role |
| --- | --- |
| `node/addon.cpp` | N-API addon source for optional `cortext_node` / `cortext.node` builds |

Language packages are published from sibling repositories:

- [cortext.py](https://github.com/augmem/cortext.py)
- [cortext.ts](https://github.com/augmem/cortext.ts)
- [cortext.go](https://github.com/augmem/cortext.go)
- [cortext.dart](https://github.com/augmem/cortext.dart)
- [cortext.wasm](https://github.com/augmem/cortext.wasm)

Enable the Node addon via CMake (`CORTEXT_BUILD_NODE_BINDINGS`) or Zig
(`-Dnode-addon=true`). On Windows, CMake searches the Node prefix for
`node.lib` (including `x64/` and `lib/`); set `-DCORTEXT_NODE_LIBRARY=...` to
override the import-library path when Node is installed elsewhere.

## Node addon surface

The addon exports `version()`, `lastError()` and the class
`NativeCortext(config?, dbPath = ":memory:")` (or `NativeCortext(dbPath)`),
whose methods wrap the C API one to one and return the C API JSON string:

| Method | C API |
| --- | --- |
| `processTextJson(text, sourceId, options?)` | `cortext_process_text_json_with_options` |
| `processTextWithMediaJson(text, sourceId, media?, mimeType?, options?)` | `cortext_process_text_with_media_json_with_options` |
| `processAudioJson(pcm, sourceId, options?)` | `cortext_process_audio_json_with_options` |
| `processAudioWithMediaJson(pcm, sourceId, media?, mimeType?, options?)` | `cortext_process_audio_with_media_json_with_options` |
| `processImageJson(data, width, height, channels, sourceId, options?)` | `cortext_process_image_json_with_options` |
| `processImageWithMediaJson(data, width, height, channels, sourceId, media?, mimeType?, options?)` | `cortext_process_image_with_media_json_with_options` |
| `embedTextJson`, `embedAudioJson`, `embedImageJson`, `consolidateJson`, `flush`, `reset` | the matching C calls |

`config` takes `focus`, `sensitivity`, `stability` and the boolean
`cortext_config` fields in camelCase (`affectInterrupt`,
`signalFilter{Audio,Image,Text}Enabled`).

`media` is bytes (`Buffer` or `Uint8Array`) stored as the payload
with `mimeType` instead of the canonical input; absent or empty media stores
the canonical input. Only `text/*` media is used as text.

Process `options` (`cortext_process_json_options`):

- `includeEmbedding` / `omitEmbedding`: keep or drop the signal embedding.
- `retention`: `"natural"` (default), `"durable"`, `"boundary"`,
  `"ephemeral"`, or the numeric `cortext_retention`.
- `evidenceBudgetBytes`: per-call evidence budget in whole bytes; absent or 0
  returns every recalled memory whole.
- `referenceMs`: query reference time in whole ms since the Unix epoch;
  absent or `null` leaves retrieval unchanged. Read by ephemeral calls only.
- `asOf`: with `referenceMs`, hide memories that start after it.

Invalid options throw a `TypeError`. The returned JSON is the C API context
packet described in the top-level README (`retrieved_memory`,
`standing_memory`, per-memory `modality` including `"mixed"`, and
`content[]` entries with their own `mimetype` and `timestamp`).

The AIST model and its late-interaction reranker (`AIST-87M-LI.gguf`) resolve
as for any engine build: linked into an embed-on library, or from
`CORTEXT_AIST_MODEL_PATH` with the reranker beside it or at
`CORTEXT_LI_MODEL_PATH`.
