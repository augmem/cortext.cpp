# External Integrations

**Analysis Date:** 2026-04-07

## APIs & External Services

**Local inference runtimes:**
- ONNX Runtime - local inference runtime for ImageBind and optional Gemma ORT execution in `CMakeLists.txt`, `src/encoder/imagebind.cpp`, and `src/encoder/embeddinggemma.cpp`
  - SDK/Client: vendored ONNX Runtime source in `third_party/onnxruntime/` plus imported `onnxruntime` shared library
  - Auth: Not applicable
- onnxruntime-genai - local Phi-4 multimodal extraction/summarization runtime in `CMakeLists.txt`, `src/extractor/phi4_extractor.cpp`, and `include/cortext/summarizer/phi4_summarizer.hpp`
  - SDK/Client: vendored `third_party/onnxruntime-genai/`
  - Auth: Not applicable
- LiteRT-LM - local Gemma extraction/summarization runtime in `CMakeLists.txt`, `src/extractor/gemma_extractor.cpp`, and `src/summarizer/gemma_summarizer.cpp`
  - SDK/Client: vendored `third_party/litert-lm/`, installed into the build tree by `cmake/litert_install.cmake`
  - Auth: Not applicable
- llama.cpp - local GGUF runtime for Liquid/LFM2 deep-consolidation and optional EmbeddingGemma GGUF execution in `CMakeLists.txt`, `src/deep_llm/deep_llm_factory.cpp`, and `src/encoder/embeddinggemma.cpp`
  - SDK/Client: system `llama`, `ggml`, and `ggml-base` libraries discovered by `find_library()` in `CMakeLists.txt`
  - Auth: Not applicable
- sherpa-onnx - local offline ASR/TTS and diarization runtime in `CMakeLists.txt`, `src/audio/sherpa_onnx.cpp`, and `examples/chat/voice_session.mm`
  - SDK/Client: vendored `third_party/sherpa-onnx/`
  - Auth: Not applicable

**Remote LLM APIs:**
- OpenAI-compatible Chat Completions API - used by the desktop chat example in `examples/chat/main.cpp` and `examples/chat/streaming_client.cpp`
  - SDK/Client: `libcurl` plus header-only `openai-cpp` declared in `examples/chat/CMakeLists.txt`
  - Auth: `OPENAI_API_KEY`; optional `OPENAI_BASE_URL`, `OPENAI_ORGANIZATION`, and `OPENAI_MODEL`
- OpenAI-compatible Chat Completions API - used for offline label-data generation in `scripts/generate_openai_label_data.py`
  - SDK/Client: Python `urllib.request`
  - Auth: `OPENAI_API_KEY`; optional `OPENAI_BASE_URL` and `OPENAI_LABEL_MODEL`

**Dataset and model distribution services:**
- Hugging Face Hub - used to download EmbeddingGemma model assets in `scripts/download_embeddinggemma.sh` and `scripts/download_embeddinggemma_onnx.sh`
  - SDK/Client: `curl`
  - Auth: `HF_TOKEN` for `litert-community/embeddinggemma-300m`; no token required in `scripts/download_embeddinggemma_onnx.sh`
- Hugging Face datasets / hub APIs - used by offline research utilities in `tools/label_bank_builder/build_label_bank.py`, `tools/centroid_vectors/build_text_emotion_centroids.py`, and `scripts/prepare_ubuntu_dialogues.py`
  - SDK/Client: Python `datasets` and `huggingface_hub`
  - Auth: None in code; uses public datasets by default
- Raw HTTP dataset downloads - used by dataset preparation scripts such as `scripts/download_taskmaster.py`, `scripts/download_goemotions.py`, `scripts/download_meld.py`, and `scripts/download_personachat.py`
  - SDK/Client: Python `urllib.request`
  - Auth: None detected

**Observability endpoints:**
- OpenTelemetry OTLP gRPC - optional exporter path for the chat example in `examples/chat/CMakeLists.txt` and `examples/chat/main.cpp`
  - SDK/Client: `opentelemetry-cpp` OTLP gRPC exporters
  - Auth: standard OTEL headers/env if configured; `examples/chat/main.cpp` checks `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS`
- OpenTelemetry OTLP file exporter - local JSONL trace/log output in `examples/otel_sqlite_smoketest/main.cpp`
  - SDK/Client: `opentelemetry-cpp` file exporters
  - Auth: Not applicable

**Platform integrations:**
- Apple CoreML provider - ONNX Runtime CoreML execution path is enabled on Apple platforms in `CMakeLists.txt` and selected by `src/encoder/imagebind.cpp`
  - SDK/Client: ONNX Runtime CoreML provider
  - Auth: Not applicable
- macOS AVFoundation audio stack - microphone capture and playback for the chat example in `examples/chat/voice_session.mm`
  - SDK/Client: `AVFoundation`, `Cocoa`, `Foundation`, `IOKit`, and `CoreVideo` linked in `examples/chat/CMakeLists.txt`
  - Auth: Not applicable
- WebAssembly / OPFS-capable build path - optional WASM target exists in `CMakePresets.json`, `cmake/EmscriptenToolchain.cmake`, and `src/wasm/auto_extensions.cpp`
  - SDK/Client: Emscripten plus embedded SQLite/objstore sources
  - Auth: Not applicable

## Data Storage

**Databases:**
- SQLite - primary memory/state store accessed through `include/cortext/store/sqlite_store.hpp`, `src/store.cpp`, and `src/store/schema.cpp`
  - Connection: passed as `db_path` to `cortext::Cortext::Create()` in `include/cortext/cortext.hpp`; the chat example maps this from `CORTEXT_CHAT_DB` in `examples/chat/main.cpp`
  - Client: custom `SQLiteStore` wrapper in `include/cortext/store/sqlite_store.hpp`
- sqlite-vec virtual table - 256-dim embedding index created in `src/store/schema.cpp`
  - Connection: same SQLite database as the core store
  - Client: embedded or dynamically loaded via `src/store/extension_loader.cpp`

**File Storage:**
- sqlite-objstore virtual table - blob/object payload storage is created in `src/store/schema.cpp` and accessed in `src/operations/memory_storage.cpp` and `src/cortext.cpp`
- Local filesystem model/data roots - runtime assets live under `models/`, datasets under `data/`, and object-store artifacts are present under `objects/`

**Caching:**
- Hugging Face cache directory is used by research tooling under `data/hf_cache/` and referenced by `tools/centroid_vectors/build_text_emotion_centroids.py`
- LiteRT/XNNPACK model caches are present under `models/gemma3n-e2b-litert/`
- No dedicated network cache service is detected

## Authentication & Identity

**Auth Provider:**
- No repo-wide authentication or identity system is implemented for the core library
  - Implementation: the core engine is local-only and relies on caller-provided filesystem paths and local model assets
- OpenAI bearer auth is used only by the chat example and labeling scripts in `examples/chat/main.cpp` and `scripts/generate_openai_label_data.py`
  - Implementation: API key passed through HTTP `Authorization: Bearer ...`
- Hugging Face token auth is used only by model download helper scripts such as `scripts/download_embeddinggemma.sh`
  - Implementation: `HF_TOKEN` is injected into `curl` request headers

## Monitoring & Observability

**Error Tracking:**
- None detected as a third-party SaaS service
- Errors, traces, metrics, and logs are emitted through `opentelemetry-cpp` wrappers in `src/telemetry/telemetry.cpp`

**Logs:**
- Core library logs through OpenTelemetry log API in `src/telemetry/telemetry.cpp`
- The chat example writes human-readable logs to `examples/chat/logs.txt` and can also export OTLP logs in `examples/chat/main.cpp`
- The smoke test writes OTLP JSONL output under `CORTEXT_OTEL_OUT_DIR` in `examples/otel_sqlite_smoketest/main.cpp`

## CI/CD & Deployment

**Hosting:**
- Not detected

**CI Pipeline:**
- GitHub Actions build/test pipeline in `.github/workflows/build.yml`
- CI installs `cmake`, `build-essential`, `pkg-config`, and `libsqlite3-dev`, then runs configure/build/ctest on `ubuntu-latest`

## Environment Configuration

**Required env vars:**
- `OPENAI_API_KEY` for `examples/chat/main.cpp` and `scripts/generate_openai_label_data.py`
- `HF_TOKEN` for `scripts/download_embeddinggemma.sh`
- `CORTEXT_DEEP_LLM_BACKEND`, `CORTEXT_LFM2_SUMMARIZER_MODEL`, and `CORTEXT_LFM2_EXTRACT_MODEL` for deep-backend overrides in `src/deep_llm/deep_llm_factory.cpp`
- `CORTEXT_EMBEDDINGGEMMA_BACKEND` for encoder backend selection in `src/encoder/embeddinggemma.cpp` and `src/encoder/text_encoder_factory.hpp`
- `SQLITE_VEC_PATH` for dynamic sqlite-vec loading in `src/store/extension_loader.cpp`
- `CORTEXT_MODELS_DIR`, `CORTEXT_CHAT_DB`, `CORTEXT_CHAT_SETTINGS`, `CORTEXT_FOCUS`, `CORTEXT_SENSITIVITY`, `CORTEXT_STABILITY`, and `CORTEXT_CHAT_STREAM_INTERRUPTS` for the desktop chat example in `examples/chat/main.cpp`
- `CORTEXT_OTEL_OUT_DIR` for file-export smoke tests in `examples/otel_sqlite_smoketest/main.cpp`
- `OTEL_EXPORTER_OTLP_ENDPOINT` and `OTEL_EXPORTER_OTLP_HEADERS` for optional OTLP gRPC export in `examples/chat/main.cpp`

**Secrets location:**
- Process environment variables are the primary secret source
- `scripts/generate_openai_label_data.py` can also load a user-supplied shell env file via `--env-file`
- No committed secret store or tracked `.env` file was detected during this scan

## Webhooks & Callbacks

**Incoming:**
- None

**Outgoing:**
- `POST {OPENAI_BASE_URL}/chat/completions` from `examples/chat/streaming_client.cpp`
- `POST {OPENAI_BASE_URL}/chat/completions` from `scripts/generate_openai_label_data.py`
- OTLP gRPC export to the endpoint configured by standard OTEL env vars from `examples/chat/main.cpp`
- Hugging Face/model and dataset HTTP downloads from `scripts/download_embeddinggemma.sh`, `scripts/download_embeddinggemma_onnx.sh`, `tools/label_bank_builder/build_label_bank.py`, and dataset prep scripts under `scripts/`

---

*Integration audit: 2026-04-07*
