# Technology Stack

**Project:** cortext
**Research Scope:** Stack choices and runtime direction for a brownfield local-first multimodal memory engine
**Researched:** 2026-04-07
**Overall confidence:** MEDIUM-HIGH

## Recommendation

The standard 2026 direction for this project is not to pick one universal AI runtime. It is to keep the C++/SQLite engine intact and assign one runtime per job:

- **SQLite** remains the durable memory substrate.
- **ONNX Runtime** becomes the default portable inference runtime for encoders, speech models, and hardware-accelerated on-device inference.
- **sherpa-onnx** becomes the default realtime voice stack.
- **llama.cpp** becomes the default GGUF text-generation, reranking, and lightweight local reasoning runtime.
- **LiteRT / LiteRT-LM** stays available, but only as a targeted mobile/NPU path for Gemma and Google-AI-Edge-optimized deployments.

That direction fits the existing cortext architecture much better than a rewrite or a single-runtime bet. The repo already has the right abstraction boundaries: `Store`, `Encoder`, `Extractor`, `Summarizer`, and backend factories. The problem is overlap and unclear defaults, not lack of runtime optionality.

## Recommended Stack

| Layer | Recommendation | Keep / Replace | Why | Confidence |
|------|----------------|----------------|-----|------------|
| Core engine | **C++20 + CMake + stable C API** | Keep | Correct for a local-first embeddable engine; aligns with current public API and bindings constraints | HIGH |
| Durable store | **SQLite + WAL + migrations + sqlite-objstore** | Keep | Still the right local-first source of truth for memories, metadata, and object payloads | HIGH |
| Dense vector retrieval | **sqlite-vec** for now | Keep, but fence scope | Correct for embedded/local retrieval and low-ops shipping; current upstream is still pre-v1, so treat it as an embedded component, not a fast-moving dependency | MEDIUM |
| Lexical retrieval | **SQLite FTS5/BM25 alongside dense retrieval** | Add | Standard 2026 memory retrieval is hybrid, not vector-only; improves recall and explainability for human and agent workflows | HIGH |
| Retrieval ranking | **C++ hybrid reranker with explicit score decomposition** | Add | Memory surfacing needs explainability; ranking should expose dense, lexical, recency, graph, and user-state contributions separately | HIGH |
| Observability | **OpenTelemetry C++ + OTLP + local trace tables in SQLite** | Keep and expand | OTel is the right telemetry backbone; local-first engines also need queryable, persisted retrieval traces for offline debugging and paper-grade analysis | HIGH |
| Portable inference | **ONNX Runtime** | Keep as primary | Best portable native runtime here because it supports multiple execution providers and on-device optimization paths across vendors | HIGH |
| GGUF text inference | **llama.cpp / libllama** | Keep as primary for GGUF | Strongest native C/C++ path for quantized local LLMs on CPU/Metal/CUDA/Vulkan-class hardware | HIGH |
| Mobile / NPU path | **LiteRT / LiteRT-LM** | Keep, but narrow | Valuable for Gemma and compiled mobile/NPU deployments; should not be the default runtime for the whole engine | MEDIUM |
| ONNX chat loop runtime | **ONNX Runtime GenAI** only where needed | Keep selectively | Useful for interactive ONNX chat/generation loops, but still preview/evolving; not stable enough to be the sole runtime contract | MEDIUM |
| Realtime voice | **sherpa-onnx** | Promote to default | Best fit because it already covers local streaming ASR, VAD, diarization, speaker ID, source separation, and TTS in one family | HIGH |
| Offline / fallback transcription | **whisper.cpp** | Keep, but demote | Excellent offline ASR fallback and Apple-first implementation, but too narrow to remain the default speech stack for the engine | HIGH |

## Runtime Ownership Matrix

This is the concrete runtime direction I would standardize on.

| Capability | Default runtime | Secondary / fallback | Notes | Confidence |
|-----------|------------------|----------------------|-------|------------|
| Text embeddings | ONNX Runtime | llama.cpp GGUF embeddings | Prefer ONNX for portable optimized embedding models; use GGUF only when packaging simplicity matters more | HIGH |
| Image / audio / multimodal encoders | ONNX Runtime | LiteRT for device-specific deployments | ONNX remains the cleanest portable path for multimodal encoders in this engine | HIGH |
| Streaming ASR | sherpa-onnx | whisper.cpp only for non-realtime fallback | sherpa-onnx is the better engine default because it is built around local streaming workflows | HIGH |
| VAD / endpointing | sherpa-onnx | separate VAD only if profiling proves need | Avoid a fragmented speech stack unless a benchmark justifies it | HIGH |
| Speaker diarization / ID | sherpa-onnx | custom stack only if quality gap is proven | Keep speaker features inside one speech runtime family | HIGH |
| TTS for local agent voice | sherpa-onnx | platform-native TTS outside engine boundary | Engine should own portable local TTS only when needed for memory capture/interaction loops | MEDIUM |
| Transcript cleanup / summarization / extraction | llama.cpp | ONNX Runtime GenAI or LiteRT-LM for specific models | GGUF is the most flexible local text-runtime path for brownfield iteration | HIGH |
| Gemma-specific mobile/NPU deployments | LiteRT-LM | ONNX Runtime if model export is available | Use LiteRT where it wins on hardware path, not as the general default | MEDIUM |
| Reranking / explanation synthesis | llama.cpp small model or C++ heuristic reranker | ONNX small reranker | Keep reranking cheap and local; avoid routing every retrieval through a large generator | MEDIUM |

## What To Keep

- **Keep the C++ engine and public API surface.** The repo already has the right embeddable boundary for augmem.ai and downstream bindings.
- **Keep SQLite as the single durable store.** Do not split memory truth across a vector DB, analytics DB, and object store unless scale proves it necessary.
- **Keep ONNX Runtime in the core.** It is still the best portable inference substrate for multimodal encoders and hardware-specific execution-provider routing.
- **Keep llama.cpp.** It should remain the default for quantized local LLM work that benefits from GGUF packaging and first-class Apple/desktop support.
- **Keep OpenTelemetry.** It should graduate from generic tracing to retrieval-native observability with stable event names and score fields.
- **Keep LiteRT available.** It is useful, but as a targeted accelerator path.

## What To Replace Or Narrow

- **Replace whisper.cpp as the default speech path with sherpa-onnx.**
  whisper.cpp is an excellent local ASR implementation, but sherpa-onnx is a better default engine stack because it covers streaming ASR, VAD, diarization, speaker ID, source separation, hotwords, WebSocket transport, and TTS in one local family.

- **Replace overlapping “any runtime can do anything” behavior with explicit ownership.**
  The current engine should stop letting Gemma/ORT/GGUF/speech backends drift into partial overlap. Pick defaults by capability and retain alternatives as opt-in backends.

- **Replace opaque retrieval scores with structured retrieval traces.**
  Every retrieval should persist candidate IDs, pre-filter counts, dense score, lexical score, graph score, recency score, final blended score, threshold decisions, and why the winner set changed.

- **Replace vector-only retrieval assumptions with hybrid retrieval.**
  For human and LLM workflows, memory recall quality improves when dense similarity, lexical matches, temporal priors, and graph relations are blended rather than treated as separate ad hoc passes.

- **Narrow LiteRT-LM to the deployments where it is structurally advantaged.**
  Keep it for Gemma and compiled mobile/NPU targets. Do not make it the core runtime contract for desktop/server-like local workflows inside cortext.

## What Not To Use

- **Do not add a remote-first vector database** such as Qdrant, Weaviate, Pinecone, or Milvus to the core engine path. That works against the repo’s local-first constraint and complicates deployability and offline behavior.
- **Do not standardize the engine around Ollama-style daemon integration.** It is fine as an external developer convenience, not as the engine’s internal runtime contract.
- **Do not make ONNX Runtime GenAI the only generation backend yet.** Its C/C++ path is real and useful, but the project is still explicitly in preview and the support matrix is still moving.
- **Do not keep a fragmented speech stack by default** where ASR, VAD, diarization, and TTS all come from unrelated runtimes unless benchmarks show a clear quality or latency win.
- **Do not introduce a generic app/backend stack replatform** around Python, Electron, or web-service orchestration. This milestone is about evolving the engine, not rebuilding it as an application platform.

## Concrete Direction For cortext

If I were standardizing this repo for the next milestone, I would formalize the stack like this:

1. **Engine core**
   - C++20
   - CMake
   - SQLite with WAL, FTS5, and migrations
   - sqlite-objstore
   - sqlite-vec

2. **Inference**
   - ONNX Runtime as the default runtime for encoders and portable on-device models
   - llama.cpp as the default runtime for GGUF local text generation and reranking
   - LiteRT / LiteRT-LM only for Gemma-specific or mobile/NPU-specific shipping targets
   - ONNX Runtime GenAI only where an ONNX-native interactive decoding loop is clearly worth it

3. **Voice**
   - sherpa-onnx for streaming ASR, VAD, diarization, speaker ID, and optional TTS
   - whisper.cpp retained only as offline ASR fallback, benchmarking baseline, or model-compatibility path

4. **Observability**
   - opentelemetry-cpp for spans/metrics/logs
   - OTLP export for external inspection
   - local SQLite retrieval-trace tables as first-class product data
   - optional Parquet export for experiment analysis

5. **Retrieval**
   - dense retrieval in sqlite-vec
   - lexical retrieval in FTS5
   - graph / temporal reranking in C++
   - explicit score decomposition surfaced to both humans and LLM callers

## Confidence Notes

| Decision | Confidence | Why |
|---------|------------|-----|
| Keep C++/SQLite core | HIGH | Fully aligned with repo constraints and still standard for embeddable local-first engines |
| ONNX Runtime as primary portable inference runtime | HIGH | Strong official support for execution providers, quantization, I/O binding, and on-device deployment paths |
| llama.cpp as default GGUF runtime | HIGH | Mature native local runtime with broad hardware support and minimal dependency burden |
| sherpa-onnx as default realtime voice stack | HIGH | Official docs show local processing plus streaming ASR, VAD, diarization, speaker ID, source separation, and TTS |
| LiteRT as targeted accelerator path, not general default | MEDIUM | Official docs show strong compiled-model and delegate support, but it is still best treated as a specialized path for this brownfield C++ engine |
| ONNX Runtime GenAI as selective, not universal | MEDIUM | Official docs and README show real utility, but preview status and evolving support matrix argue against making it the only generation contract |
| sqlite-vec as current embedded vector layer | MEDIUM | Good fit for local-first embedding storage, but upstream explicitly warns that it is still pre-v1 |

## Sources

Official sources used for this recommendation:

- OpenTelemetry C++ docs: https://opentelemetry.io/docs/languages/cpp/ [HIGH]
- ONNX Runtime execution providers overview: https://onnxruntime.ai/docs/execution-providers/ [HIGH]
- ONNX Runtime QNN EP: https://onnxruntime.ai/docs/execution-providers/QNN-ExecutionProvider.html [HIGH]
- ONNX Runtime CoreML EP: https://onnxruntime.ai/docs/execution-providers/CoreML-ExecutionProvider.html [HIGH]
- ONNX Runtime I/O Binding: https://onnxruntime.ai/docs/performance/tune-performance/iobinding.html [HIGH]
- ONNX Runtime quantization docs: https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html [HIGH]
- ONNX Runtime GenAI README: https://github.com/microsoft/onnxruntime-genai/blob/main/README.md [HIGH]
- ONNX Runtime GenAI docs: https://onnxruntime.ai/docs/genai/ [HIGH]
- ONNX Runtime GenAI migration/chat mode docs: https://onnxruntime.ai/docs/genai/howto/migrate.html [HIGH]
- sherpa-onnx docs: https://k2-fsa.github.io/sherpa/onnx/index.html [HIGH]
- whisper.cpp README: https://github.com/ggml-org/whisper.cpp/blob/master/README.md [HIGH]
- llama.cpp README: https://github.com/ggml-org/llama.cpp/blob/master/README.md [HIGH]
- LiteRT overview: https://ai.google.dev/edge/litert [HIGH]
- LiteRT build with CMake: https://ai.google.dev/edge/litert/build/cmake [HIGH]
- LiteRT delegates: https://ai.google.dev/edge/litert/performance/delegates [HIGH]
- MediaPipe LLM Inference for iOS: https://ai.google.dev/edge/mediapipe/solutions/genai/llm_inference/ios [MEDIUM]
- sqlite-vec README: https://github.com/asg017/sqlite-vec/blob/main/README.md [MEDIUM]
