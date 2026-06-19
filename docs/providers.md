# Inference providers

Status: phase 1 scaffolding (this branch). Design approved 2026-06-10.

## Why

The deep-LLM pipeline (summarizer, extractor) and the text encoder are bound
at build time (`CORTEXT_DISABLE_LITERT`, `CORTEXT_DISABLE_OGA`), discovered by
hardcoded path search (`deep_llm_factory`), and always run in-process. That
coupling has real costs:

- LiteRT has no usable GPU path on Linux servers (CPU-only litertlm sections,
  no CUDA accelerator), so evals on GPU boxes run the summarizer on CPU while
  the GPUs sit idle.
- On shared-GPU hosts (the Mac), in-process summarization and external judge
  traffic contend for the same device — the `GemmaSummarizer: Decode failed`
  crash class.
- Swapping a backend (e.g. judging summarizer quality across models) means a
  rebuild, not a config change.

The store and objstore already solved this shape of problem with a backend
registry resolved from runtime config. Providers apply the same pattern to
model inference.

## Shape

One new seam: `cortext::providers::InferenceProvider` — a transport with
`Health()`, `Capabilities()`, `Identity()`, `Generate(request)`, and
`GenerateBatch(requests)`. `GenerateBatch` is first-class because most
production engines can batch independent generations; the base implementation
falls back to sequential `Generate` calls so existing providers preserve
behavior. The existing `Summarizer`/`Extractor` interfaces stay; adapters
(`ProviderSummarizer`, `ProviderExtractor`) present any provider through
them. Both adapters now expose first-class batching at their legacy interface
level, so the operation pipeline does not need to know which transport is
serving the request.

Providers are addressed by URI and resolved through a scheme registry
(`RegisterProviderFactory` / `ResolveProvider`):

```
litert://models/gemma4-e2b-litert/gemma-4-E2B-it.litertlm   in-process LiteRT
gguf://models/LFM2.5-1.2B-Instruct-GGUF                     in-process llama.cpp
ollama://127.0.0.1:11435/gemma4:e4b                         Ollama server
openai://127.0.0.1:8000/v1/gemma4-e2b                       OpenAI-compatible
```

Configuration mechanisms live in the application layer, not the library:
the library accepts instances (`Cortext::InferenceOverrides`, mirroring the
Store/ObjectStore injection pattern), and applications turn their config
strings into instances via the registry. The benchmark CLI exposes this as:

```
cortext_chat_replay_live_run \
  --summarizer-provider ollama://127.0.0.1:11435/gemma4:e4b \
  --extractor-provider ollama://127.0.0.1:11435/gemma4:e4b
```

When both roles are injected, local model discovery is skipped entirely, so
remote-only deployments need no local weights.

There is no separate label-provider role. Labels, relations, and facts all flow
through the extractor adapter: `ConsolidationSummarize` queues extraction
requests, and `ProcessExtractionResults` uses the configured extractor to
produce the final labels and structured graph updates. A run that sets only
`--summarizer-provider` still uses the local default extractor for labeling.

Extraction batching is first-class at the extractor level:
`Extractor::ExtractBatchFromTexts` accepts independent text items with stable
ids and returns results in input order. Extractors that do not support native
batching inherit the sequential default. `ProviderExtractor` overrides the
method and forwards independent `GenerateRequest`s to
`InferenceProvider::GenerateBatch`, so provider engines can implement true
runtime batching without changing consolidation code.

Summarization batching follows the same contract:
`Summarizer::SummarizeTextBatches` accepts independent groups of source texts
with stable ids and a per-item word cap. The default implementation loops
through `SummarizeTextsLimited`; `ProviderSummarizer` forwards independent
summary requests to `InferenceProvider::GenerateBatch`. Deep consolidation
uses this path for independent cluster summaries and flushes them in bounded
batches controlled by `CORTEXT_CONSOLIDATION_SUMMARY_BATCH_SIZE`.

The current Ollama transport has no native batch endpoint, but it can run
independent requests concurrently when explicitly enabled:

```bash
CORTEXT_OLLAMA_BATCH_PARALLELISM=2 \
CORTEXT_CONSOLIDATION_SUMMARY_BATCH_SIZE=8 \
cortext_chat_replay_live_run \
  --summarizer-provider ollama://127.0.0.1:11435/gemma4:e2b \
  --extractor-provider ollama://127.0.0.1:11435/gemma4:e2b
```

The default is `1` to preserve the old deterministic request shape. On the
late six-year replay slice, parallelism `2` was a modest win; prompt-packing
multiple extraction items into one large request was a regression and should
not be treated as engine batching.

OpenAI-compatible servers use the same provider contract:

```bash
CORTEXT_OPENAI_BATCH_PARALLELISM=4 \
CORTEXT_CONSOLIDATION_SUMMARY_BATCH_SIZE=8 \
cortext_chat_replay_live_run \
  --summarizer-provider openai://127.0.0.1:8000/v1/gemma4-e2b \
  --extractor-provider openai://127.0.0.1:8000/v1/gemma4-e2b
```

The `openai://` provider is intended for local OpenAI-compatible engines such
as vLLM or llama.cpp servers. It sends text-only `/chat/completions` requests,
uses server-side JSON schema response formats for extraction, and exposes
client-side parallelism through `CORTEXT_OPENAI_BATCH_PARALLELISM` until the
server's own scheduler performs the real continuous batching. It does not yet
implement TLS, API keys, image/audio parts, or encoder embeddings; use it for
local consolidation summarization/extraction, not hosted public APIs.

### Capabilities are checked at resolve time

`Capabilities` declares modalities (text/image/audio), constrained-decoding
support, and embedding dims. `ResolveProvider` refuses unfit pairings up
front — an extractor without any constraint path, an encoder without
embedding dims — instead of failing mid-run.

Constraint support is a three-way declaration:

| Level          | Mechanism                              | Examples            |
|----------------|----------------------------------------|---------------------|
| NativeGrammar  | in-process grammar-constrained decode  | LiteRT, llama.cpp   |
| ServerSchema   | schema enforced server-side            | Ollama `format`     |
| None           | adapter validates + retries (bounded)  | bare HTTP endpoints |

All transports parse extraction output through the same
`ParseExtractionResponse` (now externally linked from `gemma_extractor.cpp`),
so output handling cannot drift between backends.

### Eval fidelity

`ProviderIdentity` (scheme, endpoint, model, detail) flows into
`DeepLlmSelection::backend_name`/model paths, which the benchmark already
records into `summary.json`. A judge artifact therefore always states which
stack produced it. In-process remains the default everywhere — the on-device
claim is unaffected unless a run explicitly opts into a server provider.

## Phases

1. `provider.hpp`, `registry.hpp/.cpp`, `adapters.hpp/.cpp`, env-override hook
   in `CreateDeepLlmSelection`.
2. `OllamaProvider` (HTTP, `/api/chat`, `format` for schema, base64 media) and
   `OpenAIProvider` (HTTP, `/chat/completions`, JSON-schema response format for
   text-only local servers) register `ollama://` and `openai://`.
3. Wrap LiteRT and llama.cpp engines as `litert://` / `gguf://` providers;
   `deep_llm_factory` resolves everything through the registry; retire the
   hardcoded search-path chain into the default URI set.
4. Encoder role joins the registry (AIST-87M in-process default); retire
   `CORTEXT_DISABLE_*` build flags where providers cover their cases.
