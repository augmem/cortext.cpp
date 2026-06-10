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
`Health()`, `Capabilities()`, `Identity()`, and a single `Generate(request)`
primitive. The existing `Summarizer`/`Extractor` interfaces stay; adapters
(`ProviderSummarizer`, `ProviderExtractor`) present any provider through
them.

Providers are addressed by URI and resolved through a scheme registry
(`RegisterProviderFactory` / `ResolveProvider`):

```
litert://models/gemma4-e2b-litert/gemma-4-E2B-it.litertlm   in-process LiteRT
gguf://models/LFM2.5-1.2B-Instruct-GGUF                     in-process llama.cpp
ollama://127.0.0.1:11435/gemma4:e4b                         Ollama server
openai://api.example.com/v1/model                           OpenAI-compatible
```

Configuration mechanisms live in the application layer, not the library:
the library accepts instances (`Cortext::InferenceOverrides`, mirroring the
Store/ObjectStore injection pattern), and applications turn their config
strings into instances via the registry. The benchmark CLI exposes this as:

```
cortext_chat_replay_live_run --summarizer-provider ollama://127.0.0.1:11435/gemma4:e4b
                       # extractor unspecified -> local auto-discovery
```

When both roles are injected, local model discovery is skipped entirely, so
remote-only deployments need no local weights.

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

1. **(this branch)** `provider.hpp`, `registry.hpp/.cpp`, `adapters.hpp/.cpp`,
   env-override hook in `CreateDeepLlmSelection`. No registered schemes yet;
   default behavior is bit-identical to main.
2. `OllamaProvider` (HTTP, `/api/chat`, `format` for schema, base64 media) —
   registers `ollama://`. Unblocks GPU summarization on Linux eval hosts via
   the already-running Ollama instances.
3. Wrap LiteRT and llama.cpp engines as `litert://` / `gguf://` providers;
   `deep_llm_factory` resolves everything through the registry; retire the
   hardcoded search-path chain into the default URI set.
4. Encoder role joins the registry (AIST-87M in-process default); retire
   `CORTEXT_DISABLE_*` build flags where providers cover their cases.
