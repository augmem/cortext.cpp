# @augmem/cortext

Node.js and TypeScript bindings for Cortext, the on-device memory engine behind
augmem. Use it to persist user or application context, retrieve relevant
memories on later turns, and inject those memories into an agent or LLM prompt.

The npm package ships with prebuilt N-API addons and the default AIST GGUF
embedding asset, so normal installs do not need a compiler or a separate model
download.

## Install

```bash
npm install @augmem/cortext
```

Supported package targets:

- `linux-x64`
- `linux-arm64`
- `darwin-x64`
- `darwin-arm64`
- `win32-x64`
- `win32-arm64`

Node.js 18 or newer is required.

## Quickstart

```ts
import { Cortext, version } from "@augmem/cortext";

console.log(version());

const memory = new Cortext(
  {
    focus: 0.55,
    sensitivity: 0.5,
    stability: 0.65,
  },
  "memory.sqlite"
);

try {
  memory.processText(
    "The garage door code is 8841.",
    "user/profile",
    { includeEmbedding: false }
  );

  const ctx = memory.processText(
    "We are leaving soon. What should I remember about the garage?",
    "chat/assistant",
    { includeEmbedding: false }
  );

  for (const item of ctx.retrieved_memory ?? []) {
    console.log(item.text, item.rel);
  }

  if (ctx.consolidation_recommended) {
    memory.consolidate();
  }
} finally {
  memory.flush();
}
```

Use `new Cortext(":memory:")` for a temporary in-memory engine. Use a file path
when memory should survive process restarts.

CommonJS works too:

```js
const { Cortext } = require("@augmem/cortext");
const memory = new Cortext("memory.sqlite");
```

## Using It With an LLM

The usual server-side loop is:

1. Store durable context with `processText`.
2. On later turns, read `ctx.retrieved_memory`.
3. Add the top memory snippets to your model prompt.
4. Call `consolidate()` when Cortext recommends it.

```ts
const ctx = memory.processText(userMessage, `conversation/${conversationId}`, {
  includeEmbedding: false,
});

const memories = (ctx.retrieved_memory ?? [])
  .slice(0, 6)
  .map((m) => `- ${m.text}`)
  .join("\n");

const prompt = `Relevant memory:
${memories || "- none"}

User:
${userMessage}`;
```

Important: `processText` is durable in the current JavaScript API. It can
retrieve memory, but it also writes the input signal into the configured store.
Use it for turns you are willing to remember. Use `embedText` for embedding-only
work that must not mutate memory.

## API Shape

```ts
new Cortext(config?: CortextConfig | null, dbPath?: string | null);
new Cortext(dbPath: string);
```

Core methods:

- `processText(text, sourceId, options?)`
- `processAudio(pcm, sourceId, options?)`
- `processImage(data, width, height, channels, sourceId, options?)`
- `processAudioWithMedia(...)`
- `processImageWithMedia(...)`
- `embedText(text)`
- `embedAudio(pcm)`
- `embedImage(data, width, height, channels)`
- `consolidate()`
- `flush()`
- `reset()`

Every `process*` and `embed*` method also has a `*Json` variant that returns
the native JSON string.

`ProcessOptions`:

```ts
interface ProcessOptions {
  includeEmbedding?: boolean;
  omitEmbedding?: boolean;
}
```

Use `{ includeEmbedding: false }` for prompt-injection workflows; returning the
embedding in every context packet is usually unnecessary.

## Returned Context

`processText`, `processAudio`, and `processImage` return parsed objects from
the native context packet. The most useful fields are:

- `retrieved_memory`: long-term memories selected for this signal.
- `working_memory`: short-term active context.
- `should_interrupt`, `interrupt_aborted`, `at_boundary`: realtime behavior
  flags.
- `consolidation_recommended`, `consolidation_required`: maintenance hints.
- `output`: scores, storage decisions, filter status, and operation timings.
- `encode_ms`, `process_ms`, `hydrate_ms`, `total_ms`: latency breakdown.
- `embedding`, `embedding_dimension`: present only when requested.

Memory entries include `text`, `source_id`, `timestamp`, `modality`,
`mimetype`, `rel`, usage counts, metric scores, and soft-anchor metadata.

## Audio and Image

Audio input is `Float32Array` containing 16 kHz mono PCM:

```ts
const pcm = new Float32Array(16000);
const ctx = memory.processAudio(pcm, "mic/main", { includeEmbedding: false });
```

Image input is row-major RGB or RGBA bytes:

```ts
const rgb = new Uint8Array(64 * 64 * 3);
const ctx = memory.processImage(rgb, 64, 64, 3, "camera/main");
```

Use media variants when you want to store original bytes next to the canonical
signal:

```ts
const media = { data: jpegBytes, mimetype: "image/jpeg" };
const ctx = memory.processImageWithMedia(
  rgb,
  64,
  64,
  3,
  "camera/main",
  media
);
```

## Runtime Assets

Normal installs include:

- A prebuilt `cortext.node` for the current supported platform.
- The default AIST GGUF embedding asset and tokenizer vocabulary.

Override paths only for local development or custom assets:

- `CORTEXT_NODE_ADDON_PATH=/path/to/cortext.node`
- `CORTEXT_AIST_MODEL_PATH=/path/to/AIST-87M_q8_0.gguf`

If the package cannot load a native addon, the thrown error lists every path it
tried and the supported target tags.

## Build From Source

From the repository root:

```bash
cd bindings/javascript
npm run build
```

Build the release tarball with all prebuilt addons and the bundled AIST asset:

```bash
npm run build:package -- --zig /path/to/zig
```

The generated tarball is written to `bindings/javascript/dist/`.

For browser demos, use the separate Emscripten wrapper under `bindings/wasm`
and serve `examples/web/` from the repository root.
