# @augmem/cortext

Node.js and TypeScript bindings for Cortext, the on-device memory engine behind
augmem. Use it to persist text, audio, and image signals, then retrieve relevant
context for agents, apps, and LLM prompts.

The npm package ships cross-platform prebuilt N-API addons for supported
desktop/server platforms. It does not embed the large AIST GGUF model. On first
engine creation, the wrapper uses `CORTEXT_AIST_MODEL_PATH` if set, then any
bundled/local model already present, otherwise it downloads the default AIST
GGUF model into the user cache and verifies its checksum before use.

## Install

```bash
npm install @augmem/cortext
```

```ts
import { Cortext, version } from "@augmem/cortext";

console.log(version());
```

Node.js 18+ is required. Release packages target `linux-x64`, `linux-arm64`,
`darwin-x64`, `darwin-arm64`, `win32-x64`, and `win32-arm64` where prebuilds
are published.

## Quickstart

<!-- quickstart retention examples matching bindings/javascript/index.d.ts -->

```ts
import { Cortext } from "@augmem/cortext";

const memory = new Cortext(
  {
    focus: 0.55,
    sensitivity: 0.5,
    stability: 0.65,
  },
  "memory.sqlite"
);

try {
  memory.processText("The garage door code is 8841.", "user/profile", {
    includeEmbedding: false,
    retention: "durable",
  });

  const ctx = memory.processText(
    "We are leaving soon. What should I remember about the garage?",
    "chat/assistant",
    { includeEmbedding: false, retention: "ephemeral" }
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

Use `new Cortext(":memory:")` for a temporary engine. Use a file path when
memories should survive process restarts. CommonJS is also supported:

```js
const { Cortext } = require("@augmem/cortext");
const memory = new Cortext("memory.sqlite");
```

## Chat Completions Memory Loop

The normal server-side loop is simple:

1. Call `processText` for turns or observations you are willing to remember.
2. On later turns, read `ctx.retrieved_memory`.
3. Pass those snippets as context messages to Chat Completions.
4. Call `consolidate()` when Cortext recommends it.

Install the OpenAI SDK and set `OPENAI_API_KEY`:

```bash
npm install openai
```

```ts
import OpenAI from "openai";
import { Cortext } from "@augmem/cortext";

const client = new OpenAI();
const memory = new Cortext("memory.sqlite");

export async function answer(conversationId: string, userMessage: string) {
  const ctx = memory.processText(
    userMessage,
    `conversation/${conversationId}`,
    { includeEmbedding: false, retention: "durable" }
  );

  const memories = (ctx.retrieved_memory ?? [])
    .slice(0, 6)
    .map((m) => m.text)
    .filter(Boolean)
    .map((text) => `- ${text}`)
    .join("\n");

  const completion = await client.chat.completions.create({
    model: process.env.OPENAI_MODEL ?? "gpt-5-mini",
    messages: [
      {
        role: "developer",
        content:
          "Use the supplied Cortext memories when they are relevant. Ignore them when they are not relevant.",
      },
      {
        role: "developer",
        content: `Cortext retrieved memory:\n${memories || "- none"}`,
      },
      { role: "user", content: userMessage },
    ],
  });

  if (ctx.consolidation_recommended) {
    memory.consolidate();
  }

  return completion.choices[0]?.message?.content ?? "";
}
```

Retention defaults to `"natural"`, so episode algorithms decide whether a
signal is stored. Use `retention: "durable"` to commit a turn immediately and
`retention: "ephemeral"` for retrieve-without-store queries. Use `embedText`,
`embedAudio`, or `embedImage` for embedding-only work.

## Returned Context

`processText`, `processAudio`, and `processImage` return parsed objects from
the native context packet. Common fields:

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
assembly, pass `{ includeEmbedding: false }`; embeddings are large and rarely
needed in the returned packet.

## Audio and Image

Audio input is a `Float32Array` containing 16 kHz mono PCM:

```ts
const pcm = new Float32Array(16000);
const ctx = memory.processAudio(pcm, "mic/main", { includeEmbedding: false });
```

Image input is row-major RGB or RGBA bytes:

```ts
const rgb = new Uint8Array(64 * 64 * 3);
const ctx = memory.processImage(rgb, 64, 64, 3, "camera/main", {
  includeEmbedding: false,
});
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
  media,
  { includeEmbedding: false }
);
```

## Runtime Assets and Model Cache

The package contains the native `cortext.node` addon for supported platforms.
The AIST GGUF model is intentionally not embedded in registry packages because
of npm size constraints.

Model resolution on engine creation:

1. `CORTEXT_AIST_MODEL_PATH=/path/to/AIST-87M_q8_0.gguf`
2. A bundled or checkout-local model, if one exists.
3. Download the default AIST GGUF model to the user cache and verify its
   checksum before loading it.

The first run may need network access and enough cache space for the model
(roughly 135-142 MiB, depending on quantization). Later runs reuse the verified
cache. Set `CORTEXT_MODEL_CACHE_DIR` to control the cache root, or set
`CORTEXT_AIST_MODEL_PATH` for offline deployments and pinned model files.

Native addon override:

- `CORTEXT_NODE_ADDON_PATH=/path/to/cortext.node`

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

Every `process*`, `embed*`, and `consolidate` method also has a `*Json` variant
that returns the raw native JSON string.

```ts
interface ProcessOptions {
  includeEmbedding?: boolean;
  omitEmbedding?: boolean;
}
```

## Troubleshooting

- First engine creation is slow: the model may be downloading and verifying.
- Model download fails: check network access, cache write permission, or set
  `CORTEXT_AIST_MODEL_PATH` to a local GGUF file.
- Checksum failure: remove the partially downloaded cache file and retry.
- Native addon cannot be loaded: install a package for a supported target or
  set `CORTEXT_NODE_ADDON_PATH`.
- Very large context objects: call `processText(..., { includeEmbedding: false })`.
- Need a clean temporary run: use `new Cortext(":memory:")`.
- Native failure details: call `lastError()` immediately after the exception.

## Build From Source

From the repository root:

```bash
cd bindings/javascript
npm run build
export CORTEXT_AIST_MODEL_PATH="$PWD/../../models/AIST-87M-GGUF/AIST-87M_q8_0.gguf"
```

Build a release tarball with native addons:

```bash
npm run build:package -- --zig /path/to/zig --skip-models
```

The tarball is written to `bindings/javascript/dist/`. Registry packages should
include native addons but leave the large AIST model to the runtime cache path
above.
