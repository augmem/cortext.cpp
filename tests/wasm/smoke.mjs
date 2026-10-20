// Node smoke test of the browser WebAssembly bundle through the demo glue
// (examples/web/cortext-wasm.js).
//
// Usage: node tests/wasm/smoke.mjs <dist-dir>
//   <dist-dir> holds cortext.js, cortext.wasm and cortext.data, built with
//   ./build-wasm.sh -DCORTEXT_WASM_PRELOAD_MODEL_ASSETS_DIR=<models> where
//   <models> has AIST-87M-GGUF/AIST-87M_q8_0.gguf, AIST-87M-GGUF/AIST-87M-LI.gguf
//   and mdbr-leaf-ir/vocab.txt (scripts/download_aist_model.py writes them).
//
// Covers an in-memory and a file database: store open, text, text with
// caller media, image, embedImage, standing memory, per-content timestamps,
// the evidence budget, the reference time with and without asOf, and engine
// errors surfacing as messages.

import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const dist = path.resolve(process.argv[2] ?? "build-wasm/dist/wasm");
const glue = path.resolve(
  path.dirname(fileURLToPath(import.meta.url)),
  "../../examples/web/cortext-wasm.js",
);

// The bundle targets web/worker; give it the globals it expects and serve its
// cortext.wasm / cortext.data fetches from disk.
globalThis.self ??= globalThis;
globalThis.fetch = async (url) => {
  const file = path.join(dist, String(url).split("/").pop());
  const type = file.endsWith(".wasm") ? "application/wasm" : "application/octet-stream";
  return new Response(fs.readFileSync(file), { headers: { "content-type": type } });
};

const { default: factory } = await import(pathToFileURL(path.join(dist, "cortext.js")).href);
const { CortextWasm } = await import(pathToFileURL(glue).href);
const rt = await CortextWasm.create(factory, { locateFile: (p) => `https://bundle/${p}` });
console.log("cortext", rt.version());

const textCost = (memories) => memories.reduce(
  (sum, m) => sum + m.content.reduce(
    (s, c) => s + (c.mimetype.startsWith("text/") ? c.size_bytes : 480), 0),
  0,
);

function run(dbPath) {
  const h = rt.createContext({ dbPath, focus: 0.7, sensitivity: 0.5, stability: 0.8 });
  try {
    const t0 = Date.now();
    const store = { retention: "durable", includeEmbedding: false };
    rt.processText(h, "Bailey the dog loves the beach at Half Moon Bay", "chat", store);
    const png = new Uint8Array([0x89, 0x50, 0x4e, 0x47, 1, 2, 3, 4]);
    rt.processTextWithMedia(h, "photo of Bailey at the beach", "chat",
                            { data: png, mimetype: "image/png" }, store);
    rt.processText(h, "Always answer in metric units", "chat", store);
    rt.flush(h);

    const query = { retention: "ephemeral", includeEmbedding: false };
    const all = rt.processText(h, "Bailey the dog beach", "q", query);
    assert.ok(all.retrieved_memory.length >= 2, "retrieved memories");
    assert.ok(all.standing_memory.length >= 1, "standing_memory");
    for (const m of all.retrieved_memory) {
      for (const c of m.content) {
        assert.equal(typeof c.timestamp, "number", "content[].timestamp");
        assert.ok(c.timestamp >= t0 - 1000, "content timestamp is the store time");
      }
    }
    assert.ok(all.retrieved_memory.some((m) => m.content.some((c) => c.mimetype === "image/png")),
              "caller media stored in place of the text");

    const budget = rt.processText(h, "Bailey the dog beach", "q",
                                  { ...query, evidenceBudgetBytes: 100 });
    const spent = textCost(budget.retrieved_memory) + textCost(budget.standing_memory);
    assert.ok(spent <= 100, `evidence within budget (${spent})`);
    assert.ok(budget.retrieved_memory.length < all.retrieved_memory.length, "budget trims");

    const asOf = rt.processText(h, "Bailey the dog beach", "q",
                                { ...query, referenceMs: 1000, asOf: true });
    assert.equal(asOf.retrieved_memory.length, 0, "asOf hides memories stored later");
    const near = rt.processText(h, "Bailey the dog beach", "q", { ...query, referenceMs: 1000 });
    assert.ok(near.retrieved_memory.length >= 1, "reference time alone hides nothing");

    const pixels = new Uint8Array(8 * 8 * 3).fill(128);
    assert.ok(Array.isArray(rt.processImage(h, pixels, 8, 8, 3, "chat", query).retrieved_memory));
    const emb = rt.embedImage(h, pixels, 8, 8, 3);
    assert.ok(emb.dimension > 0, "embedImage");

    // Invalid options are rejected before any allocation.
    assert.throws(() => rt.processText(h, "x", "q", { evidenceBudgetBytes: -1 }), RangeError);
    assert.throws(() => rt.processText(h, "x", "q", { referenceMs: 1.5 }), RangeError);
    console.log(`${dbPath}: retrieved ${all.retrieved_memory.length}, standing `
                + `${all.standing_memory.length}, budget ${spent}/100 kept `
                + `${budget.retrieved_memory.length}, asOf 0, reference `
                + `${near.retrieved_memory.length}, embedding ${emb.dimension}`);
  } finally {
    rt.freeContext(h);
  }
}

run(":memory:");
run("/tmp/cortext-smoke.db");

// An engine exception surfaces as its message through lastError(), not as a
// raw exception pointer: the browser preset builds without audio.
{
  const h = rt.createContext({ dbPath: ":memory:" });
  try {
    assert.throws(() => rt.processAudio(h, new Float32Array(1600), "mic"),
                  (err) => err instanceof Error && /audio/i.test(err.message));
    assert.match(rt.lastError(), /audio/i);
    console.log("engine error:", rt.lastError());
  } finally {
    rt.freeContext(h);
  }
}
console.log("wasm smoke OK");
