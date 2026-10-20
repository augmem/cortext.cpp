import createCortextModule from "../../build-wasm/dist/wasm/cortext.js";
// Thin demo wrapper (language package: github.com/augmem/cortext.wasm)
import { CortextWasm } from "./cortext-wasm.js";

const statusEl = document.querySelector("#status");
const outputEl = document.querySelector("#output");
const modelInput = document.querySelector("#model-file");
const liModelInput = document.querySelector("#li-model-file");
const signalInput = document.querySelector("#signal");
const mediaInput = document.querySelector("#media-file");
const budgetInput = document.querySelector("#evidence-budget");
const referenceInput = document.querySelector("#reference-time");
const asOfInput = document.querySelector("#as-of");
const createButton = document.querySelector("#create");
const processButton = document.querySelector("#process");
const recallButton = document.querySelector("#recall");
const embedButton = document.querySelector("#embed");

let runtime = null;
let handle = 0;

function setStatus(value) {
  statusEl.textContent = value;
}

function show(value) {
  outputEl.textContent = JSON.stringify(value, null, 2);
}

// Per-call evidence budget and reference time from the form.
function recallOptions() {
  const budget = Number(budgetInput.value || 0);
  const options = {
    retention: "ephemeral",
    includeEmbedding: false,
    evidenceBudgetBytes: budget,
  };
  if (referenceInput.value) {
    options.referenceMs = new Date(referenceInput.value).getTime();
    options.asOf = asOfInput.checked;
  }
  return options;
}

// Resolve cortext.wasm and a preloaded cortext.data next to cortext.js, not
// next to this page.
const wasmDir = new URL("../../build-wasm/dist/wasm/", import.meta.url);

async function boot() {
  runtime = await CortextWasm.create(createCortextModule, {
    locateFile: (path) => new URL(path, wasmDir).href,
  });
  setStatus(`Ready ${runtime.version()}`);
}

createButton.addEventListener("click", async () => {
  try {
    if (handle) {
      runtime.freeContext(handle);
      handle = 0;
    }
    const file = modelInput.files?.[0];
    if (file) {
      setStatus("Loading model");
      await runtime.writeModelFile(file);
    }
    const liFile = liModelInput.files?.[0];
    if (liFile) {
      setStatus("Loading late-interaction model");
      await runtime.writeLateInteractionModelFile(liFile);
    }
    handle = runtime.createContext();
    processButton.disabled = false;
    recallButton.disabled = false;
    embedButton.disabled = false;
    setStatus("Context ready");
    show({ handle });
  } catch (err) {
    setStatus("Error");
    show({ error: String(err?.message || err) });
  }
});

processButton.addEventListener("click", async () => {
  try {
    const options = {
      retention: "durable",
      includeEmbedding: false,
      evidenceBudgetBytes: Number(budgetInput.value || 0),
    };
    const file = mediaInput.files?.[0];
    if (file) {
      const media = {
        data: new Uint8Array(await file.arrayBuffer()),
        mimetype: file.type || "application/octet-stream",
      };
      show(runtime.processTextWithMedia(
        handle, signalInput.value, "browser", media, options,
      ));
    } else {
      show(runtime.processText(handle, signalInput.value, "browser", options));
    }
    runtime.flush(handle);
  } catch (err) {
    show({ error: String(err?.message || err) });
  }
});

recallButton.addEventListener("click", () => {
  try {
    show(runtime.processText(
      handle, signalInput.value, "browser", recallOptions(),
    ));
  } catch (err) {
    show({ error: String(err?.message || err) });
  }
});

embedButton.addEventListener("click", () => {
  try {
    const result = runtime.embedText(handle, signalInput.value);
    show({
      dimension: result.dimension,
      preview: result.embedding.slice(0, 8),
    });
  } catch (err) {
    show({ error: String(err?.message || err) });
  }
});

boot().catch((err) => {
  setStatus("Error");
  show({ error: String(err?.message || err) });
});
