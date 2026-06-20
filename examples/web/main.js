import createCortextModule from "../../build-wasm/dist/wasm/cortext.js";
import { CortextWasm } from "../../bindings/wasm/cortext.js";

const statusEl = document.querySelector("#status");
const outputEl = document.querySelector("#output");
const modelInput = document.querySelector("#model-file");
const signalInput = document.querySelector("#signal");
const createButton = document.querySelector("#create");
const processButton = document.querySelector("#process");
const embedButton = document.querySelector("#embed");

let runtime = null;
let handle = 0;

function setStatus(value) {
  statusEl.textContent = value;
}

function show(value) {
  outputEl.textContent = JSON.stringify(value, null, 2);
}

async function boot() {
  runtime = await CortextWasm.create(createCortextModule);
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
    handle = runtime.createContext();
    processButton.disabled = false;
    embedButton.disabled = false;
    setStatus("Context ready");
    show({ handle });
  } catch (err) {
    setStatus("Error");
    show({ error: String(err?.message || err) });
  }
});

processButton.addEventListener("click", () => {
  try {
    show(runtime.processText(handle, signalInput.value, "browser"));
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
