"use strict";

const crypto = require("crypto");
const fs = require("fs");
const https = require("https");
const os = require("os");
const path = require("path");
const { spawnSync } = require("child_process");

const AIST_REPO = "augmem/AIST-87M-GGUF";
const AIST_REVISION = "main";
const AIST_MODEL_FILE = {
  filename: "AIST-87M_q8_0.gguf",
  sha256: "bf4c49954eccc65183f1a97e44606e86c7ee5a4fea500457124b687a3ec97898",
  size: 141491936,
};
const TOKENIZER_REPO = "bert-base-uncased";
const TOKENIZER_REVISION = "main";
const TOKENIZER_FILE = {
  filename: "vocab.txt",
  target: path.join("mdbr-leaf-ir", "vocab.txt"),
  sha256: "07eced375cec144d27c900241f3e339478dec958f92fddbc551f295c992038a3",
  size: 231508,
};
const DOWNLOAD_TIMEOUT_MS = 60000;

function hfUrl(repo, revision, filename) {
  return `https://huggingface.co/${repo}/resolve/${revision}/${filename}?download=1`;
}

function sha256File(filePath) {
  const digest = crypto.createHash("sha256");
  const data = fs.readFileSync(filePath);
  digest.update(data);
  return digest.digest("hex");
}

function verifyFile(filePath, expectedSha256, expectedSize) {
  if (!fs.existsSync(filePath)) {
    return false;
  }
  const stat = fs.statSync(filePath);
  return stat.isFile() && stat.size === expectedSize && sha256File(filePath) === expectedSha256;
}

function modelCacheDir() {
  if (process.env.CORTEXT_MODEL_CACHE_DIR) {
    return path.resolve(process.env.CORTEXT_MODEL_CACHE_DIR);
  }

  let root;
  if (process.platform === "win32") {
    root =
      process.env.LOCALAPPDATA ||
      process.env.APPDATA ||
      path.join(os.homedir(), "AppData", "Local");
  } else if (process.platform === "darwin") {
    root = path.join(os.homedir(), "Library", "Caches");
  } else {
    root = process.env.XDG_CACHE_HOME || path.join(os.homedir(), ".cache");
  }

  return path.join(root, "augmem", "cortext", "models");
}

function aistModelPath(dir) {
  const candidates = [
    path.join(dir, "AIST-87M-GGUF", "AIST-87M_q8_0.gguf"),
    path.join(dir, "AIST-87M-GGUF", "AIST-87M_q5_1.gguf"),
  ];
  return candidates.find((candidate) => fs.existsSync(candidate));
}

function cachedDefaultAistModelPath() {
  const cacheRoot = modelCacheDir();
  const modelPath = path.join(cacheRoot, "AIST-87M-GGUF", AIST_MODEL_FILE.filename);
  const tokenizerPath = path.join(cacheRoot, TOKENIZER_FILE.target);
  if (
    verifyFile(modelPath, AIST_MODEL_FILE.sha256, AIST_MODEL_FILE.size) &&
    verifyFile(tokenizerPath, TOKENIZER_FILE.sha256, TOKENIZER_FILE.size)
  ) {
    return modelPath;
  }
  return undefined;
}

function defaultAistModelPath() {
  const packageModel = aistModelPath(path.join(__dirname, "models"));
  if (packageModel) {
    return packageModel;
  }
  const repoModel = aistModelPath(path.join(__dirname, "..", "..", "models"));
  if (repoModel) {
    return repoModel;
  }
  return cachedDefaultAistModelPath();
}

function requestToFile(url, dest, redirectCount = 0) {
  return new Promise((resolve, reject) => {
    if (redirectCount > 5) {
      reject(new Error(`too many redirects while downloading ${path.basename(dest)}`));
      return;
    }

    fs.mkdirSync(path.dirname(dest), { recursive: true });
    const tmp = path.join(
      path.dirname(dest),
      `.${path.basename(dest)}.${process.pid}.${Date.now()}.tmp`
    );
    const headers = { "User-Agent": "cortext-node-model-bootstrap/1.0" };
    const token = process.env.HF_TOKEN || process.env.HUGGINGFACE_HUB_TOKEN;
    if (token) {
      headers.Authorization = `Bearer ${token}`;
    }

    const cleanup = () => {
      try {
        fs.unlinkSync(tmp);
      } catch (_) {
        // Best effort cleanup after failed downloads.
      }
    };

    const req = https.get(url, { headers }, (res) => {
      if (
        res.statusCode >= 300 &&
        res.statusCode < 400 &&
        typeof res.headers.location === "string"
      ) {
        res.resume();
        const nextUrl = new URL(res.headers.location, url).toString();
        requestToFile(nextUrl, dest, redirectCount + 1).then(resolve, reject);
        return;
      }

      if (res.statusCode !== 200) {
        res.resume();
        cleanup();
        reject(new Error(`HTTP ${res.statusCode} while downloading ${path.basename(dest)}`));
        return;
      }

      const out = fs.createWriteStream(tmp);
      res.pipe(out);
      out.on("finish", () => {
        out.close(() => {
          fs.renameSync(tmp, dest);
          resolve();
        });
      });
      out.on("error", (err) => {
        cleanup();
        reject(err);
      });
    });

    req.setTimeout(DOWNLOAD_TIMEOUT_MS, () => {
      req.destroy(new Error(`timeout while downloading ${path.basename(dest)}`));
    });
    req.on("error", (err) => {
      cleanup();
      reject(err);
    });
  });
}

async function ensureVerifiedFile(url, dest, expectedSha256, expectedSize) {
  if (verifyFile(dest, expectedSha256, expectedSize)) {
    return;
  }
  await requestToFile(url, dest);
  const stat = fs.statSync(dest);
  const actualSha256 = sha256File(dest);
  if (stat.size !== expectedSize) {
    throw new Error(
      `${path.basename(dest)} has size ${stat.size}, expected ${expectedSize}`
    );
  }
  if (actualSha256 !== expectedSha256) {
    throw new Error(
      `${path.basename(dest)} has sha256 ${actualSha256}, expected ${expectedSha256}`
    );
  }
}

async function ensureDefaultAssets() {
  const cacheRoot = modelCacheDir();
  const modelPath = path.join(cacheRoot, "AIST-87M-GGUF", AIST_MODEL_FILE.filename);
  const tokenizerPath = path.join(cacheRoot, TOKENIZER_FILE.target);
  await ensureVerifiedFile(
    hfUrl(AIST_REPO, AIST_REVISION, AIST_MODEL_FILE.filename),
    modelPath,
    AIST_MODEL_FILE.sha256,
    AIST_MODEL_FILE.size
  );
  await ensureVerifiedFile(
    hfUrl(TOKENIZER_REPO, TOKENIZER_REVISION, TOKENIZER_FILE.filename),
    tokenizerPath,
    TOKENIZER_FILE.sha256,
    TOKENIZER_FILE.size
  );
  return modelPath;
}

function ensureDefaultAistModelPathSync() {
  // Prefer package/repo/cache models when present. Do not force a download:
  // default release natives embed AIST and assemble at load; setting
  // CORTEXT_AIST_MODEL_PATH to a freshly downloaded file would shadow that.
  // Explicit downloads remain available via `node model-bootstrap.js --download-default`.
  return defaultAistModelPath();
}

if (require.main === module && process.argv[2] === "--download-default") {
  ensureDefaultAssets().then(
    () => process.exit(0),
    (err) => {
      process.stderr.write(`${err && err.message ? err.message : err}\n`);
      process.exit(1);
    }
  );
}

module.exports = {
  defaultAistModelPath,
  ensureDefaultAistModelPathSync,
};
