#!/usr/bin/env bash
# Download EmbeddingGemma 300M seq256 mixed-precision TFLite model.
# Usage: ./scripts/download_embeddinggemma.sh [output_dir]
#
# Requires HF_TOKEN with access to litert-community/embeddinggemma-300m.

set -euo pipefail

OUTPUT_DIR="${1:-$(dirname "$0")/../models/embeddinggemma-300m-litert}"
MODEL_FILE="embeddinggemma-300M_seq256_mixed-precision.tflite"
SPM_FILE="sentencepiece.model"
MODEL_URL="https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/${MODEL_FILE}"
SPM_URL="https://huggingface.co/litert-community/embeddinggemma-300m/resolve/main/${SPM_FILE}"
OUT_PATH="${OUTPUT_DIR}/${MODEL_FILE}"
SPM_PATH="${OUTPUT_DIR}/${SPM_FILE}"

if [ -f "$OUT_PATH" ] && [ -f "$SPM_PATH" ]; then
  echo "EmbeddingGemma model already present at $OUT_PATH"
  echo "SentencePiece model already present at $SPM_PATH"
  exit 0
fi

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is required to download the model."
  echo "Export a token with access to litert-community/embeddinggemma-300m."
  exit 1
fi

mkdir -p "$OUTPUT_DIR"
if [ ! -f "$OUT_PATH" ]; then
  echo "Downloading EmbeddingGemma model to $OUT_PATH..."
  curl -L -H "Authorization: Bearer $HF_TOKEN" "$MODEL_URL" -o "$OUT_PATH"
fi
if [ ! -f "$SPM_PATH" ]; then
  echo "Downloading SentencePiece model to $SPM_PATH..."
  curl -L -H "Authorization: Bearer $HF_TOKEN" "$SPM_URL" -o "$SPM_PATH"
fi
echo "Done."
