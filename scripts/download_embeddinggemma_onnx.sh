#!/usr/bin/env bash
# Download EmbeddingGemma 300M ONNX export.
# Usage: ./scripts/download_embeddinggemma_onnx.sh [output_dir] [--variants base,q4,q8]

set -euo pipefail

OUTPUT_DIR=""
VARIANTS="base"

while [ $# -gt 0 ]; do
  case "$1" in
    --variants=*)
      VARIANTS="${1#*=}"
      shift
      ;;
    --variants)
      shift
      VARIANTS="${1:-base}"
      shift
      ;;
    -*)
      shift
      ;;
    *)
      if [ -z "$OUTPUT_DIR" ]; then
        OUTPUT_DIR="$1"
      fi
      shift
      ;;
  esac
done

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="$(dirname "$0")/../models/embeddinggemma-300m-onnx"
fi
BASE_URL="https://huggingface.co/onnx-community/embeddinggemma-300m-ONNX/resolve/main"

FILES=(
  "tokenizer.model"
  "tokenizer.json"
  "tokenizer_config.json"
  "special_tokens_map.json"
  "added_tokens.json"
  "config.json"
  "generation_config.json"
)

if [[ "$VARIANTS" == *"base"* ]]; then
  FILES+=(
    "onnx/model.onnx"
    "onnx/model.onnx_data"
  )
fi

if [[ "$VARIANTS" == *"q4"* ]]; then
  FILES+=(
    "onnx/model_q4.onnx"
    "onnx/model_q4.onnx_data"
  )
fi

if [[ "$VARIANTS" == *"q8"* ]]; then
  FILES+=(
    "onnx/model_quantized.onnx"
    "onnx/model_quantized.onnx_data"
  )
fi

mkdir -p "$OUTPUT_DIR/onnx"

for file in "${FILES[@]}"; do
  out_path="$OUTPUT_DIR/$file"
  if [ -f "$out_path" ]; then
    echo "Already present: $out_path"
    continue
  fi
  url="${BASE_URL}/${file}"
  echo "Downloading $url -> $out_path"
  curl -L "$url" -o "$out_path"
done

echo "Done."
