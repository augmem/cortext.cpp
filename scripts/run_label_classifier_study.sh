#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-logs/label_classifier_realdata_$(date +%Y%m%d_%H%M%S)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MODELS_DIR="${MODELS_DIR:-models}"
EMBEDDER_BIN="${EMBEDDER_BIN:-./build/tools/text_embedder/cortext_text_embedder}"
EMBEDDER_BACKEND="${EMBEDDER_BACKEND:-llama.cpp}"
NAME_PRIORS_PATH="${NAME_PRIORS_PATH:-}"

mkdir -p "${ROOT}"

if [[ ! -f "data/raw/names/census_surnames.csv" ]]; then
  "${PYTHON_BIN}" scripts/download_census_surnames.py \
    --limit 5000 \
    --out data/raw/names/census_surnames.csv
fi

"${PYTHON_BIN}" scripts/prepare_wnut17_label_data.py \
  --out-dir "${ROOT}/wnut17" \
  > "${ROOT}/prepare_wnut17.log"

"${PYTHON_BIN}" scripts/build_wordnet_label_index.py \
  --source english-wordnet-2025.xml \
  --out "${ROOT}/wordnet_index.json"

if [[ -n "${NAME_PRIORS_PATH}" ]]; then
  cp "${NAME_PRIORS_PATH}" "${ROOT}/name_priors.json"
else
  "${PYTHON_BIN}" scripts/build_name_priors.py \
    --no-seed \
    --kaggle-forenames data/surnames/forenames.csv \
    --kaggle-surnames data/surnames/surnames.csv \
    --surname-source data/raw/names/census_surnames.csv \
    --top-k 0 \
    --out "${ROOT}/name_priors.json"
fi

"${PYTHON_BIN}" scripts/build_label_training_data.py \
  --dataset data/topical_chat/train.jsonl \
  --dataset data/empathetic_dialogues/train.jsonl \
  --dataset data/empathetic_dialogues/valid.jsonl \
  --dataset data/ubuntu_dialogue/validation.jsonl \
  --labeled-train "${ROOT}/wnut17/train.jsonl" \
  --labeled-valid "${ROOT}/wnut17/valid.jsonl" \
  --wordnet-index "${ROOT}/wordnet_index.json" \
  --name-priors "${ROOT}/name_priors.json" \
  --out-dir "${ROOT}/training" \
  --max-messages 120

if [[ -f "data/meld/train.jsonl" ]]; then
  "${PYTHON_BIN}" scripts/build_label_training_data.py \
    --dataset data/meld/train.jsonl \
    --dataset data/meld/valid.jsonl \
    --wordnet-index "${ROOT}/wordnet_index.json" \
    --name-priors "${ROOT}/name_priors.json" \
    --out-dir "${ROOT}/meld_training" \
    --max-messages 220
  "${PYTHON_BIN}" scripts/sample_label_examples.py \
    --input "${ROOT}/meld_training/train.jsonl" \
    --output "${ROOT}/meld_training/train.sampled.jsonl" \
    --label-cap state=300 \
    --label-cap topic=140 \
    --label-cap none=120 \
    --label-cap person_entity=80 \
    --label-cap org_project=40 \
    --label-cap place=40 \
    --label-cap identity=30
  "${PYTHON_BIN}" scripts/sample_label_examples.py \
    --input "${ROOT}/meld_training/valid.jsonl" \
    --output "${ROOT}/meld_training/valid.sampled.jsonl" \
    --label-cap state=80 \
    --label-cap topic=40 \
    --label-cap none=40 \
    --label-cap person_entity=25 \
    --label-cap org_project=15 \
    --label-cap place=15 \
    --label-cap identity=10
  cat "${ROOT}/meld_training/train.sampled.jsonl" >> "${ROOT}/training/train.jsonl"
  cat "${ROOT}/meld_training/valid.sampled.jsonl" >> "${ROOT}/training/valid.jsonl"
fi

if [[ -f "data/goemotions/train.jsonl" ]]; then
  "${PYTHON_BIN}" scripts/build_label_training_data.py \
    --dataset data/goemotions/train.jsonl \
    --dataset data/goemotions/valid.jsonl \
    --wordnet-index "${ROOT}/wordnet_index.json" \
    --name-priors "${ROOT}/name_priors.json" \
    --out-dir "${ROOT}/goemotions_training" \
    --max-messages 260
  "${PYTHON_BIN}" scripts/sample_label_examples.py \
    --input "${ROOT}/goemotions_training/train.jsonl" \
    --output "${ROOT}/goemotions_training/train.sampled.jsonl" \
    --label-cap state=420 \
    --label-cap topic=120 \
    --label-cap none=120 \
    --label-cap identity=25 \
    --label-cap person_entity=40 \
    --label-cap org_project=20 \
    --label-cap place=20
  "${PYTHON_BIN}" scripts/sample_label_examples.py \
    --input "${ROOT}/goemotions_training/valid.jsonl" \
    --output "${ROOT}/goemotions_training/valid.sampled.jsonl" \
    --label-cap state=100 \
    --label-cap topic=35 \
    --label-cap none=35 \
    --label-cap identity=10 \
    --label-cap person_entity=15 \
    --label-cap org_project=10 \
    --label-cap place=10
  cat "${ROOT}/goemotions_training/train.sampled.jsonl" >> "${ROOT}/training/train.jsonl"
  cat "${ROOT}/goemotions_training/valid.sampled.jsonl" >> "${ROOT}/training/valid.jsonl"
fi

if [[ -f "data/taskmaster/train.jsonl" ]]; then
  "${PYTHON_BIN}" scripts/build_label_training_data.py \
    --dataset data/taskmaster/train.jsonl \
    --wordnet-index "${ROOT}/wordnet_index.json" \
    --name-priors "${ROOT}/name_priors.json" \
    --out-dir "${ROOT}/taskmaster_training" \
    --max-messages 120
  "${PYTHON_BIN}" scripts/sample_label_examples.py \
    --input "${ROOT}/taskmaster_training/train.jsonl" \
    --output "${ROOT}/taskmaster_training/train.sampled.jsonl" \
    --label-cap person_entity=400 \
    --label-cap topic=220 \
    --label-cap identity=60 \
    --label-cap state=40 \
    --label-cap none=60 \
    --label-cap org_project=60 \
    --label-cap place=40
  "${PYTHON_BIN}" scripts/sample_label_examples.py \
    --input "${ROOT}/taskmaster_training/valid.jsonl" \
    --output "${ROOT}/taskmaster_training/valid.sampled.jsonl" \
    --label-cap person_entity=90 \
    --label-cap topic=50 \
    --label-cap identity=20 \
    --label-cap state=10 \
    --label-cap none=20 \
    --label-cap org_project=20 \
    --label-cap place=20
  cat "${ROOT}/taskmaster_training/train.sampled.jsonl" >> "${ROOT}/training/train.jsonl"
  cat "${ROOT}/taskmaster_training/valid.sampled.jsonl" >> "${ROOT}/training/valid.jsonl"
fi

if [[ -f "data/personachat/train.jsonl" ]]; then
  "${PYTHON_BIN}" scripts/build_label_training_data.py \
    --dataset data/personachat/train.jsonl \
    --dataset data/personachat/valid.jsonl \
    --wordnet-index "${ROOT}/wordnet_index.json" \
    --name-priors "${ROOT}/name_priors.json" \
    --out-dir "${ROOT}/personachat_training" \
    --max-messages 120
  "${PYTHON_BIN}" scripts/sample_label_examples.py \
    --input "${ROOT}/personachat_training/train.jsonl" \
    --output "${ROOT}/personachat_training/train.sampled.jsonl" \
    --label-cap person_entity=450 \
    --label-cap identity=320 \
    --label-cap topic=180 \
    --label-cap state=60 \
    --label-cap none=60 \
    --label-cap place=40
  "${PYTHON_BIN}" scripts/sample_label_examples.py \
    --input "${ROOT}/personachat_training/valid.jsonl" \
    --output "${ROOT}/personachat_training/valid.sampled.jsonl" \
    --label-cap person_entity=100 \
    --label-cap identity=80 \
    --label-cap topic=50 \
    --label-cap state=15 \
    --label-cap none=20 \
    --label-cap place=10
  cat "${ROOT}/personachat_training/train.sampled.jsonl" >> "${ROOT}/training/train.jsonl"
  cat "${ROOT}/personachat_training/valid.sampled.jsonl" >> "${ROOT}/training/valid.jsonl"
fi

TRAIN_ARGS=(
  --train "${ROOT}/training/train.jsonl"
  --valid "${ROOT}/training/valid.jsonl"
  --wordnet-index "${ROOT}/wordnet_index.json"
  --name-priors "${ROOT}/name_priors.json"
  --out-dir "${ROOT}/model"
  --models-dir "${MODELS_DIR}"
  --type-class-weight state=6.0
  --type-class-weight topic=1.7
  --type-class-weight identity=1.4
  --type-class-weight none=0.9
  --promotion-class-weight provisional=2.2
  --promotion-class-weight durable=1.1
  --promotion-class-weight ignore=0.9
)

EVAL_ARGS=(
  --gold "${ROOT}/wnut17/test.jsonl"
  --wordnet-index "${ROOT}/wordnet_index.json"
  --name-priors "${ROOT}/name_priors.json"
  --model-dir "${ROOT}/model"
  --out-dir "${ROOT}/eval_wnut17"
  --models-dir "${MODELS_DIR}"
)

SMOKE_EVAL_ARGS=(
  --gold data/label_classifier/gold.jsonl
  --wordnet-index "${ROOT}/wordnet_index.json"
  --name-priors "${ROOT}/name_priors.json"
  --model-dir "${ROOT}/model"
  --out-dir "${ROOT}/eval_curated"
  --models-dir "${MODELS_DIR}"
)

if [[ -x "${EMBEDDER_BIN}" ]]; then
  TRAIN_ARGS+=(--embedder-bin "${EMBEDDER_BIN}" --embedder-backend "${EMBEDDER_BACKEND}")
  EVAL_ARGS+=(--embedder-bin "${EMBEDDER_BIN}" --embedder-backend "${EMBEDDER_BACKEND}")
  SMOKE_EVAL_ARGS+=(--embedder-bin "${EMBEDDER_BIN}" --embedder-backend "${EMBEDDER_BACKEND}")
fi

"${PYTHON_BIN}" scripts/train_label_classifier.py "${TRAIN_ARGS[@]}" \
  > "${ROOT}/train.log"
"${PYTHON_BIN}" scripts/eval_label_classifier.py "${EVAL_ARGS[@]}" \
  > "${ROOT}/eval_wnut17.log"
"${PYTHON_BIN}" scripts/eval_label_classifier.py "${SMOKE_EVAL_ARGS[@]}" \
  > "${ROOT}/eval_curated.log"

echo "[OK] label classifier study complete: ${ROOT}"
echo "  wnut: ${ROOT}/eval_wnut17.log"
echo "  gold: ${ROOT}/eval_curated.log"
echo "  train: ${ROOT}/train.log"
echo "  md   : ${ROOT}/eval_wnut17/summary.md"
