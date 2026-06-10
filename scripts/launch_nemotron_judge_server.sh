#!/usr/bin/env bash
set -euo pipefail

# Launch the local OpenAI-compatible judge endpoint used by the chat-replay/Nemotron
# evaluation scripts. The judge scripts default to this URL and model name.

MODEL_ID="${MODEL_ID:-mlx-community/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-8bit}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-nemotron-3-nano-omni-30b-a3b-8bit}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
VLLM_MLX_SPEC="${VLLM_MLX_SPEC:-vllm-mlx @ git+https://github.com/waybarrios/vllm-mlx}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
MLLM="${MLLM:-1}"

BASE_URL="http://${HOST}:${PORT}/v1"

if curl -fsS --max-time 2 "${BASE_URL}/models" >/dev/null 2>&1; then
  echo "Nemotron judge server is already responding at ${BASE_URL}"
  curl -fsS --max-time 2 "${BASE_URL}/models"
  exit 0
fi

echo "Starting Nemotron judge server:"
echo "  model: ${MODEL_ID}"
echo "  served model name: ${SERVED_MODEL_NAME}"
echo "  endpoint: ${BASE_URL}"
echo
echo "Judge scripts should use:"
echo "  CORTEXT_JUDGE_BASE_URL=${BASE_URL}"
echo "  --model ${SERVED_MODEL_NAME}"
echo

args=(
  --host "${HOST}"
  --port "${PORT}"
  --served-model-name "${SERVED_MODEL_NAME}"
)

if [[ "${TRUST_REMOTE_CODE}" == "1" || "${TRUST_REMOTE_CODE}" == "true" ]]; then
  args+=(--trust-remote-code)
fi

if [[ "${MLLM}" == "1" || "${MLLM}" == "true" ]]; then
  args+=(--mllm)
fi

exec uvx --from "${VLLM_MLX_SPEC}" vllm-mlx serve "${MODEL_ID}" \
  "${args[@]}" \
  "$@"
