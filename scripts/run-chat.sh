#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build"
chat_bin="${build_dir}/examples/chat/cortext_chat"
default_db="${repo_root}/examples/chat/chat_memory.db"
default_settings="${repo_root}/examples/chat/chat_memory.settings.json"

build_only=0
fresh_db=0

usage() {
  cat <<EOF
Usage: scripts/run-chat.sh [--build-only] [--fresh-db]

Builds cortext_chat, applies demo-safe defaults, and launches the chat app.

Options:
  --build-only   Build and print the resolved launch configuration, then exit.
  --fresh-db     Remove the current chat DB and settings file before launch.
  -h, --help     Show this help text.
EOF
}

while (($#)); do
  case "$1" in
    --build-only)
      build_only=1
      shift
      ;;
    --fresh-db)
      fresh_db=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -f "${repo_root}/env.sh" ]]; then
  # shellcheck disable=SC1091
  source "${repo_root}/env.sh"
fi

export CORTEXT_CHAT_DB="${CORTEXT_CHAT_DB:-${default_db}}"

mkdir -p "$(dirname "${CORTEXT_CHAT_DB}")"

if [[ "${fresh_db}" == "1" ]]; then
  rm -f "${CORTEXT_CHAT_DB}" "${CORTEXT_CHAT_DB}-wal" "${CORTEXT_CHAT_DB}-shm"
  if [[ "${CORTEXT_CHAT_DB}" == "${default_db}" ]]; then
    rm -f "${default_settings}"
  fi
fi

cmake -S "${repo_root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Debug
cmake --build "${build_dir}" -j8 --target cortext_chat

echo "Launching cortext_chat with:"
echo "  CORTEXT_CHAT_DB=${CORTEXT_CHAT_DB}"

if [[ "${build_only}" == "1" ]]; then
  exit 0
fi

exec "${chat_bin}"
