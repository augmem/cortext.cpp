#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${repo_root}/build"
cli_bin="${build_dir}/tools/cli/cortext_cli"
default_db="${repo_root}/tools/cli/chat_memory.db"

build_only=0
fresh_db=0

usage() {
  cat <<EOF
Usage: scripts/run-chat.sh [--build-only] [--fresh-db]

Builds cortext_cli and launches its interactive repl.

Options:
  --build-only   Build and print the resolved launch configuration, then exit.
  --fresh-db     Remove the current chat DB before launch.
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
fi

cmake -S "${repo_root}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Debug -DCORTEXT_BUILD_TOOLS=ON
cmake --build "${build_dir}" -j8 --target cortext_cli

echo "Launching cortext_cli repl with:"
echo "  CORTEXT_CHAT_DB=${CORTEXT_CHAT_DB}"

if [[ "${build_only}" == "1" ]]; then
  exit 0
fi

exec "${cli_bin}" --db "${CORTEXT_CHAT_DB}" --models "${repo_root}/models" repl
