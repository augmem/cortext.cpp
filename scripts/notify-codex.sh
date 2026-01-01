#!/usr/bin/env bash
set -euo pipefail


DEFAULT_TARGET="${TMUX_TARGET:-2:0.0}"

# Try to detect if first argument is a valid tmux target
if [ $# -gt 0 ] && tmux display-message -p -t "$1" >/dev/null 2>&1; then
  TARGET="$1"
  shift
else
  TARGET="$DEFAULT_TARGET"
fi

if [ $# -gt 0 ]; then
  PROMPT="$*"
elif [ -n "${PROMPT:-}" ]; then
  : # Use PROMPT from environment
else
  echo "ERROR: Prompt argument is required." >&2
  echo "Usage: $0 [session_id] <message>" >&2
  exit 1
fi

if ! tmux display-message -p -t "${TARGET}" "#{pane_id}" >/dev/null 2>&1; then
  echo "ERROR: tmux target not found: ${TARGET}" >&2
  echo "Tip: set TMUX_TARGET (e.g. '2:0.0') and verify with: tmux list-panes -a" >&2
  exit 2
fi

pane_in_mode="$(tmux display-message -p -t "${TARGET}" "#{pane_in_mode}" 2>/dev/null || echo 0)"
if [ "${pane_in_mode}" = "1" ]; then
  # Exit copy-mode safely before sending keys.
  tmux send-keys -t "${TARGET}" -X cancel
  sleep 0.2
fi

tmux send-keys -t "${TARGET}" C-u
sleep 1
tmux send-keys -t "${TARGET}" -l "${PROMPT}"
sleep 1
tmux send-keys -t "${TARGET}" ENTER
