#!/usr/bin/env bash
set -euo pipefail


DEFAULT_TARGET="${TMUX_TARGET:-}"

resolve_tmux_target() {
  # 1) explicit CLI target
  if [ $# -gt 0 ] && tmux display-message -p -t "$1" >/dev/null 2>&1; then
    echo "$1"
    return 0
  fi

  # 2) current pane if running inside tmux
  if [ -n "${TMUX:-}" ] && tmux display-message -p "#{session_name}:#{window_index}.#{pane_index}" >/dev/null 2>&1; then
    tmux display-message -p "#{session_name}:#{window_index}.#{pane_index}"
    return 0
  fi

  # 3) TMUX_TARGET env if valid
  if [ -n "$DEFAULT_TARGET" ] && tmux display-message -p -t "$DEFAULT_TARGET" >/dev/null 2>&1; then
    echo "$DEFAULT_TARGET"
    return 0
  fi

  # 4) most recently active pane
  local recent
  recent="$(tmux list-panes -a -F '#{last_active} #{session_name}:#{window_index}.#{pane_index}' 2>/dev/null | sort -nr | head -n 1 | awk '{print $2}')"
  if [ -n "$recent" ] && tmux display-message -p -t "$recent" >/dev/null 2>&1; then
    echo "$recent"
    return 0
  fi

  return 1
}

TARGET="$(resolve_tmux_target "${1:-}")" || {
  echo "ERROR: tmux target not found (set TMUX_TARGET or run inside tmux)." >&2
  exit 2
}

# If we used the first arg as a target, shift it away.
if [ $# -gt 0 ] && tmux display-message -p -t "$1" >/dev/null 2>&1; then
  shift
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
