"""Live-model e2e — skipped unless CORTEXT_HERMES_LIVE=1 and deps are ready."""

from __future__ import annotations

import os

import pytest

from augmem.hermes.live_e2e import format_report, live_enabled, run_live_e2e


pytestmark = pytest.mark.live


def test_live_bailey_recall_with_model() -> None:
  if not live_enabled():
    pytest.skip("set CORTEXT_HERMES_LIVE=1 to run live model verification")

  result = run_live_e2e(require_enabled=True)
  print(format_report(result))

  if result.skipped:
    pytest.skip(result.skip_reason)

  assert result.ok, (
    "live e2e failed:\n"
    + format_report(result)
    + "\nHint: build native Cortext (ffi-release), set CORTEXT_LIBRARY_PATH, "
    "and ensure OPENAI_API_KEY / OPENAI_MODEL are valid."
  )
  assert result.invisible_ok
  assert result.prefetch
  assert result.answer
  assert "cortext" not in result.prefetch.lower()


def test_live_gate_skips_by_default() -> None:
  """Default CI path: live suite is inert without the enable flag."""
  if live_enabled():
    pytest.skip("live mode enabled in this environment")
  # Ensure the helper does not accidentally call the network.
  old = os.environ.pop("OPENAI_API_KEY", None)
  try:
    result = run_live_e2e(require_enabled=True)
  finally:
    if old is not None:
      os.environ["OPENAI_API_KEY"] = old
  assert result.skipped
  assert "CORTEXT_HERMES_LIVE" in result.skip_reason
