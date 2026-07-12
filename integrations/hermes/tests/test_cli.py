from __future__ import annotations

from pathlib import Path

from augmem.hermes import cli
from augmem.hermes.cli import enable_memory_provider, install_plugin, status


def test_install_plugin_symlink(tmp_path: Path) -> None:
  dest = install_plugin(tmp_path, force=True)
  assert dest.exists()
  assert (dest / "plugin.yaml").is_file()
  assert (dest / "__init__.py").is_file()
  # second install without force should fail
  try:
    install_plugin(tmp_path, force=False)
    raised = False
  except FileExistsError:
    raised = True
  assert raised


def test_status_runs(tmp_path: Path, capsys) -> None:
  code = status(tmp_path)
  out = capsys.readouterr().out
  assert "HERMES_HOME" in out
  assert code in {0, 1}


def test_enable_memory_provider_uses_hermes_config(
  tmp_path: Path, monkeypatch
) -> None:
  calls: list[dict] = []

  monkeypatch.setattr(cli.shutil, "which", lambda command: "/usr/bin/hermes")

  def _run(*args, **kwargs):
    calls.append({"args": args, "kwargs": kwargs})
    return cli.subprocess.CompletedProcess(args[0], 0)

  monkeypatch.setattr(cli.subprocess, "run", _run)

  assert enable_memory_provider(tmp_path)
  assert calls[0]["args"][0] == [
    "/usr/bin/hermes", "config", "set", "memory.provider", "cortext"
  ]
  assert calls[0]["kwargs"]["env"]["HERMES_HOME"] == str(tmp_path)
