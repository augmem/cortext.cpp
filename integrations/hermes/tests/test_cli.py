from __future__ import annotations

from pathlib import Path

from augmem.hermes.cli import install_plugin, status


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
