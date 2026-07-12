"""CLI helpers for installing and live-verifying the Cortext Hermes provider."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path


def _package_root() -> Path:
  return Path(__file__).resolve().parent


def _plugin_source() -> Path:
  """Directory that Hermes expects under plugins/memory/cortext/."""
  return _package_root() / "plugin"


def _default_hermes_home() -> Path:
  env = os.environ.get("HERMES_HOME")
  if env:
    return Path(env).expanduser()
  return Path.home() / ".hermes"


def install_plugin(hermes_home: Path | None = None, *, force: bool = False) -> Path:
  """Symlink (or copy) the plugin tree into Hermes plugins/memory/cortext."""
  home = hermes_home or _default_hermes_home()
  source = _plugin_source()
  if not source.is_dir():
    raise FileNotFoundError(f"Plugin source missing: {source}")

  dest = home / "plugins" / "memory" / "cortext"
  dest.parent.mkdir(parents=True, exist_ok=True)

  if dest.exists() or dest.is_symlink():
    if not force:
      raise FileExistsError(
        f"{dest} already exists. Re-run with --force to replace it."
      )
    if dest.is_symlink() or dest.is_file():
      dest.unlink()
    else:
      shutil.rmtree(dest)

  try:
    dest.symlink_to(source, target_is_directory=True)
  except OSError:
    shutil.copytree(source, dest)

  return dest


def status(hermes_home: Path | None = None) -> int:
  home = hermes_home or _default_hermes_home()
  dest = home / "plugins" / "memory" / "cortext"
  config = home / "cortext.json"
  print(f"HERMES_HOME: {home}")
  print(f"plugin path: {dest}")
  print(f"plugin installed: {dest.exists()}")
  if dest.is_symlink():
    print(f"plugin target: {dest.resolve()}")
  print(f"config: {config} ({'present' if config.is_file() else 'missing'})")

  try:
    import augmem.cortext as cortext

    print(f"augmem.cortext: {cortext.version()}")
  except Exception as exc:
    print(f"augmem.cortext: unavailable ({exc})")
    return 1
  return 0


def live_e2e_cmd(*, force: bool = False, json_out: bool = False) -> int:
  from augmem.hermes.live_e2e import format_report, live_enabled, run_live_e2e

  if force and not live_enabled():
    os.environ["CORTEXT_HERMES_LIVE"] = "1"

  result = run_live_e2e(require_enabled=not force)
  if json_out:
    print(json.dumps(result.to_dict(), indent=2))
  else:
    print(format_report(result), end="")

  if result.skipped:
    return 2
  return 0 if result.ok else 1


def _add_hermes_home_arg(parser: argparse.ArgumentParser) -> None:
  parser.add_argument(
    "--hermes-home",
    type=Path,
    default=None,
    help="Hermes home directory (default: $HERMES_HOME or ~/.hermes)",
  )


def main(argv: list[str] | None = None) -> None:
  parser = argparse.ArgumentParser(
    prog="augmem-hermes",
    description="Install, inspect, and live-verify the Cortext memory provider.",
  )
  sub = parser.add_subparsers(dest="command", required=True)

  install_p = sub.add_parser(
    "install",
    help="Install plugin into $HERMES_HOME/plugins/memory/cortext",
  )
  _add_hermes_home_arg(install_p)
  install_p.add_argument(
    "--force",
    action="store_true",
    help="Replace an existing plugin install",
  )

  status_p = sub.add_parser("status", help="Show install and dependency status")
  _add_hermes_home_arg(status_p)

  live_p = sub.add_parser(
    "live-e2e",
    help=(
      "Live-model Bailey recall test (needs CORTEXT_HERMES_LIVE=1, "
      "OPENAI_API_KEY, and native Cortext)"
    ),
  )
  live_p.add_argument(
    "--force",
    action="store_true",
    help="Run even if CORTEXT_HERMES_LIVE is unset",
  )
  live_p.add_argument(
    "--json",
    action="store_true",
    help="Print machine-readable JSON result",
  )

  args = parser.parse_args(argv)
  home = args.hermes_home.expanduser() if getattr(args, "hermes_home", None) else None

  if args.command == "install":
    try:
      dest = install_plugin(home, force=args.force)
    except Exception as exc:
      print(f"install failed: {exc}", file=sys.stderr)
      raise SystemExit(1) from exc
    print(f"Installed Cortext memory plugin at {dest}")
    print("Enable with:")
    print("  hermes config set memory.provider cortext")
    print("  # or: hermes memory setup  (select cortext)")
    return

  if args.command == "status":
    raise SystemExit(status(home))

  if args.command == "live-e2e":
    raise SystemExit(live_e2e_cmd(force=args.force, json_out=args.json))


if __name__ == "__main__":
  main()
