# Copyright 2026 The ODML Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Virtual environment manager for LiteRT-LM."""

import os
import shutil
import subprocess
import sys

import click


class VenvManager:
  """Manages the virtual environment paths and binaries."""

  def __init__(self, prefer_current_venv: bool = False):
    self._self_managed_venv_dir = os.path.expanduser("~/.litert-lm/.venv")
    if not prefer_current_venv:
      self.venv_dir = self._self_managed_venv_dir
    else:
      self.venv_dir = os.environ.get(
          "VIRTUAL_ENV",
          sys.prefix
          if sys.prefix != sys.base_prefix
          else self._self_managed_venv_dir,
      )

    self.python_bin = os.path.join(self.venv_dir, "bin", "python")
    self.pip_bin = os.path.join(self.venv_dir, "bin", "pip")
    self.litert_torch_bin = os.path.join(self.venv_dir, "bin", "litert-torch")
    self.uv_bin = os.path.join(self.venv_dir, "bin", "uv")

  def ensure_venv(self):
    """Ensures that the virtual environment directory exists."""
    if os.path.exists(self.venv_dir):
      return

    if self.venv_dir != self._self_managed_venv_dir:
      # Note this should never happen.
      raise RuntimeError(
          f"Virtual environment directory not found: {self.venv_dir}"
      )

    click.echo(
        click.style(
            f"Creating virtual environment in {self.venv_dir}...", fg="cyan"
        )
    )
    os.makedirs(os.path.dirname(self.venv_dir), exist_ok=True)
    python_exe = sys.executable or "python3"
    subprocess.run([python_exe, "-m", "venv", self.venv_dir], check=True)

  def recreate_venv_if_self_managed(self):
    """Deletes and re-creates the virtual environment if it is self-managed.

    This ensures we are using the latest litert-torch-nightly. Since uv has
    local cache, if the version has been downloaded before, it will be very
    fast.
    """
    if self.venv_dir != self._self_managed_venv_dir:
      # Only recreate if it's the default venv managed by the CLI.
      return

    if os.path.exists(self.venv_dir):
      click.echo(
          click.style(
              f"Deleting virtual environment in {self.venv_dir}...", fg="cyan"
          )
      )
      shutil.rmtree(self.venv_dir)

    self.ensure_venv()

  def ensure_binary(self, binary_path):
    """Ensures the binary exists, or installs it if using the default venv."""
    if os.path.exists(binary_path):
      return

    self.ensure_venv()

    if binary_path == self.pip_bin:
      click.echo(click.style("Ensuring pip is installed...", fg="cyan"))
      subprocess.run(
          [
              self.python_bin,
              "-m",
              "ensurepip",
              "--default-pip",
          ],
          check=True,
      )
    elif binary_path == self.uv_bin:
      self.ensure_binary(self.pip_bin)
      click.echo(
          click.style(
              "Installing uv into the virtual environment...", fg="cyan"
          )
      )
      subprocess.run(
          [
              self.pip_bin,
              "install",
              "uv",
              "-i",
              "https://pypi.org/simple",
          ],
          check=True,
      )
    elif binary_path == self.litert_torch_bin:
      self.ensure_binary(self.uv_bin)
      click.echo(click.style("Installing litert-torch with uv...", fg="cyan"))
      subprocess.run(
          [
              self.uv_bin,
              "pip",
              "install",
              "-i",
              "https://pypi.org/simple",
              "litert-torch-nightly",
              "--python",
              self.python_bin,
          ],
          check=True,
      )
