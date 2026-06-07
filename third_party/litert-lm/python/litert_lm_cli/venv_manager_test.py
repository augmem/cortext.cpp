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

import os
import sys
from unittest import mock

from absl.testing import absltest

from litert_lm_cli import venv_manager


class VenvManagerTest(absltest.TestCase):

  def test_standalone_venv_by_default(self):
    vm = venv_manager.VenvManager(prefer_current_venv=False)
    expected_dir = os.path.expanduser("~/.litert-lm/.venv")
    self.assertEqual(vm.venv_dir, expected_dir)

  def test_prefer_current_venv_with_virtual_env(self):
    with mock.patch.dict(os.environ, {"VIRTUAL_ENV": "/mocked/venv/dir"}):
      vm = venv_manager.VenvManager(prefer_current_venv=True)
      self.assertEqual(vm.venv_dir, "/mocked/venv/dir")

  def test_prefer_current_venv_with_sys_prefix(self):
    with mock.patch.dict(os.environ, clear=True):
      with mock.patch.object(sys, "prefix", "/mocked/sys/prefix"):
        with mock.patch.object(sys, "base_prefix", "/mocked/sys/base_prefix"):
          vm = venv_manager.VenvManager(prefer_current_venv=True)
          self.assertEqual(vm.venv_dir, "/mocked/sys/prefix")

  def test_prefer_current_venv_fallback_to_self_managed(self):
    with mock.patch.dict(os.environ, clear=True):
      with mock.patch.object(sys, "prefix", "/mocked/sys/base_prefix"):
        with mock.patch.object(sys, "base_prefix", "/mocked/sys/base_prefix"):
          vm = venv_manager.VenvManager(prefer_current_venv=True)
          expected_dir = os.path.expanduser("~/.litert-lm/.venv")
          self.assertEqual(vm.venv_dir, expected_dir)


if __name__ == "__main__":
  absltest.main()
