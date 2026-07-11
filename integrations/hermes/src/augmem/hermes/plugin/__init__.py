"""Hermes plugin entry for plugins/memory/cortext/.

Hermes loads this package when the directory is installed under
``$HERMES_HOME/plugins/memory/cortext`` (or the agent tree's
``plugins/memory/cortext``). The real implementation lives in
``augmem.hermes``.
"""

from __future__ import annotations

from augmem.hermes.provider import CortextMemoryProvider, register

__all__ = ["CortextMemoryProvider", "register"]
