"""Cortext memory provider for Hermes Agent (``augmem.hermes``)."""

from __future__ import annotations

from augmem.hermes.provider import (
  PROVIDER_NAME,
  CortextMemoryProvider,
  load_config,
  register,
  resolve_db_path,
)

__all__ = [
  "PROVIDER_NAME",
  "CortextMemoryProvider",
  "load_config",
  "register",
  "resolve_db_path",
  "__version__",
]

__version__ = "0.1.0"
