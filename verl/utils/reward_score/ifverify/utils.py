"""Utility functions for IFVerify evaluation (self-contained copy)."""

from __future__ import annotations

from typing import Tuple, Any


def import_ifverify_modules() -> Tuple[Any, Any]:
    """Import IFVerify evaluation modules from the local package.

    Returns:
        (instructions_registry, evaluation_lib)
    """
    try:
        from . import instructions_registry, evaluation_lib

        return instructions_registry, evaluation_lib
    except Exception as e:  # pragma: no cover - defensive
        raise ImportError(f"Failed to import IFVerify modules: {e}") from e


class MockInputExample:
    """Lightweight stand-in for `evaluation_lib.InputExample`."""

    def __init__(self, instruction_id_list, kwargs, mode_list, prompt: str = ""):
        self.instruction_id_list = instruction_id_list
        self.kwargs = kwargs
        self.mode_list = mode_list
        self.prompt = prompt

