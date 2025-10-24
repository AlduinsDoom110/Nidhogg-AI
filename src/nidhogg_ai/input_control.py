"""Input helpers built on top of :mod:`pydirectinput` for Windows control."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

try:  # optional dependency guard
    import pydirectinput
except Exception:  # pragma: no cover
    pydirectinput = None  # type: ignore


@dataclass
class InputMapping:
    """Mapping from abstract action names to key combinations."""

    key_bindings: Dict[str, Iterable[str]]


class InputController:
    """Send high level actions to the operating system."""

    def __init__(self, mapping: InputMapping) -> None:
        if pydirectinput is None:  # pragma: no cover - runtime guard
            raise ImportError("pydirectinput must be installed to send inputs")
        self._mapping = mapping

    def perform(self, action: str, hold: bool = False) -> None:
        """Press the key(s) corresponding to ``action``."""

        keys = self._mapping.key_bindings.get(action)
        if not keys:
            raise KeyError(f"Unknown action: {action}")
        for key in keys:
            pydirectinput.keyDown(key)
        if not hold:
            self.release(action)

    def release(self, action: str) -> None:
        """Release keys associated with ``action``."""

        keys = self._mapping.key_bindings.get(action)
        if not keys:
            return
        for key in keys:
            pydirectinput.keyUp(key)

    def neutral(self) -> None:
        """Release all keys to avoid stuck input state."""

        for keys in self._mapping.key_bindings.values():
            for key in keys:
                pydirectinput.keyUp(key)
