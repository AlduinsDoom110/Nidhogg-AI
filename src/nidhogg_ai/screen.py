"""Screen capture utilities built on top of :mod:`mss` and :mod:`cv2`."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Deque, Iterable, Tuple

import numpy as np

try:  # mss and cv2 are optional during development
    import cv2
    import mss
except Exception:  # pragma: no cover - optional dependency guard
    cv2 = None  # type: ignore
    mss = None  # type: ignore


@dataclass
class ScreenCaptureConfig:
    """Configuration for the :class:`ScreenCapture` helper."""

    monitor: int
    size: Tuple[int, int]
    region: Tuple[int, int, int, int] | None
    grayscale: bool


class ScreenCapture:
    """Capture and preprocess frames from the game window."""

    def __init__(self, config: ScreenCaptureConfig) -> None:
        if mss is None or cv2 is None:  # pragma: no cover - runtime guard
            raise ImportError(
                "mss and opencv-python must be installed to use ScreenCapture"
            )

        self._config = config
        self._sct = mss.mss()

    def grab(self) -> np.ndarray:
        """Capture a single frame from the configured monitor or region."""

        if self._config.region is None:
            monitor = self._sct.monitors[self._config.monitor]
        else:
            monitor = {
                "top": self._config.region[1],
                "left": self._config.region[0],
                "width": self._config.region[2],
                "height": self._config.region[3],
            }

        raw = np.array(self._sct.grab(monitor))
        frame = raw[:, :, :3]  # drop alpha channel
        if self._config.grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(frame, self._config.size, interpolation=cv2.INTER_AREA)
        if self._config.grayscale:
            resized = np.expand_dims(resized, axis=-1)
        return resized.astype(np.float32) / 255.0

    @staticmethod
    def stack_frames(
        frames: Iterable[np.ndarray], deque: Deque[np.ndarray]
    ) -> np.ndarray:
        """Stack frames into a single array suitable for the neural network."""

        for frame in frames:
            deque.append(frame)
        while len(deque) > deque.maxlen:
            deque.popleft()
        return np.concatenate(list(deque), axis=-1)

    @staticmethod
    def sleep(fps: int) -> None:
        """Sleep long enough to respect the target capture frame-rate."""

        time.sleep(1.0 / max(fps, 1))
