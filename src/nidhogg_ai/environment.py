"""Game environment glue code combining screen capture and input control."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, Optional

import numpy as np

from .config import TrainingConfig
from .input_control import InputController, InputMapping
from .screen import ScreenCapture, ScreenCaptureConfig


@dataclass
class EnvironmentState:
    """Container describing the current observation and auxiliary info."""

    observation: np.ndarray
    reward: float
    done: bool
    info: Dict[str, float]


class NidhoggEnvironment:
    """Wraps the real game allowing the agent to interact with it."""

    def __init__(
        self,
        config: TrainingConfig,
        input_mapping: Optional[InputMapping] = None,
    ) -> None:
        screen_config = ScreenCaptureConfig(
            monitor=config.capture_monitor,
            size=(config.frame_width, config.frame_height),
            region=config.capture_region,
            grayscale=config.grayscale,
        )
        self._screen = ScreenCapture(screen_config)
        if input_mapping is None:
            input_mapping = InputMapping({action: ["space"] for action in config.action_names})
        self._input = InputController(input_mapping)
        self._config = config
        self._frames: Deque[np.ndarray] = deque(maxlen=config.frame_stack)

    def reset(self) -> EnvironmentState:
        """Reset the environment and return the initial stacked observation."""

        self._frames.clear()
        frame = self._screen.grab()
        for _ in range(self._config.frame_stack):
            self._frames.append(frame)
        stacked = np.concatenate(list(self._frames), axis=-1)
        return EnvironmentState(observation=stacked, reward=0.0, done=False, info={})

    def step(self, action: str) -> EnvironmentState:
        """Send an action to the game and observe the outcome."""

        self._input.perform(action)
        ScreenCapture.sleep(self._config.fps)
        next_frame = self._screen.grab()
        self._frames.append(next_frame)
        stacked = np.concatenate(list(self._frames), axis=-1)
        reward, done, info = self._compute_reward()
        if done:
            self._input.neutral()
        return EnvironmentState(observation=stacked, reward=reward, done=done, info=info)

    def _compute_reward(self) -> tuple[float, bool, Dict[str, float]]:
        """Placeholder reward function awaiting game specific heuristics."""

        # In a production system this function would read game HUD information using
        # template matching or memory inspection. For now we simply return zero reward
        # and mark the episode as ongoing.
        return 0.0, False, {}

    def close(self) -> None:
        """Release resources associated with the environment."""

        self._input.neutral()
