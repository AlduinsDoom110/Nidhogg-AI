"""Experience replay buffer implementation."""

from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Iterator, Tuple

import numpy as np
import torch


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    """Fixed-size buffer used for experience replay."""

    def __init__(self, capacity: int) -> None:
        self._buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self._buffer.append(transition)

    def __len__(self) -> int:
        return len(self._buffer)

    def sample(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self._buffer, batch_size)
        states = torch.from_numpy(np.stack([t.state for t in batch])).to(device)
        actions = torch.tensor([t.action for t in batch], dtype=torch.int64, device=device)
        rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
        next_states = torch.from_numpy(np.stack([t.next_state for t in batch])).to(device)
        dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)
        return states, actions, rewards, next_states, dones

    def iterate(self, batch_size: int) -> Iterator[Iterable[Transition]]:
        buffer_list = list(self._buffer)
        random.shuffle(buffer_list)
        for idx in range(0, len(buffer_list), batch_size):
            yield buffer_list[idx : idx + batch_size]
