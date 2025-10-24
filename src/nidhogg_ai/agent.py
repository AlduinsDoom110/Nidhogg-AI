"""DQN agent orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from .config import TrainingConfig
from .memory import ReplayBuffer, Transition
from .model import ConvDQN


@dataclass
class AgentStep:
    """Description of a single interaction with the environment."""

    action_index: int
    action_name: str
    epsilon: float


class DQNAgent:
    """Encapsulates the DQN logic including action selection and learning."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device)
        in_channels = config.frame_stack if config.grayscale else config.frame_stack * 3
        self.policy = ConvDQN(in_channels, len(config.action_names)).to(self.device)
        self.target = ConvDQN(in_channels, len(config.action_names)).to(self.device)
        self.target.load_state_dict(self.policy.state_dict())
        self.optimizer = optim.Adam(self.policy.parameters(), lr=config.learning_rate)
        self.replay = ReplayBuffer(config.buffer_size)
        self.frame = 0

    def act(self, state: np.ndarray, epsilon: float) -> AgentStep:
        state_tensor = torch.from_numpy(state).permute(2, 0, 1).unsqueeze(0).to(self.device)
        if np.random.rand() < epsilon:
            action_index = np.random.randint(len(self.config.action_names))
        else:
            with torch.no_grad():
                q_values = self.policy(state_tensor)
                action_index = int(torch.argmax(q_values, dim=1).item())
        action_name = self.config.action_names[action_index]
        return AgentStep(action_index=action_index, action_name=action_name, epsilon=epsilon)

    def remember(
        self,
        state: np.ndarray,
        action_index: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay.push(
            Transition(
                state=state.astype(np.float32),
                action=action_index,
                reward=reward,
                next_state=next_state.astype(np.float32),
                done=done,
            )
        )

    def update(self) -> Dict[str, float]:
        if len(self.replay) < self.config.batch_size:
            return {}
        states, actions, rewards, next_states, dones = self.replay.sample(
            self.config.batch_size, self.device
        )
        states = states.permute(0, 3, 1, 2)
        next_states = next_states.permute(0, 3, 1, 2)

        q_values = self.policy(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            target_q = self.target(next_states).max(1)[0]
            target = rewards + self.config.gamma * (1 - dones) * target_q

        loss = F.smooth_l1_loss(q_values, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), self.config.max_gradient_norm)
        self.optimizer.step()

        self.frame += 1
        if self.frame % self.config.target_sync_interval == 0:
            self.target.load_state_dict(self.policy.state_dict())

        return {"loss": float(loss.item()), "frame": float(self.frame)}

    def epsilon(self) -> float:
        slope = (self.config.epsilon_end - self.config.epsilon_start) / max(
            1, self.config.epsilon_decay_frames
        )
        decayed = self.config.epsilon_start + slope * max(0, self.frame - self.config.warmup_steps)
        return float(np.clip(decayed, self.config.epsilon_end, self.config.epsilon_start))

    def save(self, path: str) -> None:
        torch.save(self.policy.state_dict(), path)

    def load(self, path: str) -> None:
        state_dict = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(state_dict)
        self.target.load_state_dict(state_dict)
