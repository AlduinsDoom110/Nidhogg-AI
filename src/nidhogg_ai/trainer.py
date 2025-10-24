"""High level training loop coordinating the environment and the agent."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict

from .agent import DQNAgent
from .config import TrainingConfig
from .environment import NidhoggEnvironment


class Trainer:
    """Runs the reinforcement learning loop."""

    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.config.ensure_save_dir()
        self.agent = DQNAgent(config)
        self.log_path = Path(config.save_dir) / f"{config.run_name}_metrics.jsonl"
        self.model_path = Path(config.save_dir) / f"{config.run_name}_policy.pt"

    def train(self) -> None:
        environment = NidhoggEnvironment(self.config)
        state = environment.reset().observation
        metrics_file = self.log_path.open("a", encoding="utf-8")
        try:
            for frame in range(self.config.max_frames):
                epsilon = self._current_epsilon(frame)
                step = self.agent.act(state, epsilon)
                env_step = environment.step(step.action_name)
                self.agent.remember(
                    state,
                    step.action_index,
                    env_step.reward,
                    env_step.observation,
                    env_step.done,
                )
                logs = self.agent.update()
                if logs:
                    logs.update({"epsilon": epsilon, "reward": env_step.reward})
                    self._log(metrics_file, logs)
                state = env_step.observation
                if env_step.done:
                    state = environment.reset().observation
                if frame % self.config.save_interval == 0 and frame > 0:
                    self.agent.save(str(self.model_path))
        finally:
            environment.close()
            metrics_file.close()

    def _current_epsilon(self, frame: int) -> float:
        if frame < self.config.warmup_steps:
            return self.config.epsilon_start
        progress = min(1.0, (frame - self.config.warmup_steps) / self.config.epsilon_decay_frames)
        return self.config.epsilon_start + progress * (self.config.epsilon_end - self.config.epsilon_start)

    def _log(self, file_obj, logs: Dict[str, float]) -> None:
        record = {"timestamp": time.time(), **{k: float(v) for k, v in logs.items()}}
        file_obj.write(json.dumps(record) + "\n")
        file_obj.flush()
