"""Configuration dataclasses for training the Nidhogg AI agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class TrainingConfig:
    """Configuration parameters that control the training loop."""

    run_name: str = "nidhogg_dqn_run"
    save_dir: Path = Path("runs")
    device: str = "cuda"

    # Environment
    frame_height: int = 96
    frame_width: int = 96
    frame_stack: int = 4
    grayscale: bool = True
    capture_monitor: int = 1
    capture_region: Tuple[int, int, int, int] | None = None
    fps: int = 15

    # Agent and replay memory
    action_names: List[str] = field(
        default_factory=lambda: [
            "move_left",
            "move_right",
            "jump",
            "crouch",
            "sword_up",
            "sword_down",
            "sword_mid",
            "attack",
            "throw_sword",
            "idle",
        ]
    )
    buffer_size: int = 100_000
    batch_size: int = 64
    gamma: float = 0.99
    learning_rate: float = 1e-4
    target_sync_interval: int = 1_000
    max_gradient_norm: float = 10.0

    # Exploration
    epsilon_start: float = 1.0
    epsilon_end: float = 0.1
    epsilon_decay_frames: int = 200_000

    # Training loop
    warmup_steps: int = 10_000
    max_frames: int = 5_000_000
    log_interval: int = 1_000
    eval_interval: int = 50_000
    save_interval: int = 100_000

    # Reward shaping
    reward_win: float = 1.0
    reward_loss: float = -1.0
    reward_kill: float = 0.2
    reward_death: float = -0.2
    reward_forward: float = 0.05
    reward_backward: float = -0.05

    def ensure_save_dir(self) -> Path:
        """Create the directory where run artifacts are stored."""

        self.save_dir.mkdir(parents=True, exist_ok=True)
        return self.save_dir
