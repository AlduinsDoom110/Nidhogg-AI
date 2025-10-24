"""Command line interface for training and evaluating the agent."""

from __future__ import annotations

import argparse

from .config import TrainingConfig
from .trainer import Trainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a DQN agent for Nidhogg")
    parser.add_argument("--device", default=None, help="Override the compute device (cpu or cuda)")
    parser.add_argument("--run-name", default=None, help="Name of the training run")
    parser.add_argument("--max-frames", type=int, default=None, help="Total training frames")
    parser.add_argument("--save-dir", default=None, help="Directory to store checkpoints")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = TrainingConfig()
    if args.device:
        config.device = args.device
    if args.run_name:
        config.run_name = args.run_name
    if args.max_frames:
        config.max_frames = args.max_frames
    if args.save_dir:
        config.save_dir = args.save_dir
    trainer = Trainer(config)
    trainer.train()


if __name__ == "__main__":
    main()
