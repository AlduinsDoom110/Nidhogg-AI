# Nidhogg AI Training Framework

This project provides a modular deep reinforcement learning (RL) framework for teaching
an artificial intelligence agent to play *Nidhogg 1* by watching the screen and sending
virtual key presses. The implementation follows the five-phase roadmap outlined in the
project brief: foundational research, environment interaction, model design, training,
and evaluation.

## Features

- **Screen capture pipeline** using [`mss`](https://github.com/BoboTiG/python-mss) and
  `opencv-python` to grab, preprocess, and stack frames from the Nidhogg window.
- **Input control layer** built on top of `pydirectinput` for reliable DirectX key
  emulation.
- **Environment abstraction** that glues screen capture and inputs together so the RL
  agent interacts with the live game just like a Gym environment.
- **Deep Q-Network agent** (DQN) implemented with PyTorch and powered by an
  Atari-inspired convolutional neural network architecture.
- **Experience replay** buffer, epsilon-greedy exploration schedule, and target network
  synchronization for stable learning.
- **Training orchestration** with JSONL metric logging and periodic checkpointing.

## Getting Started

1. Install Python 3.10 or newer.
2. (Recommended) create and activate a virtual environment.
3. Install dependencies:

   ```bash
   pip install -e .
   ```

4. Install *Nidhogg 1* on the same machine and configure it to run in windowed mode.
5. Adjust the key bindings and capture region in `TrainingConfig` if your setup differs
   from the defaults.

### Running Training

Launch the trainer from a terminal:

```bash
nidhogg-train --device cuda --run-name experiment-001
```

By default the trainer captures a 96x96 grayscale window at 15 FPS, stacks four frames,
and exposes ten high-level actions (movement, sword angles, attack, and throw). During
training the agent stores transitions in the replay buffer, periodically samples random
batches to update the DQN, and writes JSON lines with training metrics to the `runs/`
folder alongside model checkpoints.

### Customization

- Modify `TrainingConfig` to change resolution, exploration schedule, rewards, or
  learning hyperparameters.
- Replace `_compute_reward` in `nidhogg_ai/environment.py` with computer-vision logic
  that derives shaped rewards from the game HUD (e.g., detecting the player that
  currently advances, deaths, or match victories).
- Update `InputMapping` in `nidhogg_ai/environment.py` to mirror the exact keyboard
  layout you use in game.

### Safety Notes

Automating inputs can interfere with normal desktop operation. Always dedicate a
separate machine or user session for running the agent, and make sure the Nidhogg
window has focus before starting training.

## Project Structure

```
src/
└── nidhogg_ai/
    ├── __init__.py            # Package exports
    ├── agent.py               # DQN agent logic
    ├── cli.py                 # Command-line interface for training
    ├── config.py              # Dataclasses containing hyperparameters
    ├── environment.py         # Screen/input glue and reward placeholder
    ├── input_control.py       # High-level action to key-press mapping
    ├── memory.py              # Experience replay buffer
    ├── model.py               # Convolutional neural network for Q-value estimation
    ├── screen.py              # Screen capture and preprocessing helpers
    └── trainer.py             # Training loop orchestration
```

## Next Steps

The groundwork is in place to extend the agent with:

- Advanced reward shaping derived from computer-vision analysis of match progress.
- Automated match resets through template matching on post-game screens.
- Policy-gradient algorithms such as PPO or recurrent layers (LSTMs/GRUs) for better
  temporal reasoning.
- Evaluation scripts that run fixed-duration matches with epsilon set to zero for
  objective performance tracking.

Happy training!
