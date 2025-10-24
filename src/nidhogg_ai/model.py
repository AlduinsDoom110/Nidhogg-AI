"""Neural network modules for the DQN agent."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvDQN(nn.Module):
    """A convolutional network similar to the architecture used by Atari DQN."""

    def __init__(self, input_channels: int, num_actions: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x / 255.0 if x.dtype == torch.uint8 else x
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
