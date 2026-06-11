
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class IOPNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Kembalikan logits atas aksi lawan."""
        return self.net(x)

    def log_prob(self, x: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return F.log_softmax(logits, dim=-1).gather(1, action.unsqueeze(1)).squeeze(1)

    def prob_action(self, x: torch.Tensor, action_idx: int) -> float:
        """Kembalikan probabilitas skalar untuk satu action index."""
        with torch.no_grad():
            logits = self.forward(x)
            probs = F.softmax(logits, dim=-1)
        return float(probs[0, action_idx].item())

    def sample_action(self, x: torch.Tensor) -> int:
        """Sample satu aksi dari distribusi IOP."""
        with torch.no_grad():
            logits = self.forward(x)
            dist = torch.distributions.Categorical(logits=logits)
            return int(dist.sample().item())
    