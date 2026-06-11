"""
Jaringan DDPG untuk DLST-ANESIA.

Tiga arsitektur actor-critic terpisah (paper Sec. 5.2):
  1. Threshold utility (ū_t)  → actor = single-output regression ANN (output ū_t ∈ [0,1])
  2. Bidding strategy         → actor = multiple-output regression ANN (δ, c, p)
  3. Acceptance strategy      → actor = multiple-output regression ANN (δ, c, p)

Critic selalu Q(s, a): menerima state + action, mengeluarkan satu nilai Q.

Catatan continuous action:
  DDPG menangani continuous action space. Output actor strategy adalah vektor
  kontinu (Persamaan 21) yang berisi:
    - δ_i        : durasi fase (continuous, di-sigmoid agar ∈ (0,1))
    - c_{i,j}    : choice parameter (continuous ∈ (0,1); di-threshold 0.5 → boolean saat inference)
    - p_{i,j}    : parameter taktik (continuous; di-tanh agar terbatas, lalu di-rescale di template)
  Untuk threshold, output tunggal ū_t ∈ (0,1) via sigmoid.
"""

import torch
import torch.nn as nn


class DDPGActorThreshold(nn.Module):
    """
    Actor untuk threshold utility (single-output regression ANN).
    Output: ū_t ∈ (0, 1) lewat sigmoid.
    """

    def __init__(self, state_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return torch.sigmoid(self.net(x))  # (B, 1) ∈ (0,1)


class DDPGActorStrategy(nn.Module):
    """
    Actor untuk strategy (bidding / acceptance) — multiple-output regression ANN.

    Output flat vector berukuran action_dim (lihat dlst_agent.StrategyTemplateSpec
    untuk layout: [δ..., c..., p...]). Bagian δ dan c di-sigmoid agar ∈ (0,1),
    bagian p di-tanh agar ∈ (-1,1). Layout indeks diberikan via split_sizes:
      split_sizes = (n_delta, n_choice, n_param)
    """

    def __init__(self, state_dim: int, action_dim: int, split_sizes, hidden_dim: int = 128):
        super().__init__()
        self.action_dim = action_dim
        self.split_sizes = tuple(split_sizes)  # (n_delta, n_choice, n_param)
        assert sum(self.split_sizes) == action_dim, (
            f"split_sizes {self.split_sizes} harus berjumlah action_dim={action_dim}"
        )
        self.backbone = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        h = self.backbone(x)
        raw = self.head(h)  # (B, action_dim)

        n_delta, n_choice, n_param = self.split_sizes
        parts = []
        idx = 0
        # δ ∈ (0,1)
        if n_delta > 0:
            parts.append(torch.sigmoid(raw[:, idx:idx + n_delta]))
            idx += n_delta
        # c ∈ (0,1)
        if n_choice > 0:
            parts.append(torch.sigmoid(raw[:, idx:idx + n_choice]))
            idx += n_choice
        # p ∈ (-1,1)
        if n_param > 0:
            parts.append(torch.tanh(raw[:, idx:idx + n_param]))
            idx += n_param

        return torch.cat(parts, dim=-1) if parts else raw


class DDPGCritic(nn.Module):
    """
    Critic Q(s, a). Menerima state (state_dim) dan action (action_dim),
    mengeluarkan satu nilai Q skalar.
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.net(x).squeeze(-1)  # (B,)
