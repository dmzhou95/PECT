
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZINBDecoder(nn.Module):
    def __init__(self, d_c: int, hidden: int, n_genes: int):
        super().__init__()
        self.hidden = nn.Sequential(nn.Linear(d_c, hidden), nn.ELU())
        self.mu = nn.Linear(hidden, n_genes)
        self.theta = nn.Linear(hidden, n_genes)
        self.pi = nn.Linear(hidden, n_genes)

    def forward(self, F_co: torch.Tensor, lib_factor: torch.Tensor):
        h = self.hidden(F_co)
        mu = lib_factor.unsqueeze(1) * F.softplus(self.mu(h))
        theta = F.softplus(self.theta(h)) + 1e-4
        pi_logits = self.pi(h)               # pi = sigmoid(pi_logits)
        return mu, theta, pi_logits
