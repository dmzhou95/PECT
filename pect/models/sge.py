
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def low_pass(H: torch.Tensor, S: torch.Tensor, alpha_g: float) -> torch.Tensor:
    """(I - alpha_g L) H with L = I - S; S is a (sparse) normalized adjacency."""
    SH = torch.sparse.mm(S, H) if S.is_sparse else S @ H
    return (1.0 - alpha_g) * H + alpha_g * SH


class SGE(nn.Module):
    def __init__(self, n_genes: int, d_g: int, n_views: int = 3, alpha_g: float = 0.2):
        super().__init__()
        if not 0.0 <= alpha_g <= 0.5:
            raise ValueError("Theorem 1 requires 0 <= alpha_g <= 1/2")
        self.alpha_g = alpha_g
        self.W = nn.ModuleList([nn.Linear(n_genes, d_g, bias=False) for _ in range(n_views)])
        self.b = nn.Parameter(torch.zeros(n_views, d_g))
        self.theta_beta = nn.Parameter(torch.zeros(n_views))
        for lin in self.W:
            nn.init.xavier_uniform_(lin.weight)

    def forward(self, X: torch.Tensor, S_views: List[torch.Tensor]):
        beta = torch.softmax(self.theta_beta, 0)
        Hm = []
        for m, (S, lin) in enumerate(zip(S_views, self.W)):
            XW = lin(X)                                    # X W_m  (N x d_g)
            SXW = torch.sparse.mm(S, XW) if S.is_sparse else S @ XW
            Hm.append(beta[m] * F.elu(SXW + self.b[m]))
        H = torch.stack(Hm, 0).sum(0)
        return H, Hm

    def smooth(self, H: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        return low_pass(H, S, self.alpha_g)
