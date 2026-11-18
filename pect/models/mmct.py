
from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class MMCTBlock(nn.Module):
    def __init__(self, d_g: int, d_s: int, d_c: int, n_heads: int, n_views: int = 3):
        super().__init__()
        assert d_c % n_heads == 0
        self.h, self.dh = n_heads, d_c // n_heads
        # intersection: d_c (3 d_g + d_s) parameters, plus W_proj
        self.U = nn.ModuleList([nn.Linear(d_g, d_c, bias=False) for _ in range(n_views)])
        self.V = nn.Linear(d_s, d_c, bias=False)
        self.W_proj = nn.Linear(d_s, d_c, bias=False)
        # neighbourhood cross-attention
        self.W_Q = nn.Linear(d_c, d_c, bias=False)
        self.W_K = nn.Linear(d_g, d_c, bias=False)
        self.W_V = nn.Linear(d_s, d_c, bias=False)
        self.W_O = nn.Linear(d_c, d_c, bias=False)
        self.W_co = nn.Linear(d_c, d_c, bias=False)
        self.ln = nn.LayerNorm(d_c)

    def interaction(self, Hbar_views: List[torch.Tensor], z_seq: torch.Tensor, z_bar: torch.Tensor):
        hU = torch.stack([U(h) for U, h in zip(self.U, Hbar_views)], 0).sum(0)
        Phi = hU * self.V(z_seq)
        return F.gelu(Phi + self.W_proj(z_bar))

    def attend(self, M_q: torch.Tensor, Hbar_src: torch.Tensor, z_bar_src: torch.Tensor,
               nbr: torch.Tensor) -> torch.Tensor:
        """M_q: n1 x d_c queries; Hbar_src / z_bar_src: n2 x (.) features of the
        source nodes; nbr: n1 x D positions of N(i) in the source nodes (-1 pad)."""
        n1, D = nbr.shape
        valid = nbr >= 0
        nb = nbr.clamp_min(0)
        q = self.W_Q(M_q).view(n1, self.h, self.dh)
        k = self.W_K(Hbar_src).view(-1, self.h, self.dh)[nb]          # n1 x D x h x dh
        v = self.W_V(z_bar_src).view(-1, self.h, self.dh)[nb]
        scores = torch.einsum("nhd,nDhd->nhD", q, k) / math.sqrt(self.dh)
        scores = scores.masked_fill(~valid.unsqueeze(1), float("-inf"))
        att = torch.softmax(scores, dim=-1)                           # N(i) always contains i
        out = torch.einsum("nhD,nDhd->nhd", att, v).reshape(n1, self.h * self.dh)
        return self.ln(M_q + self.W_co(self.W_O(out)))
