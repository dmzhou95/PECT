
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class ExpressionAwarePooling(nn.Module):
    def __init__(self, d_s: int, d_a: int):
        super().__init__()
        self.W_a = nn.Linear(d_s, d_a, bias=True)      # W_a, b_a
        self.w_a = nn.Parameter(torch.empty(d_a))
        nn.init.normal_(self.w_a, std=d_a ** -0.5)

    def forward(
        self,
        C: torch.Tensor,                 # G' x d_s per-gene base features
        gene_local: torch.Tensor,        # n x M indices into C (-1 = padding)
        mask: torch.Tensor,              # n x M bool
        u_expr: torch.Tensor,            # n x M expression u_ij
        U_ad: Optional[torch.Tensor] = None,   # n x M x r gated token means
        Wup_s: Optional[torch.Tensor] = None,  # r x d_s scaled up-projection s W_up
        return_weights: bool = False,
    ):
        n, M = gene_local.shape
        gl = gene_local.clamp_min(0)
        pre = self.W_a(C)[gl]                                   # n x M x d_a
        if U_ad is not None:
            pre = pre + U_ad @ (Wup_s @ self.W_a.weight.t())    # + u~ (s W_up W_a)
        logits = torch.tanh(pre) @ self.w_a + u_expr
        logits = logits.masked_fill(~mask, -1e9)
        a = torch.softmax(logits, dim=1) * mask                 # rows without genes -> 0
        # sum_j a_ij c_{g_ij} via a dense spot x gene weight matrix
        Wmat = C.new_zeros(n, C.shape[0]).scatter_add_(1, gl, a)
        z = Wmat @ C
        if U_ad is not None:
            z = z + torch.einsum("nm,nmr->nr", a, U_ad) @ Wup_s
        return (z, a) if return_weights else z
