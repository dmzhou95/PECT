
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


class SpatialGate(nn.Module):
    def __init__(self, d_g: int, hidden: int, r: int):
        super().__init__()
        self.W1 = nn.Linear(d_g, hidden, bias=False)     # W_phi1
        self.W2 = nn.Linear(hidden, r, bias=False)       # W_phi2
        self.b2 = nn.Parameter(torch.ones(r))            # b_phi2
        nn.init.xavier_uniform_(self.W1.weight)
        nn.init.zeros_(self.W2.weight)

    def forward(self, Hbar: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        P = self.W2(self.W1(Hbar))
        SP = torch.sparse.mm(S, P) if S.is_sparse else S @ P
        return F.gelu(SP + self.b2)


class GatedLowRankAdapter(nn.Module):
    def __init__(self, d_s: int, r: int, alpha: float):
        super().__init__()
        self.r, self.alpha = r, alpha
        self.s = alpha / r
        self.W_down = nn.Parameter(torch.randn(d_s, r) * d_s ** -0.5)
        self.b_down = nn.Parameter(torch.zeros(r))
        self.W_up = nn.Parameter(torch.zeros(r, d_s))

    def down(self, X_tok: torch.Tensor) -> torch.Tensor:
        """V = X W_down + 1 b_d^T; X_tok may be float16 (cached)."""
        if X_tok.dtype != torch.float32 and X_tok.device.type == "cuda":
            V = (X_tok @ self.W_down.to(X_tok.dtype)).float()
        else:
            V = X_tok.float() @ self.W_down
        return V + self.b_down

    def scaled_up(self) -> torch.Tensor:
        return self.s * self.W_up

    def delta_w_sqnorm(self) -> torch.Tensor:
        """||Delta W||_F^2 with Delta W = (alpha/r) W_down W_up (Eq. (15)), without forming d_s x d_s."""
        return self.s ** 2 * ((self.W_down.t() @ self.W_down) * (self.W_up @ self.W_up.t())).sum()


def _gtm_chunk(V, tok_w, gene, gate):
    Vg = V.index_select(0, gene)                       # c x n x r
    U = F.gelu(Vg * gate.unsqueeze(1))                 # GELU(V_g Diag(g_i))
    return torch.einsum("cn,cnr->cr", tok_w.index_select(0, gene), U)


def gated_token_mean(V: torch.Tensor, tok_w: torch.Tensor, pair_gene: torch.Tensor,
                     pair_gate: torch.Tensor, chunk: int = 4096) -> torch.Tensor:
    """m_g^T GELU(V_g Diag(g_i)) / 1^T m_g for every spot-gene pair (Eq. (16)).

    Pairs are processed in chunks; under autograd every chunk is checkpointed,
    so the P x n x r activations are never stored.
    """
    P = pair_gene.shape[0]
    outs = []
    use_ckpt = torch.is_grad_enabled()
    for s in range(0, P, chunk):
        args = (V, tok_w, pair_gene[s : s + chunk], pair_gate[s : s + chunk])
        if use_ckpt:
            outs.append(checkpoint(_gtm_chunk, *args, use_reentrant=False))
        else:
            outs.append(_gtm_chunk(*args))
    return torch.cat(outs, 0) if outs else V.new_zeros(0, V.shape[-1])
