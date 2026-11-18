
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .decoder import ZINBDecoder
from .mmct import MMCTBlock
from .pooling import ExpressionAwarePooling
from .sge import SGE
from .ssa import GatedLowRankAdapter, SpatialGate, gated_token_mean


@dataclass
class TorchData:
    """Section tensors on the training device."""
    X: torch.Tensor                 # N x G log-expression
    S_views: List[torch.Tensor]     # [S_s, S_f, S_c] sparse normalized adjacencies
    counts: torch.Tensor            # N x G raw counts of HVGs
    lib: torch.Tensor               # N library-size factors
    gene_idx: torch.Tensor          # N x M cache rows of the genes of each spot (-1 pad)
    gene_mask: torch.Tensor         # N x M bool
    u_expr: torch.Tensor            # N x M expression of those genes
    z_seq: Optional[torch.Tensor] = None   # N x d_s static prior (computed once)

    @property
    def S(self):
        return self.S_views[0]

    @property
    def N(self):
        return self.X.shape[0]


@dataclass
class BatchOutput:
    F_co: torch.Tensor          # |B| x d_c
    z_bar_B: torch.Tensor       # |B| x d_s block-weighted adapted prior of B
    M_nodes: torch.Tensor       # |B+| x d_c block-weighted interaction states
    z_bar_blocks: List[torch.Tensor]


class PECT(nn.Module):
    def __init__(self, cfg, n_genes: int, d_s: int, n_adapter_blocks: int):
        super().__init__()
        m = cfg.model
        self.cfg = cfg
        self.L = n_adapter_blocks
        self.alpha_g = m.alpha_g
        self.sge = SGE(n_genes, m.d_g, n_views=3, alpha_g=m.alpha_g)
        self.gates = nn.ModuleList([SpatialGate(m.d_g, m.gate_hidden, m.rank) for _ in range(self.L)])
        self.adapters = nn.ModuleList([GatedLowRankAdapter(d_s, m.rank, m.lora_alpha) for _ in range(self.L)])
        self.pool = ExpressionAwarePooling(d_s, m.d_a)
        self.mmct = nn.ModuleList([MMCTBlock(m.d_g, d_s, m.d_c, m.n_heads) for _ in range(self.L)])
        self.theta_alpha = nn.Parameter(torch.zeros(self.L))
        self.b_co = nn.Parameter(torch.zeros(m.d_c))
        self.decoder = ZINBDecoder(m.d_c, m.dec_hidden, n_genes)
        self.W_g = nn.Linear(m.d_g, m.d_c, bias=False)
        self.W_s = nn.Linear(d_s, m.d_c, bias=False)
        self.W_D = nn.Parameter(torch.empty(m.d_g, m.d_g))
        nn.init.xavier_uniform_(self.W_D)

    # ------------------------------------------------------------------
    def param_groups(self, train_cfg):
        """theta_a: adapters and gates; theta_g: everything else (backbone is frozen/cached)."""
        ad_ids = {id(p) for p in list(self.adapters.parameters()) + list(self.gates.parameters())}
        theta_a = [p for p in self.parameters() if id(p) in ad_ids]
        theta_g = [p for p in self.parameters() if id(p) not in ad_ids]
        return [
            dict(params=theta_g, lr=train_cfg.lr_graph, weight_decay=train_cfg.weight_decay),
            dict(params=theta_a, lr=train_cfg.lr_adapter, weight_decay=train_cfg.weight_decay),
        ]

    def block_weights(self):
        return torch.softmax(self.theta_alpha, 0)

    # ------------------------------------------------------------------
    def graph_branch(self, data: TorchData, X: Optional[torch.Tensor] = None):
        H, Hm = self.sge(data.X if X is None else X, data.S_views)
        Hbar = self.sge.smooth(H, data.S)
        Hbar_m = [self.sge.smooth(h, data.S) for h in Hm]
        gates = [g(Hbar, data.S) for g in self.gates]
        return dict(H=H, Hm=Hm, Hbar=Hbar, Hbar_m=Hbar_m, gates=gates)

    @staticmethod
    def _pairs(data: TorchData, nodes: torch.Tensor):
        gidx = data.gene_idx[nodes]
        gmask = data.gene_mask[nodes]
        uniq, inv = torch.unique(gidx[gmask], return_inverse=True)   # sorted cache rows
        gl = torch.full_like(gidx, -1)
        gl[gmask] = inv
        pair_spot, pair_slot = gmask.nonzero(as_tuple=True)
        return gidx, gmask, uniq, gl, pair_spot, pair_slot

    @torch.no_grad()
    def compute_static_prior(self, data: TorchData, cache, chunk: int = 1024) -> torch.Tensor:
        """z_seq_i = Pool_i(E_i) from frozen embeddings e_g (Algorithm 1, line 3).

        Computed once before training and kept fixed ("static prior")."""
        out = []
        for s in range(0, data.N, chunk):
            nodes = torch.arange(s, min(s + chunk, data.N), device=data.X.device)
            _, gmask, uniq, gl, _, _ = self._pairs(data, nodes)
            C = cache.final.index_select(0, uniq)
            out.append(self.pool(C, gl, gmask, data.u_expr[nodes]))
        return torch.cat(out, 0)

    def _down(self, li: int, cache, block: int, uniq: torch.Tensor, recompute: bool):
        ad = self.adapters[li]
        if not (recompute and torch.is_grad_enabled()):
            return ad.down(cache.tokens(block, uniq))
        rows = uniq.detach().cpu()

        def fn(W_down, b_down):
            X = cache.tokens(block, rows)
            if X.device.type == "cuda" and X.dtype != torch.float32:
                return (X @ W_down.to(X.dtype)).float() + b_down
            return X.float() @ W_down + b_down

        # token states are re-read from the host cache in the backward pass
        return checkpoint(fn, ad.W_down, ad.b_down, use_reentrant=False)

    def adapted_priors(self, data: TorchData, cache, nodes: torch.Tensor, gates: List[torch.Tensor],
                       token_chunk: int = 4096, recompute_tokens: bool = True) -> List[torch.Tensor]:
        """z_bar_i^(l) for the spots ``nodes`` and every adapter-bearing block."""
        gidx, gmask, uniq, gl, pair_spot, pair_slot = self._pairs(data, nodes)
        pair_gene = gl[pair_spot, pair_slot]
        tok_w = cache.tok_w.index_select(0, uniq)            # masked-mean weights m_g / 1^T m_g
        u = data.u_expr[nodes]
        n, M = gidx.shape
        zbars = []
        for li, block in enumerate(cache.blocks):
            V = self._down(li, cache, block, uniq, recompute_tokens)            # G' x n x r
            g_pair = gates[li].index_select(0, nodes).index_select(0, pair_spot)
            U_pair = gated_token_mean(V, tok_w, pair_gene, g_pair, token_chunk)  # P x r
            U_ad = U_pair.new_zeros(n, M, U_pair.shape[1]).index_put((pair_spot, pair_slot), U_pair)
            C = cache.pooled[block].index_select(0, uniq)                       # rho(Z_g^(l))
            zbars.append(self.pool(C, gl, gmask, u, U_ad, self.adapters[li].scaled_up()))
        return zbars

    # ------------------------------------------------------------------
    def forward_batch(self, data: TorchData, cache, sub, gb: dict, token_chunk: int = 4096,
                      recompute_tokens: bool = True) -> BatchOutput:
        """``sub`` holds the torch versions of a :class:`pect.data.dataset.Subgraph`."""
        nodes = sub["nodes"]
        zbars = self.adapted_priors(data, cache, nodes, gb["gates"], token_chunk, recompute_tokens)
        alpha = self.block_weights()
        Hbar_nodes = gb["Hbar"].index_select(0, nodes)
        Hbar_m_nodes = [h.index_select(0, nodes) for h in gb["Hbar_m"]]
        zseq_nodes = data.z_seq.index_select(0, nodes)

        M_sum, Mt_sum = 0.0, 0.0
        for li in range(self.L):
            M = self.mmct[li].interaction(Hbar_m_nodes, zseq_nodes, zbars[li])            # |B2| x d_c
            Mt = self.mmct[li].attend(M.index_select(0, sub["b1_local"]), Hbar_nodes, zbars[li], sub["b1_nbr"])
            M_sum = M_sum + alpha[li] * M
            Mt_sum = Mt_sum + alpha[li] * Mt                                               # |B1| x d_c

        # Eq. (29): F_co = (I - alpha_g L) sum_l alpha_l M~^(l) + 1 b_co^T on the targets B
        nb = sub["b_nbr_in_b1"]
        w = sub["b_nbr_w"]
        neigh = (w.unsqueeze(-1) * Mt_sum[nb.clamp_min(0)]).sum(1)                         # (S X)_i
        F_co = (1.0 - self.alpha_g) * Mt_sum.index_select(0, sub["b_in_b1"]) + self.alpha_g * neigh + self.b_co

        b_in_nodes = sub["b1_local"].index_select(0, sub["b_in_b1"])
        z_bar_B = sum(alpha[li] * zbars[li].index_select(0, b_in_nodes) for li in range(self.L))
        return BatchOutput(F_co=F_co, z_bar_B=z_bar_B, M_nodes=M_sum, z_bar_blocks=zbars)


def subgraph_to_torch(sub, device):
    t = lambda a, dt=torch.long: torch.as_tensor(a, dtype=dt, device=device)
    return dict(
        batch=t(sub.batch), nodes=t(sub.nodes), b1_local=t(sub.b1_local), b1_nbr=t(sub.b1_nbr),
        b_in_b1=t(sub.b_in_b1), b_nbr_in_b1=t(sub.b_nbr_in_b1),
        b_nbr_w=t(sub.b_nbr_w, torch.float32), edges_local=t(sub.edges_local),
    )
