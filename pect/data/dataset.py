
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Subgraph:
    batch: np.ndarray          # B, global spot ids
    nodes: np.ndarray          # B2 (= B+), global ids, sorted
    b1_local: np.ndarray       # positions of B1 inside ``nodes``
    b1_nbr: np.ndarray         # |B1| x D, neighbours of B1 as positions in ``nodes`` (-1 pad)
    b_in_b1: np.ndarray        # positions of B inside the B1 arrays
    b_nbr_in_b1: np.ndarray    # |B| x D, neighbours of B as positions in the B1 arrays (-1 pad)
    b_nbr_w: np.ndarray        # |B| x D, entries S_ij of the spatial view
    edges_local: np.ndarray    # E' x 2, ordered spatial edges inside B+ (positions in ``nodes``)


def sample_subgraph(batch: np.ndarray, nbr_idx: np.ndarray, nbr_w: np.ndarray, edges: np.ndarray) -> Subgraph:
    N = nbr_idx.shape[0]
    B = np.asarray(batch, dtype=np.int64)
    nb = nbr_idx[B]
    B1 = np.unique(nb[nb >= 0])
    nb1 = nbr_idx[B1]
    B2 = np.unique(nb1[nb1 >= 0])

    pos2 = -np.ones(N, np.int64)
    pos2[B2] = np.arange(len(B2))
    pos1 = -np.ones(N, np.int64)
    pos1[B1] = np.arange(len(B1))

    b1_nbr = np.where(nb1 >= 0, pos2[np.maximum(nb1, 0)], -1)
    b_nbr_in_b1 = np.where(nb >= 0, pos1[np.maximum(nb, 0)], -1)
    m = (pos2[edges[:, 0]] >= 0) & (pos2[edges[:, 1]] >= 0)
    edges_local = pos2[edges[m]]
    return Subgraph(
        batch=B, nodes=B2, b1_local=pos2[B1], b1_nbr=b1_nbr,
        b_in_b1=pos1[B], b_nbr_in_b1=b_nbr_in_b1, b_nbr_w=nbr_w[B],
        edges_local=edges_local,
    )


class SpotBatcher:
    """Shuffled batches of target spots for one epoch."""

    def __init__(self, n_spots: int, batch_size: int, shuffle: bool = True, seed: int = 0):
        self.n, self.bs, self.shuffle = n_spots, batch_size, shuffle
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return (self.n + self.bs - 1) // self.bs

    def __iter__(self):
        order = self.rng.permutation(self.n) if self.shuffle else np.arange(self.n)
        for s in range(0, self.n, self.bs):
            yield order[s : s + self.bs]
