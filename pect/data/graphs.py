
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors


def _symmetrize(A: sp.csr_matrix) -> sp.csr_matrix:
    return A.maximum(A.T).tocsr()


def spatial_knn_graph(coords: np.ndarray, k: int, radius: Optional[float] = None) -> sp.csr_matrix:
    N = coords.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, N)).fit(coords)
    dist, ind = nn.kneighbors(coords)
    rows = np.repeat(np.arange(N), ind.shape[1] - 1)
    cols = ind[:, 1:].ravel()
    d = dist[:, 1:].ravel()
    keep = cols != rows
    if radius is not None:
        keep &= d <= radius
    A = sp.csr_matrix((np.ones(keep.sum(), np.float32), (rows[keep], cols[keep])), shape=(N, N))
    return _symmetrize(A)


def feature_knn_graph(feat: np.ndarray, k: int) -> sp.csr_matrix:
    N = feat.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, N), metric="cosine").fit(feat)
    _, ind = nn.kneighbors(feat)
    rows = np.repeat(np.arange(N), ind.shape[1])
    cols = ind.ravel()
    keep = rows != cols
    rows, cols = rows[keep], cols[keep]
    A = sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(N, N))
    A.data[:] = 1.0
    return _symmetrize(A)


def renormalize(A: sp.csr_matrix, dtype=np.float32) -> sp.csr_matrix:
    """S = D~^{-1/2} (A + I) D~^{-1/2} with D~ = Diag((A + I) 1) (Eq. (1))."""
    At = A.astype(np.float64) + sp.eye(A.shape[0], dtype=np.float64, format="csr")
    d = np.asarray(At.sum(1)).ravel()
    dinv = 1.0 / np.sqrt(d)
    return (sp.diags(dinv) @ At @ sp.diags(dinv)).tocsr().astype(dtype)


@dataclass
class GraphViews:
    A_s: sp.csr_matrix
    A_f: sp.csr_matrix
    A_c: sp.csr_matrix
    S_s: sp.csr_matrix
    S_f: sp.csr_matrix
    S_c: sp.csr_matrix
    # padded neighbourhoods N(i) = {j : A~_ij > 0} (self included) of the
    # spatial view, with the matching entries of S = S_s
    nbr_idx: np.ndarray            # N x D_max, -1 padded
    nbr_w: np.ndarray              # N x D_max, S_ij (0 on padding)
    edges: np.ndarray              # E x 2 ordered spatial edges (no self loops)

    @property
    def views(self):
        return [self.S_s, self.S_f, self.S_c]


def padded_neighbourhoods(S: sp.csr_matrix):
    S = S.tocsr()
    S.sort_indices()
    deg = np.diff(S.indptr)
    D = int(deg.max())
    N = S.shape[0]
    idx = -np.ones((N, D), np.int64)
    w = np.zeros((N, D), np.float32)
    for i in range(N):
        a, b = S.indptr[i], S.indptr[i + 1]
        idx[i, : b - a] = S.indices[a:b]
        w[i, : b - a] = S.data[a:b]
    return idx, w


def build_graphs(coords, pca, data_cfg, um_per_unit=None, logger=None) -> GraphViews:
    radius = None
    if data_cfg.radius_um is not None:
        scale = um_per_unit if um_per_unit is not None else data_cfg.um_per_unit
        if scale is None:
            if logger:
                logger.info("No coordinate scale known: radius constraint disabled (kNN only)")
        else:
            radius = data_cfg.radius_um / scale
    A_s = spatial_knn_graph(coords, data_cfg.k_spatial, radius)
    A_f = feature_knn_graph(pca, data_cfg.k_feature)
    A_c = (data_cfg.omega * A_s + (1.0 - data_cfg.omega) * A_f).tocsr()
    S_s, S_f, S_c = renormalize(A_s), renormalize(A_f), renormalize(A_c)
    nbr_idx, nbr_w = padded_neighbourhoods(S_s)
    Acoo = A_s.tocoo()
    edges = np.stack([Acoo.row, Acoo.col], 1).astype(np.int64)
    if logger:
        logger.info(
            f"Graphs: spatial avg deg {A_s.nnz / A_s.shape[0]:.2f}, "
            f"feature avg deg {A_f.nnz / A_f.shape[0]:.2f}, max |N(i)| {nbr_idx.shape[1]}"
        )
    return GraphViews(A_s, A_f, A_c, S_s, S_f, S_c, nbr_idx, nbr_w, edges)


def dirichlet_energy(H: np.ndarray, A: sp.csr_matrix) -> float:
    """E(H) = Tr(H^T L H), L = I - S (Eq. (5) / Eq. (8))."""
    S = renormalize(A)
    return float(np.sum(H * H) - np.sum(H * (S @ H)))
