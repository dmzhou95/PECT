
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import scipy.sparse as sp

VISIUM_SPOT_DIAMETER_UM = 55.0


@dataclass
class RawST:
    counts: sp.csr_matrix          # N x G_all raw counts
    gene_names: np.ndarray         # G_all
    coords: np.ndarray             # N x 2
    barcodes: np.ndarray           # N
    labels: Optional[np.ndarray]   # N (object) or None
    um_per_unit: Optional[float]   # micrometres per coordinate unit
    gene_ids: Optional[np.ndarray] = None  # e.g. Ensembl IDs, used for TSS lookup


@dataclass
class STData:
    X: np.ndarray                  # N x G log2-normalized expression (HVGs)
    counts: np.ndarray             # N x G raw counts of HVGs (float32)
    lib_factor: np.ndarray         # N, library size / median
    pca: np.ndarray                # N x n_pca
    coords: np.ndarray             # N x 2
    gene_names: np.ndarray         # G (HVG names)
    gene_ids: Optional[np.ndarray]
    barcodes: np.ndarray
    labels: Optional[np.ndarray]
    um_per_unit: Optional[float]


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------
def _read_labels(label_file, label_col, sep, barcodes, header=True):
    df = pd.read_csv(label_file, sep=sep, index_col=0, header=0 if header else None)
    df.index = df.index.astype(str)
    col = label_col if (label_col is not None and header) else df.columns[0]
    s = df[col].reindex(barcodes)
    return s.to_numpy(dtype=object)


def load_section(data_cfg) -> RawST:
    fmt = data_cfg.format.lower()
    if fmt == "npz":
        z = np.load(data_cfg.path, allow_pickle=True)
        counts = sp.csr_matrix(z["counts"].astype(np.float32))
        n = counts.shape[0]
        labels = z["labels"].astype(object) if "labels" in z.files else None
        return RawST(
            counts=counts,
            gene_names=z["gene_names"].astype(str),
            coords=z["coords"].astype(np.float64),
            barcodes=z["barcodes"].astype(str) if "barcodes" in z.files else np.array([f"spot{i}" for i in range(n)]),
            labels=labels,
            um_per_unit=float(z["um_per_unit"]) if "um_per_unit" in z.files else data_cfg.um_per_unit,
            gene_ids=z["gene_ids"].astype(str) if "gene_ids" in z.files else None,
        )

    import scanpy as sc  # heavy import only when needed

    if fmt == "visium":
        adata = sc.read_visium(data_cfg.path, count_file=data_cfg.count_file)
    elif fmt == "h5ad":
        adata = sc.read_h5ad(data_cfg.path)
    else:
        raise ValueError(f"Unknown data format {data_cfg.format}")
    adata.var_names_make_unique()

    counts = adata.layers["counts"] if "counts" in adata.layers else adata.X
    counts = sp.csr_matrix(counts, dtype=np.float32)
    coords = np.asarray(adata.obsm["spatial"], dtype=np.float64)

    um_per_unit = data_cfg.um_per_unit
    if um_per_unit is None and "spatial" in adata.uns:
        try:
            lib = next(iter(adata.uns["spatial"].values()))
            diam = float(lib["scalefactors"]["spot_diameter_fullres"])
            um_per_unit = VISIUM_SPOT_DIAMETER_UM / diam
        except (StopIteration, KeyError, TypeError):
            um_per_unit = None

    barcodes = np.asarray(adata.obs_names, dtype=str)
    labels = None
    if data_cfg.label_file:
        labels = _read_labels(data_cfg.label_file, data_cfg.label_col, data_cfg.label_sep, barcodes,
                              data_cfg.label_header)
    elif data_cfg.label_col and data_cfg.label_col in adata.obs:
        labels = adata.obs[data_cfg.label_col].to_numpy(dtype=object)

    gene_ids = None
    for key in ("gene_ids", "gene_id", "ensembl_id"):
        if key in adata.var:
            gene_ids = adata.var[key].to_numpy(dtype=str)
            break
    return RawST(counts, np.asarray(adata.var_names, dtype=str), coords, barcodes, labels, um_per_unit, gene_ids)


# ---------------------------------------------------------------------------
# preprocessing
# ---------------------------------------------------------------------------
def _normalize_total(counts: sp.csr_matrix, target: float = 1e4) -> sp.csr_matrix:
    lib = np.asarray(counts.sum(1)).ravel()
    lib[lib == 0] = 1.0
    return sp.diags(target / lib) @ counts


def _hvg_seurat_v3(counts: sp.csr_matrix, n_top: int) -> Optional[np.ndarray]:
    """Use scanpy's seurat_v3 flavour (raw counts) when scanpy + skmisc exist."""
    try:
        import anndata as ad
        import scanpy as sc

        a = ad.AnnData(counts.copy())
        sc.pp.highly_variable_genes(a, flavor="seurat_v3", n_top_genes=n_top)
        return np.where(a.var["highly_variable"].to_numpy())[0]
    except Exception:  # ImportError or missing scikit-misc
        return None


def _hvg_dispersion(norm: sp.csr_matrix, n_top: int, n_bins: int = 20) -> np.ndarray:
    """Seurat-style binned normalized dispersion (numpy fallback)."""
    mean = np.asarray(norm.mean(0)).ravel()
    mean_sq = np.asarray(norm.multiply(norm).mean(0)).ravel()
    var = np.maximum(mean_sq - mean**2, 0.0) * norm.shape[0] / max(norm.shape[0] - 1, 1)
    mean_safe = np.where(mean == 0, 1e-12, mean)
    disp = np.log(np.maximum(var / mean_safe, 1e-12))
    logmean = np.log1p(mean)
    bins = pd.cut(logmean, n_bins, labels=False)
    z = np.zeros_like(disp)
    for b in np.unique(bins):
        idx = bins == b
        mu, sd = disp[idx].mean(), disp[idx].std()
        z[idx] = 0.0 if (sd == 0 or not np.isfinite(sd)) else (disp[idx] - mu) / sd
    z[mean == 0] = -np.inf
    return np.sort(np.argsort(-z)[:n_top])


def preprocess(raw: RawST, data_cfg, logger=None) -> STData:
    counts = raw.counts.tocsr().astype(np.float32)
    names = raw.gene_names
    gene_ids = raw.gene_ids

    # 1) genes expressed in >= min_spots spots, no mitochondrial genes
    n_spots_expr = np.asarray((counts > 0).sum(0)).ravel()
    is_mt = np.array([n.upper().startswith("MT-") for n in names])
    keep = (n_spots_expr >= data_cfg.min_spots) & (~is_mt)
    counts, names = counts[:, keep], names[keep]
    gene_ids = gene_ids[keep] if gene_ids is not None else None

    # drop empty spots (rare in filtered matrices)
    lib_all = np.asarray(counts.sum(1)).ravel()
    spot_keep = lib_all > 0
    if not spot_keep.all() and logger:
        logger.info(f"Dropping {int((~spot_keep).sum())} spots with zero counts")
    counts = counts[spot_keep]
    coords, barcodes = raw.coords[spot_keep], raw.barcodes[spot_keep]
    labels = raw.labels[spot_keep] if raw.labels is not None else None

    # 2) HVGs
    n_top = min(data_cfg.n_hvg, counts.shape[1])
    hvg = _hvg_seurat_v3(counts, n_top)
    norm = _normalize_total(counts)
    if hvg is None:
        hvg = _hvg_dispersion(norm, n_top)
        if logger:
            logger.info("scanpy/scikit-misc unavailable: HVGs by binned dispersion (numpy)")

    # 3) normalize to 1e4 and log2(x+1)
    X = np.log2(norm[:, hvg].toarray() + 1.0).astype(np.float32)
    counts_hvg = counts[:, hvg].toarray().astype(np.float32)
    lib = counts_hvg.sum(1)
    lib_factor = (lib / max(np.median(lib), 1e-8)).astype(np.float32)
    lib_factor[lib_factor <= 0] = 1e-3

    # 4) 50-d PCA space for the feature graph
    from sklearn.decomposition import PCA

    n_pca = min(data_cfg.n_pca, X.shape[1], X.shape[0] - 1)
    pca = PCA(n_components=n_pca, random_state=0).fit_transform(X - X.mean(0)).astype(np.float32)

    if logger:
        logger.info(f"Preprocessed: {X.shape[0]} spots x {X.shape[1]} HVGs")
    return STData(
        X=X, counts=counts_hvg, lib_factor=lib_factor, pca=pca, coords=coords,
        gene_names=names[hvg], gene_ids=gene_ids[hvg] if gene_ids is not None else None,
        barcodes=barcodes, labels=labels, um_per_unit=raw.um_per_unit,
    )


def build_gene_lists(X: np.ndarray, gene_ok: np.ndarray, M: int):
    """Gene list G_i of every spot: its <= M most expressed non-zero HVGs.

    Only HVGs that have a cached DNA sequence (``gene_ok``) are eligible.
    Returns ``idx`` (N x M, HVG column indices, -1 padded), ``mask`` (N x M)
    and ``u`` (N x M) with u_ij = x_{i, g_ij} (Eq. (3)).
    """
    N, G = X.shape
    Xm = np.where(gene_ok[None, :], X, 0.0)
    M_eff = min(M, G)
    top = np.argpartition(-Xm, M_eff - 1, axis=1)[:, :M_eff]
    vals = np.take_along_axis(Xm, top, 1)
    order = np.argsort(-vals, axis=1)
    top = np.take_along_axis(top, order, 1)
    vals = np.take_along_axis(vals, order, 1)
    mask = vals > 0
    idx = np.where(mask, top, -1).astype(np.int64)
    u = np.where(mask, vals, 0.0).astype(np.float32)
    if M_eff < M:
        pad = M - M_eff
        idx = np.pad(idx, ((0, 0), (0, pad)), constant_values=-1)
        u = np.pad(u, ((0, 0), (0, pad)))
        mask = np.pad(mask, ((0, 0), (0, pad)))
    return idx, mask, u
