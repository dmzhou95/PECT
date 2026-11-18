
from __future__ import annotations

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.mixture import GaussianMixture


def mclust_R(emb: np.ndarray, n_clusters: int, model: str = "EEE", seed: int = 0) -> np.ndarray:
    import rpy2.robjects as robjects
    import rpy2.robjects.numpy2ri as numpy2ri

    numpy2ri.activate()
    robjects.r.library("mclust")
    robjects.r["set.seed"](seed)
    res = robjects.r["Mclust"](numpy2ri.numpy2rpy(emb), n_clusters, model)
    labels = np.array(res.rx2("classification")).astype(int) - 1
    numpy2ri.deactivate()
    return labels


def gmm_cluster(emb: np.ndarray, n_clusters: int, seed: int = 0) -> np.ndarray:
    gmm = GaussianMixture(n_components=n_clusters, covariance_type="tied", random_state=seed, n_init=3,
                          reg_covar=1e-5, max_iter=500)
    return gmm.fit_predict(emb)


def cluster_embedding(emb: np.ndarray, n_clusters: int, method: str = "mclust", model: str = "EEE",
                      seed: int = 0, logger=None) -> np.ndarray:
    emb = np.asarray(emb, dtype=np.float64)
    if method == "mclust":
        try:
            return mclust_R(emb, n_clusters, model, seed)
        except Exception as e:  # ImportError, missing R package, fitting failure
            if logger:
                logger.info(f"mclust unavailable ({type(e).__name__}: {e}); using sklearn tied-covariance GMM")
    return gmm_cluster(emb, n_clusters, seed)


def evaluate(pred: np.ndarray, labels) -> dict:
    """ARI and NMI (in percent) on spots with an annotation; silhouette is label-free."""
    out = {}
    if labels is not None:
        lab = np.asarray(labels, dtype=object)
        ok = np.array([(l is not None) and (l == l) and str(l) not in ("", "nan", "NA") for l in lab])
        out["ARI"] = 100.0 * adjusted_rand_score(lab[ok].astype(str), pred[ok])
        out["NMI"] = 100.0 * normalized_mutual_info_score(lab[ok].astype(str), pred[ok])
        out["n_annotated"] = int(ok.sum())
    return out


def silhouette(emb: np.ndarray, pred: np.ndarray, sample_size: int = 5000, seed: int = 0) -> float:
    """Label-free model-selection criterion (Section 4.1)."""
    if len(np.unique(pred)) < 2:
        return -1.0
    n = emb.shape[0]
    return float(silhouette_score(emb, pred, sample_size=min(sample_size, n), random_state=seed))
