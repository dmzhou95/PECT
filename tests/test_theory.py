
import os
import sys

import numpy as np
import scipy.sparse as sp

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from pect.data.graphs import renormalize, spatial_knn_graph  # noqa: E402

RNG = np.random.default_rng(0)


def gelu(x):
    from math import erf, sqrt
    return 0.5 * x * (1.0 + np.vectorize(erf)(x / sqrt(2.0)))


def _graph(n=200):
    coords = RNG.uniform(0, 10, size=(n, 2))
    return spatial_knn_graph(coords, 6)


def test_theorem1_energy_and_drift():
    A = _graph()
    S = renormalize(A, np.float64).toarray()
    L = np.eye(S.shape[0]) - S
    lam = np.linalg.eigvalsh(L)
    assert lam.min() > -1e-9 and lam.max() < 2.0          # spectrum in [0, 2)
    for alpha_g in (0.0, 0.1, 0.2, 0.5):
        H = RNG.normal(size=(S.shape[0], 16))
        Hb = (np.eye(S.shape[0]) - alpha_g * L) @ H
        E = np.trace(H.T @ L @ H)
        Eb = np.trace(Hb.T @ L @ Hb)
        assert Eb <= E + 1e-8                              # Eq. (7), first part
        drift = np.linalg.norm(Hb - H)
        assert drift <= alpha_g * np.sqrt(lam.max() * E) + 1e-8   # Eq. (7), second part


def test_dirichlet_energy_matches_pairwise_form():
    A = _graph(80)
    At = (A + sp.eye(A.shape[0])).toarray()
    d = At.sum(1)
    H = RNG.normal(size=(A.shape[0], 5))
    pair = 0.5 * sum(At[i, j] * np.sum((H[i] / np.sqrt(d[i]) - H[j] / np.sqrt(d[j])) ** 2)
                     for i in range(len(d)) for j in range(len(d)) if At[i, j] > 0)
    L = np.eye(len(d)) - renormalize(A, np.float64).toarray()
    assert np.isclose(pair, np.trace(H.T @ L @ H), rtol=1e-5)         # Eq. (5) = Eq. (8)


def test_proposition1_pooled_evaluation_exact():
    n_tok, d_s, r, s = 20, 12, 4, 0.15
    Z = RNG.normal(size=(n_tok, d_s))
    X = RNG.normal(size=(n_tok, d_s))
    m = (RNG.random(n_tok) > 0.3).astype(float)
    Wd, bd, Wu = RNG.normal(size=(d_s, r)), RNG.normal(size=r), RNG.normal(size=(r, d_s))
    g = RNG.uniform(0, 1, size=r)
    V = X @ Wd + bd
    R = s * gelu(V * g) @ Wu                                 # Eq. (13), token level
    rho = lambda T: m @ T / m.sum()
    lhs = rho(Z + R)                                         # adapted readout
    rhs = rho(Z) + s * (m @ gelu(V * g) / m.sum()) @ Wu      # Eq. (16)
    assert np.allclose(lhs, rhs)


def test_theorem2_rank_and_error_split():
    n, d_s, r, s = 40, 16, 3, 0.2
    V = RNG.normal(size=(n, r))
    Wu = RNG.normal(size=(r, d_s))
    g = RNG.uniform(0, 1, size=r)
    R = s * gelu(V * g) @ Wu
    assert np.linalg.matrix_rank(R) <= r
    Pi = np.linalg.pinv(Wu) @ Wu
    assert np.allclose(R @ Pi, R)                           # row space of W_up
    T = RNG.normal(size=(n, d_s))
    lhs = np.linalg.norm(T - R) ** 2
    rhs = np.linalg.norm(T @ (np.eye(d_s) - Pi)) ** 2 + np.linalg.norm(T @ Pi - R) ** 2
    assert np.isclose(lhs, rhs)
    sv = np.linalg.svd(T, compute_uv=False)
    assert np.linalg.norm(T @ (np.eye(d_s) - Pi)) ** 2 >= np.sum(sv[r:] ** 2) - 1e-8


def test_eq26_factorized_bilinear():
    d_g, d_s, d_c = 5, 7, 6
    h, z = RNG.normal(size=d_g), RNG.normal(size=d_s)
    U, V = RNG.normal(size=(d_g, d_c)), RNG.normal(size=(d_s, d_c))
    P = np.stack([np.kron(U[:, k], V[:, k]) for k in range(d_c)], 1)   # columns u_k ⊗ v_k
    assert np.allclose(np.kron(h, z) @ P, (h @ U) * (z @ V))


def test_decomposed_pooling_equals_eq3():
    Gp, d_s, d_a, r, n, M, s = 9, 10, 6, 3, 5, 4, 0.3
    C = RNG.normal(size=(Gp, d_s))
    Wa, ba, wa = RNG.normal(size=(d_s, d_a)), RNG.normal(size=d_a), RNG.normal(size=d_a)
    Wu = RNG.normal(size=(r, d_s))
    gl = np.stack([RNG.choice(Gp, M, replace=False) for _ in range(n)])
    mask = RNG.random((n, M)) > 0.2
    mask[:, 0] = True
    u = RNG.random((n, M))
    Uad = RNG.normal(size=(n, M, r))
    # explicit Eq. (3) on y_ij = c_g + u~_ij (s W_up)
    z_ref = np.zeros((n, d_s))
    for i in range(n):
        Y = C[gl[i]] + Uad[i] @ (s * Wu)
        logit = np.tanh(Y @ Wa + ba) @ wa + u[i]
        logit[~mask[i]] = -np.inf
        a = np.exp(logit - logit.max()); a /= a.sum()
        z_ref[i] = a @ Y
    # decomposition used in pect/models/pooling.py
    pre = (C @ Wa + ba)[gl] + Uad @ (s * Wu @ Wa)
    logit = np.tanh(pre) @ wa + u
    logit = np.where(mask, logit, -1e9)
    a = np.exp(logit - logit.max(1, keepdims=True)); a /= a.sum(1, keepdims=True); a *= mask
    Wmat = np.zeros((n, Gp))
    np.add.at(Wmat, (np.repeat(np.arange(n), M), gl.ravel()), a.ravel())
    z = Wmat @ C + np.einsum("nm,nmr->nr", a, Uad) @ (s * Wu)
    assert np.allclose(z, z_ref)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
