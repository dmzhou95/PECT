
import os
import subprocess
import sys

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from pect.config import PECTConfig  # noqa: E402
from pect.data.dataset import sample_subgraph  # noqa: E402
from pect.models.pect import PECT, subgraph_to_torch  # noqa: E402
from pect.trainer import PECTRunner  # noqa: E402
from pect.utils import get_logger, set_seed  # noqa: E402


@pytest.fixture(scope="module")
def runner(tmp_path_factory):
    d = tmp_path_factory.mktemp("pect")
    py = sys.executable
    subprocess.check_call([py, os.path.join(ROOT, "scripts/make_synthetic.py"), "--out", str(d), "--rows", "16",
                           "--cols", "16", "--genes", "300", "--domains", "4"])
    sets = [f"data.path={d}/section.npz", f"backbone.cache_dir={d}/cache", "data.n_hvg=200",
            "data.n_clusters=4", "train.max_epochs=2"]
    subprocess.check_call([py, os.path.join(ROOT, "scripts/extract_tss_sequences.py"), "--config",
                           os.path.join(ROOT, "configs/default.yaml"), os.path.join(ROOT, "configs/synthetic.yaml"),
                           "--set", *sets, "--fasta", f"{d}/genome.fa", "--gtf", f"{d}/genes.gtf",
                           "--out", f"{d}/seqs.fa"])
    subprocess.check_call([py, os.path.join(ROOT, "scripts/cache_backbone.py"), "--fasta", f"{d}/seqs.fa",
                           "--out", f"{d}/cache", "--model", "toy", "--blocks", "3", "4"])
    cfg = PECTConfig.from_yaml(os.path.join(ROOT, "configs/default.yaml"),
                               os.path.join(ROOT, "configs/synthetic.yaml")).override(sets)
    r = PECTRunner(cfg, get_logger("pect-test")).prepare()
    r.out_dir = d
    return r


def _model(runner, seed=0):
    set_seed(seed)
    m = PECT(runner.cfg, runner.data.X.shape[1], runner.cache.d_s, len(runner.cache.blocks))
    runner.data.z_seq = m.compute_static_prior(runner.data, runner.cache)
    return m


def _brute_force_prior(model, runner, nodes, gates, li):
    """Token-level Eq. (12)-(14) followed by the plain Eq. (3) pooling."""
    cache, data, ad = runner.cache, runner.data, model.adapters[li]
    block = cache.blocks[li]
    out = []
    for i in nodes.tolist():
        rows = data.gene_idx[i][data.gene_mask[i]]
        u = data.u_expr[i][data.gene_mask[i]]
        Y = []
        for g in rows.tolist():
            X = cache.tokens(block, np.array([g])).float()[0]          # n x d_s
            m = cache.tok_w[g]                                         # m_g / 1^T m_g
            V = X @ ad.W_down + ad.b_down
            R = ad.s * F.gelu(V * gates[li][i]) @ ad.W_up              # Eq. (13)
            Y.append(cache.pooled[block][g] + m @ R)                   # rho(Z + R)
        Y = torch.stack(Y)
        a = torch.softmax(torch.tanh(model.pool.W_a(Y)) @ model.pool.w_a + u, 0)
        out.append(a @ Y)
    return torch.stack(out)


def test_adapted_prior_matches_token_level(runner):
    model = _model(runner)
    with torch.no_grad():
        for ad in model.adapters:
            ad.W_up.normal_(0, 0.1)                 # non-trivial adapters
        gb = model.graph_branch(runner.data)
        nodes = torch.arange(0, 6)
        zb = model.adapted_priors(runner.data, runner.cache, nodes, gb["gates"], recompute_tokens=False)
        for li in range(model.L):
            ref = _brute_force_prior(model, runner, nodes, gb["gates"], li)
            assert torch.allclose(zb[li], ref, atol=1e-4), (zb[li] - ref).abs().max()


def test_zero_init_is_frozen_prior(runner):
    model = _model(runner)
    with torch.no_grad():
        gb = model.graph_branch(runner.data)
        nodes = torch.arange(runner.data.N)
        zb = model.adapted_priors(runner.data, runner.cache, nodes, gb["gates"], recompute_tokens=False)
    # last cached block is K_b, so its adapted prior equals the static prior at initialization
    if runner.cache.blocks[-1] == runner.cache.n_blocks:
        assert torch.allclose(zb[-1], runner.data.z_seq, atol=1e-5)
    assert torch.allclose(gb["gates"][0], F.gelu(torch.ones_like(gb["gates"][0])))


def test_minibatch_equals_full_graph(runner):
    model = _model(runner)
    model.eval()
    with torch.no_grad():
        for ad in model.adapters:
            ad.W_up.normal_(0, 0.1)
        g = runner.graphs
        gb = model.graph_branch(runner.data)
        full = subgraph_to_torch(sample_subgraph(np.arange(runner.data.N), g.nbr_idx, g.nbr_w, g.edges), "cpu")
        F_full = model.forward_batch(runner.data, runner.cache, full, gb, recompute_tokens=False).F_co
        batch = np.array([3, 50, 101, 200])
        sub = subgraph_to_torch(sample_subgraph(batch, g.nbr_idx, g.nbr_w, g.edges), "cpu")
        F_b = model.forward_batch(runner.data, runner.cache, sub, gb, recompute_tokens=False).F_co
    assert torch.allclose(F_b, F_full[torch.as_tensor(batch)], atol=1e-5)


def test_sequence_to_space_gradients(runner):
    model = _model(runner)
    with torch.no_grad():  # after a few steps W_up and W_phi2 are non-zero (both start at 0)
        for ad, gate in zip(model.adapters, model.gates):
            ad.W_up.normal_(0, 0.1)
            gate.W2.weight.normal_(0, 0.1)
    gb = model.graph_branch(runner.data)
    g = runner.graphs
    sub = subgraph_to_torch(sample_subgraph(np.arange(32), g.nbr_idx, g.nbr_w, g.edges), "cpu")
    out = model.forward_batch(runner.data, runner.cache, sub, gb)
    loss = out.z_bar_B.pow(2).mean()        # depends on the graph only through the gates
    loss.backward()
    assert model.sge.W[0].weight.grad is not None and model.sge.W[0].weight.grad.abs().sum() > 0
    assert model.adapters[0].W_down.grad.abs().sum() > 0


def test_training_runs(runner):
    res = runner.run(0, os.path.join(str(runner.out_dir), "run"))
    assert np.isfinite(res["best_loss"])
    assert "ARI" in res and "NMI" in res
