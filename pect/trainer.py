"""Training and inference of PECT (Algorithm 1)."""
from __future__ import annotations

import copy
import json
import os
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch

from .backbone.cache import BackboneCache
from .clustering import cluster_embedding, evaluate, silhouette
from .data.dataset import SpotBatcher, sample_subgraph
from .data.graphs import build_graphs
from .data.preprocess import build_gene_lists, load_section, preprocess
from .losses import (adapter_reg_loss, decorrelation_loss, graph_infomax_loss, infonce_loss,
                     smoothness_loss, zinb_loss)
from .models.pect import PECT, TorchData, subgraph_to_torch
from .utils import count_parameters, resolve_device, set_seed


def scipy_to_torch_sparse(S: sp.csr_matrix, device) -> torch.Tensor:
    C = S.tocoo()
    idx = torch.as_tensor(np.vstack([C.row, C.col]), dtype=torch.long)
    val = torch.as_tensor(C.data, dtype=torch.float32)
    return torch.sparse_coo_tensor(idx, val, C.shape).coalesce().to(device)


def parameter_breakdown(model: PECT) -> dict:
    c = lambda mods: sum(count_parameters(m) for m in mods)
    mm = model.mmct
    inter = sum(count_parameters(b.U) + count_parameters(b.V) + count_parameters(b.W_proj) for b in mm)
    att = c(mm) - inter
    return {
        "SGE": count_parameters(model.sge),
        "gates": c(model.gates),
        "adapters": c(model.adapters),
        "gene_pooling": count_parameters(model.pool),
        "cross_modal_intersection": inter,
        "neighbourhood_attention": att,
        "decoder": count_parameters(model.decoder),
        "projections_and_discriminator": count_parameters(model.W_g) + count_parameters(model.W_s) + model.W_D.numel(),
        "block_weights_and_bias": model.theta_alpha.numel() + model.b_co.numel(),
        "total": count_parameters(model),
    }


class PECTRunner:
    def __init__(self, cfg, logger):
        self.cfg, self.log = cfg, logger
        self.device = resolve_device(cfg.train.device)

    # ------------------------------------------------------------------
    def prepare(self):
        """Seed-independent steps: preprocessing, graphs, cached backbone states (lines 1-2)."""
        cfg, log = self.cfg, self.log
        t0 = time.time()
        raw = load_section(cfg.data)
        st = preprocess(raw, cfg.data, log)
        graphs = build_graphs(st.coords, st.pca, cfg.data, st.um_per_unit, log)

        cache = BackboneCache(cfg.backbone.cache_dir, self.device, cfg.backbone.adapter_blocks,
                              cfg.backbone.preload_tokens_to_gpu)
        rows = cache.rows_for(st.gene_names)
        gene_ok = rows >= 0
        log.info(f"HVGs with a cached DNA sequence: {int(gene_ok.sum())}/{len(rows)}")
        if gene_ok.sum() == 0:
            raise RuntimeError("No HVG found in the backbone cache. Check gene names / cache_dir.")
        idx, mask, u = build_gene_lists(st.X, gene_ok, cfg.data.max_genes_per_spot)
        gene_rows = np.where(mask, rows[np.maximum(idx, 0)], -1)
        log.info(f"Genes per spot: mean {mask.sum(1).mean():.1f}, min {mask.sum(1).min()}")

        dev = self.device
        self.data = TorchData(
            X=torch.as_tensor(st.X, device=dev),
            S_views=[scipy_to_torch_sparse(S, dev) for S in graphs.views],
            counts=torch.as_tensor(st.counts, device=dev),
            lib=torch.as_tensor(st.lib_factor, device=dev),
            gene_idx=torch.as_tensor(gene_rows, dtype=torch.long, device=dev),
            gene_mask=torch.as_tensor(mask, device=dev),
            u_expr=torch.as_tensor(u, device=dev),
        )
        self.st, self.graphs, self.cache = st, graphs, cache
        # spatially ordered inference batches keep the two-hop neighbourhoods small
        self.infer_order = np.lexsort((st.coords[:, 0], st.coords[:, 1]))
        log.info(f"Preparation done in {time.time() - t0:.1f}s on {dev}")
        return self

    # ------------------------------------------------------------------
    def _losses(self, model: PECT, batch: np.ndarray):
        cfg, data, g = self.cfg, self.data, self.graphs
        Lc, tc = cfg.loss, cfg.train
        sub = subgraph_to_torch(sample_subgraph(batch, g.nbr_idx, g.nbr_w, g.edges), self.device)
        gb = model.graph_branch(data)
        out = model.forward_batch(data, self.cache, sub, gb, tc.token_chunk, tc.recompute_tokens)
        B = sub["batch"]
        terms = {}
        if Lc.use_zinb:
            mu, theta, pi = model.decoder(out.F_co, data.lib[B])
            terms["zinb"] = zinb_loss(data.counts[B], mu, theta, pi)
        if Lc.use_graph:
            perm = torch.randperm(data.N, device=self.device)
            H_shuf, _ = model.sge(data.X[perm], data.S_views)
            terms["graph"] = graph_infomax_loss(gb["H"], H_shuf, model.W_D)
        Hg = model.W_g(gb["Hbar"][B])
        Zs = model.W_s(out.z_bar_B)
        if Lc.use_dec:
            terms["dec"] = decorrelation_loss(Hg, Zs)
        if Lc.use_mi and Lc.eta1 > 0:
            terms["mi"] = Lc.eta1 * infonce_loss(Hg, Zs, Lc.tau)
        if Lc.use_reg and Lc.eta2 > 0:
            reg, _, _ = adapter_reg_loss([a.delta_w_sqnorm() for a in model.adapters], Hg, Zs, Lc.lam, Lc.gamma)
            terms["reg"] = Lc.eta2 * reg
        if Lc.use_sm:
            terms["sm"] = smoothness_loss(out.M_nodes, sub["edges_local"])
        total = sum(terms.values())
        return total, {k: float(v.detach()) for k, v in terms.items()}

    @torch.no_grad()
    def embed(self, model: PECT) -> np.ndarray:
        """F_co for all spots (Algorithm 1, line 14)."""
        model.eval()
        data, g, tc = self.data, self.graphs, self.cfg.train
        gb = model.graph_branch(data)
        F = torch.zeros(data.N, self.cfg.model.d_c, device=self.device)
        order = self.infer_order
        for s in range(0, len(order), tc.infer_batch_size):
            batch = order[s : s + tc.infer_batch_size]
            sub = subgraph_to_torch(sample_subgraph(batch, g.nbr_idx, g.nbr_w, g.edges), self.device)
            out = model.forward_batch(data, self.cache, sub, gb, tc.token_chunk, False)
            F[sub["batch"]] = out.F_co
        return F.cpu().numpy()

    # ------------------------------------------------------------------
    def run(self, seed: int, out_dir: str | None = None) -> dict:
        cfg, log, tc = self.cfg, self.log, self.cfg.train
        set_seed(seed)
        model = PECT(cfg, n_genes=self.data.X.shape[1], d_s=self.cache.d_s,
                     n_adapter_blocks=len(self.cache.blocks)).to(self.device)
        if not hasattr(self, "_logged_params"):
            pb = parameter_breakdown(model)
            log.info("Trainable parameters: " + ", ".join(f"{k}={v / 1e6:.3f}M" for k, v in pb.items()))
            self._logged_params = True
        self.data.z_seq = model.compute_static_prior(self.data, self.cache)

        opt = torch.optim.AdamW(model.param_groups(tc))
        batcher = SpotBatcher(self.data.N, tc.batch_size, shuffle=True, seed=seed)
        best, best_state, best_epoch, wait = float("inf"), None, -1, 0
        history = []
        for epoch in range(1, tc.max_epochs + 1):
            model.train()
            t0 = time.time()
            agg, n_steps = defaultdict(float), 0
            for step, batch in enumerate(batcher):
                loss, terms = self._losses(model, batch)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if tc.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                opt.step()
                agg["total"] += float(loss.detach())
                for k, v in terms.items():
                    agg[k] += v
                n_steps += 1
            ep = {k: v / n_steps for k, v in agg.items()}
            ep["epoch"], ep["time_s"] = epoch, time.time() - t0
            history.append(ep)
            if epoch == 1 or epoch % tc.log_every == 0:
                log.info(f"[seed {seed}] epoch {epoch:3d} | " +
                         " ".join(f"{k}={v:.4f}" for k, v in ep.items() if k not in ("epoch",)))
            if ep["total"] < best - 1e-6:
                best, best_epoch, wait = ep["total"], epoch, 0
                best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            else:
                wait += 1
                if wait >= tc.patience:
                    log.info(f"[seed {seed}] early stop at epoch {epoch} (best epoch {best_epoch})")
                    break

        model.load_state_dict(best_state)
        F_co = self.embed(model)
        pred = cluster_embedding(F_co, cfg.data.n_clusters, tc.cluster_method, tc.mclust_model, seed, log)
        res = evaluate(pred, self.st.labels)
        res["silhouette"] = silhouette(F_co, pred, seed=seed)
        res.update(seed=seed, best_epoch=best_epoch, best_loss=best)
        log.info(f"[seed {seed}] " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                              for k, v in res.items()))
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
            np.save(os.path.join(out_dir, "F_co.npy"), F_co)
            df = pd.DataFrame({"barcode": self.st.barcodes, "domain": pred,
                               "x": self.st.coords[:, 0], "y": self.st.coords[:, 1]})
            if self.st.labels is not None:
                df["annotation"] = self.st.labels
            df.to_csv(os.path.join(out_dir, "domains.csv"), index=False)
            pd.DataFrame(history).to_csv(os.path.join(out_dir, "history.csv"), index=False)
            with open(os.path.join(out_dir, "metrics.json"), "w") as f:
                json.dump(res, f, indent=2)
            if tc.save_checkpoint:
                torch.save(best_state, os.path.join(out_dir, "best.pt"))
        return res
