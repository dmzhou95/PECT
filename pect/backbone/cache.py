
from __future__ import annotations

import json
import os
from typing import List, Sequence

import numpy as np


class CacheWriter:
    def __init__(self, out_dir: str, genes: Sequence[str], n_tokens: int, d_s: int,
                 blocks: List[int], n_blocks: int, model_name: str):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir, self.blocks = out_dir, list(blocks)
        G = len(genes)
        mm = np.lib.format.open_memmap
        self.tokens = {l: mm(os.path.join(out_dir, f"tokens_b{l}.npy"), mode="w+", dtype=np.float16,
                             shape=(G, n_tokens, d_s)) for l in self.blocks}
        self.pooled = {l: mm(os.path.join(out_dir, f"pooled_b{l}.npy"), mode="w+", dtype=np.float32,
                             shape=(G, d_s)) for l in self.blocks}
        self.final = mm(os.path.join(out_dir, "final.npy"), mode="w+", dtype=np.float32, shape=(G, d_s))
        self.mask = mm(os.path.join(out_dir, "mask.npy"), mode="w+", dtype=np.uint8, shape=(G, n_tokens))
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(dict(genes=list(genes), n_tokens=n_tokens, d_s=d_s, blocks=self.blocks,
                           n_blocks=n_blocks, model_name=model_name), f)

    def write(self, start: int, mask, tokens: dict, pooled: dict, final):
        end = start + mask.shape[0]
        self.mask[start:end] = mask
        self.final[start:end] = final
        for l in self.blocks:
            self.tokens[l][start:end] = tokens[l].astype(np.float16)
            self.pooled[l][start:end] = pooled[l]

    def close(self):
        for arr in list(self.tokens.values()) + list(self.pooled.values()) + [self.final, self.mask]:
            arr.flush()


class BackboneCache:
    """Reads the cache and serves token states of the genes of a batch."""

    def __init__(self, cache_dir: str, device, blocks: List[int] | None = None, preload: bool = False):
        import torch

        with open(os.path.join(cache_dir, "meta.json")) as f:
            self.meta = json.load(f)
        self.genes = list(self.meta["genes"])
        self.gene_pos = {g: i for i, g in enumerate(self.genes)}
        self.blocks = list(blocks) if blocks is not None else list(self.meta["blocks"])
        missing = set(self.blocks) - set(self.meta["blocks"])
        if missing:
            raise ValueError(f"Blocks {sorted(missing)} are not in the cache {self.meta['blocks']}")
        self.d_s = int(self.meta["d_s"])
        self.n_blocks = int(self.meta["n_blocks"])
        self.device = device
        self.preload = preload

        mask = np.load(os.path.join(cache_dir, "mask.npy")).astype(np.float32)
        denom = np.maximum(mask.sum(1, keepdims=True), 1.0)
        # masked-mean readout rho(.; m_g) = m_g^T (.) / 1^T m_g as token weights
        self.tok_w = torch.from_numpy(mask / denom).to(device)
        self.final = torch.from_numpy(np.load(os.path.join(cache_dir, "final.npy"))).to(device)
        self.pooled = {l: torch.from_numpy(np.load(os.path.join(cache_dir, f"pooled_b{l}.npy"))).to(device)
                       for l in self.blocks}
        self._tokens = {l: np.load(os.path.join(cache_dir, f"tokens_b{l}.npy"), mmap_mode="r") for l in self.blocks}
        if preload:
            self._tokens = {l: torch.from_numpy(np.ascontiguousarray(a)).to(device) for l, a in self._tokens.items()}

    def rows_for(self, gene_names: Sequence[str]) -> np.ndarray:
        """Cache row of each HVG (-1 if the gene has no cached sequence)."""
        return np.array([self.gene_pos.get(g, -1) for g in gene_names], dtype=np.int64)

    def tokens(self, block: int, rows) -> "torch.Tensor":
        """X^(l) for the given cache rows, float16, shape (len(rows), n, d_s)."""
        import torch

        if self.preload:
            idx = rows if torch.is_tensor(rows) else torch.as_tensor(rows, device=self.device)
            return self._tokens[block].index_select(0, idx.to(self.device))
        r = rows.cpu().numpy() if torch.is_tensor(rows) else np.asarray(rows)
        arr = np.take(self._tokens[block], r, axis=0)  # rows are sorted -> near-sequential reads
        t = torch.from_numpy(arr)
        if self.device.type == "cuda":
            t = t.pin_memory().to(self.device, non_blocking=True)
        return t
