
from __future__ import annotations

import argparse
import os

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/synthetic")
    ap.add_argument("--rows", type=int, default=30)
    ap.add_argument("--cols", type=int, default=30)
    ap.add_argument("--genes", type=int, default=600)
    ap.add_argument("--domains", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    os.makedirs(args.out, exist_ok=True)

    # hexagonal lattice in micrometres
    rr, cc = np.meshgrid(np.arange(args.rows), np.arange(args.cols), indexing="ij")
    x = cc * 100.0 + (rr % 2) * 50.0
    y = rr * 100.0 * np.sqrt(3) / 2
    coords = np.stack([x.ravel(), y.ravel()], 1)
    N, G, K = coords.shape[0], args.genes, args.domains

    # layered domains with a wavy boundary
    t = (coords[:, 1] + 60 * np.sin(coords[:, 0] / 400.0)) / (coords[:, 1].max() + 1e-6)
    labels = np.clip((t * K).astype(int), 0, K - 1)

    base = rng.gamma(0.6, 1.0, size=G) + 0.05
    prog = np.ones((K, G))
    per = G // (K + 1)
    for k in range(K):
        idx = rng.choice(G, per, replace=False)
        prog[k, idx] *= rng.uniform(3, 8, size=per)
    lib = rng.lognormal(np.log(3000), 0.3, size=N)
    mu = prog[labels] * base[None]
    mu = mu / mu.sum(1, keepdims=True) * lib[:, None]
    theta = 5.0
    lam = rng.gamma(theta, mu / theta)
    counts = rng.poisson(lam)
    counts[rng.random(counts.shape) < 0.1] = 0
    # a few mitochondrial genes, removed by preprocessing
    gene_names = np.array([f"MT-G{i}" if i < 5 else f"GENE{i:04d}" for i in range(G)])

    np.savez_compressed(
        os.path.join(args.out, "section.npz"),
        counts=counts.astype(np.float32), coords=coords, gene_names=gene_names,
        labels=np.array([f"Layer{l + 1}" for l in labels], dtype=object),
        barcodes=np.array([f"SPOT{i:05d}" for i in range(N)]), um_per_unit=np.array(1.0),
    )

    # random genome with one gene every 3 kb (alternating strands)
    L = G * 3000 + 2000
    genome = "".join(rng.choice(list("ACGT"), size=L))
    with open(os.path.join(args.out, "genome.fa"), "w") as f:
        f.write(">chr1\n")
        for i in range(0, L, 60):
            f.write(genome[i : i + 60] + "\n")
    with open(os.path.join(args.out, "genes.gtf"), "w") as f:
        for i, g in enumerate(gene_names):
            start = 1000 + i * 3000 + 1
            end = start + 1500
            strand = "+" if i % 2 == 0 else "-"
            f.write(f'chr1\tsynth\tgene\t{start}\t{end}\t.\t{strand}\t.\t'
                    f'gene_id "ENSSYN{i:06d}"; gene_name "{g}";\n')
    print(f"Synthetic section: {N} spots, {G} genes, {K} domains -> {args.out}")


if __name__ == "__main__":
    main()
