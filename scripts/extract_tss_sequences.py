
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pect.config import PECTConfig  # noqa: E402
from pect.data.preprocess import load_section, preprocess  # noqa: E402
from pect.data.sequences import extract_tss_sequences, read_fasta, read_gtf_tss, write_fasta  # noqa: E402
from pect.utils import get_logger  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["configs/default.yaml"])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--fasta", required=True, help="uncompressed genome FASTA (GRCh38 / GRCm39)")
    ap.add_argument("--gtf", required=True, help="matching GTF annotation (.gtf or .gtf.gz)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--length", type=int, default=1000)
    ap.add_argument("--append", action="store_true", help="merge with an existing --out file")
    ap.add_argument("--all_genes", action="store_true", help="export all genes of the GTF instead of HVGs")
    args = ap.parse_args()
    log = get_logger("pect")

    if args.all_genes:
        by_name, _ = read_gtf_tss(args.gtf)
        genes, gene_ids = sorted(by_name), None
    else:
        cfg = PECTConfig.from_yaml(*args.config).override(args.set)
        st = preprocess(load_section(cfg.data), cfg.data, log)
        genes, gene_ids = list(st.gene_names), st.gene_ids

    seqs, found = extract_tss_sequences(genes, args.fasta, args.gtf, args.length, gene_ids)
    log.info(f"Found TSS for {sum(found.values())}/{len(found)} genes")
    missing = [g for g, ok in found.items() if not ok]
    if missing:
        log.info(f"Missing (first 20): {missing[:20]}")
    if args.append and os.path.exists(args.out):
        old = read_fasta(args.out)
        old.update(seqs)
        seqs = old
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    write_fasta(seqs, args.out)
    log.info(f"Wrote {len(seqs)} sequences to {args.out}")


if __name__ == "__main__":
    main()
