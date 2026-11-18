
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pect.data.sequences import read_fasta  # noqa: E402
from pect.utils import get_logger  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fasta", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", default="InstaDeepAI/nucleotide-transformer-500m-human-ref",
                    help='HF model id, or "toy"')
    ap.add_argument("--blocks", type=int, nargs="+", default=[21, 22, 23, 24], help="1-indexed adapter blocks")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--no_fp16", action="store_true")
    # toy encoder options
    ap.add_argument("--toy_dim", type=int, default=64)
    ap.add_argument("--toy_blocks", type=int, default=4)
    args = ap.parse_args()
    log = get_logger("pect")

    seqs = read_fasta(args.fasta)
    genes = list(seqs.keys())
    log.info(f"{len(genes)} sequences read from {args.fasta}")
    if args.model == "toy":
        from pect.backbone.toy import cache_toy_encoder

        cache_toy_encoder(genes, [seqs[g] for g in genes], args.out, d_s=args.toy_dim,
                          n_blocks=args.toy_blocks, blocks=args.blocks, logger=log)
    else:
        from pect.backbone.nucleotide_transformer import cache_nucleotide_transformer

        cache_nucleotide_transformer(genes, [seqs[g] for g in genes], args.out, model_name=args.model,
                                     blocks=args.blocks, device=args.device, batch_size=args.batch_size,
                                     fp16=not args.no_fp16, logger=log)
    log.info(f"Cache written to {args.out}")


if __name__ == "__main__":
    main()
