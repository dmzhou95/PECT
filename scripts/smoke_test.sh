
set -euo pipefail
python scripts/make_synthetic.py --out data/synthetic
python scripts/extract_tss_sequences.py --config configs/default.yaml configs/synthetic.yaml \
    --fasta data/synthetic/genome.fa --gtf data/synthetic/genes.gtf --out seqs/toy.fa
python scripts/cache_backbone.py --fasta seqs/toy.fa --out cache/toy --model toy --blocks 3 4
python -m pytest -q tests
python train.py --config configs/default.yaml configs/synthetic.yaml --seeds 0-1 --out outputs/smoke
