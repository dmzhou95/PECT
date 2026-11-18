
set -euo pipefail
DATA=${DATA:-data/DLPFC}
REF_FA=${REF_FA:-ref/Homo_sapiens.GRCh38.dna.primary_assembly.fa}
REF_GTF=${REF_GTF:-ref/Homo_sapiens.GRCh38.gtf.gz}
SEQ=${SEQ:-seqs/dlpfc_hvg.fa}
CACHE=${CACHE:-cache/nt500m_human_dlpfc}
SEEDS=${SEEDS:-0-19}
SECTIONS="151507 151508 151509 151510 151669 151670 151671 151672 151673 151674 151675 151676"

# 1) union of the HVGs of all sections -> one sequence file and one cache
for s in $SECTIONS; do
  python scripts/extract_tss_sequences.py --config configs/default.yaml configs/dlpfc.yaml \
      --set data.path=$DATA/$s --fasta $REF_FA --gtf $REF_GTF --out $SEQ --append
done
# 2) frozen Nucleotide Transformer, run once per gene
if [ ! -f "$CACHE/meta.json" ]; then
  python scripts/cache_backbone.py --fasta $SEQ --out $CACHE --blocks 21 22 23 24
fi
# 3) training + clustering
for s in $SECTIONS; do
  case $s in 151669|151670|151671|151672) K=5 ;; *) K=7 ;; esac
  python train.py --config configs/default.yaml configs/dlpfc.yaml \
      --set data.path=$DATA/$s data.label_file=$DATA/$s/${s}_truth.txt \
            data.n_clusters=$K backbone.cache_dir=$CACHE \
      --seeds $SEEDS --out outputs/DLPFC/$s
done
python scripts/summarize.py outputs/DLPFC
