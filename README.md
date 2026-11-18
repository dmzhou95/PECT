# PECT: Parameter-Efficient Co-Tuning for Spatial Transcriptomics

## 1. Directory Structure

```
PECT/
├── train.py
├── configs/
│   ├── default.yaml
│   ├── dlpfc.yaml / hbc.yaml / mba.yaml
│   └── synthetic.yaml
├── pect/
│   ├── config.py
│   ├── data/
│   │   ├── preprocess.py
│   │   ├── graphs.py
│   │   ├── sequences.py
│   │   └── dataset.py
│   ├── backbone/
│   │   ├── nucleotide_transformer.py
│   │   └── cache.py
│   ├── models/
│   │   ├── sge.py  ssa.py  mmct.py  pooling.py  decoder.py
│   │   └── pect.py
│   ├── losses.py
│   ├── clustering.py
│   └── trainer.py
├── scripts/
│   ├── extract_tss_sequences.py
│   ├── cache_backbone.py
│   ├── hparam_search.py
│   ├── summarize.py
│   ├── make_synthetic.py
│   ├── run_dlpfc.sh
│   └── smoke_test.sh
└── tests/
    ├── test_theory.py
    └── test_model.py
```

---

## 2. Environment

Environment used in the paper: Python 3.9, PyTorch 1.12, and a single RTX 4060 GPU (8 GB).

```bash
conda create -n pect python=3.9 -y && conda activate pect
pip install torch==1.12.1 --extra-index-url https://download.pytorch.org/whl/cu113   # Choose according to your CUDA version
pip install -r requirements.txt

conda install -c conda-forge r-base r-mclust rpy2 -y
```

---

## 3. Data Preparation

### 3.1 Spatial Transcriptomics Sections

| Dataset | Sections | Spots | Number of domains (K) | Configuration |
|---|---|---|---|---|
| DLPFC (spatialLIBD) | 151507–151510, 151669–151676 | 3,460–4,789 | 7 (5 for 151669–151672) | `configs/dlpfc.yaml` |
| HBC (10x Visium Human Breast Cancer, Block A Section 1) | 1 | 3,798 | 20 | `configs/hbc.yaml` |
| MBA (10x Visium Mouse Brain Anterior) | 1 | 2,695 | 52 | `configs/mba.yaml` |

For DLPFC, use the directory layout commonly used by GraphST and SEDR:

```
data/DLPFC/151507/
├── filtered_feature_bc_matrix.h5
├── spatial/ (tissue_positions_list.csv, scalefactors_json.json, images)
└── 151507_truth.txt        # One "barcode<TAB>layer" entry per line, with no header
```

### 3.2 Reference Genomes

An uncompressed genome FASTA file and its corresponding GTF file are required. Gene records in the GTF must contain `gene_name` and `gene_id`:

* Human: GRCh38 (DLPFC, HBC)
* Mouse: GRCm39 (MBA)

For example, use Ensembl's `*.dna.primary_assembly.fa` and `*.gtf.gz` files. Chromosome names in either `1` or `chr1` format can be matched.
Windows for genes on the negative strand are reverse complemented so that sequences follow the gene's 5′→3′ direction.

---

## 4. Running PECT

### 4.1 Run DLPFC in One Step

```bash
DATA=data/DLPFC REF_FA=ref/GRCh38.fa REF_GTF=ref/GRCh38.gtf.gz bash scripts/run_dlpfc.sh
```

### 4.2 Run Step by Step (Using 151507 as an Example)

```bash
# Step 1: Extract TSS sequences for HVGs (use --append to combine multiple sections into one file)
python scripts/extract_tss_sequences.py --config configs/default.yaml configs/dlpfc.yaml \
    --set data.path=data/DLPFC/151507 \
    --fasta ref/GRCh38.fa --gtf ref/GRCh38.gtf.gz --out seqs/dlpfc_hvg.fa --append

# Step 2: Run each gene through the frozen Nucleotide Transformer once and cache blocks 21–24
python scripts/cache_backbone.py --fasta seqs/dlpfc_hvg.fa --out cache/nt500m_human \
    --model InstaDeepAI/nucleotide-transformer-500m-human-ref --blocks 21 22 23 24

# Step 3: Train, cluster, and evaluate 
python train.py --config configs/default.yaml configs/dlpfc.yaml \
    --set data.path=data/DLPFC/151507 data.label_file=data/DLPFC/151507/151507_truth.txt \
          data.n_clusters=7 backbone.cache_dir=cache/nt500m_human \
    --seeds 0-19 --out outputs/DLPFC/151507

# Step 4: Summarize results
python scripts/summarize.py outputs/DLPFC
```

For HBC or MBA, change the configuration file and set `data.label_file` and `data.label_col`. Extract MBA sequences using GRCm39.

### 4.3 Smoke Test (CPU, a Few Minutes, No Model Download Required)

```bash
bash scripts/smoke_test.sh
```
