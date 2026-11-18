
from __future__ import annotations

import ast
import copy
from dataclasses import asdict, dataclass, field, fields
from typing import List, Optional

import yaml


@dataclass
class DataConfig:
    # "visium" (10x folder), "h5ad" or "npz" (synthetic / pre-exported arrays)
    format: str = "visium"
    path: str = ""
    count_file: str = "filtered_feature_bc_matrix.h5"
    label_file: Optional[str] = None      # optional TSV/CSV with barcode -> label
    label_col: Optional[str] = None       # column of label_file or obs key in h5ad
    label_sep: str = "\t"
    label_header: bool = True             # False for headerless files such as 151507_truth.txt
    species: str = "human"               # human -> GRCh38, mouse -> GRCm39
    n_clusters: int = 7                   # K, number of spatial domains
    # preprocessing (Section 4.1)
    min_spots: int = 3
    n_hvg: int = 3000
    max_genes_per_spot: int = 64          # M
    n_pca: int = 50
    # graphs (Section 3.1)
    k_spatial: int = 6
    radius_um: Optional[float] = 150.0
    um_per_unit: Optional[float] = None   # coordinate scale; inferred for Visium
    k_feature: int = 15
    omega: float = 0.5


@dataclass
class BackboneConfig:
    cache_dir: str = ""                   # output of scripts/cache_backbone.py
    model_name: str = "InstaDeepAI/nucleotide-transformer-500m-human-ref"
    n_blocks: int = 24                    # K_b
    adapter_blocks: List[int] = field(default_factory=lambda: [21, 22, 23, 24])
    seq_len_bp: int = 1000
    preload_tokens_to_gpu: bool = False   # True only if GPU memory allows


@dataclass
class ModelConfig:
    d_g: int = 64
    gate_hidden: int = 64
    d_a: int = 128
    d_c: int = 128
    n_heads: int = 4
    rank: int = 30                        # r
    lora_alpha: float = 0.6               # alpha, scale s = alpha / r
    alpha_g: float = 0.2                  # smoothing step, 0 <= alpha_g <= 1/2
    dec_hidden: int = 256


@dataclass
class LossConfig:
    tau: float = 0.5
    eta1: float = 0.1                     # weight of L_MI
    eta2: float = 0.1                     # weight of L_reg
    lam: float = 0.5                      # lambda in L_reg (adapter magnitude)
    gamma: float = 0.5                    # gamma in L_reg (KL alignment)
    # switches for ablations (RQ2)
    use_zinb: bool = True
    use_graph: bool = True
    use_dec: bool = True
    use_mi: bool = True
    use_reg: bool = True
    use_sm: bool = True


@dataclass
class TrainConfig:
    seed: int = 0
    device: str = "cuda"
    batch_size: int = 128                 # |B| target spots
    infer_batch_size: int = 512
    max_epochs: int = 60
    patience: int = 10
    lr_graph: float = 1e-3                # theta_g
    lr_adapter: float = 5e-4              # theta_a
    weight_decay: float = 1e-4
    token_chunk: int = 4096               # spot-gene pairs per checkpointed chunk
    recompute_tokens: bool = True         # re-read cached token states in backward (saves GPU memory)
    grad_clip: Optional[float] = None     # not used in the paper; set e.g. 5.0 if training is unstable
    save_checkpoint: bool = True
    # clustering
    cluster_method: str = "mclust"        # mclust (rpy2) or "gmm" (sklearn, tied cov.)
    mclust_model: str = "EEE"
    out_dir: str = "outputs"
    log_every: int = 5                    # epochs between log lines


@dataclass
class PECTConfig:
    data: DataConfig = field(default_factory=DataConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    loss: LossConfig = field(default_factory=LossConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    # ------------------------------------------------------------------
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "PECTConfig":
        cfg = cls()
        for sec in fields(cls):
            if sec.name not in (d or {}):
                continue
            sub = getattr(cfg, sec.name)
            for k, v in d[sec.name].items():
                if not hasattr(sub, k):
                    raise KeyError(f"Unknown config key: {sec.name}.{k}")
                setattr(sub, k, v)
        return cfg

    @classmethod
    def from_yaml(cls, *paths: str) -> "PECTConfig":
        """Load one or more YAML files; later files override earlier ones."""
        merged: dict = {}
        for p in paths:
            with open(p, "r", encoding="utf-8") as f:
                d = yaml.safe_load(f) or {}
            for sec, kv in d.items():
                merged.setdefault(sec, {}).update(kv or {})
        return cls.from_dict(merged)

    def override(self, items: Optional[List[str]]) -> "PECTConfig":
        """Apply CLI overrides such as ``model.rank=20 loss.eta1=0.01``."""
        cfg = copy.deepcopy(self)
        for item in items or []:
            key, val = item.split("=", 1)
            sec, name = key.split(".", 1)
            sub = getattr(cfg, sec)
            if not hasattr(sub, name):
                raise KeyError(f"Unknown config key: {key}")
            try:
                parsed = ast.literal_eval(val)
            except (ValueError, SyntaxError):
                parsed = val
            if isinstance(parsed, str) and parsed.lower() in ("none", "null"):
                parsed = None
            setattr(sub, name, parsed)
        return cfg

    def save(self, path: str):
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)
