
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from pect.config import PECTConfig
from pect.trainer import PECTRunner
from pect.utils import get_logger


def parse_seeds(s: str):
    seeds = []
    for part in s.split(","):
        if "-" in part:
            a, b = part.split("-")
            seeds.extend(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["configs/default.yaml"])
    ap.add_argument("--set", nargs="*", default=[], help="overrides, e.g. model.rank=20 loss.eta1=0.01")
    ap.add_argument("--seeds", default="0", help='e.g. "0", "0-19" or "0,3,5"')
    ap.add_argument("--out", default=None, help="output directory (default: train.out_dir)")
    args = ap.parse_args()

    cfg = PECTConfig.from_yaml(*args.config).override(args.set)
    out = args.out or cfg.train.out_dir
    os.makedirs(out, exist_ok=True)
    cfg.save(os.path.join(out, "config.yaml"))
    log = get_logger("pect", os.path.join(out, "train.log"))
    log.info(f"Config: {json.dumps(cfg.to_dict())}")

    runner = PECTRunner(cfg, log).prepare()
    results = [runner.run(seed, os.path.join(out, f"seed{seed}")) for seed in parse_seeds(args.seeds)]

    summary = {"n_runs": len(results)}
    for key in ("ARI", "NMI", "silhouette", "best_epoch"):
        vals = [r[key] for r in results if key in r]
        if vals:
            summary[key] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "values": vals}
    with open(os.path.join(out, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    msg = ", ".join(f"{k} {v['mean']:.2f}±{v['std']:.2f}" for k, v in summary.items() if isinstance(v, dict))
    log.info(f"Summary over {len(results)} run(s): {msg}")


if __name__ == "__main__":
    main()
