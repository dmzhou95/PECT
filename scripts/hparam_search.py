
from __future__ import annotations

import argparse
import ast
import itertools
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pect.config import PECTConfig  # noqa: E402
from pect.trainer import PECTRunner  # noqa: E402
from pect.utils import get_logger  # noqa: E402


def parse_grid(spec: str):
    keys, values = [], []
    for part in spec.split(";"):
        k, v = part.strip().split("=", 1)
        keys.append(k.strip())
        values.append(ast.literal_eval(v.strip()))
    return keys, values


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", nargs="+", default=["configs/default.yaml"])
    ap.add_argument("--set", nargs="*", default=[])
    ap.add_argument("--grid", action="append", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/hparam")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    log = get_logger("pect", os.path.join(args.out, "search.log"))

    cfg = PECTConfig.from_yaml(*args.config).override(args.set)
    cfg.train.save_checkpoint = False
    runner = PECTRunner(cfg, log).prepare()
    chosen, records = [], []
    for spec in args.grid:
        keys, values = parse_grid(spec)
        best = (-2.0, None)
        for combo in itertools.product(*values):
            sets = [f"{k}={v}" for k, v in zip(keys, combo)]
            runner.cfg = cfg.override(chosen + sets)
            res = runner.run(args.seed)
            records.append({"settings": chosen + sets, **res})
            log.info(f"{sets}: silhouette={res['silhouette']:.4f} (ARI={res.get('ARI', float('nan')):.2f})")
            if res["silhouette"] > best[0]:
                best = (res["silhouette"], sets)
        chosen += best[1]
        log.info(f"Selected {best[1]} (silhouette {best[0]:.4f})")
    with open(os.path.join(args.out, "search.json"), "w") as f:
        json.dump({"selected": chosen, "runs": records}, f, indent=2)
    log.info(f"Final selection: {chosen}")


if __name__ == "__main__":
    main()
