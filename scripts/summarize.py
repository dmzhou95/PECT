
from __future__ import annotations

import argparse
import glob
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()
    rows = []
    for path in sorted(glob.glob(os.path.join(args.root, "*", "summary.json"))):
        with open(path) as f:
            s = json.load(f)
        row = {"section": os.path.basename(os.path.dirname(path)), "runs": s.get("n_runs")}
        for k in ("NMI", "ARI", "silhouette"):
            if k in s:
                row[k] = f"{s[k]['mean']:.2f}±{s[k]['std']:.2f}"
                row[k + "_mean"] = s[k]["mean"]
        rows.append(row)
    if not rows:
        print("No summary.json found")
        return
    df = pd.DataFrame(rows)
    show = [c for c in ("section", "runs", "NMI", "ARI", "silhouette") if c in df]
    print(df[show].to_string(index=False))
    for k in ("NMI", "ARI"):
        if k + "_mean" in df:
            print(f"Average {k}: {df[k + '_mean'].mean():.2f}")
    if args.csv:
        df.to_csv(args.csv, index=False)


if __name__ == "__main__":
    main()
