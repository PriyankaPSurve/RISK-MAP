#!/usr/bin/env python3
"""
monte_carlo_safer.py  –  Jitter I_j and W_i and record SAFE-R variance.

USAGE
  python monte_carlo_safer.py \
      --matrix attacks_vs_defenses_normalised.csv \
      --weights attack_weights.csv \
      --impl   G1_EDU_implementation_status.csv \
      --runs   1000 \
      --jitter 0.25                       # ±25 %

Outputs
  • sensitivity_G1_EDU.csv   (run-by-run ⅁R values)
  • table_sensitivity.csv    (mean, σ, min, max for all robots)
  • histogram_G1_EDU.png     (optional, for the notebook)
"""

import argparse, pathlib, random
import numpy as np, pandas as pd

def load(matrix, weights, impl):
    A = pd.read_csv(matrix, index_col="Attack Vector")
    W = pd.read_csv(weights, index_col="Attack Vector")["Weight"]
    I = pd.read_csv(impl, index_col="Defence")["Implementation"]

    common = A.columns.intersection(I.index)
    A, I = A[common], I[common]
    return A, W, I

def score(A, W, I):
    E = A.mul(I, axis=1)
    C = 1 - (1 - E).prod(axis=1)
    return (W * C).sum() / W.sum()

def jitter_series(s: pd.Series, jitter: float):
    low, high = 1-jitter, 1+jitter
    return s * pd.Series(np.random.uniform(low, high, size=len(s)), index=s.index)

def run_mc(name, A, W, I, n, jitter):
    values = []
    for _ in range(n):
        Wj = jitter_series(W, jitter)
        Ij = jitter_series(I, jitter)
        values.append(score(A, Wj, Ij))
    out = pd.Series(values, name=name)
    out.to_csv(f"sensitivity_{name}.csv", index=False)
    return out

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--impl", nargs="+", required=True,
                   help="One or more implementation CSVs")
    p.add_argument("--runs", type=int, default=1000)
    p.add_argument("--jitter", type=float, default=0.25)
    args = p.parse_args()

    stats = []
    for path in args.impl:
        name = pathlib.Path(path).stem.replace("_implementation_status", "")
        A, W, I = load(args.matrix, args.weights, path)
        series = run_mc(name, A, W, I, args.runs, args.jitter)
        stats.append({
            "Robot": name,
            "Mean": series.mean(),
            "Std": series.std(),
            "Min": series.min(),
            "Max": series.max()
        })
    pd.DataFrame(stats).to_csv("table_sensitivity.csv", index=False)
    print("[✓] Wrote table_sensitivity.csv and per-robot CSVs")

if __name__ == "__main__":
    main()
