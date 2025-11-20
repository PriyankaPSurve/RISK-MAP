#!/usr/bin/env python3
"""
heatmap.py – Visualise SAFE-R effective coverage for the 10 riskiest attacks.

Assumes the following files are in the same directory:
  • attacks_vs_defenses_normalised.csv
  • G1_EDU_implementation_status_updated.csv
  • attack_weights.csv
"""

import pathlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE = pathlib.Path(__file__).resolve().parent

MATRIX   = BASE / "attacks_vs_defenses_normalised.csv"
MATURITY = BASE / "Digit_implementation_status.csv"
WEIGHTS  = BASE / "attack_weights.csv"
OUTPNG   = BASE / "effective_coverage_heatmap.png"

def load():
    A = pd.read_csv(MATRIX, index_col="Attack Vector")
    I = pd.read_csv(MATURITY, index_col="Defence")["Implementation"]
    W = pd.read_csv(WEIGHTS, index_col="Attack Vector")["Weight"]

    # Safely align defenses that exist in both A and I
    common_defs = A.columns.intersection(I.index)
    A = A[common_defs]
    I = I[common_defs]

    return A, I, W


def compute_top10(A: pd.DataFrame, I: pd.Series, W: pd.Series):
    E = A.mul(I, axis=1)                               # discount by maturity
    C = 1 - (1 - E).prod(axis=1)                       # combined coverage
    residual = W * (1 - C)                             # weighted risk
    top10 = residual.sort_values(ascending=False).head(10).index
    return E.loc[top10]

def plot_heatmap(heat: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(heat.values, aspect="auto", cmap="Purples")
    # y-axis: attack names
    ax.set_yticks(range(len(heat)))
    ax.set_yticklabels(heat.index, fontsize=8)
    # x-axis: abbrev. mitigation IDs
    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels([c.split()[0] for c in heat.columns],
                       rotation=90, ha="center", fontsize=6)
    ax.set_title("Effective Coverage E₍ᵢⱼ₎ – Top-10 Weighted-Risk Attacks",
                 pad=12, fontsize=11, weight="bold")
    cbar = fig.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label("Coverage (0 = none … 1 = full)", rotation=270, labelpad=15)
    fig.tight_layout()
    fig.savefig(OUTPNG, dpi=300, bbox_inches="tight")
    print(f"[+] Heat-map saved to {OUTPNG}")

def main():
    A, I, W = load()
    heat = compute_top10(A, I, W)
    plot_heatmap(heat)
    plt.show()

if __name__ == "__main__":
    main()
