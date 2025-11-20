#!/usr/bin/env python3
"""
combined_heatmap.py – annotated heat-map of the 10 highest residual-risk
attack vectors for Digit, G1 EDU and Pepper.

Run (PowerShell, use back-ticks for line continuation):

python combined_heatmap.py `
    --matrix  "..\\data\\attacks_vs_defenses_normalised.csv" `
    --weights "..\\data\\attack_weights.csv" `
    --digit   "..\\data\\Digit_implementation_status.csv" `
    --g1      "..\\data\\G1_EDU_implementation_status.csv" `
    --pepper  "..\\data\\Pepper_implementation_status.csv" `
    --out     "combined_heatmap_safe_r.png"
"""
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib

# ────────────────────────── helpers ──────────────────────────
def load_matrix(path):
    return pd.read_csv(path, index_col="Attack Vector")

def load_weights(path):
    return pd.read_csv(path, index_col="Attack Vector")["Weight"]

def load_impl(path):
    return pd.read_csv(path, index_col="Defence")["Implementation"]

def residual_risk(A, W, I):
    common = A.columns.intersection(I.index)
    E = A[common].mul(I[common], axis=1)
    C = 1 - (1 - E).prod(axis=1)
    return W * (1 - C)

def top10_residuals(A, W, impl_csv):
    return residual_risk(A, W, load_impl(impl_csv)).nlargest(10)

# ────────────────────── plotting ─────────────────────────────
def annotated_heatmap(pivot: pd.DataFrame, outfile: Path):
    pivot   = pivot.reindex(columns=["Digit", "G1_EDU", "Pepper"])
    display = pivot.fillna(-1)                       # sentinel → grey

    # palette: try seaborn icefire → fallback inferno
    # try:
    #     import seaborn as sns
    #     cmap = sns.color_palette("icefire", as_cmap=True)
    # except (ImportError, ValueError):
    #     cmap = matplotlib.colormaps.get_cmap("inferno")
    #     print("[i] Using fallback palette 'inferno'.")
    # cmap = cmap.copy()
    # cmap.set_under("#e0e0e0")                        # grey for sentinel
    # ── custom 4-step palette ─────────────────────────────────────────────
    # from matplotlib.colors import LinearSegmentedColormap
    # custom_cols = ["#916075",   # deep navy
    #             "#7bc9cc",   # dark violet
    #             "#f1f4dd"]
    #             # ,   # mid-violet
    #             # "#F8B558"]   # beige
    # cmap = LinearSegmentedColormap.from_list("custom_risk", custom_cols, N=256)
    # cmap.set_under("#fefdff")                   # grey for “not in top-10”
# ──────────────────────────────────────────────────────────────────────
    # ── viridis palette (purple → yellow) ───────────────────────────────
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap("viridis").copy()
    cmap.set_under("#e0e0e0")                    # grey for “not in top-10”
    # ────────────────────────────────────────────────────────────────────


    vmax = pivot.max().max()
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(display, vmin=0, vmax=vmax,
                   cmap=cmap, aspect="auto")

    # ticks & labels
    ax.set_xticks(range(len(display.columns)))
    ax.set_xticklabels(display.columns, fontsize=9)
    ax.set_yticks(range(len(display.index)))
    ax.set_yticklabels(display.index, fontsize=8)

    # value annotations
    for i in range(display.shape[0]):
        for j in range(display.shape[1]):
            val = pivot.iat[i, j]
            if pd.notna(val):
                txt_colour = "white" if val > vmax * 0.6 else "black"
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color=txt_colour)

    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label("Residual risk  (higher = worse)",
                   rotation=270, labelpad=15)

    ax.set_title("Top-10 Residual Risks Across Robots", pad=12,
                 fontsize=13, weight="bold")

    fig.tight_layout()
    fig.savefig(outfile, dpi=300,format='pdf', bbox_inches='tight')
    print(f"[✓] Heat-map saved to {outfile.resolve()}")

# ────────────────────────── main ────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--matrix",  required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--digit",   required=True)
    p.add_argument("--g1",      required=True)
    p.add_argument("--pepper",  required=True)
    p.add_argument("--out",     default="combined_heatmap_safe_r.png")
    args = p.parse_args()

    A = load_matrix(args.matrix)
    W = load_weights(args.weights)

    robots = {"Digit": args.digit,
              "G1_EDU": args.g1,
              "Pepper": args.pepper}

    frames = []
    for name, csv in robots.items():
        s  = top10_residuals(A, W, csv)
        df = s.reset_index()
        df.columns = ["Attack", "Residual"]
        df["Robot"] = name
        frames.append(df)

    heat  = pd.concat(frames, ignore_index=True)
    pivot = heat.pivot_table(index="Attack",
                             columns="Robot",
                             values="Residual",
                             aggfunc="first")

    annotated_heatmap(pivot, Path(args.out))

if __name__ == "__main__":
    main()
