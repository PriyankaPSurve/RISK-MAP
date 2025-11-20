#!/usr/bin/env python3
"""
score_safe_r.py  compute SAFEᴿ overall and per-layer scores
                 + write heat-map & radar chart.

USAGE
python score_safe_r.py `
    --matrix  ..\data\attacks_vs_defenses_normalised.csv `
    --weights ..\data\attack_weights.csv `
    ...
"""

import argparse, pathlib, textwrap
import numpy as np, pandas as pd, matplotlib.pyplot as plt

LAYER_MAP = {
    'P':  'Physical', 'SP': 'Sensor', 'DP': 'Data',
    'MW': 'Middleware', 'DM': 'Decision',
    'AP': 'Application', 'SI': 'Social_Interface'
}
LAYER_ORDER = ['Physical','Sensor','Data','Middleware',
               'Decision','Application','Social_Interface']

# ──────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=textwrap.dedent(__doc__))
    p.add_argument("--matrix",  required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--impl",    required=True, nargs='+',
                   help="One or more implementation CSVs")
    p.add_argument("--outdir",  default="figs",
                   help="Folder to write <robot>/heatmap.png & radar.pdf")
    return p.parse_args()

# ──────────────────────────────────────────────────────────────────────────
def layer_of(defence_id: str) -> str:
    return LAYER_MAP.get(defence_id.split('-')[0], 'Other')

def compute_scores(A, W, I):
    common = A.columns.intersection(I.index)
    E = A[common].mul(I[common], axis=1)     # effective coverage
    C = 1 - (1 - E).prod(axis=1)             # per-attack combined coverage
    overall = (W * C).sum() / W.sum()

    layer_scores = {}
    for lyr in LAYER_ORDER:
        cols = [c for c in E.columns if layer_of(c) == lyr]
        if cols:
            C_lyr = 1 - (1 - E[cols]).prod(axis=1)
            layer_scores[lyr] = (W * C_lyr).sum() / W.sum()
        else:
            layer_scores[lyr] = 0.0
    return overall, layer_scores, E

# ───────────────────────── radar_plot (show 0-10 numbers, 0-5 axis) ──
def radar_plot(layer_scores, overall_pct, out_pdf):
    labels  = list(layer_scores.keys())

    # keep the 0–10 numbers
    # radii   = [v * 10 for v in layer_scores.values()]   # e.g. 0.36 → 3.6
    true_values = [v * 5 for v in layer_scores.values()]
    radii = [np.sqrt(x) for x in true_values]   # monotonic expansion

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    labels.append(labels[0]);  radii.append(radii[0])
    angles = np.append(angles, angles[0])

    fig, ax = plt.subplots(figsize=(6.5, 6.5),
                           subplot_kw=dict(polar=True))

    ax.plot(angles, radii, marker="o", linewidth=2.2, color="#1f77b4")
    ax.fill(angles, radii, alpha=0.25, color="#1f77b4")

    # 0–5 axis (major rings every 1, dotted minor every 0.5)
    ax.set_ylim(0, 5)
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0.5, 5, 0.5), minor=True)
    ax.grid(True, which="major", color="grey", alpha=.6)
    ax.grid(True, which="minor", color="grey", linestyle=":", alpha=.3)

    ax.set_xticks(angles[:-1]);  ax.set_xticklabels(labels[:-1])
    ax.set_title("SAFE-R Layer Coverage (0–5 display)\n"
                 f"Overall: {overall_pct:.1f} %", pad=20)

    # put the original 0–10 number next to each point
    for ang, r in zip(angles[:-1], radii[:-1]):
        ax.text(ang, r + 0.15, f"{r:.1f}",
                ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
# ───────────────────────── combined_radar_plot ─────────────────────────
def combined_radar_plot(all_scores: dict[str, dict[str, float]],
                        out_pdf: pathlib.Path):
    """Plot one radar with every robot over-plotted."""
    labels  = LAYER_ORDER
    angles  = np.linspace(0, 2*np.pi, len(labels), endpoint=False)
    angles  = np.append(angles, angles[0])

    colours = ["#1f77b4", "#d62728", "#2ca02c", "#ff7f0e",
               "#9467bd", "#17becf", "#8c564b"]           # extend if needed

    fig, ax = plt.subplots(figsize=(6.5, 6.5),
                           subplot_kw=dict(polar=True))

    # 0–5 axis with minor grid every 0.5
    ax.set_ylim(0, 5)
    ax.set_yticks(np.arange(0, 6, 1))
    ax.set_yticks(np.arange(0.5, 5, 0.5), minor=True)
    ax.grid(True, which="major", color="grey", alpha=.6)
    ax.grid(True, which="minor", color="grey", linestyle=":", alpha=.3)
    ax.set_xticks(angles[:-1]);  ax.set_xticklabels(labels)

    for i, (robot, lyr_dict) in enumerate(all_scores.items()):
        radii = [v*10 for v in lyr_dict.values()]     # keep 0-10 numbers
        radii.append(radii[0])                        # close loop
        c = colours[i % len(colours)]
        ax.plot(angles, radii, marker='o', linewidth=2, color=c, label=robot)
        ax.fill(angles, radii, alpha=0.15, color=c)

        # annotate each vertex with the 0-10 figure
        for ang, r in zip(angles[:-1], radii[:-1]):
            ax.text(ang, r + 0.15, f"{r:.1f}",
                    ha="center", va="bottom", fontsize=7, color=c)

    ax.set_title("SAFE-R Layer Coverage – combined view (display 0–5)",
                 pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), frameon=False)
    fig.tight_layout()
    fig.savefig(out_pdf)
    plt.close(fig)
# ───────────────────────────────────────────────────────────────────────



# (Optional) quick “top-10” heat-map – remove if not needed
def heatmap_top10(E, W, out_png):
    risk = W * (1 - E.sum(axis=1))
    idx = risk.nlargest(10).index
    data = E.loc[idx]
    if data.empty: return
    plt.figure(figsize=(10,4)); plt.imshow(data, aspect='auto')
    plt.yticks(range(len(idx)), idx, fontsize=7)
    plt.xticks(range(len(data.columns)), data.columns, rotation=90, fontsize=6)
    plt.colorbar(label='Coverage'); plt.tight_layout(); plt.savefig(out_png); plt.close()

# ──────────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    A = pd.read_csv(args.matrix,  index_col='Attack Vector')
    W = pd.read_csv(args.weights, index_col='Attack Vector')['Weight']

    out_root = pathlib.Path(args.outdir);  out_root.mkdir(exist_ok=True)

    all_layer_scores = {}                          

    for impl_path in args.impl:
        impl_path = pathlib.Path(impl_path)
        robot = impl_path.stem.replace('_implementation_status','')
        I = pd.read_csv(impl_path, index_col='Defence')['Implementation']

        overall, layer_scores, E = compute_scores(A, W, I)
        pct = overall * 100

        all_layer_scores[robot] = layer_scores          

        rdir = out_root/robot;  rdir.mkdir(exist_ok=True)
        radar_plot(layer_scores, pct, rdir/'radar.pdf')
        heatmap_top10(E, W, rdir/'heatmap.png')
        print(f"[✓] {robot:>8}: SAFE-R {pct:5.1f} %  figs saved → {rdir}")

    if len(all_layer_scores) > 1:                
        combined_pdf = out_root / "SAFE_R_combined_radar.pdf"
        combined_radar_plot(all_layer_scores, combined_pdf)
        print(f"[✓] Combined radar saved → {combined_pdf}")

if __name__ == "__main__":     
    main()