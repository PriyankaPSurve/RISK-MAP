# RISK-MAP: Security Assessment Framework for Humanoid Robots

RISK-MAP is a quantitative security assessment artifact for humanoid robot platforms. It implements the **RISK-MAP** (Security Assessment Framework for Engineering Resilience) scoring methodology, cross-layer cascade risk modeling, and sensitivity analysis across three humanoid robots: **Digit**, **G1 EDU**, and **Pepper**.

## Overview

The framework evaluates robot security across **7 architectural layers**:

| Layer | Description |
|---|---|
| Physical | Hardware, actuators, power systems |
| Sensor | Cameras, LiDAR, IMU, acoustic sensors |
| Data | Memory, timing, concurrency |
| Middleware | ROS/DDS communication fabric |
| Decision | Planning, estimation, control |
| Application | Scripts, OTA updates, APIs |
| Social Interface | ASR, vision, human interaction |

Against **39 attack vectors** and **35 defense mechanisms**, the tool computes residual risk scores, models cascade propagation between layers, and compares platform-level security posture.

## Repository Structure

```
artifact/
└── RISK_MAP/
    ├── data/                        # Input CSVs (attack matrices, weights, defense status)
    │   ├── attacks_vs_defenses_normalised.csv
    │   ├── attack_weights.csv
    │   ├── structural_feasibility.csv
    │   ├── empirical_evidence.csv
    │   ├── {Robot}_implementation_status.csv
    │   ├── {Robot}_mitigation.csv
    │   └── layer_scores/            # Computed per-layer RISK-MAP scores
    ├── scripts/                     # Python analysis scripts
    │   ├── score_safe_r.py          # Core RISK-MAP scoring engine
    │   ├── monte_carlo_safer.py     # Monte Carlo sensitivity analysis
    │   ├── cross_layer_propagation.py  # Cascade risk metrics (CRR & CCI)
    │   ├── heatmap.py               # Per-robot top-10 risk heatmap
    │   ├── combined_heatmap.py      # Multi-robot risk comparison heatmap
    │   └── extract_layer_scores.py  # Layer score extraction utility
    ├── figs/                        # Generated figures (PNG, PDF)
    │   ├── {Robot}/                 # Per-platform radar and heatmap charts
    │   └── cascade/                 # Cross-layer propagation visualizations
    ├── Information/                 # Robot specification documents (Excel)
    └── create_mitigation_files.ps1  # Generates cross-layer mitigation CSVs
```

## Key Metrics

- **RISK-MAP Score** (0–1): Weighted overall security index per platform
- **Layer Scores** (0–5): Security depth per architectural layer
- **CRR** (Cascade Risk Ratio): Cross-layer attack propagation potential (lower is safer)
- **CCI** (Cascade Containment Index): Mitigation effectiveness across layers (higher is better)
- **Residual Risk**: Weighted attack coverage gaps remaining after defenses are applied

## Scripts

### `score_safe_r.py`

Core scoring engine. Computes RISK-MAP scores from the attack-defense effectiveness matrix, attack weight vector, and platform-specific defense implementation levels. Outputs per-layer scores, an overall score, a radar chart, and a top-10 risk heatmap.

```bash
python scripts/score_safe_r.py
```

### `monte_carlo_safer.py`

Runs Monte Carlo sensitivity analysis (default: 1000 iterations) with ±25% jitter on weights and implementation levels. Reports mean, std, min, and max of RISK-MAP scores to validate robustness.

```bash
python scripts/monte_carlo_safer.py
```

### `cross_layer_propagation.py`

Models how a compromise in one layer cascades to others. Computes CRR and CCI using four coupling models: Geometric Mean (default), Weighted Average, Maximum, and NoisyOR. Generates network graphs, hierarchical flow diagrams, and chord diagrams.

```bash
python scripts/cross_layer_propagation.py
```

### `heatmap.py` / `combined_heatmap.py`

Generate risk heatmaps. `heatmap.py` produces a per-platform view of the top 10 residual risk vectors. `combined_heatmap.py` places all three robots side by side for direct comparison.

```bash
python scripts/heatmap.py
python scripts/combined_heatmap.py
```

### `extract_layer_scores.py`

Extracts per-layer RISK-MAP scores from the full computation and writes them to `data/layer_scores/`. Used as input by `cross_layer_propagation.py`.

```bash
python scripts/extract_layer_scores.py
```

## Data

### Input Files

| File | Description |
|---|---|
| `attacks_vs_defenses_normalised.csv` | 39 × 35 normalized effectiveness matrix |
| `attack_weights.csv` | Likelihood × Impact weight per attack vector |
| `{Robot}_implementation_status.csv` | Defense maturity level (0–1) per platform |
| `structural_feasibility.csv` | 7 × 7 cross-layer coupling feasibility |
| `empirical_evidence.csv` | 7 × 7 empirical evidence strength matrix |
| `{Robot}_mitigation.csv` | Layer-to-layer mitigation effectiveness |

### Robots

Three platforms are analyzed: `Digit`, `G1_EDU`, `Pepper`.

## Requirements

```
python >= 3.8
pandas
numpy
matplotlib
seaborn        # optional, enhanced palettes
networkx       # optional, cascade network graphs
plotly         # optional, interactive plots
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn networkx plotly
```

## Outputs

All figures are written to `RISK_MAP/figs/`:

- `{Robot}/radar.pdf` — Radar chart of layer-level RISK-MAP scores
- `{Robot}/heatmap.png` — Top-10 residual risk heatmap
- `cascade/{Robot}/coupling_matrix.png` — Layer coupling visualization
- `cascade/{Robot}/network_graph.png` — Attack propagation network
- `cascade/cascade_comparison.png` — CRR/CCI comparison across platforms
- `figs/SAFE_R_combined_radar.pdf` — All-platform overlay radar chart

Sensitivity results are written to the project root as `sensitivity_{Robot}.csv` and `table_sensitivity.csv`.

## License

This project is licensed under the [GNU General Public License v3.0](LICENSE).
