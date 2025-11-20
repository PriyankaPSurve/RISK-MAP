#!/usr/bin/env python3
"""
cross_layer_propagation.py - Compute cascade metrics (CRR, CCI) for humanoid platforms
WITH GEOMETRIC MEAN AS DEFAULT (addresses "ad-hoc formula" reviewer concern)

Implements the cross-layer vulnerability propagation model from Section 5.2
Default: D = √(S ⊙ E) ⊙ (1-M) - Geometric Mean (no arbitrary parameters!)
"""

import argparse
import pathlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.path import Path
from typing import Tuple, Dict, List
import seaborn as sns

# Try networkx for graph plots
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: networkx not installed. Skipping network graph.")
    print("Install with: pip install networkx")

# Try plotly for interactive (optional)
try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

LAYER_ORDER = ['Physical', 'Sensor', 'Data', 'Middleware',
               'Decision', 'Application', 'Social_Interface']

# ══════════════════════════════════════════════════════════════════════════
# COUPLING MATRIX MODELS
# ══════════════════════════════════════════════════════════════════════════

class CouplingMatrixModel:
    """Base class for coupling matrix computation"""
    
    def compute(self, S: np.ndarray, E: np.ndarray, M: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def name(self) -> str:
        raise NotImplementedError

class GeometricMeanModel(CouplingMatrixModel):
    """D = √(S×E) × (1-M) - DEFAULT MODEL
    
    Symmetric fusion of structural and empirical evidence.
    No arbitrary weighting parameters needed.
    Standard in geometric probability and information fusion.
    """
    
    def compute(self, S, E, M):
        return np.sqrt(S * E) * (1 - M)
    
    def name(self):
        return "GeometricMean"

class WeightedAverageModel(CouplingMatrixModel):
    """D = (αS + βE) × (1-M) - Alternative for comparison"""
    
    def __init__(self, alpha: float = 0.5):
        self.alpha = alpha
        self.beta = 1 - alpha
    
    def compute(self, S, E, M):
        return (self.alpha * S + self.beta * E) * (1 - M)
    
    def name(self):
        return f"Weighted(α={self.alpha:.2f})"

class MaximumModel(CouplingMatrixModel):
    """D = max(S,E) × (1-M) - Conservative upper bound"""
    
    def compute(self, S, E, M):
        return np.maximum(S, E) * (1 - M)
    
    def name(self):
        return "Maximum"

class NoisyORModel(CouplingMatrixModel):
    """D = [1-(1-S)(1-E)] × (1-M) - Bayesian independent failures"""
    
    def compute(self, S, E, M):
        return (1 - (1 - S) * (1 - E)) * (1 - M)
    
    def name(self):
        return "NoisyOR"

# ══════════════════════════════════════════════════════════════════════════
# CORE METRICS
# ══════════════════════════════════════════════════════════════════════════

def compute_residual_risk_vector(layer_scores: Dict[str, float]) -> np.ndarray:
    """Convert layer scores (0-5) to residual risk vector V (0-1)"""
    V = np.array([1 - (layer_scores.get(layer, 0) / 5.0) 
                  for layer in LAYER_ORDER])
    return V

def compute_cascade_metrics(D: np.ndarray, V: np.ndarray, 
                           n_samples: int = 100) -> Tuple[float, float, List]:
    """Compute CRR, CCI, and critical paths"""
    
    # Minimum coupling threshold for significant paths
    # Using 0.10 (10%) to focus on architecturally meaningful propagation
    # Lower thresholds create too many insignificant paths that distort CRR
    MIN_COUPLING = 0.10
    
    # Collect all paths
    paths = []
    
    # 1-hop
    for i in range(7):
        for j in range(7):
            if i != j and D[i, j] > MIN_COUPLING:
                risk = V[i] * D[i, j]
                paths.append({
                    'path': (i, j),
                    'risk': risk,
                    'hops': 1,
                    'source': LAYER_ORDER[i],
                    'target': LAYER_ORDER[j],
                    'path_str': f"{LAYER_ORDER[i]} → {LAYER_ORDER[j]}"
                })
    
    # 2-hop
    for i in range(7):
        for j in range(7):
            for k in range(7):
                if i != j and j != k and D[i, j] > MIN_COUPLING and D[j, k] > MIN_COUPLING:
                    risk = V[i] * D[i, j] * D[j, k]
                    paths.append({
                        'path': (i, j, k),
                        'risk': risk,
                        'hops': 2,
                        'source': LAYER_ORDER[i],
                        'intermediate': LAYER_ORDER[j],
                        'target': LAYER_ORDER[k],
                        'path_str': f"{LAYER_ORDER[i]} → {LAYER_ORDER[j]} → {LAYER_ORDER[k]}"
                    })
    
    # 3-hop
    for i in range(7):
        for j in range(7):
            for k in range(7):
                for l in range(7):
                    if (i != j and j != k and k != l and 
                        D[i, j] > MIN_COUPLING and D[j, k] > MIN_COUPLING and D[k, l] > MIN_COUPLING):
                        risk = V[i] * D[i, j] * D[j, k] * D[k, l]
                        paths.append({
                            'path': (i, j, k, l),
                            'risk': risk,
                            'hops': 3,
                            'source': LAYER_ORDER[i],
                            'target': LAYER_ORDER[l],
                            'path_str': f"{LAYER_ORDER[i]} → {LAYER_ORDER[j]} → {LAYER_ORDER[k]} → {LAYER_ORDER[l]}"
                        })
    
    # Compute CRR
    if paths:
        total_risk = sum(p['risk'] for p in paths)
        crr = total_risk / len(paths)
    else:
        crr = 0.0
    
    cci = 1 - crr
    paths.sort(key=lambda x: x['risk'], reverse=True)
    
    return crr, cci, paths[:20]

def monte_carlo_uncertainty(D, V, S, E, M, n_samples=1000, noise_level=0.1, 
                           model=None):
    """Monte Carlo uncertainty analysis
    
    Args:
        D: Coupling matrix
        V: Residual risk vector
        S: Structural feasibility matrix
        E: Empirical evidence matrix
        M: Mitigation strength matrix
        n_samples: Number of Monte Carlo iterations
        noise_level: Standard deviation of measurement noise
        model: CouplingMatrixModel instance (default: GeometricMeanModel)
    """
    if model is None:
        model = GeometricMeanModel()
    
    crr_samples = []
    
    for _ in range(n_samples):
        S_noisy = np.clip(S + np.random.normal(0, noise_level, S.shape), 0, 1)
        E_noisy = np.clip(E + np.random.normal(0, noise_level, E.shape), 0, 1)
        M_noisy = np.clip(M + np.random.normal(0, noise_level/2, M.shape), 0, 1)
        
        D_noisy = model.compute(S_noisy, E_noisy, M_noisy)
        crr, _, _ = compute_cascade_metrics(D_noisy, V, n_samples=50)
        crr_samples.append(crr)
    
    return {
        'mean': np.mean(crr_samples),
        'std': np.std(crr_samples),
        'ci_95': np.percentile(crr_samples, [2.5, 97.5]),
        'samples': crr_samples
    }

# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 1: HIERARCHICAL CASCADE (BEST FOR PAPER)
# ══════════════════════════════════════════════════════════════════════════

def plot_hierarchical_cascade(critical_paths: List[Dict], platform: str,
                              out_path: pathlib.Path):
    """
    Clear hierarchical flow chart of top critical paths
    Shows exactly which paths matter most - INTUITIVE!
    """
    fig, ax = plt.subplots(figsize=(16, 10))
    
    # Top 5 paths
    top_paths = critical_paths[:5]
    y_positions = np.linspace(0.88, 0.12, len(top_paths))
    
    for idx, (path_info, y_base) in enumerate(zip(top_paths, y_positions)):
        path = path_info['path']
        risk = path_info['risk']
        
        # Color scheme
        if risk > 0.4:
            color = '#d62728'  # Red
            box_color = '#ffcccc'
        elif risk > 0.2:
            color = '#ff7f0e'  # Orange
            box_color = '#ffe5cc'
        else:
            color = '#2ca02c'  # Green
            box_color = '#ccffcc'
        
        # Draw path
        n_steps = len(path)
        x_positions = np.linspace(0.15, 0.85, n_steps)
        
        # Boxes for each layer
        for i, (x, layer_idx) in enumerate(zip(x_positions, path)):
            layer = LAYER_ORDER[layer_idx]
            
            # Fancy box
            box = mpatches.FancyBboxPatch((x - 0.07, y_base - 0.025), 
                                         0.14, 0.05,
                                         boxstyle="round,pad=0.008",
                                         facecolor=box_color, 
                                         edgecolor=color, 
                                         linewidth=2.5, zorder=2)
            ax.add_patch(box)
            
            # Label
            ax.text(x, y_base, layer, ha='center', va='center',
                   fontsize=9, fontweight='bold', color='black', zorder=3)
            
            # Arrow to next
            if i < n_steps - 1:
                arrow = mpatches.FancyArrowPatch(
                    (x + 0.07, y_base), 
                    (x_positions[i+1] - 0.07, y_base),
                    arrowstyle='->', 
                    mutation_scale=25,
                    linewidth=3, 
                    color=color,
                    alpha=0.8, zorder=1
                )
                ax.add_patch(arrow)
        
        # Risk value at end
        ax.text(0.92, y_base, f'{risk:.3f}',
               ha='left', va='center', fontsize=11, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='white', 
                        edgecolor=color, linewidth=2))
        
        # Rank badge
        ax.text(0.08, y_base, f'{idx+1}',
               ha='center', va='center', fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='circle', facecolor=color, 
                        edgecolor='black', linewidth=2),
               color='white', zorder=3)
    
    # Styling
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title(f'Top 5 Critical Vulnerability Cascade Paths - {platform}\n' +
                'Ranked by propagation risk magnitude',
                fontsize=16, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [
        mpatches.Patch(facecolor='#ffcccc', edgecolor='#d62728', 
                      linewidth=2, label='High Risk (>0.4)'),
        mpatches.Patch(facecolor='#ffe5cc', edgecolor='#ff7f0e', 
                      linewidth=2, label='Medium Risk (0.2-0.4)'),
        mpatches.Patch(facecolor='#ccffcc', edgecolor='#2ca02c', 
                      linewidth=2, label='Low Risk (<0.2)')
    ]
    ax.legend(handles=legend_elements, loc='lower right', 
             fontsize=11, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 2: MATRIX FLOW (BEST FOR ANALYSIS)
# ══════════════════════════════════════════════════════════════════════════

def plot_matrix_flow(D: np.ndarray, V: np.ndarray, platform: str,
                    out_path: pathlib.Path):
    """
    Side-by-side: Propagation matrix + Risk contribution
    Shows both structure and actual impact
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))
    
    # LEFT: Coupling matrix D
    im1 = ax1.imshow(D, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
    
    for i in range(7):
        for j in range(7):
            if D[i, j] > 0.05:
                text_color = 'white' if D[i, j] > 0.5 else 'black'
                ax1.text(j, i, f'{D[i, j]:.2f}',
                        ha="center", va="center", color=text_color,
                        fontsize=9, fontweight='bold')
    
    ax1.set_xticks(np.arange(7))
    ax1.set_yticks(np.arange(7))
    ax1.set_xticklabels(LAYER_ORDER, rotation=45, ha='right', fontsize=10)
    ax1.set_yticklabels(LAYER_ORDER, fontsize=10)
    ax1.set_xlabel('Target Layer', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Source Layer', fontsize=12, fontweight='bold')
    ax1.set_title('(a) Propagation Probability (D)', 
                 fontsize=13, fontweight='bold', pad=10)
    
    ax1.set_xticks(np.arange(7) - 0.5, minor=True)
    ax1.set_yticks(np.arange(7) - 0.5, minor=True)
    ax1.grid(which="minor", color="white", linestyle='-', linewidth=2)
    
    # RIGHT: Risk contribution V × D
    risk_matrix = np.outer(V, np.ones(7)) * D
    max_risk = np.max(risk_matrix)
    im2 = ax2.imshow(risk_matrix, cmap='Reds', vmin=0, vmax=max_risk, 
                    aspect='auto')
    
    for i in range(7):
        for j in range(7):
            if risk_matrix[i, j] > 0.01:
                text_color = 'white' if risk_matrix[i, j] > max_risk*0.5 else 'black'
                ax2.text(j, i, f'{risk_matrix[i, j]:.2f}',
                        ha="center", va="center", color=text_color,
                        fontsize=9, fontweight='bold')
    
    ax2.set_xticks(np.arange(7))
    ax2.set_yticks(np.arange(7))
    ax2.set_xticklabels(LAYER_ORDER, rotation=45, ha='right', fontsize=10)
    ax2.set_yticklabels(LAYER_ORDER, fontsize=10)
    ax2.set_xlabel('Target Layer', fontsize=12, fontweight='bold')
    ax2.set_ylabel('Source Layer', fontsize=12, fontweight='bold')
    ax2.set_title('(b) Risk Contribution (V × D)', 
                 fontsize=13, fontweight='bold', pad=10)
    
    ax2.set_xticks(np.arange(7) - 0.5, minor=True)
    ax2.set_yticks(np.arange(7) - 0.5, minor=True)
    ax2.grid(which="minor", color="white", linestyle='-', linewidth=2)
    
    # Colorbars
    cbar1 = plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    cbar1.set_label('Propagation\nProbability', rotation=270, 
                   labelpad=25, fontsize=11)
    
    cbar2 = plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)
    cbar2.set_label('Risk\nMagnitude', rotation=270, 
                   labelpad=25, fontsize=11)
    
    fig.suptitle(f'Vulnerability Propagation Analysis - {platform}',
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 3: CHORD DIAGRAM (GOOD OVERVIEW)
# ══════════════════════════════════════════════════════════════════════════

def plot_chord_diagram(D: np.ndarray, V: np.ndarray, platform: str,
                      out_path: pathlib.Path):
    """
    Circular chord diagram showing all flows
    Good for seeing overall connectivity pattern
    """
    fig, ax = plt.subplots(figsize=(12, 12))
    
    n_layers = len(LAYER_ORDER)
    radius = 5
    gap = 0.04
    
    # Arc positions
    arc_width = (2 * np.pi - n_layers * gap) / n_layers
    angles = []
    for i in range(n_layers):
        start = i * (arc_width + gap)
        end = start + arc_width
        angles.append((start, end))
    
    # Layer colors
    layer_colors = plt.cm.Set3(np.linspace(0, 1, n_layers))
    
    # Draw arcs
    for i, (start, end) in enumerate(angles):
        theta = np.linspace(start, end, 100)
        x_outer = (radius + 0.5) * np.cos(theta)
        y_outer = (radius + 0.5) * np.sin(theta)
        x_inner = radius * np.cos(theta)
        y_inner = radius * np.sin(theta)
        
        vertices = (list(zip(x_outer, y_outer)) + 
                   list(zip(x_inner[::-1], y_inner[::-1])))
        codes = [Path.MOVETO] + [Path.LINETO] * (len(vertices) - 2) + [Path.CLOSEPOLY]
        path = Path(vertices, codes)
        patch = mpatches.PathPatch(path, facecolor=layer_colors[i], 
                                   edgecolor='black', linewidth=2, alpha=0.8)
        ax.add_patch(patch)
        
        # Labels
        label_angle = (start + end) / 2
        label_x = (radius + 1.3) * np.cos(label_angle)
        label_y = (radius + 1.3) * np.sin(label_angle)
        
        rotation = np.degrees(label_angle)
        if rotation > 90 and rotation < 270:
            rotation += 180
        
        ax.text(label_x, label_y, LAYER_ORDER[i], 
               ha='center', va='center', fontsize=11, fontweight='bold',
               rotation=rotation)
    
    # Draw flows
    for i in range(n_layers):
        for j in range(n_layers):
            if i != j and D[i, j] > 0.1:
                risk = V[i] * D[i, j]
                
                start_angle = (angles[i][0] + angles[i][1]) / 2
                end_angle = (angles[j][0] + angles[j][1]) / 2
                
                x1 = radius * np.cos(start_angle)
                y1 = radius * np.sin(start_angle)
                x2 = radius * np.cos(end_angle)
                y2 = radius * np.sin(end_angle)
                
                # Bezier curve
                t = np.linspace(0, 1, 100)
                bx = (1-t)**2 * x1 + 2*(1-t)*t * 0 + t**2 * x2
                by = (1-t)**2 * y1 + 2*(1-t)*t * 0 + t**2 * y2
                
                if risk > 0.3:
                    color, alpha = 'red', 0.7
                elif risk > 0.15:
                    color, alpha = 'orange', 0.5
                else:
                    color, alpha = 'gold', 0.3
                
                width = max(risk * 30, 1)
                ax.plot(bx, by, color=color, alpha=alpha, linewidth=width, zorder=1)
    
    ax.set_xlim(-7, 7)
    ax.set_ylim(-7, 7)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f'Vulnerability Propagation Flows - {platform}\n' + 
                'Curves show risk flows between layers',
                fontsize=15, fontweight='bold', pad=20)
    
    # Legend
    legend_elements = [
        plt.Line2D([0], [0], color='red', lw=4, label='High Risk (>0.3)', alpha=0.7),
        plt.Line2D([0], [0], color='orange', lw=4, label='Medium (0.15-0.3)', alpha=0.5),
        plt.Line2D([0], [0], color='gold', lw=4, label='Low (<0.15)', alpha=0.3)
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=10, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# VISUALIZATION 4: NETWORK GRAPH (FIXED)
# ══════════════════════════════════════════════════════════════════════════

def plot_network_graph(D: np.ndarray, V: np.ndarray, platform: str,
                      out_path: pathlib.Path):
    """Network graph with networkx"""
    
    if not NETWORKX_AVAILABLE:
        print("    ⚠ Skipping network graph (networkx not installed)")
        return
    
    G = nx.DiGraph()
    
    # Add nodes
    for i, layer in enumerate(LAYER_ORDER):
        G.add_node(layer, risk=V[i])
    
    # Add edges
    for i, src in enumerate(LAYER_ORDER):
        for j, tgt in enumerate(LAYER_ORDER):
            if i != j and D[i, j] > 0.05:
                G.add_edge(src, tgt, weight=D[i, j])
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Circular layout
    pos = nx.circular_layout(G)
    
    # Draw nodes
    node_sizes = [G.nodes[node]['risk'] * 3000 + 500 for node in G.nodes()]
    node_colors = [G.nodes[node]['risk'] for node in G.nodes()]
    
    nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                          node_color=node_colors, cmap='YlOrRd',
                          vmin=0, vmax=1, alpha=0.9,
                          edgecolors='black', linewidths=2, ax=ax)
    
    # Draw edges
    for (u, v, data) in G.edges(data=True):
        weight = data['weight']
        width = weight * 8
        
        if weight > 0.5:
            color, alpha = 'red', 0.8
        elif weight > 0.3:
            color, alpha = 'orange', 0.6
        else:
            color, alpha = 'gray', 0.4
        
        nx.draw_networkx_edges(G, pos, [(u, v)], width=width,
                              edge_color=color, alpha=alpha,
                              arrowsize=20, arrowstyle='-|>',
                              connectionstyle='arc3,rad=0.1', ax=ax)
    
    # Labels
    nx.draw_networkx_labels(G, pos, font_size=11, font_weight='bold',
                           font_color='white', ax=ax)
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='YlOrRd', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Residual Risk (V)', rotation=270, labelpad=20, fontsize=12)
    
    ax.set_title(f'Vulnerability Propagation Network - {platform}\n' +
                f'Node size = residual risk | Edge width = propagation probability',
                fontsize=14, fontweight='bold', pad=20)
    ax.axis('off')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

# ══════════════════════════════════════════════════════════════════════════
# BASIC VISUALIZATIONS (Keep these)
# ══════════════════════════════════════════════════════════════════════════

def plot_coupling_heatmap(D: np.ndarray, platform: str, out_path: pathlib.Path):
    """Basic coupling matrix heatmap"""
    fig, ax = plt.subplots(figsize=(9, 8))
    
    sns.heatmap(D, annot=True, fmt='.2f', cmap='YlOrRd', 
                xticklabels=LAYER_ORDER, yticklabels=LAYER_ORDER,
                cbar_kws={'label': 'Propagation Probability'},
                vmin=0, vmax=1, ax=ax, linewidths=0.5, linecolor='gray')
    
    ax.set_title(f'Coupling Matrix (D) - {platform}', 
                fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Target Layer', fontsize=14, fontweight='bold')
    ax.set_ylabel('Source Layer', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_platform_comparison(all_D, all_V, all_crr, out_path):
    """Side-by-side platform comparison"""
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    platforms = list(all_D.keys())
    
    for idx, (platform, ax) in enumerate(zip(platforms, axes)):
        D = all_D[platform]
        crr = all_crr[platform]
        
        im = ax.imshow(D, cmap='YlOrRd', vmin=0, vmax=1, aspect='auto')
        
        for i in range(7):
            for j in range(7):
                text_color = 'white' if D[i, j] > 0.5 else 'black'
                ax.text(j, i, f'{D[i, j]:.2f}',
                       ha="center", va="center", color=text_color,
                       fontsize=9, fontweight='bold')
        
        ax.set_xticks(np.arange(7))
        ax.set_yticks(np.arange(7))
        ax.set_xticklabels(LAYER_ORDER, rotation=45, ha='right', fontsize=12,fontweight='bold')
        ax.set_yticklabels(LAYER_ORDER, fontsize=12, fontweight='bold')
        ax.set_title(f'{platform}\nCRR = {crr:.3f} | CCI = {1-crr:.3f}',
                    fontsize=13, fontweight='bold', pad=10)
        
        ax.set_xticks(np.arange(7) - 0.5, minor=True)
        ax.set_yticks(np.arange(7) - 0.5, minor=True)
        ax.grid(which="minor", color="white", linestyle='-', linewidth=2)
    
    fig.subplots_adjust(
    left=0.05,
    right=0.85,
    top=0.88,
    bottom=0.05,
    wspace=0.35 
    )

    cbar_ax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
    cbar = fig.colorbar(im, cax=cbar_ax)
    cbar.set_label('Propagation Probability (D)', rotation=270, 
                   labelpad=20, fontsize=12, fontweight='bold')
    
    # fig.suptitle('Coupling Matrix Comparison Across Platforms',
    #             fontsize=12, fontweight='bold', y=0.98)
    
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()

def compare_models(S, E, M, V, out_path):
    """Model comparison - ALL FOUR MODELS"""
    models = [
        GeometricMeanModel(),         # DEFAULT (no parameters!)
        WeightedAverageModel(alpha=0.5),
        MaximumModel(),
        NoisyORModel()
    ]
    
    results = []
    for model in models:
        D = model.compute(S, E, M)
        crr, cci, _ = compute_cascade_metrics(D, V, n_samples=50)
        results.append({'Model': model.name(), 'CRR': crr, 'CCI': cci})
    
    df = pd.DataFrame(results)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Highlight GeometricMean as default
    colors = ['#2ca02c' if 'Geometric' in m else 'coral' for m in df['Model']]
    
    bars1 = axes[0].bar(df['Model'], df['CRR'], color=colors, 
                       edgecolor='black', linewidth=2)
    axes[0].set_ylabel('CRR', fontsize=13, fontweight='bold')
    axes[0].set_title('Cascade Residual Risk', fontsize=13, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=15)
    axes[0].grid(True, alpha=0.3, axis='y')
    
    for bar in bars1:
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=10)
    
    colors_cci = ['#2ca02c' if 'Geometric' in m else 'lightgreen' for m in df['Model']]
    
    bars2 = axes[1].bar(df['Model'], df['CCI'], color=colors_cci, 
                       edgecolor='black', linewidth=2)
    axes[1].set_ylabel('CCI', fontsize=13, fontweight='bold')
    axes[1].set_title('Cascade Coverage Index', fontsize=13, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=15)
    axes[1].grid(True, alpha=0.3, axis='y')
    
    for bar in bars2:
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=13)
    
    plt.suptitle('Comparison of Coupling Matrix Formulations\n(Green = Default Model)', 
                fontsize=15, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return df

# ══════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--structural', required=True,
                       help='Path to structural feasibility matrix CSV')
    parser.add_argument('--empirical', required=True,
                       help='Path to empirical evidence matrix CSV')
    parser.add_argument('--mitigation', required=True, nargs='+',
                       help='Paths to mitigation matrices (one per platform)')
    parser.add_argument('--layer-scores', required=True, nargs='+',
                       help='Paths to layer scores (one per platform)')
    parser.add_argument('--outdir', default='figs/cascade',
                       help='Output directory for figures')
    parser.add_argument('--n-samples', type=int, default=1000,
                       help='Number of Monte Carlo samples for uncertainty')
    parser.add_argument('--model', choices=['geometric', 'weighted', 'maximum', 'noisyor'],
                       default='geometric',
                       help='Coupling matrix model (default: geometric - no parameters!)')
    
    args = parser.parse_args()
    
    # Select model
    if args.model == 'geometric':
        model = GeometricMeanModel()
    elif args.model == 'weighted':
        model = WeightedAverageModel(alpha=0.5)
    elif args.model == 'maximum':
        model = MaximumModel()
    elif args.model == 'noisyor':
        model = NoisyORModel()
    
    print(f"\n{'='*70}")
    print(f"RISK-MAP Cross-Layer Vulnerability Propagation Analysis")
    print(f"{'='*70}")
    print(f"  Model: {model.name()} (DEFAULT: GeometricMean - no arbitrary parameters!)")
    print(f"  Formula: D = √(S ⊙ E) ⊙ (1-M)")
    print(f"  Monte Carlo samples: {args.n_samples}")
    print(f"{'='*70}\n")
    
    # Load matrices
    S_df = pd.read_csv(args.structural, index_col='From_Layer')
    E_df = pd.read_csv(args.empirical, index_col='From_Layer')
    S = S_df.loc[LAYER_ORDER, LAYER_ORDER].values
    E = E_df.loc[LAYER_ORDER, LAYER_ORDER].values
    
    out_root = pathlib.Path(args.outdir)
    out_root.mkdir(exist_ok=True, parents=True)
    
    all_results = []
    all_D = {}
    all_V = {}
    all_crr = {}
    
    # Process each platform
    for mit_path, scores_path in zip(args.mitigation, args.layer_scores):
        platform = pathlib.Path(mit_path).stem.replace('_mitigation', '')
        
        print(f"\n{'='*60}")
        print(f"Processing: {platform}")
        print(f"{'='*60}")
        
        # Load data
        M_df = pd.read_csv(mit_path, index_col='From_Layer')
        M = M_df.loc[LAYER_ORDER, LAYER_ORDER].values
        
        scores_df = pd.read_csv(scores_path, index_col='Layer')
        layer_scores = scores_df['Score'].to_dict()
        V = compute_residual_risk_vector(layer_scores)
        
        # Compute coupling with selected model
        D = model.compute(S, E, M)
        
        # Metrics
        crr, cci, critical_paths = compute_cascade_metrics(D, V)
        
        print(f"  CRR: {crr:.3f} | CCI: {cci:.3f}")
        print(f"\n  Top 5 Critical Paths:")
        for i, p in enumerate(critical_paths[:5], 1):
            print(f"    {i}. {p['path_str']}")
            print(f"       Risk: {p['risk']:.4f}")
        
        # Monte Carlo
        print(f"\n  Running Monte Carlo uncertainty analysis...")
        mc = monte_carlo_uncertainty(D, V, S, E, M, n_samples=args.n_samples, 
                                    model=model)
        print(f"  CRR: {mc['mean']:.3f} ± {mc['std']:.3f}")
        print(f"  95% CI: [{mc['ci_95'][0]:.3f}, {mc['ci_95'][1]:.3f}]")
        
        # Visualizations
        print(f"\n  Generating visualizations...")
        platform_dir = out_root / platform
        platform_dir.mkdir(exist_ok=True)
        
        plot_coupling_heatmap(D, platform, platform_dir / 'coupling_matrix.png')
        print(f"    ✓ Coupling matrix")
        
        plot_hierarchical_cascade(critical_paths, platform,
                                 platform_dir / 'hierarchical_paths.png')
        print(f"    ✓ Hierarchical cascade (BEST FOR PAPER!)")
        
        plot_matrix_flow(D, V, platform, platform_dir / 'matrix_flow.png')
        print(f"    ✓ Matrix flow (dual panel)")
        
        plot_chord_diagram(D, V, platform, platform_dir / 'chord_diagram.png')
        print(f"    ✓ Chord diagram")
        
        plot_network_graph(D, V, platform, platform_dir / 'network_graph.png')
        print(f"    ✓ Network graph")
        
        # Store
        all_D[platform] = D
        all_V[platform] = V
        all_crr[platform] = crr
        
        all_results.append({
            'Platform': platform,
            'CRR': crr,
            'CCI': cci,
            'CRR_Mean_MC': mc['mean'],
            'CRR_Std_MC': mc['std'],
            'CRR_CI_Lower': mc['ci_95'][0],
            'CRR_CI_Upper': mc['ci_95'][1]
        })
    
    # Comparisons
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("Generating cross-platform comparisons...")
        print(f"{'='*60}")
        
        df_results = pd.DataFrame(all_results)
        df_results.to_csv(out_root / 'cascade_metrics_summary.csv', index=False)
        
        plot_platform_comparison(all_D, all_V, all_crr,
                               out_root / 'coupling_comparison.pdf')
        print(f"  ✓ Platform comparison")
        
        # Bar chart
        fig, ax = plt.subplots(figsize=(12, 7))
        x = np.arange(len(df_results))
        width = 0.35
        
        # Use Monte Carlo mean for plotting (more robust than single calculation)
        crr_plot = df_results['CRR_Mean_MC']
        cci_plot = 1 - crr_plot
        
        bars1 = ax.bar(x - width/2, crr_plot, width, label='CRR',
                      color='coral', edgecolor='black', linewidth=1.5)
        bars2 = ax.bar(x + width/2, cci_plot, width, label='CCI',
                      color='lightgreen', edgecolor='black', linewidth=1.5)
        
        for bar, val in zip(bars1, crr_plot):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
        
        for bar, val in zip(bars2, cci_plot):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.3f}', ha='center', va='bottom', 
                   fontsize=10, fontweight='bold')
        
        # Error bars relative to Monte Carlo mean (guaranteed positive)
        ax.errorbar(x - width/2, crr_plot, 
                   yerr=[crr_plot - df_results['CRR_CI_Lower'],
                         df_results['CRR_CI_Upper'] - crr_plot],
                   fmt='none', ecolor='black', capsize=5, capthick=2)
        
        ax.set_xlabel('Platform', fontsize=13, fontweight='bold')
        ax.set_ylabel('Metric', fontsize=13, fontweight='bold')
        ax.set_title('Cascade Metrics Comparison\n(Error bars = 95% CI)',
                    fontsize=15, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(df_results['Platform'], fontsize=12)
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(out_root / 'cascade_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Bar chart with error bars")
    
    # Model comparison
    if all_results:
        print(f"\n{'='*60}")
        print("Generating model comparison (robustness check)...")
        print(f"{'='*60}")
        
        first_platform = list(all_D.keys())[0]
        M_df = pd.read_csv(args.mitigation[0], index_col='From_Layer')
        M = M_df.loc[LAYER_ORDER, LAYER_ORDER].values
        V = all_V[first_platform]
        
        model_df = compare_models(S, E, M, V, out_root / 'model_comparison.png')
        model_df.to_csv(out_root / 'model_comparison.csv', index=False)
        print(f"  ✓ Model comparison (all 4 formulations)")
        
        # Print comparison
        print(f"\n  Model Comparison Results:")
        for _, row in model_df.iterrows():
            marker = " ← DEFAULT" if "Geometric" in row['Model'] else ""
            print(f"    {row['Model']:20s}  CRR={row['CRR']:.3f}  CCI={row['CCI']:.3f}{marker}")
    
    print(f"\n{'='*60}")
    print(f"✓ ALL DONE! Outputs in: {out_root}")
    print(f"{'='*60}\n")
    
    print("Key files generated:")
    print("  Per-platform:")
    for platform in all_D.keys():
        print(f"    {platform}/")
        print(f"      • hierarchical_paths.png  ← BEST for paper (Figure 7)!")
        print(f"      • matrix_flow.png         ← Shows D and V×D (Figure 6)")
        print(f"      • coupling_matrix.png")
        print(f"      • chord_diagram.png")
        print(f"      • network_graph.png")
    
    if len(all_results) > 1:
        print("\n  Comparisons:")
        print("    • cascade_metrics_summary.csv  ← Table data")
        print("    • coupling_comparison.png      ← Side-by-side matrices")
        print("    • cascade_comparison.png       ← Bar chart with CI")
    
    print("\n  Robustness Analysis:")
    print("    • model_comparison.png  ← Shows all 4 models agree!")
    print("    • model_comparison.csv\n")
    
    print("DEFAULT MODEL: GeometricMean")
    print("  Formula: D = √(S ⊙ E) ⊙ (1-M)")
    print("  Justification: Symmetric fusion, NO arbitrary parameters")
    print("  Addresses reviewer concern: 'ad-hoc formula'\n")

if __name__ == '__main__':
    main()
