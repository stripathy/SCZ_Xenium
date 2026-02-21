#!/usr/bin/env python3
"""
Neurons-only proportion scatter: predicted MERFISH (all cells, L1-L6) vs Xenium.

Uses pre-computed per-donor stats from proportion_stats_predicted_merfish.csv.
Left panel: MERFISH predicted vs Xenium uncropped
Right panel: MERFISH predicted vs Xenium cropped (L1-L6)

Features:
  - Neurons only (Glutamatergic + GABAergic)
  - +/-SD error bars across donors/samples
  - Frequency-weighted Pearson r (log-scale)
  - Dot size scaled by geometric mean frequency
  - Mean CV annotation

Output: output/presentation/slide_neurons_only_proportion_scatter.png
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from scipy.stats import pearsonr, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BG_COLOR, PRESENTATION_DIR, classify_celltype,
)

OUT_DIR = PRESENTATION_DIR
STATS_CSV = os.path.join(OUT_DIR, "proportion_stats_predicted_merfish.csv")

BG = BG_COLOR


def is_neuron(ct):
    _, cls = classify_celltype(ct)
    return cls in ("Glutamatergic", "GABAergic")


def weighted_corr(x, y, w):
    """Weighted Pearson correlation."""
    w = w / w.sum()
    mx = np.sum(w * x)
    my = np.sum(w * y)
    dx = x - mx
    dy = y - my
    cov = np.sum(w * dx * dy)
    sx = np.sqrt(np.sum(w * dx**2))
    sy = np.sqrt(np.sum(w * dy**2))
    if sx == 0 or sy == 0:
        return np.nan
    return cov / (sx * sy)


def plot_panel(ax, df, x_col, y_col, x_sd_col, y_sd_col, title, xlabel, ylabel,
               max_labels=12):
    """Plot neurons-only scatter with error bars, weighted corr, scaled dots."""
    # Filter to rows with valid data in both columns
    valid_mask = df[x_col].notna() & df[y_col].notna()
    valid_mask &= (df[x_col] > 0) & (df[y_col] > 0)
    d = df[valid_mask].copy()

    x = d[x_col].values
    y = d[y_col].values
    x_sd = d[x_sd_col].fillna(0).values
    y_sd = d[y_sd_col].fillna(0).values

    colors = [classify_celltype(ct)[0] for ct in d["celltype"]]

    # Size scaled by geometric mean frequency
    geo_mean = np.sqrt(x * y)
    sizes = 30 + 800 * (geo_mean / geo_mean.max())

    # Error bars
    for i in range(len(x)):
        ax.errorbar(x[i], y[i], xerr=x_sd[i], yerr=y_sd[i],
                    fmt="none", ecolor=colors[i], alpha=0.3, linewidth=1.0,
                    capsize=0, zorder=2)

    ax.scatter(x, y, c=colors, s=sizes, alpha=0.8, edgecolors="white",
               linewidths=0.5, zorder=5)

    # Label most deviant points
    d["log_ratio"] = np.log2((y + 1e-7) / (x + 1e-7))
    d["abs_log_ratio"] = d["log_ratio"].abs()
    top = d.nlargest(max_labels, "abs_log_ratio")

    for _, row in top.iterrows():
        ax.annotate(row["celltype"],
                    (row[x_col], row[y_col]),
                    fontsize=9, color="#dddddd", alpha=0.9,
                    xytext=(5, 5), textcoords="offset points")

    # Diagonal line
    lo = min(x.min(), y.min()) * 0.3
    hi = max(x.max(), y.max()) * 3
    ax.plot([lo, hi], [lo, hi], "--", color="#888888", linewidth=1.5,
            alpha=0.6, zorder=1)

    # Correlations
    log_x = np.log10(x)
    log_y = np.log10(y)

    r_log, _ = pearsonr(log_x, log_y)
    rho, _ = spearmanr(x, y)

    # Frequency-weighted correlation (on log scale)
    weights = geo_mean
    r_weighted = weighted_corr(log_x, log_y, weights)

    # Mean CV
    cv_x = x_sd / (x + 1e-10)
    cv_y = y_sd / (y + 1e-10)
    mean_cv_x = np.mean(cv_x[x_sd > 0])
    mean_cv_y = np.mean(cv_y[y_sd > 0])

    ax.text(0.04, 0.96,
            f"r = {r_log:.2f} (log)\n"
            f"r_w = {r_weighted:.2f} (freq-weighted)\n"
            f"\u03c1 = {rho:.2f}\n"
            f"n = {len(x)} types\n"
            f"CV: MERFISH={mean_cv_x:.2f}, Xenium={mean_cv_y:.2f}",
            transform=ax.transAxes, ha="left", va="top",
            fontsize=13, color="#dddddd",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#333333",
                      edgecolor="#555555", alpha=0.85))

    ax.set_xscale("log")
    ax.set_yscale("log")

    # Human-readable percentage ticks
    def pct_formatter(val, pos):
        pct = val * 100
        if pct >= 1:
            return f"{pct:.0f}%"
        elif pct >= 0.1:
            return f"{pct:.1f}%"
        elif pct >= 0.01:
            return f"{pct:.2f}%"
        else:
            return f"{pct:.3f}%"

    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_formatter(mticker.FuncFormatter(pct_formatter))
        axis.set_minor_formatter(mticker.NullFormatter())

    ax.set_xlabel(xlabel, fontsize=16, color="white")
    ax.set_ylabel(ylabel, fontsize=16, color="white")
    ax.set_title(title, fontsize=20, fontweight="bold", color="white", pad=10)
    ax.set_facecolor(BG)
    ax.tick_params(colors="white", labelsize=13)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(True, alpha=0.2, color="#555555")


def main():
    # Load pre-computed stats
    stats = pd.read_csv(STATS_CSV)
    print(f"Loaded {len(stats)} cell types from {STATS_CSV}")

    # Filter to neurons only
    neuron_mask = stats["celltype"].apply(is_neuron)
    neurons = stats[neuron_mask].copy()
    print(f"Neurons only: {neuron_mask.sum()} types")

    n_glut = sum(1 for ct in neurons["celltype"] if classify_celltype(ct)[1] == "Glutamatergic")
    n_gaba = sum(1 for ct in neurons["celltype"] if classify_celltype(ct)[1] == "GABAergic")
    print(f"  Glutamatergic: {n_glut}, GABAergic: {n_gaba}")

    # --- Plot ---
    fig, axes = plt.subplots(1, 2, figsize=(22, 10), facecolor=BG)

    # Left: MERFISH predicted vs Xenium uncropped
    plot_panel(axes[0], neurons,
               "merfish_mean", "xenium_mean_uncrop",
               "merfish_std", "xenium_std_uncrop",
               "Xenium uncropped (neurons only)",
               "MERFISH predicted (all cells, L1-L6)",
               "Xenium (all QC-pass cells)")

    # Right: MERFISH predicted vs Xenium cropped
    plot_panel(axes[1], neurons,
               "merfish_mean", "xenium_mean_crop",
               "merfish_std", "xenium_std_crop",
               "Xenium cropped to L1-L6 (neurons only)",
               "MERFISH predicted (all cells, L1-L6)",
               "Xenium (L1-L6 only)")

    # Shared legend
    legend_elements = [
        Line2D([0], [0], marker="o", color=BG, markerfacecolor="#3399dd",
               markersize=12, label="Glutamatergic", linewidth=0),
        Line2D([0], [0], marker="o", color=BG, markerfacecolor="#ee4433",
               markersize=12, label="GABAergic", linewidth=0),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2,
               fontsize=14, frameon=False, labelcolor="white",
               bbox_to_anchor=(0.5, -0.01))

    plt.tight_layout(pad=2.0)

    outpath = os.path.join(OUT_DIR, "slide_neurons_only_proportion_scatter.png")
    plt.savefig(outpath, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
