#!/usr/bin/env python3
"""
Boxplots of cell type proportions (%) for snRNAseq-significant cell types
(FDR < 0.2 from Nicole's analysis) in Xenium data, SCZ vs Control.

Organized in 3 rows:
  Row 1: Sst subtypes
  Row 2: Non-Sst, non-L6 subtypes
  Row 3: Layer 6 subtypes (L6 CT, L6 IT, L6b)

Each panel shows crumblr p-value from the Xenium compositional analysis.
Proportions reported as percent of cortical (L1-L6) cells.

Output: output/presentation/slide_xenium_proportion_boxplots.png
"""

import os
import sys
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
from scipy.stats import ttest_ind

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    SAMPLE_TO_DX, DX_COLORS, BG_COLOR, H5AD_DIR, PRESENTATION_DIR,
    CRUMBLR_DIR, SST_TYPES, CORTICAL_LAYERS, SUBCLASS_CONF_THRESH, format_pval,
)

OUT_DIR = PRESENTATION_DIR
CRUMBLR_PATH = os.path.join(CRUMBLR_DIR, "crumblr_results_supertype.csv")

BG = BG_COLOR

# snRNAseq FDR < 0.2 cell types, organized by group
OTHER_TYPES = ["Pvalb_14", "Pvalb_7", "L5/6 NP_4", "L2/3 IT_7"]
L6_TYPES = ["L6b_1", "L6b_2", "L6b_4", "L6 CT_1", "L6 CT_3"]

ALL_TYPES = SST_TYPES + OTHER_TYPES + L6_TYPES


def compute_cortical_area_mm2(coords):
    """Compute convex hull area of cortical cells. Coords in um, returns mm2."""
    if len(coords) < 3:
        return np.nan
    try:
        hull = ConvexHull(coords)
        return hull.volume / 1e6  # um2 -> mm2
    except Exception:
        return np.nan


def load_all_samples():
    """Load per-sample data: cortical proportions and densities for target cell types."""
    records = []
    h5ad_files = sorted([f for f in os.listdir(H5AD_DIR) if f.endswith("_annotated.h5ad")])

    for fname in h5ad_files:
        sample_id = fname.replace("_annotated.h5ad", "")
        fpath = os.path.join(H5AD_DIR, fname)
        print(f"  {sample_id}...", end="", flush=True)

        adata = ad.read_h5ad(fpath)

        cols = ["supertype_label", "layer", "qc_pass", "subclass_label_confidence"]
        obs = adata.obs[cols].copy()
        coords = adata.obsm["spatial"]
        obs["x"] = coords[:, 0]
        obs["y"] = coords[:, 1]

        # Filter to QC-pass
        obs = obs[obs["qc_pass"] == True]

        # Bottom-1% subclass confidence filter
        obs = obs[obs["subclass_label_confidence"].astype(float) >= SUBCLASS_CONF_THRESH]
        obs["layer"] = obs["layer"].astype(str)
        obs["supertype_label"] = obs["supertype_label"].astype(str)

        # Cortical cells only (L1-L6)
        cortical = obs[obs["layer"].isin(CORTICAL_LAYERS)]
        n_cortical = len(cortical)

        # Cortical area from convex hull
        cortical_coords = cortical[["x", "y"]].values
        area_mm2 = compute_cortical_area_mm2(cortical_coords)

        print(f" {n_cortical:,} cortical, {area_mm2:.2f} mm2")

        dx = SAMPLE_TO_DX.get(sample_id, "Unknown")

        for ct in ALL_TYPES:
            ct_count = (cortical["supertype_label"] == ct).sum()
            prop = ct_count / n_cortical if n_cortical > 0 else np.nan
            density = ct_count / area_mm2 if area_mm2 and area_mm2 > 0 else np.nan

            records.append({
                "sample_id": sample_id,
                "diagnosis": dx,
                "celltype": ct,
                "count": ct_count,
                "n_cortical": n_cortical,
                "cortical_area_mm2": area_mm2,
                "proportion_pct": prop * 100,  # as percent
                "density_per_mm2": density,
            })

    return pd.DataFrame(records)


def make_boxplot(ax, data, ct, crumblr_pval):
    """Draw a single boxplot panel for one cell type, SCZ vs Control."""
    ctrl = data[(data["celltype"] == ct) & (data["diagnosis"] == "Control")]["proportion_pct"].dropna()
    scz = data[(data["celltype"] == ct) & (data["diagnosis"] == "SCZ")]["proportion_pct"].dropna()

    positions = [0, 1]
    bp = ax.boxplot([ctrl.values, scz.values], positions=positions,
                    widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(color="#888888"),
                    capprops=dict(color="#888888"))

    for patch, color in zip(bp['boxes'], [DX_COLORS["Control"], DX_COLORS["SCZ"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.8)

    # Overlay individual points with jitter
    for i, (vals, dx) in enumerate([(ctrl, "Control"), (scz, "SCZ")]):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   c=DX_COLORS[dx], s=40, alpha=0.85, edgecolors="white",
                   linewidths=0.5, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Ctrl", "SCZ"], fontsize=13, color="white")

    # Title = cell type name
    ax.set_title(ct, fontsize=15, fontweight="bold", color="white", pad=6)

    # Crumblr p-value annotation
    pval_str = format_pval(crumblr_pval)
    pval_color = "white" if crumblr_pval < 0.05 else ("#cccccc" if crumblr_pval < 0.1 else "#888888")
    pval_weight = "bold" if crumblr_pval < 0.1 else "normal"
    ax.text(0.5, 0.02, f"Xenium {pval_str}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color=pval_color, fontweight=pval_weight,
            fontstyle="italic")

    ax.set_facecolor(BG)
    ax.tick_params(colors="white", labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(axis="y", alpha=0.15, color="#555555")


def make_density_boxplot(ax, data, ct):
    """Draw a density boxplot panel with t-test p-value."""
    ctrl = data[(data["celltype"] == ct) & (data["diagnosis"] == "Control")]["density_per_mm2"].dropna()
    scz = data[(data["celltype"] == ct) & (data["diagnosis"] == "SCZ")]["density_per_mm2"].dropna()

    positions = [0, 1]
    bp = ax.boxplot([ctrl.values, scz.values], positions=positions,
                    widths=0.5, patch_artist=True, showfliers=False,
                    medianprops=dict(color="white", linewidth=1.5),
                    whiskerprops=dict(color="#888888"),
                    capprops=dict(color="#888888"))

    for patch, color in zip(bp['boxes'], [DX_COLORS["Control"], DX_COLORS["SCZ"]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
        patch.set_edgecolor("white")
        patch.set_linewidth(0.8)

    # Overlay individual points with jitter
    for i, (vals, dx) in enumerate([(ctrl, "Control"), (scz, "SCZ")]):
        jitter = np.random.default_rng(42).uniform(-0.12, 0.12, len(vals))
        ax.scatter(np.full(len(vals), i) + jitter, vals,
                   c=DX_COLORS[dx], s=40, alpha=0.85, edgecolors="white",
                   linewidths=0.5, zorder=5)

    ax.set_xticks(positions)
    ax.set_xticklabels(["Ctrl", "SCZ"], fontsize=13, color="white")

    # Title = cell type name
    ax.set_title(ct, fontsize=15, fontweight="bold", color="white", pad=6)

    # T-test p-value
    if len(ctrl) >= 2 and len(scz) >= 2:
        _, pval = ttest_ind(ctrl.values, scz.values, equal_var=False)
    else:
        pval = np.nan

    pval_str = format_pval(pval)
    pval_color = "white" if pval < 0.05 else ("#cccccc" if pval < 0.1 else "#888888")
    pval_weight = "bold" if pval < 0.1 else "normal"
    ax.text(0.5, 0.02, f"t-test {pval_str}",
            transform=ax.transAxes, ha="center", va="bottom",
            fontsize=10, color=pval_color, fontweight=pval_weight,
            fontstyle="italic")

    ax.set_facecolor(BG)
    ax.tick_params(colors="white", labelsize=11)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(axis="y", alpha=0.15, color="#555555")

    return pval


def main():
    print("Loading Xenium samples...")
    data = load_all_samples()

    # Save raw data
    csv_path = os.path.join(OUT_DIR, "xenium_composition_by_sample.csv")
    data.to_csv(csv_path, index=False)
    print(f"\nSaved data: {csv_path}")

    # Load crumblr results for p-values
    crumblr = pd.read_csv(CRUMBLR_PATH)
    crumblr_pvals = dict(zip(crumblr["celltype"], crumblr["P.Value"]))
    print("\nCrumblr p-values:")
    for ct in ALL_TYPES:
        p = crumblr_pvals.get(ct, np.nan)
        print(f"  {ct:25s} p = {p:.4f}")

    row_groups = [
        ("Sst subtypes", SST_TYPES),
        ("Non-Sst, non-L6", OTHER_TYPES),
        ("Layer 6 subtypes", L6_TYPES),
    ]

    n_cols = max(len(SST_TYPES), len(OTHER_TYPES), len(L6_TYPES))
    n_rows = 3

    # ===== Proportion figure =====
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4.5 * n_rows),
                              facecolor=BG)

    for row_idx, (row_label, types) in enumerate(row_groups):
        for j, ct in enumerate(types):
            pval = crumblr_pvals.get(ct, np.nan)
            make_boxplot(axes[row_idx, j], data, ct, pval)

        # Hide unused columns
        for j in range(len(types), n_cols):
            axes[row_idx, j].set_visible(False)

        # Row label
        axes[row_idx, 0].set_ylabel("% of cortical cells", fontsize=13, color="white")

    # Row group labels on the left side
    row_positions = [0.83, 0.50, 0.17]
    for (label, _), ypos in zip(row_groups, row_positions):
        fig.text(0.01, ypos, label, ha="left", va="center",
                 fontsize=16, fontweight="bold", color="#dddddd",
                 rotation=90, transform=fig.transFigure)

    fig.suptitle("Xenium SCZ vs Control: cell type proportions\n"
                 "(snRNAseq FDR < 0.2 cell types)",
                 fontsize=22, fontweight="bold", color="white", y=1.01)

    plt.tight_layout(pad=1.5, rect=[0.04, 0.0, 1.0, 0.97])

    outpath = os.path.join(OUT_DIR, "slide_xenium_proportion_boxplots.png")
    plt.savefig(outpath, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {outpath}")

    # ===== Density figure =====
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.5 * n_cols, 4.5 * n_rows),
                              facecolor=BG)

    print("\nDensity t-test p-values:")
    for row_idx, (row_label, types) in enumerate(row_groups):
        for j, ct in enumerate(types):
            pval = make_density_boxplot(axes[row_idx, j], data, ct)
            print(f"  {ct:25s} t-test p = {pval:.4f}")

        # Hide unused columns
        for j in range(len(types), n_cols):
            axes[row_idx, j].set_visible(False)

        # Row label
        axes[row_idx, 0].set_ylabel("cells / mm2", fontsize=13, color="white")

    # Row group labels
    for (label, _), ypos in zip(row_groups, row_positions):
        fig.text(0.01, ypos, label, ha="left", va="center",
                 fontsize=16, fontweight="bold", color="#dddddd",
                 rotation=90, transform=fig.transFigure)

    fig.suptitle("Xenium SCZ vs Control: cell type density\n"
                 "(snRNAseq FDR < 0.2 cell types)",
                 fontsize=22, fontweight="bold", color="white", y=1.01)

    plt.tight_layout(pad=1.5, rect=[0.04, 0.0, 1.0, 0.97])

    outpath = os.path.join(OUT_DIR, "slide_xenium_density_boxplots.png")
    plt.savefig(outpath, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"\nSaved: {outpath}")


if __name__ == "__main__":
    main()
