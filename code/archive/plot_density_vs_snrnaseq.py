#!/usr/bin/env python3
"""
Presentation-quality scatter: snRNAseq meta-analysis betas vs Xenium density logFC.

Matches the aesthetic style of slide_snrnaseq_vs_xenium.png (dark background,
adjustText labels, tiered labeling by significance, correlation stats box).

Input:
  output/density_analysis/density_results_supertype_cortical.csv
  data/nicole_scz_snrnaseq_betas/final_results_crumblr_7_cohorts.csv (neuronal)
  data/nicole_scz_snrnaseq_betas/final_results_crumblr_7_nonN_cohorts.csv (non-neuronal)

Output:
  output/density_analysis/snrnaseq_vs_density_supertype.png
  output/density_analysis/snrnaseq_vs_density_comparison.csv

Usage:
    python3 -u plot_density_vs_snrnaseq.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import pearsonr
from adjustText import adjust_text

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from config import BASE_DIR, infer_class, BG_COLOR

DENSITY_PATH = os.path.join(BASE_DIR, "output", "density_analysis",
                             "density_results_supertype_cortical.csv")
NICOLE_NEURONAL_PATH = os.path.join(BASE_DIR, "data", "nicole_scz_snrnaseq_betas",
                                     "final_results_crumblr_7_cohorts.csv")
NICOLE_NONNEURONAL_PATH = os.path.join(BASE_DIR, "data", "nicole_scz_snrnaseq_betas",
                                        "final_results_crumblr_7_nonN_cohorts.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "density_analysis")

BG = BG_COLOR
CLASS_COLORS = {'Glut': '#00ADF8', 'GABA': '#F05A28', 'NN': '#808080', 'Other': '#999999'}


def main():
    # Load density results
    density = pd.read_csv(DENSITY_PATH)
    density = density.rename(columns={
        "supertype": "celltype",
        "logFC": "logFC_density",
        "pval": "pval_density",
        "fdr": "fdr_density",
        "se": "se_density",
    })

    # Load Nicole snRNAseq betas (stratified: neuronal + non-neuronal)
    dfs = []
    for path, stratum in [(NICOLE_NEURONAL_PATH, "neuronal"),
                           (NICOLE_NONNEURONAL_PATH, "non-neuronal")]:
        df = pd.read_csv(path)
        df = df[~df["CellType"].str.contains("SEAAD", na=False)]
        df["analysis_stratum"] = stratum
        dfs.append(df)
    nicole = pd.concat(dfs, ignore_index=True)
    nicole = nicole.rename(columns={
        "CellType": "celltype",
        "estimate": "beta_snrnaseq",
        "pval": "pval_snrnaseq",
        "padj": "padj_snrnaseq",
    })
    print(f"Loaded Nicole stratified data: {len(nicole)} types "
          f"({sum(nicole['analysis_stratum']=='neuronal')} neuronal, "
          f"{sum(nicole['analysis_stratum']=='non-neuronal')} non-neuronal)")

    # Merge
    merged = pd.merge(
        nicole[["celltype", "beta_snrnaseq", "se", "pval_snrnaseq", "padj_snrnaseq"]],
        density[["celltype", "logFC_density", "pval_density", "fdr_density", "se_density"]],
        on="celltype", how="inner"
    )
    merged["class"] = merged["celltype"].apply(infer_class)
    print(f"Merged: {len(merged)} shared supertypes")

    # Correlations
    valid = merged["logFC_density"].notna() & merged["beta_snrnaseq"].notna()
    m = merged[valid]
    r_all, p_all = pearsonr(m["beta_snrnaseq"], m["logFC_density"])
    neur = m[m["class"].isin(["Glut", "GABA"])]
    r_neur, p_neur = pearsonr(neur["beta_snrnaseq"], neur["logFC_density"]) if len(neur) > 3 else (np.nan, np.nan)
    print(f"All: r={r_all:.3f} (p={p_all:.1e}), Neuronal: r={r_neur:.3f} (p={p_neur:.1e})")

    # Identify snRNAseq FDR < 0.1 cell types (bold labels)
    snrna_sig = set(merged[merged["padj_snrnaseq"] < 0.1]["celltype"].values)
    print(f"snRNAseq FDR < 0.1: {len(snrna_sig)} types: {sorted(snrna_sig)}")

    # Identify snRNAseq nominal p < 0.05 but FDR >= 0.1 (lighter labels)
    snrna_nom = set(merged[(merged["pval_snrnaseq"] < 0.05) &
                            (merged["padj_snrnaseq"] >= 0.1)]["celltype"].values)
    # Also label types with large density effects
    large_density = set(merged[abs(merged["logFC_density"]) > 0.8]["celltype"].values)
    snrna_nom = snrna_nom | large_density
    print(f"snRNAseq nom p < 0.05 or large density: {len(snrna_nom)} types")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(11, 9), facecolor=BG)
    ax.set_facecolor(BG)

    # Error bars first (behind dots)
    for _, row in merged.iterrows():
        c = CLASS_COLORS[row["class"]]
        # Horizontal: snRNAseq SE
        ax.plot([row["beta_snrnaseq"] - row["se"], row["beta_snrnaseq"] + row["se"]],
                [row["logFC_density"], row["logFC_density"]],
                color=c, alpha=0.15, linewidth=1.2, zorder=2, solid_capstyle='round')
        # Vertical: density SE
        if pd.notna(row.get("se_density")) and pd.notna(row["logFC_density"]):
            ax.plot([row["beta_snrnaseq"], row["beta_snrnaseq"]],
                    [row["logFC_density"] - row["se_density"],
                     row["logFC_density"] + row["se_density"]],
                    color=c, alpha=0.15, linewidth=1.2, zorder=2, solid_capstyle='round')

    # Scatter dots
    for cls in ["Glut", "GABA", "NN", "Other"]:
        mask = merged["class"] == cls
        if mask.sum() == 0:
            continue
        sub = merged[mask]
        ax.scatter(sub["beta_snrnaseq"], sub["logFC_density"],
                   c=CLASS_COLORS[cls], s=70, alpha=0.8,
                   edgecolors="white", linewidth=0.5, zorder=5,
                   label=f"{cls} (n={mask.sum()})")

    # Regression line
    z = np.polyfit(m["beta_snrnaseq"], m["logFC_density"], 1)
    lim_x = max(abs(m["beta_snrnaseq"]).max(), 0.3) * 1.3
    x_line = np.linspace(-lim_x, lim_x, 100)
    ax.plot(x_line, np.polyval(z, x_line), color="#888888", alpha=0.5,
            linewidth=1.5, linestyle="-", zorder=1)

    # Reference lines
    ax.axhline(0, color="#555555", alpha=0.4, linewidth=0.8, zorder=1)
    ax.axvline(0, color="#555555", alpha=0.4, linewidth=0.8, zorder=1)

    # --- Labels: two tiers ---
    texts = []
    for _, row in merged.iterrows():
        ct = row["celltype"]
        if ct in snrna_sig:
            txt = ax.text(row["beta_snrnaseq"], row["logFC_density"],
                          f"  {ct}",
                          fontsize=13, fontweight="bold",
                          color="white", alpha=0.95, zorder=10)
            texts.append(txt)
        elif ct in snrna_nom:
            txt = ax.text(row["beta_snrnaseq"], row["logFC_density"],
                          f"  {ct}",
                          fontsize=11, fontweight="normal",
                          color="#bbbbbb", alpha=0.85, zorder=10)
            texts.append(txt)

    # Use adjustText to prevent overlapping labels
    if texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle="-", color="#aaaaaa",
                                   alpha=0.5, linewidth=0.7),
                    force_text=(0.9, 0.9),
                    force_points=(0.6, 0.6),
                    expand_text=(1.4, 1.4),
                    expand_points=(1.6, 1.6))

    # Correlation annotation (bottom-right)
    ax.text(0.97, 0.04,
            f"All: r = {r_all:.2f} (p = {p_all:.1e})\n"
            f"Neuronal: r = {r_neur:.2f} (p = {p_neur:.1e})\n"
            f"n = {len(m)} shared supertypes",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=14, color="#dddddd",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#222222",
                      edgecolor="#555555", alpha=0.9))

    # Axis labels and title
    ax.set_xlabel("snRNAseq meta-analysis beta (SCZ effect)", fontsize=18, color="white")
    ax.set_ylabel("Xenium density logFC (SCZ vs Control)\n(cells/mm², cortical)",
                  fontsize=18, color="white")
    ax.set_title("SCZ density effects: snRNAseq vs Xenium spatial",
                 fontsize=22, fontweight="bold", color="white", pad=12)

    ax.tick_params(colors="white", labelsize=14)
    for spine in ax.spines.values():
        spine.set_color("#555555")
    ax.grid(True, alpha=0.15, color="#555555")

    # Legend — upper-left
    legend_elements = [
        Line2D([0], [0], marker="o", color=BG, markerfacecolor=CLASS_COLORS["Glut"],
               markersize=10, label=f"Glutamatergic (n={(merged['class']=='Glut').sum()})", linewidth=0),
        Line2D([0], [0], marker="o", color=BG, markerfacecolor=CLASS_COLORS["GABA"],
               markersize=10, label=f"GABAergic (n={(merged['class']=='GABA').sum()})", linewidth=0),
        Line2D([0], [0], marker="o", color=BG, markerfacecolor=CLASS_COLORS["NN"],
               markersize=10, label=f"Non-neuronal (n={(merged['class']=='NN').sum()})", linewidth=0),
    ]
    leg = ax.legend(handles=legend_elements, loc="upper left", fontsize=13,
                    frameon=True, fancybox=True, framealpha=0.85,
                    edgecolor="#555555", labelcolor="white")
    leg.get_frame().set_facecolor("#222222")

    # Note about labeled types
    ax.text(0.03, 0.04,
            "Bold = snRNAseq FDR < 0.1  |  Light = nom. p < 0.05 or |density logFC| > 0.8",
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=12, color="#aaaaaa", fontstyle="italic")

    plt.tight_layout(pad=1.5)
    out_png = os.path.join(OUTPUT_DIR, "snrnaseq_vs_density_supertype.png")
    plt.savefig(out_png, dpi=200, facecolor=BG, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_png}")

    # Save comparison table
    out_csv = os.path.join(OUTPUT_DIR, "snrnaseq_vs_density_comparison.csv")
    merged.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Print notable concordances
    sig_either = merged[(merged["padj_snrnaseq"] < 0.05) | (merged.get("fdr_density", pd.Series(dtype=float)) < 0.05)]
    if len(sig_either) > 0:
        print(f"\nFDR-significant in either platform:")
        for _, row in sig_either.sort_values("celltype").iterrows():
            concordant = "Y" if (row["beta_snrnaseq"] * row["logFC_density"]) > 0 else "N"
            print(f"  {row['celltype']:30s} snRNA={row['beta_snrnaseq']:+.3f} "
                  f"(padj={row['padj_snrnaseq']:.3f})  "
                  f"density={row['logFC_density']:+.3f} "
                  f"(FDR={row.get('fdr_density', np.nan):.3f})  "
                  f"concordant={concordant}")


if __name__ == "__main__":
    main()
