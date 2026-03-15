#!/usr/bin/env python3
"""
Scatter plot: snRNAseq meta-analysis betas vs Xenium density log2FC.

Mirrors the existing snrnaseq_vs_xenium_supertype.png but uses density
(cells/mm²) log2FC instead of crumblr compositional logFC.

Input:
  output/density_analysis/density_results_supertype_cortical.csv
  data/nicole_scz_snrnaseq_betas/scz_coefs.xlsx

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
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR, infer_class, INFER_CLASS_COLORS as CLASS_COLORS

DENSITY_PATH = os.path.join(BASE_DIR, "output", "density_analysis",
                             "density_results_supertype_cortical.csv")
NICOLE_PATH = os.path.join(BASE_DIR, "data", "nicole_scz_snrnaseq_betas",
                            "scz_coefs.xlsx")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "density_analysis")


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

    # Load Nicole snRNAseq betas
    nicole = pd.read_excel(NICOLE_PATH)
    nicole = nicole.rename(columns={
        "CellType": "celltype",
        "estimate": "beta_snrnaseq",
        "pval": "pval_snrnaseq",
        "padj": "padj_snrnaseq",
    })

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

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(9, 8))

    for cls in ["Glut", "GABA", "NN", "Other"]:
        mask = merged["class"] == cls
        if mask.sum() == 0:
            continue
        sub = merged[mask]
        ax.scatter(sub["beta_snrnaseq"], sub["logFC_density"],
                   c=CLASS_COLORS[cls], s=60, alpha=0.7,
                   label=f"{cls} (n={mask.sum()})",
                   edgecolors="white", linewidth=0.5, zorder=5)

    # Horizontal error bars (snRNAseq SE)
    for _, row in merged.iterrows():
        ax.plot([row["beta_snrnaseq"] - row["se"], row["beta_snrnaseq"] + row["se"]],
                [row["logFC_density"], row["logFC_density"]],
                color=CLASS_COLORS[row["class"]], alpha=0.15, linewidth=1, zorder=3)

    # Vertical error bars (density SE — already on natural log scale)
    if "se_density" in merged.columns:
        for _, row in merged.iterrows():
            if pd.notna(row["se_density"]) and pd.notna(row["logFC_density"]):
                ax.plot([row["beta_snrnaseq"], row["beta_snrnaseq"]],
                        [row["logFC_density"] - row["se_density"],
                         row["logFC_density"] + row["se_density"]],
                        color=CLASS_COLORS[row["class"]], alpha=0.15,
                        linewidth=1, zorder=3)

    # Label notable types
    for _, row in merged.iterrows():
        is_sig_either = (row["padj_snrnaseq"] < 0.05) or (row.get("fdr_density", 1) < 0.05)
        is_large = abs(row["beta_snrnaseq"]) > 0.2 or abs(row["logFC_density"]) > 0.8
        is_nom_both = (row["pval_snrnaseq"] < 0.05) and (row["pval_density"] < 0.05)
        if is_sig_either or is_large or is_nom_both:
            ax.annotate(row["celltype"],
                        (row["beta_snrnaseq"], row["logFC_density"]),
                        xytext=(5, 5), textcoords="offset points",
                        fontsize=7, alpha=0.8)

    ax.axhline(0, color="grey", alpha=0.2, linewidth=0.5)
    ax.axvline(0, color="grey", alpha=0.2, linewidth=0.5)

    # Regression line
    z = np.polyfit(m["beta_snrnaseq"], m["logFC_density"], 1)
    lim_x = max(abs(m["beta_snrnaseq"]).max(), 0.3) * 1.3
    x_line = np.linspace(-lim_x, lim_x, 100)
    ax.plot(x_line, np.polyval(z, x_line), "k-", alpha=0.3, linewidth=1.5)

    ax.set_xlabel("snRNAseq meta-analysis beta (SCZ effect)", fontsize=13)
    ax.set_ylabel("Xenium density logFC (SCZ vs Control)\n(cells/mm², cortical)", fontsize=13)
    ax.set_title(
        f"SCZ Supertype Effects: snRNAseq Meta-Analysis vs Xenium Density (logFC)\n"
        f"r = {r_all:.3f} (p = {p_all:.1e}) | "
        f"Neuronal: r = {r_neur:.3f} (p = {p_neur:.1e}) | n = {len(m)}",
        fontsize=13, fontweight="bold"
    )
    ax.legend(fontsize=10, loc="lower right")

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "snrnaseq_vs_density_supertype.png")
    plt.savefig(out_png, dpi=100)
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
