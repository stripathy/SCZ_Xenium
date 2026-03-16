#!/usr/bin/env python3
"""
Scatter plot: snRNAseq meta-analysis betas vs Xenium density log2FC at SUBCLASS level.

Aggregates supertype-level data to subclass by summing densities, then computes
logFC and correlates with aggregated snRNAseq betas (inverse-variance weighted).

Input:
  output/density_analysis/density_per_sample_supertype.csv
  data/nicole_scz_snrnaseq_betas/final_results_crumblr_7_cohorts.csv (neuronal)
  data/nicole_scz_snrnaseq_betas/final_results_crumblr_7_nonN_cohorts.csv (non-neuronal)

Output:
  output/density_analysis/snrnaseq_vs_density_subclass.png
  output/density_analysis/snrnaseq_vs_density_subclass.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, ttest_ind

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BASE_DIR, EXCLUDE_SAMPLES, INFER_CLASS_COLORS as CLASS_COLORS,
    infer_class, BG_COLOR,
)

DENSITY_RAW = os.path.join(BASE_DIR, "output", "density_analysis",
                            "density_per_sample_supertype.csv")
NICOLE_NEURONAL_PATH = os.path.join(BASE_DIR, "data", "nicole_scz_snrnaseq_betas",
                                     "final_results_crumblr_7_cohorts.csv")
NICOLE_NONNEURONAL_PATH = os.path.join(BASE_DIR, "data", "nicole_scz_snrnaseq_betas",
                                        "final_results_crumblr_7_nonN_cohorts.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "output", "density_analysis")


def supertype_to_subclass(st):
    """Map supertype name to subclass by stripping trailing _N."""
    # Handle multi-word supertypes like "L2/3 IT_1" -> "L2/3 IT"
    # and "Sst Chodl_1" -> "Sst Chodl"
    parts = st.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return st


def main():
    # ── Load per-sample supertype densities ──
    raw = pd.read_csv(DENSITY_RAW)
    raw = raw[raw["region"] == "cortical"]
    raw = raw[~raw["sample_id"].isin(EXCLUDE_SAMPLES)]
    raw["subclass"] = raw["supertype"].apply(supertype_to_subclass)
    print(f"Raw data: {len(raw)} rows, {raw['sample_id'].nunique()} samples, "
          f"{raw['subclass'].nunique()} subclasses")

    # Aggregate density to subclass level per sample
    sub_density = (raw.groupby(["sample_id", "diagnosis", "subclass"])
                   .agg(density_per_mm2=("density_per_mm2", "sum"),
                        count=("count", "sum"),
                        area_mm2=("area_mm2", "first"))
                   .reset_index())

    # Compute logFC per subclass
    results = []
    for sc in sorted(sub_density["subclass"].unique()):
        sc_data = sub_density[sub_density["subclass"] == sc]
        ctrl = sc_data[sc_data["diagnosis"] == "Control"]["density_per_mm2"]
        scz = sc_data[sc_data["diagnosis"] == "SCZ"]["density_per_mm2"]

        if len(ctrl) < 3 or len(scz) < 3:
            continue

        ctrl_mean = ctrl.mean()
        scz_mean = scz.mean()
        logFC = np.log(scz_mean / ctrl_mean) if ctrl_mean > 0 else np.nan

        # Welch's t-test on log-transformed densities
        _, pval = ttest_ind(np.log1p(ctrl), np.log1p(scz), equal_var=False)

        # SE from log-scale pooled
        log_vals = np.log1p(sc_data["density_per_mm2"])
        se = log_vals.std() / np.sqrt(len(log_vals))

        results.append({
            "subclass": sc,
            "ctrl_mean_density": ctrl_mean,
            "scz_mean_density": scz_mean,
            "logFC_density": logFC,
            "pval_density": pval,
            "se_density": se,
            "n_ctrl": len(ctrl),
            "n_scz": len(scz),
            "class": infer_class(sc),
        })

    density_df = pd.DataFrame(results)
    print(f"Density results: {len(density_df)} subclasses")

    # ── Aggregate snRNAseq betas to subclass (inverse-variance weighted mean) ──
    # Load stratified results (neuronal + non-neuronal)
    dfs = []
    for path in [NICOLE_NEURONAL_PATH, NICOLE_NONNEURONAL_PATH]:
        df = pd.read_csv(path)
        df = df[~df["CellType"].str.contains("SEAAD", na=False)]
        dfs.append(df)
    nicole = pd.concat(dfs, ignore_index=True)
    nicole["subclass"] = nicole["CellType"].apply(supertype_to_subclass)
    nicole["w"] = 1 / (nicole["se"] ** 2)

    snrna_sub = []
    for sc, grp in nicole.groupby("subclass"):
        w = grp["w"].values
        beta = grp["estimate"].values
        se_vals = grp["se"].values

        weighted_beta = np.average(beta, weights=w)
        weighted_se = 1 / np.sqrt(w.sum())

        # Fisher's method for combined p-value (or just use weighted z)
        z = weighted_beta / weighted_se
        from scipy.stats import norm
        combined_p = 2 * norm.sf(abs(z))

        snrna_sub.append({
            "subclass": sc,
            "beta_snrnaseq": weighted_beta,
            "se_snrnaseq": weighted_se,
            "pval_snrnaseq": combined_p,
            "n_supertypes": len(grp),
        })

    snrna_df = pd.DataFrame(snrna_sub)
    print(f"snRNAseq aggregated: {len(snrna_df)} subclasses")

    # ── Merge ──
    merged = pd.merge(snrna_df, density_df, on="subclass", how="inner")
    print(f"Merged: {len(merged)} shared subclasses")

    # Correlations
    valid = merged["logFC_density"].notna() & merged["beta_snrnaseq"].notna()
    m = merged[valid]
    r_all, p_all = pearsonr(m["beta_snrnaseq"], m["logFC_density"])
    neur = m[m["class"].isin(["Glut", "GABA"])]
    r_neur, p_neur = pearsonr(neur["beta_snrnaseq"], neur["logFC_density"]) if len(neur) > 3 else (np.nan, np.nan)
    print(f"All: r={r_all:.3f} (p={p_all:.1e}, n={len(m)})")
    print(f"Neuronal: r={r_neur:.3f} (p={p_neur:.1e}, n={len(neur)})")

    # ── Plot ──
    fig, ax = plt.subplots(figsize=(9, 8))

    for cls in ["Glut", "GABA", "NN", "Other"]:
        mask = merged["class"] == cls
        if mask.sum() == 0:
            continue
        sub = merged[mask]
        ax.scatter(sub["beta_snrnaseq"], sub["logFC_density"],
                   c=CLASS_COLORS[cls], s=100, alpha=0.8,
                   label=f"{cls} (n={mask.sum()})",
                   edgecolors="white", linewidth=0.8, zorder=5)

    # Error bars
    for _, row in merged.iterrows():
        color = CLASS_COLORS[row["class"]]
        # Horizontal (snRNAseq SE)
        ax.plot([row["beta_snrnaseq"] - row["se_snrnaseq"],
                 row["beta_snrnaseq"] + row["se_snrnaseq"]],
                [row["logFC_density"], row["logFC_density"]],
                color=color, alpha=0.25, linewidth=1.5, zorder=3)
        # Vertical (density SE)
        if pd.notna(row["se_density"]) and pd.notna(row["logFC_density"]):
            ax.plot([row["beta_snrnaseq"], row["beta_snrnaseq"]],
                    [row["logFC_density"] - row["se_density"],
                     row["logFC_density"] + row["se_density"]],
                    color=color, alpha=0.25, linewidth=1.5, zorder=3)

    # Label ALL points (it's subclass level so there are few)
    for _, row in merged.iterrows():
        ax.annotate(row["subclass"],
                    (row["beta_snrnaseq"], row["logFC_density"]),
                    xytext=(7, 5), textcoords="offset points",
                    fontsize=10, alpha=0.9, fontweight="bold")

    ax.axhline(0, color="grey", alpha=0.3, linewidth=0.8)
    ax.axvline(0, color="grey", alpha=0.3, linewidth=0.8)

    # Regression line
    z = np.polyfit(m["beta_snrnaseq"], m["logFC_density"], 1)
    lim_x = max(abs(m["beta_snrnaseq"]).max(), 0.1) * 1.5
    x_line = np.linspace(-lim_x, lim_x, 100)
    ax.plot(x_line, np.polyval(z, x_line), "k-", alpha=0.3, linewidth=1.5)

    ax.set_xlabel("snRNAseq meta-analysis beta (SCZ effect, weighted mean)",
                  fontsize=14)
    ax.set_ylabel("Xenium density logFC (SCZ vs Control)\n(cells/mm², cortical)",
                  fontsize=14)
    ax.set_title(
        f"SCZ Subclass Effects: snRNAseq Meta-Analysis vs Xenium Density\n"
        f"r = {r_all:.2f} (p = {p_all:.1e}) | n = {len(m)} subclasses",
        fontsize=15, fontweight="bold"
    )
    ax.legend(fontsize=11, loc="lower right")
    ax.tick_params(labelsize=12)

    plt.tight_layout()
    out_png = os.path.join(OUTPUT_DIR, "snrnaseq_vs_density_subclass.png")
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"\nSaved: {out_png}")

    # Save CSV
    out_csv = os.path.join(OUTPUT_DIR, "snrnaseq_vs_density_subclass.csv")
    merged.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")

    # Print all subclasses
    print(f"\nAll subclasses (sorted by snRNAseq beta):")
    for _, row in merged.sort_values("beta_snrnaseq").iterrows():
        concordant = "✓" if (row["beta_snrnaseq"] * row["logFC_density"]) > 0 else "✗"
        print(f"  {row['subclass']:20s} snRNA={row['beta_snrnaseq']:+.3f}  "
              f"density={row['logFC_density']:+.3f}  {concordant}")


if __name__ == "__main__":
    main()
