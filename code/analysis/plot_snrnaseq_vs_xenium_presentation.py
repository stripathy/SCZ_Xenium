#!/usr/bin/env python3
"""
Presentation-quality scatter: snRNAseq meta-analysis beta vs Xenium spatial logFC.

All supertypes with snRNAseq FDR < 0.1 are explicitly labeled with large text.
Error bars (SE) shown with transparency. Legend placed to avoid obscuring data.

Output: output/presentation/slide_snrnaseq_vs_xenium.png
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    BG_COLOR, CRUMBLR_DIR, PRESENTATION_DIR,
    GABA_PREFIXES, GLUT_PREFIXES, NN_PREFIXES, infer_class,
)

RESULTS_DIR = CRUMBLR_DIR
OUT_DIR = PRESENTATION_DIR

BG = BG_COLOR

CLASS_COLORS = {'Glut': '#d73027', 'GABA': '#4575b4', 'NN': '#33a02c', 'Other': '#999999'}


def main():
    # Load comparison data
    csv_path = os.path.join(RESULTS_DIR, "snrnaseq_vs_xenium_comparison.csv")
    df = pd.read_csv(csv_path)
    df['class'] = df['celltype'].apply(infer_class)
    print(f"Loaded {len(df)} cell types from {csv_path}")

    # Identify snRNAseq FDR < 0.1 cell types (bold labels)
    snrna_sig = set(df[df['padj_snrnaseq'] < 0.1]['celltype'].values)
    print(f"snRNAseq FDR < 0.1: {len(snrna_sig)} types: {sorted(snrna_sig)}")

    # Identify snRNAseq nominal p < 0.05 but FDR >= 0.1 (lighter labels)
    snrna_nom = set(df[(df['pval_snrnaseq'] < 0.05) &
                       (df['padj_snrnaseq'] >= 0.1)]['celltype'].values)
    print(f"snRNAseq nom p < 0.05: {len(snrna_nom)} types: {sorted(snrna_nom)}")

    # Correlations
    r_all, p_all = pearsonr(df['beta_snrnaseq'], df['logFC_xenium'])
    neur = df[df['class'].isin(['Glut', 'GABA'])]
    r_neur, p_neur = pearsonr(neur['beta_snrnaseq'], neur['logFC_xenium'])
    print(f"All: r={r_all:.3f} (p={p_all:.1e}), Neuronal: r={r_neur:.3f} (p={p_neur:.1e})")

    # --- Figure ---
    fig, ax = plt.subplots(figsize=(11, 9), facecolor=BG)
    ax.set_facecolor(BG)

    # Error bars first (behind dots), with transparency
    for _, row in df.iterrows():
        c = CLASS_COLORS[row['class']]
        # Horizontal: snRNAseq SE
        ax.plot([row['beta_snrnaseq'] - row['se'], row['beta_snrnaseq'] + row['se']],
                [row['logFC_xenium'], row['logFC_xenium']],
                color=c, alpha=0.15, linewidth=1.2, zorder=2, solid_capstyle='round')
        # Vertical: Xenium SE
        if pd.notna(row.get('SE_xenium')):
            ax.plot([row['beta_snrnaseq'], row['beta_snrnaseq']],
                    [row['logFC_xenium'] - row['SE_xenium'],
                     row['logFC_xenium'] + row['SE_xenium']],
                    color=c, alpha=0.15, linewidth=1.2, zorder=2, solid_capstyle='round')

    # Scatter dots
    for cls in ['Glut', 'GABA', 'NN', 'Other']:
        mask = df['class'] == cls
        if mask.sum() == 0:
            continue
        sub = df[mask]
        ax.scatter(sub['beta_snrnaseq'], sub['logFC_xenium'],
                   c=CLASS_COLORS[cls], s=70, alpha=0.8,
                   edgecolors='white', linewidth=0.5, zorder=5,
                   label=f'{cls} (n={mask.sum()})')

    # Regression line
    z = np.polyfit(df['beta_snrnaseq'], df['logFC_xenium'], 1)
    lim_x = max(abs(df['beta_snrnaseq']).max(), 0.3) * 1.3
    x_line = np.linspace(-lim_x, lim_x, 100)
    ax.plot(x_line, np.polyval(z, x_line), color='#888888', alpha=0.5,
            linewidth=1.5, linestyle='-', zorder=1)

    # Reference lines
    ax.axhline(0, color='#555555', alpha=0.4, linewidth=0.8, zorder=1)
    ax.axvline(0, color='#555555', alpha=0.4, linewidth=0.8, zorder=1)

    # --- Labels: two tiers ---
    # Tier 1: snRNAseq FDR < 0.1 (bold, white, large)
    # Tier 2: snRNAseq nominal p < 0.05 (normal weight, lighter, slightly smaller)
    texts = []
    for _, row in df.iterrows():
        ct = row['celltype']
        if ct in snrna_sig:
            txt = ax.text(row['beta_snrnaseq'], row['logFC_xenium'],
                          f"  {ct}",
                          fontsize=13, fontweight='bold',
                          color='white', alpha=0.95, zorder=10)
            texts.append(txt)
        elif ct in snrna_nom:
            txt = ax.text(row['beta_snrnaseq'], row['logFC_xenium'],
                          f"  {ct}",
                          fontsize=11, fontweight='normal',
                          color='#bbbbbb', alpha=0.85, zorder=10)
            texts.append(txt)

    # Use adjustText to prevent overlapping labels
    if texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='#aaaaaa',
                                   alpha=0.5, linewidth=0.7),
                    force_text=(0.9, 0.9),
                    force_points=(0.6, 0.6),
                    expand_text=(1.4, 1.4),
                    expand_points=(1.6, 1.6))

    # Correlation annotation (bottom-right, away from most data)
    ax.text(0.97, 0.04,
            f"All: r = {r_all:.2f} (p = {p_all:.1e})\n"
            f"Neuronal: r = {r_neur:.2f} (p = {p_neur:.1e})\n"
            f"n = {len(df)} shared supertypes",
            transform=ax.transAxes, ha='right', va='bottom',
            fontsize=14, color='#dddddd',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#222222',
                      edgecolor='#555555', alpha=0.9))

    # Axis labels and title
    ax.set_xlabel('snRNAseq meta-analysis beta (SCZ effect)', fontsize=18, color='white')
    ax.set_ylabel('Xenium spatial logFC (SCZ vs Control)', fontsize=18, color='white')
    ax.set_title('SCZ compositional effects: snRNAseq vs Xenium spatial',
                 fontsize=22, fontweight='bold', color='white', pad=12)

    ax.tick_params(colors='white', labelsize=14)
    for spine in ax.spines.values():
        spine.set_color('#555555')
    ax.grid(True, alpha=0.15, color='#555555')

    # Legend — place in upper-left (data is sparse there)
    legend_elements = [
        Line2D([0], [0], marker='o', color=BG, markerfacecolor=CLASS_COLORS['Glut'],
               markersize=10, label=f"Glutamatergic (n={(df['class']=='Glut').sum()})", linewidth=0),
        Line2D([0], [0], marker='o', color=BG, markerfacecolor=CLASS_COLORS['GABA'],
               markersize=10, label=f"GABAergic (n={(df['class']=='GABA').sum()})", linewidth=0),
        Line2D([0], [0], marker='o', color=BG, markerfacecolor=CLASS_COLORS['NN'],
               markersize=10, label=f"Non-neuronal (n={(df['class']=='NN').sum()})", linewidth=0),
    ]
    leg = ax.legend(handles=legend_elements, loc='upper left', fontsize=13,
                    frameon=True, fancybox=True, framealpha=0.85,
                    edgecolor='#555555', labelcolor='white')
    leg.get_frame().set_facecolor('#222222')

    # Note about labeled types
    ax.text(0.03, 0.04,
            'Bold = snRNAseq FDR < 0.1  |  Light = snRNAseq nom. p < 0.05',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=12, color='#aaaaaa', fontstyle='italic')

    plt.tight_layout(pad=1.5)

    outpath = os.path.join(OUT_DIR, "slide_snrnaseq_vs_xenium.png")
    plt.savefig(outpath, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
