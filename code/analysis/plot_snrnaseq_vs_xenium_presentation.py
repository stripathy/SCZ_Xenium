#!/usr/bin/env python3
"""
Presentation-quality scatter: snRNAseq meta-analysis beta vs Xenium spatial logFC.

Two-panel layout: Neuronal supertypes (left) | Non-neuronal supertypes (right).
All supertypes with snRNAseq FDR < 0.1 are explicitly labeled with large text.
Error bars (SE) shown with transparency.

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

CLASS_COLORS = {'Glut': '#00ADF8', 'GABA': '#F05A28', 'NN': '#808080', 'Other': '#999999'}


def plot_panel(ax, panel_df, snrna_sig, snrna_nom, title, show_ylabel=True,
               xcol='beta_snrnaseq', ycol='logFC_xenium',
               se_x_col='se', se_y_col='SE_xenium',
               ylabel='Xenium spatial logFC (SCZ vs Control)'):
    """Plot a single scatter panel with error bars, labels, and regression line."""
    ax.set_facecolor(BG)

    # Class colors for this panel
    classes_present = sorted(panel_df['class'].unique())

    # Error bars first (behind dots)
    for _, row in panel_df.iterrows():
        c = CLASS_COLORS.get(row['class'], '#999999')
        # Horizontal: snRNAseq SE
        if pd.notna(row.get(se_x_col)):
            ax.plot([row[xcol] - row[se_x_col], row[xcol] + row[se_x_col]],
                    [row[ycol], row[ycol]],
                    color=c, alpha=0.15, linewidth=1.2, zorder=2, solid_capstyle='round')
        # Vertical: Xenium SE
        if pd.notna(row.get(se_y_col)):
            ax.plot([row[xcol], row[xcol]],
                    [row[ycol] - row[se_y_col], row[ycol] + row[se_y_col]],
                    color=c, alpha=0.15, linewidth=1.2, zorder=2, solid_capstyle='round')

    # Scatter dots
    for cls in ['Glut', 'GABA', 'NN', 'Other']:
        mask = panel_df['class'] == cls
        if mask.sum() == 0:
            continue
        sub = panel_df[mask]
        ax.scatter(sub[xcol], sub[ycol],
                   c=CLASS_COLORS[cls], s=70, alpha=0.8,
                   edgecolors='white', linewidth=0.5, zorder=5,
                   label=f'{cls} (n={mask.sum()})')

    # Regression line
    valid = panel_df[xcol].notna() & panel_df[ycol].notna()
    m = panel_df[valid]
    if len(m) > 2:
        z = np.polyfit(m[xcol], m[ycol], 1)
        lim_x = max(abs(m[xcol]).max(), 0.3) * 1.3
        x_line = np.linspace(-lim_x, lim_x, 100)
        ax.plot(x_line, np.polyval(z, x_line), color='#888888', alpha=0.5,
                linewidth=1.5, linestyle='-', zorder=1)

    # Reference lines
    ax.axhline(0, color='#555555', alpha=0.4, linewidth=0.8, zorder=1)
    ax.axvline(0, color='#555555', alpha=0.4, linewidth=0.8, zorder=1)

    # Labels: two tiers
    texts = []
    for _, row in panel_df.iterrows():
        ct = row['celltype']
        if ct in snrna_sig:
            txt = ax.text(row[xcol], row[ycol],
                          f"  {ct}",
                          fontsize=11, fontweight='bold',
                          color='white', alpha=0.95, zorder=10)
            texts.append(txt)
        elif ct in snrna_nom:
            txt = ax.text(row[xcol], row[ycol],
                          f"  {ct}",
                          fontsize=9, fontweight='normal',
                          color='#bbbbbb', alpha=0.85, zorder=10)
            texts.append(txt)

    if texts:
        adjust_text(texts, ax=ax,
                    arrowprops=dict(arrowstyle='-', color='#aaaaaa',
                                   alpha=0.5, linewidth=0.7),
                    force_text=(0.9, 0.9),
                    force_points=(0.6, 0.6),
                    expand_text=(1.3, 1.3),
                    expand_points=(1.5, 1.5))

    # Correlation annotation
    if len(m) > 2:
        r_val, p_val = pearsonr(m[xcol], m[ycol])
        ax.text(0.97, 0.04,
                f"r = {r_val:.2f} (p = {p_val:.1e})\nn = {len(m)} supertypes",
                transform=ax.transAxes, ha='right', va='bottom',
                fontsize=13, color='#dddddd',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='#222222',
                          edgecolor='#555555', alpha=0.9))

    ax.set_xlabel('snRNAseq meta-analysis beta', fontsize=16, color='white')
    if show_ylabel:
        ax.set_ylabel(ylabel, fontsize=16, color='white')
    ax.set_title(title, fontsize=18, fontweight='bold', color='white', pad=10)

    ax.tick_params(colors='white', labelsize=12)
    for spine in ax.spines.values():
        spine.set_color('#555555')
    ax.grid(True, alpha=0.15, color='#555555')

    # Note about labeled types
    ax.text(0.03, 0.04,
            'Bold = FDR < 0.1  |  Light = nom. p < 0.05',
            transform=ax.transAxes, ha='left', va='bottom',
            fontsize=10, color='#aaaaaa', fontstyle='italic')


def main():
    # Load comparison data
    csv_path = os.path.join(RESULTS_DIR, "snrnaseq_vs_xenium_comparison.csv")
    df = pd.read_csv(csv_path)
    df['class'] = df['celltype'].apply(infer_class)
    print(f"Loaded {len(df)} cell types from {csv_path}")

    # Split into neuronal and non-neuronal
    neuronal = df[df['class'].isin(['Glut', 'GABA'])].copy()
    nonneuronal = df[df['class'] == 'NN'].copy()
    print(f"Neuronal: {len(neuronal)}, Non-neuronal: {len(nonneuronal)}")

    # Identify snRNAseq FDR < 0.1 cell types (bold labels)
    snrna_sig = set(df[df['padj_snrnaseq'] < 0.1]['celltype'].values)
    print(f"snRNAseq FDR < 0.1: {len(snrna_sig)} types")

    # Identify snRNAseq nominal p < 0.05 but FDR >= 0.1 (lighter labels)
    snrna_nom = set(df[(df['pval_snrnaseq'] < 0.05) &
                       (df['padj_snrnaseq'] >= 0.1)]['celltype'].values)
    print(f"snRNAseq nom p < 0.05: {len(snrna_nom)} types")

    # Overall correlations for reporting
    r_all, p_all = pearsonr(df['beta_snrnaseq'], df['logFC_xenium'])
    r_neur, p_neur = pearsonr(neuronal['beta_snrnaseq'], neuronal['logFC_xenium'])
    if len(nonneuronal) > 2:
        r_nn, p_nn = pearsonr(nonneuronal['beta_snrnaseq'], nonneuronal['logFC_xenium'])
    else:
        r_nn, p_nn = np.nan, np.nan
    print(f"All: r={r_all:.3f} (p={p_all:.1e})")
    print(f"Neuronal: r={r_neur:.3f} (p={p_neur:.1e})")
    print(f"Non-neuronal: r={r_nn:.3f} (p={p_nn:.1e})")

    # --- Two-panel figure ---
    fig, (ax_n, ax_nn) = plt.subplots(1, 2, figsize=(20, 9), facecolor=BG)

    plot_panel(ax_n, neuronal, snrna_sig, snrna_nom,
              title=f'Neuronal supertypes (n={len(neuronal)})',
              show_ylabel=True)

    # For non-neuronal, label all points (there are few)
    nn_all_labels = set(nonneuronal['celltype'].values)
    plot_panel(ax_nn, nonneuronal, snrna_sig, nn_all_labels,
              title=f'Non-neuronal supertypes (n={len(nonneuronal)})',
              show_ylabel=False)

    # Legend on left panel
    legend_elements = [
        Line2D([0], [0], marker='o', color=BG, markerfacecolor=CLASS_COLORS['Glut'],
               markersize=10, label=f"Glutamatergic (n={(neuronal['class']=='Glut').sum()})", linewidth=0),
        Line2D([0], [0], marker='o', color=BG, markerfacecolor=CLASS_COLORS['GABA'],
               markersize=10, label=f"GABAergic (n={(neuronal['class']=='GABA').sum()})", linewidth=0),
    ]
    leg = ax_n.legend(handles=legend_elements, loc='upper left', fontsize=12,
                      frameon=True, fancybox=True, framealpha=0.85,
                      edgecolor='#555555', labelcolor='white')
    leg.get_frame().set_facecolor('#222222')

    # Suptitle
    fig.suptitle('SCZ compositional effects: snRNAseq meta-analysis vs Xenium spatial',
                 fontsize=22, fontweight='bold', color='white', y=1.02)

    plt.tight_layout(pad=2.0)

    outpath = os.path.join(OUT_DIR, "slide_snrnaseq_vs_xenium.png")
    plt.savefig(outpath, dpi=200, facecolor=BG, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {outpath}")


if __name__ == '__main__':
    main()
