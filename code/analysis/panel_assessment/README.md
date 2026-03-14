# Panel Assessment & Probe Design

## Motivation

Our Xenium v1 Brain panel (313 genes) and the newer 5K Prime panel (~5000 genes) have different coverage of cell-type marker genes. Several SST and L6b supertypes are particularly vulnerable to misclassification because their best discriminating markers are absent from one or both panels. Before designing add-on probe sets, we need to:

1. **Quantify the gap**: Which supertypes lack sufficient markers in each panel?
2. **Calibrate expectations**: How much expression dropout should we expect when moving from snRNAseq to spatial? Not all genes survive the transition equally.
3. **Identify reliable genes**: Which gene properties predict good cross-platform performance?
4. **Design add-ons**: Select probes that maximize supertype discrimination while being robust to spatial dropout.

## Approach

### Phase 1: Cross-Platform Calibration

We use three independent datasets to calibrate detection efficiency:

- **snRNAseq** (SEA-AD reference): Ground truth expression profiles
- **MERSCOPE 4K** (4,000-gene panel): High-gene-count spatial data for calibration
- **SEA-AD MERFISH**: Independent spatial dataset for cross-validation

The key insight is that **the same genes tend to perform poorly across all spatial platforms** — dropout is largely a property of the gene (expression level, biotype, specificity), not the platform. This means we can predict which candidate probes will work before ordering them.

### Phase 2: Marker Discovery

We use a **hierarchical two-round strategy**:

- **Round 1 (subclass-level)**: Find markers that distinguish each subclass from all others (e.g., SST vs. all non-SST). These are the "coarse" markers.
- **Round 2 (within-subclass)**: Find markers that distinguish supertypes *within* a subclass (e.g., Sst_2 vs. other SST supertypes). These are the "fine" markers that panels often lack.

Marker scoring incorporates:
- Wilcoxon effect size (AUC)
- Predicted spatial detection rate (calibrated from MERSCOPE 4K)
- Redundancy (multiple markers per type to survive dropout)

### Phase 3: Validation & Visualization

Publication-quality dotplots and heatmaps showing marker expression across supertypes, with panel coverage annotations.

## Scripts

Run in approximate order:

| Script | Purpose |
|--------|---------|
| `compare_snrnaseq_merscope_expression.py` | Pseudobulk correlations between snRNAseq and MERSCOPE 4K by cell type and gene |
| `cross_platform_gene_corr.py` | Compare gene-level correlations across MERSCOPE 4K, SEA-AD MERFISH, and Xenium |
| `characterize_poor_genes.py` | Identify gene properties (biotype, expression, specificity) that predict poor spatial performance |
| `merscope_panel_assessment.py` | Panel overlap matrices, supertype marker coverage, detection rate calibration |
| `supertype_markers_panel_overlap.py` | Wilcoxon markers at supertype level; check which are missing from each panel |
| ~~`nsforest_supertype_markers.py`~~ | *(archived to `code/archive/stale_analysis/`)* NS-Forest marker sets |
| `hierarchical_probe_selection.py` | Two-round hierarchical probe design with dropout-aware scoring |
| `visualize_marker_quality.py` | Dotplots, matrixplots, violin plots of selected markers |

## Key Outputs (in `output/marker_analysis/`)

- `probe_recommendations_xenium_5k.csv` / `probe_recommendations_xenium_v1.csv` — Ranked add-on probe lists per panel
- `supertype_marker_coverage_all_panels.csv` — Coverage gaps by supertype and panel
- `snrnaseq_vs_merscope4k_detection.csv` — Per-gene detection efficiency calibration
- `gene_properties_vs_correlation.csv` — Gene properties predicting cross-platform performance
- `hierarchical_markers_round*.csv` — Selected markers per round

## Dependencies

- Modules: `code/modules/panel_utils.py`, `code/modules/reference_utils.py`, `code/modules/gene_properties.py`, `code/modules/pseudobulk.py`
- Data: `data/merscope_4k_probe_testing/`, `data/reference/` (SEA-AD h5ad + gene symbol mapping)
- Requires MERSCOPE-annotated h5ad files in `output/merscope_h5ad/`
