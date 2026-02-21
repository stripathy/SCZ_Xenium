# SCZ Xenium Spatial Transcriptomics Pipeline

Analysis pipeline for 24 Xenium spatial transcriptomics samples from schizophrenia (SCZ) and control brains. Uses the SEA-AD MTG taxonomy via MapMyCells for cell type annotation and a MERFISH-trained depth model for cortical depth estimation.

## Pipeline Steps

The pipeline is modular — each step reads/updates per-sample h5ad files:

| Step | Script | Description |
|------|--------|-------------|
| 00 | `pipeline/00_create_h5ad.py` | Load raw Xenium .h5 + cell boundary CSV → h5ad with spatial coords |
| 01 | `pipeline/01_run_qc.py` | Cell-level QC (Kwon et al. approach), adds `qc_pass` column |
| 02 | `pipeline/02_run_mapmycells.py` | MapMyCells hierarchical annotation → `class_label`, `subclass_label`, `supertype_label` |
| 03 | `pipeline/03_run_depth_prediction.py` | Retrain MERFISH depth model, predict cortical depth → `predicted_norm_depth` |
| 04 | `pipeline/04_run_spatial_domains.py` | Spatial domain clustering + layer assignment → `layer` column |
| 05 | `pipeline/05_export_viewer.py` | Export JSON for interactive HTML viewer |

## Samples

24 Xenium samples: 12 SCZ, 12 Control (1,339,151 total cells, 300-gene panel)
- Outlier: **Br2039** (SCZ) flagged for excess white matter (48%) — exclude from analyses

## Modules (`code/modules/`)

### `depth_model.py`
MERFISH-trained cortical depth prediction from K=50 neighborhood composition features.
- GradientBoostingRegressor, test R² ≈ 0.897 on held-out donors
- Predictions NOT clamped to [0,1] (cells outside cortex can be < 0 or > 1)
- Includes OOD scoring via 1-NN distance to calibrated MERFISH reference

### `analysis.py`
Depth-stratified SCZ vs Control comparison with MERFISH validation.
- Cell type fractions per sample per depth stratum
- Mann-Whitney U tests with FDR correction
- Validation against MERFISH reference proportions

### `loading.py`
Data loading for 10x Xenium .h5 files with cell boundary CSV centroids.

### `metadata.py`
Subject metadata loading (handles non-standard Excel XML format).

### `plotting.py`
Spatial visualization with rasterized color-blended images (dark backgrounds, 20μm bins).

## Key Output Files

```
output/
  h5ad/                           # Per-sample annotated h5ad files
    {sample_id}_annotated.h5ad    # .obs: sample_id, class_label, subclass_label,
                                  #        supertype_label, predicted_norm_depth,
                                  #        qc_pass, layer
  viewer/                         # Interactive HTML viewer
    index.html                    # Multi-sample spatial viewer
    xenium_viewer_standalone.html # Self-contained standalone version
    {sample_id}.json              # Per-sample cell data (compact)
    index.json                    # Global metadata + color palettes
  depth_model_normalized.pkl      # Trained depth prediction model
  qc_summary.csv                  # Per-sample QC statistics
```

## Cell Type Taxonomy

Uses the SEA-AD MTG hierarchy (via MapMyCells):
- **Class** (3): Glutamatergic, GABAergic, Non-neuronal
- **Subclass** (24): L2/3 IT, L4 IT, ..., Sst, Pvalb, ..., Astrocyte, Oligodendrocyte, ...
- **Supertype** (127): Finest level, named `{Subclass}_{N}` (e.g., `Sst_5`, `L2/3 IT_3`, `Astro_1`)

## Depth Strata for Analyses

| Stratum | Depth Range | Description |
|---------|------------|-------------|
| L1 | < 0.10 | Molecular layer |
| L2/3 | 0.10 - 0.30 | Upper cortical layers |
| L4 | 0.30 - 0.45 | Granular layer |
| L5 | 0.45 - 0.65 | Deep output layer |
| L6 | 0.65 - 0.85 | Deep cortical layers |
| WM | > 0.85 | White matter |

## Dependencies

```
anndata, scanpy, numpy, scipy, scikit-learn, scikit-image,
matplotlib, pandas, statsmodels, h5py, cell_type_mapper
```

## Data Requirements

- Xenium data: `GSM*-cell_feature_matrix.h5` + `GSM*-cell_boundaries.csv.gz`
- SEA-AD MERFISH: `SEAAD_MTG_MERFISH.2024-12-11.h5ad` (depth model training)
- MapMyCells precomputed stats: `precomputed_stats.20231120.sea_ad.MTG.h5`
- Subject metadata: `sample_metadata.xlsx`

## Archive

Legacy code is preserved in `code/archive/` for reference:
- `label_transfer.py` — Old kNN-based label transfer (superseded by MapMyCells)
- `layers.py` — Old density-based layer segmentation (superseded by depth model)
- `legacy_runners/` — Old monolithic pipeline runners
- `ood_methods/` — Exploratory OOD detection approaches
- `spatial_domain_exploration/` — Experimental spatial domain scripts
