# SCZ Xenium Spatial Transcriptomics Pipeline

Analysis pipeline for 24 Xenium spatial transcriptomics samples from schizophrenia (SCZ) and control brains. Uses SEA-AD reference datasets for cell type annotation and cortical depth estimation.

## Overview

This pipeline:
1. **Loads** raw 10x Xenium data (300-gene panel, ~55K cells/section)
2. **Annotates** cells via kNN label transfer from SEA-AD snRNAseq reference (class, subclass, cluster levels)
3. **Predicts** normalized cortical depth (0=pia, 1=WM) using a MERFISH-trained neighborhood composition model
4. **Compares** cell type proportions between SCZ and control, stratified by cortical depth
5. **Validates** against MERFISH reference proportions to identify label transfer biases

## Samples

24 Xenium samples: 12 SCZ, 12 Control (1,339,151 total cells)
- Outlier: **Br2039** (SCZ) flagged for excess white matter (48%) - exclude from analyses

## Modules

### `loading.py`
Data loading for 10x Xenium .h5 files with cell boundary CSV centroids.
- `load_xenium_sample()` - load one sample with spatial coords
- `discover_samples()` - find all samples in a directory

### `label_transfer.py`
SEA-AD kNN label transfer at 3 taxonomy levels.
- `annotate_sample()` - full pipeline: shared genes -> normalize -> PCA -> kNN
- Uses k=15 distance-weighted kNN in 50-component shared PCA space

### `depth_model.py`
MERFISH-trained cortical depth prediction from local cell type neighborhoods.
- `train_depth_model()` - train GBR on MERFISH normalized depth from pia
- `predict_depth()` - apply to Xenium (K=50 neighborhood features)
- `build_neighborhood_features()` - construct K-NN subclass composition features
- Model: GradientBoostingRegressor, test R^2 = 0.897 on held-out donors
- Predictions are NOT clamped to [0,1]

### `layers.py`
Density-based cortical layer segmentation (legacy approach).
- Uses 2D density maps of excitatory neuron types
- Superseded by depth_model.py for quantitative analyses

### `analysis.py`
Depth-stratified SCZ vs Control comparison with MERFISH validation.
- `compute_proportions()` - cell type fractions per sample per depth stratum
- `test_case_control()` - Mann-Whitney U with FDR correction
- `flag_outlier_samples()` - identify sections with missing layers
- `validate_against_merfish()` - compare proportions to MERFISH reference

### `metadata.py`
Subject metadata loading (handles non-standard Excel XML format).
- `get_diagnosis_map()` - sample_id -> 'SCZ'/'Control'
- `get_subject_info()` - full subject table (age, sex, diagnosis)

### `plotting.py`
Spatial visualization with rasterized color-blended images.
- `build_raster()` - core rasterization engine (20um bins, dark background)
- `plot_summary()` - 2x3 summary figure (cell types + depth)
- `plot_spatial_celltype()` - single spatial panel with legend

### `run_parallel.py`
Full parallel pipeline (4 workers by default).
```bash
python run_parallel.py \
    --data_dir /path/to/xenium_data \
    --reference /path/to/seaad_10pct.h5ad \
    --depth_model /path/to/depth_model_normalized.pkl \
    --output_dir /path/to/output \
    --n_workers 4 \
    --save_combined
```

### `run_pipeline.py`
Sequential pipeline (single-threaded, useful for debugging).

## Key Output Files

```
output/
  h5ad/                           # Per-sample annotated h5ad files
    {sample_id}_annotated.h5ad    # .obs: sample_id, class_label, subclass_label,
                                  #        cluster_label, predicted_norm_depth
  plots/
    {sample_id}_summary.png       # 2x3 summary figure per sample
  all_samples_annotated.h5ad      # Combined 1.34M cells x 300 genes
  depth_model_normalized.pkl      # Trained depth prediction model
  depth_stratified_tests.csv      # All SCZ vs Control test results
  depth_stratified_proportions.csv # Per-sample proportions
  xenium_vs_merfish_proportions.csv # Label transfer validation
```

## CRITICAL: Known Label Transfer Biases

Validation against MERFISH reference (see `xenium_vs_merfish_proportions.csv`) revealed:

| Cell Type | Bias | Magnitude |
|-----------|------|-----------|
| **Sst** | OVER-represented | ~5-10x across all layers |
| **Oligodendrocyte** | OVER-represented in L6 | +14% |
| **L2/3 IT** | UNDER-represented | -12% in L2/3 |
| **L6 IT** | UNDER-represented | -9% in L6 |
| **L5 IT** | UNDER-represented | -8% in L5 |
| **Microglia-PVM** | UNDER-represented | -3-7% |
| **Endothelial** | OVER-represented | +3-5% |

**The Sst inflation is the most critical issue.** Any findings about Sst proportion changes in SCZ must be interpreted with extreme caution. The absolute Sst proportions (~20% in Xenium vs ~2% in MERFISH) are unreliable. Relative differences between SCZ and control *may* still be meaningful, but this requires further validation.

### Priority for improvement:
1. **Improve label transfer** - the 300-gene Xenium panel may lack discriminative power for Sst vs other GABAergic types. Consider:
   - Using a different classification algorithm (e.g., scANVI, CellTypist)
   - Training a Xenium-specific classifier on MERFISH data (which shares ~180 genes)
   - Using confidence scores to flag uncertain assignments
   - Evaluating gene panel overlap between Xenium and reference
2. **Validate improved labels** against MERFISH using `analysis.validate_against_merfish()`
3. **Re-run depth-stratified comparisons** with corrected labels

## Depth Strata for Analyses

| Stratum | Depth Range | Description |
|---------|------------|-------------|
| L2/3 | 0.10 - 0.30 | Upper cortical layers |
| L4 | 0.30 - 0.45 | Granular layer |
| L5 | 0.45 - 0.65 | Deep output layer |
| L6 | 0.65 - 0.85 | Deep cortical layers |

L1 and WM are excluded from default analyses because Xenium sections
universally undersample L1 (pial surface) and variably sample WM.

## Dependencies

```
anndata, scanpy, numpy, scipy, scikit-learn, scikit-image,
matplotlib, pandas, statsmodels, h5py
```

## Data Requirements

- Xenium data: `GSM*-cell_feature_matrix.h5` + `GSM*-cell_boundaries.csv.gz`
- SEA-AD snRNAseq reference: `seaad_10pct_subsample.h5ad` (10% subsample, 13,730 cells)
- SEA-AD MERFISH: `SEAAD_MTG_MERFISH.2024-12-11.h5ad` (for depth model training)
- Subject metadata: `sample_metadata.xlsx`
