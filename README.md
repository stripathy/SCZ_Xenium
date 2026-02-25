# SCZ Xenium Spatial Transcriptomics Analysis

Analysis pipeline for 24 Xenium spatial transcriptomics samples (12 schizophrenia, 12 control) from human dorsolateral prefrontal cortex (DLPFC).

## Datasets & References

### Xenium SCZ Data

- **Source:** [Kwon et al. (2026)](https://doi.org/10.64898/2026.02.16.706214) — *Mapping spatially organized molecular and genetic signatures of schizophrenia across multiple scales in human prefrontal cortex*
- **GEO Accession:** [GSE307404](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307404)
- **Description:** 24 DLPFC sections (12 SCZ, 12 control) profiled with a 300-gene Xenium panel, yielding ~1.34 million cells total

### SEA-AD Reference Datasets

Cell type annotation and cortical depth modeling use reference data from the Seattle Alzheimer's Disease Brain Cell Atlas:

- **SEA-AD MERFISH** (`SEAAD_MTG_MERFISH.2024-12-11.h5ad`) — Middle temporal gyrus MERFISH spatial reference (1.9M cells, 180 genes) used for training the cortical depth model and proportion comparisons. [Download (3.1 GB)](https://sea-ad-spatial-transcriptomics.s3.us-west-2.amazonaws.com/middle-temporal-gyrus/all_donors-h5ad/SEAAD_MTG_MERFISH.2024-12-11.h5ad)
- **SEA-AD snRNAseq** (`nicole_sea_ad_snrnaseq_reference.h5ad`) — Nicole's curated snRNAseq reference (137K cells, 36K genes, 5 neurotypical donors). Same SEA-AD MTG taxonomy. Used for ground-truth validation of doublet detection and as a full-transcriptome reference. Converted from RDS files.
- **MapMyCells precomputed stats** (`precomputed_stats.20231120.sea_ad.MTG.h5`) — Precomputed taxonomy statistics for hierarchical cell type mapping. [Download (251 MB)](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/mapmycells/SEAAD/20240831/precomputed_stats.20231120.sea_ad.MTG.h5)
- **Cell type mapper:** [AllenInstitute/cell_type_mapper](https://github.com/AllenInstitute/cell_type_mapper) (requires Python 3.10+)

**References:**
- Gabitto et al. (2024) *Integrated multimodal cell atlas of Alzheimer's disease.* Nature Neuroscience. [doi:10.1038/s41593-024-01774-5](https://doi.org/10.1038/s41593-024-01774-5)
- [Allen Brain Cell Atlas portal](https://portal.brain-map.org/)

See **[DATA.md](DATA.md)** for complete download instructions.

## Directory Structure

```
SCZ_Xenium/
├── code/
│   ├── modules/          # Core library modules
│   ├── pipeline/         # Numbered pipeline steps (run sequentially)
│   ├── analysis/         # Downstream statistical analyses & plotting
│   └── archive/          # Legacy/exploratory scripts (reference only)
├── data/
│   ├── raw/              # 72 raw Xenium files (not in repo; see DATA.md)
│   └── reference/        # SEA-AD reference datasets (not in repo; see DATA.md)
├── output/
│   ├── h5ad/             # Per-sample annotated h5ad files (generated)
│   ├── viewer/           # Interactive spatial viewer
│   ├── crumblr/          # Compositional regression results
│   ├── presentation/     # Presentation-ready figures
│   └── ...               # CSVs, model files (generated)
├── sample_metadata.xlsx  # Donor demographics & diagnosis
├── DATA.md               # Data download instructions
└── requirements.txt      # Python dependencies
```

## Pipeline

The pipeline consists of 10 numbered steps in `code/pipeline/`, run sequentially. All paths and settings are centralized in `code/pipeline/pipeline_config.py`.

```
Raw .h5 + boundaries ─→ [00] ─→ initial h5ad
                         [01] ─→ + QC columns
                         [02] ─→ + MapMyCells labels
                        [02b] ─→ + correlation classifier labels + doublet flags
                        [02c] ─→ (alternative: Harmony-based label transfer)
                         [03] ─→ transcript coordinates (for step 04)
                         [04] ─→ + nuclear doublet resolution + hybrid_qc_pass
                         [05] ─→ + depth predictions (uses hybrid_qc_pass)
                         [06] ─→ + spatial domains + layers (uses hybrid_qc_pass)
                         [07] ─→ viewer JSON + HTML
                         [08] ─→ cell + nucleus boundary polygons
```

### Step 00: Create h5ad (`00_create_h5ad.py`)

Creates initial AnnData h5ad files from raw Xenium `.h5` + cell boundary files.

**Output:** `output/h5ad/{sample}_annotated.h5ad` (24 files)

### Step 01: Cell QC (`01_run_qc.py`)

Quality control filtering matching the Kwon et al. methodology:

- Flags cells exceeding the 99th percentile in negative control probe/codeword/unassigned sums
- Flags cells with `n_genes` < 5 MADs below median or `total_counts` outside 5 MADs (per sample)
- Adds `qc_pass` boolean column (does NOT remove cells)

**Output:** Updated h5ad files; `output/qc_summary.csv`

### Step 02: Cell Type Annotation (`02_run_mapmycells.py`)

Hierarchical cell type annotation using [MapMyCells](https://github.com/AllenInstitute/cell_type_mapper) (Allen Institute `cell_type_mapper`):

- Bootstrapped mapping (100 iterations) against the SEA-AD MTG taxonomy
- Assigns labels at 3 levels: `class_label` (3 classes), `subclass_label` (24 subclasses), `supertype_label` (127+ supertypes)
- Provides bootstrapping probability as confidence metric at each level

**Requires:** Python 3.10+ and `cell_type_mapper` (`pip install "cell_type_mapper @ git+https://github.com/AllenInstitute/cell_type_mapper"`)

**Output:** Updated h5ad files with `{class,subclass,supertype}_label` and `{class,subclass,supertype}_label_confidence`

### Step 02b: Correlation Classifier (`02b_run_correlation_classifier.py`)

Two-stage hierarchical reclassification using Pearson correlation against self-built centroids:

- **Stage 1:** Classify into 24 subclasses using top-100 HANN exemplars per subclass
- **Stage 2:** Within each subclass, classify into supertypes using top-100 exemplars per supertype
- **QC 1:** Flag bottom 1% by subclass correlation margin per sample
- **QC 2:** Flag suspected spatial doublets via marker co-expression (validated against Nicole's snRNAseq reference; FP rate 0.098%)
- **Saves** pre-built centroids to `correlation_centroids.pkl` for reuse by step 04

Dramatically reduces misclassifications such as L6b appearing in upper cortical layers.

**Output:** Updated h5ad files with `corr_subclass`, `corr_supertype`, `corr_class`, correlation scores, `corr_qc_pass`, `doublet_suspect`, `doublet_type`; `output/h5ad/correlation_centroids.pkl`

### Step 02c: Harmony Transfer (alternative) (`02c_run_harmony_transfer.py`)

Three-stage hierarchical classification using PCA + Harmony + kNN at each level (Class, Subclass, Supertype). An alternative to step 02b that does not require step 02 — only requires step 01. Independent re-runs Harmony at each stage to correct for modality-specific batch effects.

**Output:** Updated h5ad files with `harmony_subclass`, `harmony_supertype`, `harmony_class`

### Step 03: Export Transcripts (`03_export_transcripts.py`)

Exports per-gene transcript molecule coordinates from raw Xenium `transcripts.zarr` files. These are used by step 04 for building nuclear count matrices and by the browser viewer for on-demand gene visualization.

**Output:** `output/viewer/transcripts/{sample}/gene_index.json` + per-gene JSON files

### Step 04: Nuclear Doublet Resolution (`04_run_nuclear_doublet_resolution.py`)

Hybrid QC approach that uses nuclear-only count matrices (transcripts within nucleus polygons) to resolve spatial doublets identified in step 02b:

- Builds nuclear count matrix from nucleus boundary polygons + transcript coordinates (step 03)
- Re-runs doublet detection on nuclear counts
- Classifies each whole-cell doublet as:
  - **resolved** — WC doublet but NOT nuclear doublet (cytoplasmic spillover, safely rescued)
  - **persistent** — doublet in both WC and nuclear (likely real doublet)
  - **nuclear_only** — NOT WC doublet but IS nuclear doublet (new catch)
  - **insufficient** — nuclear UMI too low for reliable assessment
- Rescues high-UMI cells that were excluded only due to high total counts
- Builds `hybrid_qc_pass` column that replaces blunt UMI filtering with nuclear evidence

**Requires:** Step 02b (correlation classifier) and Step 03 (transcript export)

**Output:** Updated h5ad files with `nuclear_total_counts`, `nuclear_n_genes`, `nuclear_fraction`, `nuclear_doublet_status`, `hybrid_qc_pass`

### Step 05: Depth Prediction (`05_run_depth_prediction.py`)

Trains and applies a cortical depth model from the SEA-AD MERFISH reference:

- GradientBoostingRegressor using K=50 neighborhood composition features
- Predicts normalized cortical depth (0=pia, 1=white matter)
- Fits a 1-NN model for out-of-distribution (OOD) scoring
- Uses `hybrid_qc_pass` (from step 04) to exclude confirmed doublets from neighborhood features

**Output:** `output/depth_model_normalized.pkl`; `predicted_norm_depth` column in each h5ad

### Step 06: Spatial Domains & Layers (`06_run_spatial_domains.py`)

Classifies tissue domains and assigns cortical layers:

- **Spatial clustering:** K=50 neighborhood composition -> PCA -> KNN graph -> Leiden clustering
- **Domain classification:** Vascular (>80% Endothelial+VLMC), Extra-cortical (>60% non-neuronal, low depth), Cortical (remainder)
- **Layer assignment:** Cortical cells -> depth-based layers (L1, L2/3, L4, L5, L6, WM)
- Uses `hybrid_qc_pass` (from step 04) to exclude confirmed doublets

**Output:** `spatial_domain`, `layer` columns; `output/spatial_domain_summary.csv`; merged `output/all_samples_annotated.h5ad`

### Step 07: Export Viewer (`07_export_viewer.py`)

Exports data for the interactive spatial viewer:

- Compact JSON files with coordinates, cell type labels, depth, and layer assignments
- Optional standalone HTML file via `code/modules/bundle_viewer.py`

**Output:** `output/viewer/*.json`, `output/viewer/xenium_viewer_standalone.html`

### Step 08: Export Boundaries (`08_export_boundaries.py`)

Exports cell and nucleus boundary polygons for spatial overlay visualization.

**Output:** `output/viewer/boundaries/{sample}.json` (cell), `output/viewer/boundaries/{sample}_nucleus.json` (nucleus)

## Core Modules (`code/modules/`)

| Module | Description |
|--------|-------------|
| `constants.py` | Shared project-wide constants (SAMPLE_TO_DX, SUBCLASS_TO_CLASS, etc.) |
| `loading.py` | Xenium h5 file I/O, sample discovery |
| `correlation_classifier.py` | Two-stage Pearson correlation classifier + spatial doublet detection |
| `nuclear_counts.py` | Nuclear-only transcript counting via point-in-polygon spatial queries |
| `hybrid_qc.py` | Hybrid QC: marker-based class inference + nuclear doublet QC logic |
| `depth_model.py` | GBR depth model, neighborhood features, OOD scoring |
| `spatial_domains.py` | Pia/vascular spatial domain classification |
| `cell_qc.py` | Cell QC metrics and MAD-based filtering |
| `metadata.py` | Sample metadata and diagnosis mapping |
| `plotting.py` | Rasterized spatial visualization utilities |
| `analysis.py` | Statistical testing (proportions, comparisons) |
| `bundle_viewer.py` | Standalone HTML viewer bundler |

## Analysis Scripts (`code/analysis/`)

Downstream statistical analyses comparing SCZ vs. Control. Configuration is centralized in `code/analysis/config.py` (which imports shared constants from `code/modules/constants.py`).

| Script | Description |
|--------|-------------|
| **Composition** | |
| `build_crumblr_input.py` | Prepare input for crumblr compositional regression (supports `--qc-mode hybrid`) |
| `run_crumblr.R` | Run crumblr CLR + dream linear model (supports `--hybrid` flag) |
| `run_de_edgepython.py` | Pseudobulk differential expression via edgePy (supports `--qc-mode hybrid`) |
| `compare_qc_modes.py` | Side-by-side comparison of corr vs hybrid QC results |
| **Proportion validation** | |
| `plot_xenium_composition_boxplots.py` | Per-subclass composition boxplots (SCZ vs Control) |
| `plot_cropped_proportions.py` | MERFISH vs Xenium proportion scatter (cropped/uncropped) |
| `plot_predicted_proportion_scatter.py` | Neurons-only proportion scatter with error bars |
| `plot_merfish_vs_xenium_proportions.py` | MERFISH vs Xenium naive proportion comparison |
| `check_layer_type_proportions.py` | Upper vs deep excitatory balance check |
| **Depth** | |
| `plot_depth_comparison.py` | 4-panel depth comparison: MERFISH section vs Xenium |
| `plot_median_depth_by_celltype.py` | Median depth per cell type: MERFISH vs Xenium |
| `plot_l6b_depth_filtered.py` | L6b depth distribution: MERFISH vs Xenium |
| **Results** | |
| `plot_crumblr_results.py` | crumblr effect size and volcano plots |
| `plot_snrnaseq_vs_xenium_presentation.py` | Xenium vs snRNAseq meta-analysis beta scatter |
| `plot_aggregated_boxplots.py` | Aggregated composition boxplots |
| **Spatial figures** | |
| `plot_sst_spatial_*.py` | SST subtype spatial comparison figures |
| `plot_l6b_spatial_*.py` | L6b spatial comparison figures |
| `plot_geometry_gallery.py` | Tissue geometry variability gallery |
| **Doublet validation** | |
| `plot_nuclear_doublet_validation.py` | Nuclear doublet resolution diagnostic figures |
| **Classifier** | |
| `plot_corr_classifier_validation.py` | Correlation classifier accuracy evaluation |
| `weighted_classifier.py` | Weighted gene classifier utilities |

## Quick Start

```bash
# 1. Download required datasets (see DATA.md for full instructions)
#    - Raw Xenium data -> data/raw/
#    - SEA-AD MERFISH reference -> data/reference/
#    - SEA-AD snRNAseq reference -> data/reference/
#    - MapMyCells precomputed stats -> data/reference/

# 2. Install Python dependencies
pip install -r requirements.txt
pip install "cell_type_mapper @ git+https://github.com/AllenInstitute/cell_type_mapper"  # Python 3.10+

# 3. Run the pipeline (sequentially)
python3 -u code/pipeline/00_create_h5ad.py
python3 -u code/pipeline/01_run_qc.py
python3 -u code/pipeline/02_run_mapmycells.py
python3 -u code/pipeline/02b_run_correlation_classifier.py
# python3 -u code/pipeline/02c_run_harmony_transfer.py  # (alternative to 02b)
python3 -u code/pipeline/03_export_transcripts.py
python3 -u code/pipeline/04_run_nuclear_doublet_resolution.py
python3 -u code/pipeline/05_run_depth_prediction.py
python3 -u code/pipeline/06_run_spatial_domains.py
python3 -u code/pipeline/07_export_viewer.py
python3 -u code/pipeline/08_export_boundaries.py

# 4. Open the interactive viewer
open output/viewer/xenium_viewer_standalone.html
```

**Note:** Steps 04-06 use multiprocessing (4 workers by default, configurable in `pipeline_config.py`). The full pipeline processes ~1.34M cells across 24 samples. Step 05 (depth model training) is the most time-intensive (~80 min). Total pipeline runtime is approximately 2-3 hours.

## Key Output Files

| File | Description |
|------|-------------|
| `output/h5ad/{sample}_annotated.h5ad` | Per-sample annotated data (24 files) |
| `output/h5ad/correlation_centroids.pkl` | Pre-built correlation centroids (shared by steps 02b, 06) |
| `output/all_samples_annotated.h5ad` | Combined dataset (1.34M cells x 300 genes, ~1.3 GB) |
| `output/depth_model_normalized.pkl` | Trained depth model bundle (111 MB) |
| `output/qc_summary.csv` | Per-sample QC pass rates |
| `output/spatial_domain_summary.csv` | Per-sample domain classification counts |
| `output/viewer/xenium_viewer_standalone.html` | Interactive spatial viewer (17 MB, self-contained) |

## h5ad Schema

Each `{sample}_annotated.h5ad` contains:
- `.X`: sparse count matrix (cells x 300 genes)
- `.obsm['spatial']`: (x, y) spatial coordinates

### `.obs` columns

| Column | Type | Description |
|--------|------|-------------|
| `sample_id` | category | Sample identifier |
| **QC** | | |
| `qc_pass` | bool | Overall QC pass flag |
| `total_counts` | int64 | Total UMI counts per cell |
| `n_genes` | int64 | Number of genes detected |
| `neg_probe_sum` | int64 | Negative control probe sum |
| `neg_codeword_sum` | int64 | Negative codeword sum |
| `unassigned_sum` | int64 | Unassigned transcript sum |
| `fail_neg_probe` | bool | Failed negative probe QC |
| `fail_neg_codeword` | bool | Failed negative codeword QC |
| `fail_unassigned` | bool | Failed unassigned QC |
| `fail_n_genes_low` | bool | Too few genes detected |
| `fail_total_counts_low` | bool | Too few UMIs |
| `fail_total_counts_high` | bool | Too many UMIs |
| **HANN labels (step 02)** | | |
| `class_label` | category | Broad class (3 types) |
| `subclass_label` | category | Subclass (24 types) |
| `supertype_label` | category | Supertype (127+ types) |
| `class_label_confidence` | float32 | HANN bootstrapping confidence |
| `subclass_label_confidence` | float32 | HANN bootstrapping confidence |
| `supertype_label_confidence` | float32 | HANN bootstrapping confidence |
| **Correlation classifier (step 02b)** | | |
| `corr_subclass` | category | Reclassified subclass |
| `corr_supertype` | category | Reclassified supertype |
| `corr_class` | category | Derived from corr_subclass |
| `corr_subclass_corr` | float32 | Best subclass Pearson correlation |
| `corr_subclass_margin` | float32 | Margin between top-2 subclass correlations |
| `corr_supertype_corr` | float32 | Best supertype Pearson correlation |
| `corr_qc_pass` | bool | Passes both margin QC and doublet QC |
| `doublet_suspect` | bool | Suspected spatial doublet |
| `doublet_type` | str | '' / 'Glut+GABA' / 'GABA+GABA' |
| **Depth & layers (steps 03-04)** | | |
| `predicted_norm_depth` | float64 | Predicted normalized cortical depth (0=pia, 1=WM) |
| `spatial_domain` | category | Cortical / Extra-cortical / Vascular |
| `layer` | category | L1 / L2/3 / L4 / L5 / L6 / WM / Vascular |
| `layer_depth_only` | category | Layer from depth bins only (no Vascular override) |
| **Nuclear doublet resolution (step 06)** | | |
| `nuclear_total_counts` | int | Total UMI counts from nuclear-only transcripts |
| `nuclear_n_genes` | int | Number of genes detected in nuclear transcripts |
| `nuclear_fraction` | float32 | Fraction of UMIs in nucleus vs whole cell |
| `nuclear_doublet_suspect` | bool | Nuclear-level doublet detected |
| `nuclear_doublet_type` | str | '' / 'Glut+GABA' / 'GABA+GABA' (nuclear evidence) |
| `nuclear_doublet_status` | str | 'clean' / 'resolved' / 'persistent' / 'nuclear_only' / 'insufficient' |
| `hybrid_qc_pass` | bool | Hybrid QC pass (nuclear-informed, replaces blunt UMI filter) |

**Notes:**
- The correlation classifier columns (`corr_*`, `doublet_*`) are present only in samples processed through step 02b. Analysis scripts prefer `corr_subclass`/`corr_supertype` when available, falling back to HANN labels otherwise.
- The nuclear doublet columns are present only in samples processed through step 06. Analysis scripts support `qc_mode='hybrid'` to use `hybrid_qc_pass` instead of the default `corr_qc_pass`, with automatic fallback if step 06 hasn't been run.

## Cell Type Taxonomy

The pipeline uses the [SEA-AD MTG taxonomy](https://portal.brain-map.org/):

- **3 classes:** Neuronal: Glutamatergic, Neuronal: GABAergic, Non-neuronal and Non-neural
- **24 subclasses:** L2/3 IT, L4 IT, L5 IT, L5 ET, L5/6 NP, L6 IT, L6 IT Car3, L6 CT, L6b, Sst, Sst Chodl, Pvalb, Vip, Lamp5, Lamp5 Lhx6, Sncg, Pax6, Chandelier, Astrocyte, Oligodendrocyte, OPC, Microglia-PVM, Endothelial, VLMC
- **127+ supertypes:** Fine-grained subtypes within each subclass (e.g., Sst_1 through Sst_25)

## Layer Assignment

Layers are assigned through a combined approach:

1. **Spatial domain clustering** identifies contiguous pia/meninges tissue (Extra-cortical) and scattered vascular spots (Vascular)
2. **MERFISH depth model** assigns cortical layers (L1, L2/3, L4, L5, L6, WM) based on neighborhood composition

Final layer categories: **Extra-cortical, L1, L2/3, L4, L5, L6, WM, Vascular**

## Archive (`code/archive/`)

Contains legacy and exploratory scripts from earlier iterations of the analysis:

| Directory | Contents |
|-----------|----------|
| `stale_analysis/` | Superseded analysis scripts (threshold analysis, old comparisons, scCODA) |
| `one_time_utils/` | One-time data migration scripts (rename columns, annotate MERFISH depth) |
| `legacy_runners/` | Old pipeline runners (pre-numbered-step architecture) |
| `ood_methods/` | OOD method exploration scripts |
| `spatial_domain_exploration/` | Early spatial domain clustering experiments |
| *(root)* | Archived modules: `label_transfer.py`, `layers.py`, old pipeline scripts |

These are kept for reference but are not part of the active pipeline.
