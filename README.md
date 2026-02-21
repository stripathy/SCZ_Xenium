# SCZ Xenium Spatial Transcriptomics Analysis

Analysis pipeline for 24 Xenium spatial transcriptomics samples (12 schizophrenia, 12 control) from human dorsolateral prefrontal cortex (DLPFC).

## Datasets & References

### Xenium SCZ Data

- **Source:** [Kwon et al. (2026)](https://doi.org/10.64898/2026.02.16.706214) — *Mapping spatially organized molecular and genetic signatures of schizophrenia across multiple scales in human prefrontal cortex*
- **GEO Accession:** [GSE307404](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307404)
- **Description:** 24 DLPFC sections (12 SCZ, 12 control) profiled with a 300-gene Xenium panel, yielding ~1.34 million cells total

### SEA-AD Reference Datasets

Cell type annotation and cortical depth modeling use reference data from the Seattle Alzheimer's Disease Brain Cell Atlas:

- **SEA-AD MERFISH** (`SEAAD_MTG_MERFISH.2024-12-11.h5ad`) — Middle temporal gyrus MERFISH reference used for training the cortical depth model. [Download (3.1 GB)](https://sea-ad-spatial-transcriptomics.s3.us-west-2.amazonaws.com/middle-temporal-gyrus/all_donors-h5ad/SEAAD_MTG_MERFISH.2024-12-11.h5ad)
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
│   └── ...               # CSVs, model files (generated)
├── sample_metadata.xlsx  # Donor demographics & diagnosis
├── DATA.md               # Data download instructions
└── requirements.txt      # Python dependencies
```

## Pipeline

The pipeline consists of 6 numbered steps in `code/pipeline/`, run sequentially. All paths and settings are centralized in `code/pipeline/pipeline_config.py`.

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

### Step 03: Depth Prediction (`03_run_depth_prediction.py`)

Trains and applies a cortical depth model from the SEA-AD MERFISH reference:

- GradientBoostingRegressor using K=50 neighborhood composition features
- Predicts normalized cortical depth (0=pia, 1=white matter)
- Fits a 1-NN model for out-of-distribution (OOD) scoring

**Output:** `output/depth_model_normalized.pkl`; `predicted_norm_depth` column in each h5ad

### Step 04: Spatial Domains & Layers (`04_run_spatial_domains.py`)

Classifies tissue domains and assigns cortical layers:

- **Spatial clustering:** K=50 neighborhood composition -> PCA -> KNN graph -> Leiden clustering
- **Domain classification:** Vascular (>80% Endothelial+VLMC), Extra-cortical (>60% non-neuronal, low depth), Cortical (remainder)
- **Layer assignment:** Cortical cells -> depth-based layers (L1, L2/3, L4, L5, L6, WM)

**Output:** `spatial_domain`, `layer` columns; `output/spatial_domain_summary.csv`; merged `output/all_samples_annotated.h5ad`

### Step 05: Export Viewer (`05_export_viewer.py`)

Exports data for the interactive spatial viewer:

- Compact JSON files with coordinates, cell type labels, depth, and layer assignments
- Optional standalone HTML file via `code/modules/bundle_viewer.py`

**Output:** `output/viewer/*.json`, `output/viewer/xenium_viewer_standalone.html`

## Core Modules (`code/modules/`)

| Module | Description |
|--------|-------------|
| `loading.py` | Xenium h5 file I/O, sample discovery |
| `depth_model.py` | GBR depth model, neighborhood features, OOD scoring |
| `spatial_domains.py` | Pia/vascular spatial domain classification |
| `cell_qc.py` | Cell QC metrics and MAD-based filtering |
| `metadata.py` | Sample metadata and diagnosis mapping |
| `plotting.py` | Rasterized spatial visualization utilities |
| `analysis.py` | Statistical testing (proportions, comparisons) |
| `bundle_viewer.py` | Standalone HTML viewer bundler |

## Analysis Scripts (`code/analysis/`)

Downstream statistical analyses comparing SCZ vs. Control. Configuration is centralized in `code/analysis/config.py`.

| Script | Description |
|--------|-------------|
| `compare_proportions.py` | Cell type proportion comparisons (Mann-Whitney, linear models) |
| `depth_stratified_analysis.py` | Layer-stratified cell type proportion tests |
| `build_crumblr_input.py` | Prepare input for crumblr compositional regression |
| `run_sccoda_stratified.py` | scCODA compositional analysis |
| `plot_*.py` | ~15 scripts generating presentation-ready figures |

## Quick Start

```bash
# 1. Download required datasets (see DATA.md for full instructions)
#    - Raw Xenium data -> data/raw/
#    - SEA-AD MERFISH reference -> data/reference/
#    - MapMyCells precomputed stats -> data/reference/

# 2. Install Python dependencies
pip install -r requirements.txt
pip install "cell_type_mapper @ git+https://github.com/AllenInstitute/cell_type_mapper"  # Python 3.10+

# 3. Run the pipeline (sequentially)
python3 -u code/pipeline/00_create_h5ad.py
python3 -u code/pipeline/01_run_qc.py
python3 -u code/pipeline/02_run_mapmycells.py
python3 -u code/pipeline/03_run_depth_prediction.py
python3 -u code/pipeline/04_run_spatial_domains.py
python3 -u code/pipeline/05_export_viewer.py

# 4. Open the interactive viewer
open output/viewer/xenium_viewer_standalone.html
```

**Note:** Steps 03-04 use multiprocessing (4 workers by default, configurable in `pipeline_config.py`). The full pipeline processes ~1.34M cells across 24 samples. Step 03 (depth model training) is the most time-intensive (~80 min). Total pipeline runtime is approximately 2-3 hours.

## Key Output Files

| File | Description |
|------|-------------|
| `output/h5ad/{sample}_annotated.h5ad` | Per-sample annotated data (24 files) |
| `output/all_samples_annotated.h5ad` | Combined dataset (1.34M cells x 300 genes, ~1.3 GB) |
| `output/depth_model_normalized.pkl` | Trained depth model bundle (111 MB) |
| `output/qc_summary.csv` | Per-sample QC pass rates |
| `output/spatial_domain_summary.csv` | Per-sample domain classification counts |
| `output/viewer/xenium_viewer_standalone.html` | Interactive spatial viewer (17 MB, self-contained) |

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

Contains legacy and exploratory scripts from earlier iterations of the analysis, including:
- Correlation-based label transfer (superseded by MapMyCells)
- OOD method explorations
- Spatial domain exploration scripts

These are kept for reference but are not part of the active pipeline.
