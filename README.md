# SCZ Xenium Spatial Transcriptomics Analysis

Analysis pipeline for 24 Xenium spatial transcriptomics samples (12 schizophrenia, 12 control) from human dorsolateral prefrontal cortex (DLPFC).

**Key documents:**
- **[Pipeline Rationale](#pipeline-philosophy--design-rationale)** — Why the pipeline is structured this way and key design decisions
- **[Cell Typing Methods & Benchmarking](cell_typing_methods_writeup.md)** — Detailed methods writeup with figures: classification approaches, doublet resolution, MERFISH benchmarking, ablation studies
- **[Depth & Layer Inference Methods](depth_layer_methods_writeup.md)** — Cortical depth model, spatial domain classification, layer assignment, and validation against MERFISH
- **[Data Download Instructions](data/README.md)** — How to obtain all input datasets

## Datasets & References

### Xenium SCZ Data

- **Source:** [Kwon et al. (2026)](https://doi.org/10.64898/2026.02.16.706214) — *Mapping spatially organized molecular and genetic signatures of schizophrenia across multiple scales in human prefrontal cortex*
- **GEO Accession:** [GSE307404](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307404)
- **Description:** 24 DLPFC sections (12 SCZ, 12 control) profiled with a 300-gene Xenium panel, yielding ~1.34 million cells total

### SEA-AD Reference Datasets

Cell type annotation and cortical depth modeling use reference data from the Seattle Alzheimer's Disease Brain Cell Atlas:

- **SEA-AD MERFISH** (`SEAAD_MTG_MERFISH.2024-12-11.h5ad`) — Middle temporal gyrus MERFISH spatial reference (1.9M cells, 180 genes) used for training the cortical depth model and proportion comparisons. [Download (3.1 GB)](https://sea-ad-spatial-transcriptomics.s3.us-west-2.amazonaws.com/middle-temporal-gyrus/all_donors-h5ad/SEAAD_MTG_MERFISH.2024-12-11.h5ad)
- **SEA-AD snRNAseq** (`SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad`) — Full SEA-AD MTG single-nucleus RNA-seq dataset, subset to the 5 neurotypical reference donors used in building the SEA-AD MTG taxonomy (137K cells, 36K genes). Used for ground-truth validation of doublet detection and as a full-transcriptome reference. [Download (33.8 GB)](https://sea-ad-single-cell-profiling.s3.us-west-2.amazonaws.com/MTG/RNAseq/SEAAD_MTG_RNAseq_final-nuclei.2024-02-13.h5ad)
- **MapMyCells precomputed stats** (`precomputed_stats.20231120.sea_ad.MTG.h5`) — Precomputed taxonomy statistics for hierarchical cell type mapping. [Download (251 MB)](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/mapmycells/SEAAD/20240831/precomputed_stats.20231120.sea_ad.MTG.h5)
- **Cell type mapper:** [AllenInstitute/cell_type_mapper](https://github.com/AllenInstitute/cell_type_mapper) (requires Python 3.10+)

**References:**
- Gabitto et al. (2024) *Integrated multimodal cell atlas of Alzheimer's disease.* Nature Neuroscience. [doi:10.1038/s41593-024-01774-5](https://doi.org/10.1038/s41593-024-01774-5)
- [Allen Brain Cell Atlas portal](https://portal.brain-map.org/)

See **[data/README.md](data/README.md)** for complete download instructions.

## Pipeline Philosophy & Design Rationale

### Goal

The primary goal of this pipeline is **robust cell type proportion comparisons between schizophrenia and control tissue** using compositional regression (crumblr). This requires accurate, consistent cell type assignment across all 24 Xenium samples — errors in cell typing translate directly into false positives or missed signals in downstream disease comparisons. A secondary goal is cortical depth modeling, enabling layer-specific analyses of cell type composition.

### The core challenge

Xenium measures only 300 genes per cell (vs 20,000+ in snRNAseq), which makes cell type classification fundamentally harder — many cell types that are distinguishable with a full transcriptome become ambiguous with a curated marker panel. Standard cross-platform integration methods (e.g., Harmony) introduce systematic artifacts when bridging modalities at this reduced gene count: in our benchmarking, Harmony misclassified non-neuronal types into GABAergic categories (Sst inflated ~5x) and achieved only 69% agreement with our final classifier. We needed a classification approach that works entirely *within* the Xenium feature space rather than trying to align across modalities.

### Key design decisions

- **Self-referencing classification over cross-modality integration.** Rather than integrating Xenium with snRNAseq via Harmony, we use Allen Institute's MapMyCells for initial hierarchical labels, then build a Pearson correlation classifier from the top-100 highest-confidence Xenium exemplars per cell type. Because centroids are built from Xenium data itself, they inherently capture platform-specific expression characteristics — no cross-platform normalization needed. This achieves Pearson r = 0.80 against independent MERFISH proportions (vs r = 0.73 for Harmony). See the [detailed methods writeup](cell_typing_methods_writeup.md) for full benchmarking.

- **Accuracy of cell type calls over transcript purity.** Our goal is correct cell *type* assignment for proportion analysis, not accurate per-cell expression profiles. Xenium cell segmentation inevitably captures some mRNA from neighboring cells (cytoplasmic spillover), but this does not substantially affect correlation-based cell type calls — the classifier is robust to noise in individual genes as long as the overall expression profile matches. We track spillover via `nuclear_fraction` (median ~49% of transcripts fall within the nucleus) but do not attempt to deconvolve it.

- **Flag-based QC rather than hard filtering.** The pipeline never removes cells from the h5ad files. Instead, it adds QC flag columns — `qc_pass` (step 01), `corr_qc_pass` and `doublet_suspect` (step 02b) — so that downstream analyses can choose their own filtering stringency. All data is preserved and QC decisions are transparent and reversible.

- **Simplified QC: spatial QC + margin filter + doublet exclusion.** The default QC gate (`corr_qc_pass`) uses three layers: (1) spatial QC flags cells with aberrant control probes, extreme UMI counts, or too few genes; (2) the 5th-percentile margin filter removes cells whose correlation classifier was least confident (per-sample); (3) doublet suspects flagged via validated marker co-expression rules (0.098% FP rate) are excluded. This achieves better cross-platform concordance than more complex rescue approaches — see `docs/pipeline_qc_audit.md`. An optional nuclear doublet resolution module (in `code/nuclear_resolution/`) can further arbitrate doublet calls using nuclear-only counts, but was found to have negligible impact on downstream compositional analyses.

- **External validation at every step.** Cell type proportions are benchmarked against SEA-AD MERFISH (an independent spatial dataset with 341K cortical cells), depth distributions are validated against manually annotated cortical layers, and doublet detection thresholds are calibrated against snRNAseq false-positive rates (0.098% at our threshold). No result is accepted on internal consistency alone.

### What we did NOT optimize for

- **Per-cell expression accuracy.** Some transcripts from neighboring cells leak into cell boundaries. We track this but do not attempt to correct it, since it does not substantially affect cell type assignments or compositional analyses.
- **Rare cell type discovery.** The 300-gene panel and correlation-based classifier are optimized for the 24 known SEA-AD subclasses. Cells that don't match any known type receive low correlation scores but are still assigned to the best-matching subclass.
- **Single-cell differential expression.** While we include pseudobulk DE as a downstream analysis, the pipeline is primarily designed for compositional proportion comparisons. Expression-level analyses should account for the 300-gene panel limitations and potential spillover effects.

### QC strategy

Quality control uses three layers, each adding flag columns to the h5ad (no cells are ever removed):

1. **Spatial QC** (step 01): Flags cells exceeding the 99th percentile in negative control probes/codewords/unassigned counts, or with extreme UMI/gene counts (5 MAD). Adds `qc_pass` column.
2. **Margin filter** (step 02b): Flags the bottom 5% of cells by subclass correlation margin per sample — these are cells where the classifier was least confident. Also flags spatial doublet suspects via marker co-expression rules validated against snRNAseq (0.098% FP rate). Adds `corr_qc_pass` and `doublet_suspect` columns.
3. **Default gate**: `corr_qc_pass` is the recommended filter for downstream analyses. It requires passing spatial QC, having sufficient classifier confidence, and not being a doublet suspect.

An optional nuclear doublet resolution (step 04, now in `code/nuclear_resolution/`) can further refine doublet calls using nuclear-only counts, producing `hybrid_qc_pass`. This was found to have negligible impact on compositional SCZ results (see `docs/pipeline_qc_audit.md`) and is retained as a standalone investigation.

The Br2039 sample (65% white matter, SCZ) is processed through the full pipeline but excluded from disease comparisons in analysis scripts via `EXCLUDE_SAMPLES` due to its atypical tissue composition.

### Reference datasets

Three external datasets serve distinct roles: **SEA-AD MERFISH** (1.9M cells, 180 genes) provides the training data for the cortical depth model and an independent benchmark for cell type proportions. **SEA-AD snRNAseq** (137K cells, 36K genes) provides ground-truth doublet false-positive rates and a full-transcriptome reference. **MapMyCells precomputed statistics** enable the initial hierarchical cell type annotation that seeds the correlation classifier. The 300-gene Xenium panel has 100% gene overlap with the snRNAseq reference but only 23 genes overlapping with MERFISH — which is why MERFISH is used for spatial reference and depth modeling, not direct cell typing.

## Directory Structure

```
SCZ_Xenium/
├── code/
│   ├── modules/              # Core library modules
│   ├── pipeline/             # Numbered pipeline steps (run sequentially)
│   ├── analysis/             # Downstream statistical analyses & plotting
│   ├── nuclear_resolution/   # Nuclear doublet resolution (optional side investigation)
│   └── archive/              # Legacy/exploratory scripts (reference only)
├── data/
│   ├── raw/              # 72 raw Xenium files (not in repo; see DATA.md)
│   └── reference/        # SEA-AD reference datasets (not in repo; see DATA.md)
├── output/
│   ├── h5ad/             # Per-sample annotated h5ad files (generated)
│   ├── viewer/           # Interactive spatial viewer
│   ├── crumblr/          # Compositional regression results
│   ├── presentation/     # Presentation-ready figures
│   └── ...               # CSVs, model files (generated)
├── environment.yml      # Python/conda dependencies
└── requirements.txt     # Python dependencies (pip)
```

## Pipeline

The pipeline consists of 10 numbered steps in `code/pipeline/`, run sequentially. All paths and settings are centralized in `code/pipeline/pipeline_config.py`.

```
Raw .h5 + boundaries ─→ [00] ─→ initial h5ad
                         [01] ─→ + QC columns (qc_pass)
                         [02] ─→ + MapMyCells labels
                        [02b] ─→ + correlation classifier labels + doublet flags (corr_qc_pass)
                         [03] ─→ transcript coordinates (for viewer + optional step 04)
                         [04] ─→ (optional) nuclear doublet resolution (see code/nuclear_resolution/)
                         [05] ─→ + depth predictions
                         [06] ─→ + spatial domains + layers
                         [07] ─→ viewer JSON + HTML (exports all cells with qc_status flags)
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
- **QC 1:** Flag bottom 5% by subclass correlation margin per sample
- **QC 2:** Flag suspected spatial doublets via marker co-expression (validated against Nicole's snRNAseq reference; FP rate 0.098%)
- **Saves** pre-built centroids to `correlation_centroids.pkl` for reproducibility

Dramatically reduces misclassifications such as L6b appearing in upper cortical layers.

**Output:** Updated h5ad files with `corr_subclass`, `corr_supertype`, `corr_class`, correlation scores, `corr_qc_pass`, `doublet_suspect`, `doublet_type`; `output/h5ad/correlation_centroids.pkl`

### Step 03: Export Transcripts (`03_export_transcripts.py`)

Exports per-gene transcript molecule coordinates from raw Xenium `transcripts.zarr` files. These are used by step 04 for building nuclear count matrices and by the browser viewer for on-demand gene visualization.

**Output:** `output/viewer/transcripts/{sample}/gene_index.json` + per-gene JSON files

### Step 04: Nuclear Doublet Resolution (optional, moved to `code/nuclear_resolution/`)

Optional step that uses nuclear-only count matrices to arbitrate doublet calls. Empirically shown to have negligible impact on downstream compositional analysis (see `docs/pipeline_qc_audit.md`). The `hybrid_qc_pass` column remains in h5ad files for samples where step 04 was run, but `corr_qc_pass` is now the default QC gate.

See `code/nuclear_resolution/README.md` for details on running this step and interpreting results.

### Step 05: Depth Prediction (`05_run_depth_prediction.py`)

Trains and applies a cortical depth model from the SEA-AD MERFISH reference:

- GradientBoostingRegressor using K=50 neighborhood composition features
- Predicts normalized cortical depth (0=pia, 1=white matter)
- Uses `corr_qc_pass` to exclude low-confidence cells and doublet suspects from neighborhood features

**Output:** `output/depth_model_normalized.pkl`; `predicted_norm_depth` column in each h5ad

### Step 06: BANKSY Spatial Domains & Layers (`06_run_spatial_domains.py`)

Classifies tissue domains using BANKSY (Nature Genetics 2024) and assigns cortical layers:

- **BANKSY clustering:** Gene expression + spatial neighbor expression (λ=0.8, res=0.3) produces spatially coherent domains
- **Domain classification:** Vascular (>50% Endothelial+VLMC), WM (>40% Oligodendrocyte + deep), L1 border (>50% non-neuronal + shallow — correctly classified as Cortical, not "Extra-cortical"), Cortical (remainder)
- **Layer assignment:** Cortical cells → depth-based layers (L1, L2/3, L4, L5, L6, WM); Vascular cells overridden
- **Spatial smoothing:** 3-step pipeline refines layer boundaries: (1) within-domain majority vote (k=30, 2 rounds) smooths cortical layers without crossing BANKSY domain borders; (2) vascular border trim reassigns border Vascular cells with >33% cortical neighbors; (3) BANKSY-anchored L1 contiguity promotes shallow `banksy_is_l1` cells to L1 and removes isolated L1 assignments
- Uses `corr_qc_pass` to exclude low-confidence cells and doublet suspects

BANKSY replaces the earlier K-NN Leiden approach, which misclassified L1 border cells as "Extra-cortical" and lacked white matter detection. The BANKSY approach was validated against SEA-AD MERFISH ground truth.

**Output:** `banksy_cluster`, `banksy_domain`, `banksy_is_l1`, `spatial_domain`, `layer`, `layer_unsmoothed`, `layer_depth_only` columns; `output/spatial_domain_summary.csv`; merged `output/all_samples_annotated.h5ad`

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
| `depth_model.py` | GBR depth model, neighborhood features, layer assignment, spatial layer smoothing |
| `banksy_domains.py` | BANKSY-based spatial domain classification (Cortical/Vascular/WM + L1 border) |
| `spatial_domains.py` | Legacy domain classifier; retained for `VASCULAR_TYPES`, `NON_NEURONAL_TYPES` constants |
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
| `build_crumblr_input.py` | Prepare input for crumblr compositional regression (default: corr QC, supports `--qc-mode hybrid`) |
| `run_crumblr.R` | Run crumblr CLR + dream linear model (supports `--hybrid` flag) |
| **Proportion validation** | |
| `plot_xenium_composition_boxplots.py` | Per-subclass composition boxplots (SCZ vs Control) |
| `plot_cropped_proportions.py` | MERFISH vs Xenium proportion scatter (cropped/uncropped) |
| `plot_predicted_proportion_scatter.py` | Neurons-only proportion scatter with error bars |
| `plot_merfish_vs_xenium_proportions.py` | MERFISH vs Xenium naive proportion comparison |
| `plot_merfish_xenium_benchmark.py` | MERFISH vs Xenium benchmark: proportions, median depth, depth violins |
| **Depth** | |
| `plot_depth_comparison.py` | 4-panel depth comparison: MERFISH section vs Xenium |
| `plot_median_depth_by_celltype.py` | Median depth per cell type: MERFISH vs Xenium |
| `plot_supertype_depth_violins.py` | Paired MERFISH vs Xenium depth violins per supertype (supports `--test-only`) |
| `plot_l6b_depth_filtered.py` | L6b depth distribution: MERFISH vs Xenium |
| **Results** | |
| `plot_crumblr_results.py` | crumblr effect size and volcano plots |
| `plot_snrnaseq_vs_xenium_presentation.py` | Xenium vs snRNAseq meta-analysis beta scatter |
| `plot_aggregated_boxplots.py` | Aggregated composition boxplots |
| **Spatial figures** | |
| `plot_sst_spatial_*.py` | SST subtype spatial comparison figures |
| `plot_l6b_spatial_*.py` | L6b spatial comparison figures |
| `plot_geometry_gallery.py` | Tissue geometry variability gallery |
| **Classifier** | |
| `plot_corr_classifier_validation.py` | Correlation classifier accuracy evaluation |
| `weighted_classifier.py` | Weighted gene classifier utilities |

## Quick Start

```bash
# 1. Download required datasets (see data/README.md for full instructions)
#    - Raw Xenium data -> data/raw/
#    - SEA-AD MERFISH reference -> data/reference/
#    - SEA-AD snRNAseq reference -> data/reference/
#    - MapMyCells precomputed stats -> data/reference/

# 2. Install Python dependencies
conda env create -f environment.yml   # or: pip install -r requirements.txt
pip install pybanksy  # Required for step 06 (BANKSY spatial domain classification)
pip install "cell_type_mapper @ git+https://github.com/AllenInstitute/cell_type_mapper"  # Python 3.10+

# 3. Run the pipeline (sequentially)
python3 -u code/pipeline/00_create_h5ad.py
python3 -u code/pipeline/01_run_qc.py
python3 -u code/pipeline/02_run_mapmycells.py
python3 -u code/pipeline/02b_run_correlation_classifier.py
python3 -u code/pipeline/03_export_transcripts.py
# python3 -u code/nuclear_resolution/04_run_nuclear_doublet_resolution.py  # (optional)
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
| `output/h5ad/correlation_centroids.pkl` | Pre-built correlation centroids (from step 02b) |
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
| **Depth & layers (steps 05-06)** | | |
| `predicted_norm_depth` | float64 | Predicted normalized cortical depth (0=pia, 1=WM) |
| `banksy_cluster` | int | BANKSY cluster ID (-1 for QC-failed cells) |
| `banksy_domain` | str | Cortical / Vascular / WM (BANKSY-based domain) |
| `banksy_is_l1` | bool | True if cell in L1 border cluster (shallow, non-neuronal) |
| `spatial_domain` | category | Cortical / Vascular (backward-compatible; WM mapped to Cortical) |
| `layer` | category | L1 / L2/3 / L4 / L5 / L6 / WM / Vascular (spatially smoothed) |
| `layer_unsmoothed` | category | Pre-smoothing layer (depth bins + Vascular override) |
| `layer_depth_only` | category | Layer from depth bins only (no Vascular override, no smoothing) |
| **Nuclear doublet resolution (optional step 04)** | | |
| `nuclear_total_counts` | int | Total UMI counts from nuclear-only transcripts |
| `nuclear_n_genes` | int | Number of genes detected in nuclear transcripts |
| `nuclear_fraction` | float32 | Fraction of UMIs in nucleus vs whole cell |
| `nuclear_doublet_suspect` | bool | Nuclear-level doublet detected |
| `nuclear_doublet_type` | str | '' / 'Glut+GABA' / 'GABA+GABA' (nuclear evidence) |
| `nuclear_doublet_status` | str | 'clean' / 'resolved' / 'persistent' / 'nuclear_only' / 'insufficient' |
| `hybrid_qc_pass` | bool | Hybrid QC pass (nuclear-informed; legacy, not used by default) |

**Notes:**
- The correlation classifier columns (`corr_*`, `doublet_*`) are present only in samples processed through step 02b. Analysis scripts prefer `corr_subclass`/`corr_supertype` when available, falling back to HANN labels otherwise.
- The nuclear doublet columns are present only in samples where step 04 was previously run. These columns are retained for reference but `corr_qc_pass` is the default QC gate for all analyses.

## Cell Type Taxonomy

The pipeline uses the [SEA-AD MTG taxonomy](https://portal.brain-map.org/):

- **3 classes:** Neuronal: Glutamatergic, Neuronal: GABAergic, Non-neuronal and Non-neural
- **24 subclasses:** L2/3 IT, L4 IT, L5 IT, L5 ET, L5/6 NP, L6 IT, L6 IT Car3, L6 CT, L6b, Sst, Sst Chodl, Pvalb, Vip, Lamp5, Lamp5 Lhx6, Sncg, Pax6, Chandelier, Astrocyte, Oligodendrocyte, OPC, Microglia-PVM, Endothelial, VLMC
- **127+ supertypes:** Fine-grained subtypes within each subclass (e.g., Sst_1 through Sst_25)

## Layer Assignment

Layers are assigned through a combined approach:

1. **BANKSY spatial domain classification** (step 06) uses BANKSY clustering (Nature Genetics 2024; λ=0.8, res=0.3) to identify spatially coherent tissue domains: Cortical (including L1 border cells), Vascular, and White Matter. BANKSY augments gene expression with spatial neighbor expression, producing clusters that respect tissue geometry. L1 border cells (shallow, non-neuronal-dominated clusters) are correctly identified as Cortical with a `banksy_is_l1` flag — validated by SEA-AD MERFISH comparison showing L1 has ~81% non-neuronal composition.
2. **MERFISH depth model** (step 05; GradientBoostingRegressor on K=50 neighborhood composition features) predicts normalized cortical depth (0=pia, 1=WM; R²=0.90, MAE=0.031 on held-out donors), then bins into discrete layers
3. **Layer bins:** L1 (<0.10), L2/3 (0.10–0.30), L4 (0.30–0.45), L5 (0.45–0.65), L6 (0.65–0.85), WM (>0.85)
4. **Spatial smoothing** refines layer boundaries via 3-step pipeline: within-domain majority vote (k=30, 2 rounds) smooths noisy boundaries while respecting BANKSY domains; vascular border trim reassigns Vascular cells whose neighborhoods are predominantly cortical; BANKSY-anchored L1 contiguity promotes shallow `banksy_is_l1` cells to L1 and removes isolated L1 assignments. This reduces Vascular from 17.9% (BANKSY domain) to 6.8% (smoothed layer) and improves L1 contiguity.

Final layer categories: **L1 (5.2%), L2/3 (17.6%), L4 (12.0%), L5 (23.2%), L6 (10.1%), WM (25.1%), Vascular (6.8%)** — median depth per subclass correlates at r=0.92 with the MERFISH reference.

See **[Depth & Layer Inference Methods](depth_layer_methods_writeup.md)** for full details, validation figures, and design decisions.

## Archive (`code/archive/`)

Contains legacy and exploratory scripts from earlier iterations of the analysis:

| Directory | Contents |
|-----------|----------|
| `stale_analysis/` | Archived analysis scripts: diagnostic comparisons, edgepython DE, Harmony transfer, nsforest markers, calibration scripts |
| `one_time_utils/` | One-time data migration scripts (rename columns, annotate MERFISH depth) |
| `legacy_runners/` | Old pipeline runners (pre-numbered-step architecture) |
| `ood_methods/` | OOD method exploration scripts (superseded by BANKSY domain classification) |
| `spatial_domain_exploration/` | Early spatial domain clustering experiments |
| `banksy_exploration/` | BANKSY parameter tuning (pilot), early domain+strip prototype, validation, batch runner. Logic extracted to `modules/banksy_domains.py` |
| `curved_strips/` | Curved cortex strip identification: pia curve fitting, strip boundary detection, MERFISH adapter, gallery visualization. Experimental approach for selecting cortical strips with complete L1–L6 laminar structure |
| *(root)* | Archived modules: `label_transfer.py`, `layers.py`, old pipeline scripts |

These are kept for reference but are not part of the active pipeline.
