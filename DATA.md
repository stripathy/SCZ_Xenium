# Data Download Instructions

This document describes how to obtain all input datasets required to run the SCZ Xenium analysis pipeline from scratch.

## Directory Structure

After downloading, your `data/` directory should look like:

```
data/
├── raw/                              # Xenium data from GEO (Step 1)
│   ├── GSM9223468_Br2039-cell_feature_matrix.h5
│   ├── GSM9223468_Br2039-cell_boundaries.csv.gz
│   ├── GSM9223468_Br2039-nucleus_boundaries.csv.gz
│   ├── GSM9223469_Br2719-cell_feature_matrix.h5
│   ├── ...                           # (24 samples x 3 files = 72 files)
│   └── GSM9223491_Br8772-nucleus_boundaries.csv.gz
├── reference/                        # Reference datasets (Steps 2-4)
│   ├── SEAAD_MTG_MERFISH.2024-12-11.h5ad
│   ├── precomputed_stats.20231120.sea_ad.MTG.h5
│   └── gene_symbol_to_ensembl.json   # (already in repo)
└── nicole_scz_snrnaseq_betas/        # (already in repo)
    └── scz_coefs.xlsx
```

---

## Step 1: Raw Xenium Data (GEO: GSE307404)

**Source:** [Kwon et al. (2026)](https://doi.org/10.64898/2026.02.16.706214) — Schizophrenia Xenium spatial transcriptomics of human DLPFC.

**GEO Accession:** [GSE307404](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307404)

**Note:** The GEO data may still be under embargo while the manuscript is in review. Check the GEO page for availability. Once public, download all supplementary files:

```bash
mkdir -p data/raw
cd data/raw

# Option A: Download from GEO FTP (when available)
wget -r -np -nd -A "*.h5,*.csv.gz" \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE307nnn/GSE307404/suppl/

# Option B: Use GEOquery or the GEO download manager from the accession page
```

**Contents:** 24 Xenium sections (12 schizophrenia, 12 control) from human dorsolateral prefrontal cortex (DLPFC). Each sample has 3 files:
- `cell_feature_matrix.h5` — Gene expression counts (541 features: 300 genes + controls)
- `cell_boundaries.csv.gz` — Cell boundary polygons
- `nucleus_boundaries.csv.gz` — Nucleus boundary polygons

**Total size:** ~809 MB

---

## Step 2: SEA-AD MERFISH Reference

**Source:** [Gabitto et al. (2024)](https://doi.org/10.1038/s41593-024-01774-5) — Seattle Alzheimer's Disease Brain Cell Atlas.

**Used for:** Training the cortical depth model (step 03) and fitting the OOD/spatial domain classifier (step 04).

```bash
mkdir -p data/reference

# Download MERFISH reference (~3.1 GB)
wget -O data/reference/SEAAD_MTG_MERFISH.2024-12-11.h5ad \
  "https://sea-ad-spatial-transcriptomics.s3.us-west-2.amazonaws.com/middle-temporal-gyrus/all_donors-h5ad/SEAAD_MTG_MERFISH.2024-12-11.h5ad"

# Or using AWS CLI (faster, supports resume):
aws s3 cp \
  s3://sea-ad-spatial-transcriptomics/middle-temporal-gyrus/all_donors-h5ad/SEAAD_MTG_MERFISH.2024-12-11.h5ad \
  data/reference/ --no-sign-request
```

---

## Step 3: MapMyCells Precomputed Stats

**Source:** [Allen Brain Cell Atlas](https://portal.brain-map.org/) — Precomputed statistics for hierarchical cell type mapping against the SEA-AD MTG taxonomy.

**Used for:** Cell type annotation via MapMyCells (step 02). Contains marker gene statistics and taxonomy tree for the 127-supertype SEA-AD MTG classification.

```bash
# Download precomputed stats (~251 MB)
wget -O data/reference/precomputed_stats.20231120.sea_ad.MTG.h5 \
  "https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/mapmycells/SEAAD/20240831/precomputed_stats.20231120.sea_ad.MTG.h5"
```

**Note:** Step 02 also requires the `cell_type_mapper` Python package, which needs **Python 3.10+**:

```bash
pip install "cell_type_mapper @ git+https://github.com/AllenInstitute/cell_type_mapper"
```

---

## Step 4 (Optional): SEA-AD snRNAseq Reference

**Only needed if** you want to run the legacy correlation-based label transfer (in `code/archive/`). The main pipeline uses MapMyCells (step 02) instead.

```bash
# Download snRNAseq reference (~5.9 GB)
wget -O data/reference/Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad \
  "https://sea-ad-single-cell-profiling.s3.us-west-2.amazonaws.com/MTG/RNAseq/Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad"
```

---

## Sample Metadata

The file `sample_metadata.xlsx` is already included in the repository. It contains donor demographics (age, sex, PMI, RIN) and diagnosis (SCZ vs. Control) for all 24 samples. Originally from the Kwon et al. manuscript supplementary materials.

---

## Summary

| Dataset | Size | Required? | Source |
|---------|------|-----------|--------|
| Raw Xenium data | 809 MB | Yes | [GEO GSE307404](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307404) |
| SEA-AD MERFISH | 3.1 GB | Yes | [Allen Brain Cell Atlas](https://sea-ad-spatial-transcriptomics.s3.us-west-2.amazonaws.com/middle-temporal-gyrus/all_donors-h5ad/SEAAD_MTG_MERFISH.2024-12-11.h5ad) |
| MapMyCells stats | 251 MB | Yes (step 02) | [Allen Brain Cell Atlas](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/mapmycells/SEAAD/20240831/precomputed_stats.20231120.sea_ad.MTG.h5) |
| SEA-AD snRNAseq | 5.9 GB | No (legacy only) | [Allen Brain Cell Atlas](https://sea-ad-single-cell-profiling.s3.us-west-2.amazonaws.com/MTG/RNAseq/Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad) |
| **Total required** | **~4.2 GB** | | |
