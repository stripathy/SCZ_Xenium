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
│   ├── GSM9223468_Br2039-transcripts.zarr.zip
│   ├── GSM9223469_Br2719-cell_feature_matrix.h5
│   ├── ...                           # (24 samples x 4 files = 96 files)
│   └── GSM9223491_Br8772-transcripts.zarr.zip
├── reference/                        # Reference datasets (Steps 2-4)
│   ├── SEAAD_MTG_MERFISH.2024-12-11.h5ad
│   ├── nicole_sea_ad_snrnaseq_reference.h5ad
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
wget -r -np -nd -A "*.h5,*.csv.gz,*.zarr.zip" \
  https://ftp.ncbi.nlm.nih.gov/geo/series/GSE307nnn/GSE307404/suppl/

# Option B: Use GEOquery or the GEO download manager from the accession page
```

**Contents:** 24 Xenium sections (12 schizophrenia, 12 control) from human dorsolateral prefrontal cortex (DLPFC). Each sample has 4 files:
- `cell_feature_matrix.h5` — Gene expression counts (541 features: 300 genes + controls)
- `cell_boundaries.csv.gz` — Cell boundary polygons
- `nucleus_boundaries.csv.gz` — Nucleus boundary polygons
- `transcripts.zarr.zip` — Molecule-level transcript coordinates in Zarr format (x, y positions for every detected RNA molecule; used by step 03 for transcript export and step 04 for nuclear doublet resolution)

**Total size:** ~34 GB (~809 MB for cell matrices + boundaries, ~33 GB for transcript coordinate archives)

---

## Step 2: SEA-AD MERFISH Reference

**Source:** [Gabitto et al. (2024)](https://doi.org/10.1038/s41593-024-01774-5) — Seattle Alzheimer's Disease Brain Cell Atlas.

**Used for:** Training the cortical depth model (step 05), fitting the OOD/spatial domain classifier (step 06), and cell type proportion validation against Xenium.

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

## Step 4: SEA-AD snRNAseq Reference (Nicole's)

**Source:** Nicole Comfort's curated subset of the SEA-AD MTG snRNAseq data — neurotypical donors only, with harmonized cell type annotations matching the MERFISH taxonomy.

**Used for:** Ground-truth cell type proportions, doublet detection validation (step 02b), and as a full-transcriptome reference (36,601 genes vs MERFISH's 180 genes).

**Contents:** 137,303 cells from 5 neurotypical donors. Same 24-subclass / 137-supertype SEA-AD MTG taxonomy as the MERFISH reference.

**Conversion:** The original RDS files (`Neurotypical_ref_metadata.rds`, `raw_counts_ref.rds`) were converted to h5ad format using `/tmp/convert_rds_to_h5ad.R`. The resulting file is:

```
data/reference/nicole_sea_ad_snrnaseq_reference.h5ad  (~9 GB)
```

**Key columns:** `Class` (Neuronal: Glutamatergic / Neuronal: GABAergic / Non-neuronal and Non-neural), `Subclass`, `Supertype`, `donor_id`, `Age.at.death`, `Braak.stage`, `CERAD.score`.

**Note:** This replaces the older Allen Institute snRNAseq reference (`Reference_MTG_RNAseq_final-nuclei.2022-06-07.h5ad`) which was only used by legacy scripts in `code/archive/`.

---

## Sample Metadata

The file `sample_metadata.xlsx` is already included in the repository. It contains donor demographics (age, sex, PMI, RIN) and diagnosis (SCZ vs. Control) for all 24 samples. Originally from the Kwon et al. manuscript supplementary materials.

---

## Summary

| Dataset | Size | Required? | Source |
|---------|------|-----------|--------|
| Raw Xenium data (matrices + boundaries) | 809 MB | Yes | [GEO GSE307404](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307404) |
| Raw Xenium transcript coordinates | ~33 GB | Yes (steps 03-04) | [GEO GSE307404](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE307404) |
| SEA-AD MERFISH | 3.1 GB | Yes | [Allen Brain Cell Atlas](https://sea-ad-spatial-transcriptomics.s3.us-west-2.amazonaws.com/middle-temporal-gyrus/all_donors-h5ad/SEAAD_MTG_MERFISH.2024-12-11.h5ad) |
| SEA-AD snRNAseq (Nicole's) | 9 GB | Yes | Converted from Nicole's RDS files (see Step 4) |
| MapMyCells stats | 251 MB | Yes (step 02) | [Allen Brain Cell Atlas](https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/mapmycells/SEAAD/20240831/precomputed_stats.20231120.sea_ad.MTG.h5) |
| **Total required** | **~46 GB** | | |
