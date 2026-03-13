"""
Shared configuration for the SCZ Xenium processing pipeline.

Centralizes all paths, constants, and settings used across pipeline steps
(00_create_h5ad through 08_export_boundaries). Import from here instead of
hardcoding paths in each script.

This is the pipeline equivalent of code/analysis/config.py.
"""

import os

# ──────────────────────────────────────────────────────────────────────
# Base paths
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.expanduser("~/Github/SCZ_Xenium")

# Input data
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
METADATA_PATH = os.path.join(BASE_DIR, "sample_metadata.xlsx")

# Reference data (SEA-AD)
# MERFISH spatial reference — used ONLY for depth model training (has spatial coords)
MERFISH_PATH = os.path.join(BASE_DIR, "data", "reference",
                            "SEAAD_MTG_MERFISH.2024-12-11.h5ad")
# Nicole's snRNAseq reference — primary cell type reference (137K cells, 36K genes)
SNRNASEQ_REF_PATH = os.path.join(BASE_DIR, "data", "reference",
                                  "nicole_sea_ad_snrnaseq_reference.h5ad")
PRECOMPUTED_STATS_PATH = os.path.join(
    BASE_DIR, "data", "reference",
    "precomputed_stats.20231120.sea_ad.MTG.h5"
)
GENE_MAPPING_PATH = os.path.join(
    BASE_DIR, "data", "reference",
    "gene_symbol_to_ensembl.json"
)
# Comprehensive gene mapping (36K genes, built from snRNAseq reference + precomputed stats)
GENE_MAPPING_COMPREHENSIVE_PATH = os.path.join(
    BASE_DIR, "data", "reference",
    "gene_symbol_to_ensembl_comprehensive.json"
)

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
H5AD_DIR = os.path.join(OUTPUT_DIR, "h5ad")
VIEWER_DIR = os.path.join(OUTPUT_DIR, "viewer")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Model artifacts
DEPTH_MODEL_PATH = os.path.join(OUTPUT_DIR, "depth_model_normalized.pkl")
CENTROID_PATH = os.path.join(H5AD_DIR, "correlation_centroids.pkl")

# ──────────────────────────────────────────────────────────────────────
# Processing settings
# ──────────────────────────────────────────────────────────────────────

N_WORKERS = 4  # for multiprocessing in steps 05/06

# MapMyCells settings (step 02)
MAPMYCELLS_BOOTSTRAP_ITER = 100
MAPMYCELLS_BOOTSTRAP_FACTOR = 0.5
MAPMYCELLS_N_PER_UTILITY = 30

# Taxonomy levels assigned by MapMyCells
TAXONOMY_LEVELS = ["class", "subclass", "supertype"]

# Correlation classifier settings (step 02b)
CORR_CLASSIFIER_TOP_N = 100       # exemplar cells per type for centroid building
CORR_CLASSIFIER_QC_PERCENTILE = 1.0  # bottom % of margin to flag per sample

# Nuclear doublet resolution settings (step 04)
NUCLEAR_CHUNK_SIZE = 500_000      # transcripts per STRtree query batch
NUCLEAR_MIN_UMI = 50              # min nuclear UMI for reliable doublet assessment
NUCLEAR_MIN_CORR = 0.3            # min correlation for high-confidence resolution

# Transcript export directory (step 03 output, step 04 input)
TRANSCRIPT_DIR = os.path.join(VIEWER_DIR, "transcripts")

# ──────────────────────────────────────────────────────────────────────
# Modules path helper
# ──────────────────────────────────────────────────────────────────────

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(PIPELINE_DIR)
MODULES_DIR = os.path.join(CODE_DIR, "modules")
