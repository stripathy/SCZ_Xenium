"""
Shared configuration for the SCZ Xenium processing pipeline.

Centralizes all paths, constants, and settings used across pipeline steps
(00_create_h5ad through 05_export_viewer). Import from here instead of
hardcoding paths in each script.

This is the pipeline equivalent of code/analysis/config.py.
"""

import os

# ──────────────────────────────────────────────────────────────────────
# Base paths
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = "/Users/shreejoy/Desktop/scz_xenium_test"

# Input data
RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
METADATA_PATH = os.path.join(BASE_DIR, "sample_metadata.xlsx")

# Reference data (SEA-AD)
MERFISH_PATH = os.path.join(BASE_DIR, "data", "reference",
                            "SEAAD_MTG_MERFISH.2024-12-11.h5ad")
PRECOMPUTED_STATS_PATH = os.path.join(
    BASE_DIR, "data", "reference",
    "precomputed_stats.20231120.sea_ad.MTG.h5"
)
GENE_MAPPING_PATH = os.path.join(
    BASE_DIR, "data", "reference",
    "gene_symbol_to_ensembl.json"
)

# Output directories
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
H5AD_DIR = os.path.join(OUTPUT_DIR, "h5ad")
VIEWER_DIR = os.path.join(OUTPUT_DIR, "viewer")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

# Model artifacts
DEPTH_MODEL_PATH = os.path.join(OUTPUT_DIR, "depth_model_normalized.pkl")

# ──────────────────────────────────────────────────────────────────────
# Processing settings
# ──────────────────────────────────────────────────────────────────────

N_WORKERS = 4  # for multiprocessing in steps 03/04

# MapMyCells settings (step 02)
MAPMYCELLS_BOOTSTRAP_ITER = 100
MAPMYCELLS_BOOTSTRAP_FACTOR = 0.5
MAPMYCELLS_N_PER_UTILITY = 30

# Taxonomy levels assigned by MapMyCells
TAXONOMY_LEVELS = ["class", "subclass", "supertype"]

# ──────────────────────────────────────────────────────────────────────
# Modules path helper
# ──────────────────────────────────────────────────────────────────────

PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
CODE_DIR = os.path.dirname(PIPELINE_DIR)
MODULES_DIR = os.path.join(CODE_DIR, "modules")
