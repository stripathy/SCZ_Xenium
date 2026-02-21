#!/usr/bin/env python3
"""
Build crumblr input CSVs for SCZ vs Control compositional analysis.

Uses cortical-only cells (cropped: spatial_domain=='Cortical' AND layer!='WM')
from all Xenium samples (excluding Br2039 outlier). Generates whole-composition
inputs (neurons + non-neurons together) at subclass and supertype levels.

Output:
  output/crumblr/crumblr_input_subclass.csv
  output/crumblr/crumblr_input_supertype.csv

Each CSV is long-format: donor, celltype, count, total, diagnosis, sex, age
"""

import os
import sys
import glob
import time
import pandas as pd
import anndata as ad

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BASE_DIR, H5AD_DIR, CRUMBLR_DIR, EXCLUDE_SAMPLES, SUBCLASS_CONF_THRESH

sys.path.insert(0, os.path.join(BASE_DIR, "code"))
from modules.metadata import get_subject_info

METADATA_PATH = os.path.join(BASE_DIR, "sample_metadata.xlsx")


def load_sample(fpath):
    """Load one h5ad, filter to QC-pass cortical cells with confidence filter."""
    adata = ad.read_h5ad(fpath, backed="r")
    obs = adata.obs[["sample_id", "subclass_label", "supertype_label",
                      "subclass_label_confidence",
                      "spatial_domain", "layer", "qc_pass"]].copy()

    # Filter: QC pass + cortical + not WM
    mask = (
        (obs["qc_pass"] == True) &
        (obs["spatial_domain"] == "Cortical") &
        (obs["layer"] != "WM")
    )
    obs = obs[mask]

    # Bottom-1% subclass confidence filter
    n_before = len(obs)
    obs = obs[obs["subclass_label_confidence"].astype(float) >= SUBCLASS_CONF_THRESH]
    n_dropped = n_before - len(obs)
    if n_dropped > 0:
        pct = n_dropped / n_before * 100
        print(f"    Confidence filter: dropped {n_dropped:,} cells ({pct:.1f}%)")

    return obs


def build_counts(obs_df, level_col):
    """Build long-format count table: one row per (sample, celltype)."""
    counts = obs_df.groupby(["sample_id", level_col]).size().reset_index(name="count")
    totals = obs_df.groupby("sample_id").size().reset_index(name="total")
    counts = counts.merge(totals, on="sample_id")
    counts = counts.rename(columns={"sample_id": "donor", level_col: "celltype"})
    return counts


def main():
    t0 = time.time()
    os.makedirs(CRUMBLR_DIR, exist_ok=True)

    # Load donor metadata
    print("Loading donor metadata...")
    meta = get_subject_info(METADATA_PATH)
    meta = meta.set_index("sample_id")
    print(f"  {len(meta)} Xenium subjects in metadata")

    # Discover h5ad files
    h5ad_files = sorted(glob.glob(os.path.join(H5AD_DIR, "*_annotated.h5ad")))
    print(f"\nFound {len(h5ad_files)} h5ad files")

    # Load and concatenate
    all_obs = []
    for fpath in h5ad_files:
        sid = os.path.basename(fpath).replace("_annotated.h5ad", "")
        if sid in EXCLUDE_SAMPLES:
            print(f"  Skipping {sid} (excluded)")
            continue

        obs = load_sample(fpath)
        n_total = len(obs)
        print(f"  {sid}: {n_total:,} cortical cells")
        all_obs.append(obs)

    df = pd.concat(all_obs, ignore_index=True)
    n_samples = df["sample_id"].nunique()
    print(f"\nTotal: {len(df):,} cortical cells from {n_samples} samples")

    # Build count tables at both levels
    for level_col, level_name in [("subclass_label", "subclass"),
                                   ("supertype_label", "supertype")]:
        counts = build_counts(df, level_col)

        # Attach metadata
        counts = counts.merge(
            meta[["diagnosis", "sex", "age"]].reset_index(),
            left_on="donor", right_on="sample_id", how="left"
        ).drop(columns=["sample_id"])

        # Sort for readability
        counts = counts.sort_values(["donor", "celltype"]).reset_index(drop=True)

        n_types = counts["celltype"].nunique()
        outpath = os.path.join(CRUMBLR_DIR, f"crumblr_input_{level_name}.csv")
        counts.to_csv(outpath, index=False)
        print(f"\n  {level_name}: {n_samples} donors x {n_types} types -> {outpath}")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")


if __name__ == "__main__":
    main()
