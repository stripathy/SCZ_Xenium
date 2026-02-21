#!/usr/bin/env python3
"""
Step 2: Cell type annotation using Allen Institute's MapMyCells.

Uses bootstrapped hierarchical mapping against the SEA-AD MTG taxonomy
to assign class, subclass, and supertype labels to each cell.

For each sample:
  1. Load existing h5ad (from steps 00+01, with expression + QC columns)
  2. Subset to qc_pass == True cells
  3. Convert gene symbols to Ensembl IDs (reference uses Ensembl)
  4. Run MapMyCells hierarchical mapping (100 bootstrap iterations)
  5. Parse output: extract *_name → *_label, *_bootstrapping_probability → *_label_confidence
  6. Write labels back to h5ad (QC-fail cells get 'Unassigned' + 0.0 confidence)

Requires:
  - Python 3.10+
  - cell_type_mapper >= 1.7.0:
      pip install cell_type_mapper@git+https://github.com/AllenInstitute/cell_type_mapper
  - SEA-AD MTG precomputed stats (~1-2 GB, download once):
      wget -P data/reference/ https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/
        mapmycells/SEAAD/20240831/precomputed_stats.20231120.sea_ad.MTG.h5

Usage:
    python3 -u 02_run_mapmycells.py [--sample SAMPLE_ID] [--n-workers N]
"""

import os
import sys
import time
import json
import argparse
import tempfile
import numpy as np
import pandas as pd
import anndata as ad

# Pipeline config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_config import (
    H5AD_DIR, OUTPUT_DIR, RAW_DIR, MODULES_DIR,
    PRECOMPUTED_STATS_PATH, GENE_MAPPING_PATH,
    TAXONOMY_LEVELS,
    MAPMYCELLS_BOOTSTRAP_ITER, MAPMYCELLS_BOOTSTRAP_FACTOR,
    MAPMYCELLS_N_PER_UTILITY,
)

# Modules
sys.path.insert(0, MODULES_DIR)
from loading import discover_samples


# ──────────────────────────────────────────────────────────────────────
# Gene mapping utilities
# ──────────────────────────────────────────────────────────────────────

def load_gene_mapping():
    """Load gene symbol → Ensembl ID mapping from JSON."""
    with open(GENE_MAPPING_PATH) as f:
        return json.load(f)


def convert_genes_to_ensembl(adata, gene_mapping):
    """
    Convert gene symbols in adata.var_names to Ensembl IDs.

    Drops genes that can't be mapped. Returns a new AnnData with
    Ensembl IDs as var_names.
    """
    mappable = [g for g in adata.var_names if g in gene_mapping]
    adata_sub = adata[:, mappable].copy()
    adata_sub.var_names = [gene_mapping[g] for g in mappable]
    adata_sub.var_names_make_unique()
    return adata_sub


# ──────────────────────────────────────────────────────────────────────
# MapMyCells runner
# ──────────────────────────────────────────────────────────────────────

def run_mapmycells_on_sample(query_h5ad_path, output_dir, n_processors=1):
    """
    Run MapMyCells hierarchical mapping on a single h5ad file.

    Returns (csv_path, json_path) for the results.
    """
    from cell_type_mapper.cli.map_to_on_the_fly_markers import OnTheFlyMapper

    csv_path = os.path.join(output_dir, "mapmycells_output.csv")
    json_path = os.path.join(output_dir, "mapmycells_output.json")

    config = {
        "query_path": query_h5ad_path,
        "extended_result_path": json_path,
        "csv_result_path": csv_path,
        "precomputed_stats": {
            "path": PRECOMPUTED_STATS_PATH,
        },
        "type_assignment": {
            "normalization": "raw",
            "bootstrap_iteration": MAPMYCELLS_BOOTSTRAP_ITER,
            "bootstrap_factor": MAPMYCELLS_BOOTSTRAP_FACTOR,
            "algorithm": "hierarchical",
        },
        "n_processors": n_processors,
        "query_markers": {
            "n_per_utility": MAPMYCELLS_N_PER_UTILITY,
        },
        "reference_markers": {
            "precomputed_path_list": None,
        },
    }

    runner = OnTheFlyMapper(args=[], input_data=config)
    runner.run()

    return csv_path, json_path


def parse_mapmycells_output(csv_path):
    """
    Parse MapMyCells CSV output and extract human-readable labels.

    MapMyCells outputs columns:
      {level}_label (machine ID), {level}_name (human-readable),
      {level}_bootstrapping_probability

    We extract *_name → rename to *_label, and *_bootstrapping_probability
    → rename to *_label_confidence.

    Returns a DataFrame with our standard column names:
      class_label, subclass_label, supertype_label,
      class_label_confidence, subclass_label_confidence, supertype_label_confidence
    """
    df = pd.read_csv(csv_path, comment='#')

    result = pd.DataFrame(index=range(len(df)))

    for level in TAXONOMY_LEVELS:
        name_col = f"{level}_name"
        label_col = f"{level}_label"
        prob_col = f"{level}_bootstrapping_probability"

        # Prefer human-readable *_name; fall back to *_label (machine IDs)
        if name_col in df.columns:
            result[f"{level}_label"] = df[name_col].values
        elif label_col in df.columns:
            result[f"{level}_label"] = df[label_col].values
            print(f"  WARNING: Using machine IDs for {level} (no *_name column)")

        # Bootstrapping probability → confidence
        if prob_col in df.columns:
            result[f"{level}_label_confidence"] = df[prob_col].values.astype(np.float32)
        else:
            result[f"{level}_label_confidence"] = np.float32(0.0)

    # Handle older taxonomies that use "cluster" instead of "supertype"
    if "supertype_label" not in result.columns:
        for alt in ["cluster_name", "cluster_label"]:
            if alt in df.columns:
                result["supertype_label"] = df[alt].values
                print(f"  NOTE: Used '{alt}' as supertype_label")
                break
        if "cluster_bootstrapping_probability" in df.columns:
            result["supertype_label_confidence"] = \
                df["cluster_bootstrapping_probability"].values.astype(np.float32)

    return result


# ──────────────────────────────────────────────────────────────────────
# Per-sample processing
# ──────────────────────────────────────────────────────────────────────

def process_one_sample(sample_info, gene_mapping):
    """
    Run MapMyCells label transfer on one sample.

    Reads the existing h5ad (with expression matrix + QC columns from
    steps 00 and 01), runs MapMyCells on QC-pass cells, and writes
    the label columns back.
    """
    sid = sample_info["sample_id"]
    h5ad_path = os.path.join(H5AD_DIR, f"{sid}_annotated.h5ad")
    t0 = time.time()

    try:
        # 1. Load existing h5ad
        print(f"  {sid}: Loading h5ad...")
        adata = ad.read_h5ad(h5ad_path)
        n_total = adata.shape[0]

        # 2. Get QC mask
        if "qc_pass" not in adata.obs.columns:
            print(f"  {sid}: WARNING - no qc_pass column, using all cells")
            qc_mask = np.ones(n_total, dtype=bool)
        else:
            qc_mask = adata.obs["qc_pass"].values.astype(bool)

        n_pass = int(qc_mask.sum())
        n_fail = n_total - n_pass
        print(f"  {sid}: {n_total:,} total, {n_pass:,} QC pass, "
              f"{n_fail:,} QC fail ({100*n_fail/n_total:.1f}%)")

        # 3. Subset to QC-pass cells and convert genes to Ensembl
        adata_pass = adata[qc_mask].copy()

        with tempfile.TemporaryDirectory() as tmpdir:
            adata_ensembl = convert_genes_to_ensembl(adata_pass, gene_mapping)
            n_mapped = adata_ensembl.shape[1]
            print(f"  {sid}: Mapped {n_mapped}/{adata_pass.shape[1]} genes to Ensembl IDs")

            # 4. Save temp h5ad for MapMyCells (it needs a file path)
            tmp_h5ad = os.path.join(tmpdir, f"{sid}_qcpass.h5ad")
            adata_ensembl.write_h5ad(tmp_h5ad)

            # 5. Run MapMyCells
            print(f"  {sid}: Running MapMyCells hierarchical mapping...")
            t_map = time.time()
            csv_path, json_path = run_mapmycells_on_sample(
                tmp_h5ad, tmpdir, n_processors=1
            )
            print(f"  {sid}: MapMyCells done in {time.time()-t_map:.0f}s")

            # 6. Parse results
            labels_df = parse_mapmycells_output(csv_path)

            # Log summary
            for level in TAXONOMY_LEVELS:
                col = f"{level}_label"
                conf_col = f"{level}_label_confidence"
                if col in labels_df.columns:
                    n_unique = labels_df[col].nunique()
                    mean_conf = labels_df[conf_col].mean() if conf_col in labels_df.columns else 0
                    print(f"  {sid}: {level}: {n_unique} types, "
                          f"mean confidence={mean_conf:.3f}")

        # 7. Write labels into the full h5ad
        # Initialize all cells as Unassigned
        for level in TAXONOMY_LEVELS:
            adata.obs[f"{level}_label"] = "Unassigned"
            adata.obs[f"{level}_label_confidence"] = np.float32(0.0)

        # Copy labels for QC-pass cells
        pass_indices = np.where(qc_mask)[0]
        for col in labels_df.columns:
            if col in adata.obs.columns:
                adata.obs.iloc[pass_indices,
                               adata.obs.columns.get_loc(col)] = \
                    labels_df[col].values

        # 8. Save updated h5ad
        adata.write_h5ad(h5ad_path)

        elapsed = time.time() - t0
        print(f"  {sid}: Done in {elapsed:.0f}s")

        return {
            "sample_id": sid, "status": "success",
            "n_total": n_total, "n_pass": n_pass,
            "n_fail": n_fail, "time": elapsed,
        }

    except Exception as e:
        import traceback
        elapsed = time.time() - t0
        print(f"  {sid}: FAILED after {elapsed:.0f}s - {e}")
        traceback.print_exc()
        return {
            "sample_id": sid, "status": "error",
            "error": str(e), "time": elapsed,
        }


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Step 2: Run MapMyCells label transfer on Xenium samples"
    )
    parser.add_argument("--sample", type=str, default=None,
                        help="Process only this sample ID (for testing)")
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Number of MapMyCells internal processors")
    args = parser.parse_args()

    t_start = time.time()

    # Check prerequisites
    if not os.path.exists(PRECOMPUTED_STATS_PATH):
        print(f"ERROR: SEA-AD MTG precomputed stats not found at:")
        print(f"  {PRECOMPUTED_STATS_PATH}")
        print(f"\nDownload with:")
        print(f"  wget -P {os.path.dirname(PRECOMPUTED_STATS_PATH)} \\")
        print(f"    https://allen-brain-cell-atlas.s3.us-west-2.amazonaws.com/"
              f"mapmycells/SEAAD/20240831/precomputed_stats.20231120.sea_ad.MTG.h5")
        sys.exit(1)

    if not os.path.exists(GENE_MAPPING_PATH):
        print(f"ERROR: Gene mapping not found at {GENE_MAPPING_PATH}")
        print("Run the gene mapping script first.")
        sys.exit(1)

    gene_mapping = load_gene_mapping()
    print(f"Loaded gene mapping: {len(gene_mapping)} symbol→Ensembl mappings")

    # Discover samples from raw data directory
    all_samples = discover_samples(RAW_DIR)
    print(f"Found {len(all_samples)} samples")

    if args.sample:
        all_samples = [s for s in all_samples if s["sample_id"] == args.sample]
        if not all_samples:
            print(f"ERROR: Sample '{args.sample}' not found")
            sys.exit(1)
        print(f"Processing single sample: {args.sample}")

    print(f"\nMethod: MapMyCells hierarchical mapping (SEA-AD MTG taxonomy)")
    print(f"Reference: {PRECOMPUTED_STATS_PATH}")
    print(f"{'='*60}")

    # Process sequentially (MapMyCells has its own internal parallelism)
    results = []
    for i, sample_info in enumerate(all_samples):
        print(f"\n[{i+1}/{len(all_samples)}] Processing {sample_info['sample_id']}...")
        result = process_one_sample(sample_info, gene_mapping=gene_mapping)
        results.append(result)

    # Summary
    total_time = time.time() - t_start
    n_ok = sum(1 for r in results if r["status"] == "success")
    total_cells = sum(r.get("n_total", 0) for r in results)
    total_pass = sum(r.get("n_pass", 0) for r in results)

    print(f"\n{'='*60}")
    print(f"SUMMARY - {total_time:.0f}s total")
    print(f"{'='*60}")
    print(f"  Success: {n_ok}/{len(results)}")
    if total_cells > 0:
        print(f"  Total cells: {total_cells:,}")
        print(f"  QC pass: {total_pass:,} ({100*total_pass/total_cells:.1f}%)")

    for r in results:
        status = "✓" if r["status"] == "success" else "✗"
        t = r.get("time", 0)
        n = r.get("n_total", 0)
        print(f"  {status} {r['sample_id']:10s}: {n:>6,} cells, {t:>4.0f}s")

    for r in results:
        if r["status"] == "error":
            print(f"  FAILED: {r['sample_id']}: {r.get('error', 'unknown')}")

    print(f"\nTotal time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
