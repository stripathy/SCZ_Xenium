#!/usr/bin/env python3
"""
Step 6: Spatial domain classification + depth-based layer assignment.

For each sample:
  1. Load annotated h5ad (with QC, hierarchical labels, depth predictions)
  2. Subset to hybrid_qc_pass cells (from step 04; falls back to qc_pass)
  3. Run spatial domain clustering to identify Extra-cortical / Vascular / Cortical
  4. Assign layers: ALL cells get depth-bin layer from predicted depth,
     EXCEPT Vascular cells (from OOD) which get 'Vascular' label.
     Extra-cortical cells keep their depth-bin layer (no override).
  5. Save updated h5ad with new columns

Requires: Step 04 (hybrid_qc_pass) and Step 05 (depth predictions).

Usage:
    python3 -u 06_run_spatial_domains.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import anndata as ad
from multiprocessing import Pool, current_process

# Pipeline config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_config import H5AD_DIR, OUTPUT_DIR, DEPTH_MODEL_PATH, MODULES_DIR, N_WORKERS

# Modules
sys.path.insert(0, MODULES_DIR)
from depth_model import load_model, assign_discrete_layers, LAYER_BINS
from spatial_domains import classify_spatial_domains

# Globals
_model_bundle = None


def _init_worker():
    """Load depth model once per worker (for subclass_names)."""
    global _model_bundle
    pid = current_process().pid
    print(f"  [Worker {pid}] Loading depth model...")
    _model_bundle = load_model(DEPTH_MODEL_PATH)
    print(f"  [Worker {pid}] Ready.")


def _process_one_sample(h5ad_path):
    """Process one sample: spatial domain classification + layer assignment."""
    global _model_bundle
    pid = current_process().pid
    t0 = time.time()

    sample_id = os.path.basename(h5ad_path).replace("_annotated.h5ad", "")

    try:
        adata = ad.read_h5ad(h5ad_path)
        n_total = adata.shape[0]

        # Prefer hybrid_qc_pass (from step 04) to exclude confirmed doublets;
        # fall back to qc_pass if step 04 not run
        if "hybrid_qc_pass" in adata.obs.columns:
            qc_mask = adata.obs["hybrid_qc_pass"].values.astype(bool)
        elif "qc_pass" in adata.obs.columns:
            qc_mask = adata.obs["qc_pass"].values.astype(bool)
        else:
            qc_mask = np.ones(n_total, dtype=bool)

        n_pass = qc_mask.sum()

        # Work on QC-pass cells
        adata_pass = adata[qc_mask].copy()

        # Run spatial domain classification
        # Use correlation-derived subclass labels if available (from step 02b)
        subclass_col = ('corr_subclass' if 'corr_subclass' in adata_pass.obs.columns
                        else 'subclass_label')
        subclass_names = _model_bundle['subclass_names']
        domain_labels, cluster_stats = classify_spatial_domains(
            adata_pass, subclass_names, K=50, resolution=0.8,
            subclass_col=subclass_col
        )

        n_extra = (domain_labels == 'Extra-cortical').sum()
        n_vasc = (domain_labels == 'Vascular').sum()
        n_cort = (domain_labels == 'Cortical').sum()

        # Assign layers: depth bins for all cells, only Vascular overridden from OOD
        pred_depth = adata_pass.obs['predicted_norm_depth'].values
        depth_layers = assign_discrete_layers(pred_depth)

        # Hybrid: depth bins for everything, only override Vascular from OOD
        # Extra-cortical cells KEEP their depth-bin layer (e.g. L1, L2/3)
        combined_layers = depth_layers.copy()
        combined_layers[domain_labels == 'Vascular'] = 'Vascular'

        # Store results back into the full adata (including QC-fail cells)
        # Domain labels
        full_domain = np.full(n_total, 'Unassigned', dtype=object)
        full_domain[qc_mask] = domain_labels
        adata.obs['spatial_domain'] = full_domain

        # Combined layer
        full_layer = np.full(n_total, 'Unassigned', dtype=object)
        full_layer[qc_mask] = combined_layers
        adata.obs['layer'] = full_layer

        # Also keep the old depth-only layer for comparison
        full_depth_layer = np.full(n_total, 'Unassigned', dtype=object)
        full_depth_layer[qc_mask] = depth_layers
        adata.obs['layer_depth_only'] = full_depth_layer

        # Save
        adata.write_h5ad(h5ad_path)

        elapsed = time.time() - t0
        print(f"  [{pid}] {sample_id}: {n_pass:,} pass -> "
              f"Extra-cortical:{n_extra:,}({100*n_extra/n_pass:.1f}%) "
              f"Vascular:{n_vasc:,}({100*n_vasc/n_pass:.1f}%) "
              f"Cortical:{n_cort:,}({100*n_cort/n_pass:.1f}%) "
              f"[{elapsed:.0f}s]")

        # Layer distribution
        layer_dist = {}
        for lname in list(LAYER_BINS.keys()) + ['Vascular']:
            layer_dist[lname] = int((combined_layers == lname).sum())

        return {
            "sample_id": sample_id, "status": "success",
            "n_total": n_total, "n_pass": int(n_pass),
            "n_extra": int(n_extra), "n_vasc": int(n_vasc),
            "n_cort": int(n_cort),
            "layer_dist": layer_dist,
            "cluster_stats": {cl: {k: v for k, v in s.items() if k != 'top3'}
                              for cl, s in cluster_stats.items()},
            "time": elapsed,
        }

    except Exception as e:
        import traceback
        elapsed = time.time() - t0
        print(f"  [{pid}] {sample_id}: FAILED - {e}")
        traceback.print_exc()
        return {
            "sample_id": sample_id, "status": "error",
            "error": str(e), "time": elapsed,
        }


def main():
    t_start = time.time()

    # Find all h5ad files
    h5ad_files = sorted(
        os.path.join(H5AD_DIR, f)
        for f in os.listdir(H5AD_DIR)
        if f.endswith("_annotated.h5ad")
    )
    print(f"Found {len(h5ad_files)} samples")
    print(f"Using {N_WORKERS} workers")
    print(f"Pipeline: spatial domain clustering (K=50, res=0.8) + depth layers")
    print(f"{'='*60}")

    # Process in parallel
    with Pool(processes=N_WORKERS, initializer=_init_worker) as pool:
        results = pool.map(_process_one_sample, h5ad_files)

    # ── Summary ────────────────────────────────────────────────────
    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"SUMMARY - {total_time:.0f}s total")
    print(f"{'='*60}")

    n_ok = sum(1 for r in results if r["status"] == "success")
    print(f"  Success: {n_ok}/{len(results)}")

    # Aggregate stats
    total_pass = sum(r.get("n_pass", 0) for r in results if r["status"] == "success")
    total_extra = sum(r.get("n_extra", 0) for r in results if r["status"] == "success")
    total_vasc = sum(r.get("n_vasc", 0) for r in results if r["status"] == "success")
    total_cort = sum(r.get("n_cort", 0) for r in results if r["status"] == "success")

    print(f"\n  Total QC-pass cells: {total_pass:,}")
    print(f"  Extra-cortical: {total_extra:,} ({100*total_extra/total_pass:.1f}%)")
    print(f"  Vascular:       {total_vasc:,} ({100*total_vasc/total_pass:.1f}%)")
    print(f"  Cortical:       {total_cort:,} ({100*total_cort/total_pass:.1f}%)")

    # Aggregate layer distribution
    agg_layers = {}
    for r in results:
        if r["status"] == "success":
            for lname, n in r["layer_dist"].items():
                agg_layers[lname] = agg_layers.get(lname, 0) + n

    print(f"\n  Combined layer distribution:")
    for lname in ['L1', 'L2/3', 'L4', 'L5', 'L6', 'WM', 'Vascular']:
        n = agg_layers.get(lname, 0)
        print(f"    {lname:20s}: {n:>8,} ({100*n/total_pass:5.1f}%)")

    # Per-sample table
    print(f"\n  Per-sample breakdown:")
    print(f"  {'Sample':>10s} {'QC-pass':>8s} {'Extra%':>6s} {'Vasc%':>6s} {'Time':>5s}")
    for r in sorted(results, key=lambda x: x['sample_id']):
        if r["status"] == "success":
            n = r["n_pass"]
            print(f"  {r['sample_id']:>10s} {n:>8,} "
                  f"{100*r['n_extra']/n:>5.1f}% "
                  f"{100*r['n_vasc']/n:>5.1f}% "
                  f"{r['time']:>4.0f}s")
        else:
            print(f"  {r['sample_id']:>10s} FAILED: {r.get('error', '?')}")

    # Save summary CSV
    summary_rows = []
    for r in results:
        if r["status"] == "success":
            row = {
                'sample_id': r['sample_id'],
                'n_pass': r['n_pass'],
                'n_extra_cortical': r['n_extra'],
                'n_vascular': r['n_vasc'],
                'n_cortical': r['n_cort'],
                'pct_extra_cortical': 100 * r['n_extra'] / r['n_pass'],
                'pct_vascular': 100 * r['n_vasc'] / r['n_pass'],
            }
            row.update(r['layer_dist'])
            summary_rows.append(row)

    df = pd.DataFrame(summary_rows)
    csv_path = os.path.join(OUTPUT_DIR, "spatial_domain_summary.csv")
    df.to_csv(csv_path, index=False)
    print(f"\n  Saved: {csv_path}")

    # ── Merge into combined h5ad ───────────────────────────────────
    print("\nMerging all samples into combined h5ad...")
    t_merge = time.time()
    adatas = []
    for r in results:
        if r["status"] == "success":
            path = os.path.join(H5AD_DIR, f"{r['sample_id']}_annotated.h5ad")
            adatas.append(ad.read_h5ad(path))
    combined = ad.concat(adatas, join='outer')
    combined_path = os.path.join(OUTPUT_DIR, "all_samples_annotated.h5ad")
    combined.write_h5ad(combined_path)
    file_size = os.path.getsize(combined_path) / 1e9
    print(f"  Saved: {combined_path} ({file_size:.2f} GB, {combined.shape})")
    print(f"  Merge took {time.time()-t_merge:.0f}s")

    print(f"\nTotal time: {total_time:.0f}s")


if __name__ == "__main__":
    main()
