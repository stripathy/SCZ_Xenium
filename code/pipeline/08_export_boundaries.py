#!/usr/bin/env python3
"""
Step 8: Export cell boundary polygons for the viewer.

Reads the raw cell_boundaries.csv.gz files, filters to QC-pass cells,
and writes compact JSON files that the browser viewer can load on demand
when zoomed in.

Each cell has exactly 25 boundary vertices (Xenium default).
Vertices are quantized to uint16 for compact storage.

Output:
  - output/viewer/boundaries/<sample_id>.json  (one per sample)
"""

import os
import sys
import csv
import gzip
import json
import time
import glob
import numpy as np
import anndata as ad

# Pipeline config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_config import RAW_DIR, H5AD_DIR, VIEWER_DIR

OUTPUT_DIR = os.path.join(VIEWER_DIR, "boundaries")
VERTS_PER_CELL = 25
QUANT_RESOLUTION = 0.2  # micrometers per integer step


def process_sample(sample_id, raw_dir, h5ad_dir, output_dir):
    """Export boundary polygons for one sample."""

    # Find boundary file
    boundary_files = glob.glob(
        os.path.join(raw_dir, f"*{sample_id}-cell_boundaries.csv.gz"))
    if not boundary_files:
        print(f"  {sample_id}: no boundary file found, skipping")
        return None

    boundary_path = boundary_files[0]

    # Read h5ad to get QC-pass cell IDs in order
    h5ad_path = os.path.join(h5ad_dir, f"{sample_id}_annotated.h5ad")
    if not os.path.exists(h5ad_path):
        print(f"  {sample_id}: no h5ad file found, skipping")
        return None

    adata = ad.read_h5ad(h5ad_path)
    if "qc_pass" in adata.obs.columns:
        qc_mask = adata.obs["qc_pass"].values.astype(bool)
        qc_cell_ids = adata.obs_names[qc_mask].tolist()
    else:
        qc_cell_ids = adata.obs_names.tolist()

    qc_set = set(qc_cell_ids)
    n_qc = len(qc_cell_ids)

    # Read boundary CSV
    cells = {}
    with gzip.open(boundary_path, 'rt') as f:
        reader = csv.DictReader(f)
        for row in reader:
            cid = row['cell_id']
            if cid not in qc_set:
                continue
            if cid not in cells:
                cells[cid] = []
            cells[cid].append((float(row['vertex_x']), float(row['vertex_y'])))

    # Verify coverage
    matched = sum(1 for cid in qc_cell_ids if cid in cells)

    # Compute global coordinate range
    all_x = [v[0] for vs in cells.values() for v in vs]
    all_y = [v[1] for vs in cells.values() for v in vs]
    x_min = min(all_x)
    y_min = min(all_y)
    x_max = max(all_x)
    y_max = max(all_y)

    # Verify uint16 range
    x_range_q = (x_max - x_min) / QUANT_RESOLUTION
    y_range_q = (y_max - y_min) / QUANT_RESOLUTION
    assert x_range_q < 65535, f"X range too large for uint16: {x_range_q}"
    assert y_range_q < 65535, f"Y range too large for uint16: {y_range_q}"

    # Build flat arrays aligned to QC-pass cell order
    # Each cell gets exactly VERTS_PER_CELL vertices
    # If a cell has fewer, pad with last vertex; if more, truncate
    bx = []
    by = []
    n_with_boundary = 0

    for cid in qc_cell_ids:
        if cid in cells:
            verts = cells[cid]
            n_with_boundary += 1
        else:
            # No boundary data — use centroid placeholder (will appear as a dot)
            verts = [(0, 0)]

        # Pad or truncate to exactly VERTS_PER_CELL
        while len(verts) < VERTS_PER_CELL:
            verts.append(verts[-1])
        verts = verts[:VERTS_PER_CELL]

        for vx, vy in verts:
            bx.append(int(round((vx - x_min) / QUANT_RESOLUTION)))
            by.append(int(round((vy - y_min) / QUANT_RESOLUTION)))

    # Write JSON
    data = {
        "sample_id": sample_id,
        "n_cells": n_qc,
        "n_with_boundary": n_with_boundary,
        "verts_per_cell": VERTS_PER_CELL,
        "x_offset": float(x_min),
        "y_offset": float(y_min),
        "x_scale": QUANT_RESOLUTION,
        "y_scale": QUANT_RESOLUTION,
        "bx": bx,
        "by": by,
    }

    fpath = os.path.join(output_dir, f"{sample_id}.json")
    with open(fpath, 'w') as f:
        json.dump(data, f, separators=(',', ':'))

    fsize = os.path.getsize(fpath) / 1024 / 1024
    print(f"  {sample_id}: {n_qc:,} cells, {n_with_boundary:,} with boundaries, "
          f"{fsize:.1f} MB")

    return fsize


def main():
    t0 = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find all annotated h5ad files
    fnames = sorted(f for f in os.listdir(H5AD_DIR) if f.endswith("_annotated.h5ad"))
    sample_ids = [f.replace("_annotated.h5ad", "") for f in fnames]

    print(f"Exporting cell boundaries for {len(sample_ids)} samples")
    print(f"Output: {OUTPUT_DIR}/\n")

    total_size = 0
    n_exported = 0

    for sid in sample_ids:
        fsize = process_sample(sid, RAW_DIR, H5AD_DIR, OUTPUT_DIR)
        if fsize is not None:
            total_size += fsize
            n_exported += 1

    elapsed = time.time() - t0
    print(f"\nDone! {n_exported} samples exported in {elapsed:.1f}s")
    print(f"Total size: {total_size:.1f} MB")
    print(f"Output: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
