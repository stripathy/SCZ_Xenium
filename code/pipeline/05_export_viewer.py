#!/usr/bin/env python3
"""
Step 5: Export viewer JSON files from annotated h5ad files.

Reads per-sample h5ad files, filters to QC-pass cells only, and writes
compact JSON files for the interactive Xenium spatial viewer.

Output:
  - output/viewer/<sample_id>.json  (one per sample)
  - output/viewer/index.json        (global index with sample list + color palettes)
"""

import os
import sys
import json
import time
import numpy as np
import anndata as ad

# Pipeline config
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline_config import H5AD_DIR, VIEWER_DIR, METADATA_PATH, MODULES_DIR

# Modules
sys.path.insert(0, MODULES_DIR)
from metadata import get_diagnosis_map

# ── Layer categories (depth bins + Vascular from OOD) ──
LAYER_CATS = ["L1", "L2/3", "L4", "L5", "L6", "WM", "Vascular"]


# ── Color palettes ──
# Official SEA-AD colors (from MERFISH .uns['Subclass_colors'])
SUBCLASS_COLORS = {
    "L2/3 IT": "#B1EC30",
    "L4 IT": "#00E5E5",
    "L5 IT": "#50B2AD",
    "L5 ET": "#0D5B78",
    "L5/6 NP": "#3E9E64",
    "L6 IT": "#A19922",
    "L6 IT Car3": "#5100FF",
    "L6 CT": "#2D8CB8",
    "L6b": "#7044AA",
    "Sst": "#FF9900",
    "Sst Chodl": "#B1B10C",
    "Pvalb": "#D93137",
    "Vip": "#A45FBF",
    "Lamp5": "#DA808C",
    "Lamp5 Lhx6": "#935F50",
    "Sncg": "#DF70FF",
    "Pax6": "#71238C",
    "Chandelier": "#F641A8",
    "Astrocyte": "#665C47",
    "Oligodendrocyte": "#53776C",
    "OPC": "#374A45",
    "Microglia-PVM": "#94AF97",
    "Endothelial": "#8D6C62",
    "VLMC": "#697255",
}

CLASS_COLORS = {
    "Neuronal: Glutamatergic": "#00ADF8",
    "Neuronal: GABAergic": "#F05A28",
    "Non-neuronal and Non-neural": "#808080",
}

LAYER_COLORS = {
    "L1": "#f1c40f",
    "L2/3": "#e67e22",
    "L4": "#f39c12",
    "L5": "#2ecc71",
    "L6": "#3498db",
    "WM": "#9b59b6",
    "Vascular": "#1abc9c",
}

# Supertype → Subclass mapping for non-neuronal types whose supertype names
# don't match their subclass names (e.g., "Astro_1" → "Astrocyte")
SUPERTYPE_SUBCLASS_MAP = {
    "Astro": "Astrocyte",
    "Oligo": "Oligodendrocyte",
    "Micro-PVM": "Microglia-PVM",
    "Endo": "Endothelial",
    "Pericyte": "VLMC",
    "SMC": "VLMC",
}


def main():
    t0 = time.time()

    os.makedirs(VIEWER_DIR, exist_ok=True)

    # Load diagnosis mapping
    dx_map = get_diagnosis_map(METADATA_PATH)
    print(f"Loaded diagnosis map: {len(dx_map)} subjects")

    # Process each sample
    sample_index = []
    total_cells_all = 0
    total_qc_pass = 0

    fnames = sorted(f for f in os.listdir(H5AD_DIR) if f.endswith("_annotated.h5ad"))
    print(f"Found {len(fnames)} h5ad files\n")

    for fname in fnames:
        sample_id = fname.replace("_annotated.h5ad", "")
        path = os.path.join(H5AD_DIR, fname)

        print(f"Processing {sample_id}...", end=" ", flush=True)
        adata = ad.read_h5ad(path)
        n_total = adata.n_obs

        # Filter to QC-pass cells only
        if "qc_pass" in adata.obs.columns:
            qc_mask = adata.obs["qc_pass"].values.astype(bool)
            n_fail = (~qc_mask).sum()
            adata = adata[qc_mask].copy()
            print(f"({n_total} total, {n_fail} QC-fail removed, "
                  f"{adata.n_obs} kept)", end=" ")
        else:
            print(f"(no qc_pass column, keeping all {n_total})", end=" ")

        total_cells_all += n_total
        total_qc_pass += adata.n_obs

        # Extract data
        x = adata.obsm["spatial"][:, 0].astype(np.float32)
        y = adata.obsm["spatial"][:, 1].astype(np.float32)
        subclass = adata.obs["subclass_label"].values.astype(str)
        supertype = adata.obs["supertype_label"].values.astype(str)
        class_label = adata.obs["class_label"].values.astype(str)
        depth = adata.obs["predicted_norm_depth"].values.astype(np.float32)

        # Round coordinates to save space
        x = np.round(x, 1)
        y = np.round(y, 1)
        depth = np.round(depth, 3)

        # Build compact format: encode categories as integers
        subclass_cats = sorted(set(subclass))
        supertype_cats = sorted(set(supertype))
        class_cats = sorted(set(class_label))

        subclass_idx = {c: i for i, c in enumerate(subclass_cats)}
        supertype_idx = {c: i for i, c in enumerate(supertype_cats)}
        class_idx = {c: i for i, c in enumerate(class_cats)}

        # Layer column (from step 04)
        if "layer" in adata.obs.columns:
            layers = adata.obs["layer"].values.astype(str)
        else:
            # Fallback: assign from depth only
            from depth_model import assign_discrete_layers
            layers = assign_discrete_layers(depth)

        # Ensure all layer values are in LAYER_CATS
        layer_cats = list(LAYER_CATS)  # local copy to avoid mutation
        layer_idx = {c: i for i, c in enumerate(layer_cats)}
        for lbl in set(layers):
            if lbl not in layer_idx:
                layer_cats.append(lbl)
                layer_idx[lbl] = len(layer_cats) - 1

        data = {
            "sample_id": sample_id,
            "diagnosis": dx_map.get(sample_id, "Unknown"),
            "n_cells": len(x),
            "x_range": [float(x.min()), float(x.max())],
            "y_range": [float(y.min()), float(y.max())],
            "subclass_cats": subclass_cats,
            "supertype_cats": supertype_cats,
            "class_cats": class_cats,
            "x": x.tolist(),
            "y": y.tolist(),
            "subclass": [subclass_idx[s] for s in subclass],
            "supertype": [supertype_idx[c] for c in supertype],
            "class": [class_idx[c] for c in class_label],
            "depth": depth.tolist(),
            "layer_cats": layer_cats,
            "layer": [layer_idx[lbl] for lbl in layers],
        }

        # Write compact JSON (no whitespace)
        json_path = os.path.join(VIEWER_DIR, f"{sample_id}.json")
        with open(json_path, "w") as f:
            json.dump(data, f, separators=(",", ":"))

        file_size = os.path.getsize(json_path) / 1024 / 1024
        print(f"-> {file_size:.1f}MB")

        sample_index.append({
            "sample_id": sample_id,
            "diagnosis": dx_map.get(sample_id, "Unknown"),
            "n_cells": len(x),
        })

    # Write index.json
    index_data = {
        "samples": sample_index,
        "subclass_colors": SUBCLASS_COLORS,
        "class_colors": CLASS_COLORS,
        "layer_colors": LAYER_COLORS,
        "supertype_subclass_map": SUPERTYPE_SUBCLASS_MAP,
    }

    index_path = os.path.join(VIEWER_DIR, "index.json")
    with open(index_path, "w") as f:
        json.dump(index_data, f, indent=2)

    # Build standalone HTML viewer
    try:
        from bundle_viewer import bundle_standalone_html
        html_path = os.path.join(VIEWER_DIR, "xenium_viewer_standalone.html")
        bundle_standalone_html(VIEWER_DIR, html_path)
        print(f"\nStandalone HTML viewer: {html_path}")
    except ImportError:
        print("\nNote: bundle_viewer module not available, skipping HTML build")
    except Exception as e:
        print(f"\nHTML viewer build failed: {e}")

    elapsed = time.time() - t0
    print(f"\nDone! Processed {len(fnames)} samples in {elapsed:.1f}s")
    print(f"  Total cells across all samples: {total_cells_all:,}")
    print(f"  QC-pass cells exported: {total_qc_pass:,}")
    print(f"  QC-fail cells removed: {total_cells_all - total_qc_pass:,}")
    print(f"  Output: {VIEWER_DIR}/")


if __name__ == "__main__":
    main()
