"""
Shared constants and utility functions for SCZ Xenium presentation analysis.

This module centralizes all configuration that was previously duplicated across
~20 analysis scripts. Import from here instead of hardcoding constants.

NOTE on LAYER_COLORS: These are the PRESENTATION layer colors, intentionally
different from code/modules/depth_model.py's LAYER_COLORS (which uses different
hues for L5 and L6). Do not "fix" this — the divergence is deliberate.

NOTE on SAMPLE_TO_DX: This is a hardcoded copy of the diagnosis mapping derived
from code/modules/metadata.py::get_diagnosis_map(). Hardcoded here to avoid
needing the Excel metadata file at import time.
"""

import os
import anndata as ad
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

# ──────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────

BASE_DIR = "/Users/shreejoy/Desktop/scz_xenium_test"
H5AD_DIR = os.path.join(BASE_DIR, "output", "h5ad")
MERFISH_PATH = os.path.join(BASE_DIR, "data", "reference",
                             "SEAAD_MTG_MERFISH.2024-12-11.h5ad")
METADATA_PATH = os.path.join(BASE_DIR, "sample_metadata.xlsx")
CRUMBLR_DIR = os.path.join(BASE_DIR, "output", "crumblr")
PRESENTATION_DIR = os.path.join(BASE_DIR, "output", "presentation")

# ──────────────────────────────────────────────────────────────────────
# Sample metadata
# ──────────────────────────────────────────────────────────────────────

SAMPLE_TO_DX = {
    'Br5400': 'Control', 'Br2039': 'SCZ', 'Br2719': 'Control',
    'Br1113': 'Control', 'Br5373': 'SCZ', 'Br5590': 'SCZ',
    'Br6432': 'Control', 'Br5314': 'Control', 'Br5436': 'Control',
    'Br8772': 'SCZ', 'Br8433': 'Control', 'Br5746': 'SCZ',
    'Br5588': 'SCZ', 'Br5973': 'SCZ', 'Br6032': 'SCZ',
    'Br6437': 'SCZ', 'Br5639': 'Control', 'Br6389': 'Control',
    'Br5622': 'Control', 'Br1139': 'SCZ', 'Br2421': 'SCZ',
    'Br5931': 'Control', 'Br6496': 'SCZ', 'Br8667': 'Control',
}

CONTROL_SAMPLES = sorted(k for k, v in SAMPLE_TO_DX.items() if v == "Control")
SCZ_SAMPLES = sorted(k for k, v in SAMPLE_TO_DX.items() if v == "SCZ")
EXCLUDE_SAMPLES = {"Br2039"}  # WM outlier (54% white matter)

# ──────────────────────────────────────────────────────────────────────
# MapMyCells confidence filter
# ──────────────────────────────────────────────────────────────────────
# Bottom-1% filter on subclass bootstrapping probability.
# Removes ~0.8% of cells whose subclass assignment is least reliable.
# Threshold = 1st percentile of subclass_label_confidence across all
# QC-pass cortical cells (computed from full dataset).
SUBCLASS_CONF_THRESH = 0.280

# ──────────────────────────────────────────────────────────────────────
# Presentation colors
# ──────────────────────────────────────────────────────────────────────

BG_COLOR = "#0a0a0a"
DX_COLORS = {"Control": "#4fc3f7", "SCZ": "#ef5350"}

# Presentation layer colors — distinct from depth_model.py's palette
LAYER_COLORS = {
    "L1":       (0.9, 0.3, 0.3),
    "L2/3":     (0.3, 0.8, 0.3),
    "L4":       (0.3, 0.3, 0.9),
    "L5":       (0.9, 0.9, 0.2),
    "L6":       (0.9, 0.5, 0.1),
    "WM":       (0.6, 0.6, 0.6),
    "Vascular": (0.95, 0.3, 0.6),
}
LAYER_ORDER = ["L1", "L2/3", "L4", "L5", "L6", "WM", "Vascular"]
CORTICAL_LAYERS = {"L1", "L2/3", "L4", "L5", "L6"}

# ──────────────────────────────────────────────────────────────────────
# Vulnerable cell type groups
# ──────────────────────────────────────────────────────────────────────

SST_TYPES = ["Sst_2", "Sst_22", "Sst_25", "Sst_20", "Sst_3"]
SST_COLORS = {
    "Sst_2":  "#e41a1c",
    "Sst_22": "#377eb8",
    "Sst_25": "#984ea3",
    "Sst_20": "#ff7f00",
    "Sst_3":  "#ffff33",
}

L6B_TYPES = ["L6b_1", "L6b_2", "L6b_4"]
L6B_COLORS = {
    "L6b_1": "#1f78b4",
    "L6b_2": "#33a02c",
    "L6b_4": "#e31a1c",
}

# ──────────────────────────────────────────────────────────────────────
# Representative samples (median for both SST and L6b, good cortical coverage)
# ──────────────────────────────────────────────────────────────────────

REPRESENTATIVE_SAMPLES = [
    ("Br6389", "Control"),
    ("Br8433", "Control"),
    ("Br6437", "SCZ"),
    ("Br2421", "SCZ"),
]

# ──────────────────────────────────────────────────────────────────────
# Cell class classification
# ──────────────────────────────────────────────────────────────────────

CLASS_COLORS = {
    "Glutamatergic": "#e74c3c",
    "GABAergic":     "#3498db",
    "Non-neuronal":  "#2ecc71",
}

SUBCLASS_TO_CLASS = {
    # Glutamatergic
    "L2/3 IT": "Glutamatergic", "L4 IT": "Glutamatergic",
    "L5 IT": "Glutamatergic", "L5 ET": "Glutamatergic",
    "L5/6 NP": "Glutamatergic", "L6 IT": "Glutamatergic",
    "L6 IT Car3": "Glutamatergic", "L6 CT": "Glutamatergic",
    "L6b": "Glutamatergic",
    # GABAergic
    "Lamp5": "GABAergic", "Lamp5 Lhx6": "GABAergic",
    "Sncg": "GABAergic", "Vip": "GABAergic",
    "Pax6": "GABAergic", "Chandelier": "GABAergic",
    "Pvalb": "GABAergic", "Sst": "GABAergic",
    "Sst Chodl": "GABAergic",
    # Non-neuronal
    "Astrocyte": "Non-neuronal", "Oligodendrocyte": "Non-neuronal",
    "OPC": "Non-neuronal", "Microglia-PVM": "Non-neuronal",
    "Endothelial": "Non-neuronal", "VLMC": "Non-neuronal",
    "SMC": "Non-neuronal", "Pericyte": "Non-neuronal",
}

# Prefix-based inference (for crumblr/snRNAseq scripts)
GABA_PREFIXES = ['Sst', 'Pvalb', 'Vip', 'Lamp5', 'Sncg', 'Pax6', 'Chandelier']
GLUT_PREFIXES = ['L2/3', 'L4', 'L5', 'L6']
NN_PREFIXES = ['Astro', 'Oligo', 'OPC', 'Micro', 'Endo', 'VLMC', 'SMC', 'Pericyte']

# ──────────────────────────────────────────────────────────────────────
# Spatial plot defaults
# ──────────────────────────────────────────────────────────────────────

ALL_CELL_COLOR = "#444444"
MARKER_SIZE_BG = 0.3

# ──────────────────────────────────────────────────────────────────────
# Data loading utilities
# ──────────────────────────────────────────────────────────────────────


def load_cortical(sample_id):
    """Load QC-pass cortical cells (L1-L6) with spatial coordinates.

    Applies bottom-1% subclass confidence filter (SUBCLASS_CONF_THRESH).

    Returns DataFrame with columns: sample_id, subclass_label,
    supertype_label, spatial_domain, layer, qc_pass, x, y.
    """
    fpath = os.path.join(H5AD_DIR, f"{sample_id}_annotated.h5ad")
    adata = ad.read_h5ad(fpath, backed="r")

    cols = ["sample_id", "subclass_label", "supertype_label",
            "spatial_domain", "layer", "qc_pass",
            "subclass_label_confidence"]
    obs = adata.obs[cols].copy()

    coords = adata.obsm["spatial"]
    obs["x"] = coords[:, 0]
    obs["y"] = coords[:, 1]
    obs["layer"] = obs["layer"].astype(str)
    obs["supertype_label"] = obs["supertype_label"].astype(str)

    mask = (obs["qc_pass"] == True) & (obs["layer"].isin(CORTICAL_LAYERS))
    obs = obs[mask]

    # Apply bottom-1% subclass confidence filter
    obs = obs[obs["subclass_label_confidence"].astype(float)
              >= SUBCLASS_CONF_THRESH]

    return obs.copy()


def load_all_cells(sample_id):
    """Load all QC-pass cells with spatial coordinates and layer info.

    Applies bottom-1% subclass confidence filter (SUBCLASS_CONF_THRESH).

    Returns DataFrame with columns: sample_id, subclass_label,
    supertype_label, spatial_domain, layer, qc_pass, x, y.
    """
    fpath = os.path.join(H5AD_DIR, f"{sample_id}_annotated.h5ad")
    adata = ad.read_h5ad(fpath, backed="r")

    cols = ["sample_id", "subclass_label", "supertype_label",
            "spatial_domain", "layer", "qc_pass",
            "subclass_label_confidence"]
    obs = adata.obs[cols].copy()

    coords = adata.obsm["spatial"]
    obs["x"] = coords[:, 0]
    obs["y"] = coords[:, 1]
    obs["layer"] = obs["layer"].astype(str)
    obs["supertype_label"] = obs["supertype_label"].astype(str)
    obs = obs[obs["qc_pass"] == True]

    # Apply bottom-1% subclass confidence filter
    obs = obs[obs["subclass_label_confidence"].astype(float)
              >= SUBCLASS_CONF_THRESH]

    return obs.copy()


# ──────────────────────────────────────────────────────────────────────
# Spatial visualization helpers
# ──────────────────────────────────────────────────────────────────────


def style_dark_axis(ax):
    """Apply standard dark presentation styling to a spatial axis."""
    ax.set_facecolor(BG_COLOR)
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_inset(ax_main, df, loc="lower right"):
    """Draw a small full-tissue inset with layer-colored cells.

    Parameters
    ----------
    ax_main : matplotlib.axes.Axes
        The main axis to attach the inset to.
    df : DataFrame
        All QC-pass cells (from load_all_cells), must have x, y, layer columns.
    loc : str
        Location for the inset (e.g., "lower right", "lower left").
    """
    ax_inset = inset_axes(ax_main, width="30%", height="30%", loc=loc,
                           borderpad=0.5)
    ax_inset.set_facecolor("#111111")

    # Non-cortical cells dim
    non_cortical = df[~df["layer"].isin(CORTICAL_LAYERS)]
    if len(non_cortical) > 0:
        ax_inset.scatter(non_cortical["x"], non_cortical["y"],
                         s=0.05, c="#333333", alpha=0.3,
                         rasterized=True, linewidths=0)

    # Cortical cells colored by layer
    for layer in ["L1", "L2/3", "L4", "L5", "L6"]:
        layer_cells = df[df["layer"] == layer]
        if len(layer_cells) > 0:
            ax_inset.scatter(layer_cells["x"], layer_cells["y"],
                             s=0.08, c=[LAYER_COLORS[layer]], alpha=0.5,
                             rasterized=True, linewidths=0)

    ax_inset.set_aspect("equal")
    ax_inset.invert_yaxis()
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for spine in ax_inset.spines.values():
        spine.set_color("#555555")
        spine.set_linewidth(0.8)

    ax_inset.text(0.5, 0.02, "full section", transform=ax_inset.transAxes,
                  ha="center", va="bottom", fontsize=7, color="#aaaaaa",
                  fontstyle="italic")


def draw_layer_shading(ax, df, alpha=0.12):
    """Draw very faint layer-colored scatter for cortical context.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis to draw on.
    df : DataFrame
        Cells to shade (should include WM for context). Must have x, y, layer.
    alpha : float
        Transparency for the shading dots (default 0.12).
    """
    for layer in LAYER_ORDER:
        layer_cells = df[df["layer"] == layer]
        if len(layer_cells) < 10:
            continue
        color = LAYER_COLORS[layer]
        ax.scatter(layer_cells["x"], layer_cells["y"],
                   s=0.8, c=[color], alpha=alpha,
                   rasterized=True, linewidths=0, zorder=1)


# ──────────────────────────────────────────────────────────────────────
# Statistics helpers
# ──────────────────────────────────────────────────────────────────────


def format_pval(p):
    """Format p-value for display on figures."""
    if p < 0.001:
        return f"p = {p:.1e}"
    elif p < 0.01:
        return f"p = {p:.3f}"
    elif p < 0.1:
        return f"p = {p:.2f}"
    else:
        return f"p = {p:.2f}"


# ──────────────────────────────────────────────────────────────────────
# Cell type classification
# ──────────────────────────────────────────────────────────────────────


def infer_class(celltype):
    """Infer broad cell class from celltype name using prefix matching.

    Returns 'GABA', 'Glut', 'NN', or 'Other'.
    """
    for p in GABA_PREFIXES:
        if celltype.startswith(p):
            return 'GABA'
    for p in GLUT_PREFIXES:
        if celltype.startswith(p):
            return 'Glut'
    for p in NN_PREFIXES:
        if celltype.startswith(p):
            return 'NN'
    return 'Other'


def classify_celltype(ct):
    """Return (hex_color, class_name) for a cell type name.

    Uses case-insensitive substring matching for robustness with
    both subclass and supertype names.
    """
    ct_lower = ct.lower()
    glut_markers = ["l2/3", "l4 ", "l4_", "l5 ", "l5_", "l5/6", "l6 ", "l6_",
                    "l6b", "it_", " it"]
    gaba_markers = ["sst", "pvalb", "vip", "lamp5", "sncg", "chandelier", "pax6"]
    if any(g in ct_lower for g in glut_markers):
        return "#3399dd", "Glutamatergic"
    elif any(g in ct_lower for g in gaba_markers):
        return "#ee4433", "GABAergic"
    else:
        return "#44bb44", "Non-neuronal"
