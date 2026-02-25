# Cortical Depth and Layer Inference: Methods and Validation

## 1. Overview

Assigning cortical depth and layer identity to spatially resolved cells requires either labor-intensive manual annotation or a computational model that can generalize across tissue sections with variable geometry. We use a neighborhood composition-based depth model trained on the SEA-AD MERFISH reference (which has manual cortical depth annotations) to predict normalized cortical depth for every Xenium cell, then combine this with unsupervised spatial domain classification to assign discrete layer labels and identify non-cortical tissue regions (pia/meninges, vascular clusters).

The pipeline produces three annotations per cell: a continuous depth value (0 = pia, 1 = white matter boundary), a spatial domain label (Cortical, Extra-cortical, or Vascular), and a discrete layer assignment (L1 through WM, or Vascular).

---

## 2. Depth Prediction Model

### Feature construction

Rather than predicting depth from a cell's own gene expression (which would be sensitive to individual cell misclassification), the model uses **local neighborhood composition** as features. For each cell, the K=50 nearest spatial neighbors are identified (ball tree algorithm), and the fraction of each subclass among those neighbors is computed. This produces a feature vector of length 2 x n_subclass: the first half encodes neighbor composition fractions, the second half is a one-hot encoding of the cell's own subclass. The key insight is that neighborhood composition captures local tissue context — a region dominated by L2/3 IT neurons is likely superficial cortex regardless of any individual cell's label — making the model robust to sporadic cell misclassification.

### MERFISH reference training

The model is trained on SEA-AD MERFISH data (cells with manual "Normalized depth from pia" annotations). Training uses a donor-level split: the 3 donors with the fewest depth-annotated cells are held out for testing; the remaining donors form the training set. This ensures the model generalizes across individuals rather than memorizing section-specific patterns.

### Model and performance

The depth model is a `GradientBoostingRegressor` (n_estimators=300, max_depth=5, learning_rate=0.1, subsample=0.8, min_samples_leaf=20). Performance on the held-out MERFISH test donors:

| Metric | Train | Test |
|--------|-------|------|
| R^2 | 0.92 | 0.90 |
| MAE | 0.025 | 0.031 |
| Pearson r | 0.96 | 0.95 |

Predictions are deliberately **not clamped** to [0, 1]. Cells in white matter receive depth > 1 and cells above the pia receive depth < 0, providing natural tissue boundary detection without requiring hard cutoffs.

![Depth model diagnostics](output/depth_model_normalized_diagnostics.png)
*Figure 1. Depth model training and validation. Predicted vs actual normalized depth for train and test (held-out donor) sets, with R^2 and MAE metrics.*

### Depth coordinate system

| Normalized depth | Cortical region |
|-----------------|----------------|
| < 0.00 | Above pia (meninges) |
| 0.00 - 0.10 | Layer 1 |
| 0.10 - 0.30 | Layer 2/3 |
| 0.30 - 0.45 | Layer 4 |
| 0.45 - 0.65 | Layer 5 |
| 0.65 - 0.85 | Layer 6 |
| > 0.85 | White matter |

![Depth comparison: MERFISH vs Xenium](output/presentation/slide_depth_comparison.png)
*Figure 2. Spatial depth maps comparing MERFISH manual annotations and Xenium model predictions. Viridis colormap encodes normalized depth (dark = superficial, bright = deep).*

---

## 3. Spatial Domain Classification

### Motivation

Not all tissue on a Xenium section is cortex. Sections may include pia/meningeal tissue (cell-sparse, dominated by astrocytes and microglia), vascular clusters (concentrated Endothelial and VLMC cells), and white matter. The depth model is trained on cortical tissue from MERFISH, so its predictions for non-cortical regions are unreliable. Spatial domain classification identifies these regions so they can be handled appropriately.

### Method

For each sample, we perform unsupervised clustering on the same K=50 neighborhood composition features used by the depth model:

1. **PCA** on neighbor fraction vectors (20 components)
2. **KNN graph** in PCA space (n_neighbors=30)
3. **Leiden clustering** (resolution=0.8, igraph flavor, 2 iterations)

Each resulting cluster is classified by its cell type composition:

| Domain | Rule |
|--------|------|
| **Vascular** | >80% Endothelial + VLMC |
| **Extra-cortical** | >60% non-neuronal AND mean predicted depth < 0.15 |
| **Cortical** | Everything else |

The Vascular threshold is stringent (80%) because blood vessels are scattered throughout cortex and must be sharply distinguished from nearby cortical tissue. The Extra-cortical rule combines composition (high non-neuronal) with spatial position (shallow depth) to avoid misclassifying deep non-neuronal-rich regions (e.g., white matter oligodendrocyte clusters) as pia.

### Out-of-distribution detection

The depth model also includes a 1-NN reference model that measures each cell's Euclidean distance in neighborhood composition space to the nearest MERFISH training point. Cells whose neighborhoods are far from any training example (above the 99th percentile of held-out test distances) have compositions not represented in the cortical MERFISH reference — typically pia or meningeal tissue. This OOD score provides an independent signal that complements the Leiden-based domain classification.

### Aggregate domain breakdown

Across all 24 samples (1,302,631 QC-pass cells):

| Domain | Cells | % |
|--------|-------|---|
| Cortical | 1,183,158 | 90.8% |
| Extra-cortical | 74,298 | 5.7% |
| Vascular | 45,175 | 3.5% |

Extra-cortical fraction varies by sample (1.6% - 9.7%), reflecting differences in how much pia/meningeal tissue was captured. Vascular fraction ranges from 0.0% to 8.3%.

![Layer composition by sample](output/presentation/slide_layer_stacked_bar.png)
*Figure 3. Per-sample layer composition showing the proportion of cells in each cortical layer, white matter, and vascular domains.*

---

## 4. Layer Assignment

Discrete layer labels are assigned by binning the continuous depth predictions:

| Layer | Depth range |
|-------|------------|
| L1 | < 0.10 |
| L2/3 | 0.10 - 0.30 |
| L4 | 0.30 - 0.45 |
| L5 | 0.45 - 0.65 |
| L6 | 0.65 - 0.85 |
| WM | > 0.85 |

Vascular-domain cells are overridden to "Vascular" regardless of predicted depth, since depth predictions are unreliable for spatially isolated vascular clusters. Extra-cortical cells retain their depth-bin layer (they are typically assigned to L1 or L2/3 given their shallow position).

### Aggregate layer distribution

| Layer | Cells | % |
|-------|-------|---|
| L1 | 51,843 | 4.0% |
| L2/3 | 251,982 | 19.3% |
| L4 | 174,661 | 13.4% |
| L5 | 324,864 | 24.9% |
| L6 | 145,203 | 11.1% |
| WM | 308,903 | 23.7% |
| Vascular | 45,175 | 3.5% |

### Output columns

Each annotated h5ad file receives three new columns:

| Column | Values | Description |
|--------|--------|-------------|
| `predicted_norm_depth` | float | Continuous depth (0 = pia, 1 = WM boundary) |
| `spatial_domain` | Cortical / Extra-cortical / Vascular / Unassigned | Tissue domain |
| `layer` | L1 / L2-3 / L4 / L5 / L6 / WM / Vascular / Unassigned | Discrete layer (depth bins + Vascular override) |

---

## 5. Validation

### Median depth per cell type: Xenium vs MERFISH

The primary validation is whether cell types end up at the expected cortical depths. Comparing median predicted depth per subclass against the MERFISH reference:

| Level | Pearson r | n types |
|-------|-----------|---------|
| Subclass | 0.92 | 24 |

The strong correlation confirms that the depth model recovers biologically meaningful laminar positions. Excitatory neurons show the expected superficial-to-deep ordering: L2/3 IT is shallowest (median depth ~0.24 in Xenium, ~0.26 in MERFISH), progressing through L4 IT, L5 IT, L5 ET, L5/6 NP, L6 IT, L6 CT, to L6b (median depth ~0.80 in both datasets). Non-neuronal types show more variable depths due to their distribution across all layers.

![Median depth by cell type](output/presentation/slide_median_depth_by_celltype.png)
*Figure 4. Median cortical depth per cell type: Xenium predictions vs MERFISH reference. Left: subclass level. Right: supertype level. Strong correlation at both resolution levels confirms the model recovers biologically meaningful depth assignments.*

### Supertype-level depth distributions

At finer resolution, we compare depth distributions for every supertype within each subclass (for subclasses with ≥4 supertypes). Supertypes are ordered by median MERFISH depth, and paired MERFISH (manual depth) and Xenium (predicted depth) violins are shown side by side:

![Supertype depth distributions — Glutamatergic](output/presentation/supertype_depth_violins_glutamatergic.png)
*Figure 5. Supertype depth distributions for glutamatergic subclasses. Green = MERFISH manual depth, orange = Xenium predicted depth. Counts shown as (MERFISH / Xenium). Within each subclass, supertypes are ordered by median MERFISH depth (shallowest left, deepest right).*

![Supertype depth distributions — GABAergic](output/presentation/supertype_depth_violins_gabaergic.png)
*Figure 6. Supertype depth distributions for GABAergic subclasses. Interneuron supertypes show broader depth distributions than excitatory types, consistent with their wider laminar spread, but the overall ordering and distribution shapes are well-matched between MERFISH and Xenium.*

### Key design decisions

1. **Neighborhood-based features** rather than per-cell expression: robust to individual cell misclassification; captures local tissue context
2. **Unclamped predictions**: naturally detects cells outside the cortical column (depth < 0 or > 1) without arbitrary thresholds
3. **Donor-level train/test split**: ensures the model generalizes to new individuals, not just new cells from the same donors
4. **Combined domain classification**: unsupervised clustering captures contiguous pia regions, while composition thresholds identify scattered vascular spots — together they cover both spatially coherent and distributed non-cortical tissue
