"""
Cortical depth prediction from local cell type neighborhood composition.

Trains a GradientBoostingRegressor on SEA-AD MERFISH data to predict
normalized depth from pia (0 = pial surface, 1 = white matter boundary)
using K-nearest-neighbor subclass composition features.

The model can then be applied to Xenium data to assign a predicted
cortical depth to every cell, enabling depth-stratified analyses that
control for variable tissue sampling geometry across sections.

Key insight: predictions are NOT clamped to [0,1] — cells in white matter
may receive depth > 1 and cells above pia may receive depth < 0.

OOD scoring: cells whose local neighborhood composition is far from any
MERFISH training neighborhood are flagged as "Extra-cortical" (e.g.,
pia, meninges, dura mater) since these tissue types were not sampled
by the SEA-AD MERFISH reference.
"""

import numpy as np
import pickle
import time
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import r2_score, mean_absolute_error
from scipy.stats import pearsonr


# Default depth strata for downstream analyses
DEPTH_STRATA = {
    'L2/3': (0.10, 0.30),
    'L4': (0.30, 0.45),
    'L5': (0.45, 0.65),
    'L6': (0.65, 0.85),
}

# Discrete layer bins (including L1, WM, and Extra-cortical)
LAYER_BINS = {
    'L1': (-np.inf, 0.10),
    'L2/3': (0.10, 0.30),
    'L4': (0.30, 0.45),
    'L5': (0.45, 0.65),
    'L6': (0.65, 0.85),
    'WM': (0.85, np.inf),
}

LAYER_COLORS = {
    'L1': (0.9, 0.3, 0.3),
    'L2/3': (0.3, 0.8, 0.3),
    'L4': (0.3, 0.3, 0.9),
    'L5': (0.9, 0.6, 0.1),
    'L6': (0.7, 0.3, 0.8),
    'WM': (0.5, 0.5, 0.5),
    'Extra-cortical': (0.8, 0.8, 0.2),
}


def _vectorized_neighbor_fractions(nn_idx_neighbors, sub_idx_all, n_sub):
    """
    Vectorized computation of neighbor subclass fractions.

    Parameters
    ----------
    nn_idx_neighbors : np.ndarray (n_cells, K)
        Indices of K nearest neighbors for each cell (self excluded).
    sub_idx_all : np.ndarray (n_total,)
        Integer subclass index for each cell (-1 for unknown).
    n_sub : int
        Number of subclass categories.

    Returns
    -------
    np.ndarray (n_cells, n_sub)
        Fraction of each subclass among neighbors.
    """
    n_cells, K = nn_idx_neighbors.shape
    # Look up subclass index for every neighbor
    neighbor_types = sub_idx_all[nn_idx_neighbors]  # (n_cells, K)
    # One-hot encode and sum across neighbors
    # Use np.eye trick: create indicator matrix then sum
    valid = neighbor_types >= 0  # mask out unknown types
    neighbor_types_safe = np.where(valid, neighbor_types, 0)

    # Vectorized bincount per row using broadcasting
    fractions = np.zeros((n_cells, n_sub), dtype=np.float32)
    for k in range(K):
        col = neighbor_types_safe[:, k]
        mask = valid[:, k]
        np.add.at(fractions, (np.arange(n_cells)[mask], col[mask]), 1)

    # Normalize by actual neighbor count
    n_valid = valid.sum(axis=1, keepdims=True).astype(np.float32)
    n_valid = np.maximum(n_valid, 1)  # avoid division by zero
    fractions /= n_valid
    return fractions


def build_neighborhood_features(coords, subclass_labels, subclass_names,
                                 K=50, sections=None):
    """
    Build K-nearest-neighbor subclass composition features.

    For each cell, finds its K spatial nearest neighbors (within the same
    section if sections is provided), computes the fraction of each subclass
    among neighbors, and adds a one-hot encoding of the cell's own type.

    Parameters
    ----------
    coords : np.ndarray
        Spatial coordinates (n_cells x 2).
    subclass_labels : np.ndarray of str
        Subclass label per cell.
    subclass_names : list of str
        Ordered list of all possible subclass names.
    K : int
        Number of nearest neighbors.
    sections : np.ndarray of str, optional
        Section ID per cell. If provided, neighbors are found within
        each section separately (required for MERFISH multi-section data).

    Returns
    -------
    np.ndarray
        Feature matrix (n_cells x 2*n_subclass).
        First n_sub columns: neighbor fractions.
        Last n_sub columns: own-type one-hot.
    """
    n_cells = len(subclass_labels)
    n_sub = len(subclass_names)
    sub_to_idx = {s: i for i, s in enumerate(subclass_names)}

    sub_idx = np.array([sub_to_idx.get(s, -1) for s in subclass_labels])
    features = np.zeros((n_cells, n_sub * 2), dtype=np.float32)

    if sections is not None:
        # Build per-section
        unique_sections = np.unique(sections)
        for sec_i, sec in enumerate(unique_sections):
            if sec_i % 20 == 0:
                print(f"  Section {sec_i+1}/{len(unique_sections)}")
            sec_mask = sections == sec
            sec_indices = np.where(sec_mask)[0]
            sec_coords = coords[sec_indices]

            k_actual = min(K + 1, len(sec_indices))
            nn = NearestNeighbors(n_neighbors=k_actual, algorithm='ball_tree')
            nn.fit(sec_coords)
            _, nn_idx_local = nn.kneighbors(sec_coords)

            # Map local indices to global
            nn_neighbors_local = nn_idx_local[:, 1:K+1]  # exclude self
            nn_neighbors_global = sec_indices[nn_neighbors_local]

            # Vectorized neighbor fractions
            fracs = _vectorized_neighbor_fractions(
                nn_neighbors_global, sub_idx, n_sub
            )
            features[sec_indices, :n_sub] = fracs

            # Own type one-hot (vectorized)
            own_types = sub_idx[sec_indices]
            valid_own = own_types >= 0
            features[sec_indices[valid_own], n_sub + own_types[valid_own]] = 1
    else:
        # Single section / sample
        nn = NearestNeighbors(n_neighbors=K+1, algorithm='ball_tree')
        nn.fit(coords)
        _, nn_idx = nn.kneighbors(coords)

        nn_neighbors = nn_idx[:, 1:]  # exclude self
        fracs = _vectorized_neighbor_fractions(nn_neighbors, sub_idx, n_sub)
        features[:, :n_sub] = fracs

        # Own type one-hot (vectorized)
        valid_own = sub_idx >= 0
        features[np.where(valid_own)[0], n_sub + sub_idx[valid_own]] = 1

    return features


def train_depth_model(merfish_adata, test_donors=None, K=50,
                      n_estimators=300, max_depth=5, learning_rate=0.1):
    """
    Train a depth prediction model on MERFISH data.

    Uses cells with 'Normalized depth from pia' annotations as training
    targets, with K-nearest-neighbor subclass composition as features.

    Parameters
    ----------
    merfish_adata : anndata.AnnData
        SEA-AD MERFISH dataset with 'Subclass', 'Normalized depth from pia',
        'Section', 'Donor ID' in .obs, and spatial coordinates in
        .obsm['X_spatial_raw'].
    test_donors : list of str, optional
        Donor IDs to hold out for testing. If None, uses the 3 donors
        with the fewest depth-annotated cells.
    K : int
        Number of nearest neighbors for features.
    n_estimators : int
        Number of boosting rounds.
    max_depth : int
        Max tree depth.
    learning_rate : float
        Boosting learning rate.

    Returns
    -------
    dict
        Model bundle containing: 'model', 'subclass_names', 'sub_to_idx',
        'K', 'n_sub', 'feature_names', 'train_donors', 'test_donors',
        'train_r2', 'test_r2', 'train_mae', 'test_mae', 'target'.
    """
    print("Preparing MERFISH depth model training data...")

    # Get depth annotations
    depth = merfish_adata.obs['Normalized depth from pia'].values.astype(float)
    has_depth = ~np.isnan(depth)
    print(f"  Cells with depth annotation: {has_depth.sum():,} / {len(depth):,}")

    # Subclass info
    subclass = merfish_adata.obs['Subclass'].values.astype(str)
    subclass_names = sorted(set(subclass))
    n_sub = len(subclass_names)
    sub_to_idx = {s: i for i, s in enumerate(subclass_names)}

    # Donor info
    donor_col = 'Donor ID' if 'Donor ID' in merfish_adata.obs.columns else 'Specimen Barcode'
    donors = merfish_adata.obs[donor_col].values.astype(str)
    sections = merfish_adata.obs['Section'].values.astype(str)

    # Select test donors
    if test_donors is None:
        donor_depth_counts = {}
        for d in np.unique(donors[has_depth]):
            donor_depth_counts[d] = (donors[has_depth] == d).sum()
        all_donors = sorted(donor_depth_counts.keys(),
                            key=lambda d: -donor_depth_counts[d])
        test_donors = all_donors[-3:]

    train_donors = sorted(set(np.unique(donors[has_depth])) - set(test_donors))
    print(f"  Train donors: {len(train_donors)}, Test donors: {len(test_donors)}")

    # Build features for depth-annotated cells
    print(f"  Building K={K} neighborhood features...")
    t0 = time.time()
    coords = merfish_adata.obsm['X_spatial_raw']

    features = build_neighborhood_features(
        coords[has_depth], subclass[has_depth], subclass_names,
        K=K, sections=sections[has_depth]
    )
    depths = depth[has_depth]
    feat_donors = donors[has_depth]
    print(f"  Feature building: {time.time()-t0:.0f}s")

    # Train/test split
    train_mask = np.isin(feat_donors, train_donors)
    test_mask = np.isin(feat_donors, test_donors)
    X_train, y_train = features[train_mask], depths[train_mask]
    X_test, y_test = features[test_mask], depths[test_mask]
    print(f"  Train: {X_train.shape[0]:,}, Test: {X_test.shape[0]:,}")

    # Train
    print(f"  Training GBR (n_estimators={n_estimators})...")
    t1 = time.time()
    model = GradientBoostingRegressor(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=learning_rate, subsample=0.8,
        random_state=42, min_samples_leaf=20
    )
    model.fit(X_train, y_train)
    print(f"  Training: {time.time()-t1:.0f}s")

    # Evaluate (no clamping)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    metrics = {
        'train_r2': r2_score(y_train, y_pred_train),
        'test_r2': r2_score(y_test, y_pred_test),
        'train_mae': mean_absolute_error(y_train, y_pred_train),
        'test_mae': mean_absolute_error(y_test, y_pred_test),
        'train_r': pearsonr(y_train, y_pred_train)[0],
        'test_r': pearsonr(y_test, y_pred_test)[0],
    }
    print(f"\n  Train: R²={metrics['train_r2']:.3f}, MAE={metrics['train_mae']:.4f}")
    print(f"  Test:  R²={metrics['test_r2']:.3f}, MAE={metrics['test_mae']:.4f}")

    feature_names = ([f'neigh_{s}' for s in subclass_names] +
                     [f'own_{s}' for s in subclass_names])

    # Fit 1-NN model on training features for OOD scoring
    # Use only the neighborhood composition features (first n_sub columns),
    # not the own-type one-hot, since we want to assess whether the local
    # *neighborhood* is represented in the reference
    print("  Fitting 1-NN model on training neighborhood features for OOD scoring...")
    t_ood = time.time()
    X_train_neigh = X_train[:, :n_sub]  # neighborhood fractions only
    ood_nn = NearestNeighbors(n_neighbors=1, algorithm='ball_tree', metric='euclidean')
    ood_nn.fit(X_train_neigh)

    # Calibrate OOD threshold using held-out TEST set distances
    # Test cells are in-distribution (cortical tissue from MERFISH) but were
    # not in the BallTree, so their distances represent genuine
    # "new cell → nearest training cell" distances
    X_test_neigh = X_test[:, :n_sub]
    test_dists, _ = ood_nn.kneighbors(X_test_neigh)
    test_dists = test_dists.ravel()
    ood_threshold_99 = float(np.percentile(test_dists, 99))
    ood_threshold_95 = float(np.percentile(test_dists, 95))
    print(f"  OOD 1-NN fitting + calibration: {time.time()-t_ood:.0f}s")
    print(f"  Test-set distance stats: median={np.median(test_dists):.4f}, "
          f"mean={np.mean(test_dists):.4f}, "
          f"95th={ood_threshold_95:.4f}, 99th={ood_threshold_99:.4f}, "
          f"max={np.max(test_dists):.4f}")

    return {
        'model': model,
        'subclass_names': subclass_names,
        'sub_to_idx': sub_to_idx,
        'K': K,
        'n_sub': n_sub,
        'feature_names': feature_names,
        'target': 'normalized_depth_from_pia',
        'train_donors': train_donors,
        'test_donors': test_donors,
        'ood_nn': ood_nn,
        'ood_threshold_99': ood_threshold_99,
        'ood_threshold_95': ood_threshold_95,
        'ood_test_dist_median': float(np.median(test_dists)),
        **metrics,
    }


def save_model(model_bundle, path):
    """Save model bundle to pickle file."""
    with open(path, 'wb') as f:
        pickle.dump(model_bundle, f)
    print(f"Saved model: {path}")


def load_model(path):
    """Load model bundle from pickle file."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def compute_ood_scores(features, model_bundle):
    """
    Compute out-of-distribution scores for query cells.

    For each cell, measures the Euclidean distance from its neighborhood
    composition feature vector to the nearest MERFISH training point.
    High distance = neighborhood not well represented in the reference
    (e.g., meninges, pia, dura mater).

    Parameters
    ----------
    features : np.ndarray
        Feature matrix from build_neighborhood_features() (n_cells x 2*n_sub).
    model_bundle : dict
        Trained model bundle containing 'ood_nn' and 'n_sub'.

    Returns
    -------
    np.ndarray
        OOD distance per cell (float). Higher = more out-of-distribution.
    """
    ood_nn = model_bundle['ood_nn']
    n_sub = model_bundle['n_sub']

    # Use only neighborhood composition features (first n_sub columns)
    X_neigh = features[:, :n_sub]
    dists, _ = ood_nn.kneighbors(X_neigh)
    return dists.ravel()


def predict_depth(adata, model_bundle, subclass_col='subclass_label',
                  compute_ood=True):
    """
    Predict normalized cortical depth for all cells in a Xenium sample.

    Builds K-nearest-neighbor features from spatial coordinates and
    subclass labels, then applies the trained model. Predictions are
    NOT clamped to [0,1].

    If compute_ood=True and the model bundle contains an OOD model,
    also computes OOD scores for each cell.

    Parameters
    ----------
    adata : anndata.AnnData
        Annotated Xenium sample with subclass labels in .obs and
        spatial coordinates in .obsm['spatial'].
    model_bundle : dict
        Trained model from train_depth_model() or load_model().
    subclass_col : str
        Column name for subclass labels in adata.obs.
    compute_ood : bool
        Whether to compute OOD scores (default True).

    Returns
    -------
    np.ndarray or tuple of (np.ndarray, np.ndarray)
        If compute_ood=False: predicted depth per cell.
        If compute_ood=True: (predicted_depth, ood_scores).
    """
    model = model_bundle['model']
    subclass_names = model_bundle['subclass_names']
    K = model_bundle['K']

    coords = adata.obsm['spatial']
    subclass = adata.obs[subclass_col].values.astype(str)

    features = build_neighborhood_features(
        coords, subclass, subclass_names, K=K, sections=None
    )

    pred_depth = model.predict(features)

    if compute_ood and 'ood_nn' in model_bundle:
        ood_scores = compute_ood_scores(features, model_bundle)
        return pred_depth, ood_scores
    else:
        return pred_depth


def assign_discrete_layers(depths, layer_bins=None):
    """
    Assign discrete cortical layer labels from continuous depth values.

    Parameters
    ----------
    depths : np.ndarray
        Predicted normalized depth values.
    layer_bins : dict, optional
        {layer_name: (lower, upper)} boundaries. Defaults to LAYER_BINS.

    Returns
    -------
    np.ndarray of str
        Layer label per cell.
    """
    if layer_bins is None:
        layer_bins = LAYER_BINS

    layers = np.full(len(depths), '', dtype=object)
    for lname, (lo, hi) in layer_bins.items():
        mask = (depths >= lo) & (depths < hi)
        layers[mask] = lname
    return layers


def assign_layers_with_ood(depths, ood_scores, ood_threshold=None,
                            model_bundle=None, layer_bins=None):
    """
    Assign discrete cortical layers, marking OOD cells as 'Extra-cortical'.

    Cells whose neighborhood is not well-represented in the MERFISH
    reference (ood_score > threshold) are assigned to 'Extra-cortical'
    regardless of their predicted depth. In-distribution cells get
    standard layer assignments.

    Parameters
    ----------
    depths : np.ndarray
        Predicted normalized depth values.
    ood_scores : np.ndarray
        OOD distance scores from compute_ood_scores().
    ood_threshold : float, optional
        Cells with OOD score above this are Extra-cortical.
        If None, uses model_bundle['ood_threshold_99'].
    model_bundle : dict, optional
        Model bundle (used to get default threshold).
    layer_bins : dict, optional
        Layer boundaries. Defaults to LAYER_BINS.

    Returns
    -------
    np.ndarray of str
        Layer label per cell, with 'Extra-cortical' for OOD cells.
    """
    if ood_threshold is None:
        if model_bundle is not None and 'ood_threshold_99' in model_bundle:
            ood_threshold = model_bundle['ood_threshold_99']
        else:
            raise ValueError("Must provide ood_threshold or model_bundle "
                             "with 'ood_threshold_99'")

    # Start with standard layer assignment
    layers = assign_discrete_layers(depths, layer_bins)

    # Override OOD cells
    ood_mask = ood_scores > ood_threshold
    layers[ood_mask] = 'Extra-cortical'

    return layers
