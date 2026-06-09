"""
utils_smoothing.py
------------------
Core utilities for t-SNE affinity smoothing experiments.

Contains all functions needed to:
  - Load datasets (mouse cortex, MNIST, Adult)
  - Build conditional affinity matrices with true HD rank order
  - Apply row-wise power smoothing / sharpening
  - Run standard and modified t-SNE via openTSNE
  - Orchestrate perplexity sweeps and gamma sweeps
  - Compute NH, trustworthiness, and continuity metric curves
  - Compute effective perplexity per point
  - Save / load embeddings and metric curves
  - Plot metric curves and perplexity histograms

Functions are collected from main_mouse.ipynb (cell 0), utils_plot.py,
and utils_data.py.  Minimal changes applied: two bug-fixes and removal of
global-constant references inside run_tsne_from_joint_P.
"""

import os
import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.neighbors import NearestNeighbors
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from openTSNE import TSNEEmbedding, initialization, affinity
from scipy.sparse import csr_matrix
from matplotlib.lines import Line2D


# =============================================================================
# Data loading
# =============================================================================

def load_mouse_data(pickle_path, data_dir=None, return_highdim=False):
    """
    Load preprocessed mouse cortex data from a pickle file.

    Parameters
    ----------
    pickle_path : str
        Path to tasic2018.pickle.
    data_dir : str, optional
        Directory containing importantGenesTasic2018.npy.
    return_highdim : bool, optional
        If True, also return the mean-centred log-CPM matrix (n × n_genes)
        BEFORE the SVD step.  This is the correct high-dimensional space for
        neighbourhood evaluation (NH@k, trustworthiness, Spearman).  The PCA-50
        output should only be used as input to t-SNE optimisation.

    Returns
    -------
    X_pca : ndarray, shape (n, 50)
    X_high : ndarray, shape (n, n_genes)  – only when return_highdim=True
    y, labels, colors
    """
    import pickle

    with open(pickle_path, "rb") as f:
        tasic2018 = pickle.load(f)

    # Gene selection mask
    genes_path = None
    if data_dir is not None:
        genes_path = os.path.join(data_dir, "importantGenesTasic2018.npy")

    if genes_path is not None and os.path.exists(genes_path):
        importantGenes = np.load(genes_path)
    else:
        import rnaseqTools
        markerGenes = [
            "Snap25", "Gad1", "Slc17a7", "Pvalb", "Sst", "Vip", "Aqp4",
            "Mog", "Itgam", "Pdgfra", "Flt1", "Bgn", "Rorb", "Foxp2",
        ]
        importantGenes = rnaseqTools.geneSelection(
            tasic2018["counts"], n=3000, threshold=32,
            markers=markerGenes, genes=tasic2018["genes"],
        )
        if genes_path is not None:
            np.save(genes_path, importantGenes)

    counts = tasic2018["counts"]
    librarySizes = np.asarray(counts.sum(axis=1)).ravel()
    counts_subset = counts[:, importantGenes]
    if hasattr(counts_subset, "toarray"):
        counts_subset = counts_subset.toarray()
    X = np.log2(counts_subset / librarySizes[:, None] * 1e6 + 1)

    X = np.array(X)

    # Remove cells that have zero library size (→ Inf/NaN in log-CPM),
    # then remove any remaining NaN/Inf rows that would segfault openTSNE.
    finite_mask = np.isfinite(X).all(axis=1)
    n_bad = (~finite_mask).sum()
    if n_bad > 0:
        import warnings
        warnings.warn(
            f"load_mouse_data: dropping {n_bad} cells with NaN/Inf log-CPM values.",
            RuntimeWarning, stacklevel=2,
        )
        X      = X[finite_mask]
        y_raw  = tasic2018["clusters"][finite_mask]
    else:
        y_raw = tasic2018["clusters"]

    X = X - X.mean(axis=0)
    X_high = X.astype(np.float32)      # mean-centred log-CPM, kept for metrics

    U, s, V = np.linalg.svd(X, full_matrices=False)
    U[:, np.sum(V, axis=1) < 0] *= -1
    X = np.dot(U, np.diag(s))
    X = X[:, np.argsort(s)[::-1]][:, :50]

    y      = y_raw
    labels = tasic2018["clusterNames"][y]
    colors = tasic2018["clusterColors"][y]

    if return_highdim:
        return X, X_high, y, labels, colors
    return X, y, labels, colors


def load_mnist_data(n_pca=50, random_state=42):
    """
    Load MNIST (70 000 × 784), normalise to [0, 1], reduce to `n_pca` dims.

    Returns
    -------
    X_pca : ndarray, shape (70000, n_pca)
    y     : ndarray of str, shape (70000,)
    X_raw : ndarray, shape (70000, 784)  – raw pixel values / 255
    """
    ds = fetch_openml("mnist_784", version=1, as_frame=False)
    X_raw = ds.data.astype(np.float64) / 255.0
    y = ds.target.astype(str)

    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(X_raw)

    return X_pca, y, X_raw


def load_adult_data(max_rows=10000, random_state=42):
    """
    Load and preprocess the Adult (census-income) dataset.

    Returns
    -------
    X : ndarray, shape (max_rows, n_features)
    y : ndarray of int, shape (max_rows,)  – binary income label
    """
    adult = fetch_openml(name="adult", version=2, as_frame=True)
    X_df = adult.data.copy()
    y_raw = adult.target.astype(str).str.strip()
    y = (y_raw == ">50K").astype(int).to_numpy()

    if max_rows is not None and len(X_df) > max_rows:
        rng = np.random.default_rng(random_state)
        sel = np.sort(rng.choice(len(X_df), size=max_rows, replace=False))
        X_df = X_df.iloc[sel].reset_index(drop=True)
        y = y[sel]

    numeric_cols = X_df.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = [c for c in X_df.columns if c not in numeric_cols]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer([
        ("num", num_pipe, numeric_cols),
        ("cat", cat_pipe, categorical_cols),
    ])

    X = pre.fit_transform(X_df).astype(np.float32)
    return X, y


# =============================================================================
# Affinity utilities
# =============================================================================

def row_normalize_csr(csr):
    csr = csr.tocsr(copy=True)
    row_sums = np.asarray(csr.sum(axis=1)).ravel()
    row_sums[row_sums == 0] = 1.0
    inv = 1.0 / row_sums
    return sp.diags(inv) @ csr


def _binary_search_row_beta(dist2, target_perplexity, tol=1e-5, max_iter=50):
    """Binary-search for bandwidth beta so the row perplexity equals
    target_perplexity.  Returns the normalised probability vector.

    When convergence fails — most commonly for points in very dense regions
    where all k_hd neighbours are nearly equidistant, so the Gaussian kernel
    cannot concentrate below log(k_hd) — the distribution stays uniform at
    every beta value, and no amount of iteration helps.  In that case we fall
    back to uniform over the top ceil(target_perplexity) nearest neighbours
    rather than over all k_hd neighbours.  This keeps the effective perplexity
    close to the target (≈ 30 instead of 90) for those problematic points.
    """
    logU = np.log(target_perplexity)
    k = len(dist2)

    beta_min = -np.inf
    beta_max = np.inf
    beta = 1.0
    best_P = None
    best_Hdiff = np.inf  # |H - logU| of the closest solution seen so far

    for _ in range(max_iter):
        P = np.exp(-dist2 * beta)
        sumP = np.sum(P)

        if sumP <= 0:
            # All probabilities underflowed; stop and use best saved solution.
            break

        P_norm = P / sumP
        H = np.log(sumP) + beta * np.sum(dist2 * P) / sumP
        Hdiff = H - logU

        if np.abs(Hdiff) < tol:
            return P_norm  # converged

        if np.abs(Hdiff) < best_Hdiff:
            best_Hdiff = np.abs(Hdiff)
            best_P = P_norm  # closest solution so far

        if Hdiff > 0:
            beta_min = beta
            beta = 2.0 * beta if np.isinf(beta_max) else 0.5 * (beta + beta_max)
        else:
            beta_max = beta
            beta = 0.5 * beta if np.isinf(beta_min) else 0.5 * (beta + beta_min)

    # Did not converge.
    # If the best achieved entropy is still more than log(2) above the target
    # (effective perplexity > 2× target), the distribution is stuck near
    # uniform — equidistant-neighbour case.  Use uniform over the top
    # ceil(target_perplexity) nearest neighbours so the effective perplexity
    # stays close to the target rather than jumping to k_hd.
    if best_P is None or best_Hdiff > np.log(2.0):
        k_eff = max(1, min(int(np.ceil(target_perplexity)), k))
        out = np.zeros(k, dtype=np.float64)
        out[:k_eff] = 1.0 / k_eff
        return out

    return best_P


def build_conditional_affinity_and_knn(X, perplexity=30, k_hd=90,
                                       metric="euclidean", n_jobs=-1):
    """
    Build conditional affinity matrix P_cond and ranked HD neighbours.

    Uses an independent binary-search (not openTSNE's internal one) so that
    the true HD rank order of every point is available for downstream mass
    transforms.

    Returns
    -------
    P_cond      : csr_matrix, shape (n, n)  – row-normalised conditional P
    knn_indices : ndarray, shape (n, k_hd)  – ranked HD neighbour indices
    knn_distances : ndarray, shape (n, k_hd)
    """
    n = X.shape[0]

    nn = NearestNeighbors(n_neighbors=k_hd + 1, metric=metric, n_jobs=n_jobs)
    nn.fit(X)
    distances, indices = nn.kneighbors(X, return_distance=True)

    knn_distances = distances[:, 1:]
    knn_indices = indices[:, 1:]

    indptr = [0]
    all_indices = []
    all_data = []

    for i in range(n):
        dist2 = knn_distances[i] ** 2
        probs = _binary_search_row_beta(dist2, target_perplexity=perplexity)

        all_indices.extend(knn_indices[i].tolist())
        all_data.extend(probs.tolist())
        indptr.append(len(all_indices))

    P_cond = sp.csr_matrix(
        (
            np.asarray(all_data, dtype=np.float64),
            np.asarray(all_indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int64),
        ),
        shape=(n, n),
    )

    P_cond = row_normalize_csr(P_cond)
    return P_cond, knn_indices, knn_distances


def get_ranked_row_values_from_P(P_cond, i, ranked_neighbors):
    """Return row i probabilities from P_cond in the exact order of
    ranked_neighbors.  Neighbours absent from the sparse row (e.g. rows
    truncated by the equidistant fallback in _binary_search_row_beta) are
    returned as 0.0."""
    P_cond = P_cond.tocsr()
    s, e = P_cond.indptr[i], P_cond.indptr[i + 1]
    row_cols = P_cond.indices[s:e]
    row_vals = P_cond.data[s:e]

    row_map = {j: p for j, p in zip(row_cols, row_vals)}
    vals = np.array([row_map.get(j, 0.0) for j in ranked_neighbors], dtype=np.float64)
    return vals


def build_P_from_ranked_neighbors_and_values(knn_indices, row_values, n):
    """Construct a CSR matrix from (knn_indices[i], row_values[i]) pairs,
    then row-normalise."""
    indptr  = [0]
    indices = []
    data    = []

    for i in range(n):
        indices.extend(knn_indices[i].tolist())
        data.extend(row_values[i].tolist())
        indptr.append(len(indices))

    P = sp.csr_matrix(
        (
            np.asarray(data,    dtype=np.float64),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr,  dtype=np.int64),
        ),
        shape=(n, n),
    )
    return row_normalize_csr(P)


def symmetrize_conditional_to_joint(P_cond):
    """Convert row-normalised conditional P into a symmetric joint P that
    sums to 1, following the standard t-SNE symmetrisation."""
    n = P_cond.shape[0]
    P_joint = (P_cond + P_cond.T) * (1.0 / (2.0 * n))
    P_joint = P_joint.tocsr()
    total = P_joint.data.sum()
    if not np.isfinite(total) or total <= 0:
        raise ValueError(
            f"Joint P has non-positive or NaN total mass ({total}). "
            "This usually means NaN/Inf values reached the affinity matrix."
        )
    P_joint.data /= total
    return P_joint


# =============================================================================
# Mass transforms (smoothing / sharpening)
# =============================================================================

def _power_transform_row(vals, gamma, eps=1e-300):
    """
    Apply p -> p^gamma row-wise and renormalise.

    gamma < 1  -> smoother (flatter)
    gamma = 0  -> uniform
    gamma = 1  -> unchanged
    gamma > 1  -> sharper

    Rank order is restored after the transform (monotone non-increasing).
    """
    vals = np.asarray(vals, dtype=np.float64)

    if np.isclose(gamma, 0.0):
        # Special case: uniform distribution
        out = np.ones(len(vals), dtype=np.float64) / len(vals)
        return out

    vals = np.maximum(vals, eps)
    out = vals ** gamma
    out /= out.sum()

    # Restore monotone non-increasing rank order
    out = np.sort(out)[::-1]
    out /= out.sum()
    return out


def perturb_mass_smooth_true_rank_order(P_cond, knn_indices, gamma=0.5):
    """
    Flatten (smooth) the row-wise probability distribution.

    Preserves: same neighbours, same rank order, same support.
    gamma must satisfy  0 <= gamma < 1  (0 gives uniform rows).
    """
    if not (0.0 <= gamma < 1.0):
        raise ValueError("For smoothing, gamma must satisfy 0 <= gamma < 1.")

    n, _ = knn_indices.shape
    new_row_values = []

    for i in range(n):
        ranked_neighbors = knn_indices[i]
        vals    = get_ranked_row_values_from_P(P_cond, i, ranked_neighbors).copy()
        smoothed = _power_transform_row(vals, gamma=gamma)
        new_row_values.append(smoothed)

    new_row_values = np.asarray(new_row_values, dtype=np.float64)
    return build_P_from_ranked_neighbors_and_values(knn_indices, new_row_values, n)


def perturb_mass_sharpen_true_rank_order(P_cond, knn_indices, gamma=2.0):
    """
    Sharpen (concentrate) the row-wise probability distribution.

    Preserves: same neighbours, same rank order, same support.
    gamma must satisfy  gamma > 1.
    """
    if not (gamma > 1.0):
        raise ValueError("For sharpening, gamma must satisfy gamma > 1.")

    n, _ = knn_indices.shape
    new_row_values = []

    for i in range(n):
        ranked_neighbors = knn_indices[i]
        vals      = get_ranked_row_values_from_P(P_cond, i, ranked_neighbors).copy()
        sharpened = _power_transform_row(vals, gamma=gamma)
        new_row_values.append(sharpened)

    new_row_values = np.asarray(new_row_values, dtype=np.float64)
    return build_P_from_ranked_neighbors_and_values(knn_indices, new_row_values, n)


# =============================================================================
# t-SNE runners
# =============================================================================

def run_standard_opentsne_baseline(
    X,
    perplexity=30,
    metric="euclidean",
    n_jobs=-1,
    random_state=42,
    n_iter_ee=250,
    n_iter_main=750,
    ee=12,
    learning_rate="auto",
):
    """
    Run standard openTSNE with its built-in affinity construction.
    No custom P is injected.

    Returns
    -------
    Y   : ndarray, shape (n, 2)
    aff : openTSNE affinity object
    """
    aff = affinity.PerplexityBasedNN(
        X,
        perplexity=perplexity,
        metric=metric,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=False,
    )

    Y0 = initialization.pca(X, random_state=random_state, verbose=False)

    emb = TSNEEmbedding(
        Y0,
        aff,
        negative_gradient_method="fft",
        n_jobs=n_jobs,
        learning_rate=learning_rate,
        random_state=random_state,
        verbose=False,
    )

    emb.optimize(n_iter=n_iter_ee, exaggeration=ee, momentum=0.5, inplace=True)
    emb.optimize(n_iter=n_iter_main, exaggeration=1.0, momentum=0.8, inplace=True)

    return np.asarray(emb), aff


def run_tsne_from_joint_P(
    X,
    P_joint,
    perplexity=30,
    n_jobs=-1,
    random_state=42,
    n_iter_ee=250,
    n_iter_main=750,
    ee=12,
    learning_rate="auto",
):
    """
    Run t-SNE from a custom joint probability matrix P_joint.

    The openTSNE affinity object is created only to satisfy the API; its
    internal P is immediately replaced by P_joint.

    Returns
    -------
    Y : ndarray, shape (n, 2)
    """
    aff = affinity.PerplexityBasedNN(
        X,
        perplexity=perplexity,
        metric="euclidean",
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=False,
    )
    aff.P = P_joint.tocsr()

    Y0 = initialization.pca(X, random_state=random_state, verbose=False)

    emb = TSNEEmbedding(
        Y0,
        aff,
        negative_gradient_method="fft",
        n_jobs=n_jobs,
        learning_rate=learning_rate,
        random_state=random_state,
        verbose=False,
    )

    emb.optimize(n_iter=n_iter_ee, exaggeration=ee, momentum=0.5, inplace=True)
    emb.optimize(n_iter=n_iter_main, exaggeration=1.0, momentum=0.8, inplace=True)

    return np.asarray(emb)


# =============================================================================
# Metrics
# =============================================================================

def topk_neighbors(A, K, metric="euclidean", n_jobs=None, algorithm="auto"):
    """Return the K nearest-neighbour indices for each row of A
    (self excluded)."""
    n = A.shape[0]
    K = min(K, n - 1)
    nn = NearestNeighbors(
        n_neighbors=K + 1, metric=metric, n_jobs=n_jobs, algorithm=algorithm
    )
    nn.fit(A)
    _, idx = nn.kneighbors(A, return_distance=True)
    return idx[:, 1: K + 1]


def compute_neighborhood_hit_curve(
    X_high,
    Y_emb,
    k_values,
    metric_hd="euclidean",
    metric_ld="euclidean",
    n_jobs=None,
    algorithm="auto",
):
    """
    Compute NH@k = mean over points of  |N_k^HD ∩ N_k^LD| / k
    for each k in k_values.

    Returns
    -------
    nh_curve : ndarray, shape (len(k_values),)
    """
    k_values = np.asarray(
        sorted(set(int(k) for k in k_values if k >= 1)), dtype=int
    )
    k_max = int(k_values.max())

    hd_nbrs = topk_neighbors(
        X_high, k_max, metric=metric_hd, n_jobs=n_jobs, algorithm=algorithm
    )
    ld_nbrs = topk_neighbors(
        Y_emb, k_max, metric=metric_ld, n_jobs=n_jobs, algorithm=algorithm
    )

    n = X_high.shape[0]
    nh_curve = []

    for k in k_values:
        overlaps = np.empty(n, dtype=float)
        for i in range(n):
            hd_set = set(hd_nbrs[i, :k])
            ld_set = set(ld_nbrs[i, :k])
            overlaps[i] = len(hd_set.intersection(ld_set)) / k
        nh_curve.append(overlaps.mean())

    return np.asarray(nh_curve)


def compute_trustworthiness_curve(X_high, Y_emb, k_values, metric="euclidean"):
    """
    Compute sklearn trustworthiness for multiple k values.

    Returns
    -------
    curve : ndarray, shape (len(k_values),)
    """
    k_values = np.asarray(
        sorted(set(int(k) for k in k_values if k >= 1)), dtype=int
    )
    curve = []
    for k in k_values:
        t = sklearn_trustworthiness(
            X_high, Y_emb, n_neighbors=int(k), metric=metric
        )
        curve.append(float(t))
    return np.asarray(curve)


def compute_continuity_curve(
    X_high,
    Y_emb,
    k_values,
    metric_hd="euclidean",
    metric_ld="euclidean",
    n_jobs=None,
    algorithm="auto",
):
    """
    Compute continuity for multiple k values.

    Note: builds full n×n rank matrices — slow for large n.

    Returns
    -------
    curve : ndarray, shape (len(k_values),)
    """
    n = X_high.shape[0]
    k_values = np.asarray(
        sorted(set(int(k) for k in k_values if k >= 1)), dtype=int
    )

    nn_hd = NearestNeighbors(
        n_neighbors=n, metric=metric_hd, n_jobs=n_jobs, algorithm=algorithm
    )
    nn_hd.fit(X_high)
    hd_full = nn_hd.kneighbors(X_high, return_distance=False)[:, 1:]

    nn_ld = NearestNeighbors(
        n_neighbors=n, metric=metric_ld, n_jobs=n_jobs, algorithm=algorithm
    )
    nn_ld.fit(Y_emb)
    ld_full = nn_ld.kneighbors(Y_emb, return_distance=False)[:, 1:]

    ld_rank = np.empty((n, n), dtype=np.int32)
    ld_rank.fill(0)
    for i in range(n):
        ld_rank[i, ld_full[i]] = np.arange(1, n, dtype=np.int32)

    curve = []
    for k in k_values:
        if k >= n / 2:
            raise ValueError("k must satisfy k < n/2 for continuity.")
        norm = 2.0 / (k * (2 * n - 3 * k - 1))
        cont_vals = np.empty(n, dtype=float)
        for i in range(n):
            hd_topk_i = set(hd_full[i, :k])
            ld_topk_i = set(ld_full[i, :k])
            V = [j for j in hd_topk_i if j not in ld_topk_i]
            c_pen = sum(ld_rank[i, j] - k for j in V)
            cont_vals[i] = 1.0 - norm * c_pen
        curve.append(float(cont_vals.mean()))

    return np.asarray(curve)


def compute_effective_perplexity_per_point(P_cond, knn_indices=None, eps=1e-300):
    """
    Compute per-row effective perplexity: Perp_i = exp(-sum_j p log p).

    If knn_indices is given, row values are extracted in true HD rank order.

    Returns
    -------
    dict with keys:
        "perplexity_per_point" : ndarray (n,)
        "entropy_per_point"    : ndarray (n,)
        "summary"              : dict of statistics
    """
    n = P_cond.shape[0]
    entropy = np.empty(n, dtype=float)
    perplexity = np.empty(n, dtype=float)

    for i in range(n):
        if knn_indices is not None:
            vals = get_ranked_row_values_from_P(P_cond, i, knn_indices[i])
        else:
            s, e = P_cond.indptr[i], P_cond.indptr[i + 1]
            vals = P_cond.data[s:e]

        vals = np.asarray(vals, dtype=float)
        vals = np.maximum(vals, eps)   # floor before filtering so log(0) is never called
        vals = vals[vals > 0]          # drop exact zeros (should be none after floor, but guard)

        H = -np.sum(vals * np.log(vals))
        entropy[i] = H
        perplexity[i] = np.exp(H)

    rounded = np.round(perplexity, 3)
    vals_unique, counts = np.unique(rounded, return_counts=True)
    mode_val = vals_unique[np.argmax(counts)]

    summary = {
        "mean": float(perplexity.mean()),
        "median": float(np.median(perplexity)),
        "std": float(perplexity.std()),
        "min": float(perplexity.min()),
        "max": float(perplexity.max()),
        "mode_rounded_3dp": float(mode_val),
    }

    return {
        "perplexity_per_point": perplexity,
        "entropy_per_point": entropy,
        "summary": summary,
    }


# =============================================================================
# Experiment orchestration
# =============================================================================

def run_embeddings_for_perplexities(
    X,
    perplexities=(20, 30, 50, 70, 100),
    gamma=0.7,
    metric="euclidean",
    k_hd=None,
    random_state=42,
    n_jobs=-1,
    ee=12,
    n_iter_ee=250,
    n_iter_main=750,
    learning_rate="auto",
):
    """
    Run standard and smooth t-SNE at multiple perplexities.

    For each perplexity p:
      - standard_p{p}           : openTSNE with native affinity
      - smooth_p{p}_gamma{gamma}: custom affinity with gamma smoothing

    k_hd defaults to 3 × perplexity per model (same as the original code).

    Returns
    -------
    result : dict with keys "mode", "perplexities", "gammas", "embeddings"
    """
    perplexities = [int(p) for p in perplexities]
    k_hd_candidates = [p * 3 for p in perplexities]

    embeddings = {}

    for p, k_hd_p in zip(perplexities, k_hd_candidates):
        print(f"\n===== Perplexity {p} =====")

        print(f"  Running standard t-SNE (p={p}) ...")
        Y_standard, _ = run_standard_opentsne_baseline(
            X,
            perplexity=p,
            metric=metric,
            n_jobs=n_jobs,
            random_state=random_state,
            n_iter_ee=n_iter_ee,
            n_iter_main=n_iter_main,
            ee=ee,
            learning_rate=learning_rate,
        )
        embeddings[f"standard_p{p}"] = np.asarray(Y_standard, dtype=np.float32)

        print(f"  Building conditional affinity (p={p}, k_hd={k_hd_p}) ...")
        P_cond_base, knn_indices_base, _ = build_conditional_affinity_and_knn(
            X,
            perplexity=p,
            k_hd=k_hd_p,
            metric=metric,
            n_jobs=n_jobs,
        )

        print(f"  Smoothing with gamma={gamma} ...")
        P_cond_smooth = perturb_mass_smooth_true_rank_order(
            P_cond=P_cond_base,
            knn_indices=knn_indices_base,
            gamma=gamma,
        )
        P_joint_smooth = symmetrize_conditional_to_joint(P_cond_smooth)

        print(f"  Running smooth t-SNE (p={p}, gamma={gamma}) ...")
        Y_smooth = run_tsne_from_joint_P(
            X,
            P_joint_smooth,
            perplexity=p,
            n_jobs=n_jobs,
            random_state=random_state,
            n_iter_ee=n_iter_ee,
            n_iter_main=n_iter_main,
            ee=ee,
            learning_rate=learning_rate,
        )
        embeddings[f"smooth_p{p}_gamma{gamma}"] = np.asarray(Y_smooth, dtype=np.float32)

    return {
        "mode": "perplexity_sweep",
        "perplexities": list(perplexities),
        "gammas": [gamma],
        "embeddings": embeddings,
    }


def run_embeddings_for_gammas(
    X,
    perplexity=30,
    gammas=(0.5, 0.7, 0.9, 1.1, 1.3, 1.5),
    metric="euclidean",
    k_hd=None,
    random_state=42,
    n_jobs=-1,
    ee=12,
    n_iter_ee=250,
    n_iter_main=750,
    learning_rate="auto",
    include_standard=True,
):
    """
    Run embeddings at a fixed perplexity for multiple gamma values.

    gamma < 1  -> smoothing  -> model name: smooth_p{p}_gamma{gamma}
    gamma > 1  -> sharpening -> model name: sharp_p{p}_gamma{gamma}
    gamma = 1  -> skipped (equivalent to base)

    Returns
    -------
    result : dict with keys "mode", "perplexities", "gammas", "embeddings"
    """
    perplexity = int(perplexity)
    gammas = [float(g) for g in gammas]

    if k_hd is None:
        k_hd = max(perplexity, 90)

    embeddings = {}

    if include_standard:
        print(f"Running standard t-SNE (p={perplexity}) ...")
        Y_standard, _ = run_standard_opentsne_baseline(
            X,
            perplexity=perplexity,
            metric=metric,
            n_jobs=n_jobs,
            random_state=random_state,
            n_iter_ee=n_iter_ee,
            n_iter_main=n_iter_main,
            ee=ee,
            learning_rate=learning_rate,
        )
        embeddings[f"standard_p{perplexity}"] = np.asarray(Y_standard, dtype=np.float32)

    print(f"Building base conditional affinity (p={perplexity}, k_hd={k_hd}) ...")
    P_cond_base, knn_indices_base, _ = build_conditional_affinity_and_knn(
        X,
        perplexity=perplexity,
        k_hd=k_hd,
        metric=metric,
        n_jobs=n_jobs,
    )

    for gamma in gammas:
        print(f"\n===== gamma = {gamma} =====")

        if np.isclose(gamma, 1.0):
            print("  Skipping gamma=1.0 (equivalent to unmodified base affinity).")
            continue

        if gamma < 1.0:
            print(f"  Smoothing with gamma={gamma} ...")
            P_cond_mod = perturb_mass_smooth_true_rank_order(
                P_cond=P_cond_base,
                knn_indices=knn_indices_base,
                gamma=gamma,
            )
            model_name = f"smooth_p{perplexity}_gamma{gamma}"
        else:
            print(f"  Sharpening with gamma={gamma} ...")
            P_cond_mod = perturb_mass_sharpen_true_rank_order(
                P_cond=P_cond_base,
                knn_indices=knn_indices_base,
                gamma=gamma,
            )
            model_name = f"sharp_p{perplexity}_gamma{gamma}"

        P_joint_mod = symmetrize_conditional_to_joint(P_cond_mod)

        print(f"  Running t-SNE for gamma={gamma} ...")
        Y_mod = run_tsne_from_joint_P(
            X,
            P_joint_mod,
            perplexity=perplexity,
            n_jobs=n_jobs,
            random_state=random_state,
            n_iter_ee=n_iter_ee,
            n_iter_main=n_iter_main,
            ee=ee,
            learning_rate=learning_rate,
        )
        embeddings[model_name] = np.asarray(Y_mod, dtype=np.float32)

    return {
        "mode": "gamma_sweep",
        "perplexities": [perplexity],
        "gammas": list(gammas),
        "embeddings": embeddings,
    }


def compute_metric_curves_from_embeddings(
    X,
    result,
    metric_name="nh",
    k_values=np.arange(1, 101),
    metric_hd="euclidean",
    metric_ld="euclidean",
    n_jobs=None,
    algorithm="auto",
):
    """
    Compute a metric curve for every embedding stored in result["embeddings"].

    metric_name : {"nh", "trustworthiness", "continuity"}

    Returns
    -------
    dict with keys: "metric_name", "k_values", "curves",
                    "perplexities", "gammas"
    """
    if "embeddings" not in result:
        raise ValueError("result must contain 'embeddings'")

    embeddings = result["embeddings"]
    k_values = np.asarray(
        sorted(set(int(k) for k in k_values if k >= 1)), dtype=int
    )
    curves = {}

    for name, Y in embeddings.items():
        print(f"  Computing {metric_name} curve for {name} ...")

        if metric_name == "nh":
            curve = compute_neighborhood_hit_curve(
                X_high=X, Y_emb=Y, k_values=k_values,
                metric_hd=metric_hd, metric_ld=metric_ld,
                n_jobs=n_jobs, algorithm=algorithm,
            )
        elif metric_name == "trustworthiness":
            curve = compute_trustworthiness_curve(
                X_high=X, Y_emb=Y, k_values=k_values, metric=metric_hd,
            )
        elif metric_name == "continuity":
            curve = compute_continuity_curve(
                X_high=X, Y_emb=Y, k_values=k_values,
                metric_hd=metric_hd, metric_ld=metric_ld,
                n_jobs=n_jobs, algorithm=algorithm,
            )
        else:
            raise ValueError(
                "metric_name must be one of: 'nh', 'trustworthiness', 'continuity'"
            )

        curves[name] = curve

    return {
        "metric_name": metric_name,
        "k_values": k_values,
        "curves": curves,
        "perplexities": result.get("perplexities"),
        "gammas": result.get("gammas"),
    }


# =============================================================================
# Save / load
# =============================================================================

def save_embedding_experiment_result(
    result,
    save_dir="embedding_results",
    experiment_name="experiment",
):
    """
    Save embeddings + metadata to a compressed NPZ file.
    Works for both perplexity sweeps and gamma sweeps.

    Returns
    -------
    npz_path : str
    """
    os.makedirs(save_dir, exist_ok=True)

    perplexities = result.get("perplexities")
    gammas = result.get("gammas")

    def _fmt(values, prefix):
        if not values:
            return None
        safe = [str(v).replace(".", "p") for v in values]
        return f"{prefix}_{'_'.join(safe)}"

    parts = [experiment_name]
    p_part = _fmt(perplexities, "perp")
    g_part = _fmt(gammas, "gamma")
    if p_part:
        parts.append(p_part)
    if g_part:
        parts.append(g_part)

    npz_path = os.path.join(save_dir, "__".join(parts) + "__embeddings.npz")

    npz_data = {}
    if perplexities:
        npz_data["perplexities"] = np.asarray(perplexities, dtype=np.int32)
    if gammas:
        npz_data["gammas"] = np.asarray(gammas, dtype=np.float32)
    for name, Y in result["embeddings"].items():
        npz_data[f"embedding__{name}"] = np.asarray(Y, dtype=np.float32)

    np.savez_compressed(npz_path, **npz_data)
    print(f"Saved embeddings to: {npz_path}")
    return npz_path


def load_embedding_experiment_result(npz_path):
    """
    Load embeddings + metadata from a compressed NPZ file.

    Returns
    -------
    dict with keys: "embeddings", "perplexities", "gammas"
    """
    data = np.load(npz_path)
    embeddings = {}
    for key in data.files:
        if key.startswith("embedding__"):
            embeddings[key.replace("embedding__", "")] = data[key]

    return {
        "embeddings": embeddings,
        "perplexities": data["perplexities"].tolist() if "perplexities" in data else None,
        "gammas": data["gammas"].tolist() if "gammas" in data else None,
    }


def save_gamma_sweep_embedding_result(
    result,
    save_dir="embedding_gamma_results",
    experiment_name="experiment",
):
    """Save gamma-sweep embeddings to NPZ.  Returns the file path."""
    os.makedirs(save_dir, exist_ok=True)

    perplexity = int(result["perplexities"][0])
    npz_path = os.path.join(
        save_dir,
        f"{experiment_name}__p{perplexity}__gamma_sweep__embeddings.npz",
    )

    npz_data = {
        "mode": np.array(["gamma_sweep"]),
        "perplexities": np.asarray([perplexity], dtype=np.int32),
        "gammas": np.asarray(result["gammas"], dtype=np.float32),
    }
    for name, Y in result["embeddings"].items():
        npz_data[f"embedding__{name}"] = np.asarray(Y, dtype=np.float32)

    np.savez_compressed(npz_path, **npz_data)
    print(f"Saved gamma-sweep embeddings to: {npz_path}")
    return npz_path


def load_gamma_sweep_embedding_result(npz_path):
    """
    Load gamma-sweep embeddings + metadata.

    Returns
    -------
    dict with keys: "mode", "perplexities", "gammas", "embeddings"
    """
    data = np.load(npz_path, allow_pickle=True)

    embeddings = {}
    for key in data.files:
        if key.startswith("embedding__"):
            embeddings[key.replace("embedding__", "")] = data[key]

    return {
        "mode": str(data["mode"][0]),
        "perplexities": data["perplexities"].tolist(),   # fixed: was data["perplexity"]
        "gammas": data["gammas"].tolist(),
        "embeddings": embeddings,
    }


def save_metric_curves_result(
    curves_result,
    save_dir="results",
    experiment_name="experiment",
):
    """
    Save metric curves to CSV and NPZ.

    Returns
    -------
    dict with keys "csv_path", "npz_path"
    """
    os.makedirs(save_dir, exist_ok=True)

    metric_name = curves_result["metric_name"]
    k_values = curves_result["k_values"]
    curves = curves_result["curves"]
    perplexities = curves_result.get("perplexities") or []
    gammas = curves_result.get("gammas") or []

    # Build a safe filename suffix for gamma
    g = gammas[0] if gammas else None
    safe_gamma = str(g).replace(".", "p") if g is not None else "None"

    base = f"{experiment_name}__gamma_{safe_gamma}__{metric_name}"
    csv_path = os.path.join(save_dir, f"{base}__curves.csv")
    npz_path = os.path.join(save_dir, f"{base}__curves.npz")

    rows = []
    for name, curve in curves.items():
        family = "smooth" if name.startswith("smooth_") else "standard"
        p = next((pp for pp in perplexities if f"_p{pp}" in name), None)
        gamma_val = g if family == "smooth" else float("nan")

        for k, val in zip(k_values, curve):
            rows.append({
                "model": name,
                "family": family,
                "perplexity": p,
                "gamma": gamma_val,
                "k": int(k),
                f"{metric_name}_at_k": float(val),
            })

    df = pd.DataFrame(rows)
    df.to_csv(csv_path, index=False)

    npz_data = {
        "k_values": np.asarray(k_values, dtype=np.int32),
        "perplexities": np.asarray(perplexities, dtype=np.int32),
        "gamma": np.asarray([g] if g is not None else [], dtype=np.float32),
    }
    for name, curve in curves.items():
        npz_data[f"curve__{name}"] = np.asarray(curve, dtype=np.float32)

    np.savez_compressed(npz_path, **npz_data)
    print(f"Saved {metric_name} curves CSV to: {csv_path}")
    print(f"Saved {metric_name} curves NPZ to: {npz_path}")
    return {"csv_path": csv_path, "npz_path": npz_path}


def load_saved_metric_curves(npz_path, metric_name=None):
    """
    Load metric curves from a compressed NPZ file.

    Returns
    -------
    dict with keys: "metric_name", "k_values", "curves"
    """
    data = np.load(npz_path, allow_pickle=True)

    curves = {}
    for key in data.files:
        if key.startswith("curve__"):
            curves[key.replace("curve__", "")] = data[key]

    return {
        "metric_name": metric_name,
        "k_values": data["k_values"],
        "curves": curves,
    }


# =============================================================================
# Plotting helpers
# =============================================================================

def _split_standard_smooth_curves(curves_result):
    """Split a curves_result dict into standard and smooth sub-dicts,
    both keyed by perplexity integer."""
    curves = curves_result["curves"]
    standard = {}
    smooth = {}

    for name, curve in curves.items():
        if name.startswith("standard_p"):
            try:
                p = int(name.split("standard_p")[1])
                standard[p] = (name, curve)
            except Exception:
                pass
        elif name.startswith("smooth_p"):
            try:
                p = int(name.split("smooth_p")[1].split("_gamma")[0])
                smooth[p] = (name, curve)
            except Exception:
                pass

    perplexities = sorted(set(standard.keys()).union(set(smooth.keys())))
    return standard, smooth, perplexities


def _metric_label_and_title(metric_name):
    ylabel_map = {
        "nh": "NH@k",
        "trustworthiness": "Trustworthiness",
        "continuity": "Continuity",
    }
    title_map = {
        "nh": "Neighborhood Hit",
        "trustworthiness": "Trustworthiness",
        "continuity": "Continuity",
    }
    return ylabel_map.get(metric_name, metric_name), title_map.get(metric_name, metric_name)


def plot_metric_curves_for_perplexities(curves_result, figsize=(10, 6)):
    """
    Plot standard (solid) and smooth (dashed) curves, same colour per perplexity.
    """
    metric_name = curves_result["metric_name"]
    k_values = curves_result["k_values"]
    curves = curves_result["curves"]
    perplexities = curves_result.get("perplexities") or []

    ylabel, title = _metric_label_and_title(metric_name)

    plt.figure(figsize=figsize)
    cmap = plt.get_cmap("tab10")
    color_map = {p: cmap(i % 10) for i, p in enumerate(perplexities)}

    for p in perplexities:
        name = f"standard_p{p}"
        if name in curves:
            plt.plot(k_values, curves[name], color=color_map[p],
                     linestyle="-", linewidth=2.2, label=f"standard p={p}")

    for name, curve in curves.items():
        if not name.startswith("smooth_p"):
            continue
        try:
            p = int(name.split("smooth_p")[1].split("_gamma")[0])
        except Exception:
            continue
        gamma_str = name.split("_gamma")[-1] if "_gamma" in name else "?"
        plt.plot(k_values, curve, color=color_map.get(p, "black"),
                 linestyle="--", linewidth=2.2,
                 label=f"smooth p={p}, γ={gamma_str}")

    plt.xlabel("k")
    plt.ylabel(ylabel)
    plt.title(f"{title} comparison across perplexities")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()


def plot_metric_curves_standard_vs_smooth_separate(curves_result, figsize=(12, 5)):
    """
    Left subplot: all standard curves.
    Right subplot: all smooth curves.
    """
    metric_name = curves_result["metric_name"]
    ylabel, metric_title = _metric_label_and_title(metric_name)
    k_values = curves_result["k_values"]
    standard, smooth, perplexities = _split_standard_smooth_curves(curves_result)

    cmap = plt.get_cmap("tab10")
    color_map = {p: cmap(i % 10) for i, p in enumerate(perplexities)}

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for p in perplexities:
        if p in standard:
            name, curve = standard[p]
            axes[0].plot(k_values, curve, color=color_map[p],
                         linestyle="-", linewidth=2.2, label=f"p={p}")
    axes[0].set_title(f"Standard t-SNE ({metric_title})")
    axes[0].set_xlabel("k")
    axes[0].set_ylabel(ylabel)
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    for p in perplexities:
        if p in smooth:
            name, curve = smooth[p]
            gamma_str = name.split("_gamma")[-1] if "_gamma" in name else "?"
            axes[1].plot(k_values, curve, color=color_map[p],
                         linestyle="--", linewidth=2.2,
                         label=f"p={p}, γ={gamma_str}")
    axes[1].set_title(f"Smooth t-SNE ({metric_title})")
    axes[1].set_xlabel("k")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_metric_curves_paired_by_perplexity(
    curves_result, ncols=2, figsize_per_subplot=(5, 4)
):
    """One subplot per perplexity showing standard and smooth together."""
    metric_name = curves_result["metric_name"]
    ylabel, metric_title = _metric_label_and_title(metric_name)
    k_values = curves_result["k_values"]
    standard, smooth, perplexities = _split_standard_smooth_curves(curves_result)

    n = len(perplexities)
    ncols = min(ncols, n)
    nrows = int(np.ceil(n / ncols))

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(figsize_per_subplot[0] * ncols, figsize_per_subplot[1] * nrows),
        sharex=True, sharey=True,
    )

    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = np.array([axes])
    elif ncols == 1:
        axes = np.array([[ax] for ax in axes])

    axes_flat = axes.ravel()
    cmap = plt.get_cmap("tab10")
    color_map = {p: cmap(i % 10) for i, p in enumerate(perplexities)}

    for ax, p in zip(axes_flat, perplexities):
        color = color_map[p]
        if p in standard:
            _, curve_standard = standard[p]
            ax.plot(k_values, curve_standard, color=color,
                    linestyle="-", linewidth=2.2, label=f"standard p={p}")
        if p in smooth:
            name_smooth, curve_smooth = smooth[p]
            gamma_str = name_smooth.split("_gamma")[-1] if "_gamma" in name_smooth else "?"
            ax.plot(k_values, curve_smooth, color=color,
                    linestyle="--", linewidth=2.2,
                    label=f"smooth p={p}, γ={gamma_str}")
        ax.set_title(f"{metric_title}, p={p}")
        ax.set_xlabel("k")
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()

    for ax in axes_flat[len(perplexities):]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    return fig, axes


def plot_metric_curves_for_gammas(curves_result, figsize=(10, 6)):
    """
    Plot curves for a gamma sweep (fixed perplexity).
    Standard = black solid.  Smooth = blue shades (dashed).  Sharp = red shades (dash-dot).
    """
    metric_name = curves_result["metric_name"]
    ylabel, metric_title = _metric_label_and_title(metric_name)
    k_values = curves_result["k_values"]
    curves = curves_result["curves"]

    def _extract_gamma(name):
        try:
            return float(name.split("_gamma")[-1])
        except Exception:
            return np.nan

    standard_names = sorted([n for n in curves if n.startswith("standard_")])
    smooth_names = sorted([n for n in curves if n.startswith("smooth_")],
                          key=_extract_gamma)
    sharp_names = sorted([n for n in curves if n.startswith("sharp_")],
                         key=_extract_gamma)

    plt.figure(figsize=figsize)

    for name in standard_names:
        plt.plot(k_values, curves[name], color="black",
                 linestyle="-", linewidth=2.5, label=name)

    blues = plt.get_cmap("Blues")
    for i, name in enumerate(smooth_names):
        c = blues(0.45 + 0.45 * (i / max(1, len(smooth_names) - 1)))
        plt.plot(k_values, curves[name], color=c,
                 linestyle="--", linewidth=2.2, label=name)

    reds = plt.get_cmap("OrRd")
    for i, name in enumerate(sharp_names):
        c = reds(0.45 + 0.45 * (i / max(1, len(sharp_names) - 1)))
        plt.plot(k_values, curves[name], color=c,
                 linestyle="-.", linewidth=2.2, label=name)

    plt.xlabel("k")
    plt.ylabel(ylabel)
    plt.title(f"{metric_title} across gamma values")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.show()


def plot_effective_perplexity_histogram(
    perplexity_result,
    bins=40,
    figsize=(7, 5),
    title="Effective perplexity per point",
):
    """Histogram of per-point effective perplexities with mean/median/mode lines."""
    perp = perplexity_result["perplexity_per_point"]
    s = perplexity_result["summary"]

    plt.figure(figsize=figsize)
    plt.hist(perp, bins=bins, alpha=0.8)
    plt.axvline(s["mean"], linestyle="--", linewidth=2, label=f"mean = {s['mean']:.3f}")
    plt.axvline(s["median"], linestyle=":", linewidth=2, label=f"median = {s['median']:.3f}")
    plt.axvline(s["mode_rounded_3dp"], linestyle="-.", linewidth=2,
                label=f"mode≈ {s['mode_rounded_3dp']:.3f}")
    plt.xlabel("Effective perplexity")
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.show()
    return s


def compare_effective_perplexity_histograms(
    P_cond_dict,
    knn_indices=None,
    bins=40,
    figsize=(12, 4),
):
    """
    Side-by-side histogram of effective perplexity for multiple P_cond matrices.

    Parameters
    ----------
    P_cond_dict : dict  e.g. {"base": P_cond_base, "smooth": P_cond_smooth}
    knn_indices : ndarray, optional – for rank-ordered extraction

    Returns
    -------
    dict with "results" and "summary_df"
    """
    names = list(P_cond_dict.keys())
    m = len(names)

    fig, axes = plt.subplots(1, m, figsize=(figsize[0], figsize[1]))
    if m == 1:
        axes = [axes]

    all_results = {}
    for ax, name in zip(axes, names):
        res = compute_effective_perplexity_per_point(
            P_cond=P_cond_dict[name], knn_indices=knn_indices
        )
        all_results[name] = res

        perp = res["perplexity_per_point"]
        s = res["summary"]

        ax.hist(perp, bins=bins, alpha=0.8)
        ax.axvline(s["mean"], linestyle="--", linewidth=2,
                   label=f"mean={s['mean']:.2f}")
        ax.axvline(s["mode_rounded_3dp"], linestyle="-.", linewidth=2,
                   label=f"mode≈{s['mode_rounded_3dp']:.2f}")
        ax.set_title(f"{name}\nmean={s['mean']:.2f}, std={s['std']:.2f}")
        ax.set_xlabel("Effective perplexity")
        ax.set_ylabel("Count")
        ax.legend()

    plt.tight_layout()
    plt.show()

    rows = [{"model": name, **res["summary"]} for name, res in all_results.items()]
    summary_df = pd.DataFrame(rows)
    print(summary_df.round(4).to_string(index=False))

    return {"results": all_results, "summary_df": summary_df}
