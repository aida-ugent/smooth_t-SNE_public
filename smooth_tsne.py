"""
smooth_tsne.py
==============
Reproducible experiments for the paper:

  "Affinity Row Smoothing in t-SNE:
   A Power Transform on Conditional Probabilities"

All experiments are run on MNIST (default) or mouse cortex / Adult.
Each experiment saves both PDF figures and CSV result files so that
plots can be regenerated without re-running the computations.

Eight experiments
-----------------
1.  Affinity row sharpness for a single representative point
    (a) standard t-SNE  (b) after smoothing with γ=0.5
2.  Effective perplexity
    (a) distribution across points for standard / smooth / sharp
    (b) median eff. perplexity heatmap over γ × perplexity
3.  Per-point Δρ correlations with top-5 mass and σ_i
4.  Neighborhood Overlap curves: fixed perplexity, varying γ
5.  t-SNE embedding: standard vs smooth (side by side)
6.  Neighborhood Overlap sensitivity heatmaps (AUC 1–10 and AUC 11–90)
7.  Neighborhood Overlap comparison: standard / smooth / sharp / matched-perplexity
8.  Global Spearman vs γ across perplexity settings

Usage
-----
  python smooth_tsne.py --dataset mnist
  python smooth_tsne.py --dataset mnist --skip_exp2 --skip_exp6
  python smooth_tsne.py --dataset mouse --out_dir results/mouse_paper

Requirements
------------
  numpy, scipy, matplotlib, pandas, scikit-learn, openTSNE
  utils_smoothing.py  (in the same directory)
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.stats import spearmanr
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from utils_smoothing import (
    load_mouse_data,
    load_mnist_data,
    load_adult_data,
    build_conditional_affinity_and_knn,
    perturb_mass_smooth_true_rank_order,
    perturb_mass_sharpen_true_rank_order,
    symmetrize_conditional_to_joint,
    run_standard_opentsne_baseline,
    run_tsne_from_joint_P,
    compute_effective_perplexity_per_point,
    compute_neighborhood_hit_curve,
    topk_neighbors,
    get_ranked_row_values_from_P,
)

# =============================================================================
# Global plot style
# =============================================================================

plt.rcParams.update({
    "font.size":        14,
    "axes.labelsize":   15,
    "xtick.labelsize":  13,
    "ytick.labelsize":  13,
    "legend.fontsize":  13,
    "axes.titlesize":   17,
    "figure.dpi":       150,
})

# =============================================================================
# Constants
# =============================================================================

C_STD     = "black"
C_SMOOTH  = "#4393C3"
C_SHARP   = "#E8601C"
C_MATCHED = "#878787"

# ColorBrewer Dark2
_PERP_PALETTE = ["#1B9E77", "#D95F02", "#7570B3", "#E7298A",
                  "#66A61E", "#E6AB02"]

# Color-blind safe palette (Wong 2011) for exp8
_CB_PALETTE = ["#0072B2", "#E69F00", "#009E73", "#CC79A7"]

# =============================================================================
# Shared helpers
# =============================================================================

_fig_counter   = [0]
_plot_save_dir = [None]


def _save_fig(name=""):
    fig      = plt.gcf()
    idx      = _fig_counter[0]
    _fig_counter[0] += 1
    save_dir = _plot_save_dir[0] or "."
    suffix   = f"_{name}" if name else ""
    path     = os.path.join(save_dir, f"plot_{idx:03d}{suffix}.pdf")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    print(f"  [saved] {path}")
    plt.close("all")


def _clean_axes(ax):
    """Remove top/right spines and grid (paper-style open axes)."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)


def _global_spearman(X, Y, n_sample=2000, random_state=42):
    n   = X.shape[0]
    rng = np.random.default_rng(random_state)
    if n > n_sample:
        idx = np.sort(rng.choice(n, size=n_sample, replace=False))
        X, Y = X[idx], Y[idx]
    rho, _ = spearmanr(pdist(X), pdist(Y))
    return float(rho)


def _nh_auc(curve, k_values, k_lo, k_hi):
    mask = (k_values >= k_lo) & (k_values <= k_hi)
    return float(curve[mask].mean()) if mask.sum() > 0 else float("nan")


def _gamma_color(g, smooth_list, sharp_list):
    if np.isclose(g, 1.0):
        return C_STD
    if g < 1.0:
        frac = (g - min(smooth_list)) / max(max(smooth_list) - min(smooth_list), 1e-9)
        return plt.cm.Blues(0.4 + 0.5 * frac)
    frac = (g - min(sharp_list)) / max(max(sharp_list) - min(sharp_list), 1e-9)
    return plt.cm.Reds(0.4 + 0.5 * frac)


def _tsne_bandwidths(knn_distances, perplexity, tol=1e-5, max_iter=50):
    n = knn_distances.shape[0]
    sigmas = np.empty(n)
    logU   = np.log(perplexity)
    for i in range(n):
        dist2 = knn_distances[i].astype(float) ** 2
        bmin, bmax, beta = -np.inf, np.inf, 1.0
        for _ in range(max_iter):
            P    = np.exp(-dist2 * beta)
            sumP = P.sum()
            if sumP <= 0:
                break
            H     = np.log(sumP) + beta * np.dot(dist2, P) / sumP
            Hdiff = H - logU
            if abs(Hdiff) < tol:
                break
            if Hdiff > 0:
                bmin = beta
                beta = 2 * beta if np.isinf(bmax) else 0.5 * (beta + bmax)
            else:
                bmax = beta
                beta = 0.5 * beta if np.isinf(bmin) else 0.5 * (beta + bmin)
        sigmas[i] = 1.0 / np.sqrt(2.0 * max(beta, 1e-300))
    return sigmas


def _knn(X, k, n_jobs=-1):
    nn = NearestNeighbors(n_neighbors=k + 1, metric="euclidean", n_jobs=n_jobs)
    nn.fit(X)
    dist, idx = nn.kneighbors(X, return_distance=True)
    return idx[:, 1:], dist[:, 1:]


def _heatmap(ax, pivot, title, cmap):
    """Draw a labelled heatmap; text colour chosen by background luminance."""
    im   = ax.imshow(pivot.values, aspect="auto", cmap=cmap)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{g:.2f}" for g in pivot.columns],
                       rotation=45, fontsize=12)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index.astype(int), fontsize=12)
    ax.set_xlabel("γ", fontsize=14)
    ax.set_ylabel("Initial perplexity", fontsize=14)
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    _clean_axes(ax)
    ax.spines["left"].set_visible(True)
    ax.spines["bottom"].set_visible(True)

    cmap_fn  = plt.get_cmap(cmap)
    vmin     = np.nanmin(pivot.values)
    vmax     = np.nanmax(pivot.values)
    for ri in range(pivot.shape[0]):
        for ci in range(pivot.shape[1]):
            v = pivot.values[ri, ci]
            if np.isnan(v):
                continue
            norm_v = (v - vmin) / max(vmax - vmin, 1e-9)
            r, g_c, b, _ = cmap_fn(norm_v)
            lum = 0.2126 * r + 0.7152 * g_c + 0.0722 * b
            tc  = "black" if lum > 0.35 else "white"
            ax.text(ci, ri, f"{v:.3f}", ha="center", va="center",
                    fontsize=9, color=tc)
    return im


# =============================================================================
# Experiment 1 – Affinity row sharpness (single representative point)
# =============================================================================

def exp1_affinity_sharpness(X_pca, out_dir,
                              perplexity=30, k_hd=None, gamma_s=0.5,
                              n_subsample=5000, random_state=42,
                              n_jobs=-1):
    """
    (a) Bar chart of P(j|i) vs neighbour rank — standard t-SNE.
    (b) Same point after power transform with γ=gamma_s.

    CSV: affinity_rows.csv  (rank, p_standard, p_smooth)
    """
    print("\n--- Exp 1: Affinity row sharpness ---")
    plots_dir = os.path.join(out_dir, "exp1_affinity_sharpness")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    if k_hd is None:
        k_hd = perplexity * 3

    rng = np.random.default_rng(random_state)
    if n_subsample and n_subsample < len(X_pca):
        sel   = np.sort(rng.choice(len(X_pca), size=n_subsample, replace=False))
        X_pca = X_pca[sel]
    n = X_pca.shape[0]

    print(f"  Building conditional affinity (ρ={perplexity}, k_hd={k_hd}) ...")
    P_cond, knn_idx, _ = build_conditional_affinity_and_knn(
        X_pca, perplexity=perplexity, k_hd=k_hd, n_jobs=n_jobs
    )
    rows = np.zeros((n, k_hd), dtype=np.float64)
    for i in range(n):
        rows[i] = get_ranked_row_values_from_P(P_cond, i, knn_idx[i])

    ranks = np.arange(1, k_hd + 1)
    top5  = rows[:, :5].sum(axis=1)
    idx_p = int(np.argmin(np.abs(top5 - top5.mean())))

    vals_std = rows[idx_p].copy()
    vals_smo = np.maximum(vals_std, 1e-300) ** gamma_s
    vals_smo /= vals_smo.sum()
    vals_smo = np.sort(vals_smo)[::-1]
    vals_smo /= vals_smo.sum()

    # CSV
    pd.DataFrame({
        "rank": ranks,
        "p_standard": vals_std,
        f"p_smooth_gamma{gamma_s}": vals_smo,
    }).to_csv(os.path.join(plots_dir, "affinity_rows.csv"), index=False)

    C_BAR_STD = "#C0392B"
    C_BAR_SMO = "#2471A3"

    def _bar(vals, color, title, fname):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(ranks, vals, width=0.85, color=color, alpha=0.85)
        ax.set_xlabel("Neighbor rank")
        ax.set_ylabel("Conditional probability  P(j|i)")
        ax.set_title(title)
        _clean_axes(ax)
        plt.tight_layout()
        _save_fig(fname)

    _bar(vals_std, C_BAR_STD,
         f"(a) Standard t-SNE  (ρ={perplexity})",
         "01a_affinity_standard")
    _bar(vals_smo, C_BAR_SMO,
         f"(b) Smooth t-SNE  (ρ={perplexity},  γ={gamma_s})",
         "01b_affinity_smooth")
    print(f"  Exp 1 done → {plots_dir}")


# =============================================================================
# Experiment 2 – Effective perplexity
# =============================================================================

def exp2_effective_perplexity(P_cond_base, P_cond_smooth, P_cond_sharp,
                                knn_indices_pca, out_dir,
                                gamma_s=0.7, gamma_h=1.5, perplexity=30):
    """
    (a) Distribution of effective perplexities — linear scale only.

    CSV: eff_perplexity_per_point.csv  (standard, smooth, sharp columns)
    """
    print("\n--- Exp 2a: Effective perplexity distribution ---")
    plots_dir = os.path.join(out_dir, "exp2_eff_perplexity")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    _CONST_THR = 0.005

    def _ep(P):
        return compute_effective_perplexity_per_point(P, knn_indices_pca)[
            "perplexity_per_point"
        ]

    ep_std = _ep(P_cond_base)
    ep_smo = _ep(P_cond_smooth)
    ep_shp = _ep(P_cond_sharp)

    # CSV
    pd.DataFrame({
        "standard":         ep_std,
        f"smooth_g{gamma_s}": ep_smo,
        f"sharp_g{gamma_h}":  ep_shp,
    }).to_csv(os.path.join(plots_dir, "eff_perplexity_per_point.csv"), index=False)

    triples = [
        (ep_std, f"standard  (ρ={perplexity})", C_STD),
        (ep_smo, f"smooth  γ={gamma_s}",         "#2980B9"),   # distinct blue
        (ep_shp, f"sharp  γ={gamma_h}",           "#E91E8C"),   # pink
    ]

    n_pts   = len(ep_std)
    varying = [d for d, _, _ in triples if d.std() >= _CONST_THR]
    lo = min(d.min() for d in varying) if varying else 0
    hi = max(d.max() for d in varying) if varying else 1
    if lo == hi:
        lo, hi = lo - 0.5, hi + 0.5
    bins = np.linspace(lo, hi, 51)

    max_count = max(
        (int(np.histogram(d, bins=bins)[0].max())
         for d, _, _ in triples if d.std() >= _CONST_THR),
        default=0,
    )
    max_count = max(max_count, n_pts)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_ylim(0, max_count * 1.25)

    for d, lbl, c in triples:
        if d.std() < _CONST_THR:
            ax.vlines(float(d.mean()), 0, n_pts, colors=c, lw=4, zorder=5,
                      label=lbl)
        else:
            ax.hist(d, bins=bins, alpha=0.65, color=c, label=lbl)

    ax.set_xlabel("Effective perplexity")
    ax.set_ylabel("Count")
    ax.set_title("(a) Effective perplexity distribution")
    ax.legend(frameon=False)
    _clean_axes(ax)
    plt.tight_layout()
    _save_fig("02a_eff_perp")
    print(f"  Exp 2a done → {plots_dir}")


# =============================================================================
# Experiment 2b – Median effective perplexity heatmap
# =============================================================================

def exp2b_median_eff_perp_heatmap(df, out_dir):
    """
    CSV: already saved as sensitivity_grid.csv by _run_sensitivity_grid.
    """
    print("\n--- Exp 2b: Median effective perplexity heatmap ---")
    plots_dir = os.path.join(out_dir, "exp2_eff_perplexity")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    pivot = df.pivot(index="perplexity", columns="gamma",
                     values="median_eff_perp")
    fig, ax = plt.subplots(figsize=(8, 5))
    _heatmap(ax, pivot, "(b) Median effective perplexity", "cividis")
    plt.tight_layout()
    _save_fig("02b_median_eff_perp_heatmap")
    print(f"  Exp 2b done → {plots_dir}")


# =============================================================================
# Shared: sensitivity grid  (used by Exp 2b and Exp 6)
# =============================================================================

def _run_sensitivity_grid(X_pca, X_eval, gammas, perplexities, out_dir,
                           n_jobs=-1, random_state=42,
                           n_iter_ee=250, n_iter_main=750):
    print("\n  Running sensitivity grid ...")
    k_values = np.arange(1, 91)
    records  = []

    for p in sorted(perplexities):
        k_hd = p * 3
        print(f"\n    ρ={p}: building affinity ...")
        P_base, knn_idx, _ = build_conditional_affinity_and_knn(
            X_pca, perplexity=p, k_hd=k_hd, n_jobs=n_jobs
        )
        base_perp = compute_effective_perplexity_per_point(P_base, knn_idx)
        Y_std, _  = run_standard_opentsne_baseline(
            X_pca, perplexity=p, n_jobs=n_jobs, random_state=random_state,
            n_iter_ee=n_iter_ee, n_iter_main=n_iter_main,
        )
        nh_std = compute_neighborhood_hit_curve(X_eval, Y_std, k_values,
                                                n_jobs=n_jobs)
        for g in sorted(gammas):
            print(f"      γ={g:.2f} ...", end=" ", flush=True)
            if np.isclose(g, 1.0):
                Y_mod, nh_mod, perp_res = Y_std, nh_std, base_perp
            elif g < 1.0:
                P_mod    = perturb_mass_smooth_true_rank_order(P_base, knn_idx, gamma=g)
                perp_res = compute_effective_perplexity_per_point(P_mod, knn_idx)
                Y_mod    = run_tsne_from_joint_P(
                    X_pca, symmetrize_conditional_to_joint(P_mod),
                    perplexity=p, n_jobs=n_jobs, random_state=random_state,
                    n_iter_ee=n_iter_ee, n_iter_main=n_iter_main,
                )
                nh_mod = compute_neighborhood_hit_curve(X_eval, Y_mod, k_values,
                                                        n_jobs=n_jobs)
            else:
                P_mod    = perturb_mass_sharpen_true_rank_order(P_base, knn_idx, gamma=g)
                perp_res = compute_effective_perplexity_per_point(P_mod, knn_idx)
                Y_mod    = run_tsne_from_joint_P(
                    X_pca, symmetrize_conditional_to_joint(P_mod),
                    perplexity=p, n_jobs=n_jobs, random_state=random_state,
                    n_iter_ee=n_iter_ee, n_iter_main=n_iter_main,
                )
                nh_mod = compute_neighborhood_hit_curve(X_eval, Y_mod, k_values,
                                                        n_jobs=n_jobs)
            records.append({
                "perplexity":      p,
                "gamma":           g,
                "AUC_1_10":        _nh_auc(nh_mod, k_values, 1, 10),
                "AUC_11_90":       _nh_auc(nh_mod, k_values, 11, 90),
                "global_spearman": _global_spearman(X_eval, np.asarray(Y_mod),
                                                     random_state=random_state),
                "median_eff_perp": float(np.median(
                    perp_res["perplexity_per_point"])),
            })
            print("done")

    df = pd.DataFrame(records)
    csv_path = os.path.join(out_dir, "sensitivity_grid.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Grid saved → {csv_path}")
    return df


# =============================================================================
# Experiment 3 – Per-point Δρ correlations
# =============================================================================

def exp3_delta_perp_correlations(P_cond_base, P_cond_smooth, knn_indices_pca,
                                   knn_distances_pca, out_dir,
                                   perplexity=30, gamma_s=0.7):
    """
    Hexbin scatter plots of Δρ (change in effective perplexity after smoothing)
    versus (a) top-5 conditional mass, (b) t-SNE bandwidth σ_i.

    CSV: delta_perp_per_point.csv
    """
    print("\n--- Exp 3: Δρ correlations ---")
    plots_dir = os.path.join(out_dir, "exp3_delta_perp")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    res_base   = compute_effective_perplexity_per_point(P_cond_base,   knn_indices_pca)
    res_smooth = compute_effective_perplexity_per_point(P_cond_smooth, knn_indices_pca)
    delta_perp = (res_smooth["perplexity_per_point"] -
                  res_base["perplexity_per_point"])

    n    = P_cond_base.shape[0]
    top5 = np.zeros(n)
    for i in range(n):
        vals    = get_ranked_row_values_from_P(P_cond_base, i, knn_indices_pca[i])
        top5[i] = float(np.maximum(vals, 1e-300)[:5].sum())

    sigmas = _tsne_bandwidths(knn_distances_pca, perplexity)

    # CSV
    pd.DataFrame({
        "delta_rho":  delta_perp,
        "top5_mass":  top5,
        "sigma_i":    sigmas,
    }).to_csv(os.path.join(plots_dir, "delta_perp_per_point.csv"), index=False)

    candidates = [
        (top5,   "Top-5 conditional mass",       "03a_delta_rho_vs_top5",  "(a)"),
        (sigmas, r"t-SNE bandwidth  $\sigma_i$", "03b_delta_rho_vs_sigma", "(b)"),
    ]

    for xvar, xlabel, fname, prefix in candidates:
        rho, pval = spearmanr(xvar, delta_perp)
        fig, ax = plt.subplots(figsize=(6, 5))
        hb = ax.hexbin(xvar, delta_perp, gridsize=45, cmap="Blues", mincnt=1)
        plt.colorbar(hb, ax=ax, label="Count")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(r"$\Delta$ perplexity")
        ax.set_title(f"{prefix} Spearman ρ = {rho:.3f}")
        ax.set_ylim(bottom=0)
        _clean_axes(ax)
        plt.tight_layout()
        _save_fig(fname)

    print(f"  Exp 3 done → {plots_dir}")


# =============================================================================
# Experiment 4 – Neighborhood Overlap curves: fixed perplexity, varying γ
# =============================================================================

def exp4_nh_gamma_sweep(X_pca, X_eval, out_dir,
                          perplexity=30, k_max=200,
                          gammas=(0.0, 0.3, 0.5, 0.7, 0.9,
                                  1.0, 1.2, 1.5, 2.0),
                          n_jobs=-1, random_state=42,
                          n_iter_ee=250, n_iter_main=750, ee=12):
    """
    Neighborhood Overlap for k = 1…k_max: fixed perplexity, varying gamma.

    CSV: neighborhood_overlap_gamma_sweep.csv
    """
    print("\n--- Exp 4: Neighborhood Overlap gamma sweep ---")
    plots_dir = os.path.join(out_dir, "exp4_neighborhood_overlap_gamma_sweep")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    gammas      = list(gammas)
    smooth_list = [g for g in gammas if g < 1.0]
    sharp_list  = [g for g in gammas if g > 1.0]
    k_values    = np.arange(1, k_max + 1)

    print(f"  Pre-computing HD KNN at k={k_max} ...")
    hd_nbrs, _ = _knn(X_eval, k=k_max, n_jobs=n_jobs)

    k_hd  = perplexity * 3
    P_base, ki, _ = build_conditional_affinity_and_knn(
        X_pca, perplexity=perplexity, k_hd=k_hd, n_jobs=n_jobs
    )

    curves = {}
    for g in gammas:
        print(f"  γ={g:.2f} ...", end=" ", flush=True)
        if np.isclose(g, 1.0):
            Y, _ = run_standard_opentsne_baseline(
                X_pca, perplexity=perplexity, n_jobs=n_jobs,
                random_state=random_state,
                n_iter_ee=n_iter_ee, n_iter_main=n_iter_main, ee=ee,
            )
        elif g < 1.0:
            P_mod = perturb_mass_smooth_true_rank_order(P_base, ki, gamma=g)
            Y = run_tsne_from_joint_P(
                X_pca, symmetrize_conditional_to_joint(P_mod),
                perplexity=perplexity, n_jobs=n_jobs, random_state=random_state,
                n_iter_ee=n_iter_ee, n_iter_main=n_iter_main, ee=ee,
            )
        else:
            P_mod = perturb_mass_sharpen_true_rank_order(P_base, ki, gamma=g)
            Y = run_tsne_from_joint_P(
                X_pca, symmetrize_conditional_to_joint(P_mod),
                perplexity=perplexity, n_jobs=n_jobs, random_state=random_state,
                n_iter_ee=n_iter_ee, n_iter_main=n_iter_main, ee=ee,
            )
        ld_nbrs  = topk_neighbors(np.asarray(Y), k_max, n_jobs=n_jobs)
        nh       = np.array([
            np.mean([len(set(hd_nbrs[i, :k]) & set(ld_nbrs[i, :k])) / k
                     for i in range(len(hd_nbrs))])
            for k in k_values
        ])
        curves[g] = nh
        print("done")

    # CSV
    csv_df = pd.DataFrame({"k": k_values})
    for g in gammas:
        csv_df[f"gamma_{g:.2f}"] = curves[g]
    csv_df.to_csv(os.path.join(plots_dir, "neighborhood_overlap_gamma_sweep.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for g in gammas:
        c   = _gamma_color(g, smooth_list, sharp_list)
        lw  = 2.6 if np.isclose(g, 1.0) else 2.0
        ls  = "-" if np.isclose(g, 1.0) else ("--" if g < 1.0 else "-.")
        lbl = f"γ={g:.2f}" + ("  (standard)" if np.isclose(g, 1.0) else "")
        ax.plot(k_values, curves[g], color=c, lw=lw, ls=ls, label=lbl)

    ax.set_xlabel("k")
    ax.set_ylabel("Neighborhood Overlap")
    ax.set_title(f"Neighborhood Overlap  —  ρ={perplexity}, varying γ")
    ax.legend(ncol=2, fontsize=12, frameon=False, loc="lower right")
    _clean_axes(ax)
    plt.tight_layout()
    _save_fig("04_neighborhood_overlap_gamma_sweep")
    print(f"  Exp 4 done → {plots_dir}")


# =============================================================================
# Experiment 5 – Embedding comparison (standard vs smooth)
# =============================================================================

def exp5_embedding_comparison(X_pca, y, out_dir,
                                dataset="mnist",
                                perplexity=30, gamma_s=0.5,
                                cluster_colors_arr=None,
                                n_jobs=-1, random_state=42,
                                n_iter_ee=250, n_iter_main=750, ee=12,
                                point_size=4, alpha=0.8):
    """
    Side-by-side scatter: standard vs smooth t-SNE.
    No CSV (embeddings are 2-D point clouds; too large).
    """
    print(f"\n--- Exp 5: Embedding comparison ({dataset}) ---")
    plots_dir = os.path.join(out_dir, "exp5_embedding")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    print(f"  Running standard t-SNE (ρ={perplexity}) ...")
    Y_std, _ = run_standard_opentsne_baseline(
        X_pca, perplexity=perplexity, n_jobs=n_jobs,
        random_state=random_state,
        n_iter_ee=n_iter_ee, n_iter_main=n_iter_main, ee=ee,
    )
    print(f"  Running smooth t-SNE (γ={gamma_s}) ...")
    k_hd  = perplexity * 3
    P_c, ki, _ = build_conditional_affinity_and_knn(
        X_pca, perplexity=perplexity, k_hd=k_hd, n_jobs=n_jobs
    )
    P_sm  = perturb_mass_smooth_true_rank_order(P_c, ki, gamma=gamma_s)
    Y_smo = run_tsne_from_joint_P(
        X_pca, symmetrize_conditional_to_joint(P_sm),
        perplexity=perplexity, n_jobs=n_jobs, random_state=random_state,
        n_iter_ee=n_iter_ee, n_iter_main=n_iter_main, ee=ee,
    )

    if dataset == "mouse" and cluster_colors_arr is not None:
        point_colors = cluster_colors_arr[y]

        def _get_major_class(name):
            if name.startswith(("Lamp5", "Vip", "Pvalb", "Sst")):
                return "Inhibitory"
            elif name.startswith(("L2/3", "L5", "L6")):
                return "Excitatory"
            return "Non-neuronal"

        major_color = {"Excitatory":   "#1f77b4",
                       "Inhibitory":   "#d62728",
                       "Non-neuronal": "#7f7f7f"}
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markersize=8,
                   markerfacecolor=major_color[cls], label=cls)
            for cls in ["Excitatory", "Inhibitory", "Non-neuronal"]
        ]
        legend_title = "Cell type"

    elif dataset == "mnist":
        palette      = plt.cm.tab10(np.linspace(0, 0.9, 10))
        digits       = np.array([int(d) for d in y])
        point_colors = palette[digits]
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markersize=8,
                   markerfacecolor=palette[d], label=str(d))
            for d in range(10)
        ]
        legend_title = "Digit"

    else:
        pal          = {0: "#4393C3", 1: "#D6604D"}
        point_colors = [pal[int(yi)] for yi in y]
        legend_handles = [
            Line2D([0], [0], marker="o", color="w", markersize=8,
                   markerfacecolor=pal[v],
                   label="Income ≤50K" if v == 0 else "Income >50K")
            for v in [0, 1]
        ]
        legend_title = "Income"

    # Save embeddings + labels so plot_only mode can regenerate without rerunning
    Y_std_arr = np.asarray(Y_std)
    Y_smo_arr = np.asarray(Y_smo)
    y_str     = [str(yi) for yi in y]
    pd.DataFrame({"x": Y_std_arr[:, 0], "y": Y_std_arr[:, 1],
                  "label": y_str}).to_csv(
        os.path.join(plots_dir, "embedding_standard.csv"), index=False)
    pd.DataFrame({"x": Y_smo_arr[:, 0], "y": Y_smo_arr[:, 1],
                  "label": y_str}).to_csv(
        os.path.join(plots_dir, "embedding_smooth.csv"), index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, Y, ttl in [
        (axes[0], Y_std_arr,
         f"Standard t-SNE  (ρ={perplexity})"),
        (axes[1], Y_smo_arr,
         f"Smooth t-SNE  (ρ={perplexity},  γ={gamma_s})"),
    ]:
        ax.scatter(Y[:, 0], Y[:, 1], c=point_colors,
                   s=point_size, alpha=alpha, edgecolors="none", rasterized=True)
        ax.set_title(ttl)
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)

    if legend_handles:
        axes[1].legend(handles=legend_handles, title=legend_title,
                       fontsize=12, title_fontsize=12, frameon=False,
                       loc="upper left",
                       bbox_to_anchor=(1.02, 1.0),
                       bbox_transform=axes[1].transAxes)
    plt.tight_layout()
    _save_fig(f"05_embedding_{dataset}")
    print(f"  Exp 5 done → {plots_dir}")


# =============================================================================
# Experiment 6 – Neighborhood Overlap sensitivity heatmaps  (smooth side only)
# =============================================================================

def exp6_sensitivity_heatmaps(df, out_dir):
    """
    Heatmaps of AUC 1–10 and AUC 11–90 over γ × perplexity.
    Only smooth and standard variants (γ ≤ 1) are shown.

    CSV: already saved as sensitivity_grid.csv
    """
    print("\n--- Exp 6: Neighborhood Overlap sensitivity heatmaps ---")
    plots_dir = os.path.join(out_dir, "exp6_sensitivity")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    # Save a copy of the full grid so this experiment is self-contained
    # and can be replotted without knowing about the sensitivity_grid subfolder.
    csv_path = os.path.join(plots_dir, "sensitivity_grid.csv")
    df.to_csv(csv_path, index=False)
    print(f"  Grid saved → {csv_path}")

    specs = [
        ("AUC_1_10",  "(a) Neighborhood Overlap  AUC  (k=1–10)",   "06a_neighborhood_overlap_auc_1_10",  "RdYlGn"),
        ("AUC_11_90", "(b) Neighborhood Overlap  AUC  (k=11–90)",  "06b_neighborhood_overlap_auc_11_90", "RdYlGn"),
    ]
    for metric, title, fname, cmap in specs:
        pivot = df.pivot(index="perplexity", columns="gamma", values=metric)
        fig, ax = plt.subplots(figsize=(7, 5))
        _heatmap(ax, pivot, title, cmap)
        plt.tight_layout()
        _save_fig(fname)

    print(f"  Exp 6 done → {plots_dir}")


# =============================================================================
# Experiment 7 – Neighborhood Overlap comparison: standard / smooth / sharp / matched
# =============================================================================

def exp7_nh_comparison(X_pca, X_eval, out_dir,
                         perplexity=30, gamma_s=0.5, gamma_h=1.5,
                         k_max=200,
                         n_jobs=-1, random_state=42,
                         n_iter_ee=250, n_iter_main=750, ee=12):
    """
    Neighborhood Overlap curves for five variants:
      1. standard  ρ=perplexity              black,      solid
      2. smooth    γ=gamma_s                  steel-blue, dashed
      3. standard  ρ=matched_smooth (median)  dark-blue,  solid
      4. sharp     γ=gamma_h                  pink,       dashed
      5. standard  ρ=matched_sharp  (median)  dark-pink,  solid

    CSV: neighborhood_overlap_comparison.csv
    """
    print("\n--- Exp 7: Neighborhood Overlap comparison ---")
    plots_dir = os.path.join(out_dir, "exp7_neighborhood_overlap_comparison")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    k_values = np.arange(1, k_max + 1)

    print(f"  Building base affinity (ρ={perplexity}) ...")
    k_hd  = perplexity * 3
    P_base, ki, _ = build_conditional_affinity_and_knn(
        X_pca, perplexity=perplexity, k_hd=k_hd, n_jobs=n_jobs
    )

    P_smo = perturb_mass_smooth_true_rank_order(P_base, ki, gamma=gamma_s)

    p_matched_smo = max(
        int(round(float(np.median(
            compute_effective_perplexity_per_point(P_smo, ki)["perplexity_per_point"]
        )))),
        perplexity + 1,
    )
    print(f"  ρ_matched (smooth γ={gamma_s}) = {p_matched_smo}")

    C_GREEN   = "#27AE60"   # beautiful green for smooth
    C_MATCHED = "#E69F00"   # bright orange — Wong color-blind safe palette   # soft purple for matched-perplexity standard

    # (label, color, ls, lw, P_mod, p_override)
    variants_spec = [
        (f"standard  ρ={perplexity}",
         C_STD,     "-",  2.6, None,  perplexity),
        (f"smooth  γ={gamma_s}",
         C_GREEN,   "--", 2.2, P_smo, None),
        (f"standard  ρ={p_matched_smo}  (matched)",
         C_MATCHED, "-",  2.2, None,  p_matched_smo),
    ]

    embeddings = {}
    for label, _, _, _, P_mod, p_run in variants_spec:
        print(f"  Running: {label} ...")
        if P_mod is not None:
            Y = run_tsne_from_joint_P(
                X_pca, symmetrize_conditional_to_joint(P_mod),
                perplexity=perplexity, n_jobs=n_jobs, random_state=random_state,
                n_iter_ee=n_iter_ee, n_iter_main=n_iter_main, ee=ee,
            )
        else:
            Y, _ = run_standard_opentsne_baseline(
                X_pca, perplexity=p_run, n_jobs=n_jobs,
                random_state=random_state,
                n_iter_ee=n_iter_ee, n_iter_main=n_iter_main, ee=ee,
            )
        embeddings[label] = np.asarray(Y)

    print("  Computing Neighborhood Overlap curves ...")
    nh_curves = {
        label: compute_neighborhood_hit_curve(X_eval, Y, k_values, n_jobs=n_jobs)
        for label, Y in embeddings.items()
    }

    # CSV
    csv_df = pd.DataFrame({"k": k_values})
    for label in embeddings:
        csv_df[label] = nh_curves[label]
    csv_df.to_csv(os.path.join(plots_dir, "neighborhood_overlap_comparison.csv"), index=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    for label, color, ls, lw, _, _ in variants_spec:
        ax.plot(k_values, nh_curves[label],
                color=color, ls=ls, lw=lw, label=label)
    ax.set_xlabel("k")
    ax.set_ylabel("Neighborhood Overlap")
    ax.set_ylim(0.2, 0.65)
    ax.set_title(f"Neighborhood Overlap  (ρ={perplexity})")
    ax.legend(fontsize=11, frameon=False)
    _clean_axes(ax)
    plt.tight_layout()
    _save_fig("07_neighborhood_overlap_comparison")
    print(f"  Exp 7 done → {plots_dir}")


# =============================================================================
# Experiment 8 – Global Spearman vs γ
# =============================================================================

def exp8_global_spearman_vs_gamma(X_pca, X_eval, out_dir,
                                    gammas, perplexities,
                                    n_jobs=-1, random_state=42,
                                    n_iter_ee=250, n_iter_main=750, ee=12):
    """
    One coloured line per perplexity; color-blind safe palette.

    CSV: global_spearman_vs_gamma.csv
    """
    print("\n--- Exp 8: Global Spearman vs γ ---")
    plots_dir = os.path.join(out_dir, "exp8_global_spearman")
    os.makedirs(plots_dir, exist_ok=True)
    _plot_save_dir[0] = plots_dir

    gammas       = sorted(gammas)
    perplexities = sorted(perplexities)
    records      = []

    for p in perplexities:
        k_hd = p * 3
        print(f"\n  ρ={p}: building affinity ...")
        P_base, ki, _ = build_conditional_affinity_and_knn(
            X_pca, perplexity=p, k_hd=k_hd, n_jobs=n_jobs
        )
        for g in gammas:
            print(f"    γ={g:.2f} ...", end=" ", flush=True)
            if np.isclose(g, 1.0):
                Y, _ = run_standard_opentsne_baseline(
                    X_pca, perplexity=p, n_jobs=n_jobs,
                    random_state=random_state,
                    n_iter_ee=n_iter_ee, n_iter_main=n_iter_main, ee=ee,
                )
            elif g < 1.0:
                P_mod = perturb_mass_smooth_true_rank_order(P_base, ki, gamma=g)
                Y = run_tsne_from_joint_P(
                    X_pca, symmetrize_conditional_to_joint(P_mod),
                    perplexity=p, n_jobs=n_jobs, random_state=random_state,
                    n_iter_ee=n_iter_ee, n_iter_main=n_iter_main, ee=ee,
                )
            else:
                P_mod = perturb_mass_sharpen_true_rank_order(P_base, ki, gamma=g)
                Y = run_tsne_from_joint_P(
                    X_pca, symmetrize_conditional_to_joint(P_mod),
                    perplexity=p, n_jobs=n_jobs, random_state=random_state,
                    n_iter_ee=n_iter_ee, n_iter_main=n_iter_main, ee=ee,
                )
            rho = _global_spearman(X_eval, np.asarray(Y),
                                   random_state=random_state)
            records.append({"perplexity": p, "gamma": g, "global_spearman": rho})
            print(f"ρ={rho:.4f}")

    df = pd.DataFrame(records)
    df.to_csv(os.path.join(plots_dir, "global_spearman_vs_gamma.csv"),
              index=False)

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors  = _CB_PALETTE[:len(perplexities)]
    markers = ["o", "s", "^", "D"]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_facecolor("white")

    for p, c, m in zip(perplexities, colors, markers):
        sub = df[df["perplexity"] == p].sort_values("gamma")
        ax.plot(sub["gamma"], sub["global_spearman"],
                color=c, marker=m, markersize=7, lw=2.2, label=f"ρ={p}")

    ax.set_xlabel("γ")
    ax.set_ylabel("Global Spearman ρ")
    ax.set_title("Effect of γ on global structure preservation")
    ax.legend(fontsize=12, frameon=False)
    _clean_axes(ax)
    plt.tight_layout()
    _save_fig("08_global_spearman_vs_gamma")
    print(f"  Exp 8 done → {plots_dir}")
    return df


# =============================================================================
# Data loading
# =============================================================================

def load_data(dataset, args):
    extra = {}
    if dataset == "mouse":
        import pickle
        data_dir = args.mouse_data_dir or os.path.join(_SCRIPT_DIR, "..",
                                                        "mouse-data")
        pkl_path = os.path.join(data_dir, "tasic2018.pickle")
        print(f"Loading mouse cortex from {pkl_path} ...")
        X_pca, _, y, _, _ = load_mouse_data(pkl_path, data_dir=data_dir,
                                              return_highdim=True)
        with open(pkl_path, "rb") as fh:
            tasic = pickle.load(fh)
        extra["cluster_colors_arr"] = tasic["clusterColors"]
        if args.mouse_subsample and args.mouse_subsample < len(X_pca):
            rng = np.random.default_rng(args.random_state)
            sel = np.sort(rng.choice(len(X_pca), size=args.mouse_subsample,
                                      replace=False))
            X_pca, y = X_pca[sel], y[sel]

    elif dataset == "mnist":
        print("Loading MNIST (PCA-50) ...")
        X_pca, y, _ = load_mnist_data(n_pca=50, random_state=args.random_state)
        if args.mnist_subsample and args.mnist_subsample < len(X_pca):
            rng = np.random.default_rng(args.random_state)
            sel = np.sort(rng.choice(len(X_pca), size=args.mnist_subsample,
                                      replace=False))
            X_pca, y = X_pca[sel], y[sel]

    elif dataset == "adult":
        print("Loading Adult census data ...")
        X_pca, y = load_adult_data(max_rows=args.adult_max_rows,
                                    random_state=args.random_state)
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    X_eval = X_pca
    print(f"  n={len(X_pca)}  d={X_pca.shape[1]}")
    return X_pca, X_eval, y, extra


# =============================================================================
# Plot-only mode  (regenerate figures from saved CSVs)
# =============================================================================

def _csv(plots_dir, fname):
    """Return full path to a CSV; raise FileNotFoundError with a clear message."""
    path = os.path.join(plots_dir, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CSV not found: {path}\n"
            "Run without --plot_only first to generate the data."
        )
    return path


def plot_all_from_csv(args):
    """
    Regenerate every figure by reading previously saved CSV files.
    No t-SNE is run; only plotting code executes.

    Usage
    -----
      python smooth_tsne.py --dataset mnist --plot_only
    """
    out_dir = args.out_dir
    print(f"\n{'='*60}")
    print("  PLOT-ONLY mode: reading CSVs from", out_dir)
    print(f"{'='*60}\n")

    # ── Exp 1 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp1:
        plots_dir = os.path.join(out_dir, "exp1_affinity_sharpness")
        _plot_save_dir[0] = plots_dir
        df1 = pd.read_csv(_csv(plots_dir, "affinity_rows.csv"))
        ranks     = df1["rank"].values
        vals_std  = df1["p_standard"].values
        smo_col   = [c for c in df1.columns if c.startswith("p_smooth")][0]
        vals_smo  = df1[smo_col].values
        gamma_s1  = float(smo_col.split("gamma")[-1])
        C_BAR_STD = "#C0392B"
        C_BAR_SMO = "#2471A3"
        for vals, color, title, fname in [
            (vals_std, C_BAR_STD,
             f"(a) Standard t-SNE  (ρ={args.perplexity})",
             "01a_affinity_standard"),
            (vals_smo, C_BAR_SMO,
             f"(b) Smooth t-SNE  (ρ={args.perplexity},  γ={gamma_s1})",
             "01b_affinity_smooth"),
        ]:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.bar(ranks, vals, width=0.85, color=color, alpha=0.85)
            ax.set_xlabel("Neighbor rank")
            ax.set_ylabel("Conditional probability  P(j|i)")
            ax.set_title(title)
            _clean_axes(ax)
            plt.tight_layout()
            _save_fig(fname)
        print("  Exp 1 replotted.")

    # ── Exp 2a ────────────────────────────────────────────────────────────────
    if not args.skip_exp2:
        plots_dir = os.path.join(out_dir, "exp2_eff_perplexity")
        _plot_save_dir[0] = plots_dir
        df2 = pd.read_csv(_csv(plots_dir, "eff_perplexity_per_point.csv"))
        col_std = "standard"
        col_smo = [c for c in df2.columns if "smooth" in c][0]
        col_shp = [c for c in df2.columns if "sharp"  in c][0]
        gs = float(col_smo.split("_g")[-1])
        gh = float(col_shp.split("_g")[-1])
        _CONST_THR = 0.005
        triples = [
            (df2[col_std].values, f"standard  (ρ={args.perplexity})", C_STD),
            (df2[col_smo].values, f"smooth  γ={gs}",  "#2980B9"),
            (df2[col_shp].values, f"sharp  γ={gh}",   "#E91E8C"),
        ]
        n_pts   = len(triples[0][0])
        varying = [d for d, _, _ in triples if d.std() >= _CONST_THR]
        lo = min(d.min() for d in varying); hi = max(d.max() for d in varying)
        bins = np.linspace(lo, hi, 51)
        max_count = max(
            (int(np.histogram(d, bins=bins)[0].max())
             for d, _, _ in triples if d.std() >= _CONST_THR), default=0)
        max_count = max(max_count, n_pts)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_ylim(0, max_count * 1.25)
        for d, lbl, c in triples:
            if d.std() < _CONST_THR:
                ax.vlines(float(d.mean()), 0, n_pts, colors=c, lw=4, zorder=5,
                          label=lbl)
            else:
                ax.hist(d, bins=bins, alpha=0.65, color=c, label=lbl)
        ax.set_xlabel("Effective perplexity")
        ax.set_ylabel("Count")
        ax.set_title("(a) Effective perplexity distribution")
        ax.legend(frameon=False)
        _clean_axes(ax)
        plt.tight_layout()
        _save_fig("02a_eff_perp")

        # 2b — median eff perp heatmap from sensitivity grid
        grid_csv = os.path.join(out_dir, "sensitivity_grid", "sensitivity_grid.csv")
        if os.path.exists(grid_csv):
            df_grid = pd.read_csv(grid_csv)
            exp2b_median_eff_perp_heatmap(df_grid, out_dir)
        print("  Exp 2 replotted.")

    # ── Exp 3 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp3:
        plots_dir = os.path.join(out_dir, "exp3_delta_perp")
        _plot_save_dir[0] = plots_dir
        df3 = pd.read_csv(_csv(plots_dir, "delta_perp_per_point.csv"))
        candidates = [
            (df3["top5_mass"].values,  "Top-5 conditional mass",
             "03a_delta_rho_vs_top5",  "(a)"),
            (df3["sigma_i"].values,    r"t-SNE bandwidth  $\sigma_i$",
             "03b_delta_rho_vs_sigma", "(b)"),
        ]
        for xvar, xlabel, fname, prefix in candidates:
            rho, pval = spearmanr(xvar, df3["delta_rho"].values)
            fig, ax = plt.subplots(figsize=(6, 5))
            hb = ax.hexbin(xvar, df3["delta_rho"].values,
                           gridsize=45, cmap="Blues", mincnt=1)
            plt.colorbar(hb, ax=ax, label="Count")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(r"$\Delta$ perplexity")
            ax.set_title(f"{prefix} Spearman ρ = {rho:.3f}")
            ax.set_ylim(bottom=0)
            _clean_axes(ax)
            plt.tight_layout()
            _save_fig(fname)
        print("  Exp 3 replotted.")

    # ── Exp 4 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp4:
        plots_dir = os.path.join(out_dir, "exp4_neighborhood_overlap_gamma_sweep")
        _plot_save_dir[0] = plots_dir
        df4      = pd.read_csv(_csv(plots_dir,
                                    "neighborhood_overlap_gamma_sweep.csv"))
        k_values = df4["k"].values
        gammas   = sorted([float(c.split("_")[-1])
                           for c in df4.columns if c.startswith("gamma_")])
        smooth_list = [g for g in gammas if g < 1.0]
        sharp_list  = [g for g in gammas if g > 1.0]
        fig, ax = plt.subplots(figsize=(8, 5))
        for g in gammas:
            c   = _gamma_color(g, smooth_list, sharp_list)
            lw  = 2.6 if np.isclose(g, 1.0) else 2.0
            ls  = "-" if np.isclose(g, 1.0) else ("--" if g < 1.0 else "-.")
            lbl = f"γ={g:.2f}" + ("  (standard)" if np.isclose(g, 1.0) else "")
            ax.plot(k_values, df4[f"gamma_{g:.2f}"].values,
                    color=c, lw=lw, ls=ls, label=lbl)
        ax.set_xlabel("k")
        ax.set_ylabel("Neighborhood Overlap")
        ax.set_title(f"Neighborhood Overlap  —  ρ={args.perplexity}, varying γ")
        ax.legend(ncol=2, fontsize=12, frameon=False, loc="lower right")
        _clean_axes(ax)
        plt.tight_layout()
        _save_fig("04_neighborhood_overlap_gamma_sweep")
        print("  Exp 4 replotted.")

    # ── Exp 5 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp5:
        plots_dir = os.path.join(out_dir, "exp5_embedding")
        _plot_save_dir[0] = plots_dir
        df_std = pd.read_csv(_csv(plots_dir, "embedding_standard.csv"))
        df_smo = pd.read_csv(_csv(plots_dir, "embedding_smooth.csv"))
        labels = df_std["label"].values

        # Reconstruct per-point colours from saved labels
        dataset = args.dataset
        if dataset == "mnist":
            palette      = plt.cm.tab10(np.linspace(0, 0.9, 10))
            digits       = np.array([int(d) for d in labels])
            point_colors = palette[digits]
            legend_handles = [
                Line2D([0], [0], marker="o", color="w", markersize=8,
                       markerfacecolor=palette[d], label=str(d))
                for d in range(10)
            ]
            legend_title = "Digit"
        elif dataset == "adult":
            pal          = {0: "#4393C3", 1: "#D6604D"}
            point_colors = [pal[int(v)] for v in labels]
            legend_handles = [
                Line2D([0], [0], marker="o", color="w", markersize=8,
                       markerfacecolor=pal[v],
                       label="Income ≤50K" if v == 0 else "Income >50K")
                for v in [0, 1]
            ]
            legend_title = "Income"
        else:
            # Mouse — colours stored as hex strings in the label column
            point_colors = labels
            legend_handles, legend_title = [], "Cell type"

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        for ax, df_emb, ttl in [
            (axes[0], df_std,
             f"Standard t-SNE  (ρ={args.perplexity})"),
            (axes[1], df_smo,
             f"Smooth t-SNE  (ρ={args.perplexity},  γ={args.exp5_gamma})"),
        ]:
            ax.scatter(df_emb["x"], df_emb["y"], c=point_colors,
                       s=4, alpha=0.8, edgecolors="none", rasterized=True)
            ax.set_title(ttl)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
        if legend_handles:
            axes[1].legend(handles=legend_handles, title=legend_title,
                           fontsize=12, title_fontsize=12, frameon=False,
                           loc="upper left",
                           bbox_to_anchor=(1.02, 1.0),
                           bbox_transform=axes[1].transAxes)
        plt.tight_layout()
        _save_fig(f"05_embedding_{dataset}")
        print("  Exp 5 replotted.")

    # ── Exp 6 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp6:
        exp6_csv = os.path.join(out_dir, "exp6_sensitivity", "sensitivity_grid.csv")
        exp6_sensitivity_heatmaps(pd.read_csv(_csv(
            os.path.join(out_dir, "exp6_sensitivity"), "sensitivity_grid.csv"
        )), out_dir)
        print("  Exp 6 replotted.")

    # ── Exp 7 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp7:
        plots_dir = os.path.join(out_dir, "exp7_neighborhood_overlap_comparison")
        _plot_save_dir[0] = plots_dir
        df7      = pd.read_csv(_csv(plots_dir,
                                    "neighborhood_overlap_comparison.csv"))
        k_values = df7["k"].values
        labels   = [c for c in df7.columns if c != "k"]
        C_GREEN   = "#27AE60"
        C_MATCHED = "#E69F00"   # bright orange — Wong color-blind safe palette
        # Map label to (color, ls, lw) based on content
        def _style7(lbl):
            if "smooth" in lbl.lower():
                return C_GREEN,   "--", 2.2
            if "matched" in lbl.lower():
                return C_MATCHED, "-",  2.2
            return C_STD, "-", 2.6
        fig, ax = plt.subplots(figsize=(8, 5))
        for lbl in labels:
            c, ls, lw = _style7(lbl)
            ax.plot(k_values, df7[lbl].values, color=c, ls=ls, lw=lw, label=lbl)
        ax.set_xlabel("k")
        ax.set_ylabel("Neighborhood Overlap")
        ax.set_ylim(0.2, 0.65)
        ax.set_title(f"Neighborhood Overlap  (ρ={args.perplexity})")
        ax.legend(fontsize=11, frameon=False)
        _clean_axes(ax)
        plt.tight_layout()
        _save_fig("07_neighborhood_overlap_comparison")
        print("  Exp 7 replotted.")

    # ── Exp 8 ─────────────────────────────────────────────────────────────────
    if not args.skip_exp8:
        plots_dir = os.path.join(out_dir, "exp8_global_spearman")
        _plot_save_dir[0] = plots_dir
        df8          = pd.read_csv(_csv(plots_dir,
                                        "global_spearman_vs_gamma.csv"))
        perplexities = sorted(df8["perplexity"].unique())
        colors       = _CB_PALETTE[:len(perplexities)]
        markers      = ["o", "s", "^", "D"]
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.set_facecolor("white")
        for p, c, m in zip(perplexities, colors, markers):
            sub = df8[df8["perplexity"] == p].sort_values("gamma")
            ax.plot(sub["gamma"], sub["global_spearman"],
                    color=c, marker=m, markersize=7, lw=2.2, label=f"ρ={p}")
        ax.set_xlabel("γ")
        ax.set_ylabel("Global Spearman ρ")
        ax.set_title("Effect of γ on global structure preservation")
        ax.legend(fontsize=12, frameon=False)
        _clean_axes(ax)
        plt.tight_layout()
        _save_fig("08_global_spearman_vs_gamma")
        print("  Exp 8 replotted.")

    print(f"\nAll plots regenerated in: {out_dir}\n")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Smooth t-SNE paper experiments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dataset",         required=True,
                   choices=["mnist", "mouse", "adult"])
    p.add_argument("--out_dir",         default=None)
    p.add_argument("--mouse_data_dir",  default=None)
    p.add_argument("--mouse_subsample", type=int, default=None,
                   help="Subsample mouse cells (default: use all).")
    p.add_argument("--mnist_subsample", type=int, default=5000)
    p.add_argument("--adult_max_rows",  type=int, default=5000)
    p.add_argument("--perplexity",      type=int,   default=30)
    p.add_argument("--gamma_s",         type=float, default=0.7)
    p.add_argument("--gamma_h",         type=float, default=1.5)
    p.add_argument("--exp1_gamma",      type=float, default=0.5)
    p.add_argument("--exp4_gammas",     type=float, nargs="+",
                   default=[0.0, 0.3, 0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0])
    p.add_argument("--exp5_gamma",      type=float, default=0.5)
    p.add_argument("--exp_grid_gammas", type=float, nargs="+",
                   default=[0.0, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0])
    p.add_argument("--exp_grid_perps",  type=int,   nargs="+",
                   default=[30, 50, 100, 200])
    p.add_argument("--k_max",           type=int,   default=200)
    p.add_argument("--exp4_k_max",      type=int,   default=200)
    p.add_argument("--n_iter_ee",       type=int,   default=250)
    p.add_argument("--n_iter_main",     type=int,   default=750)
    p.add_argument("--ee",              type=float, default=12.0)
    p.add_argument("--n_jobs",          type=int,   default=-1)
    p.add_argument("--random_state",    type=int,   default=42)
    for i in range(1, 9):
        p.add_argument(f"--skip_exp{i}", action="store_true")
    p.add_argument("--plot_only", action="store_true",
                   help="Skip all computation; regenerate figures from saved CSVs.")
    return p.parse_args()


# =============================================================================
# Main
# =============================================================================

def main():
    args = parse_args()

    if args.out_dir is None:
        args.out_dir = os.path.join(_SCRIPT_DIR, "..", "results",
                                    f"{args.dataset}_smooth_tsne")
    args.out_dir = os.path.abspath(args.out_dir)
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  dataset    : {args.dataset}")
    print(f"  out_dir    : {args.out_dir}")
    print(f"  perplexity : {args.perplexity}")
    print(f"  gamma_s/h  : {args.gamma_s} / {args.gamma_h}")
    print(f"  plot_only  : {args.plot_only}")
    print(f"{'='*60}\n")

    # ── Plot-only mode: no computation, just read CSVs ─────────────────────
    if args.plot_only:
        plot_all_from_csv(args)
        return

    nj = args.n_jobs
    X_pca, X_eval, y, extra = load_data(args.dataset, args)

    # ── Shared pre-computation ─────────────────────────────────────────────
    print("Building shared conditional affinities ...")
    k_hd = args.perplexity * 3
    P_base, knn_idx_pca, knn_dist_pca = build_conditional_affinity_and_knn(
        X_pca, perplexity=args.perplexity, k_hd=k_hd, n_jobs=nj
    )
    P_smooth = perturb_mass_smooth_true_rank_order(
        P_base, knn_idx_pca, gamma=args.gamma_s
    )
    P_sharp  = perturb_mass_sharpen_true_rank_order(
        P_base, knn_idx_pca, gamma=args.gamma_h
    )

    # ── Sensitivity grid (shared by Exp 2b and Exp 6) ─────────────────────
    df_grid    = None
    need_grid  = not args.skip_exp2 or not args.skip_exp6
    if need_grid:
        print("Computing sensitivity grid ...")
        grid_dir = os.path.join(args.out_dir, "sensitivity_grid")
        os.makedirs(grid_dir, exist_ok=True)
        df_grid = _run_sensitivity_grid(
            X_pca, X_eval,
            gammas=args.exp_grid_gammas,
            perplexities=args.exp_grid_perps,
            out_dir=grid_dir,
            n_jobs=nj, random_state=args.random_state,
            n_iter_ee=args.n_iter_ee, n_iter_main=args.n_iter_main,
        )

    if not args.skip_exp1:
        exp1_affinity_sharpness(
            X_pca, args.out_dir,
            perplexity=args.perplexity,
            k_hd=k_hd, gamma_s=args.exp1_gamma,
            n_jobs=nj, random_state=args.random_state,
        )

    if not args.skip_exp2:
        exp2_effective_perplexity(
            P_base, P_smooth, P_sharp,
            knn_idx_pca, args.out_dir,
            gamma_s=args.gamma_s, gamma_h=args.gamma_h,
            perplexity=args.perplexity,
        )
        if df_grid is not None:
            exp2b_median_eff_perp_heatmap(df_grid, args.out_dir)

    if not args.skip_exp3:
        exp3_delta_perp_correlations(
            P_base, P_smooth, knn_idx_pca, knn_dist_pca, args.out_dir,
            perplexity=args.perplexity, gamma_s=args.gamma_s,
        )

    if not args.skip_exp4:
        exp4_nh_gamma_sweep(
            X_pca, X_eval, args.out_dir,
            perplexity=args.perplexity,
            k_max=args.exp4_k_max,
            gammas=args.exp4_gammas,
            n_jobs=nj, random_state=args.random_state,
            n_iter_ee=args.n_iter_ee, n_iter_main=args.n_iter_main, ee=args.ee,
        )

    if not args.skip_exp5:
        exp5_embedding_comparison(
            X_pca, y, args.out_dir,
            dataset=args.dataset,
            perplexity=args.perplexity,
            gamma_s=args.exp5_gamma,
            cluster_colors_arr=extra.get("cluster_colors_arr"),
            n_jobs=nj, random_state=args.random_state,
            n_iter_ee=args.n_iter_ee, n_iter_main=args.n_iter_main, ee=args.ee,
        )

    if not args.skip_exp6 and df_grid is not None:
        exp6_sensitivity_heatmaps(df_grid, args.out_dir)

    if not args.skip_exp7:
        exp7_nh_comparison(
            X_pca, X_eval, args.out_dir,
            perplexity=args.perplexity,
            gamma_s=args.gamma_s,
            gamma_h=args.gamma_h,
            k_max=args.k_max,
            n_jobs=nj, random_state=args.random_state,
            n_iter_ee=args.n_iter_ee, n_iter_main=args.n_iter_main, ee=args.ee,
        )

    if not args.skip_exp8:
        exp8_global_spearman_vs_gamma(
            X_pca, X_eval, args.out_dir,
            gammas=args.exp_grid_gammas,
            perplexities=args.exp_grid_perps,
            n_jobs=nj, random_state=args.random_state,
            n_iter_ee=args.n_iter_ee, n_iter_main=args.n_iter_main, ee=args.ee,
        )

    print(f"\n{'='*60}")
    print("ALL EXPERIMENTS DONE")
    print(f"Results in: {args.out_dir}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
