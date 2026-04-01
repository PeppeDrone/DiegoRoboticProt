#!/usr/bin/env python3
"""B5_eval_number_meta_clusters.py

Evaluate the optimal number of meta-synergies (K) using:
  - Elbow method on inertia curve (primary choice)
  - Silhouette score (left axis) for reference

Single-panel plot; optimal K marked with x and vertical dashed line.
Use --force-k 4 to fix K=4 regardless of elbow.

Usage:
  python B5_eval_number_meta_clusters.py --force-k 4
  python B5_eval_number_meta_clusters.py --results-dir results/synergies --k-max 10
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Reuse data loading from B5
try:
    from B5_meta_synergy_clustering import load_and_stack_w_vectors
except ImportError:
    load_and_stack_w_vectors = None


def evaluate_k_range(
    stacked: np.ndarray,
    k_min: int,
    k_max: int,
    random_state: int = 42,
) -> Tuple[List[int], List[float], List[float]]:
    """
    Run K-means for each K in [k_min, k_max], return (k_list, inertias, silhouettes).
    Silhouette requires at least 2 clusters and 2 samples per cluster.
    """
    k_list: List[int] = []
    inertias: List[float] = []
    silhouettes: List[float] = []

    for k in range(k_min, k_max + 1):
        if k > stacked.shape[0]:
            break
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(stacked)
        inertia = float(kmeans.inertia_)
        k_list.append(k)
        inertias.append(inertia)

        # Silhouette: need at least 2 clusters and 2 samples per cluster
        if k >= 2 and k < stacked.shape[0]:
            n_per_cluster = np.bincount(labels, minlength=k)
            if np.all(n_per_cluster >= 2):
                sil = silhouette_score(stacked, labels)
                silhouettes.append(float(sil))
            else:
                silhouettes.append(np.nan)
        else:
            silhouettes.append(np.nan)

    return k_list, inertias, silhouettes


def elbow_k_from_inertia(k_list: List[int], inertias: List[float]) -> int:
    """
    Elbow method: find K where inertia curve bends (max perpendicular distance
    from line connecting first and last point).
    """
    n = len(k_list)
    if n < 3:
        return k_list[-1] if k_list else 2
    k_arr = np.array(k_list, dtype=float)
    in_arr = np.array(inertias, dtype=float)
    p1 = np.array([k_arr[0], in_arr[0]])
    p2 = np.array([k_arr[-1], in_arr[-1]])
    line_vec = p2 - p1
    line_len = np.linalg.norm(line_vec)
    if line_len < 1e-12:
        return int(k_list[0])
    max_dist = -1.0
    best_idx = 0
    for i in range(n):
        p = np.array([k_arr[i], in_arr[i]])
        dist = np.abs(np.cross(line_vec, p - p1) / line_len)
        if dist > max_dist:
            max_dist = dist
            best_idx = i
    return int(k_list[best_idx])


def plot_evaluation(
    k_list: List[int],
    inertias: List[float],
    silhouettes: List[float],
    k_opt: int,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Single panel: inertia (right, primary), silhouette (left); choice = elbow on inertia."""
    k_arr = np.array(k_list)
    in_arr = np.array(inertias)
    sil_arr = np.array(silhouettes)

    fig, ax1 = plt.subplots(figsize=(6, 4))

    # Left axis: silhouette
    color_sil = "#3182bd"
    ax1.set_xlabel("Number of meta-clusters (K)")
    ax1.set_ylabel("Silhouette score", color=color_sil)
    ax1.tick_params(axis="y", labelcolor=color_sil)
    ax1.plot(k_arr, sil_arr, "o-", color=color_sil, linewidth=2, markersize=8, label="Silhouette")

    # Right axis: inertia
    ax2 = ax1.twinx()
    color_in = "#756bb1"
    ax2.set_ylabel("Inertia", color=color_in)
    ax2.tick_params(axis="y", labelcolor=color_in)
    ax2.plot(k_arr, in_arr, "s--", color=color_in, linewidth=1.5, markersize=6, label="Inertia")

    # Optimum: elbow on inertia - vertical dashed line and x on inertia curve
    ax1.axvline(k_opt, color="#31a354", linestyle="--", linewidth=1.5)
    idx_opt = np.where(k_arr == k_opt)[0]
    if len(idx_opt) > 0:
        ax2.scatter(
            [k_opt], [in_arr[idx_opt[0]]],
            marker="x", s=150, linewidths=3, color="#31a354", zorder=5,
            label=f"Optimal K={k_opt} (elbow)",
        )

    ax1.set_xticks(k_arr)
    ax1.spines["top"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    lns1, labs1 = ax1.get_legend_handles_labels()
    lns2, labs2 = ax2.get_legend_handles_labels()
    ax1.legend(lns1 + lns2, labs1 + labs2, loc="upper right", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate optimal number of meta-synergies via max silhouette (inertia for reference)"
    )
    ap.add_argument("--results-dir", type=Path, default=Path("results/synergies"))
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--k-min", type=int, default=2)
    ap.add_argument("--k-max", type=int, default=8)
    ap.add_argument("--force-k", type=int, default=4, metavar="K",
                    help="Force optimal K (default: 4). Use 0 to use elbow method instead.")
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--dpi", type=int, default=150)
    args = ap.parse_args()

    if load_and_stack_w_vectors is None:
        raise ImportError("Could not import load_and_stack_w_vectors from B5_meta_synergy_clustering. Ensure B5 is in the same directory.")

    results_dir = Path(args.results_dir)
    out_dir = args.out_dir or (results_dir / "meta_synergy_clustering")
    out_path = out_dir / "figures" / "meta_clusters_eval.png"

    print("Loading and stacking W vectors...")
    stacked, muscles, _, _ = load_and_stack_w_vectors(results_dir)
    if stacked.size == 0:
        print("No W_global files found. Exiting.")
        return
    print(f"  Stacked shape: {stacked.shape}, muscles: {len(muscles)}")

    k_min = max(2, args.k_min)
    k_max = min(args.k_max, stacked.shape[0] - 1)
    if k_max < k_min:
        k_max = k_min
    print(f"Evaluating K from {k_min} to {k_max}...")

    k_list, inertias, silhouettes = evaluate_k_range(
        stacked, k_min, k_max, random_state=args.random_state
    )

    k_arr = np.array(k_list)
    k_elbow = elbow_k_from_inertia(k_list, inertias)
    k_opt = args.force_k if args.force_k > 0 else k_elbow
    if k_opt not in k_list:
        k_opt = min(k_list, key=lambda x: abs(x - k_opt)) if k_list else k_list[0]
    print(f"  Elbow K (inertia): {k_elbow}, using K={k_opt}" + (" (forced)" if args.force_k > 0 else ""))

    plot_evaluation(k_list, inertias, silhouettes, k_opt, out_path, args.dpi)
    print(f"Plot saved to {out_path}")


if __name__ == "__main__":
    main()
