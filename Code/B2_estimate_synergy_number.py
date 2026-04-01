#!/usr/bin/env python3
"""B2_estimate_synergy_number.py

Estimate the number of muscle synergies per patient/task.
Uses same preprocessing, normalization, segmentation, and concatenation as B3.

Primary selection rule (sole criterion for k):
  Clark-inspired / Clark-style adaptation using per-muscle VAF only.
  Reference: Clark DJ, Ting LH, Zajac FE, Neptune RR, Kautz SA. 2010.
  "Merging of healthy motor modules predicts reduced locomotor performance
   and muscle coordination complexity post-stroke." Journal of Neurophysiology.

  The original Clark et al. 2010 criterion used muscle-level and region-level
  VAF in gait-cycle regions. This adaptation uses per-muscle VAF only, suited
  to the current segmented/concatenated EMG pipeline.

  Near-zero-energy muscles (total energy <= --muscle-energy-epsilon): per-muscle
  VAF is set to NaN (non-evaluable), never 1.0. They are excluded from Clark
  evaluation. Diagnostics are for audit only and do not alter selection.

  - Criterion C: smallest k such that every evaluable muscle reaches VAF >= threshold.
  - Criterion D (Clark stopping rule): if C never met, compare k and k+1;
    identify the muscle(s) with lowest VAF at k; if adding k+1 does not
    improve those muscles by more than the improvement threshold (default 5%),
    stop and select k.
  - Criterion F: if neither C nor D met by k_max, select k_max and document.

AIC, BIC, stability, global VAF, delta VAF, etc. remain in outputs for
reporting and diagnostics but do NOT determine the selected k.

Outputs: CSV metrics, recommendations, and config JSON.
Figures: Generated separately by B2_plot_synergy_estimation.py.

Usage:
  python B2_estimate_synergy_number.py
  python B2_estimate_synergy_number.py --data-dir data/emg_structured
  python B2_estimate_synergy_number.py --patients CROSS_001 --k-max 8
  python B2_plot_synergy_estimation.py   # after estimation, to generate figures
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.decomposition import NMF

from B3_emg_feature_computation import (
    DEFAULT_MUSCLES,
    PreprocessingConfig,
    build_emg_matrix,
    build_segmented_concat,
    compute_per_muscle_scale,
    apply_per_muscle_scale,
    discover_patients_with_data,
    discover_tasks_for_patient,
    extract_segment_data,
    load_emg_record,
    load_manual_segments,
    resolve_muscles_to_channels,
)

# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
EPS = 1e-12
# Clark et al. 2010 thresholds (used for primary selection only)
DEFAULT_CLARK_MUSCLE_VAF_THRESHOLD = 0.90
DEFAULT_CLARK_IMPROVEMENT_THRESHOLD = 0.01 #5
# Reporting-only thresholds (NOT used for Clark selection)
DEFAULT_GLOBAL_VAF_THRESHOLD = 0.90
DEFAULT_PER_MUSCLE_VAF_THRESHOLD = 0.75
DEFAULT_DELTA_VAF_THRESHOLD = 0.01
DEFAULT_STABILITY_THRESHOLD = 0.85

LOG = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    """Configure logging for the script."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    LOG.setLevel(level)


# -----------------------------------------------------------------------------
# NMF fitting with real multi-start
# -----------------------------------------------------------------------------
def fit_nmf_for_k(
    X: np.ndarray,
    k: int,
    n_restarts: int = 10,
    base_random_state: Optional[int] = 42,
) -> Tuple[np.ndarray, np.ndarray, float, int, int, bool, List[np.ndarray]]:
    """Fit NMF with k components using mixed init (nndsvda + random restarts).

    First restart uses init="nndsvda"; subsequent restarts use init="random"
    for truly different initializations. max_iter=1000 for better convergence.

    Returns:
        best_W, best_H, best_rss, n_success, n_failed, hit_max_iter, all_W_list
        all_W_list: list of W from successful runs (for stability computation).
        If all restarts fail, returns (empty arrays, np.nan, 0, n_restarts, True, []).
    """
    X = np.maximum(np.asarray(X, dtype=np.float64), 0.0)
    n_samples, n_muscles = X.shape
    max_iter = 1000
    k = min(k, n_muscles, n_samples)
    if k < 1:
        return np.zeros((n_muscles, 0)), np.zeros((0, n_samples)), np.nan, 0, 0, False, []

    best_rss = np.inf
    best_W, best_H = None, None
    n_success = 0
    n_failed = 0
    hit_max_iter = False
    all_W_list: List[np.ndarray] = []

    for r in range(n_restarts):
        init_str = "nndsvda" if r == 0 else "random"
        rs = (base_random_state + r) if base_random_state is not None else None
        nmf = NMF(
            n_components=k,
            init=init_str,
            solver="cd",
            beta_loss=2.0,
            max_iter=max_iter,
            random_state=rs,
        )
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                W = nmf.fit_transform(X.T)
                if w:
                    for wi in w:
                        if "Maximum" in str(wi.message) or "max_iter" in str(wi.message).lower():
                            hit_max_iter = True
                if hasattr(nmf, "n_iter_") and nmf.n_iter_ >= max_iter:
                    hit_max_iter = True
            H = nmf.components_
            X_hat = (W @ H).T
            rss = float(np.sum((X - X_hat) ** 2))
            n_success += 1
            norms = np.linalg.norm(W, axis=0, keepdims=True)
            norms = np.where(norms < EPS, 1.0, norms)
            all_W_list.append((W / norms).copy())
            if rss < best_rss:
                best_rss = rss
                best_W, best_H = W.copy(), H.copy()
        except Exception:
            n_failed += 1
            continue

    if best_W is None:
        return (
            np.zeros((n_muscles, 0)),
            np.zeros((0, n_samples)),
            np.nan,
            0,
            n_restarts,
            hit_max_iter,
            [],
        )
    return best_W, best_H, best_rss, n_success, n_failed, hit_max_iter, all_W_list


# -----------------------------------------------------------------------------
# VAF metrics
# -----------------------------------------------------------------------------
def compute_global_vaf(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Uncentered global VAF: 1 - sum((X - X_hat)^2) / sum(X^2)."""
    ss_tot = np.sum(X ** 2)
    if ss_tot < EPS:
        return 1.0
    ss_res = np.sum((X - X_hat) ** 2)
    return float(1.0 - ss_res / ss_tot)


def compute_per_muscle_vaf(
    X: np.ndarray,
    X_hat: np.ndarray,
    muscle_energy_epsilon: float = 1e-10,
) -> Tuple[np.ndarray, np.ndarray]:
    """Per-muscle VAF: VAF_j = 1 - sum((X[:,j]-X_hat[:,j])^2) / sum(X[:,j]^2).

    Near-zero-energy muscles (ss_tot <= muscle_energy_epsilon) are treated as
    non-evaluable: VAF is set to np.nan and valid_mask[j] = False. They are
    never assigned VAF = 1.0 (which could artificially help Clark-style criteria).

    Returns:
        vaf_j: per-muscle VAF array (NaN for non-evaluable near-zero-energy muscles)
        valid_mask: boolean mask, True where muscle energy is high enough to evaluate
    """
    n_muscles = X.shape[1]
    vaf_j = np.full(n_muscles, np.nan, dtype=np.float64)
    valid_mask = np.ones(n_muscles, dtype=bool)
    for j in range(n_muscles):
        ss_tot = np.sum(X[:, j] ** 2)
        if ss_tot <= muscle_energy_epsilon:
            vaf_j[j] = np.nan
            valid_mask[j] = False
        else:
            ss_res = np.sum((X[:, j] - X_hat[:, j]) ** 2)
            vaf_j[j] = 1.0 - ss_res / ss_tot
            valid_mask[j] = True
    return vaf_j, valid_mask

# -----------------------------------------------------------------------------
# AIC / BIC (heuristic; for reporting only, NOT used for Clark selection)
# -----------------------------------------------------------------------------
def compute_aic_bic(
    X: np.ndarray, X_hat: np.ndarray, k: int
) -> Tuple[float, float]:
    """AIC and BIC for NMF (heuristic; reporting/diagnostics only).

    Parameter count: n_params = k * (n_muscles + n_samples) - k * k
    RSS = sum((X - X_hat)^2), n = n_samples * n_muscles
    AIC = n * log(RSS/n) + 2 * n_params
    BIC = n * log(RSS/n) + log(n) * n_params
    """
    n_samples, n_muscles = X.shape
    n = n_samples * n_muscles
    n_params = k * (n_muscles + n_samples) - k * k
    n_params = max(n_params, 0)
    rss = np.sum((X - X_hat) ** 2)
    rss = max(rss, EPS)
    if n < 2:
        return np.nan, np.nan
    log_rss_n = np.log(rss / n)
    aic = n * log_rss_n + 2 * n_params
    bic = n * log_rss_n + np.log(n) * n_params
    return float(aic), float(bic)


# -----------------------------------------------------------------------------
# Stability across restarts (for reporting only, NOT used for Clark selection)
# -----------------------------------------------------------------------------
def compute_stability_from_W_list(
    W_list: List[np.ndarray],
) -> Tuple[float, float, int]:
    """Compute restart-to-restart similarity via Hungarian alignment of normalized W.

    W_list: list of W matrices from successful NMF runs (cols already unit norm).
    Returns (mean_similarity, std_similarity, n_pairs).
    If too few restarts (< 2), returns (np.nan, np.nan, 0).
    """
    if len(W_list) < 2:
        return np.nan, np.nan, 0

    sims: List[float] = []
    for i in range(len(W_list)):
        for j in range(i + 1, len(W_list)):
            Wi, Wj = W_list[i], W_list[j]
            sim_matrix = np.dot(Wi.T, Wj)
            sim_matrix = np.clip(sim_matrix, -1.0, 1.0)
            cost = -sim_matrix
            row_idx, col_idx = linear_sum_assignment(cost)
            paired_sim = sim_matrix[row_idx, col_idx]
            mean_pair = float(np.mean(paired_sim))
            sims.append(mean_pair)

    if not sims:
        return np.nan, np.nan, 0
    return float(np.mean(sims)), float(np.std(sims)), len(sims)


# -----------------------------------------------------------------------------
# Clark diagnostics helpers (audit only; do NOT alter selection logic)
# -----------------------------------------------------------------------------
def _json_safe_value(x: Any) -> Any:
    """JSON-serialize numeric values: finite float/int -> rounded value; NaN/inf/None -> None."""
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return None
    return round(float(x), 6) if isinstance(x, (int, float)) else x


def summarize_worst_muscles(
    muscle_names: List[str],
    per_muscle_vaf: np.ndarray,
    atol: float = 1e-8,
) -> Dict[str, Any]:
    """Return diagnostics for the worst valid muscle(s) at a given k.

    Ignores NaN values. Uses np.isclose(..., atol=atol) for tied worst muscles.
    Diagnostics are for audit only and do not alter selection.

    Returns:
        Dict with min_valid_vaf, worst_indices, worst_muscles, worst_muscles_json,
        n_worst_tied. Empty/NA if all muscles are NaN.
    """
    arr = np.atleast_1d(np.asarray(per_muscle_vaf, dtype=float))
    valid_mask = np.isfinite(arr)
    n_valid = int(np.sum(valid_mask))
    if n_valid == 0:
        return {
            "min_valid_vaf": None,
            "worst_indices": [],
            "worst_muscles": [],
            "worst_muscles_json": "[]",
            "n_worst_tied": 0,
        }
    min_vaf = float(np.nanmin(arr))
    worst_indices = np.where(valid_mask & np.isclose(arr, min_vaf, atol=atol))[0].tolist()
    n_muscles = len(muscle_names) if muscle_names else arr.shape[0]
    worst_muscles = [
        muscle_names[j] if j < n_muscles else f"muscle_{j}"
        for j in worst_indices
    ]
    return {
        "min_valid_vaf": min_vaf,
        "worst_indices": worst_indices,
        "worst_muscles": worst_muscles,
        "worst_muscles_json": json.dumps(worst_muscles),
        "n_worst_tied": len(worst_indices),
    }


def build_clark_transition_diagnostics(
    df_task: pd.DataFrame,
    improvement_thresh: float,
    atol: float = 1e-8,
) -> pd.DataFrame:
    """Build per-k transition diagnostics for Clark stopping rule audit.

    One row per k->k+1 transition. Does NOT alter selection; for observability only.
    """
    df = df_task.sort_values("k").reset_index(drop=True)
    muscle_names: List[str] = []
    if "muscle_names_json" in df.columns and len(df) > 0:
        mn = df.iloc[0].get("muscle_names_json")
        if mn:
            muscle_names = json.loads(mn) if isinstance(mn, str) else mn

    rows: List[Dict[str, Any]] = []
    k_max = int(df["k"].max())
    for i in range(len(df) - 1):
        row = df.iloc[i]
        row_next = df.iloc[i + 1]
        k = int(row["k"])
        k_next = int(row_next["k"])
        if k_next != k + 1:
            continue
        per_muscle = row.get("per_muscle_vaf")
        per_muscle_next = row_next.get("per_muscle_vaf")
        if per_muscle is None or per_muscle_next is None:
            continue
        pm = np.atleast_1d(np.asarray(per_muscle, dtype=float))
        pm_next = np.atleast_1d(np.asarray(per_muscle_next, dtype=float))
        if len(pm) != len(pm_next):
            continue
        valid_mask = np.isfinite(pm)
        n_valid = int(np.sum(valid_mask))
        if n_valid == 0:
            continue
        n_invalid = int(np.sum(~valid_mask))
        min_vaf = float(np.nanmin(pm))
        wd = summarize_worst_muscles(muscle_names, pm, atol=atol)
        worst_indices = wd["worst_indices"]
        worst_muscles = wd["worst_muscles"]
        worst_vafs = [float(pm[j]) for j in worst_indices]
        same_vafs_next = [float(pm_next[j]) if np.isfinite(pm_next[j]) else None for j in worst_indices]
        improvements = [
            float(pm_next[j] - pm[j]) if np.isfinite(pm_next[j]) else None
            for j in worst_indices
        ]
        max_imp = max((imp for imp in improvements if imp is not None), default=None)
        all_below = (
            len(improvements) == len(worst_indices)
            and all(imp is not None and imp <= improvement_thresh for imp in improvements)
        ) if improvements else False
        would_stop = all_below
        rows.append({
            "patient_id": row.get("patient_id", ""),
            "task_name": row.get("task_name", ""),
            "k": k,
            "k_next": k_next,
            "min_valid_vaf_at_k": min_vaf,
            "worst_muscles_at_k": worst_muscles,
            "worst_muscles_at_k_json": json.dumps(worst_muscles),
            "worst_vafs_at_k_json": json.dumps([_json_safe_value(v) for v in worst_vafs]),
            "same_muscles_vaf_at_kplus1_json": json.dumps([_json_safe_value(v) for v in same_vafs_next]),
            "improvements_json": json.dumps([_json_safe_value(v) for v in improvements]),
            "max_improvement_among_worst": _json_safe_value(max_imp),
            "all_worst_improvements_leq_threshold": all_below,
            "would_stop_here": would_stop,
            "n_valid_muscles_at_k": n_valid,
            "n_invalid_low_energy_muscles_at_k": row.get("n_invalid_low_energy_muscles", n_invalid),
        })
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Clark et al. 2010 selection (primary criterion only)
# -----------------------------------------------------------------------------
def _compute_clark2010_recommendation(
    df: pd.DataFrame,
    muscle_vaf_thresh: float = DEFAULT_CLARK_MUSCLE_VAF_THRESHOLD,
    improvement_thresh: float = DEFAULT_CLARK_IMPROVEMENT_THRESHOLD,
) -> Tuple[int, str, Dict[str, Any]]:
    """Select k using Clark et al. 2010 (J Neurophysiol) criterion only.

    Returns (selected_k, reason, diagnostics_dict). Diagnostics are for audit only;
    they do NOT alter selection logic.

    Clark-inspired adaptation to segmented EMG: uses per-muscle VAF only.
    Original Clark et al. 2010 used muscle-level and region-level VAF in
    gait-cycle regions; this adaptation uses per-muscle VAF only.

    Logic:
      C) Smallest k such that every muscle reaches VAF >= muscle_vaf_thresh.
      D) If C never met: Clark stopping rule - compare k and k+1; identify
         muscle(s) with lowest VAF at k (using np.isclose for ties); if ALL
         of those worst muscles fail to improve by more than improvement_thresh
         when moving from k to k+1, stop at k.
      F) If neither C nor D met by k_max: select k_max, document.

    Note: The Clark stopping rule is evaluated only when both k and k+1 have
    valid per-muscle VAF arrays. If either row is missing (e.g. NMF failed at
    that k), the selector continues scanning upward through k without applying D.

    df must have columns: k, per_muscle_vaf (list of floats per row).
    Near-zero-energy muscles yield NaN in per_muscle_vaf; Clark selection uses
    only evaluable (finite) muscles.
    """
    df_sorted = df.sort_values("k").reset_index(drop=True)
    k_max = int(df_sorted["k"].max())

    # Resolve muscle names for diagnostics (audit only)
    muscle_names: List[str] = []
    if "muscle_names_json" in df_sorted.columns and len(df_sorted) > 0:
        mn = df_sorted.iloc[0].get("muscle_names_json")
        if mn:
            muscle_names = json.loads(mn) if isinstance(mn, str) else mn

    # If no row has any evaluable muscles, document clearly
    if "n_valid_muscles_for_vaf" in df_sorted.columns:
        max_valid = df_sorted["n_valid_muscles_for_vaf"].max()
        if max_valid == 0:
            diag = {
                "selected_k": k_max,
                "selection_reason": "no_valid_muscles_for_vaf_all_k",
                "criterion_used": "F_kmax_fallback",
                "k_max": k_max,
                "notes": "no_valid_muscles_for_vaf_all_k",
            }
            return k_max, "no_valid_muscles_for_vaf_all_k", diag

    for i in range(len(df_sorted)):
        row = df_sorted.iloc[i]
        k = int(row["k"])
        per_muscle = row.get("per_muscle_vaf")
        if per_muscle is None or not isinstance(per_muscle, (list, np.ndarray)):
            continue
        per_muscle = np.atleast_1d(np.asarray(per_muscle, dtype=float))
        if len(per_muscle) == 0:
            continue

        # Ignore NaN (non-evaluable near-zero-energy) muscles; Clark applied only to valid muscles
        valid_mask = np.isfinite(per_muscle)
        n_valid = int(np.sum(valid_mask))
        if n_valid == 0:
            continue  # No evaluable muscles at this k; skip, do not treat as satisfying C

        min_vaf = float(np.nanmin(per_muscle))

        # C) All valid muscles >= threshold
        if min_vaf >= muscle_vaf_thresh:
            wd = summarize_worst_muscles(muscle_names, per_muscle)
            diag = {
                "selected_k": k,
                "selection_reason": "all_evaluable_muscles_reached_threshold",
                "criterion_used": "C_all_muscles_threshold",
                "selected_k_min_valid_vaf": min_vaf,
                "selected_k_worst_muscles": wd["worst_muscles"],
                "selected_k_worst_muscles_json": wd["worst_muscles_json"],
            }
            return k, "all_evaluable_muscles_reached_threshold", diag

        # D) Clark stopping rule: compare k and k+1; identify worst muscles only among valid ones
        if k < k_max:
            next_rows = df_sorted[df_sorted["k"] == k + 1]
            if not next_rows.empty:
                row_next = next_rows.iloc[0]
                per_muscle_next = row_next.get("per_muscle_vaf")
                if per_muscle_next is not None and isinstance(per_muscle_next, (list, np.ndarray)):
                    per_muscle_next = np.atleast_1d(np.asarray(per_muscle_next, dtype=float))
                    if len(per_muscle_next) == len(per_muscle):
                        next_valid = np.isfinite(per_muscle_next)
                        n_valid_next = int(np.sum(next_valid))
                        if n_valid_next == 0:
                            continue
                        # Worst indices: valid muscles at k with lowest valid VAF
                        worst_indices = np.where(
                            valid_mask & np.isclose(per_muscle, min_vaf)
                        )[0]
                        improvements = [
                            per_muscle_next[j] - per_muscle[j]
                            for j in worst_indices
                            if np.isfinite(per_muscle_next[j])
                        ]
                        # Stop at k only if we can evaluate ALL worst muscles and they all fail to improve
                        if (
                            len(improvements) == len(worst_indices)
                            and improvements
                            and all(imp <= improvement_thresh for imp in improvements)
                        ):
                            worst_muscles = [
                                muscle_names[j] if j < len(muscle_names) else f"M{j}"
                                for j in worst_indices
                            ]
                            worst_vafs = [float(per_muscle[j]) for j in worst_indices]
                            same_vafs_kp1 = [
                                float(per_muscle_next[j]) if np.isfinite(per_muscle_next[j]) else None
                                for j in worst_indices
                            ]
                            imp_by_muscle = {
                                m: round(float(v), 6) if v is not None and np.isfinite(v) else None
                                for m, v in zip(worst_muscles, improvements)
                            }
                            max_imp = max(improvements)
                            diag = {
                                "selected_k": k,
                                "selection_reason": "worst_muscles_improvement_below_threshold",
                                "criterion_used": "D_worst_muscle_stopping",
                                "k_stop": k,
                                "k_next": k + 1,
                                "k_stop_min_valid_vaf": min_vaf,
                                "k_stop_worst_muscles": worst_muscles,
                                "k_stop_worst_muscles_json": json.dumps(worst_muscles),
                                "k_stop_worst_indices": [int(j) for j in worst_indices],
                                "k_stop_worst_vafs": worst_vafs,
                                "k_next_same_muscles_vafs": same_vafs_kp1,
                                "worst_muscle_improvements": imp_by_muscle,
                                "worst_muscle_improvements_json": json.dumps(imp_by_muscle),
                                "max_worst_muscle_improvement": round(max_imp, 6),
                                "all_worst_muscles_below_improvement_threshold": True,
                            }
                            return k, "worst_muscles_improvement_below_threshold", diag

    # F) Fallback: select k_max
    diag = {
        "selected_k": k_max,
        "selection_reason": "reached_kmax_without_meeting_clark_criterion",
        "criterion_used": "F_kmax_fallback",
        "k_max": k_max,
        "notes": "reached_kmax_without_meeting_clark_criterion",
    }
    # Optionally add worst muscles at k_max for audit
    k_max_rows = df_sorted[df_sorted["k"] == k_max]
    if not k_max_rows.empty:
        row_km = k_max_rows.iloc[0]
        pm_km = row_km.get("per_muscle_vaf")
        if pm_km is not None:
            pm_km = np.atleast_1d(np.asarray(pm_km, dtype=float))
            if np.any(np.isfinite(pm_km)):
                wd = summarize_worst_muscles(muscle_names, pm_km)
                diag["selected_k_worst_muscles_json"] = wd["worst_muscles_json"]
                diag["selected_k_min_valid_vaf"] = wd["min_valid_vaf"]
    return k_max, "reached_kmax_without_meeting_clark_criterion", diag


# -----------------------------------------------------------------------------
# Per-task estimation
# -----------------------------------------------------------------------------
def estimate_per_task(
    patient_id: str,
    task_name: str,
    muscles: List[str],
    data_dir: Path,
    k_max: int,
    preprocessing: PreprocessingConfig,
    n_restarts: int = 10,
    base_random_state: Optional[int] = 42,
    clark_muscle_vaf_thresh: float = DEFAULT_CLARK_MUSCLE_VAF_THRESHOLD,
    clark_improvement_thresh: float = DEFAULT_CLARK_IMPROVEMENT_THRESHOLD,
    muscle_energy_epsilon: float = 1e-10,
) -> Optional[Dict[str, Any]]:
    """Estimate synergy number for one patient/task.

    Primary selector: Clark et al. 2010 (k_recommended_clark2010).
    All other metrics (AIC, BIC, stability, etc.) are for reporting only.
    """
    rec = load_emg_record(patient_id, task_name, data_dir)
    if rec is None:
        LOG.warning("Skipping %s %s: no EMG record found", patient_id, task_name)
        return None

    emg_chans = [c for c in rec.data.keys() if "EMG" in c.upper() and "ACC" not in c.upper()]
    mapping, _ = resolve_muscles_to_channels(muscles, emg_chans)
    if not mapping:
        LOG.warning("Skipping %s %s: no valid channel mapping for muscles", patient_id, task_name)
        return None

    muscle_names = [m for m in muscles if m in mapping]
    channel_order = [mapping[m] for m in muscle_names]
    times, X_full, _ = build_emg_matrix(rec, channel_order, preprocessing)
    if X_full.size == 0:
        LOG.warning("Skipping %s %s: empty EMG matrix after build", patient_id, task_name)
        return None

    segments = load_manual_segments(patient_id, task_name, data_dir)
    if not segments:
        LOG.warning("Skipping %s %s: no manual segments", patient_id, task_name)
        return None

    parent_data = []
    for s, e in segments:
        _, M_s = extract_segment_data(times, X_full, s, e)
        if M_s.size > 0:
            parent_data.append(M_s)
    if not parent_data:
        LOG.warning("Skipping %s %s: no valid segment data", patient_id, task_name)
        return None

    scale_vector = compute_per_muscle_scale(np.vstack(parent_data), preprocessing.scale_percentile)
    X_norm = apply_per_muscle_scale(X_full, scale_vector)
    _, X_concat, _, _ = build_segmented_concat(times, X_norm, segments)
    if X_concat.size == 0:
        LOG.warning("Skipping %s %s: empty concatenated matrix", patient_id, task_name)
        return None

    n_samples, n_muscles = X_concat.shape
    k_max = min(k_max, n_muscles, n_samples - 1)
    if k_max < 1:
        LOG.warning(
            "Skipping %s %s: k_max < 1 (n_samples=%d, n_muscles=%d)",
            patient_id, task_name, n_samples, n_muscles,
        )
        return None

    LOG.info("  %s | %s: fitting k=1..%d (matrix %d×%d)", patient_id, task_name, k_max, n_samples, n_muscles)
    rows: List[Dict[str, Any]] = []
    prev_global_vaf: Optional[float] = None
    for k in range(1, k_max + 1):
        LOG.info("    %s | %s: k=%d/%d", patient_id, task_name, k, k_max)
        W, H, rss, n_succ, n_fail, hit_max, all_W_list = fit_nmf_for_k(
            X_concat, k, n_restarts=n_restarts, base_random_state=base_random_state
        )
        if np.isnan(rss):
            LOG.warning(
                "%s %s k=%d: all NMF restarts failed (n_fail=%d)",
                patient_id, task_name, k, n_fail,
            )
            rows.append({
                "patient_id": patient_id,
                "task_name": task_name,
                "k": k,
                "nmf_fit_failed": True,
                "global_vaf": np.nan,
                "mean_per_muscle_vaf": np.nan,
                "min_per_muscle_vaf": np.nan,
                "per_muscle_vaf": None,
                "per_muscle_vaf_json": "",
                "delta_vaf": np.nan,
                "aic": np.nan,
                "bic": np.nan,
                "rss": np.nan,
                "stability_mean": np.nan,
                "stability_std": np.nan,
                "n_successful_restarts": 0,
                "n_failed_restarts": n_fail,
                "hit_max_iter": hit_max,
                "n_samples": n_samples,
                "n_muscles": n_muscles,
                "n_valid_muscles_for_vaf": np.nan,
                "n_invalid_low_energy_muscles": np.nan,
                "valid_muscle_mask_json": "",
                "muscle_names_json": json.dumps(muscle_names),
                "per_muscle_vaf_by_muscle_json": "",
            })
            prev_global_vaf = None
            continue

        X_hat = (W @ H).T if W.size > 0 else np.zeros_like(X_concat)
        global_vaf = compute_global_vaf(X_concat, X_hat)
        vaf_j, valid_mask = compute_per_muscle_vaf(
            X_concat, X_hat, muscle_energy_epsilon=muscle_energy_epsilon
        )
        # NaN for non-evaluable muscles; serialize as null in JSON
        per_muscle_list = [float(v) if np.isfinite(v) else None for v in vaf_j]
        n_valid = int(np.sum(valid_mask))
        n_invalid = int(np.sum(~valid_mask))
        mean_pm = float(np.nanmean(vaf_j)) if n_valid > 0 else np.nan
        min_pm = float(np.nanmin(vaf_j)) if n_valid > 0 else np.nan
        delta_vaf = (global_vaf - prev_global_vaf) if prev_global_vaf is not None else np.nan
        prev_global_vaf = global_vaf

        aic, bic = compute_aic_bic(X_concat, X_hat, k)
        stab_mean, stab_std, _ = compute_stability_from_W_list(all_W_list)
        if n_succ < 2 and np.isnan(stab_mean):
            LOG.debug(
                "%s %s k=%d: stability NaN (insufficient successful restarts)",
                patient_id, task_name, k,
            )

        rows.append({
            "patient_id": patient_id,
            "task_name": task_name,
            "k": k,
            "nmf_fit_failed": False,
            "global_vaf": global_vaf,
            "mean_per_muscle_vaf": mean_pm,
            "min_per_muscle_vaf": min_pm,
            "per_muscle_vaf": per_muscle_list,
            "per_muscle_vaf_json": json.dumps([round(v, 6) if v is not None else None for v in per_muscle_list]),
            "delta_vaf": delta_vaf,
            "aic": aic,
            "bic": bic,
            "rss": rss,
            "stability_mean": stab_mean,
            "stability_std": stab_std,
            "n_successful_restarts": n_succ,
            "n_failed_restarts": n_fail,
            "hit_max_iter": hit_max,
            "n_samples": n_samples,
            "n_muscles": n_muscles,
            "n_valid_muscles_for_vaf": n_valid,
            "n_invalid_low_energy_muscles": n_invalid,
            "valid_muscle_mask_json": json.dumps(valid_mask.tolist()),
            "muscle_names_json": json.dumps(muscle_names),
            "per_muscle_vaf_by_muscle_json": json.dumps({
                m: round(v, 6) if v is not None else None
                for m, v in zip(muscle_names, per_muscle_list)
            }),
        })

    df = pd.DataFrame(rows)
    k_clark, reason_clark, clark_diagnostics = _compute_clark2010_recommendation(
        df,
        muscle_vaf_thresh=clark_muscle_vaf_thresh,
        improvement_thresh=clark_improvement_thresh,
    )

    # Secondary descriptors (for reporting only; NOT used for selection)
    k_vaf_global = df[df["global_vaf"] >= DEFAULT_GLOBAL_VAF_THRESHOLD]["k"].min()
    if pd.isna(k_vaf_global):
        k_vaf_global = int(df["k"].max())
    k_aic = int(df.loc[df["aic"].idxmin(), "k"]) if df["aic"].notna().any() else int(df["k"].iloc[-1])
    k_bic = int(df.loc[df["bic"].idxmin(), "k"]) if df["bic"].notna().any() else int(df["k"].iloc[-1])
    best_row = df.loc[df["global_vaf"].idxmax()] if df["global_vaf"].notna().any() else df.iloc[-1]
    best_global_vaf = float(best_row["global_vaf"]) if pd.notna(best_row["global_vaf"]) else np.nan
    best_min_pm = float(best_row["min_per_muscle_vaf"]) if pd.notna(best_row["min_per_muscle_vaf"]) else np.nan
    best_stab = float(best_row["stability_mean"]) if pd.notna(best_row["stability_mean"]) else np.nan

    return {
        "patient_id": patient_id,
        "task_name": task_name,
        "metrics": rows,
        "k_recommended_clark2010": k_clark,
        "reason_for_clark2010_recommendation": reason_clark,
        "clark_diagnostics": clark_diagnostics,
        "k_recommended_vaf_global": int(k_vaf_global),
        "k_recommended_aic": k_aic,
        "k_recommended_bic": k_bic,
        "best_global_vaf": best_global_vaf,
        "best_min_per_muscle_vaf": best_min_pm,
        "best_stability": best_stab,
        "n_samples": n_samples,
        "n_muscles": n_muscles,
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(
        description="Estimate muscle synergy number (Clark et al. 2010 selection)"
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data/emg_structured"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/synergy_estimation"))
    ap.add_argument("--patients", type=str, nargs="*", help="Limit to these patient IDs")
    ap.add_argument("--k-max", type=int, default=8, help="Max synergies to test")
    ap.add_argument("--n-restarts", type=int, default=10, help="NMF restarts per k")
    ap.add_argument(
        "--clark-muscle-vaf-threshold",
        type=float,
        default=DEFAULT_CLARK_MUSCLE_VAF_THRESHOLD,
        help="Clark 2010: muscle VAF threshold (default 0.90)",
    )
    ap.add_argument(
        "--clark-improvement-threshold",
        type=float,
        default=DEFAULT_CLARK_IMPROVEMENT_THRESHOLD,
        help="Clark 2010: min improvement for worst muscle when adding synergy (default 0.05)",
    )
    ap.add_argument(
        "--muscle-energy-epsilon",
        type=float,
        default=1e-10,
        help="Per-muscle total energy threshold below which per-muscle VAF is treated as non-evaluable (NaN)",
    )
    ap.add_argument(
        "--global-vaf-threshold",
        type=float,
        default=DEFAULT_GLOBAL_VAF_THRESHOLD,
        help="Reporting only (not for Clark selection)",
    )
    ap.add_argument(
        "--stability-threshold",
        type=float,
        default=DEFAULT_STABILITY_THRESHOLD,
        help="Reporting only (not for Clark selection)",
    )
    ap.add_argument("--random-state", type=int, default=42, help="Base random state for NMF")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    _setup_logging(args.verbose)

    LOG.info("Starting synergy estimation (Clark et al. 2010)")
    LOG.info("  data-dir=%s  out-dir=%s  k-max=%d  n-restarts=%d", args.data_dir, args.out_dir, args.k_max, args.n_restarts)
    LOG.info("  Clark muscle VAF threshold=%.2f  improvement threshold=%.2f", args.clark_muscle_vaf_threshold, args.clark_improvement_threshold)

    muscles = DEFAULT_MUSCLES.copy()
    preprocessing = PreprocessingConfig()

    patients = discover_patients_with_data(args.data_dir)
    if args.patients:
        patients = [p for p in patients if p in args.patients]
    LOG.info("Discovered %d patient(s): %s", len(patients), patients[:5] if len(patients) > 5 else patients)

    all_metrics: List[Dict[str, Any]] = []
    results: List[Dict[str, Any]] = []
    total_tasks = sum(len(discover_tasks_for_patient(p, args.data_dir)) for p in patients)
    LOG.info("Total tasks to process: %d", total_tasks)

    task_idx = 0
    for patient_id in patients:
        tasks = discover_tasks_for_patient(patient_id, args.data_dir)
        for task_name in tasks:
            task_idx += 1
            LOG.info("Processing %s | %s (%d/%d)", patient_id, task_name, task_idx, total_tasks)
            res = estimate_per_task(
                patient_id,
                task_name,
                muscles,
                args.data_dir,
                k_max=args.k_max,
                preprocessing=preprocessing,
                n_restarts=args.n_restarts,
                base_random_state=args.random_state,
                clark_muscle_vaf_thresh=args.clark_muscle_vaf_threshold,
                clark_improvement_thresh=args.clark_improvement_threshold,
                muscle_energy_epsilon=getattr(args, "muscle_energy_epsilon", 1e-10),
            )
            if res is not None:
                results.append(res)
                all_metrics.extend(res["metrics"])

    if not results:
        LOG.warning("No data processed. Check --data-dir and manual segments.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = args.out_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    # Metrics CSV: flatten per_muscle_vaf for export (use json column)
    metrics_export = []
    for row in all_metrics:
        r = dict(row)
        if "per_muscle_vaf" in r and isinstance(r["per_muscle_vaf"], list):
            r["per_muscle_vaf"] = r.get("per_muscle_vaf_json", json.dumps(r["per_muscle_vaf"]))
        metrics_export.append(r)
    metrics_df = pd.DataFrame(metrics_export)
    if "per_muscle_vaf_json" in metrics_df.columns:
        metrics_df = metrics_df.drop(columns=["per_muscle_vaf_json"], errors="ignore")
    metrics_df.to_csv(summary_dir / "synergy_metrics_by_k.csv", index=False)

    # Recommendations CSV (Clark 2010 primary) with diagnostic columns
    recs = []
    for res in results:
        diag = res.get("clark_diagnostics") or {}
        rec = {
            "patient_id": res["patient_id"],
            "task_name": res["task_name"],
            "k_recommended_clark2010": res["k_recommended_clark2010"],
            "reason_for_clark2010_recommendation": res["reason_for_clark2010_recommendation"],
            "criterion_used": diag.get("criterion_used"),
            "selected_k_min_valid_vaf": diag.get("selected_k_min_valid_vaf"),
            "selected_k_worst_muscles_json": diag.get("selected_k_worst_muscles_json"),
            "k_stop": diag.get("k_stop"),
            "k_next": diag.get("k_next"),
            "k_stop_worst_muscles_json": diag.get("k_stop_worst_muscles_json"),
            "worst_muscle_improvements_json": diag.get("worst_muscle_improvements_json"),
            "max_worst_muscle_improvement": diag.get("max_worst_muscle_improvement"),
            "all_worst_muscles_below_improvement_threshold": diag.get(
                "all_worst_muscles_below_improvement_threshold"
            ),
            "k_recommended_vaf_global": res["k_recommended_vaf_global"],
            "k_recommended_aic": res["k_recommended_aic"],
            "k_recommended_bic": res["k_recommended_bic"],
            "best_global_vaf": res["best_global_vaf"],
            "best_min_per_muscle_vaf": res["best_min_per_muscle_vaf"],
            "best_stability": res["best_stability"],
            "n_samples": res["n_samples"],
            "n_muscles": res["n_muscles"],
        }
        recs.append(rec)
    pd.DataFrame(recs).to_csv(summary_dir / "synergy_recommendations.csv", index=False)

    # Clark stopping diagnostics: one row per task per k->k+1 transition
    clark_transition_rows: List[Dict[str, Any]] = []
    for res in results:
        df_task = pd.DataFrame(res["metrics"])
        trans_df = build_clark_transition_diagnostics(
            df_task,
            improvement_thresh=args.clark_improvement_threshold,
        )
        if not trans_df.empty:
            clark_transition_rows.extend(trans_df.to_dict("records"))
    if clark_transition_rows:
        pd.DataFrame(clark_transition_rows).to_csv(
            summary_dir / "clark_stopping_diagnostics.csv", index=False
        )

    # Clark stop driver summary: one row per task, muscles that drove the selection
    driver_rows: List[Dict[str, Any]] = []
    for res in results:
        diag = res.get("clark_diagnostics") or {}
        criterion = diag.get("criterion_used", "")
        selected_k = res["k_recommended_clark2010"]
        if criterion == "D_worst_muscle_stopping":
            stop_json = diag.get("k_stop_worst_muscles_json")
            imp_json = diag.get("worst_muscle_improvements_json")
            min_vaf = diag.get("k_stop_min_valid_vaf")
            max_imp = diag.get("max_worst_muscle_improvement")
        elif criterion == "C_all_muscles_threshold":
            stop_json = diag.get("selected_k_worst_muscles_json")
            imp_json = None
            min_vaf = diag.get("selected_k_min_valid_vaf")
            max_imp = None
        else:
            stop_json = diag.get("selected_k_worst_muscles_json")
            imp_json = None
            min_vaf = diag.get("selected_k_min_valid_vaf")
            max_imp = None
        driver_rows.append({
            "patient_id": res["patient_id"],
            "task_name": res["task_name"],
            "selected_k": selected_k,
            "criterion_used": criterion,
            "stop_driver_muscles_json": stop_json,
            "stop_driver_improvements_json": imp_json,
            "stop_driver_min_valid_vaf": min_vaf,
            "stop_driver_max_improvement": max_imp,
            "n_stop_driver_muscles": (
                len(json.loads(stop_json)) if stop_json else 0
            ),
        })
    pd.DataFrame(driver_rows).to_csv(
        summary_dir / "clark_stop_driver_summary.csv", index=False
    )

    # JSON run settings
    run_config = {
        "selection_method": "Clark et al. 2010 (J Neurophysiol)",
        "data_dir": str(args.data_dir),
        "out_dir": str(args.out_dir),
        "k_max": args.k_max,
        "n_restarts": args.n_restarts,
        "clark_muscle_vaf_threshold": args.clark_muscle_vaf_threshold,
        "clark_improvement_threshold": args.clark_improvement_threshold,
        "muscle_energy_epsilon": getattr(args, "muscle_energy_epsilon", 1e-10),
        "global_vaf_threshold_reporting_only": args.global_vaf_threshold,
        "stability_threshold_reporting_only": args.stability_threshold,
        "random_state": args.random_state,
        "n_tasks_processed": len(results),
    }
    with open(summary_dir / "synergy_estimation_config.json", "w") as f:
        json.dump(run_config, f, indent=2)

    LOG.info("Estimation complete. Run B2_plot_synergy_estimation.py to generate figures.")
    print(f"Synergy estimation written to {args.out_dir}")
    print(f"  Selection: Clark et al. 2010 (per-muscle VAF only)")
    print(f"  Tasks processed: {len(results)}")
    print(f"  Metrics: {summary_dir / 'synergy_metrics_by_k.csv'}")
    print(f"  Recommendations: {summary_dir / 'synergy_recommendations.csv'}")
    print(f"  Clark stopping diagnostics: {summary_dir / 'clark_stopping_diagnostics.csv'}")
    print(f"  Clark stop driver summary: {summary_dir / 'clark_stop_driver_summary.csv'}")
    print(f"  Config: {summary_dir / 'synergy_estimation_config.json'}")


if __name__ == "__main__":
    main()
