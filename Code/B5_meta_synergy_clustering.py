#!/usr/bin/env python3
"""B5_meta_synergy_clustering.py

Meta-cluster individual patient synergies into cohort-level meta-synergies.
Supports variable task-specific k (from B2) and produces fixed-K cohort-level
comparable representation via nearest-cluster assignment + H aggregation.

Products saved to results_dir/meta_synergy_clustering/:
  - meta_centroids.csv           : K x muscles (meta-synergy weight vectors)
  - cluster_assignment.csv       : per-task individual -> meta mapping (any k_i)
  - meta_task_summary_metrics.csv : fixed-K task-level features for B7
  - H_meta_windows.npz (per task) : fixed-K aggregated activations
  - confusion_matrices/          : NxM similarity matrices (n_syn x K)
  - h_paired_summary_clustered.csv
  - task_summary_by_cluster.csv  : task-level metrics remapped by cluster
"""

from __future__ import annotations

import argparse
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

PROFILE_NPTS = 101
EPS = 1e-12
DEFAULT_N_PERM = 5000


def _curve_diff_stat(curves_a: List[np.ndarray], curves_b: List[np.ndarray]) -> float:
    """Mean absolute difference between the two mean curves (test statistic for T0 vs T1)."""
    if not curves_a or not curves_b:
        return 0.0
    mean_a = np.mean(np.array(curves_a), axis=0)
    mean_b = np.mean(np.array(curves_b), axis=0)
    return float(np.mean(np.abs(mean_a - mean_b)))


def _permutation_test_t0_vs_t1(
    curves_t0: List[np.ndarray],
    curves_t1: List[np.ndarray],
    n_perm: int = DEFAULT_N_PERM,
    random_state: Optional[int] = None,
) -> Tuple[float, float]:
    """
    Permutation test comparing two sets of activation curves (T0 vs T1).
    Null: curves come from the same distribution. Test statistic: mean absolute difference
    between mean curves. Returns (observed_stat, p_value).
    """
    n0, n1 = len(curves_t0), len(curves_t1)
    if n0 < 2 or n1 < 2:
        return 0.0, 1.0
    pooled = np.array(curves_t0 + curves_t1)
    observed = _curve_diff_stat(curves_t0, curves_t1)
    rng = np.random.default_rng(random_state)
    count_ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(n0 + n1)
        s0 = pooled[perm[:n0]]
        s1 = pooled[perm[n0:]]
        stat = _curve_diff_stat(list(s0), list(s1))
        if stat >= observed:
            count_ge += 1
    p = (1 + count_ge) / (1 + n_perm)
    return observed, p


def run_permutation_tests_meta_synergy_affected(
    curves_by_key: Dict[Tuple[str, str, int], List[np.ndarray]],
    out_dir: Path,
    n_clusters: int,
    n_perm: int = DEFAULT_N_PERM,
    random_state: Optional[int] = None,
) -> Dict[Tuple[str, int], float]:
    """
    Run permutation test T0 vs T1 for each (aff_group, cluster).
    Saves results to meta_synergy_permutation_tests.csv.
    Returns dict mapping (aff_group, cluster_idx) -> p_value.
    """
    rows: List[Dict[str, Any]] = []
    p_values: Dict[Tuple[str, int], float] = {}
    for aff_group in ("more_affected", "less_affected"):
        for k in range(n_clusters):
            if k == 0:
                continue  # C0 excluded from permutation test comparison
            key_t0 = (aff_group, "T0", k)
            key_t1 = (aff_group, "T1", k)
            curves_t0 = curves_by_key.get(key_t0, [])
            curves_t1 = curves_by_key.get(key_t1, [])
            if curves_t0 and curves_t1:
                stat_obs, p_val = _permutation_test_t0_vs_t1(
                    curves_t0, curves_t1, n_perm=n_perm, random_state=random_state
                )
                p_values[(aff_group, k)] = p_val
                rows.append({
                    "affected_group": aff_group,
                    "cluster": f"C{k}",
                    "cluster_idx": k,
                    "n_T0": len(curves_t0),
                    "n_T1": len(curves_t1),
                    "stat_observed": stat_obs,
                    "p_value": p_val,
                    "n_permutations": n_perm,
                })
            else:
                p_values[(aff_group, k)] = 1.0
                rows.append({
                    "affected_group": aff_group,
                    "cluster": f"C{k}",
                    "cluster_idx": k,
                    "n_T0": len(curves_t0),
                    "n_T1": len(curves_t1),
                    "stat_observed": np.nan,
                    "p_value": np.nan,
                    "n_permutations": n_perm,
                })
    if rows:
        df = pd.DataFrame(rows)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "meta_synergy_permutation_tests.csv"
        df.to_csv(out_path, index=False)
        print(f"  Saved {out_path.name}")
    return p_values


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Version-safe trapezoidal integration for NumPy 1.x/2.x."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


# -----------------------------------------------------------------------------
# Discovery and loading (mirrors B4_report_emg_features)
# -----------------------------------------------------------------------------


def parse_session_condition(task_name: str) -> Tuple[str, str]:
    task_upper = task_name.upper()
    session = "T0" if "T0" in task_upper else "T1" if "T1" in task_upper else ""
    condition = "SN" if "SN" in task_upper else "DS" if "DS" in task_upper else ""
    return session, condition


def discover_task_dirs(results_dir: Path) -> List[Tuple[str, str, str, str, Path]]:
    discovered: List[Tuple[str, str, str, str, Path]] = []
    if not results_dir.exists():
        return discovered
    for patient_dir in sorted(results_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        if patient_dir.name.startswith("plots_") or patient_dir.name == "meta_synergy_clustering":
            continue
        patient_id = patient_dir.name
        for task_dir in sorted(patient_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            session, condition = parse_session_condition(task_dir.name)
            if not session or not condition:
                continue
            discovered.append((patient_id, task_dir.name, session, condition, task_dir))
    return discovered


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def read_npz_array(path: Path, key: str) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        npz = np.load(path, allow_pickle=True)
        return npz[key] if key in npz.files else None
    except Exception:
        return None


def _read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Excel file."""
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path, engine="openpyxl")
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def load_demographics(path: Optional[Path]) -> pd.DataFrame:
    """Load demographics CSV/Excel. Must have id (or patient_id) column."""
    if path is None or not path.exists():
        return pd.DataFrame()
    df = _read_table(path)
    if df.empty:
        return pd.DataFrame()
    # Find id column (case-insensitive, flexible naming)
    id_col = None
    for c in df.columns:
        nc = str(c).strip().lower()
        if nc in ("id", "patient_id", "patient id", "codice", "numero"):
            id_col = c
            break
    if id_col is None:
        return pd.DataFrame()
    if id_col != "id":
        df = df.rename(columns={id_col: "id"})
    return df


def _normalize_side(val: Any) -> str:
    """Normalize DX/SX. Accepts str, strips, uppercases."""
    v = str(val).strip().upper()
    if not v:
        return ""
    if v in ("DX", "DESTRO", "RIGHT", "D"):
        return "DX"
    if v in ("SX", "SINISTRO", "LEFT", "L"):
        return "SX"
    if len(v) == 2 and v[0] in "DS" and v[1] in "X":
        return "DX" if "D" in v else "SX"
    return v


def _condition_to_performed_side(condition: str) -> str:
    """DS -> DX, SN -> SX."""
    c = str(condition).strip().upper()
    if c == "DS":
        return "DX"
    if c == "SN":
        return "SX"
    return c if c in ("DX", "SX") else ""


AFFECTED_SIDE_ALIASES = [
    "lato piu colpito", "lato_piu_colpito", "lato più colpito", "lato piu colpito",
    "latopiucolpito", "affected_side", "affected side",
]


def _normalize_col_for_match(s: str) -> str:
    """Normalize string for column matching (handles accents, encoding mojibake)."""
    s = str(s).strip().lower()
    # NFKD decomposition + ASCII: "più" -> "piu", handles replacement char etc.
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.replace(" ", "_").replace("-", "_")
    return s


def _find_affected_column(df: pd.DataFrame) -> Optional[str]:
    """Find column for affected side (lato più colpito). Uses 'colpito' as key (avoids 'id' matching 'affected_side')."""
    for col in df.columns:
        norm = _normalize_col_for_match(col)
        raw = str(col).strip().lower()
        # Require "colpito" – distinctive keyword; prevents "id" falsely matching "affected_side"
        if "colpito" not in norm and "colpito" not in raw:
            continue
        # Exclude id-like columns (safety)
        if norm in ("id", "patient_id", "patient id", "codice", "numero"):
            continue
        return col
    return None


def build_patient_cond_to_affected(demo_path: Optional[Path]) -> Dict[Tuple[str, str], str]:
    """
    Build (patient_id, condition) -> "more_affected" | "less_affected".
    DS -> performed DX, SN -> performed SX. more_affected when performed == affected_side.
    Returns empty dict if demographics missing or invalid.
    """
    out: Dict[Tuple[str, str], str] = {}
    demo = load_demographics(demo_path)
    if demo.empty:
        return out
    aff_col = _find_affected_column(demo)
    if not aff_col or aff_col not in demo.columns:
        return out
    for _, row in demo.iterrows():
        pid = str(row.get("id", "")).strip()
        if not pid:
            continue
        affected = _normalize_side(row.get(aff_col, ""))
        if affected not in ("DX", "SX"):
            continue
        for cond in ("SN", "DS"):
            performed = _condition_to_performed_side(cond)
            if not performed:
                continue
            out[(pid, cond)] = "more_affected" if performed == affected else "less_affected"
    return out


def _diagnose_demographics_failure(demo_path: Path) -> None:
    """Print helpful diagnostics when build_patient_cond_to_affected returns empty."""
    demo = load_demographics(demo_path)
    if demo.empty:
        print(f"  Skipped affected-side time course: could not load {demo_path} (empty or unreadable)")
        return
    aff_col = _find_affected_column(demo)
    if not aff_col:
        print(f"  Skipped affected-side time course: no 'lato più colpito' column in {demo_path.name} (cols: {list(demo.columns)[:6]}...)")
        return
    n_valid = 0
    for _, row in demo.iterrows():
        pid = str(row.get("id", "")).strip()
        affected = _normalize_side(row.get(aff_col, ""))
        if pid and affected in ("DX", "SX"):
            n_valid += 1
    if n_valid == 0:
        print(f"  Skipped affected-side time course: no rows with valid id + DX/SX in '{aff_col}'")
        return
    print(f"  Skipped affected-side time course: unexpected (found {n_valid} valid rows)")


# -----------------------------------------------------------------------------
# Cosine similarity and Hungarian alignment
# -----------------------------------------------------------------------------


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Pairwise cosine similarity between columns of A and B. A, B are (n_features, n_cols)."""
    def norm_cols(M: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(M, axis=0, keepdims=True)
        n[n < 1e-12] = 1.0
        return M / n
    return norm_cols(A).T @ norm_cols(B)


def hungarian_match_to_reference(W_new: np.ndarray, W_ref: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorder columns of W_new to maximize similarity to W_ref.
    Returns (permutation, similarity_matrix).
    permutation[k] = index in W_ref that W_new column k was matched to.
    similarity_matrix: n_new x n_ref, cell = cosine sim between new col i and ref col j.
    Used for optional diagnostics only; primary assignment uses assign_task_synergies_to_meta_clusters.
    """
    sim = cosine_similarity_matrix(W_new, W_ref)
    cost = -sim
    row_idx, col_idx = linear_sum_assignment(cost)
    perm = np.zeros(W_new.shape[1], dtype=int)
    perm[row_idx] = col_idx
    return perm, sim


def assign_task_synergies_to_meta_clusters(
    W_task: np.ndarray,
    meta_centroids: np.ndarray,
) -> pd.DataFrame:
    """
    Assign each original synergy column to the best-matching meta-cluster via argmax cosine similarity.
    Supports variable k_i: multiple synergies may map to same cluster, some clusters may be absent.

    W_task: shape (n_muscles, k_i) - columns are synergy vectors
    meta_centroids: shape (K, n_muscles)
    Returns DataFrame with: syn_idx, meta_cluster, similarity_to_cluster
    """
    if W_task.size == 0 or meta_centroids.size == 0:
        return pd.DataFrame(columns=["syn_idx", "meta_cluster", "similarity_to_cluster"])
    # W_task (n_muscles, k_i), centroids (K, n_muscles) -> need (n_muscles, K) for sim
    W_cols = W_task  # (n_muscles, k_i)
    centroids_T = np.asarray(meta_centroids, dtype=np.float64).T  # (n_muscles, K)
    sim = cosine_similarity_matrix(W_cols, centroids_T)  # (k_i, K)
    rows = []
    for syn_idx in range(sim.shape[0]):
        best_c = int(np.argmax(sim[syn_idx, :]))
        sim_val = float(sim[syn_idx, best_c])
        rows.append({"syn_idx": syn_idx, "meta_cluster": best_c, "similarity_to_cluster": sim_val})
    return pd.DataFrame(rows)


def build_meta_h_windows(
    H_windows: np.ndarray,
    assignment_df: pd.DataFrame,
    n_clusters: int,
    agg: str = "sum",
) -> np.ndarray:
    """
    Aggregate original H activations into fixed-K meta-activation representation.
    H_windows: (n_windows, k_i, n_samples)
    assignment_df: must have syn_idx, meta_cluster (syn_idx -> cluster mapping)
    Returns H_meta_windows: (n_windows, K, n_samples)
    """
    n_win, k_i, n_samp = H_windows.shape
    H_meta = np.zeros((n_win, n_clusters, n_samp), dtype=np.float64)
    for _, row in assignment_df.iterrows():
        syn_idx = int(row["syn_idx"])
        meta_c = int(row["meta_cluster"])
        if 0 <= syn_idx < k_i and 0 <= meta_c < n_clusters:
            H_meta[:, meta_c, :] += H_windows[:, syn_idx, :]
    return H_meta


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Version-safe trapezoidal integration for NumPy 1.x/2.x."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def _meta_synergy_activation_summary(H_meta_windows: np.ndarray) -> Dict[str, float]:
    """
    Compute task-level mean (over windows) of mean, max, auc, centroid, peak_time per meta-synergy.
    H_meta_windows: (n_windows, K, n_samples).
    Returns mean over windows of each metric (analogous to B3 mean_synergy_* from window metrics).
    """
    n_win, n_clust, n_samp = H_meta_windows.shape
    phase = np.linspace(0.0, 1.0, n_samp, dtype=np.float64) if n_samp > 0 else np.array([])
    out = {}
    for c in range(n_clust):
        means_w, maxs_w, aucs_w, centroids_w, peaks_w = [], [], [], [], []
        for w in range(n_win):
            h = np.asarray(H_meta_windows[w, c, :], dtype=np.float64)
            if h.size == 0:
                continue
            means_w.append(float(np.mean(h)))
            maxs_w.append(float(np.max(h)))
            aucs_w.append(_trapezoid_integral(h, phase) if n_samp > 1 else 0.0)
            total = np.sum(h)
            centroids_w.append(float(np.sum(phase * h) / total) if total > EPS else np.nan)
            peaks_w.append(float(phase[int(np.argmax(h))]) if n_samp > 0 else np.nan)
        out[f"mean_meta_synergy_{c}_mean"] = float(np.nanmean(means_w)) if means_w else np.nan
        out[f"mean_meta_synergy_{c}_max"] = float(np.nanmean(maxs_w)) if maxs_w else np.nan
        out[f"mean_meta_synergy_{c}_auc"] = float(np.nanmean(aucs_w)) if aucs_w else np.nan
        out[f"mean_meta_synergy_{c}_centroid"] = float(np.nanmean([x for x in centroids_w if np.isfinite(x)])) if any(np.isfinite(x) for x in centroids_w) else np.nan
        out[f"mean_meta_synergy_{c}_peak_time"] = float(np.nanmean([x for x in peaks_w if np.isfinite(x)])) if any(np.isfinite(x) for x in peaks_w) else np.nan
    return out


def _compute_presence_and_simmax(
    task_assign: pd.DataFrame,
    n_clusters: int,
) -> Dict[str, Any]:
    """
    For each meta-cluster, compute:
      presence_Ck = 1 if at least one synergy assigned to Ck, else 0
      simmax_Ck   = max similarity_to_cluster among synergies assigned to Ck, else NaN if none

    Returns dict with mean_meta_synergy_0_presence, mean_meta_synergy_0_simmax, ... for B7 compatibility.
    """
    out: Dict[str, Any] = {}
    for c in range(n_clusters):
        mask = task_assign["meta_cluster"] == c
        out[f"mean_meta_synergy_{c}_presence"] = 1 if mask.any() else 0
        if mask.any() and "similarity_to_cluster" in task_assign.columns:
            out[f"mean_meta_synergy_{c}_simmax"] = float(task_assign.loc[mask, "similarity_to_cluster"].max())
        else:
            out[f"mean_meta_synergy_{c}_simmax"] = np.nan
    return out


# -----------------------------------------------------------------------------
# Meta-clustering pipeline
# -----------------------------------------------------------------------------


def load_and_stack_w_vectors(
    results_dir: Path,
    return_weights: bool = False,
) -> Tuple[np.ndarray, List[str], pd.DataFrame, Optional[np.ndarray]]:
    """
    Load all W_global matrices, stack columns into (N_vectors, n_muscles).
    Returns (stacked_vectors, muscle_order, metadata_df, weights_or_None).
    metadata_df: patient_id, task_name, session, condition, syn_idx.
    If return_weights: weights[i] = 1/k_i for vector i (each task contributes total weight 1).
    Uses intersection of muscles across all tasks.
    """
    discovered = discover_task_dirs(results_dir)
    all_W: List[pd.DataFrame] = []
    all_meta: List[Dict[str, Any]] = []

    # First pass: collect all W and muscle sets
    muscle_sets: List[set] = []
    for patient_id, task_name, session, condition, task_dir in discovered:
        w_path = task_dir / "W_global.csv"
        if not w_path.exists():
            continue
        try:
            w_df = pd.read_csv(w_path, index_col=0)
        except Exception:
            continue
        if w_df.empty or w_df.shape[1] == 0:
            continue
        all_W.append((patient_id, task_name, session, condition, w_df))
        muscle_sets.append(set(w_df.index))

    if not muscle_sets:
        return np.array([]).reshape(0, 0), [], pd.DataFrame(), None

    muscles_common = sorted(set.intersection(*muscle_sets))
    if not muscles_common:
        return np.array([]).reshape(0, 0), [], pd.DataFrame(), None

    # Second pass: extract vectors with common muscle order
    vectors = []
    weights_list: List[float] = []
    for patient_id, task_name, session, condition, w_df in all_W:
        w_aligned = w_df.reindex(muscles_common).fillna(0).to_numpy(dtype=np.float64)
        k_i = w_aligned.shape[1]
        w_i = 1.0 / k_i if return_weights else 1.0
        for k in range(k_i):
            vec = w_aligned[:, k]
            vec_norm = np.linalg.norm(vec)
            if vec_norm < 1e-12:
                vec_norm = 1.0
            vectors.append(vec / vec_norm)
            if return_weights:
                weights_list.append(w_i)
            all_meta.append({
                "patient_id": patient_id,
                "task_name": task_name,
                "session": session,
                "condition": condition,
                "syn_idx": k,
            })

    stack = np.vstack(vectors)
    meta_df = pd.DataFrame(all_meta)
    weights = np.array(weights_list, dtype=np.float64) if return_weights else None
    return stack, muscles_common, meta_df, weights


def compute_meta_clusters(
    stacked: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """K-means on stacked vectors. Returns (centroids, labels)."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(stacked, sample_weight=sample_weight)
    centroids = kmeans.cluster_centers_
    # L2-normalize centroids for consistency
    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    centroids = centroids / norms
    return centroids, labels


def _trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Version-safe trapezoidal integration."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def compute_per_task_assignment_and_confusion(
    results_dir: Path,
    meta_centroids: np.ndarray,
    muscles: List[str],
    n_clusters: int,
) -> Tuple[pd.DataFrame, Dict[Tuple[str, str, str], np.ndarray], Dict[Tuple[str, str, str], np.ndarray]]:
    """
    For each task, assign original synergies to meta-clusters via nearest-cluster (argmax similarity).
    Supports variable k_i: tasks with any valid k_i >= 1 are processed.

    Returns:
        assignment_df: full cluster_assignment.csv (patient_id, task_name, session, condition,
                       syn_idx, meta_cluster, similarity_to_cluster, n_synergies_task, n_meta_clusters)
        confusion_matrices: (n_syn x K) similarity matrix per (patient_id, condition, session)
        mappings: (patient_id, condition, session) -> perm of length n_syn, perm[syn_idx] = meta_cluster
    """
    discovered = discover_task_dirs(results_dir)
    assignment_rows: List[Dict[str, Any]] = []
    confusion_matrices: Dict[Tuple[str, str, str], np.ndarray] = {}
    mappings: Dict[Tuple[str, str, str], np.ndarray] = {}

    for patient_id, task_name, session, condition, task_dir in discovered:
        w_path = task_dir / "W_global.csv"
        if not w_path.exists():
            continue
        try:
            w_df = pd.read_csv(w_path, index_col=0)
        except Exception:
            continue
        w_df = w_df.reindex(muscles).fillna(0)
        W = w_df.to_numpy(dtype=np.float64)
        k_i = W.shape[1]
        if k_i < 1:
            continue

        assign_df = assign_task_synergies_to_meta_clusters(W, meta_centroids)
        if assign_df.empty:
            continue

        sim = cosine_similarity_matrix(W, meta_centroids.T)
        key = (patient_id, condition, session)
        confusion_matrices[key] = sim
        perm = assign_df["meta_cluster"].to_numpy(dtype=int)
        mappings[key] = perm

        for _, row in assign_df.iterrows():
            assignment_rows.append({
                "patient_id": patient_id,
                "task_name": task_name,
                "session": session,
                "condition": condition,
                "syn_idx": int(row["syn_idx"]),
                "meta_cluster": int(row["meta_cluster"]),
                "similarity_to_cluster": float(row["similarity_to_cluster"]),
                "n_synergies_task": k_i,
                "n_meta_clusters": n_clusters,
            })

    assignment_df = pd.DataFrame(assignment_rows)
    return assignment_df, confusion_matrices, mappings


def build_h_paired_summary_individual(task_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build paired T0/T1 summary for individual synergy H metrics (S0, S1, S2) - same as B4."""
    if task_summary_df.empty:
        return pd.DataFrame()
    pattern = re.compile(r"mean_synergy_(\d+)_(auc|centroid|peak_time|mean|max)$")
    cols_found = [(int(m.group(1)), m.group(2)) for c in task_summary_df.columns for m in [pattern.match(c)] if m]
    if not cols_found:
        return pd.DataFrame()
    synergy_ids = sorted({s for s, _ in cols_found})
    metrics = sorted({m for _, m in cols_found})
    rows: List[Dict[str, Any]] = []
    for (patient_id, condition), grp in task_summary_df.groupby(["patient_id", "condition"]):
        t0_row = grp[grp["session"] == "T0"]
        t1_row = grp[grp["session"] == "T1"]
        if t0_row.empty or t1_row.empty:
            continue
        t0_row = t0_row.iloc[0]
        t1_row = t1_row.iloc[0]
        for syn_id in synergy_ids:
            row = {"patient_id": patient_id, "condition": condition, "synergy_label": f"S{syn_id}"}
            for metric in metrics:
                col = f"mean_synergy_{syn_id}_{metric}"
                if col not in task_summary_df.columns:
                    continue
                v0 = t0_row[col]
                v1 = t1_row[col]
                if pd.notna(v0) or pd.notna(v1):
                    row[f"{metric}_T0"] = float(v0) if pd.notna(v0) else np.nan
                    row[f"{metric}_T1"] = float(v1) if pd.notna(v1) else np.nan
                    if pd.notna(v0) and pd.notna(v1):
                        row[f"{metric}_delta"] = float(v1) - float(v0)
            if any(k.endswith("_T0") or k.endswith("_T1") for k in row):
                rows.append(row)
    return pd.DataFrame(rows)


def build_h_paired_summary_clustered(
    task_summary_df: pd.DataFrame,
    mappings: Dict[Tuple[str, str, str], np.ndarray],
    n_meta_clusters: int,
) -> pd.DataFrame:
    """
    Rebuild h_paired_summary using cluster labels. For each (patient, condition), we have
    T0 and T1 rows. perm maps syn_idx -> meta_cluster. Iterate over fixed K meta-clusters.
    When multiple synergies map to same cluster, average their metric values.
    """
    if task_summary_df.empty:
        return pd.DataFrame()
    pattern = re.compile(r"mean_synergy_(\d+)_(auc|centroid|peak_time|mean|max)$")
    cols_found = [(int(m.group(1)), m.group(2)) for c in task_summary_df.columns for m in [pattern.match(c)] if m]
    if not cols_found:
        return pd.DataFrame()
    synergy_ids = sorted({s for s, _ in cols_found})
    metrics = sorted({m for _, m in cols_found})

    rows: List[Dict[str, Any]] = []
    for (patient_id, condition), grp in task_summary_df.groupby(["patient_id", "condition"]):
        t0_row = grp[grp["session"] == "T0"]
        t1_row = grp[grp["session"] == "T1"]
        if t0_row.empty or t1_row.empty:
            continue
        t0_row = t0_row.iloc[0]
        t1_row = t1_row.iloc[0]

        key_t0 = (patient_id, condition, "T0")
        key_t1 = (patient_id, condition, "T1")
        perm_t0 = mappings.get(key_t0)
        perm_t1 = mappings.get(key_t1)
        if perm_t0 is None or perm_t1 is None:
            continue

        for meta_c in range(n_meta_clusters):
            syn_t0 = np.where(perm_t0 == meta_c)[0]
            syn_t1 = np.where(perm_t1 == meta_c)[0]
            if len(syn_t0) == 0 and len(syn_t1) == 0:
                continue

            row = {"patient_id": patient_id, "condition": condition, "synergy_label": f"C{meta_c}"}
            for metric in metrics:
                v0_list = [t0_row.get(f"mean_synergy_{s}_{metric}", np.nan) for s in syn_t0 if f"mean_synergy_{s}_{metric}" in task_summary_df.columns]
                v1_list = [t1_row.get(f"mean_synergy_{s}_{metric}", np.nan) for s in syn_t1 if f"mean_synergy_{s}_{metric}" in task_summary_df.columns]
                v0 = float(np.nanmean(v0_list)) if v0_list else np.nan
                v1 = float(np.nanmean(v1_list)) if v1_list else np.nan
                if pd.notna(v0) or pd.notna(v1):
                    row[f"{metric}_T0"] = v0
                    row[f"{metric}_T1"] = v1
                    if pd.notna(v0) and pd.notna(v1):
                        row[f"{metric}_delta"] = float(v1) - float(v0)
            if any(k.endswith("_T0") or k.endswith("_T1") for k in row):
                rows.append(row)
    return pd.DataFrame(rows)


def build_task_summary_by_cluster(
    task_summary_df: pd.DataFrame,
    mappings: Dict[Tuple[str, str, str], np.ndarray],
    n_meta_clusters: int,
) -> pd.DataFrame:
    """Remap task_summary metrics from synergy index to cluster label. Supports variable k_i."""
    if task_summary_df.empty:
        return pd.DataFrame()
    pattern = re.compile(r"mean_synergy_(\d+)_(auc|centroid|peak_time|mean|max)$")
    base_cols = [c for c in task_summary_df.columns if not pattern.match(c)]

    out_rows: List[Dict[str, Any]] = []
    for _, r in task_summary_df.iterrows():
        pid = r.get("patient_id")
        cond = r.get("condition")
        sess = r.get("session")
        if pd.isna(pid) or pd.isna(cond) or pd.isna(sess):
            continue
        key = (str(pid), str(cond), str(sess))
        perm = mappings.get(key)
        if perm is None:
            continue

        for meta_c in range(n_meta_clusters):
            syn_idxs = np.where(perm == meta_c)[0]
            if len(syn_idxs) == 0:
                continue
            row = {c: r[c] for c in base_cols}
            row["synergy_label"] = f"C{meta_c}"
            row["original_syn_idx"] = int(syn_idxs[0]) if len(syn_idxs) == 1 else f"avg({syn_idxs.tolist()})"
            for metric in ["auc", "centroid", "peak_time", "mean", "max"]:
                vals = [r.get(f"mean_synergy_{s}_{metric}", np.nan) for s in syn_idxs if f"mean_synergy_{s}_{metric}" in task_summary_df.columns]
                row[f"{metric}"] = float(np.nanmean(vals)) if vals and np.any(np.isfinite(vals)) else np.nan
            out_rows.append(row)
    return pd.DataFrame(out_rows)


# -----------------------------------------------------------------------------
# Statistics (mirrors B4_report_emg_features)
# -----------------------------------------------------------------------------


def _paired_ttest(v0: np.ndarray, v1: np.ndarray) -> float:
    mask = np.isfinite(v0) & np.isfinite(v1)
    v0, v1 = np.asarray(v0)[mask], np.asarray(v1)[mask]
    if len(v0) < 2:
        return np.nan
    _, p = scipy_stats.ttest_rel(v0, v1)
    return float(p)


def _paired_wilcoxon(v0: np.ndarray, v1: np.ndarray) -> float:
    mask = np.isfinite(v0) & np.isfinite(v1)
    v0, v1 = np.asarray(v0)[mask], np.asarray(v1)[mask]
    if len(v0) < 2:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                _, p = scipy_stats.wilcoxon(v0, v1, method="exact" if len(v0) <= 20 else "asymptotic")
            except TypeError:
                _, p = scipy_stats.wilcoxon(v0, v1)
        return float(p)
    except Exception:
        return np.nan


def _paired_effect_size(v0: np.ndarray, v1: np.ndarray) -> float:
    mask = np.isfinite(v0) & np.isfinite(v1)
    diff = np.asarray(v1)[mask] - np.asarray(v0)[mask]
    if diff.size < 2:
        return np.nan
    sd = float(np.std(diff, ddof=1))
    if sd <= EPS:
        return np.nan
    return float(np.mean(diff) / sd)


def _apply_fdr(stats_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    if stats_df.empty or "p_value" not in stats_df.columns:
        return stats_df
    out_parts = []
    for _, group in stats_df.groupby(list(group_cols), dropna=False):
        group = group.copy()
        p = group["p_value"].to_numpy(dtype=float)
        finite_mask = np.isfinite(p)
        q = np.full(len(group), np.nan, dtype=float)
        if np.any(finite_mask):
            idx = np.where(finite_mask)[0]
            p_finite = p[finite_mask]
            order = np.argsort(p_finite)
            ranked = p_finite[order]
            m = len(ranked)
            bh = ranked * m / np.arange(1, m + 1)
            bh = np.minimum.accumulate(bh[::-1])[::-1]
            q_vals = np.clip(bh, 0.0, 1.0)
            q[idx[order]] = q_vals
        group["q_value"] = q
        out_parts.append(group)
    return pd.concat(out_parts, ignore_index=True)


def compute_dynamic_paired_stats(
    df: pd.DataFrame,
    metric_col_base: str,
    label_col: str,
    family: str,
    title_prefix: str,
) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    t0_col = f"{metric_col_base}_T0"
    t1_col = f"{metric_col_base}_T1"
    for (condition, label), subset in df.groupby(["condition", label_col]):
        vals = subset[[t0_col, t1_col]].dropna()
        if vals.empty:
            continue
        v0 = vals[t0_col].to_numpy(dtype=float)
        v1 = vals[t1_col].to_numpy(dtype=float)
        n_val = len(vals)
        p_ttest = _paired_ttest(v0, v1)
        p_wilcox = _paired_wilcoxon(v0, v1)
        p_primary = p_ttest  # always use paired t-test
        rows.append({
            "family": family,
            "metric_key": metric_col_base,
            "label": label,
            "title": f"{title_prefix}: {label}",
            "condition": condition,
            "n": int(n_val),
            "mean_t0": float(np.mean(v0)),
            "mean_t1": float(np.mean(v1)),
            "mean_delta": float(np.mean(v1 - v0)),
            "std_delta": float(np.std(v1 - v0, ddof=1)) if len(v0) > 1 else np.nan,
            "p_value": p_primary,
            "p_ttest": p_ttest,
            "p_wilcoxon": p_wilcox,
            "effect_size_dz": _paired_effect_size(v0, v1),
        })
    stats_df = pd.DataFrame(rows)
    return _apply_fdr(stats_df, group_cols=["family", "condition"])


def compute_synergy_h_stats(h_paired_df: pd.DataFrame, family: str) -> pd.DataFrame:
    """Compute paired T0 vs T1 stats for all synergy H metrics (auc, centroid, peak_time, mean, max)."""
    metric_specs = [
        ("auc", "Synergy AUC"),
        ("centroid", "Centroid"),
        ("peak_time", "Peak time"),
        ("mean", "Mean activation"),
        ("max", "Max activation"),
    ]
    parts = []
    for metric_col_base, title_prefix in metric_specs:
        if f"{metric_col_base}_T0" not in h_paired_df.columns or f"{metric_col_base}_T1" not in h_paired_df.columns:
            continue
        part = compute_dynamic_paired_stats(
            h_paired_df,
            metric_col_base=metric_col_base,
            label_col="synergy_label",
            family=family,
            title_prefix=title_prefix,
        )
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def _format_pvalue(p: float) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def _format_qvalue(q: float) -> str:
    if not np.isfinite(q):
        return "q=n/a"
    return "q<0.001" if q < 0.001 else f"q={q:.3f}"


def _extract_stats_row(stats_df: pd.DataFrame, family: str, metric_key: str, condition: str, label: str) -> Optional[pd.Series]:
    if stats_df.empty:
        return None
    subset = stats_df[
        (stats_df["family"] == family)
        & (stats_df["metric_key"] == metric_key)
        & (stats_df["condition"] == condition)
        & (stats_df["label"] == label)
    ]
    if subset.empty:
        return None
    return subset.iloc[0]


def _annotate_stats(ax: plt.Axes, stats_row: Optional[pd.Series]) -> None:
    if stats_row is None:
        return
    n = int(stats_row["n"])
    mean_delta = stats_row.get("mean_delta", np.nan)
    delta_str = f"Δ={mean_delta:+.3f}" if np.isfinite(mean_delta) else "Δ=n/a"
    dz = stats_row.get("effect_size_dz", np.nan)
    dz_str = f"dz={dz:.2f}" if np.isfinite(dz) else "dz=n/a"
    parts = [f"n={n}", delta_str, dz_str, _format_pvalue(stats_row["p_value"]), _format_qvalue(stats_row.get("q_value", np.nan))]
    ax.text(0.98, 0.98, " | ".join(parts), transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))


def _plot_paired_slope(
    ax: plt.Axes,
    v0: np.ndarray,
    v1: np.ndarray,
    ylabel: str,
    title: str,
    stats_row: Optional[pd.Series] = None,
) -> None:
    mask = np.isfinite(v0) & np.isfinite(v1)
    v0, v1 = np.asarray(v0)[mask], np.asarray(v1)[mask]
    if len(v0) == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return
    n = len(v0)
    x0 = np.full(n, 0.0) + np.linspace(-0.06, 0.06, n)
    x1 = np.full(n, 1.0) + np.linspace(-0.06, 0.06, n)
    for i in range(n):
        ax.plot([x0[i], x1[i]], [v0[i], v1[i]], color="0.7", linewidth=1, zorder=1)
    ax.scatter(x0, v0, color="#3182bd", s=26, zorder=3)
    ax.scatter(x1, v1, color="#e6550d", s=26, zorder=3)
    means = [np.mean(v0), np.mean(v1)]
    sems = [scipy_stats.sem(v0) if n > 1 else 0.0, scipy_stats.sem(v1) if n > 1 else 0.0]
    ax.errorbar([0, 1], means, yerr=sems, color="black", linewidth=2, marker="D", markersize=5, capsize=4, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["T0", "T1"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    _annotate_stats(ax, stats_row)


def plot_synergy_h_summary(
    h_paired_df: pd.DataFrame,
    h_stats_df: pd.DataFrame,
    out_dir: Path,
    conditions: List[str],
    family: str,
    subdir: str,
    title_prefix: str,
    file_prefix: str,
    dpi: int = 150,
) -> None:
    """Paired slope plots for synergy H metrics with p-values, q-values, effect sizes."""
    if h_paired_df.empty:
        return
    metrics = [
        ("auc", "Synergy AUC", "AUC"),
        ("centroid", "Centroid (phase)", "Centroid"),
        ("peak_time", "Peak time (phase)", "Peak time"),
        ("mean", "Mean activation", "Mean"),
        ("max", "Max activation", "Max"),
    ]
    available = [(m, yl, t) for m, yl, t in metrics if f"{m}_T0" in h_paired_df.columns and f"{m}_T1" in h_paired_df.columns]
    if not available:
        return
    synergies = sorted(h_paired_df["synergy_label"].unique().tolist())
    h_dir = out_dir / "figures" / subdir
    h_dir.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        cond_df = h_paired_df[h_paired_df["condition"] == condition]
        if cond_df.empty:
            continue
        fig, axes = plt.subplots(len(synergies), len(available), figsize=(4.5 * len(available), 4 * len(synergies)), squeeze=False)
        for i, syn in enumerate(synergies):
            syn_df = cond_df[cond_df["synergy_label"] == syn]
            for j, (metric, ylabel, title) in enumerate(available):
                stats_row = _extract_stats_row(h_stats_df, family, metric, condition, label=syn)
                _plot_paired_slope(
                    axes[i, j],
                    syn_df[f"{metric}_T0"].to_numpy(dtype=float),
                    syn_df[f"{metric}_T1"].to_numpy(dtype=float),
                    ylabel,
                    f"{syn}: {title}" if i == 0 else title,
                    stats_row,
                )
        fig.suptitle(f"{title_prefix} ({condition})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(h_dir / f"paired_{file_prefix}_{condition}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_confusion_matrix_summary(
    confusion_matrices: Dict[Tuple[str, str, str], np.ndarray],
    out_dir: Path,
    conditions: List[str],
    dpi: int = 150,
) -> None:
    """Create figures for confusion matrices (n_syn x n_meta).

    1. Cohort mean confusion matrix per (condition, session)
    2. Grid of small heatmaps per patient for each condition/session
    """
    if not confusion_matrices:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Infer dimensions from first matrix (n_syn, n_meta)
    sample_shape = next(iter(confusion_matrices.values())).shape
    n_syn, n_meta = sample_shape[0], sample_shape[1]
    row_labels = [f"S{k}" for k in range(n_syn)]
    col_labels = [f"C{k}" for k in range(n_meta)]

    # 1. Mean confusion matrix per (condition, session)
    sessions = ["T0", "T1"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    axes = axes.flatten()
    idx = 0
    for condition in conditions:
        for session in sessions:
            ax = axes[idx]
            stack = []
            for (pid, cond, sess), sim in confusion_matrices.items():
                if cond == condition and sess == session and sim.shape == sample_shape:
                    stack.append(sim)
            if stack:
                mean_sim = np.nanmean(np.stack(stack), axis=0)
                im = ax.imshow(mean_sim, vmin=0, vmax=1, cmap="Blues", aspect="equal")
                for i in range(mean_sim.shape[0]):
                    for j in range(mean_sim.shape[1]):
                        ax.text(j, i, f"{mean_sim[i, j]:.2f}", ha="center", va="center", fontsize=11)
                plt.colorbar(im, ax=ax, shrink=0.8, label="Cosine similarity")
            ax.set_xticks(range(n_meta))
            ax.set_xticklabels(col_labels)
            ax.set_yticks(range(n_syn))
            ax.set_yticklabels(row_labels)
            ax.set_xlabel("Meta-synergy")
            ax.set_ylabel("Individual synergy")
            ax.set_title(f"{condition} {session} (cohort mean, n={len(stack)})")
            idx += 1
    fig.suptitle("Cohort mean individual↔meta similarity", fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "confusion_matrix_cohort_mean.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    # 2. Grid of per-patient confusion matrices for each (condition, session)
    for condition in conditions:
        for session in sessions:
            keys = [(p, c, s) for (p, c, s) in confusion_matrices if c == condition and s == session]
            if not keys:
                continue
            patients = sorted({k[0] for k in keys})
            n_pat = len(patients)
            n_cols = min(4, n_pat)
            n_rows = (n_pat + n_cols - 1) // n_cols
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(3 * n_cols, 3 * n_rows))
            if n_rows == 1 and n_cols == 1:
                axes = np.array([[axes]])
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            for i, patient_id in enumerate(patients):
                row, col = i // n_cols, i % n_cols
                ax = axes[row, col]
                key = (patient_id, condition, session)
                sim = confusion_matrices.get(key)
                if sim is not None:
                    nr, nc = sim.shape[0], sim.shape[1]
                    ax.imshow(sim, vmin=0, vmax=1, cmap="Blues", aspect="equal")
                    for ri in range(nr):
                        for cj in range(nc):
                            ax.text(cj, ri, f"{sim[ri, cj]:.2f}", ha="center", va="center", fontsize=9)
                    ax.set_xticks(range(nc))
                    ax.set_xticklabels([f"C{k}" for k in range(nc)], fontsize=8)
                    ax.set_yticks(range(nr))
                    ax.set_yticklabels([f"S{k}" for k in range(nr)], fontsize=8)
                ax.set_title(patient_id, fontsize=9)
            for j in range(len(patients), n_rows * n_cols):
                row, col = j // n_cols, j % n_cols
                axes[row, col].axis("off")
            fig.suptitle(f"Individual↔meta similarity – {condition} {session}", fontsize=11)
            fig.tight_layout()
            fig.savefig(fig_dir / f"confusion_matrix_grid_{condition}_{session}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)


def plot_meta_synergy_spatial(
    meta_centroids: np.ndarray,
    muscles: List[str],
    out_dir: Path,
    dpi: int = 150,
) -> None:
    """Heatmap of meta-synergy weights (muscles x clusters)."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, max(6, len(muscles) * 0.35)))
    W_display = meta_centroids.T
    im = ax.imshow(W_display, aspect="auto", cmap="viridis", vmin=0)
    ax.set_yticks(range(len(muscles)))
    ax.set_yticklabels(muscles, fontsize=9)
    ax.set_xticks(range(W_display.shape[1]))
    ax.set_xticklabels([f"C{k}" for k in range(W_display.shape[1])])
    ax.set_xlabel("Meta-synergy")
    ax.set_ylabel("Muscle")
    plt.colorbar(im, ax=ax, label="Weight")
    ax.set_title("Meta-synergy muscle weights (spatial patterns)")
    fig.tight_layout()
    fig.savefig(fig_dir / "meta_synergy_spatial_heatmap.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_meta_synergy_radar(
    meta_centroids: np.ndarray,
    muscles: List[str],
    out_dir: Path,
    dpi: int = 150,
) -> None:
    """Radar plot of meta-synergy weights. Each polygon = one meta-synergy (C0, C1, C2)."""
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    n_muscles = len(muscles)
    n_clusters = meta_centroids.shape[0]
    angles = np.linspace(0, 2 * np.pi, n_muscles, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection="polar"))
    colors = plt.cm.Set1(np.linspace(0, 0.8, n_clusters))

    for k in range(n_clusters):
        values = meta_centroids[k, :].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=f"C{k}", color=colors[k])
        ax.fill(angles, values, alpha=0.2, color=colors[k])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(muscles, fontsize=8, ha="center")
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
    ax.set_title("Meta-synergy muscle weights", pad=20, fontsize=12)
    fig.tight_layout()
    fig.savefig(fig_dir / "meta_synergy_radar.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def _load_presence_by_task(assignment_path: Path) -> Dict[Tuple[str, str, str, str], Set[int]]:
    """
    Load cluster_assignment.csv and build presence map: (patient_id, task_name, session, condition)
    -> set of meta_cluster indices that have at least one synergy assigned in that task.
    Returns {} if file not found.
    """
    if not assignment_path.exists():
        return {}
    try:
        df = pd.read_csv(assignment_path)
    except Exception:
        return {}
    if "patient_id" not in df.columns or "meta_cluster" not in df.columns:
        return {}
    out: Dict[Tuple[str, str, str, str], Set[int]] = {}
    for _, row in df.iterrows():
        key = (str(row["patient_id"]), str(row["task_name"]), str(row["session"]), str(row["condition"]))
        out.setdefault(key, set()).add(int(row["meta_cluster"]))
    return out


def load_h_curves_by_cluster(
    results_dir: Path,
    mappings: Dict[Tuple[str, str, str], np.ndarray],
    n_phase_pts: int = PROFILE_NPTS,
    out_dir: Optional[Path] = None,
) -> Tuple[Dict[Tuple[str, str, str], List[np.ndarray]], Dict[Tuple[str, str, str], int], Dict[Tuple[str, str, str], Set[str]]]:
    """
    Load H_meta_windows from each task (aggregated meta-activation, same source as meta_task_summary_metrics).
    Resample each window's H_meta to common phase grid, collect curves per (condition, session, cluster).
    Uses H_meta (sum of synergies per cluster) so amplitude matches emg_meta_C*_mean in Excel.

    When out_dir is provided and cluster_assignment.csv exists, only includes curves where the
    meta-synergy was actually present in that task (at least one individual synergy assigned to it).
    Excludes zero-vector instances from tasks where that meta-cluster had no synergies.

    Returns (curves_by_key, n_subjects_by_key) where n_subjects_by_key gives unique subject count per key.
    """
    phase_new = np.linspace(0.0, 1.0, n_phase_pts)
    curves_by_key: Dict[Tuple[str, str, str], List[np.ndarray]] = {}
    subjects_by_key: Dict[Tuple[str, str, str], Set[str]] = {}

    assignment_path = (out_dir / "cluster_assignment.csv") if out_dir else None
    presence = _load_presence_by_task(assignment_path) if assignment_path else {}
    use_presence = bool(presence)

    for patient_id, task_name, session, condition, task_dir in discover_task_dirs(results_dir):
        H_meta = read_npz_array(task_dir / "H_meta_windows.npz", "H_meta")
        if H_meta is None or H_meta.ndim != 3:
            continue
        n_windows, n_clusters, n_samples = H_meta.shape
        phase_old = np.linspace(0.0, 1.0, n_samples) if n_samples > 0 else np.array([])
        task_key = (str(patient_id), str(task_name), str(session), str(condition))
        present_clusters = presence.get(task_key, set(range(n_clusters))) if use_presence else set(range(n_clusters))
        pid = str(patient_id)

        for win in range(n_windows):
            for meta_c in range(n_clusters):
                if meta_c not in present_clusters:
                    continue
                key = (condition, session, meta_c)
                h_win = np.asarray(H_meta[win, meta_c, :], dtype=np.float64)
                if n_samples > 1:
                    curve = np.interp(phase_new, phase_old, h_win)
                else:
                    curve = np.full(n_phase_pts, float(h_win[0]) if h_win.size else 0.0)
                curves_by_key.setdefault(key, []).append(curve)
                subjects_by_key.setdefault(key, set()).add(pid)

    n_subjects_by_key = {k: len(s) for k, s in subjects_by_key.items()}
    return curves_by_key, n_subjects_by_key, subjects_by_key


def load_h_curves_by_cluster_affected(
    results_dir: Path,
    patient_cond_to_affected: Dict[Tuple[str, str], str],
    n_phase_pts: int = PROFILE_NPTS,
    out_dir: Optional[Path] = None,
) -> Tuple[Dict[Tuple[str, str, int], List[np.ndarray]], Dict[Tuple[str, str, int], int]]:
    """
    Load H_meta_windows and tag curves by (affected_group, session, cluster).
    Pools across SN/DS: more_affected = DS from DX-affected + SN from SX-affected (matches B7 stats).
    Key: (affected_group, session, meta_c).

    When out_dir is provided and cluster_assignment.csv exists, only includes curves where the
    meta-synergy was actually present in that task. Excludes zero-vector instances from tasks
    where that meta-cluster had no synergies assigned.

    Returns (curves_by_key, n_subjects_by_key) where n_subjects_by_key gives unique subject count per key.
    """
    phase_new = np.linspace(0.0, 1.0, n_phase_pts)
    curves_by_key: Dict[Tuple[str, str, int], List[np.ndarray]] = {}
    subjects_by_key: Dict[Tuple[str, str, int], Set[str]] = {}

    assignment_path = (out_dir / "cluster_assignment.csv") if out_dir else None
    presence = _load_presence_by_task(assignment_path) if assignment_path else {}
    use_presence = bool(presence)

    for patient_id, task_name, session, condition, task_dir in discover_task_dirs(results_dir):
        aff_group = patient_cond_to_affected.get((str(patient_id), condition))
        if not aff_group:
            continue
        H_meta = read_npz_array(task_dir / "H_meta_windows.npz", "H_meta")
        if H_meta is None or H_meta.ndim != 3:
            continue
        n_windows, n_clusters, n_samples = H_meta.shape
        phase_old = np.linspace(0.0, 1.0, n_samples) if n_samples > 0 else np.array([])
        task_key = (str(patient_id), str(task_name), str(session), str(condition))
        present_clusters = presence.get(task_key, set(range(n_clusters))) if use_presence else set(range(n_clusters))
        pid = str(patient_id)

        for win in range(n_windows):
            for meta_c in range(n_clusters):
                if meta_c not in present_clusters:
                    continue
                key = (aff_group, session, meta_c)
                h_win = np.asarray(H_meta[win, meta_c, :], dtype=np.float64)
                if n_samples > 1:
                    curve = np.interp(phase_new, phase_old, h_win)
                else:
                    curve = np.full(n_phase_pts, float(h_win[0]) if h_win.size else 0.0)
                curves_by_key.setdefault(key, []).append(curve)
                subjects_by_key.setdefault(key, set()).add(pid)

    n_subjects_by_key = {k: len(s) for k, s in subjects_by_key.items()}
    return curves_by_key, n_subjects_by_key


def plot_meta_synergy_timecourse_by_affected(
    results_dir: Path,
    patient_cond_to_affected: Dict[Tuple[str, str], str],
    out_dir: Path,
    conditions: List[str],
    n_clusters: int,
    dpi: int = 150,
    p_values: Optional[Dict[Tuple[str, int], float]] = None,
) -> None:
    """
    Plot meta-synergy activation time courses split by more_affected and less_affected.
    Pools across SN/DS (matches B7 stats_affected_paired). Each figure: one panel,
    T0 (solid) vs T1 (dashed) for each cluster.
    If p_values is provided, adds permutation test p-values (T0 vs T1) per cluster in legend.
    """
    curves_by_key, n_subjects_by_key = load_h_curves_by_cluster_affected(
        results_dir, patient_cond_to_affected, out_dir=out_dir
    )
    if not curves_by_key:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    phase = np.linspace(0, 1, PROFILE_NPTS) * 100
    clusters = [f"C{k}" for k in range(n_clusters)]
    colors = plt.cm.Set1(np.linspace(0, 0.8, n_clusters))

    for aff_group in ("more_affected", "less_affected"):
        fig, ax = plt.subplots(figsize=(8, 5))
        for k, cluster in enumerate(clusters):
            for session, linestyle in [("T0", "-"), ("T1", "--")]:
                key = (aff_group, session, k)
                curves = curves_by_key.get(key, [])
                if curves:
                    curves_arr = np.array(curves)
                    mean_c = np.mean(curves_arr, axis=0)
                    n_subj = n_subjects_by_key.get(key, 0)
                    lab = f"{cluster} {session} (n={n_subj})"
                    ax.plot(
                        phase,
                        mean_c,
                        color=colors[k],
                        linestyle=linestyle,
                        label=lab,
                        linewidth=1.5,
                    )
        # Add permutation test p-values (T0 vs T1 per cluster) as text box
        if p_values is not None:
            p_text_lines = []
            for k in range(n_clusters):
                p = p_values.get((aff_group, k))
                if p is not None and not np.isnan(p):
                    p_str = f"p={p:.3g}" if p >= 0.001 else "p<0.001"
                    p_text_lines.append(f"{clusters[k]} T0 vs T1: {p_str}")
            if p_text_lines:
                p_text = "\n".join(p_text_lines)
                ax.text(
                    0.02, 0.02, f"Permutation test (T0 vs T1):\n{p_text}",
                    transform=ax.transAxes, fontsize=7, verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
                )
        ax.set_ylim(0, 1)
        ax.set_xlabel("Normalized phase (%)")
        ax.set_ylabel("Activation (a.u.)")
        ax.set_title(f"Meta-synergy activation: T0 (solid) vs T1 (dashed) – {aff_group.replace('_', ' ')}")
        ax.legend(loc="upper right", ncol=2, fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        fname = f"meta_synergy_timecourse_{aff_group}.png"
        fig.savefig(fig_dir / fname, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


def plot_meta_synergy_timecourse(
    results_dir: Path,
    mappings: Dict[Tuple[str, str, str], np.ndarray],
    out_dir: Path,
    conditions: List[str],
    n_clusters: int = 3,
    dpi: int = 150,
) -> None:
    """Plot meta-synergy activation time courses from H_meta (same source as meta_task_summary_metrics).

    Loads H_meta_windows.npz from each task, resamples to common phase grid, plots mean across
    windows and patients. Amplitude matches emg_meta_C*_mean in Excel. Compare with stats_condition_paired (SN/DS).
    """
    curves_by_key, n_subjects_by_key, subjects_by_key = load_h_curves_by_cluster(
        results_dir, mappings, out_dir=out_dir
    )
    if not curves_by_key:
        return
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    phase = np.linspace(0, 1, PROFILE_NPTS) * 100
    clusters = [f"C{k}" for k in range(n_clusters)]
    colors = plt.cm.Set1(np.linspace(0, 0.8, n_clusters))

    for condition in conditions:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        for sess_idx, session in enumerate(["T0", "T1"]):
            ax = axes[sess_idx]
            for k, cluster in enumerate(clusters):
                key = (condition, session, k)
                curves = curves_by_key.get(key, [])
                if curves:
                    curves_arr = np.array(curves)
                    mean_c = np.mean(curves_arr, axis=0)
                    n_subj = n_subjects_by_key.get(key, 0)
                    ax.plot(phase, mean_c, color=colors[k], label=f"{cluster} (n={n_subj})", linewidth=2)
            ax.set_ylim(0, 1)
            ax.set_xlabel("Normalized phase (%)")
            ax.set_ylabel("Activation (a.u.)")
            ax.set_title(f"{condition} {session}")
            ax.legend(loc="upper right", fontsize=9)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
        fig.suptitle(f"Meta-synergy activation time course – {condition}", fontsize=11)
        fig.tight_layout()
        fig.savefig(fig_dir / f"meta_synergy_timecourse_{condition}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)

    # Combined T0 vs T1
    fig, ax = plt.subplots(figsize=(8, 5))
    for k, cluster in enumerate(clusters):
        for sess_idx, (session, linestyle) in enumerate([("T0", "-"), ("T1", "--")]):
            curves_list = []
            subjects_pooled: Set[str] = set()
            for cond in conditions:
                key = (cond, session, k)
                curves_list.extend(curves_by_key.get(key, []))
                subjects_pooled.update(subjects_by_key.get(key, set()))
            if curves_list:
                curves_arr = np.array(curves_list)
                mean_c = np.mean(curves_arr, axis=0)
                n_subj = len(subjects_pooled)
                ax.plot(phase, mean_c, color=colors[k], linestyle=linestyle, label=f"{cluster} {session} (n={n_subj})", linewidth=1.5)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Normalized phase (%)")
    ax.set_ylabel("Activation (a.u.)")
    ax.set_title("Meta-synergy activation: T0 (solid) vs T1 (dashed)")
    ax.legend(loc="upper right", ncol=2, fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "meta_synergy_timecourse_T0_vs_T1.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Meta-cluster synergies, compute per-patient confusion matrices, rebuild h_paired with cluster labels"
    )
    ap.add_argument("--results-dir", type=Path, default=Path("results/synergies"))
    ap.add_argument("--out-dir", type=Path, default=None, help="Default: results-dir/meta_synergy_clustering")
    ap.add_argument("--n-clusters", type=int, default=4)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--conditions", type=str, nargs="*", default=["SN", "DS"])
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument(
        "--weight-by-task",
        action="store_true",
        help="Weight each vector by 1/k_i so each task contributes total weight 1 (reduces influence of tasks with many synergies)",
    )
    ap.add_argument(
        "--demographic",
        type=Path,
        default=None,
        help="Demographics CSV/Excel (e.g. demografica.xlsx) with id and lato più colpito for affected-side split plots",
    )
    ap.add_argument(
        "--n-permutations",
        type=int,
        default=DEFAULT_N_PERM,
        help=f"Number of permutations for T0 vs T1 curve comparison (default: {DEFAULT_N_PERM})",
    )
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = args.out_dir or (results_dir / "meta_synergy_clustering")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "confusion_matrices").mkdir(parents=True, exist_ok=True)

    # 1. Load and stack W vectors
    print("Loading and stacking W vectors...")
    stacked, muscles, meta_df, sample_weight = load_and_stack_w_vectors(
        results_dir, return_weights=args.weight_by_task
    )
    if stacked.size == 0:
        print("No W_global files found. Exiting.")
        return
    print(f"  Stacked shape: {stacked.shape}, muscles: {len(muscles)}")
    if sample_weight is not None:
        print(f"  Using per-task weights (each task contributes total weight 1)")

    # 2. K-means meta-clustering
    print("Running K-means meta-clustering...")
    centroids, labels = compute_meta_clusters(
        stacked,
        n_clusters=args.n_clusters,
        random_state=args.random_state,
        sample_weight=sample_weight,
    )
    meta_df["meta_cluster"] = labels
    print(f"  Centroids shape: {centroids.shape}")

    # 3. Save meta centroids
    centroids_df = pd.DataFrame(centroids.T, index=muscles, columns=[f"C{k}" for k in range(centroids.shape[0])])
    centroids_df.to_csv(out_dir / "meta_centroids.csv")
    meta_df.to_csv(out_dir / "stack_metadata.csv", index=False)

    # 4. Per-task assignment (nearest-cluster), confusion matrices, mappings
    print("Computing per-task assignment and confusion matrices...")
    assignment_df, confusion_matrices, mappings = compute_per_task_assignment_and_confusion(
        results_dir, centroids, muscles, n_clusters=args.n_clusters
    )
    assignment_df.to_csv(out_dir / "cluster_assignment.csv", index=False)

    # 4b. Build H_meta per task, save H_meta_windows.npz, compute meta_task_summary_metrics
    print("Building fixed-K meta-H and meta_task_summary_metrics...")
    meta_task_rows: List[Dict[str, Any]] = []
    for patient_id, task_name, session, condition, task_dir in discover_task_dirs(results_dir):
        H = read_npz_array(task_dir / "H_windows.npz", "H")
        if H is None or H.ndim != 3:
            continue
        task_assign = assignment_df[
            (assignment_df["patient_id"] == patient_id)
            & (assignment_df["task_name"] == task_name)
            & (assignment_df["session"] == session)
            & (assignment_df["condition"] == condition)
        ]
        if task_assign.empty:
            continue
        H_meta = build_meta_h_windows(H, task_assign, args.n_clusters)
        window_centers = read_npz_array(task_dir / "H_windows.npz", "window_centers")
        if window_centers is None:
            window_centers = np.arange(H_meta.shape[0], dtype=np.float64)
        np.savez_compressed(
            task_dir / "H_meta_windows.npz",
            H_meta=H_meta,
            window_centers=window_centers,
            patient_id=np.array(patient_id, dtype=object),
            task_name=np.array(task_name, dtype=object),
            session=np.array(session, dtype=object),
            condition=np.array(condition, dtype=object),
            n_meta_clusters=np.array(args.n_clusters),
        )
        summary = _meta_synergy_activation_summary(H_meta)
        presence_simmax = _compute_presence_and_simmax(task_assign, args.n_clusters)
        row = {
            "patient_id": patient_id,
            "task_name": task_name,
            "session": session,
            "condition": condition,
            "n_synergies_task": int(task_assign["n_synergies_task"].iloc[0]),
            "n_meta_clusters": args.n_clusters,
            **summary,
            **presence_simmax,
        }
        meta_task_rows.append(row)
    if meta_task_rows:
        pd.DataFrame(meta_task_rows).to_csv(out_dir / "meta_task_summary_metrics.csv", index=False)
        print(f"  Saved meta_task_summary_metrics.csv ({len(meta_task_rows)} rows)")

    for (patient_id, condition, session), sim in confusion_matrices.items():
        sim_df = pd.DataFrame(
            sim,
            index=[f"S{k}" for k in range(sim.shape[0])],
            columns=[f"C{k}" for k in range(sim.shape[1])],
        )
        fname = f"confusion_{patient_id}_{condition}_{session}.csv"
        sim_df.to_csv(out_dir / "confusion_matrices" / fname)

    # 5. Load task summary and rebuild h_paired with cluster labels
    print("Rebuilding h_paired_summary with cluster labels...")
    root_task = read_csv_if_exists(results_dir / "patient_task_metrics.csv")
    task_records = []
    for patient_id, task_name, session, condition, task_dir in discover_task_dirs(results_dir):
        ts = read_csv_if_exists(task_dir / "task_summary_metrics.csv")
        if ts.empty:
            continue
        if "patient_id" not in ts.columns:
            ts.insert(0, "patient_id", patient_id)
        if "task_name" not in ts.columns:
            ts.insert(1, "task_name", task_name)
        ts["session"] = session
        ts["condition"] = condition
        task_records.append(ts)
    task_summary_df = root_task if root_task is not None and not root_task.empty else pd.concat(task_records, ignore_index=True)

    if task_summary_df.empty:
        print("No task summary found. Skipping h_paired rebuild.")
        h_paired_clustered = pd.DataFrame()
    else:
        h_paired_clustered = build_h_paired_summary_clustered(
            task_summary_df, mappings, n_meta_clusters=args.n_clusters
        )
        h_paired_clustered.to_csv(out_dir / "h_paired_summary_clustered.csv", index=False)
        print(f"  Saved h_paired_summary_clustered.csv ({len(h_paired_clustered)} rows)")

        task_by_cluster = build_task_summary_by_cluster(
            task_summary_df, mappings, n_meta_clusters=args.n_clusters
        )
        if not task_by_cluster.empty:
            task_by_cluster.to_csv(out_dir / "task_summary_by_cluster.csv", index=False)
            print(f"  Saved task_summary_by_cluster.csv ({len(task_by_cluster)} rows)")

        # Statistical results: individual synergy (S0, S1, S2) and meta-synergy (C0, C1, C2)
        (out_dir / "summary_tables").mkdir(parents=True, exist_ok=True)
        h_paired_individual = build_h_paired_summary_individual(task_summary_df)
        if not h_paired_individual.empty:
            h_paired_individual.to_csv(out_dir / "h_paired_summary_individual.csv", index=False)
            indiv_stats = compute_synergy_h_stats(h_paired_individual, family="individual_synergy_h")
            if not indiv_stats.empty:
                indiv_stats.to_csv(out_dir / "summary_tables" / "individual_synergy_h_stats.csv", index=False)
                print(f"  Saved individual_synergy_h_stats.csv ({len(indiv_stats)} rows)")
                plot_synergy_h_summary(
                    h_paired_individual, indiv_stats, out_dir, args.conditions,
                    family="individual_synergy_h", subdir="individual_synergy_h_summaries",
                    title_prefix="Individual synergy H timing/amplitude", file_prefix="individual_synergy", dpi=args.dpi,
                )
        h_stats = compute_synergy_h_stats(h_paired_clustered, family="meta_synergy_h")
        if not h_stats.empty:
            h_stats.to_csv(out_dir / "summary_tables" / "meta_synergy_h_stats.csv", index=False)
            print(f"  Saved meta_synergy_h_stats.csv ({len(h_stats)} rows)")
            plot_synergy_h_summary(
                h_paired_clustered, h_stats, out_dir, args.conditions,
                family="meta_synergy_h", subdir="meta_synergy_h_summaries",
                title_prefix="Meta-synergy H timing/amplitude", file_prefix="meta_synergy", dpi=args.dpi,
            )

    # 6. Create figures: confusion matrices and meta-synergy time courses
    print("Creating figures...")
    plot_confusion_matrix_summary(confusion_matrices, out_dir, args.conditions, dpi=args.dpi)
    plot_meta_synergy_spatial(centroids, muscles, out_dir, dpi=args.dpi)
    plot_meta_synergy_radar(centroids, muscles, out_dir, dpi=args.dpi)
    plot_meta_synergy_timecourse(
        results_dir, mappings, out_dir, args.conditions,
        n_clusters=args.n_clusters, dpi=args.dpi,
    )
    # Affected-side split plots (more_affected / less_affected) when demographics provided
    demo_path = getattr(args, "demographic", None)
    script_dir = Path(__file__).resolve().parent
    if demo_path is not None and not Path(demo_path).exists():
        # Resolve relative path against script dir (robust when cwd differs)
        alt = script_dir / Path(demo_path).name
        if alt.exists():
            demo_path = alt
    if demo_path is None:
        # Try script dir first (robust when cwd differs), then cwd, then results
        for candidate in [
            script_dir / "demografica.xlsx",
            script_dir / "demographics.csv",
            Path("demografica.xlsx"),
            Path("demographics.csv"),
            results_dir.parent / "demografica.xlsx",
        ]:
            if candidate.exists():
                demo_path = candidate
                break
    elif not demo_path.is_absolute() and not demo_path.exists():
        # Resolve relative --demographic against script dir when cwd differs
        alt = script_dir / demo_path
        if alt.exists():
            demo_path = alt
    if demo_path is not None:
        patient_cond_to_affected = build_patient_cond_to_affected(demo_path)
        if patient_cond_to_affected:
            curves_by_key, _ = load_h_curves_by_cluster_affected(
                results_dir, patient_cond_to_affected, out_dir=out_dir
            )
            p_values: Optional[Dict[Tuple[str, int], float]] = None
            if curves_by_key:
                print("Running permutation tests (T0 vs T1 per meta-synergy, more/less affected)...")
                p_values = run_permutation_tests_meta_synergy_affected(
                    curves_by_key,
                    out_dir,
                    n_clusters=args.n_clusters,
                    n_perm=args.n_permutations,
                    random_state=args.random_state,
                )
            plot_meta_synergy_timecourse_by_affected(
                results_dir, patient_cond_to_affected, out_dir, args.conditions,
                n_clusters=args.n_clusters, dpi=args.dpi, p_values=p_values,
            )
        else:
            _diagnose_demographics_failure(demo_path)

    print(f"All outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
