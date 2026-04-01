#!/usr/bin/env python3
"""B6_muscle_synergy_manuscript_figure.py

Publication-quality 6-panel figure summarizing muscle synergy results for manuscript.
- Panel A: Synergy dimensionality (boxplots, x=T0/T1, hue=more/less affected)
- Panel B: Reconstruction quality (boxplots, x=T0/T1, hue=more/less affected, ylim 0.6-1)
- Panel C: Meta-synergy structure (radar)
- Panel D: Effect size for all meta-synergies AuC and peak time (more affected)
- Panel E: Time activation more affected (T0 vs T1, perm test)
- Panel F: Time activation less affected (T0 vs T1, perm test)

Design: All panels built from CSV/NPZ data. No figure loading.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multitest import multipletests

# -----------------------------------------------------------------------------
# Path configuration (edit these)
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results" / "synergies" / "meta_synergy_clustering"
FIGURES_DIR = RESULTS_ROOT / "figures"
SUMMARY_DIR = RESULTS_ROOT / "summary_tables"
RESULTS_SYNERGIES = SCRIPT_DIR / "results" / "synergies"
OUTPUT_DIR = SCRIPT_DIR / "results" / "synergy_manuscript_figure"

# CSV paths
META_CENTROIDS_CSV = RESULTS_ROOT / "meta_centroids.csv"
CLUSTER_ASSIGNMENT_CSV = RESULTS_ROOT / "cluster_assignment.csv"
META_TASK_SUMMARY_CSV = RESULTS_ROOT / "meta_task_summary_metrics.csv"
H_PAIRED_CLUSTERED_CSV = RESULTS_ROOT / "h_paired_summary_clustered.csv"
TASK_SUMMARY_BY_CLUSTER_CSV = RESULTS_ROOT / "task_summary_by_cluster.csv"
META_PERMUTATION_TESTS_CSV = RESULTS_ROOT / "meta_synergy_permutation_tests.csv"
PATIENT_TASK_METRICS_CSV = RESULTS_SYNERGIES / "patient_task_metrics.csv"
META_H_STATS_CSV = SUMMARY_DIR / "meta_synergy_h_stats.csv"

# Demographics for more/less affected mapping
DEFAULT_DEMO_PATHS = [
    SCRIPT_DIR / "demografica.xlsx",
    SCRIPT_DIR / "demographics.csv",
]

# -----------------------------------------------------------------------------
# Style constants
# -----------------------------------------------------------------------------

FONT_PANEL = 16
FONT_TITLE = 12
FONT_AXIS = 11
FONT_TICK = 10
FONT_TICK_LABEL = FONT_TICK + 1  # Tick labels +1 for legibility
COLOR_MORE = "#4a5568"
COLOR_LESS = "#718096"
COLOR_LINE = "#a0aec0"
COLOR_REF = "#cbd5e0"
PROFILE_NPTS = 101
ALPHA_SIGNIFICANCE = 0.05  # before FDR
ALPHA_PANEL_D = 0.1  # Panel D: draw significance squares when p < 0.1
COLOR_SIG_MARKER = "#c53030"  # red for significance
# Panel D: same as B3 EMG figure - more = full circles, less = empty circles
COLOR_D_POINT = "#4a5568"  # slate (full for more affected, edge for less affected)

# Cluster colors (match B5)
def _cluster_colors(n: int) -> np.ndarray:
    return plt.cm.Set1(np.linspace(0, 0.8, n))


# -----------------------------------------------------------------------------
# Utility: demographics and affected-side mapping
# -----------------------------------------------------------------------------

def _find_affected_column(df: pd.DataFrame) -> Optional[str]:
    for col in df.columns:
        c = str(col).lower()
        if "colpito" in c or ("affected" in c and "side" in c):
            return col
    return None


def _normalize_side(val: Any) -> str:
    s = str(val).upper().strip()
    if s in ("DX", "D", "DESTRO", "RIGHT"):
        return "DX"
    if s in ("SX", "S", "SINISTRO", "LEFT"):
        return "SX"
    return ""


def _condition_to_performed_side(cond: str) -> Optional[str]:
    m = {"DS": "DX", "SN": "SX"}
    return m.get(str(cond).upper())


def load_demographics(path: Optional[Path]) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path)
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def build_patient_cond_to_affected(demo_path: Optional[Path]) -> Dict[Tuple[str, str], str]:
    """(patient_id, condition) -> 'more_affected' | 'less_affected'."""
    out: Dict[Tuple[str, str], str] = {}
    demo = load_demographics(demo_path)
    if demo.empty:
        return out
    aff_col = _find_affected_column(demo)
    if not aff_col:
        return out
    id_col = "id" if "id" in demo.columns else "patient_id" if "patient_id" in demo.columns else demo.columns[0]
    for _, row in demo.iterrows():
        pid = str(row.get(id_col, "")).strip()
        if not pid:
            continue
        affected = _normalize_side(row.get(aff_col, ""))
        if affected not in ("DX", "SX"):
            continue
        for cond in ("SN", "DS"):
            performed = _condition_to_performed_side(cond)
            if performed:
                out[(pid, cond)] = "more_affected" if performed == affected else "less_affected"
    return out


# -----------------------------------------------------------------------------
# Utility: file checks, discovery, data loading
# -----------------------------------------------------------------------------

def file_exists(path: Optional[Path]) -> bool:
    return path is not None and path.exists()


def parse_session_condition(task_name: str) -> Tuple[str, str]:
    task_upper = str(task_name).upper()
    session = "T0" if "T0" in task_upper else "T1" if "T1" in task_upper else ""
    condition = "SN" if "SN" in task_upper else "DS" if "DS" in task_upper else ""
    return session, condition


def discover_task_dirs(results_dir: Path) -> List[Tuple[str, str, str, str, Path]]:
    """Discover (patient_id, task_name, session, condition, task_dir)."""
    discovered: List[Tuple[str, str, str, str, Path]] = []
    if not results_dir.exists():
        return discovered
    for patient_dir in sorted(results_dir.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("plots_") or patient_dir.name == "meta_synergy_clustering":
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


def read_npz_array(path: Path, key: str) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        npz = np.load(path, allow_pickle=True)
        return npz[key] if key in npz.files else None
    except Exception:
        return None


def _load_presence_by_task(assignment_path: Path) -> Dict[Tuple[str, str, str, str], Set[int]]:
    """(patient_id, task_name, session, condition) -> set of present meta_cluster indices."""
    if not assignment_path.exists():
        return {}
    try:
        df = pd.read_csv(assignment_path)
    except Exception:
        return {}
    if "patient_id" not in df.columns or "meta_cluster" not in df.columns or "task_name" not in df.columns:
        return {}
    out: Dict[Tuple[str, str, str, str], Set[int]] = {}
    for _, row in df.iterrows():
        key = (str(row["patient_id"]), str(row["task_name"]), str(row["session"]), str(row["condition"]))
        out.setdefault(key, set()).add(int(row["meta_cluster"]))
    return out


def load_h_curves_by_cluster_affected(
    results_dir: Path,
    patient_cond_to_affected: Dict[Tuple[str, str], str],
    out_dir: Optional[Path] = None,
    n_phase_pts: int = PROFILE_NPTS,
) -> Tuple[Dict[Tuple[str, str, int], List[np.ndarray]], Dict[Tuple[str, str, int], int]]:
    """Load H_meta curves by (affected_group, session, cluster). Returns (curves_by_key, n_subjects_by_key)."""
    phase_new = np.linspace(0.0, 1.0, n_phase_pts)
    curves_by_key: Dict[Tuple[str, str, int], List[np.ndarray]] = {}
    subjects_by_key: Dict[Tuple[str, str, int], Set[str]] = {}
    assignment_path = (out_dir / "cluster_assignment.csv") if out_dir else None
    presence = _load_presence_by_task(assignment_path) if assignment_path and assignment_path.exists() else {}
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


def _cohens_dz(deltas: List[float]) -> float:
    """Cohen's d_z for paired data: mean(delta) / SD(delta). Returns 0 if n < 2 or SD = 0."""
    arr = np.array([float(x) for x in deltas if np.isfinite(x)], dtype=float)
    if len(arr) < 2:
        return 0.0
    sd = np.std(arr)
    if sd < 1e-12:
        return 0.0
    return float(np.mean(arr) / sd)


def _paired_pvalue(v0: np.ndarray, v1: np.ndarray) -> float:
    """Paired t-test p-value (before FDR). Returns 1.0 if n < 2 or invalid."""
    v0, v1 = np.asarray(v0).flatten(), np.asarray(v1).flatten()
    v0, v1 = v0[~np.isnan(v0)], v1[~np.isnan(v1)]
    n = min(len(v0), len(v1))
    if n < 2:
        return 1.0
    v0, v1 = v0[:n], v1[:n]
    try:
        _, p = scipy_stats.ttest_rel(v0, v1)
        return float(p) if np.isfinite(p) else 1.0
    except Exception:
        return 1.0


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_TICK_LABEL)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=FONT_PANEL, fontweight="bold", va="top")


# -----------------------------------------------------------------------------
# Utility: boxplot by session (T0/T1) and hue (more/less affected)
# -----------------------------------------------------------------------------

def _boxplot_by_session_hue(
    ax: plt.Axes,
    t0_more: np.ndarray,
    t1_more: np.ndarray,
    t0_less: np.ndarray,
    t1_less: np.ndarray,
    ylabel: str = "",
    xtick_labels: Tuple[str, str] = ("T0", "T1"),
    showfliers: bool = True,
    show_legend: bool = True,
    legend_loc: str = "upper right",
    legend_fontsize: Optional[int] = None,
) -> None:
    """Boxplots with x=T0/T1, hue=more/less affected."""
    if legend_fontsize is None:
        legend_fontsize = FONT_TICK - 1
    # Positions: T0 at 0 (more at -0.15, less at 0.15), T1 at 1 (more at 0.85, less at 1.15)
    data = [t0_more, t0_less, t1_more, t1_less]
    positions = [-0.15, 0.15, 0.85, 1.15]
    colors = [COLOR_MORE, COLOR_LESS, COLOR_MORE, COLOR_LESS]
    data_ok = [(d, positions[i], colors[i]) for i, d in enumerate(data) if len(d) > 0]
    if not data_ok:
        return
    bp = ax.boxplot(
        [x[0] for x in data_ok],
        positions=[x[1] for x in data_ok],
        widths=0.22,
        patch_artist=True,
        showfliers=showfliers,
        flierprops=dict(marker="o", markersize=4, alpha=0.7),
    )
    for box, (_, _, c) in zip(bp["boxes"], data_ok):
        box.set_facecolor(c)
        box.set_alpha(0.7)
    for whisker in bp["whiskers"]:
        whisker.set_color(COLOR_LINE)
    for cap in bp["caps"]:
        cap.set_color(COLOR_LINE)
    for median in bp["medians"]:
        median.set_color("#1a202c")
        median.set_linewidth(1.5)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(xtick_labels, fontsize=FONT_TICK_LABEL)
    ax.set_ylabel(ylabel)
    ax.set_xlim(-0.5, 1.5)
    if show_legend:
        legend_elements = [
            mpatches.Patch(facecolor=COLOR_MORE, alpha=0.7, label="More affected"),
            mpatches.Patch(facecolor=COLOR_LESS, alpha=0.7, label="Less affected"),
        ]
        ax.legend(handles=legend_elements, loc=legend_loc, fontsize=legend_fontsize)


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_patient_task_metrics() -> pd.DataFrame:
    if not file_exists(PATIENT_TASK_METRICS_CSV):
        return pd.DataFrame()
    return pd.read_csv(PATIENT_TASK_METRICS_CSV)


def load_paired_n_synergies_vaf(
    patient_cond_to_affected: Dict[Tuple[str, str], str],
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray],
           Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Returns (n_T0_more, n_T1_more, n_T0_less, n_T1_less, vaf_T0_more, vaf_T1_more, vaf_T0_less, vaf_T1_less).
    Each patient contributes one value per (session, affected_group) from the condition that maps to that group.
    """
    df = load_patient_task_metrics()
    if df.empty or "patient_id" not in df.columns or "session" not in df.columns:
        return (None,) * 8
    if "n_synergies" not in df.columns or "mean_global_vaf" not in df.columns or "condition" not in df.columns:
        return (None,) * 8
    if not patient_cond_to_affected:
        return (None,) * 8

    by_key: Dict[Tuple[str, str, str], Tuple[Optional[float], Optional[float]]] = {}
    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        sess = str(row["session"]).upper()
        cond = str(row.get("condition", "")).upper()
        if sess not in ("T0", "T1") or cond not in ("SN", "DS"):
            continue
        aff = patient_cond_to_affected.get((pid, cond), "")
        if not aff:
            continue
        key = (pid, sess, aff)
        n_syn = row.get("n_synergies")
        vaf = row.get("mean_global_vaf")
        n_val = float(n_syn) if pd.notna(n_syn) else None
        v_val = float(vaf) if pd.notna(vaf) else None
        if key not in by_key:
            by_key[key] = (n_val, v_val)
        else:
            existing = by_key[key]
            if n_val is not None and existing[0] is None:
                by_key[key] = (n_val, existing[1])
            if v_val is not None and existing[1] is None:
                by_key[key] = (existing[0], v_val)

    patients = sorted({k[0] for k in by_key})
    common_more = {p for p in patients
                   if (p, "T0", "more_affected") in by_key and (p, "T1", "more_affected") in by_key
                   and by_key[(p, "T0", "more_affected")][0] is not None and by_key[(p, "T1", "more_affected")][0] is not None}
    common_less = {p for p in patients
                   if (p, "T0", "less_affected") in by_key and (p, "T1", "less_affected") in by_key
                   and by_key[(p, "T0", "less_affected")][0] is not None and by_key[(p, "T1", "less_affected")][0] is not None}

    def _extract(aff: str, metric: str) -> Tuple[np.ndarray, np.ndarray]:
        common = common_more if aff == "more_affected" else common_less
        idx = 0 if metric == "n" else 1
        t0_list, t1_list = [], []
        for p in common:
            v0 = by_key.get((p, "T0", aff), (None, None))[idx]
            v1 = by_key.get((p, "T1", aff), (None, None))[idx]
            if v0 is not None and v1 is not None:
                t0_list.append(v0)
                t1_list.append(v1)
        return np.array(t0_list), np.array(t1_list)

    n_t0_m, n_t1_m = _extract("more_affected", "n")
    n_t0_l, n_t1_l = _extract("less_affected", "n")
    v_t0_m, v_t1_m = _extract("more_affected", "vaf")
    v_t0_l, v_t1_l = _extract("less_affected", "vaf")

    return n_t0_m, n_t1_m, n_t0_l, n_t1_l, v_t0_m, v_t1_m, v_t0_l, v_t1_l


def load_meta_centroids() -> Tuple[Optional[np.ndarray], Optional[List[str]]]:
    if not file_exists(META_CENTROIDS_CSV):
        return None, None
    df = pd.read_csv(META_CENTROIDS_CSV)
    if df.empty:
        return None, None
    cluster_cols = [c for c in df.columns if c in ("C0", "C1", "C2", "C3")]
    if not cluster_cols:
        return None, None
    # Meta_centroids: rows = muscles, first col = muscle names, C0-C3 = cluster weights
    non_cluster = [c for c in df.columns if c not in cluster_cols]
    if non_cluster:
        muscles = df[non_cluster[0]].astype(str).tolist()
    else:
        muscles = [f"M{i}" for i in range(len(df))]
    mat = df[cluster_cols].values.T  # (n_clusters, n_muscles)
    if not muscles or len(muscles) != mat.shape[1]:
        muscles = [f"M{i}" for i in range(mat.shape[1])]
    # Reorder by dominant meta-synergy (data-driven): C0, C1, C2, C3; within each group by descending loading
    n_m = mat.shape[1]
    indices = list(range(n_m))
    dominant = np.argmax(mat, axis=0)
    loadings = np.array([mat[dominant[j], j] for j in range(n_m)])
    sorted_idx = sorted(
        indices,
        key=lambda j: (int(dominant[j]), -float(loadings[j])),
    )
    mat = mat[:, sorted_idx]
    muscles = [muscles[j] for j in sorted_idx]
    return mat, muscles


def load_h_paired_from_meta_task(
    patient_cond_to_affected: Dict[Tuple[str, str], str],
) -> Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
    """
    Build paired AuC and peak_time from meta_task_summary_metrics (same source as B7 and Panels E/F).
    Aligns Panel D effect sizes with timecourse plots and Excel stats.
    Returns: {cluster: {metric: {aff: (t0_arr, t1_arr)}}}
    """
    METRICS = ("centroid", "peak_time", "mean")
    out: Dict[str, Dict[str, Dict[str, Tuple[List[float], List[float]]]]] = {
        cl: {m: {"more_affected": ([], []), "less_affected": ([], [])} for m in METRICS}
        for cl in ("C0", "C1", "C2", "C3")
    }
    if not file_exists(META_TASK_SUMMARY_CSV):
        return {cl: {m: {a: (np.array(t0), np.array(t1)) for a, (t0, t1) in o.items()} for m, o in v.items()} for cl, v in out.items()}
    df = pd.read_csv(META_TASK_SUMMARY_CSV)
    if df.empty or "patient_id" not in df.columns or "condition" not in df.columns or "session" not in df.columns:
        return {cl: {m: {a: (np.array([]), np.array([])) for a in ("more_affected", "less_affected")} for m in ("auc", "peak_time")} for cl in ("C0", "C1", "C2", "C3")}
    # Aggregate per (patient_id, condition, session): mean across tasks
    agg_cols = [c for c in df.columns if c.startswith("mean_meta_synergy_") and (c.endswith("_auc") or c.endswith("_peak_time"))]
    if not agg_cols:
        return {cl: {m: {a: (np.array([]), np.array([])) for a in ("more_affected", "less_affected")} for m in ("auc", "peak_time")} for cl in ("C0", "C1", "C2", "C3")}
    grouped = df.groupby(["patient_id", "condition", "session"])[agg_cols].mean().reset_index()
    # Pivot to one row per (patient_id, condition) with T0 and T1 columns
    t0 = grouped[grouped["session"] == "T0"].drop(columns=["session"])
    t1 = grouped[grouped["session"] == "T1"].drop(columns=["session"])
    merged = t0.merge(t1, on=["patient_id", "condition"], suffixes=("_T0", "_T1"))
    for _, row in merged.iterrows():
        pid = str(row["patient_id"])
        cond = str(row.get("condition", "")).upper()
        if cond not in ("SN", "DS"):
            continue
        aff = patient_cond_to_affected.get((pid, cond), "")
        if aff not in ("more_affected", "less_affected"):
            continue
        for k in range(4):
            cl = f"C{k}"
            a0_col = f"mean_meta_synergy_{k}_auc_T0"
            a1_col = f"mean_meta_synergy_{k}_auc_T1"
            p0_col = f"mean_meta_synergy_{k}_peak_time_T0"
            p1_col = f"mean_meta_synergy_{k}_peak_time_T1"
            a0, a1 = row.get(a0_col), row.get(a1_col)
            p0, p1 = row.get(p0_col), row.get(p1_col)
            if pd.notna(a0) and pd.notna(a1):
                out[cl]["auc"][aff][0].append(float(a0))
                out[cl]["auc"][aff][1].append(float(a1))
            if pd.notna(p0) and pd.notna(p1):
                out[cl]["peak_time"][aff][0].append(float(p0))
                out[cl]["peak_time"][aff][1].append(float(p1))
    return {cl: {m: {a: (np.array(t0), np.array(t1)) for a, (t0, t1) in o.items()} for m, o in v.items()} for cl, v in out.items()}


def load_h_paired_by_cluster(
    patient_cond_to_affected: Dict[Tuple[str, str], str],
) -> Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
    """
    Load paired AuC and peak_time per cluster from h_paired_summary_clustered (legacy).
    NOTE: Prefer load_h_paired_from_meta_task for consistency with B7 and Panels E/F.
    Returns: {cluster: {metric: {aff: (t0_arr, t1_arr)}}}
    """
    out: Dict[str, Dict[str, Dict[str, Tuple[List[float], List[float]]]]] = {}
    for cl in ("C0", "C1", "C2", "C3"):
        out[cl] = {"auc": {"more_affected": ([], []), "less_affected": ([], [])},
                   "peak_time": {"more_affected": ([], []), "less_affected": ([], [])}}
    if not file_exists(H_PAIRED_CLUSTERED_CSV):
        return {cl: {m: {a: (np.array(t0), np.array(t1)) for a, (t0, t1) in o.items()} for m, o in v.items()} for cl, v in out.items()}
    df = pd.read_csv(H_PAIRED_CLUSTERED_CSV)
    if df.empty:
        return {k: {m: {a: (np.array([]), np.array([])) for a in ("more_affected", "less_affected")} for m in ("auc", "peak_time")} for k in ("C0", "C1", "C2", "C3")}
    for _, row in df.iterrows():
        pid = str(row["patient_id"])
        cond = str(row.get("condition", "")).upper()
        cl = str(row.get("synergy_label", ""))
        if cl not in out:
            continue
        aff = patient_cond_to_affected.get((pid, cond), "")
        if aff not in ("more_affected", "less_affected"):
            continue
        a0, a1 = row.get("auc_T0"), row.get("auc_T1")
        p0, p1 = row.get("peak_time_T0"), row.get("peak_time_T1")
        if pd.notna(a0) and pd.notna(a1):
            out[cl]["auc"][aff][0].append(float(a0))
            out[cl]["auc"][aff][1].append(float(a1))
        if pd.notna(p0) and pd.notna(p1):
            out[cl]["peak_time"][aff][0].append(float(p0))
            out[cl]["peak_time"][aff][1].append(float(p1))
    result: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for cl in ("C0", "C1", "C2", "C3"):
        result[cl] = {}
        for m in ("auc", "peak_time"):
            result[cl][m] = {}
            for aff in ("more_affected", "less_affected"):
                t0, t1 = out[cl][m][aff]
                result[cl][m][aff] = (np.array(t0), np.array(t1))
    return result


def load_h_paired_from_meta_task(
    patient_cond_to_affected: Dict[Tuple[str, str], str],
) -> Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
    """
    Build paired AuC and peak_time from meta_task_summary_metrics (same source as B7 and Panels E/F).

    This aligns Panel D effect sizes with the timecourse plots (E/F) and B7 Excel stats.
    Uses mean_meta_synergy_X_auc and mean_meta_synergy_X_peak_time from H_meta (not task_summary
    remapped via h_paired_summary_clustered, which uses different metrics).

    Returns: {cluster: {metric: {aff: (t0_arr, t1_arr)}}}
    """
    out: Dict[str, Dict[str, Dict[str, Tuple[List[float], List[float]]]]] = {}
    for cl in ("C0", "C1", "C2", "C3"):
        out[cl] = {"auc": {"more_affected": ([], []), "less_affected": ([], [])},
                   "peak_time": {"more_affected": ([], []), "less_affected": ([], [])}}
    if not file_exists(META_TASK_SUMMARY_CSV):
        return {cl: {m: {a: (np.array(t0), np.array(t1)) for a, (t0, t1) in o.items()} for m, o in v.items()} for cl, v in out.items()}
    df = pd.read_csv(META_TASK_SUMMARY_CSV)
    if df.empty or "patient_id" not in df.columns or "session" not in df.columns or "condition" not in df.columns:
        return {cl: {m: {a: (np.array(t0), np.array(t1)) for a, (t0, t1) in o.items()} for m, o in v.items()} for cl, v in out.items()}
    n_clusters = 4
    # Aggregate by (patient_id, condition, session): mean across tasks
    agg_cols = {}
    for k in range(n_clusters):
        for m, col_suffix in [("auc", "auc"), ("peak_time", "peak_time")]:
            col = f"mean_meta_synergy_{k}_{col_suffix}"
            if col in df.columns:
                agg_cols[col] = "mean"
    if not agg_cols:
        return {cl: {m: {a: (np.array(t0), np.array(t1)) for a, (t0, t1) in o.items()} for m, o in v.items()} for cl, v in out.items()}
    agg_df = df.groupby(["patient_id", "condition", "session"], as_index=False)[list(agg_cols.keys())].mean()
    for (pid, cond), grp in agg_df.groupby(["patient_id", "condition"]):
        cond = str(cond).upper()
        if cond not in ("SN", "DS"):
            continue
        aff = patient_cond_to_affected.get((str(pid), cond), "")
        if aff not in ("more_affected", "less_affected"):
            continue
        t0_r = grp[grp["session"] == "T0"]
        t1_r = grp[grp["session"] == "T1"]
        if t0_r.empty or t1_r.empty:
            continue
        t0_row, t1_row = t0_r.iloc[0], t1_r.iloc[0]
        for k in range(n_clusters):
            cl = f"C{k}"
            a0 = t0_row.get(f"mean_meta_synergy_{k}_auc", np.nan)
            a1 = t1_row.get(f"mean_meta_synergy_{k}_auc", np.nan)
            p0 = t0_row.get(f"mean_meta_synergy_{k}_peak_time", np.nan)
            p1 = t1_row.get(f"mean_meta_synergy_{k}_peak_time", np.nan)
            if pd.notna(a0) and pd.notna(a1):
                out[cl]["auc"][aff][0].append(float(a0))
                out[cl]["auc"][aff][1].append(float(a1))
            if pd.notna(p0) and pd.notna(p1):
                out[cl]["peak_time"][aff][0].append(float(p0))
                out[cl]["peak_time"][aff][1].append(float(p1))
    result: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for cl in ("C0", "C1", "C2", "C3"):
        result[cl] = {}
        for m in ("auc", "peak_time"):
            result[cl][m] = {}
            for aff in ("more_affected", "less_affected"):
                t0, t1 = out[cl][m][aff]
                result[cl][m][aff] = (np.array(t0), np.array(t1))
    return result


def load_h_paired_from_meta_task(
    patient_cond_to_affected: Dict[Tuple[str, str], str],
    n_clusters: int = 4,
) -> Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
    """
    Build paired AuC and peak_time from meta_task_summary_metrics (same source as B7 and Panels E/F).
    Aligns Panel D effect sizes with the timecourse figures and Excel emg_meta_C* columns.

    For each (patient, condition): mean across tasks of mean_meta_synergy_k_auc / peak_time for T0 and T1.
    Returns: {cluster: {metric: {aff: (t0_arr, t1_arr)}}}
    """
    out: Dict[str, Dict[str, Dict[str, Tuple[List[float], List[float]]]]] = {}
    for cl in ("C0", "C1", "C2", "C3"):
        out[cl] = {"auc": {"more_affected": ([], []), "less_affected": ([], [])},
                   "peak_time": {"more_affected": ([], []), "less_affected": ([], [])}}
    if not file_exists(META_TASK_SUMMARY_CSV):
        return {cl: {m: {a: (np.array(t0), np.array(t1)) for a, (t0, t1) in o.items()} for m, o in v.items()} for cl, v in out.items()}
    df = pd.read_csv(META_TASK_SUMMARY_CSV)
    if df.empty or "patient_id" not in df.columns or "condition" not in df.columns or "session" not in df.columns:
        return {k: {m: {a: (np.array([]), np.array([])) for a in ("more_affected", "less_affected")} for m in ("auc", "peak_time")} for k in ("C0", "C1", "C2", "C3")}
    # Aggregate to (patient_id, condition, session): mean across tasks
    agg_cols = [f"mean_meta_synergy_{k}_auc" for k in range(n_clusters)]
    agg_cols += [f"mean_meta_synergy_{k}_peak_time" for k in range(n_clusters)]
    agg_cols = [c for c in agg_cols if c in df.columns]
    if not agg_cols:
        return {k: {m: {a: (np.array([]), np.array([])) for a in ("more_affected", "less_affected")} for m in ("auc", "peak_time")} for k in ("C0", "C1", "C2", "C3")}
    patient_cond_sess = df.groupby(["patient_id", "condition", "session"])[agg_cols].mean().reset_index()
    for (pid, cond), grp in patient_cond_sess.groupby(["patient_id", "condition"]):
        cond_upper = str(cond).upper()
        aff = patient_cond_to_affected.get((str(pid), cond_upper), "")
        if aff not in ("more_affected", "less_affected"):
            continue
        t0_rows = grp[grp["session"] == "T0"]
        t1_rows = grp[grp["session"] == "T1"]
        if t0_rows.empty or t1_rows.empty:
            continue
        t0_row = t0_rows.iloc[0]
        t1_row = t1_rows.iloc[0]
        for k in range(n_clusters):
            cl = f"C{k}"
            auc_col = f"mean_meta_synergy_{k}_auc"
            peak_col = f"mean_meta_synergy_{k}_peak_time"
            if auc_col in agg_cols and pd.notna(t0_row.get(auc_col)) and pd.notna(t1_row.get(auc_col)):
                out[cl]["auc"][aff][0].append(float(t0_row[auc_col]))
                out[cl]["auc"][aff][1].append(float(t1_row[auc_col]))
            if peak_col in agg_cols and pd.notna(t0_row.get(peak_col)) and pd.notna(t1_row.get(peak_col)):
                out[cl]["peak_time"][aff][0].append(float(t0_row[peak_col]))
                out[cl]["peak_time"][aff][1].append(float(t1_row[peak_col]))
    result: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for cl in ("C0", "C1", "C2", "C3"):
        result[cl] = {}
        for m in ("auc", "peak_time"):
            result[cl][m] = {}
            for aff in ("more_affected", "less_affected"):
                t0, t1 = out[cl][m][aff]
                result[cl][m][aff] = (np.array(t0), np.array(t1))
    return result


def load_h_paired_from_meta_task(
    patient_cond_to_affected: Dict[Tuple[str, str], str],
) -> Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
    """
    Load paired AuC and peak_time from meta_task_summary_metrics (same source as B7 Excel and Panels E/F).
    Aligns Panel D effect sizes with B7 stats and timecourse plots.
    Averages across tasks per (patient, condition, session), then splits by more/less affected.
    Returns: {cluster: {metric: {aff: (t0_arr, t1_arr)}}}
    """
    out: Dict[str, Dict[str, Dict[str, Tuple[List[float], List[float]]]]] = {}
    for cl in ("C0", "C1", "C2", "C3"):
        out[cl] = {"auc": {"more_affected": ([], []), "less_affected": ([], [])},
                   "peak_time": {"more_affected": ([], []), "less_affected": ([], [])}}
    if not file_exists(META_TASK_SUMMARY_CSV):
        return {cl: {m: {a: (np.array(t0), np.array(t1)) for a, (t0, t1) in o.items()} for m, o in v.items()} for cl, v in out.items()}
    df = pd.read_csv(META_TASK_SUMMARY_CSV)
    if df.empty or "patient_id" not in df.columns or "condition" not in df.columns or "session" not in df.columns:
        return {k: {m: {a: (np.array([]), np.array([])) for a in ("more_affected", "less_affected")} for m in ("auc", "peak_time")} for k in ("C0", "C1", "C2", "C3")}
    df["condition"] = df["condition"].astype(str).str.upper()
    df["session"] = df["session"].astype(str).str.upper()
    for (pid, cond), grp in df.groupby(["patient_id", "condition"]):
        if cond not in ("SN", "DS"):
            continue
        aff = patient_cond_to_affected.get((str(pid), cond), "")
        if aff not in ("more_affected", "less_affected"):
            continue
        t0_grp = grp[grp["session"] == "T0"]
        t1_grp = grp[grp["session"] == "T1"]
        if t0_grp.empty or t1_grp.empty:
            continue
        for k in range(4):
            cl = f"C{k}"
            auc_col = f"mean_meta_synergy_{k}_auc"
            pt_col = f"mean_meta_synergy_{k}_peak_time"
            if auc_col in grp.columns:
                a0 = t0_grp[auc_col].dropna()
                a1 = t1_grp[auc_col].dropna()
                if len(a0) > 0 and len(a1) > 0:
                    out[cl]["auc"][aff][0].append(float(a0.mean()))
                    out[cl]["auc"][aff][1].append(float(a1.mean()))
            if pt_col in grp.columns:
                p0 = t0_grp[pt_col].dropna()
                p1 = t1_grp[pt_col].dropna()
                if len(p0) > 0 and len(p1) > 0:
                    out[cl]["peak_time"][aff][0].append(float(p0.mean()))
                    out[cl]["peak_time"][aff][1].append(float(p1.mean()))
    result: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    for cl in ("C0", "C1", "C2", "C3"):
        result[cl] = {}
        for m in ("auc", "peak_time"):
            result[cl][m] = {}
            for aff in ("more_affected", "less_affected"):
                t0, t1 = out[cl][m][aff]
                result[cl][m][aff] = (np.array(t0), np.array(t1))
    return result


def load_h_paired_from_meta_task(
    patient_cond_to_affected: Dict[Tuple[str, str], str],
    n_clusters: int = 4,
) -> Dict[str, Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]]]:
    """
    Build paired AuC and peak_time from meta_task_summary_metrics (same source as B7 and Panels E/F).
    Uses mean_meta_synergy_X_auc and mean_meta_synergy_X_peak_time from H_meta.
    Aggregates across tasks per (patient, condition) by mean before pairing T0/T1.
    Returns: {cluster: {metric: {aff: (t0_arr, t1_arr)}}}
    """
    metrics = ("centroid", "peak_time", "mean")
    out: Dict[str, Dict[str, Dict[str, Tuple[List[float], List[float]]]]] = {}
    for cl in ("C0", "C1", "C2", "C3"):
        out[cl] = {m: {"more_affected": ([], []), "less_affected": ([], [])} for m in metrics}
    empty_ret = {cl: {m: {a: (np.array([]), np.array([])) for a in ("more_affected", "less_affected")} for m in metrics} for cl in ("C0", "C1", "C2", "C3")}
    if not file_exists(META_TASK_SUMMARY_CSV):
        return {cl: {m: {a: (np.array(t0), np.array(t1)) for a, (t0, t1) in o.items()} for m, o in v.items()} for cl, v in out.items()}
    df = pd.read_csv(META_TASK_SUMMARY_CSV)
    if df.empty or "patient_id" not in df.columns or "condition" not in df.columns or "session" not in df.columns:
        return empty_ret

    # Aggregate per (patient_id, condition, session): mean across tasks
    agg_cols = []
    for k in range(n_clusters):
        for m in metrics:
            col = f"mean_meta_synergy_{k}_{m}"
            if col in df.columns:
                agg_cols.append(col)
    if not agg_cols:
        return empty_ret

    agg_df = df.groupby(["patient_id", "condition", "session"], as_index=False)[agg_cols].mean()

    for (patient_id, condition), grp in agg_df.groupby(["patient_id", "condition"]):
        t0_row = grp[grp["session"] == "T0"]
        t1_row = grp[grp["session"] == "T1"]
        if t0_row.empty or t1_row.empty:
            continue
        t0_row, t1_row = t0_row.iloc[0], t1_row.iloc[0]
        aff = patient_cond_to_affected.get((str(patient_id), str(condition).upper()), "")
        if aff not in ("more_affected", "less_affected"):
            continue
        for k in range(n_clusters):
            cl = f"C{k}"
            if cl not in out:
                continue
            for m in metrics:
                col = f"mean_meta_synergy_{k}_{m}"
                if col not in agg_cols:
                    continue
                v0, v1 = t0_row.get(col, np.nan), t1_row.get(col, np.nan)
                if pd.notna(v0) and pd.notna(v1) and np.isfinite(v0) and np.isfinite(v1):
                    out[cl][m][aff][0].append(float(v0))
                    out[cl][m][aff][1].append(float(v1))

    return {cl: {m: {a: (np.array(t0), np.array(t1)) for a, (t0, t1) in o.items()} for m, o in v.items()} for cl, v in out.items()}


def load_permutation_pvalues() -> Dict[Tuple[str, int], float]:
    out: Dict[Tuple[str, int], float] = {}
    if not file_exists(META_PERMUTATION_TESTS_CSV):
        return out
    df = pd.read_csv(META_PERMUTATION_TESTS_CSV)
    for _, row in df.iterrows():
        aff = str(row.get("affected_group", ""))
        idx = int(row.get("cluster_idx", -1))
        p = float(row.get("p_value", 1.0))
        out[(aff, idx)] = p
    return out


def apply_fdr_to_permutation_pvalues(
    p_values: Dict[Tuple[str, int], float],
    n_clusters: int = 4,
    method: str = "fdr_bh",
) -> Dict[Tuple[str, int], float]:
    """Apply Benjamini-Hochberg FDR correction across all permutation test p-values."""
    if not p_values:
        return {}
    keys_ordered: List[Tuple[str, int]] = []
    pvals_list: List[float] = []
    for aff in ("more_affected", "less_affected"):
        for k in range(n_clusters):
            key = (aff, k)
            if key in p_values and p_values[key] is not None and not np.isnan(p_values[key]):
                keys_ordered.append(key)
                pvals_list.append(float(p_values[key]))
    if not pvals_list:
        return p_values
    _, qvals, _, _ = multipletests(np.array(pvals_list), alpha=0.05, method=method)
    out: Dict[Tuple[str, int], float] = dict(p_values)
    for key, q in zip(keys_ordered, qvals):
        out[key] = float(q)
    return out


# -----------------------------------------------------------------------------
# Panel builders
# -----------------------------------------------------------------------------

def _safe_less(t0: Optional[np.ndarray], t1: Optional[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    empty = np.array([])
    if t0 is None or t1 is None or len(t0) == 0 or len(t1) == 0:
        return empty, empty
    return t0, t1


def build_panel_a(
    ax: plt.Axes,
    patient_cond_to_affected: Dict[Tuple[str, str], str],
) -> bool:
    """Panel A: Synergy dimensionality (boxplots, x=T0/T1, hue=more/less affected)."""
    n_t0_m, n_t1_m, n_t0_l, n_t1_l, _, _, _, _ = load_paired_n_synergies_vaf(patient_cond_to_affected)
    if n_t0_m is None or len(n_t0_m) == 0:
        ax.text(0.5, 0.5, "No n_synergies data.\nRun B5 and ensure demographics.", ha="center", va="center", transform=ax.transAxes)
        return False
    t0_l, t1_l = _safe_less(n_t0_l, n_t1_l)
    _boxplot_by_session_hue(ax, n_t0_m, n_t1_m, t0_l, t1_l,
                            ylabel="Number of synergies", xtick_labels=("T0", "T1"),
                            showfliers=False, show_legend=False)
    ax.set_title("Synergy dimensionality", fontsize=FONT_TITLE)
    ax.set_ylim(0, None)
    return True


def build_panel_b(
    ax: plt.Axes,
    patient_cond_to_affected: Dict[Tuple[str, str], str],
) -> bool:
    """Panel B: Reconstruction quality (boxplots, x=T0/T1, hue=more/less affected)."""
    _, _, _, _, v_t0_m, v_t1_m, v_t0_l, v_t1_l = load_paired_n_synergies_vaf(patient_cond_to_affected)
    if v_t0_m is None or len(v_t0_m) == 0:
        ax.text(0.5, 0.5, "No VAF data.", ha="center", va="center", transform=ax.transAxes)
        return False
    t0_l, t1_l = _safe_less(v_t0_l, v_t1_l)
    _boxplot_by_session_hue(ax, v_t0_m, v_t1_m, t0_l, t1_l,
                            ylabel="Global VAF", xtick_labels=("T0", "T1"),
                            showfliers=False, show_legend=True, legend_loc="lower right",
                            legend_fontsize=FONT_TICK + 1)
    ax.set_title("Reconstruction quality", fontsize=FONT_TITLE)
    ax.set_ylim(0.6, 1.0)
    return True


def build_panel_c(ax: plt.Axes) -> bool:
    """Panel C: Meta-synergy structure (radar) from meta_centroids.csv."""
    mat, muscles = load_meta_centroids()
    if mat is None or muscles is None or mat.size == 0:
        ax.text(0.5, 0.5, "Run B5 to generate meta_centroids.csv",
                ha="center", va="center", transform=ax.transAxes, fontsize=FONT_TICK)
        return False
    n_clusters = mat.shape[0]
    n_muscles = len(muscles)
    angles = np.linspace(0, 2 * np.pi, n_muscles, endpoint=False).tolist()
    angles += angles[:1]
    colors = _cluster_colors(n_clusters)
    for k in range(n_clusters):
        values = mat[k, :].tolist()
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=f"C{k}", color=colors[k])
        ax.fill(angles, values, alpha=0.2, color=colors[k])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(muscles, fontsize=FONT_TICK_LABEL - 2, ha="center")  # was FONT_TICK-2, now +1
    ax.set_yticklabels([])  # Remove radial axis tick labels; keep muscle names only
    ax.set_ylim(0, None)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.05), fontsize=FONT_TICK_LABEL)
    ax.set_title("Meta-synergy structure", fontsize=FONT_TITLE)
    return True


def build_panel_d(
    ax: plt.Axes,
    patient_cond_to_affected: Dict[Tuple[str, str], str],
    p_values: Optional[Dict[Tuple[str, int], float]] = None,
) -> bool:
    """Panel D: Effect size (Cohen's d_z) for all meta-synergies centroid, peak time, mean (more + less affected).
    Significance squares only drawn when BOTH paired t-test p<0.1 AND permutation test for that synergy is significant.
    Uses meta_task_summary_metrics (same source as B7 Excel and Panels E/F) for consistency."""
    paired = load_h_paired_from_meta_task(patient_cond_to_affected)
    labels: List[str] = []
    es_more: List[Optional[float]] = []
    sig_more: List[bool] = []
    es_less: List[Optional[float]] = []
    sig_less: List[bool] = []
    print("\n--- Panel D: p-values (paired t-test T0 vs T1) ---")
    METRICS_PANEL_D = [("centroid", "centroid"), ("peak_time", "peak time"), ("mean", "mean")]
    for cl in ("C0", "C1", "C2", "C3"):
        for metric, lab in METRICS_PANEL_D:
            labels.append(f"{cl} {lab}")
            # More affected
            t0_m, t1_m = paired.get(cl, {}).get(metric, {}).get("more_affected", (np.array([]), np.array([])))
            if len(t0_m) >= 2 and len(t1_m) >= 2:
                n = min(len(t0_m), len(t1_m))
                deltas = (np.asarray(t1_m[:n]) - np.asarray(t0_m[:n])).tolist()
                p_more = _paired_pvalue(t0_m[:n], t1_m[:n])
                es_more.append(_cohens_dz(deltas))
                sig_more.append(p_more < ALPHA_PANEL_D)
                print(f"  {cl} {lab} (more_affected): n={n}, p={p_more:.4f}")
            else:
                es_more.append(None)
                sig_more.append(False)
                print(f"  {cl} {lab} (more_affected): n<2, p=N/A")
            # Less affected
            t0_l, t1_l = paired.get(cl, {}).get(metric, {}).get("less_affected", (np.array([]), np.array([])))
            if len(t0_l) >= 2 and len(t1_l) >= 2:
                n = min(len(t0_l), len(t1_l))
                deltas = (np.asarray(t1_l[:n]) - np.asarray(t0_l[:n])).tolist()
                p_less = _paired_pvalue(t0_l[:n], t1_l[:n])
                es_less.append(_cohens_dz(deltas))
                sig_less.append(p_less < ALPHA_PANEL_D)
                print(f"  {cl} {lab} (less_affected): n={n}, p={p_less:.4f}")
            else:
                es_less.append(None)
                sig_less.append(False)
                print(f"  {cl} {lab} (less_affected): n<2, p=N/A")
    print("---\n")
    if not labels:
        ax.text(0.5, 0.5, "No paired AuC/peak_time data.\nRun B5 and ensure h_paired_summary_clustered.csv.",
                ha="center", va="center", transform=ax.transAxes, fontsize=FONT_TICK)
        return False
    y = np.arange(len(labels))
    ax.axvline(0, color=COLOR_LINE, lw=1, ls="--")
    # Permutation-gated significance: only show square when BOTH paired t-test sig AND permutation sig
    def _perm_sig(aff: str, k: int) -> bool:
        if p_values is None:
            return True  # backward compat: no perm data => use paired only
        q = p_values.get((aff, k))
        return q is not None and not np.isnan(q) and float(q) < 0.05

    # More affected: full circles (same as B3 Panel D)
    for i in range(len(labels)):
        if es_more[i] is not None:
            ax.scatter(es_more[i], y[i], c=COLOR_D_POINT, s=54, zorder=2)
            k = i // 3
            if sig_more[i] and _perm_sig("more_affected", k):
                ax.scatter(es_more[i], y[i], marker="s", s=90, facecolors="none", edgecolors=COLOR_SIG_MARKER,
                           linewidths=1.5, zorder=3)
    # Less affected: empty circles with edge (same as B3 Panel D)
    for i in range(len(labels)):
        if es_less[i] is not None:
            ax.scatter(es_less[i], y[i], facecolors="none", edgecolors=COLOR_D_POINT, s=54,
                       linewidths=1.5, zorder=1)
            k = i // 3
            if sig_less[i] and _perm_sig("less_affected", k):
                ax.scatter(es_less[i], y[i], marker="s", s=90, facecolors="none", edgecolors=COLOR_SIG_MARKER,
                           linewidths=1.5, zorder=3)
    # "Not Available" for rows with no data (e.g. C0 often missing)
    for i in range(len(labels)):
        if es_more[i] is None and es_less[i] is None:
            ax.text(0, y[i], "Not Available", ha="center", va="center", fontsize=FONT_TICK - 1,
                    style="italic", color="#718096")
    ax.legend(handles=[
        plt.Line2D([0], [0], marker="o", ls="", color=COLOR_D_POINT, markersize=10, label="More affected"),
        plt.Line2D([0], [0], marker="o", ls="", markerfacecolor="none", markeredgecolor=COLOR_D_POINT,
                   markeredgewidth=1.5, markersize=10, label="Less affected"),
    ], loc="upper left", fontsize=FONT_TICK - 1)
    ax.set_yticks(y)
    tick_labels = ax.set_yticklabels(labels, fontsize=FONT_TICK_LABEL)
    colors = _cluster_colors(4)
    for i, lbl in enumerate(tick_labels):
        k = i // 3  # cluster index (3 metrics per cluster)
        lbl.set_color(colors[k])
    ax.set_xlabel(r"Effect size, $d_z$", fontsize=FONT_AXIS)
    ax.set_title("Meta-synergy features", fontsize=FONT_TITLE)
    all_es = [e for e in es_more + es_less if e is not None]
    lim = max(abs(np.array(all_es)).max() * 1.2, 0.1) if all_es else 0.1
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.8, len(labels) - 0.5)
    return True


CROP_PHASE_FRAC = 0.02  # Crop first and last 2% of phase
GREY_BAR_EXTENT = 2  # Grey bars span ±2% (i.e., -2 to +2 at start, 98 to 102 at end)


def _draw_timecourse_panel(
    ax: plt.Axes,
    aff_group: str,
    curves_by_key: Dict[Tuple[str, str, int], List[np.ndarray]],
    n_subjects_by_key: Dict[Tuple[str, str, int], int],
    p_values: Dict[Tuple[str, int], float],
    n_clusters: int = 4,
) -> None:
    """Draw meta-synergy timecourse: T0 (solid) vs T1 (dashed) per cluster. Crops first/last 4% with grey bars."""
    phase_full = np.linspace(0, 1, PROFILE_NPTS) * 100
    n_pts = len(phase_full)
    n_crop = max(1, int(n_pts * CROP_PHASE_FRAC))
    keep_slice = slice(n_crop, n_pts - n_crop)
    phase = phase_full[keep_slice]

    # Grey shaded vertical bars: span ±GREY_BAR_EXTENT% (-2 to +2 at start, 98 to 102 at end)
    ax.axvspan(-GREY_BAR_EXTENT, GREY_BAR_EXTENT, facecolor="#e2e8f0", alpha=0.6, zorder=0)
    ax.axvspan(100 - GREY_BAR_EXTENT, 100 + GREY_BAR_EXTENT, facecolor="#e2e8f0", alpha=0.6, zorder=0)

    clusters = [f"C{k}" for k in range(n_clusters)]
    colors = _cluster_colors(n_clusters)
    for k, cluster in enumerate(clusters):
        for session, linestyle in [("T0", "-"), ("T1", "--")]:
            key = (aff_group, session, k)
            curves = curves_by_key.get(key, [])
            if curves:
                curves_arr = np.array(curves)
                mean_c = np.mean(curves_arr, axis=0)
                mean_cropped = mean_c[keep_slice]
                n_subj = n_subjects_by_key.get(key, 0)
                lab = f"{cluster} {session} (n={n_subj})"
                ax.plot(phase, mean_cropped, color=colors[k], linestyle=linestyle, label=lab, linewidth=1.5, zorder=2)
    if p_values:
        p_lines = []
        for k in range(n_clusters):
            if k == 0:
                continue  # C0 excluded from permutation test comparison
            q = p_values.get((aff_group, k))
            if q is not None and not np.isnan(q):
                q_str = f"q={q:.3g}" if q >= 0.001 else "q<0.001"
                p_lines.append(f"{clusters[k]} T0 vs T1: {q_str}")
        if p_lines:
            ax.text(0.02, 0.02, "Permutation test (T0 vs T1, FDR-corrected):\n" + "\n".join(p_lines),
                    transform=ax.transAxes, fontsize=7, verticalalignment="bottom",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    ax.set_ylim(0, 1)
    ax.set_xlim(-GREY_BAR_EXTENT, 100 + GREY_BAR_EXTENT)
    ax.set_xlabel("Normalized phase (%)")
    ax.set_ylabel("Activation (a.u.)")
    ax.set_title(f"Meta-synergy activation: {aff_group.replace('_', ' ')}")
    ax.legend(loc="upper right", ncol=2, fontsize=FONT_TICK_LABEL)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def build_panel_e(
    ax: plt.Axes,
    patient_cond_to_affected: Dict[Tuple[str, str], str],
    p_values: Dict[Tuple[str, int], float],
) -> bool:
    """Panel E: Time activation more affected (T0 vs T1) with perm test results."""
    curves_by_key, n_subjects_by_key = load_h_curves_by_cluster_affected(
        RESULTS_SYNERGIES, patient_cond_to_affected, out_dir=RESULTS_ROOT
    )
    if not curves_by_key:
        ax.text(0.5, 0.5, "No H_meta curves found.\nRun B5 to generate H_meta_windows.npz.",
                ha="center", va="center", transform=ax.transAxes, fontsize=FONT_TICK)
        return False
    n_clusters = 4
    _draw_timecourse_panel(ax, "more_affected", curves_by_key, n_subjects_by_key, p_values, n_clusters)
    return True


def build_panel_f(
    ax: plt.Axes,
    patient_cond_to_affected: Dict[Tuple[str, str], str],
    p_values: Dict[Tuple[str, int], float],
) -> bool:
    """Panel F: Time activation less affected (T0 vs T1) with perm test results."""
    curves_by_key, n_subjects_by_key = load_h_curves_by_cluster_affected(
        RESULTS_SYNERGIES, patient_cond_to_affected, out_dir=RESULTS_ROOT
    )
    if not curves_by_key:
        ax.text(0.5, 0.5, "No H_meta curves found.\nRun B5 to generate H_meta_windows.npz.",
                ha="center", va="center", transform=ax.transAxes, fontsize=FONT_TICK)
        return False
    n_clusters = 4
    _draw_timecourse_panel(ax, "less_affected", curves_by_key, n_subjects_by_key, p_values, n_clusters)
    return True


# -----------------------------------------------------------------------------
# Main: create multi-panel figure
# -----------------------------------------------------------------------------

def create_manuscript_figure(
    out_path: Optional[Path] = None,
    dpi: int = 300,
) -> None:
    """Create the 6-panel manuscript figure."""
    demo_path = None
    for p in DEFAULT_DEMO_PATHS:
        if p.exists():
            demo_path = p
            break
    patient_cond_to_affected = build_patient_cond_to_affected(demo_path)
    if not patient_cond_to_affected:
        print("Warning: No demographics found. more/less affected split may be missing.")

    p_values = load_permutation_pvalues()
    p_values = apply_fdr_to_permutation_pvalues(p_values)

    fig, axes = plt.subplots(3, 2, figsize=(11, 13.5), constrained_layout=True)
    axes = axes.flatten()

    # Panel A
    ok_a = build_panel_a(axes[0], patient_cond_to_affected)
    if ok_a:
        _style_axes(axes[0])
        _add_panel_label(axes[0], "A")

    # Panel B
    ok_b = build_panel_b(axes[1], patient_cond_to_affected)
    if ok_b:
        _style_axes(axes[1])
        _add_panel_label(axes[1], "B")

    # Panel C: replace with polar axes for radar, shifted left to avoid overlap with Panel D labels
    ax_c = axes[2]
    pos_c = ax_c.get_position()
    ax_c.remove()
    # Shrink width and keep left-aligned so muscle names don't overlap Panel D y-axis
    rect_c = [pos_c.x0, pos_c.y0, pos_c.width * 0.78, pos_c.height]
    ax_c = fig.add_axes(rect_c, projection="polar")
    ok_c = build_panel_c(ax_c)
    if ok_c:
        _add_panel_label(ax_c, "C")

    # Panel D: effect size for all synergies centroid/peak time/mean (permutation-gated significance)
    ok_d = build_panel_d(axes[3], patient_cond_to_affected, p_values=p_values)
    if ok_d:
        _style_axes(axes[3])
        _add_panel_label(axes[3], "D")

    # Panel E: time activation more affected
    ok_e = build_panel_e(axes[4], patient_cond_to_affected, p_values)
    if ok_e:
        _style_axes(axes[4])
        _add_panel_label(axes[4], "E")

    # Panel F: time activation less affected
    ok_f = build_panel_f(axes[5], patient_cond_to_affected, p_values)
    if ok_f:
        _style_axes(axes[5])
        _add_panel_label(axes[5], "F")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = out_path or OUTPUT_DIR / "muscle_synergy_manuscript_figure"
    png_path = Path(str(out_path) + ".png")
    pdf_path = Path(str(out_path) + ".pdf")
    svg_path = Path(str(out_path) + ".svg")
    fig.savefig(png_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
    fig.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close(fig)
    print(f"Figure saved: {png_path}, {pdf_path}, {svg_path}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    create_manuscript_figure()
