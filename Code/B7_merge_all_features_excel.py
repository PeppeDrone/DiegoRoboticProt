#!/usr/bin/env python3
"""B7_merge_all_features_excel.py

Merge ALL EMG, IMU, EEG, and demographic features into one Excel file.
Each row: id (patient_id), time (T0/T1), condition (SN/DS).
EEG has no condition - features are repeated for both SN and DS rows.
Demographics merged on id.

Usage:
  python B7_merge_all_features_excel.py
  python B7_merge_all_features_excel.py --results-dir results --eeg eeg_combined_features.csv
  python B7_merge_all_features_excel.py --demographic demographics.csv
"""

from __future__ import annotations

import argparse
import re
import unicodedata
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from statsmodels.stats.multitest import multipletests


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def slugify(s: str) -> str:
    """Make string safe for column name."""
    return re.sub(r"[^a-zA-Z0-9]+", "_", str(s).strip()).strip("_")


def _normalize_col_name(s: str) -> str:
    """Lowercase, strip, remove accents, collapse spaces. For column detection only."""
    s = str(s).strip().lower()
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _find_column(df: pd.DataFrame, candidates: List[str], aliases: Optional[List[str]] = None) -> Optional[str]:
    """
    Find first column whose normalized name matches any candidate or alias.
    candidates: e.g. ["id", "patient_id", "patient id"]
    aliases: extra variants for affected-side column.
    """
    norm_to_col = {_normalize_col_name(c): c for c in df.columns}
    search = list(candidates or [])
    if aliases:
        search = search + aliases
    for cand in search:
        n = _normalize_col_name(cand)
        for norm, col in norm_to_col.items():
            if n in norm or norm in n or n == norm:
                return col
    return None


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
        return "DX" if "D" in v or "X" in v and v[0] == "D" else "SX"
    return v


def discover_task_dirs(results_dir: Path) -> List[Tuple[str, str, str, str, Path]]:
    """(patient_id, task_name, session, condition, task_dir)."""
    discovered: List[Tuple[str, str, str, str, Path]] = []
    if not results_dir.exists():
        return discovered
    for patient_dir in sorted(results_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        if patient_dir.name.startswith("plots_") or patient_dir.name in ("meta_synergy_clustering",):
            continue
        patient_id = patient_dir.name
        for task_dir in sorted(patient_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_upper = task_dir.name.upper()
            session = "T0" if "T0" in task_upper else "T1" if "T1" in task_upper else ""
            condition = "SN" if "SN" in task_upper else "DS" if "DS" in task_upper else ""
            if session and condition:
                discovered.append((patient_id, task_dir.name, session, condition, task_dir))
    return discovered


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def read_table(path: Path) -> pd.DataFrame:
    """Read CSV or Excel file."""
    if not path.exists():
        return pd.DataFrame()
    try:
        if path.suffix.lower() in (".xlsx", ".xls"):
            return pd.read_excel(path, engine="openpyxl")
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


# -----------------------------------------------------------------------------
# EMG: task-level + stability
# -----------------------------------------------------------------------------


def load_emg_task_metrics(emg_dir: Path) -> pd.DataFrame:
    """Load patient_task_metrics or aggregate from task_summary_metrics."""
    root = read_csv(emg_dir / "patient_task_metrics.csv")
    if not root.empty:
        return root
    # Fallback: collect from each task dir
    rows = []
    for patient_id, task_name, session, condition, task_dir in discover_task_dirs(emg_dir):
        ts = read_csv(task_dir / "task_summary_metrics.csv")
        if ts.empty:
            continue
        ts = ts.copy()
        ts["patient_id"] = patient_id
        ts["task_name"] = task_name
        ts["session"] = session
        ts["condition"] = condition
        rows.append(ts)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def load_meta_synergy_task_metrics(meta_dir: Path) -> pd.DataFrame:
    """
    Load B5 meta_task_summary_metrics.csv.
    Returns one row per (patient_id, task_name, session, condition).
    If file is missing, returns empty DataFrame (no crash).
    """
    path = meta_dir / "meta_task_summary_metrics.csv"
    if not path.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        if df.empty or "patient_id" not in df.columns or "task_name" not in df.columns:
            return pd.DataFrame()
        return df
    except Exception:
        return pd.DataFrame()


def _prefix_meta_synergy_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rename meta-synergy feature columns with emg_meta_C{idx}_* prefix.
    Keeps key columns (patient_id, task_name, session, condition) and excludes bookkeeping.
    """
    BOOKKEEPING = {"n_synergies_task", "n_meta_clusters"}
    KEY_COLS = {"patient_id", "task_name", "session", "condition"}
    rename_map = {}
    for c in df.columns:
        if c in KEY_COLS or c in BOOKKEEPING:
            continue
        if c.startswith("mean_meta_synergy_"):
            # mean_meta_synergy_0_mean -> emg_meta_C0_mean
            m = re.match(r"mean_meta_synergy_(\d+)_(.*)", c)
            if m:
                idx, suffix = m.group(1), m.group(2)
                rename_map[c] = f"emg_meta_C{idx}_{suffix}"
    return df.rename(columns=rename_map)


def load_emg_stability_wide(emg_dir: Path) -> pd.DataFrame:
    """Load stability_summary_metrics and pivot to wide (one row per task)."""
    rows = []
    for patient_id, task_name, session, condition, task_dir in discover_task_dirs(emg_dir):
        df = read_csv(task_dir / "stability_summary_metrics.csv")
        if df.empty:
            continue
        row = {"patient_id": patient_id, "task_name": task_name, "session": session, "condition": condition}
        for _, r in df.iterrows():
            muscle = slugify(str(r.get("muscle", "")))
            if not muscle:
                continue
            for col in ["cv_mean_amp", "cv_peak_amp", "cv_auc", "mean_similarity_to_mean_cycle", "std_similarity_to_mean_cycle"]:
                if col in r:
                    row[f"emg_stability_{muscle}_{col}"] = r[col]
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Demographics (optional)
# -----------------------------------------------------------------------------


def load_demographics(path: Optional[Path]) -> pd.DataFrame:
    """Load demographics CSV or Excel. Must have id (or patient_id) column."""
    if path is None or not path.exists():
        return pd.DataFrame()
    df = read_table(path)
    if df.empty:
        return pd.DataFrame()
    if "id" not in df.columns:
        if "patient_id" in df.columns:
            df = df.rename(columns={"patient_id": "id"})
        elif "ID" in df.columns:
            df = df.rename(columns={"ID": "id"})
        else:
            return pd.DataFrame()
    return df


# -----------------------------------------------------------------------------
# EEG (no condition: repeat for SN and DS)
# -----------------------------------------------------------------------------


def load_eeg_expanded(path: Optional[Path]) -> pd.DataFrame:
    """
    Load EEG features. EEG has no condition (SN/DS); IDs like CROSS_001_F.
    Expand: for each (patient_id, time) create one row per condition (SN, DS)
    with the same EEG features.
    """
    if path is None or not path.exists():
        return pd.DataFrame()
    df = read_csv(path)
    if df.empty or "ID" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    # Parse patient_id from ID (CROSS_001_F -> CROSS_001)
    df["patient_id"] = df["ID"].astype(str).str.replace(r"_F$|_M$", "", regex=True)
    if "timing" in df.columns:
        df["time"] = df["timing"]
    elif "time" not in df.columns:
        return pd.DataFrame()
    feat_cols = [c for c in df.columns if c not in ("ID", "patient_id", "timing", "time")]
    rows = []
    for _, r in df.iterrows():
        for cond in ["SN", "DS"]:
            row = {"patient_id": r["patient_id"], "time": r["time"], "condition": cond}
            for fc in feat_cols:
                row[f"eeg_{fc}"] = r[fc]
            rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# IMU: pivot to wide per (patient, session, condition)
# -----------------------------------------------------------------------------


# IMU feature columns (B6 output; supports both legacy and revised non-amplitude features)
IMU_FEATURE_COLS = [
    "cycle_duration_s", "dominant_frequency_hz", "spectral_entropy",
    "normalized_rms_derivative", "lag1_autocorr", "permutation_entropy",
    # Legacy (B6 pre-refactor):
    "range_span", "iqr", "std", "rms_derivative",
]


def load_imu_wide(imu_dir: Path) -> pd.DataFrame:
    """Load imu_features_per_task and pivot to wide: imu_<sensor>_<axis>_<feat>."""
    path = imu_dir / "summary_tables" / "imu_features_per_task_sensor_axis.csv"
    if not path.exists():
        path = imu_dir / "summary_tables" / "imu_features_per_task.csv"
    if not path.exists():
        path = imu_dir / "imu_features_per_task.csv"
    df = read_csv(path)
    if df.empty or "sensor" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["sensor_slug"] = df["sensor"].apply(lambda x: slugify(str(x).replace(": Acc", "Acc").replace("Acc ", "Acc")))
    feat_cols = [c for c in df.columns if c in IMU_FEATURE_COLS]
    if not feat_cols:
        return pd.DataFrame()
    # Pivot: one row per (patient_id, task_name, session, condition)
    wide_parts = []
    for (pid, task, sess, cond), grp in df.groupby(["patient_id", "task_name", "session", "condition"]):
        row = {"patient_id": pid, "task_name": task, "session": sess, "condition": cond}
        for _, r in grp.iterrows():
            base = f"imu_{r['sensor_slug']}_{r['axis']}"
            for fc in feat_cols:
                if fc in r:
                    row[f"{base}_{fc}"] = r[fc]
        wide_parts.append(row)
    if not wide_parts:
        return pd.DataFrame()
    return pd.DataFrame(wide_parts)


def load_imu_aggregated(imu_dir: Path) -> pd.DataFrame:
    """Load imu_features_per_task, aggregate mean across sensors/axes per task."""
    path = imu_dir / "summary_tables" / "imu_features_per_task_sensor_axis.csv"
    if not path.exists():
        path = imu_dir / "summary_tables" / "imu_features_per_task.csv"
    if not path.exists():
        path = imu_dir / "imu_features_per_task.csv"
    df = read_csv(path)
    if df.empty:
        return pd.DataFrame()
    present = [c for c in IMU_FEATURE_COLS if c in df.columns]
    if not present:
        return pd.DataFrame()
    agg_dict = {c: "mean" for c in present}
    agg = df.groupby(["patient_id", "task_name", "session", "condition"], as_index=False).agg(agg_dict)
    agg.columns = ["patient_id", "task_name", "session", "condition"] + [f"imu_{c}" for c in present]
    return agg


# -----------------------------------------------------------------------------
# Merge and save
# -----------------------------------------------------------------------------


def merge_all(
    emg_dir: Path,
    imu_dir: Path,
    eeg_path: Optional[Path] = None,
    demographic_path: Optional[Path] = None,
    imu_wide: bool = False,
) -> pd.DataFrame:
    """
    Merge EMG + stability + IMU + EEG + demographics.
    Key: (patient_id, task_name, session, condition).
    EEG: no condition - repeated for both SN and DS.
    Demographics: merged on id.
    """
    emg = load_emg_task_metrics(emg_dir)
    if emg.empty:
        return pd.DataFrame()
    emg_base = emg.copy()

    stability = load_emg_stability_wide(emg_dir)
    if not stability.empty:
        merge_cols = ["patient_id", "task_name", "session", "condition"]
        st_cols = [c for c in stability.columns if c not in merge_cols]
        emg_base = emg_base.merge(stability[merge_cols + st_cols], on=merge_cols, how="left")

    # B5 meta-synergy task metrics (fixed-K comparable features)
    meta_dir = emg_dir / "meta_synergy_clustering"
    meta_synergy = load_meta_synergy_task_metrics(meta_dir)
    if not meta_synergy.empty:
        meta_prefixed = _prefix_meta_synergy_columns(meta_synergy)
        merge_cols = ["patient_id", "task_name", "session", "condition"]
        meta_feat = [c for c in meta_prefixed.columns if c not in merge_cols and c not in ("n_synergies_task", "n_meta_clusters")]
        if meta_feat:
            emg_base = emg_base.merge(
                meta_prefixed[merge_cols + meta_feat],
                on=merge_cols,
                how="left",
            )

    if imu_wide:
        imu = load_imu_wide(imu_dir)
    else:
        imu = load_imu_aggregated(imu_dir)
    if not imu.empty:
        merge_cols = ["patient_id", "task_name", "session", "condition"]
        imu_add = [c for c in imu.columns if c not in merge_cols]
        emg_base = emg_base.merge(imu[merge_cols + imu_add], on=merge_cols, how="left")

    # EEG: expand for both SN and DS, merge on (patient_id, time, condition)
    eeg = load_eeg_expanded(eeg_path)
    if not eeg.empty:
        eeg_add = [c for c in eeg.columns if c not in ("patient_id", "time", "condition")]
        emg_base = emg_base.merge(
            eeg[["patient_id", "time", "condition"] + eeg_add],
            left_on=["patient_id", "session", "condition"],
            right_on=["patient_id", "time", "condition"],
            how="left",
        )
        emg_base = emg_base.drop(columns=["time"], errors="ignore")  # keep session, same as time

    # Demographics: merge on id (patient_id)
    demo = load_demographics(demographic_path)
    if not demo.empty:
        demo_cols = [c for c in demo.columns if c != "id"]
        demo_sub = demo[["id"] + demo_cols] if demo_cols else demo[["id"]]
        emg_base = emg_base.merge(demo_sub, left_on="patient_id", right_on="id", how="left")
        emg_base = emg_base.drop(columns=["id"], errors="ignore")

    # Rename to id, time, condition
    out = emg_base.copy()
    out = out.rename(columns={"patient_id": "id", "session": "time"})
    if "id" not in out.columns and "patient_id" in out.columns:
        out = out.rename(columns={"patient_id": "id"})
    first = ["id", "time", "condition"]
    rest = [c for c in out.columns if c not in first]
    out = out[[c for c in first if c in out.columns] + rest]
    return out


# -----------------------------------------------------------------------------
# Column resolution for analysis
# -----------------------------------------------------------------------------

# Aliases for "lato più colpito" (affected side)
AFFECTED_SIDE_ALIASES = [
    "lato piu colpito",
    "lato_piu_colpito",
    "lato più colpito",
    "lato piu colpito",
    "latopiucolpito",
    "affected_side",
    "affected side",
]
DOMINANCE_ALIASES = ["dominanza", "dominance", "dominant_side", "dominant side"]


def _resolve_analysis_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Resolve id, time, condition, Dominanza, lato più colpito. Returns {logical_name: actual_col}."""
    out: Dict[str, str] = {}
    id_col = _find_column(df, ["id", "patient_id", "patient id", "ID"])
    if id_col:
        out["id"] = id_col
    time_col = _find_column(df, ["time", "timing", "session", "Time"])
    if time_col:
        out["time"] = time_col
    cond_col = _find_column(df, ["condition", "cond", "Condition"])
    if cond_col:
        out["condition"] = cond_col
    dom_col = _find_column(df, ["dominanza", "dominance"], DOMINANCE_ALIASES)
    if dom_col:
        out["dominanza"] = dom_col
    aff_col = _find_column(df, ["lato più colpito", "lato piu colpito"], AFFECTED_SIDE_ALIASES)
    if aff_col:
        out["lato_piu_colpito"] = aff_col
    return out


# -----------------------------------------------------------------------------
# Metric families
# -----------------------------------------------------------------------------


def get_metric_groups(df: pd.DataFrame) -> Dict[str, List[str]]:
    """
    Classify numeric columns into families.
    Returns dict: family_name -> [list of column names].
    Excludes constant, all-NaN, ID-like, and meta columns.
    """
    META = {"id", "patient_id", "time", "timing", "session", "condition", "task_name", "cond"}
    res = {
        "EEG": [],
        "IMU": [],
        "emg_stability": [],
        "emg_meta_synergy": [],
        "emg_movement": [],
    }
    excluded: Dict[str, str] = {}  # col -> reason

    for col in df.columns:
        cnorm = _normalize_col_name(col)
        if cnorm in META or any(m in cnorm for m in ["patient", "task_name", "segment"]):
            excluded[col] = "meta_id"
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            excluded[col] = "non_numeric"
            continue
        if df[col].isna().all():
            excluded[col] = "all_missing"
            continue
        if df[col].nunique(dropna=True) < 2:
            excluded[col] = "constant"
            continue

        if col.startswith("eeg_"):
            res["EEG"].append(col)
        elif col.startswith("imu_"):
            res["IMU"].append(col)
        elif col.startswith("emg_stability_"):
            res["emg_stability"].append(col)
        elif col.startswith("emg_meta_"):
            res["emg_meta_synergy"].append(col)
        else:
            res["emg_movement"].append(col)

    # Store excluded in a shared structure for metric_groups sheet
    get_metric_groups._excluded = excluded  # type: ignore
    return res


def build_metric_groups_sheet(groups: Dict[str, List[str]]) -> pd.DataFrame:
    """Build metric_groups sheet: metric, metric_family, included_in_stats, exclusion_reason."""
    excluded = getattr(get_metric_groups, "_excluded", {})
    rows = []
    for family, cols in groups.items():
        for c in cols:
            rows.append({"metric": c, "metric_family": family, "included_in_stats": True, "exclusion_reason": ""})
    for c, reason in excluded.items():
        rows.append({"metric": c, "metric_family": "", "included_in_stats": False, "exclusion_reason": reason})
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# EEG deduplication and reshaping
# -----------------------------------------------------------------------------


def build_eeg_unique(
    df: pd.DataFrame,
    eeg_cols: List[str],
    id_col: str,
    time_col: str,
    strict: bool = True,
) -> pd.DataFrame:
    """
    One row per (id, time). Verifies EEG identical across SN/DS; if not, raises if strict.
    """
    if not eeg_cols:
        return pd.DataFrame(columns=[id_col, time_col])

    keep = [id_col, time_col] + eeg_cols
    present = [c for c in keep if c in df.columns]
    if len(present) < 3:
        return pd.DataFrame()

    sub = df[present].drop_duplicates(subset=[id_col, time_col], keep="first")
    for (i, t), grp in df.groupby([id_col, time_col]):
        if len(grp) < 2:
            continue
        vals = grp[eeg_cols].astype(float)
        if vals.isna().all().all():
            continue
        if not vals.iloc[0].equals(vals.iloc[1]):
            diff = (vals.iloc[0] - vals.iloc[1]).abs()
            bad = diff[diff > 1e-9]
            if strict and not bad.empty:
                raise ValueError(
                    f"EEG should be condition-independent but differs for ({i}, {t}). "
                    f"Sample cols: {list(bad.index[:3])}. Use --no-strict-eeg-check to relax."
                )
            warnings.warn(f"EEG differs for ({i}, {t}); keeping first row.")

    return sub


def build_eeg_wide(eeg_long: pd.DataFrame, eeg_cols: List[str], id_col: str, time_col: str) -> pd.DataFrame:
    """Wide EEG: one row per id, columns like eeg_X__T0, eeg_X__T1."""
    if eeg_long.empty or not eeg_cols:
        return pd.DataFrame()

    pivot = eeg_long.pivot(index=id_col, columns=time_col, values=eeg_cols)
    if pivot.columns.nlevels == 1:
        pivot.columns = [f"{c}__{pivot.columns.name}" for c in pivot.columns]
    else:
        flat = []
        for c in pivot.columns:
            # Pivot MultiIndex: (values_col, columns_val) or (columns_val, values_col) depending on pandas
            # We want metric__time (e.g. eeg_rel_alpha__T0)
            m, t = c[0], c[1]
            if str(t) in ("T0", "T1") and str(m).startswith("eeg_"):
                flat.append(f"{m}__{t}")
            elif str(m) in ("T0", "T1") and str(t).startswith("eeg_"):
                flat.append(f"{t}__{m}")
            else:
                flat.append(f"{m}__{t}")
        pivot.columns = flat
    return pivot.reset_index()


# -----------------------------------------------------------------------------
# Movement side recoding
# -----------------------------------------------------------------------------


def condition_to_performed_side(condition: str) -> str:
    """DS -> DX, SN -> SX."""
    c = str(condition).strip().upper()
    if c == "DS":
        return "DX"
    if c == "SN":
        return "SX"
    return c if c in ("DX", "SX") else ""


def build_movement_recoded(
    df: pd.DataFrame,
    movement_cols: List[str],
    col_map: Dict[str, str],
    use_affected: bool,
) -> pd.DataFrame:
    """
    use_affected=True -> recoded_group in (more_affected, less_affected)
    use_affected=False -> recoded_group in (dominant, non_dominant)
    """
    id_col = col_map.get("id")
    time_col = col_map.get("time")
    cond_col = col_map.get("condition")
    if not all([id_col, time_col, cond_col]):
        return pd.DataFrame()

    if use_affected:
        aff_col = col_map.get("lato_piu_colpito")
        if not aff_col or aff_col not in df.columns:
            return pd.DataFrame()
    else:
        dom_col = col_map.get("dominanza")
        if not dom_col or dom_col not in df.columns:
            return pd.DataFrame()

    base = [id_col, time_col, cond_col]
    if use_affected:
        base.extend([aff_col])
    else:
        base.extend([dom_col])

    select = [c for c in base + movement_cols if c in df.columns]
    sub = df[select].copy()
    sub["performed_side"] = sub[cond_col].apply(condition_to_performed_side)

    def safe_norm(val: Any) -> str:
        v = _normalize_side(val)
        if v not in ("DX", "SX"):
            return ""
        return v

    if use_affected:
        sub["affected_side_orig"] = sub[aff_col].apply(safe_norm)
        # Exclude rows with invalid affected side (cannot classify)
        sub = sub[sub["affected_side_orig"] != ""].copy()
        sub["recoded_group"] = sub.apply(
            lambda r: "more_affected" if r["performed_side"] == r["affected_side_orig"] else "less_affected",
            axis=1,
        )
    else:
        sub["dominance_orig"] = sub[dom_col].apply(safe_norm)
        # Exclude rows with invalid dominant side (cannot classify)
        sub = sub[sub["dominance_orig"] != ""].copy()
        sub["recoded_group"] = sub.apply(
            lambda r: "dominant" if r["performed_side"] == r["dominance_orig"] else "non_dominant",
            axis=1,
        )

    # Verify: each (id, time) has one per recoded group
    for (i, t), grp in sub.groupby([id_col, time_col]):
        uniq = grp["recoded_group"].dropna().unique()
        expected = {"more_affected", "less_affected"} if use_affected else {"dominant", "non_dominant"}
        if set(uniq) != expected:
            missing = expected - set(uniq)
            invalid = grp["recoded_group"].isna() | (grp["recoded_group"] == "")
            if invalid.any():
                warnings.warn(
                    f"Subject {i}, time {t}: recoded_group has invalid/missing values. "
                    f"Expected {expected}, got {list(uniq)}. Check Dominanza / lato più colpito."
                )

    return sub


def build_movement_wide(long_df: pd.DataFrame, metrics: List[str], id_col: str, time_col: str, group_col: str) -> pd.DataFrame:
    """Pivot: metric__group__T0, metric__group__T1."""
    if long_df.empty or not metrics:
        return pd.DataFrame()
    present = [c for c in [id_col, time_col, group_col] + metrics if c in long_df.columns]
    if len(present) < 4:
        return pd.DataFrame()
    sub = long_df[present].copy()
    rows = []
    for idx, grp in sub.groupby(id_col):
        row = {id_col: idx}
        for m in metrics:
            if m not in grp.columns:
                continue
            for (g, t), v in grp.groupby([group_col, time_col])[m].first().items():
                row[f"{m}__{g}__{t}"] = v
        rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Statistical helpers
# -----------------------------------------------------------------------------

EPS = 1e-12


def is_binary(t0: np.ndarray, t1: np.ndarray) -> bool:
    """Check if paired values are binary (at most 2 distinct values)."""
    t0 = np.asarray(t0, dtype=float)
    t1 = np.asarray(t1, dtype=float)
    mask = np.isfinite(t0) & np.isfinite(t1)
    combined = np.concatenate([t0[mask], t1[mask]])
    if len(combined) < 2:
        return False
    uniq = np.unique(combined)
    return len(uniq) <= 2


def is_presence_metric(metric_name: str) -> bool:
    """True if metric is a presence variable (0/1, use counts/percentages + chi2)."""
    return "_presence" in str(metric_name).lower()


def paired_presence_stats(T0: np.ndarray, T1: np.ndarray) -> Dict[str, Any]:
    """
    For presence variables: counts, percentages, McNemar chi-square test.
    Presence=1, absent=0. Returns count_present_T0/T1, count_absent_T0/T1, pct_present_T0/T1.
    """
    t0v = np.asarray(T0, dtype=float)
    t1v = np.asarray(T1, dtype=float)
    mask = np.isfinite(t0v) & np.isfinite(t1v)
    t0v = t0v[mask]
    t1v = t1v[mask]
    n = int(len(t0v))
    if n < 2:
        return {
            "n_pairs": n,
            "count_present_T0": np.nan,
            "count_absent_T0": np.nan,
            "pct_present_T0": np.nan,
            "count_present_T1": np.nan,
            "count_absent_T1": np.nan,
            "pct_present_T1": np.nan,
            "mean_T0": np.nan,
            "sd_T0": np.nan,
            "median_T0": np.nan,
            "mean_T1": np.nan,
            "sd_T1": np.nan,
            "median_T1": np.nan,
            "mean_change": np.nan,
            "sd_change": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "t_stat": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "notes": "too_few_pairs",
            "test_type": "chi2",
        }
    # 1 = present, 0 = absent (standard for presence variables)
    v0 = (t0v > 0.5).astype(int)
    v1 = (t1v > 0.5).astype(int)
    count_present_T0 = int(np.sum(v0))
    count_absent_T0 = n - count_present_T0
    count_present_T1 = int(np.sum(v1))
    count_absent_T1 = n - count_present_T1
    pct_present_T0 = 100.0 * count_present_T0 / n if n > 0 else np.nan
    pct_present_T1 = 100.0 * count_present_T1 / n if n > 0 else np.nan
    mean_change = (pct_present_T1 - pct_present_T0) / 100.0
    # McNemar chi-square for paired 2x2
    b = int(np.sum((v0 == 0) & (v1 == 1)))
    c = int(np.sum((v0 == 1) & (v1 == 0)))
    if b + c == 0:
        chi2, p_val = 0.0, 1.0
    else:
        chi2 = (abs(b - c) - 1) ** 2 / (b + c)
        p_val = float(scipy_stats.chi2.sf(chi2, 1))
    return {
        "n_pairs": n,
        "count_present_T0": count_present_T0,
        "count_absent_T0": count_absent_T0,
        "pct_present_T0": pct_present_T0,
        "count_present_T1": count_present_T1,
        "count_absent_T1": count_absent_T1,
        "pct_present_T1": pct_present_T1,
        "mean_T0": pct_present_T0 / 100.0,
        "sd_T0": np.nan,
        "median_T0": np.nan,
        "mean_T1": pct_present_T1 / 100.0,
        "sd_T1": np.nan,
        "median_T1": np.nan,
        "mean_change": mean_change,
        "sd_change": np.nan,
        "ci95_low": np.nan,
        "ci95_high": np.nan,
        "t_stat": float(chi2),
        "df": 1.0,
        "p_value": float(p_val),
        "cohens_dz": np.sqrt(chi2 / n) if n > 0 and np.isfinite(chi2) else np.nan,
        "notes": "",
        "test_type": "chi2",
    }


def paired_mcnemar(T0: np.ndarray, T1: np.ndarray) -> Dict[str, Any]:
    """
    McNemar's test for paired binary data (T0 vs T1).
    Returns same structure as paired_ttest: t_stat holds chi2, cohens_dz holds phi.
    """
    t0v = np.asarray(T0, dtype=float)
    t1v = np.asarray(T1, dtype=float)
    mask = np.isfinite(t0v) & np.isfinite(t1v)
    t0v = t0v[mask]
    t1v = t1v[mask]
    n = int(len(t0v))
    if n < 2:
        return {
            "n_pairs": n,
            "mean_T0": np.nan,
            "sd_T0": np.nan,
            "median_T0": np.nan,
            "mean_T1": np.nan,
            "sd_T1": np.nan,
            "median_T1": np.nan,
            "mean_change": np.nan,
            "sd_change": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "t_stat": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "cohens_dz": np.nan,
            "notes": "too_few_pairs" if n < 2 else "ok",
            "test_type": "mcnemar",
        }
    # Map to 0/1 (smaller value -> 0)
    uniq = np.unique(np.concatenate([t0v, t1v]))
    v0 = (t0v != uniq[0]).astype(int)  # len(uniq)==1 gives all zeros
    v1 = (t1v != uniq[0]).astype(int)
    # Contingency: rows=T0 (0/1), cols=T1 (0/1)
    a = np.sum((v0 == 0) & (v1 == 0))
    b = np.sum((v0 == 0) & (v1 == 1))
    c = np.sum((v0 == 1) & (v1 == 0))
    d = np.sum((v0 == 1) & (v1 == 1))
    table = np.array([[a, b], [c, d]])
    mean_T0 = float(np.mean(v0))
    mean_T1 = float(np.mean(v1))
    mean_change = mean_T1 - mean_T0
    sd_T0 = float(np.std(v0, ddof=1)) if n > 1 else 0.0
    sd_T1 = float(np.std(v1, ddof=1)) if n > 1 else 0.0
    median_T0 = float(np.median(v0))
    median_T1 = float(np.median(v1))
    try:
        # McNemar: chi2 = (|b-c|-1)^2/(b+c), df=1 (continuity correction)
        if b + c == 0:
            chi2, p_val = 0.0, 1.0
        else:
            chi2 = (abs(b - c) - 1) ** 2 / (b + c)
            p_val = float(scipy_stats.chi2.sf(chi2, 1))  # upper tail
        phi = np.sqrt(chi2 / n) if n > 0 and np.isfinite(chi2) else np.nan
        # CI for proportion difference (simplified Wilson-style for paired)
        se_diff = np.sqrt((mean_T0 * (1 - mean_T0) + mean_T1 * (1 - mean_T1) - 2 * (np.mean(v0 * v1) - mean_T0 * mean_T1)) / n) if n > 0 else np.nan
        if np.isfinite(se_diff) and se_diff > EPS:
            ci95_low = mean_change - 1.96 * se_diff
            ci95_high = mean_change + 1.96 * se_diff
        else:
            ci95_low = ci95_high = np.nan
    except Exception:
        chi2, p_val, phi = np.nan, np.nan, np.nan
        ci95_low, ci95_high = np.nan, np.nan
    diff = v1.astype(float) - v0.astype(float)
    sd_change = float(np.std(diff, ddof=1)) if n > 1 else 0.0
    return {
        "n_pairs": n,
        "mean_T0": mean_T0,
        "sd_T0": sd_T0,
        "median_T0": median_T0,
        "mean_T1": mean_T1,
        "sd_T1": sd_T1,
        "median_T1": median_T1,
        "mean_change": mean_change,
        "sd_change": sd_change,
        "ci95_low": float(ci95_low) if np.isfinite(ci95_low) else np.nan,
        "ci95_high": float(ci95_high) if np.isfinite(ci95_high) else np.nan,
        "t_stat": float(chi2),
        "df": 1.0,
        "p_value": float(p_val),
        "cohens_dz": float(phi),
        "notes": "",
        "test_type": "mcnemar",
    }


def paired_ttest_or_chi2(T0: np.ndarray, T1: np.ndarray) -> Dict[str, Any]:
    """Use paired t-test for continuous, McNemar for binary. Same output structure."""
    if is_binary(T0, T1):
        return paired_mcnemar(T0, T1)
    res = paired_ttest(T0, T1)
    res["test_type"] = "paired_t"
    return res


def paired_ttest(T0: np.ndarray, T1: np.ndarray) -> Dict[str, Any]:
    """
    Paired T0 vs T1. Returns dict with mean_T0, mean_T1, mean_change, sd_change,
    ci95_low, ci95_high, t_stat, df, p_value, cohens_dz, notes.
    """
    diff = np.asarray(T1, dtype=float) - np.asarray(T0, dtype=float)
    mask = np.isfinite(diff)
    n = int(np.sum(mask))
    if n < 2:
        return {
            "n_pairs": n,
            "mean_T0": np.nan,
            "sd_T0": np.nan,
            "median_T0": np.nan,
            "mean_T1": np.nan,
            "sd_T1": np.nan,
            "median_T1": np.nan,
            "mean_change": np.nan,
            "sd_change": np.nan,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "t_stat": np.nan,
            "df": np.nan,
            "p_value": np.nan,
            "cohens_dz": np.nan,
            "notes": "too_few_pairs" if n < 2 else "ok",
            "test_type": "paired_t",
        }
    d = diff[mask]
    t0v = np.asarray(T0, dtype=float)[mask]
    t1v = np.asarray(T1, dtype=float)[mask]
    mean_change = float(np.mean(d))
    sd_change = float(np.std(d, ddof=1))
    if sd_change < EPS:
        return {
            "n_pairs": n,
            "mean_T0": float(np.mean(t0v)),
            "sd_T0": float(np.std(t0v, ddof=1)),
            "median_T0": float(np.median(t0v)),
            "mean_T1": float(np.mean(t1v)),
            "sd_T1": float(np.std(t1v, ddof=1)),
            "median_T1": float(np.median(t1v)),
            "mean_change": mean_change,
            "sd_change": sd_change,
            "ci95_low": np.nan,
            "ci95_high": np.nan,
            "t_stat": np.nan,
            "df": n - 1,
            "p_value": np.nan,
            "cohens_dz": np.nan,
            "notes": "zero_variance_in_difference",
        }
    t_stat, p_val = scipy_stats.ttest_rel(t1v, t0v)
    se = sd_change / np.sqrt(n)
    t_crit = scipy_stats.t.ppf(0.975, n - 1)
    return {
        "n_pairs": n,
        "mean_T0": float(np.mean(t0v)),
        "sd_T0": float(np.std(t0v, ddof=1)),
        "median_T0": float(np.median(t0v)),
        "mean_T1": float(np.mean(t1v)),
        "sd_T1": float(np.std(t1v, ddof=1)),
        "median_T1": float(np.median(t1v)),
        "mean_change": mean_change,
        "sd_change": sd_change,
        "ci95_low": mean_change - t_crit * se,
        "ci95_high": mean_change + t_crit * se,
        "t_stat": float(t_stat),
        "df": n - 1,
        "p_value": float(p_val),
        "cohens_dz": mean_change / sd_change,
        "notes": "",
    }


def apply_fdr(pvals: np.ndarray, method: str = "fdr_bh") -> np.ndarray:
    """Benjamini-Hochberg FDR correction."""
    mask = np.isfinite(pvals)
    out = np.full_like(pvals, np.nan)
    if mask.sum() == 0:
        return out
    rej, pcorr, _, _ = multipletests(pvals[mask], alpha=0.05, method=method)
    out[mask] = pcorr
    return out


# -----------------------------------------------------------------------------
# Build all stats tables
# -----------------------------------------------------------------------------


def run_eeg_stats(eeg_wide: pd.DataFrame, eeg_cols: List[str], id_col: str) -> pd.DataFrame:
    """One row per EEG metric: paired T0 vs T1."""
    if eeg_wide.empty or not eeg_cols:
        return pd.DataFrame()
    rows = []
    for m in eeg_cols:
        t0_col = f"{m}__T0"
        t1_col = f"{m}__T1"
        if t0_col not in eeg_wide.columns or t1_col not in eeg_wide.columns:
            continue
        t0 = eeg_wide[t0_col].values
        t1 = eeg_wide[t1_col].values
        res = paired_presence_stats(t0, t1) if is_presence_metric(m) else paired_ttest_or_chi2(t0, t1)
        rows.append({
            "metric": m,
            "metric_family": "EEG",
            "analysis_domain": "EEG",
            "recoding_framework": "none",
            "contrast": "T0_vs_T1",
            "group_label": "all",
            "direction_label": "increase" if res["mean_change"] > 0 else "decrease" if res["mean_change"] < 0 else "none",
            **res,
        })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    pvals = df["p_value"].values
    df["p_value_fdr"] = apply_fdr(pvals)
    return df


def run_movement_paired_stats(
    long_df: pd.DataFrame,
    metrics: List[str],
    col_map: Dict[str, str],
    framework: str,
    id_col: str,
    time_col: str,
    group_col: str,
    allowed_groups: Optional[Tuple[str, ...]] = None,
) -> pd.DataFrame:
    """One row per metric per group. Default groups: more_affected/less_affected or dominant/non_dominant.
    Use allowed_groups=('SN','DS') for condition-level stats (matches meta_synergy timecourse)."""
    if long_df.empty or not metrics:
        return pd.DataFrame()
    if allowed_groups is None:
        allowed_groups = ("more_affected", "less_affected", "dominant", "non_dominant")
    rows = []
    for m in metrics:
        if m not in long_df.columns:
            continue
        for grp in long_df[group_col].dropna().unique():
            if grp not in allowed_groups:
                continue
            sub = long_df[(long_df[group_col] == grp) & long_df[m].notna()]
            if sub.empty:
                continue
            pivot = sub.pivot_table(index=id_col, columns=time_col, values=m, aggfunc="first")
            if "T0" not in pivot.columns or "T1" not in pivot.columns:
                continue
            t0 = pivot["T0"].values
            t1 = pivot["T1"].values
            res = paired_presence_stats(t0, t1) if is_presence_metric(m) else paired_ttest_or_chi2(t0, t1)
            rows.append({
                "metric": m,
                "metric_family": _infer_family(m),
                "analysis_domain": "movement",
                "recoding_framework": framework,
                "contrast": "T0_vs_T1",
                "group_label": grp,
                "direction_label": "increase" if res["mean_change"] and res["mean_change"] > 0 else "decrease" if res["mean_change"] and res["mean_change"] < 0 else "none",
                **res,
            })
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["p_value_fdr"] = np.nan
    for grp in df["group_label"].unique():
        mask = df["group_label"] == grp
        df.loc[mask, "p_value_fdr"] = apply_fdr(df.loc[mask, "p_value"].values)
    return df


def _infer_family(m: str) -> str:
    if m.startswith("emg_stability_"):
        return "emg_stability"
    if m.startswith("emg_meta_"):
        return "emg_meta_synergy"
    if m.startswith("imu_"):
        return "IMU"
    return "emg_movement"


def main():
    ap = argparse.ArgumentParser(description="Merge EMG + IMU features into one Excel")
    ap.add_argument("--results-dir", type=Path, default=Path("results"), help="Root results dir (EMG in {dir}/synergies, IMU in {dir}/imu_features)")
    ap.add_argument("--emg-dir", type=Path, default=None, help="Override: EMG/synergy outputs (default: {results-dir}/synergies)")
    ap.add_argument("--imu-dir", type=Path, default=None, help="Override: IMU outputs (default: {results-dir}/imu_features)")
    ap.add_argument("--eeg", type=Path, default=None, help="EEG CSV (default: eeg_combined_features.csv); no condition, repeated for SN/DS")
    ap.add_argument("--demographic", type=Path, default=None, help="Demographics CSV/Excel (default: demografica.xlsx); must have id column")
    ap.add_argument("--out", type=Path, default=None, help="Output Excel (default: {results-dir}/merged_all_features.xlsx)")
    ap.add_argument("--stats", action="store_true", help="Run reshaping + statistical analysis; write full workbook")
    ap.add_argument("--strict-eeg-check", action="store_true", default=True, help="Strict EEG identity check across SN/DS (default: True)")
    ap.add_argument("--no-strict-eeg-check", action="store_false", dest="strict_eeg_check", help="Relax EEG identity check")
    ap.add_argument("--alpha", type=float, default=0.05, help="Alpha for FDR (default: 0.05)")
    ap.add_argument("--fdr-method", type=str, default="fdr_bh", help="FDR method (default: fdr_bh)")
    ap.add_argument("--imu-wide", action="store_true", help="Include per-sensor IMU columns (many columns)")
    args = ap.parse_args()

    results_dir = args.results_dir
    emg_dir = args.emg_dir or (results_dir / "synergies")
    imu_dir = args.imu_dir or (results_dir / "imu_features")
    out_path = args.out or (results_dir / "merged_all_features.xlsx")
    eeg_path = args.eeg or Path("eeg_combined_features.csv")
    demo_path = args.demographic or Path("demografica.xlsx")

    # Resolve EEG path: try cwd, script dir, results_dir, results_dir.parent (project root)
    _eeg_resolved = None
    if eeg_path.exists():
        _eeg_resolved = eeg_path
    else:
        _script_dir = Path(__file__).resolve().parent
        for candidate in [
            _script_dir / eeg_path.name,
            results_dir / eeg_path.name,
            results_dir.resolve().parent / eeg_path.name,
        ]:
            if candidate.exists():
                _eeg_resolved = candidate
                break
    if _eeg_resolved is None:
        print(f"Note: EEG file not found ({eeg_path}); EEG features will not be included.")

    merged = merge_all(
        emg_dir, imu_dir,
        eeg_path=_eeg_resolved,
        demographic_path=demo_path if demo_path.exists() else args.demographic,
        imu_wide=args.imu_wide,
    )
    if merged.empty:
        print("No data to merge. Check --results-dir (EMG in synergies/, IMU in imu_features/).")
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)

    if args.stats:
        # Full analysis: reshaped datasets + stats
        out_path = out_path.with_name(out_path.stem + "_with_stats.xlsx")
        col_map = _resolve_analysis_columns(merged)
        id_col = col_map.get("id", "id")
        time_col = col_map.get("time", "time")
        cond_col = col_map.get("condition", "condition")

        groups = get_metric_groups(merged)
        eeg_cols = groups.get("EEG", [])

        # Diagnostic: why EEG might be missing from stats
        eeg_in_merged = [c for c in merged.columns if str(c).startswith("eeg_")]
        if eeg_in_merged and not eeg_cols:
            excluded = getattr(get_metric_groups, "_excluded", {})
            eeg_excluded = {c: excluded.get(c, "?") for c in eeg_in_merged[:5]}
            print(f"EEG columns present in merged ({len(eeg_in_merged)} total) but excluded from stats. Sample reasons: {eeg_excluded}")
        elif not eeg_in_merged and not eeg_cols:
            print("No EEG columns in merged. Check: (1) EEG file path and (2) patient_id/time/condition match (EMG: CROSS_001, EEG: CROSS_001 from CROSS_001_F).")

        movement_cols = (
            groups.get("emg_movement", [])
            + groups.get("emg_stability", [])
            + groups.get("emg_meta_synergy", [])
            + groups.get("IMU", [])
        )

        imu_cols = groups.get("IMU", [])
        if imu_cols:
            print(f"IMU columns in stats: {len(imu_cols)} (use --imu-wide for per-sensor-axis features)")
        elif any(c.startswith("imu_") for c in merged.columns):
            print("Note: IMU columns in merged but excluded from stats (e.g. all-missing, constant). Check metric_groups sheet.")

        # EEG dedup and reshape
        eeg_long_unique = pd.DataFrame()
        eeg_wide_paired = pd.DataFrame()
        if eeg_cols and id_col in merged.columns and time_col in merged.columns:
            try:
                eeg_long_unique = build_eeg_unique(
                    merged, eeg_cols, id_col, time_col, strict=args.strict_eeg_check
                )
                eeg_wide_paired = build_eeg_wide(eeg_long_unique, eeg_cols, id_col, time_col)
            except ValueError as e:
                print(f"EEG check failed: {e}")
                if args.strict_eeg_check:
                    return

        # Movement recoded datasets
        movement_affected_long = build_movement_recoded(
            merged, movement_cols, col_map, use_affected=True
        )
        movement_dominance_long = build_movement_recoded(
            merged, movement_cols, col_map, use_affected=False
        )
        # Diagnostic: n_pairs often differs because Dominanza vs lato più colpito have different missingness
        if not movement_affected_long.empty and not movement_dominance_long.empty:
            aff_ids = set(movement_affected_long[id_col].dropna().unique())
            dom_ids = set(movement_dominance_long[id_col].dropna().unique())
            if len(aff_ids) != len(dom_ids):
                only_aff = aff_ids - dom_ids
                only_dom = dom_ids - aff_ids
                if only_aff:
                    print(f"Note: {len(only_aff)} patient(s) in affected analysis but not dominance (Dominanza missing/invalid): {sorted(only_aff)}")
                if only_dom:
                    print(f"Note: {len(only_dom)} patient(s) in dominance but not affected (lato più colpito missing/invalid): {sorted(only_dom)}")
        # Diagnostic: patients in affected vs dominance
        ids_affected = set(movement_affected_long[id_col].dropna().unique()) if not movement_affected_long.empty else set()
        ids_dominance = set(movement_dominance_long[id_col].dropna().unique()) if not movement_dominance_long.empty else set()
        if ids_affected and ids_affected != ids_dominance:
            missing_dom = ids_affected - ids_dominance
            if missing_dom:
                print(f"Note: {len(missing_dom)} patient(s) in affected analysis but excluded from dominance: {sorted(missing_dom)}. Check Dominanza column in demographics.")
        group_col = "recoded_group"
        movement_affected_wide = (
            build_movement_wide(movement_affected_long, movement_cols, id_col, time_col, group_col)
            if not movement_affected_long.empty
            else pd.DataFrame()
        )
        movement_dominance_wide = (
            build_movement_wide(movement_dominance_long, movement_cols, id_col, time_col, group_col)
            if not movement_dominance_long.empty
            else pd.DataFrame()
        )

        # Condition-level (SN, DS) for direct comparison with meta_synergy timecourse
        cond_col = col_map.get("condition", "condition")
        movement_condition_long = pd.DataFrame()
        if cond_col in merged.columns and movement_cols:
            agg_cols = [c for c in [id_col, time_col, cond_col] + movement_cols if c in merged.columns]
            if len(agg_cols) >= 4:
                movement_condition_long = (
                    merged[agg_cols]
                    .groupby([id_col, time_col, cond_col], as_index=False)[movement_cols]
                    .mean()
                )

        # Stats tables
        stats_eeg = run_eeg_stats(eeg_wide_paired, eeg_cols, id_col) if not eeg_wide_paired.empty else pd.DataFrame()
        # EMG + IMU: paired T0 vs T1 in more_affected, less_affected, dominant, non_dominant
        stats_affected_paired = (
            run_movement_paired_stats(
                movement_affected_long, movement_cols, col_map,
                framework="affected", id_col=id_col, time_col=time_col, group_col=group_col,
            )
            if not movement_affected_long.empty
            else pd.DataFrame()
        )
        stats_dominance_paired = (
            run_movement_paired_stats(
                movement_dominance_long, movement_cols, col_map,
                framework="dominance", id_col=id_col, time_col=time_col, group_col=group_col,
            )
            if not movement_dominance_long.empty
            else pd.DataFrame()
        )
        stats_condition_paired = (
            run_movement_paired_stats(
                movement_condition_long, movement_cols, col_map,
                framework="condition", id_col=id_col, time_col=time_col, group_col=cond_col,
                allowed_groups=("SN", "DS"),
            )
            if not movement_condition_long.empty
            else pd.DataFrame()
        )
        # EEG: paired T0 vs T1 only
        stats_all = pd.concat(
            [df for df in [stats_eeg, stats_affected_paired, stats_dominance_paired, stats_condition_paired] if not df.empty],
            ignore_index=True,
        ) if any(not df.empty for df in [stats_eeg, stats_affected_paired, stats_dominance_paired]) else pd.DataFrame()

        metric_groups_sheet = build_metric_groups_sheet(groups)

        sheets = [
            ("merged_raw", merged),
            ("eeg_long_unique", eeg_long_unique),
            ("eeg_wide_paired", eeg_wide_paired),
            ("movement_by_affected_long", movement_affected_long),
            ("movement_by_affected_wide", movement_affected_wide),
            ("movement_by_dominance_long", movement_dominance_long),
            ("movement_by_dominance_wide", movement_dominance_wide),
            ("movement_by_condition_long", movement_condition_long),
            ("metric_groups", metric_groups_sheet),
            ("stats_eeg_paired", stats_eeg),
            ("stats_affected_paired", stats_affected_paired),
            ("stats_dominance_paired", stats_dominance_paired),
            ("stats_condition_paired", stats_condition_paired),
            ("stats_all", stats_all),
        ]
        with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
            for name, df in sheets:
                if df is not None and not df.empty:
                    name_safe = name[:31]
                    df.to_excel(writer, sheet_name=name_safe, index=False)
                    # Optional CSV export for stats tables
                    if name.startswith("stats_") and name != "stats_all":
                        csv_path = out_path.with_name(out_path.stem + f"_{name}.csv")
                        df.to_csv(csv_path, index=False)
                elif name == "merged_raw":
                    merged.to_excel(writer, sheet_name="merged_raw", index=False)
        print(f"Merged {len(merged)} rows; wrote full workbook with stats -> {out_path}")
    else:
        merged.to_excel(out_path, index=False, engine="openpyxl")
        print(f"Merged {len(merged)} rows, {len(merged.columns)} columns -> {out_path}")


if __name__ == "__main__":
    main()
