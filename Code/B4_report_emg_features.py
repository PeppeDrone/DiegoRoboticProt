#!/usr/bin/env python3
"""plot_paired_synergy_metrics.py

Reporting and plotting utility for paired EMG synergy analysis outputs.

Features:
  - metric-family aware paired reporting
  - paired slope plots and delta plots with effect sizes
  - FDR correction within family/condition
  - muscle-level heatmaps and selected-muscle small multiples
  - cycle profile plots for synergies and optional muscles
  - W heatmap comparisons (T0, T1, delta)
  - H timing/amplitude summary plots
  - coordination plots: CCI slope plots, correlation heatmaps

Usage:
  python plot_paired_synergy_metrics.py --results-dir results/synergies
  python plot_paired_synergy_metrics.py --results-dir results/synergies --selected-muscles "Anterior Deltoid" "Posterior Deltoid"
"""

from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment


EPS = 1e-12
PROFILE_NPTS = 101
DEFAULT_SELECTED_MUSCLES = [
    "Anterior Deltoid",
    "Posterior Deltoid",
    "Biceps Brachii Long Head",
    "Triceps Brachii Long Head",
    "Brachioradialis",
]
DEFAULT_SELECTED_SYNERGIES = [0, 1, 2]
DEFAULT_CONDITIONS = ["SN", "DS"]

ROOT_METRIC_SPECS = [
    {
        "key": "global_vaf",
        "family": "synergy",
        "kind": "paired",
        "t0": "vaf_T0",
        "t1": "vaf_T1",
        "ylabel": "Mean global VAF",
        "title": "Global VAF",
        "higher_is_better": True,
    },
    {
        "key": "reconstruction_error",
        "family": "synergy",
        "kind": "paired",
        "t0": "recon_err_T0",
        "t1": "recon_err_T1",
        "ylabel": "Mean reconstruction error",
        "title": "Reconstruction Error",
        "higher_is_better": False,
    },
    {
        "key": "w_sparsity",
        "family": "synergy",
        "kind": "paired",
        "t0": "w_sparsity_T0",
        "t1": "w_sparsity_T1",
        "ylabel": "W sparsity",
        "title": "W Sparsity",
        "higher_is_better": None,
    },
    {
        "key": "mean_per_muscle_vaf",
        "family": "synergy",
        "kind": "paired",
        "t0": "mean_mean_per_muscle_vaf_T0",
        "t1": "mean_mean_per_muscle_vaf_T1",
        "ylabel": "Mean per-muscle VAF",
        "title": "Per-Muscle VAF",
        "higher_is_better": True,
    },
    {
        "key": "min_per_muscle_vaf",
        "family": "synergy",
        "kind": "paired",
        "t0": "mean_min_per_muscle_vaf_T0",
        "t1": "mean_min_per_muscle_vaf_T1",
        "ylabel": "Minimum per-muscle VAF",
        "title": "Worst-Muscle VAF",
        "higher_is_better": True,
    },
    {
        "key": "mean_cycle_mean_amp",
        "family": "muscle_amplitude",
        "kind": "paired",
        "t0": "mean_cycle_mean_amp_T0",
        "t1": "mean_cycle_mean_amp_T1",
        "ylabel": "Mean muscle amplitude",
        "title": "Cycle Mean Amplitude",
        "higher_is_better": None,
    },
    {
        "key": "mean_cycle_peak_amp",
        "family": "muscle_amplitude",
        "kind": "paired",
        "t0": "mean_cycle_peak_amp_T0",
        "t1": "mean_cycle_peak_amp_T1",
        "ylabel": "Mean muscle peak amplitude",
        "title": "Cycle Peak Amplitude",
        "higher_is_better": None,
    },
    {
        "key": "mean_cycle_auc",
        "family": "muscle_amplitude",
        "kind": "paired",
        "t0": "mean_cycle_auc_T0",
        "t1": "mean_cycle_auc_T1",
        "ylabel": "Mean muscle AUC",
        "title": "Cycle AUC",
        "higher_is_better": None,
    },
    {
        "key": "mean_cv_mean_amp",
        "family": "stability",
        "kind": "paired",
        "t0": "mean_cv_mean_amp_T0",
        "t1": "mean_cv_mean_amp_T1",
        "ylabel": "CV of mean amplitude",
        "title": "Stability: CV Mean Amplitude",
        "higher_is_better": False,
    },
    {
        "key": "mean_cv_peak_amp",
        "family": "stability",
        "kind": "paired",
        "t0": "mean_cv_peak_amp_T0",
        "t1": "mean_cv_peak_amp_T1",
        "ylabel": "CV of peak amplitude",
        "title": "Stability: CV Peak Amplitude",
        "higher_is_better": False,
    },
    {
        "key": "mean_cv_auc",
        "family": "stability",
        "kind": "paired",
        "t0": "mean_cv_auc_T0",
        "t1": "mean_cv_auc_T1",
        "ylabel": "CV of AUC",
        "title": "Stability: CV AUC",
        "higher_is_better": False,
    },
    {
        "key": "mean_similarity_to_mean_cycle",
        "family": "stability",
        "kind": "paired",
        "t0": "mean_similarity_to_mean_cycle_T0",
        "t1": "mean_similarity_to_mean_cycle_T1",
        "ylabel": "Similarity to mean cycle",
        "title": "Stability: Similarity to Mean Cycle",
        "higher_is_better": True,
    },
    {
        "key": "w_similarity",
        "family": "synergy",
        "kind": "single",
        "col": "w_similarity",
        "ylabel": "W similarity (T0↔T1)",
        "title": "W Structure Similarity",
        "higher_is_better": True,
    },
    {
        "key": "h_profile_similarity",
        "family": "synergy",
        "kind": "single",
        "col": "h_profile_similarity",
        "ylabel": "H profile similarity",
        "title": "H Profile Similarity",
        "higher_is_better": True,
    },
]


@dataclass
class TaskResult:
    patient_id: str
    task_name: str
    session: str
    condition: str
    task_dir: Path
    task_summary: pd.DataFrame
    window_metrics: pd.DataFrame
    cycle_muscle_metrics: pd.DataFrame
    cycle_cci_metrics: pd.DataFrame
    cycle_corr_metrics: pd.DataFrame
    cycle_similarity_metrics: pd.DataFrame
    stability_summary_metrics: pd.DataFrame
    W_global: Optional[pd.DataFrame]
    H_windows: Optional[np.ndarray]
    phase: Optional[np.ndarray]
    X_cycles: Optional[np.ndarray]


# -----------------------------------------------------------------------------
# Data loading / discovery
# -----------------------------------------------------------------------------


def slugify(text: str) -> str:
    return re.sub(r"[^0-9a-zA-Z]+", "_", str(text).strip()).strip("_")


def parse_session_condition(task_name: str) -> Tuple[str, str]:
    task_upper = task_name.upper()
    session = "T0" if "T0" in task_upper else "T1" if "T1" in task_upper else ""
    condition = "SN" if "SN" in task_upper else "DS" if "DS" in task_upper else ""
    return session, condition


def read_csv_if_exists(path: Path) -> pd.DataFrame:
    if path.exists():
        try:
            return pd.read_csv(path)
        except Exception:
            return pd.DataFrame()
    return pd.DataFrame()


def read_numpy_if_exists(path: Path) -> Optional[np.ndarray]:
    if path.exists():
        try:
            return np.load(path, allow_pickle=True)
        except Exception:
            return None
    return None


def read_npz_array(path: Path, key: str) -> Optional[np.ndarray]:
    if not path.exists():
        return None
    try:
        npz = np.load(path, allow_pickle=True)
        return npz[key] if key in npz.files else None
    except Exception:
        return None


def load_root_tables(results_dir: Path) -> Dict[str, pd.DataFrame]:
    """Load root-level summary tables when present."""
    return {
        "paired": read_csv_if_exists(results_dir / "patient_paired_metrics.csv"),
        "task": read_csv_if_exists(results_dir / "patient_task_metrics.csv"),
        "cohort": read_csv_if_exists(results_dir / "cohort_summary.csv"),
    }


def discover_task_dirs(results_dir: Path) -> List[Tuple[str, str, str, str, Path]]:
    """Discover patient/task directories that encode session and condition."""
    discovered: List[Tuple[str, str, str, str, Path]] = []
    if not results_dir.exists():
        return discovered
    for patient_dir in sorted(results_dir.iterdir()):
        if not patient_dir.is_dir():
            continue
        if patient_dir.name.startswith("plots_"):
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


def load_task_level_tables(results_dir: Path) -> List[TaskResult]:
    """Load per-task tables and arrays from patient/task directories."""
    records: List[TaskResult] = []
    for patient_id, task_name, session, condition, task_dir in discover_task_dirs(results_dir):
        w_global = None
        w_path = task_dir / "W_global.csv"
        if w_path.exists():
            try:
                w_global = pd.read_csv(w_path, index_col=0)
            except Exception:
                w_global = None

        task_summary = read_csv_if_exists(task_dir / "task_summary_metrics.csv")
        if not task_summary.empty:
            if "patient_id" not in task_summary.columns:
                task_summary.insert(0, "patient_id", patient_id)
            if "task_name" not in task_summary.columns:
                task_summary.insert(1, "task_name", task_name)
            task_summary["session"] = session
            task_summary["condition"] = condition

        records.append(
            TaskResult(
                patient_id=patient_id,
                task_name=task_name,
                session=session,
                condition=condition,
                task_dir=task_dir,
                task_summary=task_summary,
                window_metrics=read_csv_if_exists(task_dir / "window_metrics.csv"),
                cycle_muscle_metrics=read_csv_if_exists(task_dir / "cycle_muscle_metrics.csv"),
                cycle_cci_metrics=read_csv_if_exists(task_dir / "cycle_cci_metrics.csv"),
                cycle_corr_metrics=read_csv_if_exists(task_dir / "cycle_pairwise_correlations.csv"),
                cycle_similarity_metrics=read_csv_if_exists(task_dir / "cycle_similarity_to_mean.csv"),
                stability_summary_metrics=read_csv_if_exists(task_dir / "stability_summary_metrics.csv"),
                W_global=w_global,
                H_windows=read_npz_array(task_dir / "H_windows.npz", "H"),
                phase=read_numpy_if_exists(task_dir / "phase.npy"),
                X_cycles=read_npz_array(task_dir / "X_cycles.npz", "X_cycles"),
            )
        )
    return records


def combine_task_summary_tables(root_task_df: pd.DataFrame, task_records: List[TaskResult]) -> pd.DataFrame:
    """Use root patient_task_metrics when available, otherwise combine task-level summaries."""
    if root_task_df is not None and not root_task_df.empty:
        return root_task_df.copy()
    rows = [rec.task_summary for rec in task_records if rec.task_summary is not None and not rec.task_summary.empty]
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


# -----------------------------------------------------------------------------
# Summary-table construction
# -----------------------------------------------------------------------------


def _merge_task_columns_to_paired(
    paired_df: pd.DataFrame,
    task_df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """Merge selected task-level T0/T1 columns into the paired summary."""
    if paired_df.empty or task_df is None or task_df.empty:
        return paired_df
    merged = paired_df.copy()
    for col in columns:
        if col not in task_df.columns:
            continue
        t0 = task_df[task_df["session"] == "T0"][["patient_id", "condition", col]].copy()
        t1 = task_df[task_df["session"] == "T1"][["patient_id", "condition", col]].copy()
        t0 = t0.rename(columns={col: f"{col}_T0"})
        t1 = t1.rename(columns={col: f"{col}_T1"})
        merged = merged.merge(t0, on=["patient_id", "condition"], how="left")
        merged = merged.merge(t1, on=["patient_id", "condition"], how="left")
        if f"{col}_T0" in merged.columns and f"{col}_T1" in merged.columns:
            merged[f"{col}_delta"] = merged[f"{col}_T1"] - merged[f"{col}_T0"]
    return merged


def build_paired_summary_df(
    root_tables: Dict[str, pd.DataFrame],
    task_records: List[TaskResult],
) -> pd.DataFrame:
    """Build paired patient-level summary table from root files and task summaries."""
    paired_df = root_tables.get("paired", pd.DataFrame()).copy()
    task_df = combine_task_summary_tables(root_tables.get("task", pd.DataFrame()), task_records)

    if paired_df.empty and not task_df.empty:
        metrics = [
            "mean_global_vaf",
            "mean_reconstruction_error",
            "mean_cycle_mean_amp",
            "mean_cycle_peak_amp",
            "mean_cycle_auc",
            "mean_cv_mean_amp",
            "mean_cv_peak_amp",
            "mean_cv_auc",
            "mean_similarity_to_mean_cycle",
        ]
        base = task_df[["patient_id", "condition"]].drop_duplicates().copy()
        paired_df = base
        for col in metrics:
            if col not in task_df.columns:
                continue
            t0 = task_df[task_df["session"] == "T0"][["patient_id", "condition", col]].copy()
            t1 = task_df[task_df["session"] == "T1"][["patient_id", "condition", col]].copy()
            t0 = t0.rename(columns={col: f"{col}_T0"})
            t1 = t1.rename(columns={col: f"{col}_T1"})
            paired_df = paired_df.merge(t0, on=["patient_id", "condition"], how="left")
            paired_df = paired_df.merge(t1, on=["patient_id", "condition"], how="left")
            paired_df[f"{col}_delta"] = paired_df[f"{col}_T1"] - paired_df[f"{col}_T0"]
        if "mean_global_vaf_T0" in paired_df.columns:
            paired_df["vaf_T0"] = paired_df["mean_global_vaf_T0"]
            paired_df["vaf_T1"] = paired_df["mean_global_vaf_T1"]
            paired_df["vaf_delta"] = paired_df["vaf_T1"] - paired_df["vaf_T0"]
        if "mean_reconstruction_error_T0" in paired_df.columns:
            paired_df["recon_err_T0"] = paired_df["mean_reconstruction_error_T0"]
            paired_df["recon_err_T1"] = paired_df["mean_reconstruction_error_T1"]
            paired_df["recon_err_ratio"] = paired_df["recon_err_T1"] / paired_df["recon_err_T0"].clip(lower=EPS)

    paired_df = _merge_task_columns_to_paired(
        paired_df,
        task_df,
        columns=["w_sparsity", "mean_mean_per_muscle_vaf", "mean_min_per_muscle_vaf"],
    )
    return paired_df


def build_muscle_summary_df(task_records: List[TaskResult]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-muscle metrics by task, then pair T0/T1 per patient-condition-muscle."""
    rows: List[pd.DataFrame] = []
    for rec in task_records:
        if rec.cycle_muscle_metrics.empty:
            continue
        amp = (
            rec.cycle_muscle_metrics.groupby(["patient_id", "task_name", "muscle"], as_index=False)[
                ["mean_amp", "peak_amp", "auc", "onset", "offset", "duration", "centroid"]
            ]
            .mean()
        )
        amp["session"] = rec.session
        amp["condition"] = rec.condition
        if not rec.stability_summary_metrics.empty:
            stab = rec.stability_summary_metrics.copy()
            keep = [
                "patient_id",
                "task_name",
                "muscle",
                "cv_mean_amp",
                "cv_peak_amp",
                "cv_auc",
                "mean_similarity_to_mean_cycle",
                "std_similarity_to_mean_cycle",
            ]
            stab = stab[[c for c in keep if c in stab.columns]]
            amp = amp.merge(stab, on=["patient_id", "task_name", "muscle"], how="left")
        rows.append(amp)
    long_df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if long_df.empty:
        return long_df, pd.DataFrame()

    pair_cols = [
        "mean_amp",
        "peak_amp",
        "auc",
        "onset",
        "offset",
        "duration",
        "centroid",
        "cv_mean_amp",
        "cv_peak_amp",
        "cv_auc",
        "mean_similarity_to_mean_cycle",
        "std_similarity_to_mean_cycle",
    ]
    index_cols = ["patient_id", "condition", "muscle"]
    wide = long_df[index_cols + ["session"] + [c for c in pair_cols if c in long_df.columns]].copy()
    paired = wide.pivot_table(index=index_cols, columns="session", values=pair_cols, aggfunc="mean")
    paired.columns = [f"{metric}_{session}" for metric, session in paired.columns]
    paired = paired.reset_index()
    for metric in pair_cols:
        t0_col = f"{metric}_T0"
        t1_col = f"{metric}_T1"
        if t0_col in paired.columns and t1_col in paired.columns:
            paired[f"{metric}_delta"] = paired[t1_col] - paired[t0_col]
    return long_df, paired


def build_coordination_summary_df(task_records: List[TaskResult]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build paired CCI summaries and long-form correlation summaries."""
    cci_rows: List[pd.DataFrame] = []
    corr_rows: List[pd.DataFrame] = []
    for rec in task_records:
        if not rec.cycle_cci_metrics.empty:
            cci = (
                rec.cycle_cci_metrics.groupby(["patient_id", "task_name", "muscle_a", "muscle_b"], as_index=False)[["cci"]]
                .mean()
            )
            cci["session"] = rec.session
            cci["condition"] = rec.condition
            cci_rows.append(cci)
        if not rec.cycle_corr_metrics.empty:
            corr = (
                rec.cycle_corr_metrics.groupby(["patient_id", "task_name", "muscle_a", "muscle_b"], as_index=False)[["corr"]]
                .mean()
            )
            corr["session"] = rec.session
            corr["condition"] = rec.condition
            corr_rows.append(corr)

    cci_long = pd.concat(cci_rows, ignore_index=True) if cci_rows else pd.DataFrame()
    corr_long = pd.concat(corr_rows, ignore_index=True) if corr_rows else pd.DataFrame()

    if cci_long.empty:
        return cci_long, pd.DataFrame(), corr_long

    cci_long["pair_label"] = cci_long["muscle_a"] + " vs " + cci_long["muscle_b"]
    paired = cci_long.pivot_table(
        index=["patient_id", "condition", "pair_label", "muscle_a", "muscle_b"],
        columns="session",
        values="cci",
        aggfunc="mean",
    ).reset_index()
    paired.columns.name = None
    if "T0" in paired.columns and "T1" in paired.columns:
        paired = paired.rename(columns={"T0": "cci_T0", "T1": "cci_T1"})
        paired["cci_delta"] = paired["cci_T1"] - paired["cci_T0"]
    return cci_long, paired, corr_long


def build_cycle_profile_summary_df(
    task_records: List[TaskResult],
    selected_muscles: Sequence[str],
    selected_synergies: Sequence[int],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build cohort-ready long tables of cycle/task profiles for muscles and synergies."""
    x_new = np.linspace(0.0, 1.0, PROFILE_NPTS)
    muscle_rows: List[Dict[str, Any]] = []
    synergy_rows: List[Dict[str, Any]] = []

    for rec in task_records:
        if rec.H_windows is not None and rec.H_windows.ndim == 3:
            n_windows = rec.H_windows.shape[0]
            x_old = np.linspace(0.0, 1.0, n_windows) if n_windows > 1 else np.array([0.0])
            for syn_idx in selected_synergies:
                if syn_idx >= rec.H_windows.shape[1]:
                    continue
                y = np.mean(rec.H_windows[:, syn_idx, :], axis=1)
                if n_windows == 1:
                    y_new = np.repeat(y[0], PROFILE_NPTS)
                else:
                    y_new = np.interp(x_new, x_old, y)
                for phase, value in zip(x_new, y_new):
                    synergy_rows.append(
                        {
                            "patient_id": rec.patient_id,
                            "task_name": rec.task_name,
                            "session": rec.session,
                            "condition": rec.condition,
                            "label": f"S{syn_idx}",
                            "profile_phase": phase,
                            "value": float(value),
                        }
                    )

        if rec.X_cycles is not None and rec.phase is not None and rec.X_cycles.ndim == 3 and rec.W_global is not None:
            muscle_names = list(rec.W_global.index)
            muscle_idx = {m: i for i, m in enumerate(muscle_names)}
            for muscle in selected_muscles:
                if muscle not in muscle_idx:
                    continue
                idx = muscle_idx[muscle]
                mean_cycle = np.mean(rec.X_cycles[:, :, idx], axis=0)
                y_new = np.interp(x_new, rec.phase, mean_cycle)
                for phase, value in zip(x_new, y_new):
                    muscle_rows.append(
                        {
                            "patient_id": rec.patient_id,
                            "task_name": rec.task_name,
                            "session": rec.session,
                            "condition": rec.condition,
                            "label": muscle,
                            "profile_phase": phase,
                            "value": float(value),
                        }
                    )
    return pd.DataFrame(muscle_rows), pd.DataFrame(synergy_rows)


def build_h_paired_summary_df(task_summary_df: pd.DataFrame) -> pd.DataFrame:
    """Build paired T0/T1 summary for synergy H metrics (auc, centroid, peak_time) from task_summary_metrics."""
    if task_summary_df.empty:
        return pd.DataFrame()
    pattern = re.compile(r"mean_synergy_(\d+)_(auc|centroid|peak_time)$")
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
                if pd.isna(v0) or pd.isna(v1):
                    continue
                row[f"{metric}_T0"] = float(v0)
                row[f"{metric}_T1"] = float(v1)
                row[f"{metric}_delta"] = float(v1) - float(v0)
            if any(k.endswith("_T0") for k in row):
                rows.append(row)
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Statistics helpers
# -----------------------------------------------------------------------------


def paired_ttest(v0: np.ndarray, v1: np.ndarray) -> float:
    mask = np.isfinite(v0) & np.isfinite(v1)
    v0 = np.asarray(v0)[mask]
    v1 = np.asarray(v1)[mask]
    if len(v0) < 2:
        return np.nan
    _, p = stats.ttest_rel(v0, v1)
    return float(p)


def paired_wilcoxon(v0: np.ndarray, v1: np.ndarray) -> float:
    mask = np.isfinite(v0) & np.isfinite(v1)
    v0 = np.asarray(v0)[mask]
    v1 = np.asarray(v1)[mask]
    if len(v0) < 2:
        return np.nan
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            try:
                stat, p = stats.wilcoxon(v0, v1, method="exact" if len(v0) <= 20 else "asymptotic")
            except TypeError:
                stat, p = stats.wilcoxon(v0, v1)
        return float(p)
    except Exception:
        return np.nan


def paired_effect_size(v0: np.ndarray, v1: np.ndarray) -> float:
    """Paired Cohen's dz."""
    mask = np.isfinite(v0) & np.isfinite(v1)
    diff = np.asarray(v1)[mask] - np.asarray(v0)[mask]
    if diff.size < 2:
        return np.nan
    sd = float(np.std(diff, ddof=1))
    if sd <= EPS:
        return np.nan
    return float(np.mean(diff) / sd)


def apply_fdr(stats_df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    """Benjamini-Hochberg FDR correction within each group."""
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


def compute_paired_stats(
    df: pd.DataFrame,
    metric_specs: Sequence[Dict[str, Any]],
) -> pd.DataFrame:
    """Compute paired t-tests and effect sizes for family-aware scalar metrics."""
    rows: List[Dict[str, Any]] = []
    if df.empty:
        return pd.DataFrame()
    for spec in metric_specs:
        if spec.get("kind") != "paired":
            continue
        t0_col, t1_col = spec["t0"], spec["t1"]
        if t0_col not in df.columns or t1_col not in df.columns:
            continue
        for condition, subset in df.groupby("condition"):
            vals = subset[[t0_col, t1_col]].dropna()
            if vals.empty:
                continue
            v0 = vals[t0_col].to_numpy(dtype=float)
            v1 = vals[t1_col].to_numpy(dtype=float)
            p_ttest = paired_ttest(v0, v1)
            p_wilcox = paired_wilcoxon(v0, v1)
            n_val = len(vals)
            p_primary = p_ttest  # always use paired t-test
            rows.append(
                {
                    "family": spec["family"],
                    "metric_key": spec["key"],
                    "title": spec["title"],
                    "condition": condition,
                    "n": int(n_val),
                    "mean_t0": float(np.mean(v0)),
                    "mean_t1": float(np.mean(v1)),
                    "mean_delta": float(np.mean(v1 - v0)),
                    "std_delta": float(np.std(v1 - v0, ddof=1)) if len(v0) > 1 else np.nan,
                    "p_value": p_primary,
                    "p_ttest": p_ttest,
                    "p_wilcoxon": p_wilcox,
                    "effect_size_dz": paired_effect_size(v0, v1),
                }
            )
    stats_df = pd.DataFrame(rows)
    return apply_fdr(stats_df, group_cols=["family", "condition"])


def compute_dynamic_paired_stats(
    df: pd.DataFrame,
    metric_col_base: str,
    label_col: str,
    family: str,
    title_prefix: str,
) -> pd.DataFrame:
    """Paired stats for long-derived summaries like muscles or CCI pairs."""
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
        p_ttest = paired_ttest(v0, v1)
        p_wilcox = paired_wilcoxon(v0, v1)
        p_primary = p_ttest  # always use paired t-test
        rows.append(
            {
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
                "effect_size_dz": paired_effect_size(v0, v1),
            }
        )
    stats_df = pd.DataFrame(rows)
    return apply_fdr(stats_df, group_cols=["family", "condition"])


def compute_multi_metric_paired_stats(
    df: pd.DataFrame,
    metric_specs: Sequence[Tuple[str, str, str]],
    label_col: str,
) -> pd.DataFrame:
    """Compute paired stats for multiple derived long-format metrics."""
    parts = []
    for metric_col_base, family, title_prefix in metric_specs:
        if f"{metric_col_base}_T0" not in df.columns or f"{metric_col_base}_T1" not in df.columns:
            continue
        part = compute_dynamic_paired_stats(
            df,
            metric_col_base=metric_col_base,
            label_col=label_col,
            family=family,
            title_prefix=title_prefix,
        )
        if not part.empty:
            parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


# -----------------------------------------------------------------------------
# Generic plotting helpers
# -----------------------------------------------------------------------------


def format_pvalue(p: float) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def format_qvalue(q: float) -> str:
    if not np.isfinite(q):
        return "q=n/a"
    return "q<0.001" if q < 0.001 else f"q={q:.3f}"


def extract_stats_row(stats_df: pd.DataFrame, family: str, metric_key: str, condition: str, label: Optional[str] = None) -> Optional[pd.Series]:
    if stats_df.empty:
        return None
    subset = stats_df[(stats_df["family"] == family) & (stats_df["metric_key"] == metric_key) & (stats_df["condition"] == condition)]
    if label is not None and "label" in subset.columns:
        subset = subset[subset["label"] == label]
    if subset.empty:
        return None
    return subset.iloc[0]


def _fmt_delta(d: float) -> str:
    if not np.isfinite(d):
        return "Δ=n/a"
    sign = "+" if d >= 0 else ""
    return f"Δ={sign}{d:.3f}"


def annotate_stats(ax: plt.Axes, stats_row: Optional[pd.Series]) -> None:
    if stats_row is None:
        return
    n = int(stats_row["n"])
    delta_str = _fmt_delta(stats_row.get("mean_delta", np.nan))
    dz = stats_row.get("effect_size_dz", np.nan)
    dz_str = f"dz={dz:.2f}" if np.isfinite(dz) else "dz=n/a"
    parts = [f"n={n}", delta_str, dz_str, format_pvalue(stats_row["p_value"]), format_qvalue(stats_row.get("q_value", np.nan))]
    txt = " | ".join(parts)
    ax.text(0.98, 0.98, txt, transform=ax.transAxes, ha="right", va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7, edgecolor="none"))


def plot_paired_slope(
    ax: plt.Axes,
    v0: np.ndarray,
    v1: np.ndarray,
    ylabel: str,
    title: str,
    stats_row: Optional[pd.Series] = None,
) -> None:
    """Paired slope plot with mean/SEM overlay."""
    mask = np.isfinite(v0) & np.isfinite(v1)
    v0 = np.asarray(v0)[mask]
    v1 = np.asarray(v1)[mask]
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
    sems = [stats.sem(v0) if n > 1 else 0.0, stats.sem(v1) if n > 1 else 0.0]
    ax.errorbar([0, 1], means, yerr=sems, color="black", linewidth=2, marker="D", markersize=5, capsize=4, zorder=4)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["T0", "T1"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    annotate_stats(ax, stats_row)


def plot_delta_strip(
    ax: plt.Axes,
    delta: np.ndarray,
    ylabel: str,
    title: str,
    stats_row: Optional[pd.Series] = None,
) -> None:
    """Delta strip plot for paired differences (T1-T0)."""
    delta = np.asarray(delta, dtype=float)
    delta = delta[np.isfinite(delta)]
    if delta.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return
    x = np.linspace(-0.08, 0.08, len(delta))
    ax.axhline(0.0, color="0.5", linestyle="--", linewidth=1)
    ax.scatter(x, delta, color="#756bb1", s=28, zorder=3)
    mean_delta = np.mean(delta)
    sem_delta = stats.sem(delta) if len(delta) > 1 else 0.0
    ax.errorbar([0], [mean_delta], yerr=[sem_delta], color="black", marker="D", markersize=5, capsize=4, zorder=4)
    ax.set_xlim(-0.2, 0.2)
    ax.set_xticks([0])
    ax.set_xticklabels(["T1 - T0"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    annotate_stats(ax, stats_row)


def plot_single_distribution(
    ax: plt.Axes,
    values: np.ndarray,
    ylabel: str,
    title: str,
) -> None:
    """Single-condition strip + mean/SEM overlay."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return
    x = np.linspace(-0.08, 0.08, len(values))
    ax.scatter(x, values, color="#3182bd", s=30)
    ax.errorbar([0], [np.mean(values)], yerr=[stats.sem(values) if len(values) > 1 else 0.0],
                color="black", marker="D", markersize=5, capsize=4)
    ax.set_xlim(-0.2, 0.2)
    ax.set_xticks([0])
    ax.set_xticklabels([f"n={len(values)}"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_matrix_heatmap(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: Sequence[str],
    col_labels: Sequence[str],
    title: str,
    cmap: str = "viridis",
    center_zero: bool = False,
    show_row_labels: bool = True,
    show_col_labels: bool = True,
) -> None:
    """Generic heatmap with labels."""
    if matrix.size == 0:
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(title)
        return
    vlim = np.nanmax(np.abs(matrix)) if center_zero else None
    im = ax.imshow(
        matrix,
        aspect="auto",
        cmap=cmap,
        vmin=-vlim if center_zero else None,
        vmax=vlim if center_zero else None,
    )
    ax.set_title(title)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels if show_row_labels else [""] * len(row_labels), fontsize=8)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels if show_col_labels else [""] * len(col_labels), rotation=45, ha="right", fontsize=8)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)


def _plot_single_row_heatmap(
    ax: plt.Axes,
    row_data: np.ndarray,
    col_labels: Sequence[str],
    row_label: str,
    cmap: str,
    center_zero: bool,
    show_col_labels: bool = True,
) -> None:
    """Plot one row as a heatmap with its own color scale and compact horizontal colorbar."""
    if row_data.size == 0 or not np.any(np.isfinite(row_data)):
        ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
        ax.set_ylabel(row_label, fontsize=8)
        return
    arr = np.atleast_2d(np.asarray(row_data, dtype=float))
    vmin = np.nanmin(arr)
    vmax = np.nanmax(arr)
    if center_zero:
        vlim = max(np.abs(vmin), np.abs(vmax), 1e-12)
        vmin, vmax = -vlim, vlim
    if vmax <= vmin:
        vmax = vmin + 1e-12
    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_ylabel(row_label, fontsize=8)
    ax.set_yticks([0])
    ax.set_yticklabels([""])
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels if show_col_labels else [""] * len(col_labels), rotation=45, ha="right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    cbar = plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.04, pad=0.08, shrink=0.7, aspect=20)
    cbar.ax.tick_params(labelsize=6)
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([f"{vmin:.2g}", f"{vmax:.2g}"])


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    def _norm_cols(M: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(M, axis=0, keepdims=True)
        n[n < EPS] = 1.0
        return M / n
    return _norm_cols(A).T @ _norm_cols(B)


def align_columns_to_reference(new_matrix: np.ndarray, ref_matrix: np.ndarray) -> np.ndarray:
    """Align W columns with Hungarian assignment on cosine similarity."""
    if new_matrix.shape != ref_matrix.shape:
        return new_matrix
    sim = cosine_similarity_matrix(new_matrix, ref_matrix)
    row_idx, col_idx = linear_sum_assignment(-sim)
    order = np.zeros(new_matrix.shape[1], dtype=int)
    order[col_idx] = row_idx
    return new_matrix[:, order]


# -----------------------------------------------------------------------------
# Plot-family generators
# -----------------------------------------------------------------------------


def plot_individual_paired_metric_figures(
    paired_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    metric_specs: Sequence[Dict[str, Any]],
    out_dir: Path,
    conditions: Sequence[str],
    dpi: int,
) -> None:
    """Create individual paired slope and delta figures per metric per condition."""
    if paired_df.empty:
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in metric_specs:
        if spec.get("kind") != "paired":
            continue
        t0_col, t1_col = spec["t0"], spec["t1"]
        if t0_col not in paired_df.columns or t1_col not in paired_df.columns:
            continue
        key = spec["key"]
        for condition in conditions:
            subset = paired_df[paired_df["condition"] == condition]
            if subset.empty:
                continue
            stats_row = extract_stats_row(stats_df, spec["family"], key, condition)
            fig, ax = plt.subplots(figsize=(5, 4))
            plot_paired_slope(
                ax,
                subset[t0_col].to_numpy(dtype=float),
                subset[t1_col].to_numpy(dtype=float),
                spec["ylabel"],
                f"{spec['title']} ({condition})",
                stats_row,
            )
            fig.tight_layout()
            fig.savefig(out_dir / f"paired_{key}_{condition}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)
            delta = subset[t1_col].to_numpy(dtype=float) - subset[t0_col].to_numpy(dtype=float)
            fig2, ax2 = plt.subplots(figsize=(4, 4))
            plot_delta_strip(ax2, delta, f"Delta {spec['ylabel']}", f"{spec['title']} ({condition})", stats_row)
            fig2.tight_layout()
            fig2.savefig(out_dir / f"delta_{key}_{condition}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig2)


def plot_paired_family_figures(
    paired_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    metric_specs: Sequence[Dict[str, Any]],
    out_dir: Path,
    conditions: Sequence[str],
    dpi: int,
) -> None:
    """Create family-level paired slope and delta figures."""
    if paired_df.empty:
        return
    families = sorted({spec["family"] for spec in metric_specs if spec.get("kind") == "paired"})
    for family in families:
        family_specs = [
            spec for spec in metric_specs
            if spec.get("kind") == "paired"
            and spec["family"] == family
            and spec["t0"] in paired_df.columns
            and spec["t1"] in paired_df.columns
        ]
        if not family_specs:
            continue
        fig_s, axes_s = plt.subplots(len(family_specs), len(conditions), figsize=(5 * len(conditions), 4.2 * len(family_specs)), squeeze=False)
        fig_d, axes_d = plt.subplots(len(family_specs), len(conditions), figsize=(4.6 * len(conditions), 4.0 * len(family_specs)), squeeze=False)
        for i, spec in enumerate(family_specs):
            for j, condition in enumerate(conditions):
                subset = paired_df[paired_df["condition"] == condition]
                if subset.empty:
                    axes_s[i, j].text(0.5, 0.5, "No data", transform=axes_s[i, j].transAxes, ha="center", va="center")
                    axes_d[i, j].text(0.5, 0.5, "No data", transform=axes_d[i, j].transAxes, ha="center", va="center")
                    continue
                stats_row = extract_stats_row(stats_df, family, spec["key"], condition)
                plot_paired_slope(
                    axes_s[i, j],
                    subset[spec["t0"]].to_numpy(dtype=float),
                    subset[spec["t1"]].to_numpy(dtype=float),
                    spec["ylabel"],
                    f"{spec['title']} ({condition})",
                    stats_row,
                )
                delta = subset[spec["t1"]].to_numpy(dtype=float) - subset[spec["t0"]].to_numpy(dtype=float)
                plot_delta_strip(
                    axes_d[i, j],
                    delta,
                    f"Delta {spec['ylabel']}",
                    f"{spec['title']} delta ({condition})",
                    stats_row,
                )
        fig_s.tight_layout()
        fig_d.tight_layout()
        out_dir.mkdir(parents=True, exist_ok=True)
        fam_name = slugify(family)
        fig_s.savefig(out_dir / f"paired_{fam_name}_metrics.png", dpi=dpi, bbox_inches="tight")
        fig_d.savefig(out_dir / f"delta_{fam_name}_metrics.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig_s)
        plt.close(fig_d)


def plot_single_metric_figures(
    paired_df: pd.DataFrame,
    metric_specs: Sequence[Dict[str, Any]],
    out_dir: Path,
    conditions: Sequence[str],
    dpi: int,
) -> None:
    """Plot single-value pair-level metrics such as W similarity."""
    if paired_df.empty:
        return
    single_specs = [spec for spec in metric_specs if spec.get("kind") == "single" and spec.get("col") in paired_df.columns]
    if not single_specs:
        return
    for family in sorted({spec["family"] for spec in single_specs}):
        family_specs = [spec for spec in single_specs if spec["family"] == family]
        fig, axes = plt.subplots(len(family_specs), len(conditions), figsize=(4.5 * len(conditions), 4.0 * len(family_specs)), squeeze=False)
        for i, spec in enumerate(family_specs):
            for j, condition in enumerate(conditions):
                subset = paired_df[(paired_df["condition"] == condition) & paired_df[spec["col"]].notna()]
                plot_single_distribution(
                    axes[i, j],
                    subset[spec["col"]].to_numpy(dtype=float),
                    spec["ylabel"],
                    f"{spec['title']} ({condition})",
                )
        fig.tight_layout()
        family_dir = out_dir / family
        family_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(family_dir / f"{family}_single_metrics.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_muscle_heatmaps(
    muscle_paired_df: pd.DataFrame,
    out_dir: Path,
    conditions: Sequence[str],
    dpi: int,
) -> None:
    """Heatmaps of T0, T1, and delta across muscles. One subpanel per metric row with its own colorbar."""
    if muscle_paired_df.empty:
        return
    families = {
        "amplitude": ["mean_amp", "peak_amp", "auc"],
        "timing": ["onset", "offset", "duration", "centroid"],
        "stability": ["cv_mean_amp", "cv_peak_amp", "cv_auc", "mean_similarity_to_mean_cycle"],
    }
    for condition in conditions:
        cond_df = muscle_paired_df[muscle_paired_df["condition"] == condition]
        if cond_df.empty:
            continue
        muscles = sorted(cond_df["muscle"].dropna().unique().tolist())
        for family_name, metrics in families.items():
            available_metrics = [m for m in metrics if f"{m}_T0" in cond_df.columns and f"{m}_T1" in cond_df.columns]
            if not available_metrics:
                continue
            n_rows = len(available_metrics)
            n_cols = 3  # T0, T1, delta
            fig, axes = plt.subplots(
                n_rows,
                n_cols,
                figsize=(3.5 * n_cols + 0.4 * len(muscles), 1.2 * n_rows),
                squeeze=False,
                gridspec_kw={"hspace": 0.35, "wspace": 0.25},
                layout="constrained",
            )
            for suffix, col_idx, cmap, center_zero in [
                ("T0", 0, "viridis", False),
                ("T1", 1, "viridis", False),
                ("delta", 2, "coolwarm", True),
            ]:
                for i, metric in enumerate(available_metrics):
                    col = f"{metric}_{suffix}"
                    row_data = np.full(len(muscles), np.nan, dtype=float)
                    if col in cond_df.columns:
                        for j, muscle in enumerate(muscles):
                            vals = cond_df.loc[cond_df["muscle"] == muscle, col].to_numpy(dtype=float)
                            row_data[j] = np.nanmean(vals) if vals.size else np.nan
                    _plot_single_row_heatmap(
                        axes[i, col_idx],
                        row_data,
                        muscles,
                        metric,
                        cmap=cmap,
                        center_zero=center_zero,
                        show_col_labels=(i == 0),
                    )
                    if i == 0:
                        axes[i, col_idx].set_title(f"{family_name.title()} {suffix} ({condition})", fontsize=9)
            hm_dir = out_dir / "muscle_heatmaps"
            hm_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(hm_dir / f"heatmap_muscle_deltas_{slugify(family_name)}_{condition}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)


def plot_small_multiples(
    muscle_paired_df: pd.DataFrame,
    selected_muscles: Sequence[str],
    out_dir: Path,
    conditions: Sequence[str],
    dpi: int,
) -> None:
    """Paired slope small multiples for selected muscles."""
    if muscle_paired_df.empty:
        return
    metrics = [
        ("auc", "AUC"),
        ("duration", "Duration"),
        ("centroid", "Centroid"),
        ("mean_similarity_to_mean_cycle", "Similarity to mean cycle"),
    ]
    for condition in conditions:
        cond_df = muscle_paired_df[muscle_paired_df["condition"] == condition]
        present_muscles = [m for m in selected_muscles if m in cond_df["muscle"].unique()]
        if not present_muscles:
            continue
        fig, axes = plt.subplots(len(present_muscles), len(metrics), figsize=(4.6 * len(metrics), 3.6 * len(present_muscles)), squeeze=False)
        for i, muscle in enumerate(present_muscles):
            mdf = cond_df[cond_df["muscle"] == muscle]
            for j, (metric, ylabel) in enumerate(metrics):
                t0_col, t1_col = f"{metric}_T0", f"{metric}_T1"
                if t0_col not in mdf.columns or t1_col not in mdf.columns:
                    axes[i, j].text(0.5, 0.5, "No data", transform=axes[i, j].transAxes, ha="center", va="center")
                    continue
                plot_paired_slope(
                    axes[i, j],
                    mdf[t0_col].to_numpy(dtype=float),
                    mdf[t1_col].to_numpy(dtype=float),
                    ylabel,
                    muscle if j == 0 else ylabel,
                )
        fig.tight_layout()
        small_dir = out_dir / "muscle_small_multiples"
        small_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(small_dir / f"small_multiples_muscle_metrics_{condition}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_cycle_profiles(
    muscle_profile_df: pd.DataFrame,
    synergy_profile_df: pd.DataFrame,
    out_dir: Path,
    conditions: Sequence[str],
    dpi: int,
) -> None:
    """Mean ± SD cycle/task profile plots for muscles and synergies."""
    def _plot_profile_family(df: pd.DataFrame, family_name: str) -> None:
        if df.empty:
            return
        labels = sorted(df["label"].unique().tolist())
        for condition in conditions:
            cond_df = df[df["condition"] == condition]
            if cond_df.empty:
                continue
            fig, axes = plt.subplots(len(labels), 1, figsize=(8, 2.8 * len(labels)), squeeze=False, sharex=True)
            for i, label in enumerate(labels):
                ax = axes[i, 0]
                lab_df = cond_df[cond_df["label"] == label]
                for session, color in [("T0", "#3182bd"), ("T1", "#e6550d")]:
                    sdf = lab_df[lab_df["session"] == session]
                    if sdf.empty:
                        continue
                    summary = sdf.groupby("profile_phase")["value"].agg(["mean", "std"]).reset_index()
                    ax.plot(summary["profile_phase"] * 100.0, summary["mean"], color=color, label=session, linewidth=2)
                    ax.fill_between(summary["profile_phase"] * 100.0,
                                    summary["mean"] - summary["std"].fillna(0.0),
                                    summary["mean"] + summary["std"].fillna(0.0),
                                    color=color, alpha=0.2)
                ax.set_ylabel(label)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                if i == 0:
                    ax.legend(loc="upper right", fontsize=8)
            axes[-1, 0].set_xlabel("Normalized phase (%)")
            fig.suptitle(f"{family_name.title()} profiles ({condition})", fontsize=12)
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            profile_dir = out_dir / "cycle_profiles"
            profile_dir.mkdir(parents=True, exist_ok=True)
            fname = f"synergy_profiles_{condition}.png" if family_name == "synergy_profiles" else f"cycle_profiles_selected_muscles_{condition}.png"
            fig.savefig(profile_dir / fname, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

    _plot_profile_family(synergy_profile_df, "synergy_profiles")
    _plot_profile_family(muscle_profile_df, "muscle_profiles")


def plot_w_heatmaps(
    task_records: List[TaskResult],
    out_dir: Path,
    conditions: Sequence[str],
    dpi: int,
) -> None:
    """Plot cohort mean W heatmaps for T0, T1, and delta."""
    for condition in conditions:
        by_patient: Dict[str, Dict[str, TaskResult]] = {}
        for rec in task_records:
            if rec.condition != condition or rec.W_global is None:
                continue
            by_patient.setdefault(rec.patient_id, {})[rec.session] = rec
        pairs = [(d["T0"], d["T1"]) for d in by_patient.values() if "T0" in d and "T1" in d]
        if not pairs:
            continue
        ref_df = pairs[0][0].W_global
        if ref_df is None:
            continue
        ref = ref_df.to_numpy(dtype=float)
        muscles = ref_df.index.tolist()
        cols = ref_df.columns.tolist()
        t0_stack, t1_stack, delta_stack = [], [], []
        for rec0, rec1 in pairs:
            if rec0.W_global is None or rec1.W_global is None:
                continue
            if list(rec0.W_global.index) != muscles or list(rec1.W_global.index) != muscles:
                continue
            W0 = align_columns_to_reference(rec0.W_global.to_numpy(dtype=float), ref)
            W1 = align_columns_to_reference(rec1.W_global.to_numpy(dtype=float), ref)
            t0_stack.append(W0)
            t1_stack.append(W1)
            delta_stack.append(W1 - W0)
        if not t0_stack:
            continue
        mean_t0 = np.mean(np.stack(t0_stack, axis=0), axis=0)
        mean_t1 = np.mean(np.stack(t1_stack, axis=0), axis=0)
        mean_delta = np.mean(np.stack(delta_stack, axis=0), axis=0)
        fig, axes = plt.subplots(1, 3, figsize=(14, 8))
        plot_matrix_heatmap(axes[0], mean_t0, muscles, cols, f"W mean T0 ({condition})", cmap="viridis", show_row_labels=True, show_col_labels=True)
        plot_matrix_heatmap(axes[1], mean_t1, muscles, cols, f"W mean T1 ({condition})", cmap="viridis", show_row_labels=False, show_col_labels=False)
        plot_matrix_heatmap(axes[2], mean_delta, muscles, cols, f"W delta ({condition})", cmap="coolwarm", center_zero=True, show_row_labels=False, show_col_labels=False)
        fig.tight_layout()
        w_dir = out_dir / "w_heatmaps"
        w_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(w_dir / f"W_heatmap_comparison_{condition}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_corr_heatmap(
    corr_long_df: pd.DataFrame,
    out_dir: Path,
    conditions: Sequence[str],
    dpi: int,
) -> None:
    """Plot pairwise muscle correlation heatmaps for T0, T1, and delta.

    T0 and T1 share one colormap (-1 to 1). Delta has its own colormap.
    Y-tick labels (muscle names) only on the first left matrix.
    All matrices are square (aspect='equal').
    """
    if corr_long_df.empty:
        return
    muscles = sorted(pd.unique(pd.concat([corr_long_df["muscle_a"], corr_long_df["muscle_b"]], ignore_index=True)).tolist())

    def _matrix_from_long(df: pd.DataFrame, value_col: str) -> np.ndarray:
        mat = np.full((len(muscles), len(muscles)), np.nan, dtype=float)
        np.fill_diagonal(mat, 1.0)
        idx = {m: i for i, m in enumerate(muscles)}
        for _, row in df.iterrows():
            i = idx[row["muscle_a"]]
            j = idx[row["muscle_b"]]
            mat[i, j] = row[value_col]
            mat[j, i] = row[value_col]
        return mat

    def _plot_corr_panel(
        ax: plt.Axes,
        matrix: np.ndarray,
        vmin: float,
        vmax: float,
        title: str,
        show_row_labels: bool,
        show_col_labels: bool,
    ):
        if matrix.size == 0:
            ax.text(0.5, 0.5, "No data", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(title)
            return None
        im = ax.imshow(matrix, aspect="equal", cmap="coolwarm", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_yticks(range(len(muscles)))
        ax.set_yticklabels(muscles if show_row_labels else [""] * len(muscles), fontsize=8)
        ax.set_xticks(range(len(muscles)))
        ax.set_xticklabels(muscles if show_col_labels else [""] * len(muscles), rotation=45, ha="right", fontsize=8)
        return im

    for condition in conditions:
        cond_df = corr_long_df[corr_long_df["condition"] == condition]
        if cond_df.empty:
            continue
        session_mats = {}
        for session in ["T0", "T1"]:
            sdf = cond_df[cond_df["session"] == session]
            agg = sdf.groupby(["muscle_a", "muscle_b"], as_index=False)[["corr"]].mean()
            session_mats[session] = _matrix_from_long(agg, "corr")
        if "T0" not in session_mats or "T1" not in session_mats:
            continue

        mat_t0 = session_mats["T0"]
        mat_t1 = session_mats["T1"]
        mat_delta = mat_t1 - mat_t0

        # Shared scale for T0 and T1 (correlations -1 to 1)
        vmin_shared, vmax_shared = -1.0, 1.0
        # Delta: symmetric around 0
        dlim = float(np.nanmax(np.abs(mat_delta)))
        dlim = max(dlim, 1e-12)

        n = len(muscles)
        panel_size = max(5, n * 0.4)
        fig, axes = plt.subplots(1, 3, figsize=(3 * panel_size, panel_size), squeeze=True)

        im0 = _plot_corr_panel(
            axes[0], mat_t0, vmin_shared, vmax_shared,
            f"Correlation T0 ({condition})", show_row_labels=True, show_col_labels=True,
        )
        im1 = _plot_corr_panel(
            axes[1], mat_t1, vmin_shared, vmax_shared,
            f"Correlation T1 ({condition})", show_row_labels=False, show_col_labels=False,
        )
        im2 = _plot_corr_panel(
            axes[2], mat_delta, -dlim, dlim,
            f"Correlation delta ({condition})", show_row_labels=False, show_col_labels=False,
        )

        # Single colorbar shared by T0 and T1 (first two panels)
        if im0 is not None and im1 is not None:
            cbar_shared = fig.colorbar(im0, ax=[axes[0], axes[1]], fraction=0.046, pad=0.04, shrink=0.8)
            cbar_shared.set_label("Correlation", fontsize=9)
        # Delta has its own colorbar
        if im2 is not None:
            cbar_delta = fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)
            cbar_delta.set_label("Delta", fontsize=9)

        fig.tight_layout()
        corr_dir = out_dir / "coordination"
        corr_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(corr_dir / f"corr_matrix_comparison_{condition}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_h_timing_amplitude_summary(
    h_paired_df: pd.DataFrame,
    h_stats_df: pd.DataFrame,
    out_dir: Path,
    conditions: Sequence[str],
    dpi: int,
) -> None:
    """Paired slope plots for synergy H metrics (auc, centroid, peak_time) with effect sizes and q-values."""
    if h_paired_df.empty:
        return
    metrics = [
        ("auc", "Synergy AUC", "AUC"),
        ("centroid", "Centroid (phase)", "Centroid"),
        ("peak_time", "Peak time (phase)", "Peak time"),
    ]
    available = [(m, yl, t) for m, yl, t in metrics if f"{m}_T0" in h_paired_df.columns and f"{m}_T1" in h_paired_df.columns]
    if not available:
        return
    synergies = sorted(h_paired_df["synergy_label"].unique().tolist())
    for condition in conditions:
        cond_df = h_paired_df[h_paired_df["condition"] == condition]
        if cond_df.empty:
            continue
        fig, axes = plt.subplots(len(synergies), len(available), figsize=(4.5 * len(available), 4 * len(synergies)), squeeze=False)
        for i, syn in enumerate(synergies):
            syn_df = cond_df[cond_df["synergy_label"] == syn]
            for j, (metric, ylabel, title) in enumerate(available):
                t0_col, t1_col = f"{metric}_T0", f"{metric}_T1"
                stats_row = extract_stats_row(h_stats_df, "synergy_h", metric, condition, label=syn)
                plot_paired_slope(
                    axes[i, j],
                    syn_df[t0_col].to_numpy(dtype=float),
                    syn_df[t1_col].to_numpy(dtype=float),
                    ylabel,
                    f"{syn}: {title}" if i == 0 else title,
                    stats_row,
                )
        fig.suptitle(f"Synergy H timing/amplitude ({condition})", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        h_dir = out_dir / "h_summaries"
        h_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(h_dir / f"paired_synergy_component_metrics_{condition}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


def plot_cci_family(
    cci_paired_df: pd.DataFrame,
    cci_stats_df: pd.DataFrame,
    out_dir: Path,
    conditions: Sequence[str],
    dpi: int,
) -> None:
    """Paired slope and delta plots for CCI summaries (one figure per condition)."""
    if cci_paired_df.empty:
        return
    pairs = sorted(cci_paired_df["pair_label"].unique().tolist())
    coord_dir = out_dir / "coordination"
    coord_dir.mkdir(parents=True, exist_ok=True)
    for condition in conditions:
        fig_s, axes_s_sub = plt.subplots(len(pairs), 1, figsize=(5, 4 * len(pairs)), squeeze=False)
        fig_d, axes_d_sub = plt.subplots(len(pairs), 1, figsize=(4.5, 4 * len(pairs)), squeeze=False)
        for i, pair in enumerate(pairs):
            subset = cci_paired_df[(cci_paired_df["condition"] == condition) & (cci_paired_df["pair_label"] == pair)]
            stats_row = extract_stats_row(cci_stats_df, "coordination", "cci", condition, label=pair)
            plot_paired_slope(
                axes_s_sub[i, 0],
                subset.get("cci_T0", pd.Series(dtype=float)).to_numpy(dtype=float),
                subset.get("cci_T1", pd.Series(dtype=float)).to_numpy(dtype=float),
                "CCI",
                f"{pair} ({condition})",
                stats_row,
            )
            plot_delta_strip(
                axes_d_sub[i, 0],
                subset.get("cci_delta", pd.Series(dtype=float)).to_numpy(dtype=float),
                "CCI delta",
                f"{pair} delta ({condition})",
                stats_row,
            )
        fig_s.tight_layout()
        fig_d.tight_layout()
        fig_s.savefig(coord_dir / f"paired_cci_{condition}.png", dpi=dpi, bbox_inches="tight")
        fig_d.savefig(coord_dir / f"delta_cci_{condition}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig_s)
        plt.close(fig_d)


# -----------------------------------------------------------------------------
# Summary-table saving
# -----------------------------------------------------------------------------


def save_summary_tables(
    out_dir: Path,
    paired_stats: pd.DataFrame,
    muscle_stats: pd.DataFrame,
    cci_stats: pd.DataFrame,
    paired_df: pd.DataFrame,
    muscle_paired_df: pd.DataFrame,
    cci_paired_df: pd.DataFrame,
) -> None:
    """Write derived summary/statistics tables separately from plotting."""
    summary_dir = out_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)
    if not paired_df.empty:
        paired_df.to_csv(summary_dir / "paired_summary_expanded.csv", index=False)
    if not muscle_paired_df.empty:
        muscle_paired_df.to_csv(summary_dir / "muscle_summary_paired.csv", index=False)
    if not cci_paired_df.empty:
        cci_paired_df.to_csv(summary_dir / "coordination_summary_paired.csv", index=False)
    if not paired_stats.empty:
        paired_stats.to_csv(summary_dir / "paired_metric_stats.csv", index=False)
    if not muscle_stats.empty:
        muscle_stats.to_csv(summary_dir / "muscle_metric_stats.csv", index=False)
    if not cci_stats.empty:
        cci_stats.to_csv(summary_dir / "coordination_metric_stats.csv", index=False)


# -----------------------------------------------------------------------------
# Main CLI / orchestration
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot and report paired EMG synergy metrics from results directories"
    )
    ap.add_argument("--results-dir", type=Path, default=Path("results/synergies"))
    ap.add_argument("--out-dir", type=Path, default=None, help="Default: results-dir/plots_paired")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--conditions", type=str, nargs="*", default=DEFAULT_CONDITIONS, help="Conditions to include")
    ap.add_argument("--selected-muscles", type=str, nargs="*", default=None, help="Selected muscles for small-multiples/profile plots")
    ap.add_argument("--selected-synergies", type=int, nargs="*", default=None, help="Selected synergy indices for profile plots")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = args.out_dir or (results_dir / "plots_paired")
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_muscles = args.selected_muscles or DEFAULT_SELECTED_MUSCLES
    selected_synergies = args.selected_synergies or DEFAULT_SELECTED_SYNERGIES

    root_tables = load_root_tables(results_dir)
    task_records = load_task_level_tables(results_dir)
    task_summary_df = combine_task_summary_tables(root_tables.get("task", pd.DataFrame()), task_records)
    paired_df = build_paired_summary_df(root_tables, task_records)
    muscle_long_df, muscle_paired_df = build_muscle_summary_df(task_records)
    cci_long_df, cci_paired_df, corr_long_df = build_coordination_summary_df(task_records)
    h_paired_df = build_h_paired_summary_df(task_summary_df)
    muscle_profile_df, synergy_profile_df = build_cycle_profile_summary_df(
        task_records,
        selected_muscles=selected_muscles,
        selected_synergies=selected_synergies,
    )

    if paired_df.empty and muscle_paired_df.empty and cci_paired_df.empty:
        print("No compatible summary tables found. Nothing to plot.")
        return

    paired_stats = compute_paired_stats(paired_df, ROOT_METRIC_SPECS)
    muscle_stats = compute_multi_metric_paired_stats(
        muscle_paired_df,
        metric_specs=[
            ("mean_amp", "muscle_amplitude", "Muscle mean amplitude"),
            ("peak_amp", "muscle_amplitude", "Muscle peak amplitude"),
            ("auc", "muscle_amplitude", "Muscle AUC"),
            ("duration", "muscle_timing", "Muscle duration"),
            ("centroid", "muscle_timing", "Muscle centroid"),
            ("cv_mean_amp", "stability", "Muscle CV mean amplitude"),
            ("cv_peak_amp", "stability", "Muscle CV peak amplitude"),
            ("cv_auc", "stability", "Muscle CV AUC"),
            ("mean_similarity_to_mean_cycle", "stability", "Similarity to mean cycle"),
        ],
        label_col="muscle",
    )
    cci_stats = compute_dynamic_paired_stats(
        cci_paired_df,
        metric_col_base="cci",
        label_col="pair_label",
        family="coordination",
        title_prefix="CCI",
    )
    h_stats = compute_multi_metric_paired_stats(
        h_paired_df,
        metric_specs=[
            ("auc", "synergy_h", "Synergy AUC"),
            ("centroid", "synergy_h", "Centroid"),
            ("peak_time", "synergy_h", "Peak time"),
        ],
        label_col="synergy_label",
    )

    save_summary_tables(out_dir, paired_stats, muscle_stats, cci_stats, paired_df, muscle_paired_df, cci_paired_df)

    plot_individual_paired_metric_figures(paired_df, paired_stats, ROOT_METRIC_SPECS, out_dir, args.conditions, args.dpi)
    plot_paired_family_figures(paired_df, paired_stats, ROOT_METRIC_SPECS, out_dir, args.conditions, args.dpi)
    plot_single_metric_figures(paired_df, ROOT_METRIC_SPECS, out_dir, args.conditions, args.dpi)
    plot_muscle_heatmaps(muscle_paired_df, out_dir, args.conditions, args.dpi)
    plot_small_multiples(muscle_paired_df, selected_muscles, out_dir, args.conditions, args.dpi)
    plot_cycle_profiles(muscle_profile_df, synergy_profile_df, out_dir, args.conditions, args.dpi)
    plot_w_heatmaps(task_records, out_dir, args.conditions, args.dpi)
    plot_corr_heatmap(corr_long_df, out_dir, args.conditions, args.dpi)
    plot_h_timing_amplitude_summary(h_paired_df, h_stats, out_dir, args.conditions, args.dpi)
    plot_cci_family(cci_paired_df, cci_stats, out_dir, args.conditions, args.dpi)

    print(f"Done. Outputs in {out_dir}")


if __name__ == "__main__":
    main()
