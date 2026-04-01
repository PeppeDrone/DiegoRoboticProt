#!/usr/bin/env python3
"""synergy_estimation_methodology_figure.py

Publication-quality multi-panel figure summarizing individual-level optimization
of EMG synergy number and NMF solution stability across restarts.

Selection based on Clark-inspired per-muscle VAF criterion; stability and global
fit shown for diagnostics only.

Usage:
  python synergy_estimation_methodology_figure.py
  python synergy_estimation_methodology_figure.py --results-dir results/synergy_estimation
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Path configuration
# -----------------------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = SCRIPT_DIR / "results" / "synergy_estimation"
SUMMARY_DIR = RESULTS_ROOT / "summary_tables"
OUT_DIR = RESULTS_ROOT / "figures"

# File paths
SYNERGY_RECOMMENDATIONS_CSV = SUMMARY_DIR / "synergy_recommendations.csv"
SYNERGY_METRICS_BY_K_CSV = SUMMARY_DIR / "synergy_metrics_by_k.csv"
CLARK_STOPPING_CSV = SUMMARY_DIR / "clark_stopping_diagnostics.csv"
CLARK_STOP_DRIVER_CSV = SUMMARY_DIR / "clark_stop_driver_summary.csv"
SYNERGY_CONFIG_JSON = SUMMARY_DIR / "synergy_estimation_config.json"

# Style constants
FONT_PANEL = 15
FONT_TITLE = 11
FONT_AXIS = 10
FONT_TICK = 9
CLARK_VAF_THRESHOLD = 0.90
CLARK_IMPROVEMENT_THRESHOLD = 0.01
STABILITY_THRESHOLD = 0.85
COLOR_C = "#2ca02c"   # C: all muscles threshold (green)
COLOR_D = "#1f77b4"   # D: worst muscle stopping (blue)
COLOR_F = "#d62728"   # F: k_max fallback (red)
COLOR_NEUTRAL = "#4a5568"

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def safe_read_csv(path: Path) -> pd.DataFrame:
    """Load CSV if it exists; return empty DataFrame otherwise."""
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON if it exists."""
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def parse_json_list(s: Any) -> List[str]:
    """Parse JSON list from string or value."""
    if pd.isna(s) or s is None:
        return []
    if isinstance(s, list):
        return [str(x) for x in s]
    try:
        parsed = json.loads(s) if isinstance(s, str) else s
        return [str(x) for x in parsed] if isinstance(parsed, list) else []
    except Exception:
        return []


def parse_json_dict(s: Any) -> Dict[str, float]:
    """Parse JSON dict from string."""
    if pd.isna(s) or s is None:
        return {}
    try:
        parsed = json.loads(s) if isinstance(s, str) else s
        return dict(parsed) if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def shorten_task_name(name: str) -> str:
    """Shorten task labels for display."""
    if not name:
        return ""
    # e.g. Task_T0_DS -> T0_DS, Task_DS_T0 -> DS_T0
    m = re.match(r"Task[_\-]?(.+)", name, re.I)
    return m.group(1) if m else name[:10]


def example_row_label(idx: int) -> str:
    """Label for representative example rows: Sample A, B, C."""
    return f"Sample {chr(65 + idx)}"


def select_representative_tasks(
    rec_df: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> Dict[str, Tuple[str, str]]:
    """
    Select one representative task per criterion (C, D, F).
    Returns {criterion: (patient_id, task_name)}
    """
    if rec_df.empty or "criterion_used" not in rec_df.columns:
        return {}
    out: Dict[str, Tuple[str, str]] = {}
    for crit in ("C_all_muscles_threshold", "D_worst_muscle_stopping", "F_kmax_fallback"):
        subset = rec_df[rec_df["criterion_used"] == crit]
        if subset.empty:
            continue
        k_col = "k_recommended_clark2010" if "k_recommended_clark2010" in subset.columns else "selected_k"
        if k_col not in subset.columns:
            k_col = subset.columns[0]
        median_k = subset[k_col].median()
        # Pick task with k closest to cohort median among this criterion
        subset = subset.copy()
        subset["_dist"] = (subset[k_col] - median_k).abs()
        best = subset.loc[subset["_dist"].idxmin()]
        out[crit] = (str(best["patient_id"]), str(best["task_name"]))
    return out


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_TICK)


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=FONT_PANEL,
            fontweight="bold", va="top")


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_all_data(
    results_dir: Path,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, Optional[Dict]]:
    """Load all input files. Returns (rec, metrics, stopping, driver, config)."""
    summary_dir = results_dir / "summary_tables"
    rec = safe_read_csv(summary_dir / "synergy_recommendations.csv")
    metrics = safe_read_csv(summary_dir / "synergy_metrics_by_k.csv")
    stopping = safe_read_csv(summary_dir / "clark_stopping_diagnostics.csv")
    driver = safe_read_csv(summary_dir / "clark_stop_driver_summary.csv")
    config = safe_read_json(summary_dir / "synergy_estimation_config.json")
    return rec, metrics, stopping, driver, config


# -----------------------------------------------------------------------------
# Panel builders
# -----------------------------------------------------------------------------


def build_panel_a(
    ax: plt.Axes, rec_df: pd.DataFrame, k_range: Optional[List[int]] = None
) -> bool:
    """Panel A: Distribution of selected k across tasks, stratified by criterion.

    If k_range is provided (e.g., from metrics), uses the same k axis as Panel B
    for visual consistency. Bars for k with no selections show zero height.
    """
    k_col = "k_recommended_clark2010" if "k_recommended_clark2010" in rec_df.columns else None
    if k_col is None or k_col not in rec_df.columns:
        ax.text(0.5, 0.5, "Data unavailable\n(synergy_recommendations.csv)", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_TICK)
        return False
    crit_col = "criterion_used" if "criterion_used" in rec_df.columns else None
    k_vals = rec_df[k_col].dropna().astype(int)
    if k_vals.empty:
        ax.text(0.5, 0.5, "No k data.", ha="center", va="center", transform=ax.transAxes)
        return False
    k_counts = k_vals.value_counts().sort_index()
    # Use k_range if provided (align with Panel B); otherwise use only selected k values
    if k_range is not None:
        k_index = list(k_range)
    else:
        k_index = list(k_counts.index.astype(int))
    x = np.arange(len(k_index))
    w = 0.65
    if crit_col:
        c_counts = rec_df.groupby([k_col, crit_col]).size().unstack(fill_value=0)
        c_counts = c_counts.reindex(k_index, fill_value=0)
        bottom = np.zeros(len(k_index))
        colors_map = {
            "C_all_muscles_threshold": COLOR_C,
            "D_worst_muscle_stopping": COLOR_D,
            "F_kmax_fallback": COLOR_F,
        }
        for col in c_counts.columns:
            vals = c_counts[col].fillna(0).values
            ax.bar(x + (1 - w) / 2, vals, width=w, bottom=bottom, label=col.replace("_", " ").replace("C all muscles threshold", "All muscles ≥0.90").replace("D worst muscle stopping", "Worst muscle plateau").replace("F kmax fallback", "k_max fallback"), color=colors_map.get(col, COLOR_NEUTRAL))
            bottom = bottom + vals
    else:
        vals = [k_counts.get(k, 0) for k in k_index]
        ax.bar(x + (1 - w) / 2, vals, width=w, color=COLOR_NEUTRAL)
    ax.set_xticks(x)
    ax.set_xticklabels(k_index)
    ax.set_xlabel("Selected k", fontsize=FONT_AXIS)
    ax.set_ylabel("Number of tasks", fontsize=FONT_AXIS)
    ax.set_title("Distribution of selected synergies", fontsize=FONT_TITLE)
    if crit_col:
        ax.legend(loc="upper right", fontsize=FONT_TICK - 1)
    return True


def build_panel_b(ax: plt.Axes, metrics_df: pd.DataFrame) -> bool:
    """Panel B: Cohort mean ± SEM for global_vaf, min_per_muscle_vaf, stability_mean vs k."""
    if metrics_df.empty or "k" not in metrics_df.columns:
        ax.text(0.5, 0.5, "Data unavailable\n(synergy_metrics_by_k.csv)", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_TICK)
        return False
    required = ["global_vaf", "min_per_muscle_vaf", "stability_mean"]
    if not all(c in metrics_df.columns for c in required):
        ax.text(0.5, 0.5, "Required columns missing.", ha="center", va="center", transform=ax.transAxes)
        return False
    gs = ax.get_subplotspec().subgridspec(3, 1, hspace=0.35)
    fig = ax.figure
    ax.remove()
    k_vals = sorted(metrics_df["k"].dropna().unique())
    k_arr = np.array(k_vals)
    for i, (metric, ylabel, ref) in enumerate([
        ("global_vaf", "Global VAF", None),
        ("min_per_muscle_vaf", "Min per-muscle VAF", CLARK_VAF_THRESHOLD),
        ("stability_mean", "Restart stability", STABILITY_THRESHOLD),
    ]):
        ax_i = fig.add_subplot(gs[i])
        means, sems = [], []
        for k in k_vals:
            v = metrics_df.loc[metrics_df["k"] == k, metric].dropna()
            means.append(v.mean())
            sems.append(v.std() / np.sqrt(len(v)) if len(v) > 1 else 0)
        err_label = "Mean ± SEM"
        if i == 1:
            avg_sem = float(np.mean(sems))
            err_label = f"Mean ± SEM (Avg SEM = {avg_sem:.3f})"
        ax_i.errorbar(k_arr, means, yerr=sems, fmt="o-", color=COLOR_NEUTRAL, capsize=3, markersize=5, label=err_label)
        if ref is not None:
            ax_i.axhline(ref, color="#a0aec0", ls="--", lw=1)
        ax_i.set_ylabel(ylabel, fontsize=FONT_AXIS - 1)
        ax_i.set_ylim(0, 1.02)
        ax_i.set_xlim(min(k_arr) - 0.3, max(k_arr) + 0.3)
        if i == 0:
            _add_panel_label(ax_i, "B")
            ax_i.set_title("Cohort optimization trajectories", fontsize=FONT_TITLE, pad=4)
        if i == 1:
            ax_i.legend(loc="lower right", fontsize=FONT_TICK - 1)
        _style_axes(ax_i)
    ax_i.set_xlabel("Number of synergies (k)", fontsize=FONT_AXIS)
    return True


def build_panel_d(ax: plt.Axes, stopping_df: pd.DataFrame) -> bool:
    """Panel D: Clark stopping diagnostics – max_improvement vs k."""
    if stopping_df.empty or "max_improvement_among_worst" not in stopping_df.columns:
        ax.text(0.5, 0.5, "Data unavailable\n(clark_stopping_diagnostics.csv)", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_TICK)
        return False
    ax.axhline(CLARK_IMPROVEMENT_THRESHOLD, color="#a0aec0", ls="--", lw=1.5, label=f"Threshold ({CLARK_IMPROVEMENT_THRESHOLD})")
    stop = stopping_df[stopping_df["would_stop_here"] == True]
    cont = stopping_df[stopping_df["would_stop_here"] == False]
    if not stop.empty:
        ax.scatter(stop["k"], stop["max_improvement_among_worst"], c=COLOR_D, s=25, alpha=0.8,
                   label="Stopped here", zorder=3)
    if not cont.empty:
        ax.scatter(cont["k"], cont["max_improvement_among_worst"], c=COLOR_NEUTRAL, s=15, alpha=0.5,
                   label="Continued", zorder=1)
    ax.set_xlabel("k (transition k→k+1)", fontsize=FONT_AXIS)
    ax.set_ylabel("Max improvement among worst muscles", fontsize=FONT_AXIS)
    ax.set_title("Clark stopping-rule diagnostics", fontsize=FONT_TITLE)
    ax.legend(loc="upper right", fontsize=FONT_TICK - 1)
    ax.set_ylim(-0.02, min(0.2, stopping_df["max_improvement_among_worst"].max() * 1.2) if len(stopping_df) else 0.2)
    _style_axes(ax)
    return True


def build_panel_e(ax: plt.Axes, driver_df: pd.DataFrame) -> bool:
    """Panel E: Stop-driver muscles frequency (horizontal bar)."""
    if driver_df.empty or "stop_driver_muscles_json" not in driver_df.columns:
        ax.text(0.5, 0.5, "Data unavailable\n(clark_stop_driver_summary.csv)", ha="center", va="center",
                transform=ax.transAxes, fontsize=FONT_TICK)
        return False
    counts: Dict[str, int] = {}
    for _, row in driver_df.iterrows():
        muscles = parse_json_list(row["stop_driver_muscles_json"])
        for m in muscles:
            counts[m] = counts.get(m, 0) + 1
    if not counts:
        ax.text(0.5, 0.5, "No stop-driver data.", ha="center", va="center", transform=ax.transAxes)
        return False
    sorted_items = sorted(counts.items(), key=lambda x: -x[1])
    muscles = [x[0].replace(" Brachii ", " B. ").replace(" Triceps ", " Tri. ") for x in sorted_items]
    freqs = [x[1] for x in sorted_items]
    y = np.arange(len(muscles))
    ax.barh(y, freqs, color=COLOR_D, height=0.7, alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(muscles, fontsize=FONT_TICK - 1)
    ax.set_xlabel("Frequency as stop-driver", fontsize=FONT_AXIS)
    _style_axes(ax)
    return True


def build_panel_f(ax: plt.Axes, rec_df: pd.DataFrame, metrics_df: pd.DataFrame, reps: Dict[str, Tuple[str, str]]) -> bool:
    """Panel F: Representative individual examples (2–3 tasks)."""
    if metrics_df.empty or "k" not in metrics_df.columns:
        ax.text(0.5, 0.5, "Data unavailable", ha="center", va="center", transform=ax.transAxes)
        return False
    required = ["global_vaf", "min_per_muscle_vaf", "stability_mean"]
    if not all(c in metrics_df.columns for c in required):
        ax.text(0.5, 0.5, "Required columns missing.", ha="center", va="center", transform=ax.transAxes)
        return False
    examples: List[Tuple[str, str, str]] = []
    for crit, label in [
        ("C_all_muscles_threshold", "All muscles ≥0.90"),
        ("D_worst_muscle_stopping", "Worst muscle plateau"),
        ("F_kmax_fallback", "k_max fallback"),
    ]:
        if crit in reps:
            pid, tname = reps[crit]
            examples.append((pid, tname, label))
    if not examples:
        examples = [(str(rec_df["patient_id"].iloc[0]), str(rec_df["task_name"].iloc[0]), "Example")]
    n_ex = len(examples)
    gs = ax.get_subplotspec().subgridspec(n_ex, 3, hspace=0.25, wspace=0.2)
    fig = ax.figure
    ax.remove()
    for ex_idx, (pid, tname, reason) in enumerate(examples):
        sub = metrics_df[(metrics_df["patient_id"].astype(str) == pid) & (metrics_df["task_name"].astype(str) == tname)]
        if sub.empty:
            continue
        sel_k = rec_df[(rec_df["patient_id"].astype(str) == pid) & (rec_df["task_name"].astype(str) == tname)]
        sel_k_val = int(sel_k["k_recommended_clark2010"].iloc[0]) if "k_recommended_clark2010" in sel_k.columns and not sel_k.empty else int(sub["k"].max())
        k_vals = sub["k"].values
        for m_idx, (metric, ylab) in enumerate([
            ("global_vaf", "Global VAF"),
            ("min_per_muscle_vaf", "Min muscle VAF"),
            ("stability_mean", "Stability"),
        ]):
            ax_ij = fig.add_subplot(gs[ex_idx, m_idx])
            ax_ij.plot(k_vals, sub[metric].values, "o-", color=COLOR_NEUTRAL, markersize=4)
            ax_ij.axvline(sel_k_val, color=COLOR_D, ls="--", lw=1)
            row_lab = example_row_label(ex_idx)
            if m_idx == 0:
                ax_ij.set_ylabel(f"{row_lab}\n{ylab}", fontsize=FONT_AXIS - 1)
            else:
                ax_ij.set_ylabel(ylab, fontsize=FONT_AXIS - 1)
            ax_ij.set_ylim(0, 1.02)
            # Only center column (Min muscle VAF) gets a title
            if ex_idx == 0 and m_idx == 1:
                ax_ij.set_title("Min muscle VAF", fontsize=FONT_TITLE - 1)
            if ex_idx == len(examples) - 1:
                ax_ij.set_xlabel("k", fontsize=FONT_AXIS - 1)
            # Grey criterion text only on center (middle) subpanel
            if m_idx == 1:
                ax_ij.text(0.05, 0.95, reason, transform=ax_ij.transAxes, fontsize=FONT_TICK - 2,
                           va="top", style="italic", color="#718096")
            if ex_idx == 0 and m_idx == 0:
                _add_panel_label(ax_ij, "E")
            _style_axes(ax_ij)
    return True


# -----------------------------------------------------------------------------
# Figure data summary for reproducibility
# -----------------------------------------------------------------------------


def _build_figure_data_summary(
    rec: pd.DataFrame,
    metrics: pd.DataFrame,
    stopping: pd.DataFrame,
    driver: pd.DataFrame,
    config: Optional[Dict[str, Any]],
    reps: Dict[str, Tuple[str, str]],
    results_dir: Path,
) -> Dict[str, Any]:
    """Build a JSON-serializable summary of all data used for the figure."""
    summary_dir = results_dir / "summary_tables"
    summary: Dict[str, Any] = {
        "input_files": {
            "synergy_recommendations.csv": str(summary_dir / "synergy_recommendations.csv"),
            "synergy_metrics_by_k.csv": str(summary_dir / "synergy_metrics_by_k.csv"),
            "clark_stopping_diagnostics.csv": str(summary_dir / "clark_stopping_diagnostics.csv"),
            "clark_stop_driver_summary.csv": str(summary_dir / "clark_stop_driver_summary.csv"),
            "synergy_estimation_config.json": str(summary_dir / "synergy_estimation_config.json"),
        },
        "config": dict(config) if config else {},
        "thresholds_used": {
            "clark_vaf_threshold": CLARK_VAF_THRESHOLD,
            "clark_improvement_threshold": CLARK_IMPROVEMENT_THRESHOLD,
            "stability_threshold": STABILITY_THRESHOLD,
        },
        "panel_a": {},
        "panel_b": {},
        "panel_c": {},
        "panel_d": {},
        "panel_e": {},
    }
    # Panel A: k distribution by criterion
    if not rec.empty and "k_recommended_clark2010" in rec.columns:
        k_col = "k_recommended_clark2010"
        crit_col = "criterion_used" if "criterion_used" in rec.columns else None
        k_counts = rec[k_col].dropna().astype(int).value_counts().sort_index()
        summary["panel_a"]["k_counts"] = {int(k): int(v) for k, v in k_counts.items()}
        if crit_col:
            summary["panel_a"]["by_criterion"] = {
                str(c): int(n) for c, n in rec.groupby(crit_col).size().items()
            }
        summary["panel_a"]["n_tasks"] = int(len(rec))
    # Panel B: cohort trajectories
    if not metrics.empty and "k" in metrics.columns:
        k_vals = sorted(metrics["k"].dropna().unique())
        traj = []
        for k in k_vals:
            sub = metrics[metrics["k"] == k]
            row = {"k": int(k)}
            for m in ("global_vaf", "min_per_muscle_vaf", "stability_mean"):
                if m in sub.columns:
                    v = sub[m].dropna()
                    row[m] = {"mean": float(v.mean()), "sem": float(v.std() / np.sqrt(len(v))) if len(v) > 1 else 0.0, "n": int(len(v))}
            traj.append(row)
        summary["panel_b"]["trajectories"] = traj
    # Panel C: Clark stopping
    if not stopping.empty:
        summary["panel_c"]["n_transitions"] = int(len(stopping))
        if "would_stop_here" in stopping.columns:
            summary["panel_c"]["n_stopped"] = int((stopping["would_stop_here"] == True).sum())
        if "max_improvement_among_worst" in stopping.columns:
            imp = stopping["max_improvement_among_worst"].dropna()
            summary["panel_c"]["max_improvement_range"] = [float(imp.min()), float(imp.max())] if len(imp) else []
    # Panel D: stop-driver muscles
    if not driver.empty and "stop_driver_muscles_json" in driver.columns:
        counts: Dict[str, int] = {}
        for _, row in driver.iterrows():
            for m in parse_json_list(row["stop_driver_muscles_json"]):
                counts[m] = counts.get(m, 0) + 1
        summary["panel_d"]["stop_driver_frequencies"] = counts
    # Panel E: representative examples
    summary["panel_e"]["representative_tasks"] = {
        str(crit): {"patient_id": str(pid), "task_name": str(tname)}
        for crit, (pid, tname) in reps.items()
    }
    for crit, (pid, tname) in reps.items():
        sel = rec[(rec["patient_id"].astype(str) == pid) & (rec["task_name"].astype(str) == tname)]
        if not sel.empty and "k_recommended_clark2010" in sel.columns:
            summary["panel_e"]["representative_tasks"][str(crit)]["selected_k"] = int(sel["k_recommended_clark2010"].iloc[0])
    return summary


def _print_figure_data_summary(summary: Dict[str, Any], summary_path: Path) -> None:
    """Print key figure data to terminal for quick inspection."""
    print("\n--- Synergy estimation methodology figure: data summary ---")
    print(f"Input dir: {summary.get('input_files', {}).get('synergy_recommendations.csv', '?')}")
    print(f"Thresholds: VAF={summary.get('thresholds_used', {}).get('clark_vaf_threshold')}, improvement={summary.get('thresholds_used', {}).get('clark_improvement_threshold')}")
    if summary.get("panel_a"):
        print(f"Panel A: n_tasks={summary['panel_a'].get('n_tasks')}, k_distribution={summary['panel_a'].get('k_counts', {})}")
    if summary.get("panel_b", {}).get("trajectories"):
        print(f"Panel B: k range {[t['k'] for t in summary['panel_b']['trajectories']]}")
    if summary.get("panel_c"):
        print(f"Panel C: n_transitions={summary['panel_c'].get('n_transitions')}, n_stopped={summary['panel_c'].get('n_stopped')}")
    if summary.get("panel_d", {}).get("stop_driver_frequencies"):
        print(f"Panel D: top stop-drivers = {dict(sorted(summary['panel_d']['stop_driver_frequencies'].items(), key=lambda x: -x[1])[:5])}")
    if summary.get("panel_e", {}).get("representative_tasks"):
        for c, r in summary["panel_e"]["representative_tasks"].items():
            print(f"Panel E: {c} -> {r.get('patient_id')}/{r.get('task_name')} k={r.get('selected_k', '?')}")
    print(f"Full summary saved: {summary_path}")
    print("---\n")


# -----------------------------------------------------------------------------
# Main figure creation
# -----------------------------------------------------------------------------


def create_figure(
    results_dir: Path,
    out_path: Path,
    dpi: int = 300,
) -> None:
    """Create the 5-panel methodology figure (A,B,C on row 1; D,E on row 2)."""
    rec, metrics, stopping, driver, config = load_all_data(results_dir)
    if config:
        global CLARK_VAF_THRESHOLD, CLARK_IMPROVEMENT_THRESHOLD, STABILITY_THRESHOLD
        CLARK_VAF_THRESHOLD = config.get("clark_muscle_vaf_threshold", 0.9)
        CLARK_IMPROVEMENT_THRESHOLD = config.get("clark_improvement_threshold", 0.01)
        STABILITY_THRESHOLD = config.get("stability_threshold_reporting_only", 0.85)

    reps = select_representative_tasks(rec, metrics) if not rec.empty else {}

    fig = plt.figure(figsize=(11, 11), facecolor="white")
    fig.patch.set_facecolor("white")
    # Row 1: A,B,C  |  Row 2: D (narrower), E (wider, more space for examples)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.35)
    gs_top = gs[0].subgridspec(1, 3, wspace=0.25)
    gs_bot = gs[1].subgridspec(1, 2, wspace=0.12, width_ratios=[0.45, 0.55])

    ax_a = fig.add_subplot(gs_top[0, 0])
    ax_b = fig.add_subplot(gs_top[0, 1])
    ax_c = fig.add_subplot(gs_top[0, 2])  # Clark stopping (was D)
    ax_d = fig.add_subplot(gs_bot[0, 0])  # Stop drivers (was E)
    ax_e = fig.add_subplot(gs_bot[0, 1])  # Representative examples (was F)

    # Panel A: use same k range as Panel B for aligned x-axes
    k_range = None
    if not metrics.empty and "k" in metrics.columns:
        k_range = sorted(int(k) for k in metrics["k"].dropna().unique())
    ok_a = build_panel_a(ax_a, rec, k_range=k_range)
    if ok_a:
        _style_axes(ax_a)
        _add_panel_label(ax_a, "A")

    # Panel B (replaces ax_b with subplots)
    ok_b = build_panel_b(ax_b, metrics)
    if not ok_b:
        _add_panel_label(ax_b, "B")

    # Panel C (Clark stopping, was D)
    ok_c = build_panel_d(ax_c, stopping)
    if ok_c:
        _add_panel_label(ax_c, "C")

    # Panel D (Stop drivers, was E)
    ok_d = build_panel_e(ax_d, driver)
    if ok_d:
        _add_panel_label(ax_d, "D")

    # Panel E (Representative examples, was F)
    ok_e = build_panel_f(ax_e, rec, metrics, reps)
    if not ok_e:
        _add_panel_label(ax_e, "E")

    # Method note
    fig.text(0.5, 0.01, "Selection based on per-muscle VAF criterion; stability and global fit shown for diagnostics only.",
             ha="center", fontsize=FONT_TICK - 1, style="italic", color="#718096")

    plt.subplots_adjust(left=0.08, right=0.96, top=0.94, bottom=0.05, hspace=0.35, wspace=0.25)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    base = out_path.with_suffix("")
    fig.savefig(base.with_suffix(".png"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".pdf"), dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(base.with_suffix(".svg"), format="svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # Write and print figure data summary for reproducibility
    summary = _build_figure_data_summary(rec, metrics, stopping, driver, config, reps, results_dir)
    summary_path = base.with_suffix(".figure_data.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    _print_figure_data_summary(summary, summary_path)
    print(f"Figure saved: {base.with_suffix('.png')}, {base.with_suffix('.pdf')}, {base.with_suffix('.svg')}")


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synergy estimation methodology figure")
    ap.add_argument("--results-dir", type=Path, default=RESULTS_ROOT)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--dpi", type=int, default=300)
    args = ap.parse_args()
    out_dir = args.out_dir or args.results_dir / "figures"
    out_path = out_dir / "synergy_estimation_methodology_figure.png"
    create_figure(args.results_dir, out_path, args.dpi)


if __name__ == "__main__":
    main()
