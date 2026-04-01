#!/usr/bin/env python3
"""B2_plot_synergy_estimation.py

Generate synergy estimation figures from CSV outputs of B2_estimate_synergy_number.py.

Reads:
  - synergy_metrics_by_k.csv
  - synergy_recommendations.csv
  - synergy_estimation_config.json

Produces:
  - One combined figure per patient/task (2x3 panels: global VAF, min per-muscle VAF,
    mean per-muscle VAF, delta VAF, AIC/BIC, stability)
  - All-tasks summary figure

Usage:
  python B2_plot_synergy_estimation.py
  python B2_plot_synergy_estimation.py --results-dir results/synergy_estimation
  python B2_plot_synergy_estimation.py --results-dir results/synergy_estimation --dpi 200
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    LOG.setLevel(level)


def load_results(results_dir: Path) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Load metrics, recommendations, and config from CSV/JSON. Return (results, config)."""
    summary_dir = results_dir / "summary_tables"
    metrics_path = summary_dir / "synergy_metrics_by_k.csv"
    recs_path = summary_dir / "synergy_recommendations.csv"
    config_path = summary_dir / "synergy_estimation_config.json"

    if not metrics_path.exists():
        raise FileNotFoundError(f"Metrics CSV not found: {metrics_path}. Run B2_estimate_synergy_number.py first.")
    if not recs_path.exists():
        raise FileNotFoundError(f"Recommendations CSV not found: {recs_path}.")

    metrics_df = pd.read_csv(metrics_path)
    recs_df = pd.read_csv(recs_path)
    config: Dict[str, Any] = {}
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)

    # Build results list: one dict per task with metrics + recommendation fields
    results: List[Dict[str, Any]] = []
    for _, rec_row in recs_df.iterrows():
        pid = rec_row["patient_id"]
        tname = rec_row["task_name"]
        task_metrics = metrics_df[(metrics_df["patient_id"] == pid) & (metrics_df["task_name"] == tname)]
        rows = task_metrics.to_dict("records")
        # Parse per_muscle_vaf if it's a json string
        for r in rows:
            if "per_muscle_vaf" in r and isinstance(r["per_muscle_vaf"], str):
                try:
                    r["per_muscle_vaf"] = json.loads(r["per_muscle_vaf"])
                except (json.JSONDecodeError, TypeError):
                    pass
        res = {
            "patient_id": pid,
            "task_name": tname,
            "metrics": rows,
            "k_recommended_clark2010": int(rec_row["k_recommended_clark2010"]),
            "reason_for_clark2010_recommendation": str(rec_row["reason_for_clark2010_recommendation"]),
            "k_recommended_aic": int(rec_row["k_recommended_aic"]),
            "k_recommended_bic": int(rec_row["k_recommended_bic"]),
            "n_samples": int(rec_row["n_samples"]),
            "n_muscles": int(rec_row["n_muscles"]),
        }
        results.append(res)

    return results, config


def plot_per_task(
    res: Dict[str, Any],
    out_dir: Path,
    dpi: int,
    clark_muscle_thresh: float,
    clark_improvement_thresh: float,
    stability_thresh: float,
) -> None:
    """Single combined 2x2 figure: global VAF + delta VAF, min + mean per-muscle VAF, AIC/BIC, stability."""
    df = pd.DataFrame(res["metrics"])
    if df.empty:
        LOG.warning("No metrics for %s %s, skipping figure", res["patient_id"], res["task_name"])
        return
    k_vals = df["k"].to_numpy()
    k_selected = res["k_recommended_clark2010"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True)

    # Panel (0,0): Global VAF (left axis) + Delta VAF (right axis), vertical dashed grey at selected k
    ax0 = axes[0, 0]
    line_global, = ax0.plot(k_vals, df["global_vaf"], "o-", color="steelblue", linewidth=2, markersize=6, label="Global VAF")
    vline_selected = ax0.axvline(k_selected, color="gray", linestyle="--", alpha=0.8, linewidth=1.5, label="Selected N")
    ax0.set_xlabel("k")
    ax0.set_ylabel("Global VAF", color="steelblue")
    ax0.tick_params(axis="y", labelcolor="steelblue")
    ax0.set_xticks(k_vals.astype(int))
    ax0.grid(True, alpha=0.3)
    ax0.set_ylim(0, 1.05)

    ax0_twin = ax0.twinx()
    delta = df["delta_vaf"].to_numpy()
    valid_delta = ~np.isnan(delta)
    line_delta = None
    if np.any(valid_delta):
        line_delta, = ax0_twin.plot(k_vals[valid_delta], delta[valid_delta], "s-", color="darkviolet", linewidth=2, markersize=5, label="Delta VAF")
    ax0_twin.set_ylabel("Delta VAF", color="darkviolet")
    ax0_twin.tick_params(axis="y", labelcolor="darkviolet")
    # Combined legend: Global VAF, Delta VAF, Selected N
    legs = [line_global]
    labs = ["Global VAF"]
    if line_delta is not None:
        legs.append(line_delta)
        labs.append("Delta VAF")
    legs.append(vline_selected)
    labs.append("Selected N")
    ax0.legend(legs, labs, loc="upper left", fontsize=7)

    # Panel (0,1): Min + Mean per-muscle VAF, vertical dashed at selected k
    ax1 = axes[0, 1]
    ax1.plot(k_vals, df["min_per_muscle_vaf"], "o-", color="coral", linewidth=2, markersize=6, label="Min per-muscle VAF")
    ax1.plot(k_vals, df["mean_per_muscle_vaf"], "o-", color="teal", linewidth=2, markersize=6, label="Mean per-muscle VAF")
    ax1.axvline(k_selected, color="gray", linestyle="--", alpha=0.8, linewidth=1.5)
    ax1.set_xlabel("k")
    ax1.set_ylabel("Per-muscle VAF")
    ax1.set_xticks(k_vals.astype(int))
    ax1.legend(fontsize=7)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 1.05)

    # Panel (1,0): AIC and BIC only, no selection markers
    ax2 = axes[1, 0]
    ax2.plot(k_vals, df["aic"], "o-", color="darkorange", linewidth=2, markersize=6, label="AIC")
    ax2.plot(k_vals, df["bic"], "s-", color="forestgreen", linewidth=2, markersize=5, label="BIC")
    ax2.set_xlabel("k")
    ax2.set_ylabel("AIC / BIC")
    ax2.set_xticks(k_vals.astype(int))
    ax2.legend(fontsize=7)
    ax2.grid(True, alpha=0.3)

    # Panel (1,1): Stability, y-axis 0.5–1, no legend
    ax3 = axes[1, 1]
    stab_mean = df["stability_mean"].to_numpy()
    stab_std = df["stability_std"].to_numpy()
    valid_stab = ~np.isnan(stab_mean)
    if np.any(valid_stab):
        ax3.plot(k_vals[valid_stab], stab_mean[valid_stab], "o-", color="purple", linewidth=2, markersize=6)
        valid_band = valid_stab & ~np.isnan(stab_std)
        if np.any(valid_band):
            ax3.fill_between(
                k_vals[valid_band],
                (stab_mean - stab_std)[valid_band],
                (stab_mean + stab_std)[valid_band],
                alpha=0.2,
                color="purple",
            )
    else:
        ax3.text(0.5, 0.5, "Stability unavailable", ha="center", va="center", transform=ax3.transAxes)
    ax3.set_xlabel("k")
    ax3.set_ylabel("Stability")
    ax3.set_xticks(k_vals.astype(int))
    ax3.set_ylim(0.7, 1.3)
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"{res['patient_id']} | {res['task_name']} | Clark 2010: k={res['k_recommended_clark2010']} ({res['reason_for_clark2010_recommendation']})",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    safe_name = res["task_name"].replace(" ", "_").replace("/", "_")
    fig.savefig(out_dir / f"synergy_estimation_{res['patient_id']}_{safe_name}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_all_tasks_summary(
    results: List[Dict[str, Any]],
    out_dir: Path,
    dpi: int,
    clark_muscle_thresh: float,
    stability_thresh: float,
) -> None:
    """Summary: gray individual trajectories (one per patient/task), T0/T1 dashed averages."""
    if len(results) < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)

    # Layout: (0,0)=Global VAF, (0,1)=Min per-muscle VAF, (1,0)=AIC, (1,1)=Stability
    for res in results:
        df = pd.DataFrame(res["metrics"])
        axes[0, 0].plot(df["k"], df["global_vaf"], "-", color="gray", linewidth=1, alpha=0.6)
        axes[0, 1].plot(df["k"], df["aic"], "-", color="gray", linewidth=1, alpha=0.6)
        axes[1, 0].plot(df["k"], df["min_per_muscle_vaf"], "-", color="gray", linewidth=1, alpha=0.6)
        stab = df["stability_mean"].to_numpy()
        valid = ~np.isnan(stab)
        if np.any(valid):
            axes[1, 1].plot(df["k"].values[valid], stab[valid], "-", color="gray", linewidth=1, alpha=0.6)

    def avg_by_k(rlist: List[Dict], col: str) -> Tuple[np.ndarray, np.ndarray]:
        if not rlist:
            return np.array([]), np.array([])
        dfs = [pd.DataFrame(r["metrics"]) for r in rlist]
        all_k = np.unique(np.concatenate([d["k"].values for d in dfs])).astype(int)
        rows = []
        for d in dfs:
            dk = d.set_index("k").reindex(all_k)
            rows.append(dk[col].values)
        mean = np.nanmean(rows, axis=0)
        valid = ~np.isnan(mean)
        return all_k[valid], mean[valid]

    t0_res = [r for r in results if "T0" in r["task_name"]]
    t1_res = [r for r in results if "T1" in r["task_name"]]

    if t0_res:
        k, v = avg_by_k(t0_res, "global_vaf")
        axes[0, 0].plot(k, v, "--", color="tab:blue", linewidth=2.5, label="T0")
        k, v = avg_by_k(t0_res, "min_per_muscle_vaf")
        axes[1, 0].plot(k, v, "--", color="tab:blue", linewidth=2.5, label="T0")
        k, v = avg_by_k(t0_res, "aic")
        axes[0, 1].plot(k, v, "--", color="tab:blue", linewidth=2.5, label="T0")
        k, v = avg_by_k(t0_res, "stability_mean")
        if len(k) > 0:
            axes[1, 1].plot(k, v, "--", color="tab:blue", linewidth=2.5, label="T0")
    if t1_res:
        k, v = avg_by_k(t1_res, "global_vaf")
        axes[0, 0].plot(k, v, "--", color="tab:orange", linewidth=2.5, label="T1")
        k, v = avg_by_k(t1_res, "min_per_muscle_vaf")
        axes[1, 0].plot(k, v, "--", color="tab:orange", linewidth=2.5, label="T1")
        k, v = avg_by_k(t1_res, "aic")
        axes[0, 1].plot(k, v, "--", color="tab:orange", linewidth=2.5, label="T1")
        k, v = avg_by_k(t1_res, "stability_mean")
        if len(k) > 0:
            axes[1, 1].plot(k, v, "--", color="tab:orange", linewidth=2.5, label="T1")

    axes[0, 0].set_xlabel("k")
    axes[0, 0].set_ylabel("Global VAF")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("k")
    axes[0, 1].set_ylabel("AIC")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("k")
    axes[1, 0].set_ylabel("Min per-muscle VAF")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("k")
    axes[1, 1].set_ylabel("Stability")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].set_ylim(0.7, 1.3)
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(
        "Synergy estimation: all tasks (gray=individual task, dashed=T0/T1 mean)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_dir / "synergy_estimation_all_tasks.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Plot synergy estimation figures from B2 CSV outputs"
    )
    ap.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results/synergy_estimation"),
        help="Directory containing summary_tables/ (from B2_estimate_synergy_number.py)",
    )
    ap.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    _setup_logging(args.verbose)

    LOG.info("Loading results from %s", args.results_dir)
    results, config = load_results(args.results_dir)
    LOG.info("Loaded %d tasks", len(results))

    clark_muscle = config.get("clark_muscle_vaf_threshold", 0.90)
    clark_improvement = config.get("clark_improvement_threshold", 0.05)
    stability = config.get("stability_threshold_reporting_only", 0.85)

    figures_dir = args.results_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    for i, res in enumerate(results):
        LOG.info("Plotting %s | %s (%d/%d)", res["patient_id"], res["task_name"], i + 1, len(results))
        plot_per_task(
            res,
            figures_dir,
            args.dpi,
            clark_muscle_thresh=clark_muscle,
            clark_improvement_thresh=clark_improvement,
            stability_thresh=stability,
        )

    plot_all_tasks_summary(
        results,
        figures_dir,
        args.dpi,
        clark_muscle_thresh=clark_muscle,
        stability_thresh=stability,
    )

    print(f"Figures written to {figures_dir}")


if __name__ == "__main__":
    main()
