#!/usr/bin/env python3
"""emg_channel_qc_report.py

Standalone QC pipeline for raw EMG channels.
Scans all patients/tasks, computes quality metrics, saves tables, and generates cohort figures.

Usage:
  python emg_channel_qc_report.py --data-dir data/emg_structured --out-dir results/emg_qc
  python emg_channel_qc_report.py --patients CROSS_001 CROSS_003 --max-tasks 5
"""

from __future__ import annotations

import argparse
import pickle
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.signal import butter, filtfilt, welch

# Pickle compatibility for EMGRecord
import B0_parse_emg_patient_task
sys.modules["__main__"].ChannelInfo = B0_parse_emg_patient_task.ChannelInfo
sys.modules["__main__"].EMGRecord = B0_parse_emg_patient_task.EMGRecord

from B0_parse_emg_patient_task import EMGRecord

# -----------------------------------------------------------------------------
# QC thresholds (heuristic, not absolute truth)
# -----------------------------------------------------------------------------
MIN_SAMPLES = 50
MOTION_ARTIFACT_RATIO_THRESH = 0.30
OFFSET_RATIO_THRESH = 0.10
DRIFT_RATIO_THRESH = 0.20
CLIPPING_FRACTION_THRESH = 0.01
FLATLINE_FRACTION_THRESH = 0.05
LINE_NOISE_RATIO_THRESH = 0.20
RMS_NEAR_ZERO_THRESH = 1e-9  # RMS below this = likely dead channel
QC_BAD_SCORE = 4
QC_WARNING_SCORE = 2
CLIPPING_TOLERANCE_PCT = 0.005  # 0.5% of dynamic range
FLATLINE_DIFF_THRESH_PCT = 0.0001  # diff threshold as fraction of dynamic range
EMG_BAND_LOW_HZ = 20.0
EMG_BAND_HIGH_HZ = 450.0
MOTION_BAND_LOW_HZ = 1.0
MOTION_BAND_HIGH_HZ = 20.0
LINE_NOISE_BW_HZ = 1.0  # ±1 Hz around line freq
DRIFT_LOWPASS_HZ = 5.0
DRIFT_ORDER = 4

QC_FLAG_COLORS = {"ok": "#2e7d32", "warning": "#ef6c00", "bad": "#c62828", "too_short": "#757575"}
QC_FLAG_ORDER = ["ok", "warning", "bad", "too_short"]


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_emg_record(patient_id: str, task_name: str, data_dir: Path) -> Optional[EMGRecord]:
    """Load EMG pickle. Returns None on failure."""
    pkl_path = data_dir / f"{patient_id}_EMG" / f"{task_name}_emg.pkl"
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        warnings.warn(f"Failed to load {pkl_path}: {e}")
        return None


def get_fs_from_record(rec: EMGRecord) -> float:
    """Extract sampling frequency from record."""
    if not rec.channels:
        return 150.0
    return float(rec.channels[0].sampling_freq)


def discover_patients_with_data(data_dir: Path) -> List[str]:
    """Return sorted patient IDs with EMG folders."""
    if not data_dir.exists():
        return []
    return sorted(
        pdir.name.replace("_EMG", "")
        for pdir in data_dir.iterdir()
        if pdir.is_dir() and pdir.name.endswith("_EMG")
    )


def discover_tasks_for_patient(patient_id: str, data_dir: Path) -> List[str]:
    """Return sorted task names (excluding Calibrazione)."""
    pdir = data_dir / f"{patient_id}_EMG"
    if not pdir.exists():
        return []
    out = []
    for p in sorted(pdir.glob("*_emg.pkl")):
        task = p.stem.replace("_emg", "")
        if "Calibrazione" not in task:
            out.append(task)
    return out


def get_emg_channels(rec: EMGRecord) -> List[str]:
    """Return EMG channel names, excluding Acc/IMU."""
    if hasattr(rec, "get_emg_channels") and rec.channels:
        chans = rec.get_emg_channels()
        if chans:
            return chans
    return [c for c in rec.data if "EMG" in c and "Acc" not in c and "acc" not in c.lower()]


# -----------------------------------------------------------------------------
# Signal processing
# -----------------------------------------------------------------------------


def bandpower(psd: np.ndarray, freqs: np.ndarray, low_hz: float, high_hz: float, fs: float) -> Optional[float]:
    """Integrate PSD over frequency band. Returns None if band empty."""
    mask = (freqs >= low_hz) & (freqs <= high_hz)
    if not np.any(mask):
        return None
    return float(trapezoid(psd[mask], freqs[mask]))


def lowpass_signal(x: np.ndarray, fs: float, cutoff_hz: float = DRIFT_LOWPASS_HZ, order: int = DRIFT_ORDER) -> np.ndarray:
    """Butterworth low-pass with filtfilt."""
    nyq = fs / 2
    if cutoff_hz >= nyq:
        return x.copy()
    low = np.clip(cutoff_hz / nyq, 1e-6, 0.99)
    b, a = butter(order, low, btype="low")
    return filtfilt(b, a, np.asarray(x, dtype=float))


def compute_emg_channel_qc(
    raw: np.ndarray,
    fs: float,
    line_freq: float = 50.0,
) -> Dict[str, Any]:
    """Compute QC metrics for one raw EMG channel. Returns dict of metrics."""
    raw = np.asarray(raw, dtype=float)
    raw = raw[np.isfinite(raw)]
    n = len(raw)
    out: Dict[str, Any] = {
        "n_samples": n,
        "duration_s": n / fs if fs > 0 else np.nan,
        "fs": fs,
        "signal_mean": np.nan,
        "signal_std": np.nan,
        "signal_min": np.nan,
        "signal_max": np.nan,
        "rms": np.nan,
        "motion_artifact_ratio": np.nan,
        "offset_ratio": np.nan,
        "drift_ratio": np.nan,
        "clipping_fraction": np.nan,
        "flatline_fraction": np.nan,
        "line_noise_ratio": np.nan,
    }
    if n < MIN_SAMPLES:
        return out

    mean_v = float(np.mean(raw))
    std_v = float(np.std(raw))
    rms = float(np.sqrt(np.mean(raw ** 2)))
    vmin, vmax = float(np.min(raw)), float(np.max(raw))
    dyn = vmax - vmin
    if dyn < 1e-30:
        dyn = 1e-30

    out["signal_mean"] = mean_v
    out["signal_std"] = std_v
    out["signal_min"] = vmin
    out["signal_max"] = vmax
    out["rms"] = rms

    # offset_ratio
    if std_v > 1e-30:
        out["offset_ratio"] = float(np.abs(mean_v) / std_v)
    else:
        out["offset_ratio"] = np.nan if mean_v == 0 else np.inf

    # drift_ratio: lowpass at 5 Hz, RMS of trend / RMS of raw
    try:
        trend = lowpass_signal(raw, fs, cutoff_hz=DRIFT_LOWPASS_HZ)
        rms_trend = float(np.sqrt(np.mean(trend ** 2)))
        if rms > 1e-30:
            out["drift_ratio"] = rms_trend / rms
        else:
            out["drift_ratio"] = np.nan
    except Exception:
        out["drift_ratio"] = np.nan

    # clipping_fraction: samples within 0.5% of min or max
    tol = CLIPPING_TOLERANCE_PCT * dyn
    near_min = np.sum(raw <= vmin + tol)
    near_max = np.sum(raw >= vmax - tol)
    out["clipping_fraction"] = (near_min + near_max) / n

    # flatline_fraction: first diff below threshold
    diff = np.abs(np.diff(raw))
    thresh = max(FLATLINE_DIFF_THRESH_PCT * dyn, 1e-30)
    out["flatline_fraction"] = float(np.sum(diff < thresh) / (n - 1)) if n > 1 else 0.0

    # Welch PSD for motion, line noise, EMG band
    nyq = fs / 2
    nperseg = min(n, max(256, int(fs * 2)))
    if nperseg < 8:
        return out
    try:
        freqs, psd = welch(raw, fs=fs, nperseg=nperseg)
    except Exception:
        return out

    emg_high = min(EMG_BAND_HIGH_HZ, nyq * 0.95)
    emg_power = bandpower(psd, freqs, EMG_BAND_LOW_HZ, emg_high, fs)
    if emg_power is None or emg_power <= 0:
        emg_power = 1e-30

    # motion_artifact_ratio
    motion_power = bandpower(psd, freqs, MOTION_BAND_LOW_HZ, MOTION_BAND_HIGH_HZ, fs)
    if motion_power is not None and emg_power > 0:
        out["motion_artifact_ratio"] = motion_power / emg_power

    # line_noise_ratio
    line_lo = line_freq - LINE_NOISE_BW_HZ
    line_hi = line_freq + LINE_NOISE_BW_HZ
    line_power = bandpower(psd, freqs, line_lo, line_hi, fs)
    if line_power is not None and emg_power > 0:
        out["line_noise_ratio"] = line_power / emg_power

    return out


def compute_qc_score(row: Dict[str, Any]) -> Tuple[int, str]:
    """Rule-based QC score and flag from metrics dict."""
    score = 0
    if row.get("motion_artifact_ratio", 0) is not None and np.isfinite(row["motion_artifact_ratio"]) and row["motion_artifact_ratio"] > MOTION_ARTIFACT_RATIO_THRESH:
        score += 1
    if row.get("offset_ratio", 0) is not None and np.isfinite(row["offset_ratio"]) and row["offset_ratio"] > OFFSET_RATIO_THRESH:
        score += 1
    if row.get("drift_ratio", 0) is not None and np.isfinite(row["drift_ratio"]) and row["drift_ratio"] > DRIFT_RATIO_THRESH:
        score += 1
    if row.get("clipping_fraction", 0) is not None and np.isfinite(row["clipping_fraction"]) and row["clipping_fraction"] > CLIPPING_FRACTION_THRESH:
        score += 2
    if row.get("flatline_fraction", 0) is not None and np.isfinite(row["flatline_fraction"]) and row["flatline_fraction"] > FLATLINE_FRACTION_THRESH:
        score += 2
    if row.get("line_noise_ratio", 0) is not None and np.isfinite(row["line_noise_ratio"]) and row["line_noise_ratio"] > LINE_NOISE_RATIO_THRESH:
        score += 1
    rms = row.get("rms")
    if rms is not None and np.isfinite(rms) and rms < RMS_NEAR_ZERO_THRESH:
        score += 2

    n = row.get("n_samples", 0)
    if n is not None and n < MIN_SAMPLES:
        return int(score), "too_short"
    if score >= QC_BAD_SCORE:
        return int(score), "bad"
    if score >= QC_WARNING_SCORE:
        return int(score), "warning"
    return int(score), "ok"


# -----------------------------------------------------------------------------
# Per-task QC extraction
# -----------------------------------------------------------------------------


def extract_emg_qc_rows_for_task(
    patient_id: str,
    task_name: str,
    data_dir: Path,
    line_freq: float,
) -> List[Dict[str, Any]]:
    """Extract QC rows for all EMG channels in one task."""
    rec = load_emg_record(patient_id, task_name, data_dir)
    if rec is None:
        return []
    fs = get_fs_from_record(rec)
    channels = get_emg_channels(rec)
    rows = []
    for idx, ch in enumerate(channels):
        if ch not in rec.data:
            continue
        try:
            v = rec.data[ch]["values"]
            v = np.asarray(v, dtype=float)
        except (KeyError, TypeError):
            continue
        qc = compute_emg_channel_qc(v, fs, line_freq=line_freq)
        score, flag = compute_qc_score(qc)
        row = {
            "patient_id": patient_id,
            "task_name": task_name,
            "channel": ch,
            "channel_index": idx,
            **qc,
            "qc_score": score,
            "qc_flag": flag,
        }
        rows.append(row)
    return rows


# -----------------------------------------------------------------------------
# Main QC collection
# -----------------------------------------------------------------------------


def collect_all_qc(
    data_dir: Path,
    patients: Optional[Sequence[str]] = None,
    tasks: Optional[Sequence[str]] = None,
    max_patients: Optional[int] = None,
    max_tasks: Optional[int] = None,
    line_freq: float = 50.0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Collect QC rows for all patients and tasks."""
    all_rows: List[Dict[str, Any]] = []
    pat_list = discover_patients_with_data(data_dir)
    if patients:
        pat_list = [p for p in pat_list if p in patients]
    if max_patients:
        pat_list = pat_list[:max_patients]

    n_patients = len(pat_list)
    print(f"Discovering patients and tasks... {n_patients} patients")

    for i, patient_id in enumerate(pat_list):
        task_list = discover_tasks_for_patient(patient_id, data_dir)
        if tasks:
            task_list = [t for t in task_list if t in tasks]
        if max_tasks:
            task_list = task_list[:max_tasks]
        print(f"\n--- [{i + 1}/{n_patients}] {patient_id} ({len(task_list)} tasks) ---")
        pat_bad, pat_warn, pat_ok = 0, 0, 0
        for task_name in task_list:
            rows = extract_emg_qc_rows_for_task(patient_id, task_name, data_dir, line_freq)
            n_bad = sum(1 for r in rows if r.get("qc_flag") == "bad")
            n_warn = sum(1 for r in rows if r.get("qc_flag") == "warning")
            n_ok = sum(1 for r in rows if r.get("qc_flag") == "ok")
            pat_bad += n_bad
            pat_warn += n_warn
            pat_ok += n_ok
            all_rows.extend(rows)
            if rows:
                print(f"  {task_name}: {len(rows)} ch (bad={n_bad}, warn={n_warn}, ok={n_ok})")
            else:
                print(f"  {task_name}: (no EMG data)")
        if pat_bad + pat_warn + pat_ok > 0:
            print(f"  → {patient_id} total: {pat_bad + pat_warn + pat_ok} ch (bad={pat_bad}, warn={pat_warn}, ok={pat_ok})")

    return pd.DataFrame(all_rows)


# -----------------------------------------------------------------------------
# Summary tables
# -----------------------------------------------------------------------------


def build_patient_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate QC metrics by patient."""
    if df.empty:
        return pd.DataFrame()
    cols = ["motion_artifact_ratio", "offset_ratio", "drift_ratio", "clipping_fraction", "flatline_fraction", "line_noise_ratio", "rms", "qc_score"]
    agg = {"channel": "count"}
    for c in cols:
        if c in df.columns:
            agg[c] = "mean"
    grp = df.groupby("patient_id", as_index=False).agg(agg)
    grp = grp.rename(columns={"channel": "n_channels"})
    for flag in ["bad", "warning", "ok"]:
        cnt = df[df["qc_flag"] == flag].groupby("patient_id").size().reindex(grp["patient_id"]).fillna(0).astype(int)
        grp[f"n_{flag}"] = cnt.values
    grp["frac_bad"] = (grp["n_bad"] / grp["n_channels"]).fillna(0)
    grp["frac_warning"] = (grp["n_warning"] / grp["n_channels"]).fillna(0)
    return grp


def build_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate QC metrics by task."""
    if df.empty:
        return pd.DataFrame()
    cols = ["motion_artifact_ratio", "offset_ratio", "drift_ratio", "clipping_fraction", "flatline_fraction", "line_noise_ratio", "rms", "qc_score"]
    agg = {"channel": "count"}
    for c in cols:
        if c in df.columns:
            agg[c] = "mean"
    grp = df.groupby("task_name", as_index=False).agg(agg)
    grp = grp.rename(columns={"channel": "n_channels"})
    for flag in ["bad", "warning", "ok"]:
        cnt = df[df["qc_flag"] == flag].groupby("task_name").size().reindex(grp["task_name"]).fillna(0).astype(int)
        grp[f"n_{flag}"] = cnt.values
    grp["frac_bad"] = (grp["n_bad"] / grp["n_channels"]).fillna(0)
    grp["frac_warning"] = (grp["n_warning"] / grp["n_channels"]).fillna(0)
    return grp


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def _style_ax(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_metric_strip_by_patient(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Strip plot: x=patient, y=metric, points colored by qc_flag."""
    if df.empty or metric not in df.columns:
        return
    sub = df[[metric, "patient_id", "qc_flag"]].dropna(subset=[metric])
    if sub.empty:
        return
    patients = sub["patient_id"].unique()
    fig, ax = plt.subplots(figsize=(max(8, len(patients) * 0.4), 5))
    for flag in QC_FLAG_ORDER:
        mask = sub["qc_flag"] == flag
        if not mask.any():
            continue
        d = sub.loc[mask]
        x = [list(patients).index(p) for p in d["patient_id"]]
        jitter = np.random.uniform(-0.15, 0.15, len(x))
        ax.scatter(np.array(x) + jitter, d[metric], c=QC_FLAG_COLORS[flag], alpha=0.6, s=12, label=flag)
    ax.set_xticks(range(len(patients)))
    ax.set_xticklabels(patients, rotation=45, ha="right")
    ax.set_xlabel("Patient")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by patient (n={len(sub)})")
    ax.legend(loc="upper right", fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_qc_flag_stackedbars(
    df: pd.DataFrame,
    group_col: str,
    out_path_counts: Path,
    out_path_frac: Path,
    dpi: int = 150,
) -> None:
    """Stacked bar: counts and fractions of ok/warning/bad per group."""
    if df.empty or group_col not in df.columns:
        return
    grp = df.groupby([group_col, "qc_flag"]).size().unstack(fill_value=0)
    order = [c for c in QC_FLAG_ORDER if c in grp.columns]
    if not order:
        return
    grp = grp[order]
    if grp.empty:
        return
    fig, ax = plt.subplots(figsize=(max(8, grp.shape[0] * 0.5), 5))
    bottom = np.zeros(len(grp))
    for col in grp.columns:
        ax.bar(range(len(grp)), grp[col], bottom=bottom, color=QC_FLAG_COLORS.get(col, "#999"), label=col)
        bottom += grp[col].values
    ax.set_xticks(range(len(grp)))
    ax.set_xticklabels(grp.index, rotation=45, ha="right")
    ax.set_ylabel("Count")
    ax.set_title(f"QC flags by {group_col}")
    ax.legend(loc="upper right")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path_counts, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    frac = grp.div(grp.sum(axis=1), axis=0).fillna(0)
    fig, ax = plt.subplots(figsize=(max(8, frac.shape[0] * 0.5), 5))
    bottom = np.zeros(len(frac))
    for col in frac.columns:
        ax.bar(range(len(frac)), frac[col], bottom=bottom, color=QC_FLAG_COLORS.get(col, "#999"), label=col)
        bottom += frac[col].values
    ax.set_xticks(range(len(frac)))
    ax.set_xticklabels(frac.index, rotation=45, ha="right")
    ax.set_ylabel("Fraction")
    ax.set_title(f"QC flags by {group_col} (fraction)")
    ax.legend(loc="upper right")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path_frac, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metric_strip_by_channel(
    df: pd.DataFrame,
    metric: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Strip plot: x=channel, y=metric, points colored by qc_flag."""
    if df.empty or metric not in df.columns:
        return
    sub = df[[metric, "channel", "qc_flag"]].dropna(subset=[metric])
    if sub.empty:
        return
    channels = sub["channel"].unique()
    fig, ax = plt.subplots(figsize=(max(8, len(channels) * 0.45), 5))
    for flag in QC_FLAG_ORDER:
        mask = sub["qc_flag"] == flag
        if not mask.any():
            continue
        d = sub.loc[mask]
        ch_to_idx = {c: i for i, c in enumerate(channels)}
        x = [ch_to_idx[c] for c in d["channel"]]
        jitter = np.random.uniform(-0.15, 0.15, len(x))
        ax.scatter(np.array(x) + jitter, d[metric], c=QC_FLAG_COLORS.get(flag, "#999"), alpha=0.6, s=12, label=flag)
    ax.set_xticks(range(len(channels)))
    ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Channel")
    ax.set_ylabel(metric)
    ax.set_title(f"{metric} by channel (n={len(sub)})")
    ax.legend(loc="upper right", fontsize=8)
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metric_violin_with_points(
    df: pd.DataFrame,
    metrics: List[str],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Violin + jittered points by patient for multiple metrics."""
    if df.empty:
        return
    patients = df["patient_id"].unique()
    n_metrics = len([m for m in metrics if m in df.columns])
    if n_metrics == 0:
        return
    fig, axes = plt.subplots(n_metrics, 1, figsize=(max(8, len(patients) * 0.35), 3.5 * n_metrics), sharex=True, squeeze=False)
    ax_idx = 0
    for metric in metrics:
        if metric not in df.columns:
            continue
        ax = axes[ax_idx, 0]
        sub = df[[metric, "patient_id", "qc_flag"]].dropna(subset=[metric])
        if sub.empty:
            ax_idx += 1
            continue
        for i, p in enumerate(patients):
            d = sub[sub["patient_id"] == p]
            if d.empty:
                continue
            for flag in QC_FLAG_ORDER:
                m = d["qc_flag"] == flag
                if not m.any():
                    continue
                vals = d.loc[m, metric]
                jitter = np.random.uniform(-0.2, 0.2, len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals, c=QC_FLAG_COLORS[flag], alpha=0.5, s=10)
        ax.set_xticks(range(len(patients)))
        ax.set_xticklabels(patients, rotation=45, ha="right")
        ax.set_ylabel(metric)
        ax.set_title(metric)
        _style_ax(ax)
        ax_idx += 1
    axes[-1, 0].set_xlabel("Patient")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_metric_violin_by_channel(
    df: pd.DataFrame,
    metrics: List[str],
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Violin + jittered points by channel for multiple metrics."""
    if df.empty:
        return
    channels = df["channel"].unique()
    n_metrics = len([m for m in metrics if m in df.columns])
    if n_metrics == 0:
        return
    fig, axes = plt.subplots(n_metrics, 1, figsize=(max(10, len(channels) * 0.4), 3.5 * n_metrics), sharex=True, squeeze=False)
    ax_idx = 0
    for metric in metrics:
        if metric not in df.columns:
            continue
        ax = axes[ax_idx, 0]
        sub = df[[metric, "channel", "qc_flag"]].dropna(subset=[metric])
        if sub.empty:
            ax_idx += 1
            continue
        for i, ch in enumerate(channels):
            d = sub[sub["channel"] == ch]
            if d.empty:
                continue
            for flag in QC_FLAG_ORDER:
                m = d["qc_flag"] == flag
                if not m.any():
                    continue
                vals = d.loc[m, metric]
                jitter = np.random.uniform(-0.2, 0.2, len(vals))
                ax.scatter(np.full(len(vals), i) + jitter, vals, c=QC_FLAG_COLORS.get(flag, "#999"), alpha=0.5, s=10)
        ax.set_xticks(range(len(channels)))
        ax.set_xticklabels(channels, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(metric)
        ax.set_title(metric)
        _style_ax(ax)
        ax_idx += 1
    axes[-1, 0].set_xlabel("Channel")
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_patient_channel_heatmap(
    df: pd.DataFrame,
    patient_id: str,
    out_path: Path,
    dpi: int = 150,
) -> None:
    """Heatmap: rows=tasks, cols=channels, values=qc_score."""
    sub = df[df["patient_id"] == patient_id]
    if sub.empty:
        return
    pivot = sub.pivot_table(index="task_name", columns="channel", values="qc_score", aggfunc="first")
    pivot = pivot.fillna(-1)
    tasks = pivot.index.tolist()
    chans = pivot.columns.tolist()
    fig, ax = plt.subplots(figsize=(max(6, len(chans) * 0.5), max(4, len(tasks) * 0.4)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=-0.5, vmax=6)
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels(tasks, fontsize=8)
    ax.set_xticks(range(len(chans)))
    ax.set_xticklabels(chans, rotation=45, ha="right", fontsize=8)
    ax.set_title(f"{patient_id} — QC score (task × channel)")
    plt.colorbar(im, ax=ax, label="QC score")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_channel_cohort_heatmap(df: pd.DataFrame, out_path: Path, dpi: int = 150) -> None:
    """Cohort heatmap: rows=patients, cols=channels, values=mean qc_score per patient-channel."""
    if df.empty:
        return
    pivot = df.pivot_table(index="patient_id", columns="channel", values="qc_score", aggfunc="mean")
    pivot = pivot.fillna(-1)
    patients = pivot.index.tolist()
    chans = pivot.columns.tolist()
    fig, ax = plt.subplots(figsize=(max(8, len(chans) * 0.45), max(5, len(patients) * 0.35)))
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdYlGn_r", vmin=-0.5, vmax=6)
    ax.set_yticks(range(len(patients)))
    ax.set_yticklabels(patients, fontsize=8)
    ax.set_xticks(range(len(chans)))
    ax.set_xticklabels(chans, rotation=45, ha="right", fontsize=8)
    ax.set_title("QC score by patient × channel (mean across tasks)")
    plt.colorbar(im, ax=ax, label="QC score")
    _style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_summary_dashboard(df: pd.DataFrame, out_path: Path, dpi: int = 150) -> None:
    """Multi-panel cohort summary."""
    if df.empty:
        return
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

    ax1 = fig.add_subplot(gs[0, 0])
    if "qc_score" in df.columns:
        ax1.hist(df["qc_score"].dropna(), bins=range(0, 10), color="#1565c0", alpha=0.7, edgecolor="white")
    ax1.set_xlabel("QC score")
    ax1.set_ylabel("Count")
    ax1.set_title("QC score distribution")
    _style_ax(ax1)

    ax2 = fig.add_subplot(gs[0, 1])
    if "qc_flag" in df.columns:
        flag_counts = df["qc_flag"].value_counts()
        colors = [QC_FLAG_COLORS.get(f, "#999") for f in flag_counts.index]
        ax2.bar(flag_counts.index, flag_counts.values, color=colors)
    ax2.set_xlabel("QC flag")
    ax2.set_ylabel("Count")
    ax2.set_title("QC flag counts")
    _style_ax(ax2)

    ax3 = fig.add_subplot(gs[1, 0])
    if "patient_id" in df.columns and "qc_flag" in df.columns:
        pt = df.groupby("patient_id").apply(lambda g: (g["qc_flag"] == "bad").mean() * 100)
        pt = pt.sort_values(ascending=False)
        ax3.barh(range(len(pt)), pt.values, color="#c62828", alpha=0.7)
        ax3.set_yticks(range(len(pt)))
        ax3.set_yticklabels(pt.index, fontsize=8)
    ax3.set_xlabel("Bad %")
    ax3.set_title("Bad channel fraction per patient")
    _style_ax(ax3)

    ax4 = fig.add_subplot(gs[1, 1])
    if "task_name" in df.columns and "qc_flag" in df.columns:
        tk = df.groupby("task_name").apply(lambda g: (g["qc_flag"] == "bad").mean() * 100)
        tk = tk.sort_values(ascending=False)
        n_show = min(15, len(tk))
        ax4.barh(range(n_show), tk.values[:n_show], color="#c62828", alpha=0.7)
        ax4.set_yticks(range(n_show))
        ax4.set_yticklabels(tk.index[:n_show], fontsize=8)
    ax4.set_xlabel("Bad %")
    ax4.set_title("Bad channel fraction per task (top 15)")
    _style_ax(ax4)

    fig.suptitle("EMG channel QC summary", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


# -----------------------------------------------------------------------------
# Optional: worst channels table
# -----------------------------------------------------------------------------


def save_worst_channels(df: pd.DataFrame, out_path: Path, top_n: int = 50) -> None:
    """Save top-N worst channels sorted by qc_score, clipping, flatline."""
    if df.empty:
        return
    sub = df.sort_values(
        by=["qc_score", "clipping_fraction", "flatline_fraction"],
        ascending=[False, False, False],
    ).head(top_n)
    sub.to_csv(out_path, index=False)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(description="EMG channel QC report: scan raw channels, compute metrics, generate figures")
    ap.add_argument("--data-dir", type=Path, default=Path("data/emg_structured"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/emg_qc"), help="Tables output directory")
    ap.add_argument("--figures-dir", type=Path, default=None, help="Figures directory (default: results/figures/emg_qc)")
    ap.add_argument("--patients", type=str, nargs="*", default=None)
    ap.add_argument("--tasks", type=str, nargs="*", default=None)
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--line-freq", type=float, default=50.0)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = args.figures_dir if args.figures_dir is not None else Path("results/figures/emg_qc")
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = collect_all_qc(
        data_dir,
        patients=args.patients,
        tasks=args.tasks,
        max_patients=args.max_patients,
        max_tasks=args.max_tasks,
        line_freq=args.line_freq,
        verbose=args.verbose,
    )

    if df.empty:
        print("No EMG channels found. Check data-dir and filters.")
        return

    # Save tables
    print("\nSaving tables...")
    df.to_csv(out_dir / "channel_qc_all.csv", index=False)
    patient_summary = build_patient_summary(df)
    task_summary = build_task_summary(df)
    if not patient_summary.empty:
        patient_summary.to_csv(out_dir / "patient_qc_summary.csv", index=False)
    if not task_summary.empty:
        task_summary.to_csv(out_dir / "task_qc_summary.csv", index=False)
    save_worst_channels(df, out_dir / "worst_channels_top50.csv")

    # Figures: metric strip by patient
    print("\nGenerating cohort figures...")
    metrics_for_strip = [
        "motion_artifact_ratio", "offset_ratio", "drift_ratio",
        "clipping_fraction", "flatline_fraction", "line_noise_ratio",
        "rms", "qc_score",
    ]
    for metric in metrics_for_strip:
        if metric in df.columns:
            plot_metric_strip_by_patient(
                df, metric,
                fig_dir / f"metric_by_patient_{metric}.png",
                dpi=args.dpi,
            )

    # Combined violin/jitter by patient
    plot_metric_violin_with_points(
        df, metrics_for_strip[:6],
        fig_dir / "qc_metrics_by_patient_combined.png",
        dpi=args.dpi,
    )

    # Figures grouped by channel
    for metric in metrics_for_strip:
        if metric in df.columns:
            plot_metric_strip_by_channel(
                df, metric,
                fig_dir / f"metric_by_channel_{metric}.png",
                dpi=args.dpi,
            )
    plot_metric_violin_by_channel(
        df, metrics_for_strip[:6],
        fig_dir / "qc_metrics_by_channel_combined.png",
        dpi=args.dpi,
    )
    plot_qc_flag_stackedbars(
        df, "channel",
        fig_dir / "qc_flags_by_channel_counts.png",
        fig_dir / "qc_flags_by_channel_fraction.png",
        dpi=args.dpi,
    )
    plot_channel_cohort_heatmap(df, fig_dir / "qc_heatmap_patient_x_channel.png", dpi=args.dpi)

    # Stacked bar: flags by patient
    plot_qc_flag_stackedbars(
        df, "patient_id",
        fig_dir / "qc_flags_by_patient_counts.png",
        fig_dir / "qc_flags_by_patient_fraction.png",
        dpi=args.dpi,
    )

    # Stacked bar: flags by task
    plot_qc_flag_stackedbars(
        df, "task_name",
        fig_dir / "qc_flags_by_task_counts.png",
        fig_dir / "qc_flags_by_task_fraction.png",
        dpi=args.dpi,
    )

    # Per-patient heatmaps
    print("\nGenerating per-patient heatmaps...")
    for pid in df["patient_id"].unique():
        plot_patient_channel_heatmap(df, pid, fig_dir / f"patient_{pid}_qc_heatmap.png", dpi=args.dpi)
        print(f"  {pid}: patient_{pid}_qc_heatmap.png")

    # Dashboard
    plot_summary_dashboard(df, fig_dir / "qc_summary_dashboard.png", dpi=args.dpi)

    n_bad = (df["qc_flag"] == "bad").sum()
    n_warn = (df["qc_flag"] == "warning").sum()
    n_ok = (df["qc_flag"] == "ok").sum()
    n_total = len(df)
    n_patients = df["patient_id"].nunique()
    n_tasks = df["task_name"].nunique()

    print("\n" + "=" * 50)
    print("EMG Channel QC Report — Summary")
    print("=" * 50)
    print(f"Patients:        {n_patients}")
    print(f"Tasks:           {n_tasks}")
    print(f"Channels:        {n_total}")
    print(f"Bad:             {n_bad} ({100*n_bad/n_total:.1f}%)")
    print(f"Warning:         {n_warn} ({100*n_warn/n_total:.1f}%)")
    print(f"OK:              {n_ok} ({100*n_ok/n_total:.1f}%)")
    print(f"Tables:          {out_dir}")
    print(f"Figures:         {fig_dir}")
    print("=" * 50)


if __name__ == "__main__":
    main()
