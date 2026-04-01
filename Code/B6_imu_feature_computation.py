#!/usr/bin/env python3
"""B6_imu_feature_computation.py

IMU feature computation for non-calibrated sensors.
Uses manual segment indexes to extract movement cycles per sensor-axis.
Focus: timing, smoothness, complexity, temporal regularity.
NO amplitude-dependent features (range, std, vector magnitude, etc.).

Features are computed in TRUE TIME from raw segments.
Phase-normalized profiles are used only for visualization (shape comparison).

Usage:
  python B6_imu_feature_computation.py --data-dir data/emg_structured
  python B6_imu_feature_computation.py --data-dir data/emg_structured --out-dir results/imu_features
  python B6_imu_feature_computation.py --min-cycles 3 --lowpass-hz 10 --cycle-normalization zscore
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import pickle
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.signal import butter, filtfilt, welch

# Pickle compatibility for EMGRecord
try:
    import B0_parse_emg_patient_task as _parse_mod
    sys.modules["__main__"].ChannelInfo = _parse_mod.ChannelInfo
    sys.modules["__main__"].EMGRecord = _parse_mod.EMGRecord
    from B0_parse_emg_patient_task import EMGRecord
except ImportError:
    EMGRecord = Any

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# Constants (centralized)
# -----------------------------------------------------------------------------

# Non-amplitude-dependent IMU features (interpretable without calibration)
IMU_FEATURES = [
    "cycle_duration_s",
    "dominant_frequency_hz",
    "spectral_entropy",
    "normalized_rms_derivative",
    "lag1_autocorr",
    "permutation_entropy",
]

FEATURE_LABELS = {
    "cycle_duration_s": "Cycle duration (s)",
    "dominant_frequency_hz": "Dominant frequency (Hz)",
    "spectral_entropy": "Spectral entropy",
    "normalized_rms_derivative": "Normalized RMS derivative",
    "lag1_autocorr": "Lag-1 autocorrelation",
    "permutation_entropy": "Permutation entropy",
}

N_PHASE_POINTS = 101
DEFAULT_LOWPASS_HZ = 10.0
DEFAULT_LOWPASS_ORDER = 4
DEFAULT_MIN_CYCLES = 3
DEFAULT_CONDITIONS = ["SN", "DS"]
EPS = 1e-12
# Permutation entropy: embedding dimension (3 is stable for short segments)
PE_ORDER = 3
PE_DELAY = 1

# -----------------------------------------------------------------------------
# Pickle loading (numpy 1.x / 2.x compatibility)
# -----------------------------------------------------------------------------


def _compat_unpickle(path: Path) -> Any:
    """Load pickle with numpy 1.x/2.x compatibility."""
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except ModuleNotFoundError as e:
        if "numpy._core" in str(e):
            if "numpy._core" not in sys.modules:
                sys.modules["numpy._core"] = type(sys)("numpy._core")
            if "numpy._core.numeric" not in sys.modules:
                try:
                    sys.modules["numpy._core.numeric"] = np.core.numeric
                except AttributeError:
                    sys.modules["numpy._core.numeric"] = np
            with open(path, "rb") as f:
                return pickle.load(f)
        raise


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_imu_record(patient_id: str, task_name: str, data_dir: Path) -> Optional[EMGRecord]:
    pkl_path = data_dir / f"{patient_id}_EMG" / f"{task_name}_imu.pkl"
    if not pkl_path.exists():
        logger.warning("IMU file not found: %s", pkl_path)
        return None
    try:
        return _compat_unpickle(pkl_path)
    except Exception as e:
        logger.warning("Failed to load %s: %s", pkl_path, e)
        return None


def load_manual_segments(
    patient_id: str,
    task_name: str,
    data_dir: Path,
) -> Optional[List[Tuple[float, float]]]:
    path = data_dir / f"{patient_id}_EMG" / f"{task_name}_manual_segments.json"
    if not path.exists():
        logger.info("  [%s/%s] Manual segments file not found: %s", patient_id, task_name, path)
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        pairs = [(p["start_s"], p["end_s"]) for p in data.get("pairs", [])]
        return pairs if pairs else None
    except Exception as e:
        logger.warning("Failed to load segments %s: %s", path, e)
        return None


def get_fs_from_record(rec: EMGRecord) -> float:
    if not rec.channels:
        return 150.0
    return float(rec.channels[0].sampling_freq)


def discover_patients(data_dir: Path) -> List[str]:
    if not data_dir.exists():
        logger.warning("Data dir does not exist: %s", data_dir)
        return []
    return sorted(
        pdir.name.replace("_EMG", "")
        for pdir in data_dir.iterdir()
        if pdir.is_dir() and pdir.name.endswith("_EMG")
    )


def discover_tasks_with_segments(patient_id: str, data_dir: Path) -> List[str]:
    pdir = data_dir / f"{patient_id}_EMG"
    if not pdir.exists():
        return []
    tasks = []
    for seg in sorted(pdir.glob("*_manual_segments.json")):
        task_name = seg.stem.replace("_manual_segments", "")
        if "Calibrazione" in task_name:
            continue
        imu_path = pdir / f"{task_name}_imu.pkl"
        if imu_path.exists():
            tasks.append(task_name)
    return tasks


def parse_session_condition(task_name: str) -> Tuple[str, str]:
    task_upper = task_name.upper()
    session = "T0" if "T0" in task_upper else "T1" if "T1" in task_upper else ""
    condition = "SN" if "SN" in task_upper else "DS" if "DS" in task_upper else ""
    return session, condition


# -----------------------------------------------------------------------------
# IMU sensor grouping
# -----------------------------------------------------------------------------


def group_acc_channels_by_sensor(acc_channels: List[str]) -> Dict[str, Tuple[str, str, str]]:
    """Group Acc channels by sensor. Returns {sensor_key: (ch_x, ch_y, ch_z)}."""
    groups: Dict[str, Dict[str, str]] = {}
    for ch in acc_channels:
        m = re.search(r"^(.+?):\s*Acc[\.\s]*(\d+)\s*\.([XYZ])$", ch, re.IGNORECASE)
        if not m:
            m = re.search(r"^(.+?):\s*Acc\s*\.([XYZ])\s*(\d+)$", ch, re.IGNORECASE)
        if not m:
            continue
        if len(m.groups()) == 3:
            base = f"{m.group(1).strip()}: Acc {m.group(2)}"
            axis = m.group(3).upper()
        else:
            base = f"{m.group(1).strip()}: Acc {m.group(3)}"
            axis = m.group(2).upper()
        if base not in groups:
            groups[base] = {}
        groups[base][axis] = ch
    out = {}
    for base, axes in groups.items():
        if set(axes.keys()) >= {"X", "Y", "Z"}:
            out[base] = (axes["X"], axes["Y"], axes["Z"])
    return out


# -----------------------------------------------------------------------------
# Preprocessing (smoothing only; NOT calibration)
# -----------------------------------------------------------------------------


def filter_lowpass(
    data: np.ndarray,
    fs: float,
    cutoff_hz: float = DEFAULT_LOWPASS_HZ,
    order: int = DEFAULT_LOWPASS_ORDER,
) -> np.ndarray:
    """Low-pass filter for smoothing. Does NOT calibrate or change interpretation."""
    if len(data) < 10 or cutoff_hz <= 0:
        return np.asarray(data, dtype=np.float64)
    nyq = fs / 2
    cutoff = min(cutoff_hz, nyq * 0.99)
    if cutoff < 0.5:
        cutoff = 0.5
    b, a = butter(order, cutoff, btype="low", fs=fs)
    return filtfilt(b, a, np.asarray(data, dtype=np.float64))


def normalize_segment(v: np.ndarray, mode: str = "zscore") -> np.ndarray:
    """Normalize segment for amplitude-independent features. Returns demeaned or z-scored signal."""
    v = np.asarray(v, dtype=np.float64)
    v = v - np.nanmean(v)  # demean always
    if mode == "zscore":
        sd = np.nanstd(v, ddof=1)
        if sd > EPS:
            v = v / sd
    return v


# -----------------------------------------------------------------------------
# Extraction: TRUE-TIME raw segments vs phase-normalized (visualization only)
# -----------------------------------------------------------------------------


def extract_raw_segments(
    t: np.ndarray,
    v: np.ndarray,
    segments: List[Tuple[float, float]],
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Extract segments in TRUE TIME.
    Returns list of (t_seg, v_seg) per cycle.
    Use for feature computation (duration, frequency, entropy, derivative).
    """
    result = []
    for start_s, end_s in segments:
        mask = (t >= start_s) & (t <= end_s)
        if not np.any(mask):
            continue
        t_seg = np.asarray(t[mask], dtype=np.float64)
        v_seg = np.asarray(v[mask], dtype=np.float64)
        if len(t_seg) < 2 or t_seg[-1] <= t_seg[0]:
            continue
        result.append((t_seg, v_seg))
    return result


def extract_phase_normalized_cycles(
    t: np.ndarray,
    v: np.ndarray,
    segments: List[Tuple[float, float]],
    n_pts: int = N_PHASE_POINTS,
) -> np.ndarray:
    """
    Extract segments, interpolate to 0-100% phase.
    FOR VISUALIZATION / SHAPE COMPARISON ONLY.
    Do NOT use for derivative, frequency, or entropy in physical time.
    """
    phase_new = np.linspace(0.0, 1.0, n_pts, dtype=np.float64)
    stacked = []
    for start_s, end_s in segments:
        mask = (t >= start_s) & (t <= end_s)
        if not np.any(mask):
            continue
        t_seg = t[mask]
        v_seg = v[mask]
        if len(t_seg) < 2 or t_seg[-1] <= t_seg[0]:
            continue
        phase_old = (t_seg - t_seg[0]) / (t_seg[-1] - t_seg[0])
        v_res = np.interp(phase_new, phase_old, v_seg)
        stacked.append(v_res)
    return np.asarray(stacked) if stacked else np.array([]).reshape(0, n_pts)


# -----------------------------------------------------------------------------
# TRUE-TIME features (no amplitude dependence)
# -----------------------------------------------------------------------------


def feat_cycle_duration_s(t_seg: np.ndarray) -> float:
    """Cycle duration in seconds. Direct time metric."""
    if t_seg.size < 2:
        return np.nan
    return float(t_seg[-1] - t_seg[0])


def feat_dominant_frequency_hz(v: np.ndarray, fs: float) -> float:
    """Dominant frequency from Welch PSD. Amplitude-independent (peak location)."""
    if v.size < 16:
        return np.nan
    v = v - np.nanmean(v)  # demean
    nperseg = min(256, len(v))
    nperseg = max(nperseg, 16)
    f, psd = welch(v, fs=fs, nperseg=nperseg)
    idx = np.argmax(psd)
    return float(f[idx])


def feat_spectral_entropy(v: np.ndarray, fs: float) -> float:
    """Spectral entropy on demeaned signal. Normalized 0-1."""
    if v.size < 16:
        return np.nan
    v = np.asarray(v, dtype=np.float64) - np.nanmean(v)
    if np.nanstd(v) < EPS:
        return 0.0
    nperseg = min(256, len(v))
    nperseg = max(nperseg, 16)
    f, psd = welch(v, fs=fs, nperseg=nperseg)
    psd = np.maximum(psd, EPS)
    psd_norm = psd / np.sum(psd)
    psd_norm = psd_norm[psd_norm > 0]
    n_bins = len(psd_norm)
    if n_bins < 2:
        return np.nan
    h = -np.sum(psd_norm * np.log2(psd_norm + EPS))
    h_max = np.log2(n_bins)
    return float(h / h_max) if h_max > 0 else np.nan


def feat_normalized_rms_derivative(v: np.ndarray) -> float:
    """RMS of derivative of normalized (demeaned + unit-variance) signal. Reflects roughness/smoothness."""
    if v.size < 3:
        return np.nan
    v = np.asarray(v, dtype=np.float64)
    v = v - np.nanmean(v)
    sd = np.nanstd(v, ddof=1)
    if sd < EPS:
        return 0.0
    v = v / sd
    diff = np.diff(v)
    return float(np.sqrt(np.nanmean(diff ** 2) + EPS))


def feat_lag1_autocorr(v: np.ndarray) -> float:
    """Lag-1 autocorrelation of normalized cycle. Shape/smoothness regularity."""
    if v.size < 3:
        return np.nan
    v = np.asarray(v, dtype=np.float64)
    v = v - np.nanmean(v)
    sd = np.nanstd(v, ddof=1)
    if sd < EPS:
        return np.nan
    v = v / sd
    c0 = np.dot(v[:-1], v[:-1]) / (len(v) - 1)
    c1 = np.dot(v[:-1], v[1:]) / (len(v) - 1)
    if c0 < EPS:
        return np.nan
    return float(c1 / c0)


def permutation_entropy(v: np.ndarray, order: int = PE_ORDER, delay: int = PE_DELAY) -> float:
    """Permutation entropy (normalized). Stable for short segments."""
    n = len(v)
    if n < order + delay * (order - 1) + 1:
        return np.nan
    v = np.asarray(v, dtype=np.float64)
    v = v - np.nanmean(v)
    sd = np.nanstd(v, ddof=1)
    if sd > EPS:
        v = v / sd
    # Build ordinal patterns
    n_patterns = 1
    for i in range(1, order + 1):
        n_patterns *= i
    count = np.zeros(n_patterns, dtype=np.int64)
    indices = np.arange(order, dtype=np.int64)
    for i in range(n - delay * (order - 1) - 1):
        # Extract embedded vector
        emb = v[i: i + delay * order: delay]
        # Ordinal pattern: argsort gives permutation rank
        perm = np.argsort(emb)
        # Map permutation to index (Lehmer code)
        rank = 0
        used = np.zeros(order, dtype=bool)
        for j, p in enumerate(perm):
            k = np.sum(~used[:p])
            rank += k * math.factorial(order - 1 - j)
            used[p] = True
        count[rank] += 1
    p = count / np.sum(count)
    p = p[p > 0]
    h = -np.sum(p * np.log2(p))
    h_max = np.log2(n_patterns)
    return float(h / h_max) if h_max > 0 else np.nan


def compute_cycle_features(
    t_seg: np.ndarray,
    v_seg: np.ndarray,
    fs: float,
    norm_mode: str = "zscore",
) -> Dict[str, float]:
    """Compute all TRUE-TIME features for one raw segment."""
    v_norm = normalize_segment(v_seg, mode=norm_mode)
    return {
        "cycle_duration_s": feat_cycle_duration_s(t_seg),
        "dominant_frequency_hz": feat_dominant_frequency_hz(v_norm, fs),
        "spectral_entropy": feat_spectral_entropy(v_norm, fs),
        "normalized_rms_derivative": feat_normalized_rms_derivative(v_norm),
        "lag1_autocorr": feat_lag1_autocorr(v_norm),
        "permutation_entropy": permutation_entropy(v_norm, order=PE_ORDER, delay=PE_DELAY),
    }


def aggregate_cycle_features(
    raw_segments: List[Tuple[np.ndarray, np.ndarray]],
    fs: float,
    min_cycles: int,
    norm_mode: str = "zscore",
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """
    Mean of features across cycles.
    Returns (feats_dict, qc_info).
    If n_valid < min_cycles, feats are NaN and qc_info has reason.
    """
    qc_info: Dict[str, Any] = {
        "n_segments": len(raw_segments),
        "n_valid": 0,
        "n_discarded": 0,
        "median_duration_s": np.nan,
        "min_duration_s": np.nan,
        "max_duration_s": np.nan,
    }
    if not raw_segments:
        qc_info["exclusion_reason"] = "no_segments"
        return {k: np.nan for k in IMU_FEATURES}, qc_info
    feats_list = []
    durations = []
    for t_seg, v_seg in raw_segments:
        f = compute_cycle_features(t_seg, v_seg, fs, norm_mode)
        feats_list.append(f)
        if np.isfinite(f["cycle_duration_s"]):
            durations.append(f["cycle_duration_s"])
    n_valid = len(feats_list)
    n_discarded = len(raw_segments) - n_valid
    qc_info["n_valid"] = n_valid
    qc_info["n_discarded"] = n_discarded
    if durations:
        qc_info["median_duration_s"] = float(np.median(durations))
        qc_info["min_duration_s"] = float(np.min(durations))
        qc_info["max_duration_s"] = float(np.max(durations))
    if n_valid < min_cycles:
        qc_info["exclusion_reason"] = f"too_few_cycles (n={n_valid} < min={min_cycles})"
        return {k: np.nan for k in IMU_FEATURES}, qc_info
    out = {}
    for k in IMU_FEATURES:
        vals = [f[k] for f in feats_list if np.isfinite(f[k])]
        out[k] = float(np.mean(vals)) if vals else np.nan
    return out, qc_info


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------


def process_patient_task(
    patient_id: str,
    task_name: str,
    data_dir: Path,
    lowpass_hz: float,
    min_cycles: int,
    norm_mode: str,
) -> Tuple[Optional[pd.DataFrame], Optional[pd.DataFrame]]:
    """
    Process one patient/task.
    Returns (task_df, qc_df).
    task_df: patient_id, task_name, session, condition, sensor, axis, <features>
    qc_df: patient_id, task_name, session, condition, sensor, axis, n_segments, n_valid, ...
    """
    logger.info("  [%s/%s] Loading IMU record...", patient_id, task_name)
    rec = load_imu_record(patient_id, task_name, data_dir)
    if rec is None:
        logger.info("  [%s/%s] SKIP: IMU record not loaded (file missing or parse error)", patient_id, task_name)
        return None, None
    segments = load_manual_segments(patient_id, task_name, data_dir)
    if not segments:
        logger.info("  [%s/%s] SKIP: no manual segments", patient_id, task_name)
        return None, None
    acc_chans = rec.get_acc_channels()
    if not acc_chans:
        logger.info("  [%s/%s] SKIP: no accelerometer channels in record", patient_id, task_name)
        return None, None
    sensor_groups = group_acc_channels_by_sensor(acc_chans)
    if not sensor_groups:
        logger.info("  [%s/%s] SKIP: could not parse sensor-axis groups (need X,Y,Z per sensor)", patient_id, task_name)
        return None, None
    fs = get_fs_from_record(rec)
    session, condition = parse_session_condition(task_name)
    if not session or not condition:
        logger.info("  [%s/%s] SKIP: invalid session/condition (task name must contain T0/T1 and SN/DS)", patient_id, task_name)
        return None, None
    rows = []
    qc_rows = []
    for sensor_label, (ch_x, ch_y, ch_z) in sensor_groups.items():
        for axis, ch in [("X", ch_x), ("Y", ch_y), ("Z", ch_z)]:
            t = np.asarray(rec.data[ch]["times"], dtype=np.float64)
            v = np.asarray(rec.data[ch]["values"], dtype=np.float64)
            v_filt = filter_lowpass(v, fs, lowpass_hz)
            raw_segs = extract_raw_segments(t[: len(v_filt)], v_filt, segments)
            feats, qc_info = aggregate_cycle_features(raw_segs, fs, min_cycles, norm_mode)
            if qc_info.get("exclusion_reason"):
                logger.debug("      [%s/%s] %s %s: %s (n_seg=%d, n_valid=%d)",
                            patient_id, task_name, sensor_label, axis,
                            qc_info["exclusion_reason"], qc_info["n_segments"], qc_info["n_valid"])
            qc_row = {
                "patient_id": patient_id,
                "task_name": task_name,
                "session": session,
                "condition": condition,
                "sensor": sensor_label,
                "axis": axis,
                "n_segments": qc_info["n_segments"],
                "n_valid": qc_info["n_valid"],
                "n_discarded": qc_info["n_discarded"],
                "median_duration_s": qc_info.get("median_duration_s", np.nan),
                "min_duration_s": qc_info.get("min_duration_s", np.nan),
                "max_duration_s": qc_info.get("max_duration_s", np.nan),
                "fs": fs,
                "pairing_available": qc_info["n_valid"] >= min_cycles,
                "exclusion_reason": qc_info.get("exclusion_reason", ""),
            }
            qc_rows.append(qc_row)
            row = {
                "patient_id": patient_id,
                "task_name": task_name,
                "session": session,
                "condition": condition,
                "sensor": sensor_label,
                "axis": axis,
                **feats,
            }
            rows.append(row)
    if not rows:
        n_ok = sum(1 for r in qc_rows if r.get("pairing_available"))
        n_fail = len(qc_rows) - n_ok
        logger.info("  [%s/%s] SKIP: no valid features (%d sensor-axes had too_few_cycles, %d had valid cycles but excluded)", patient_id, task_name, n_fail, n_ok)
        return None, None
    n_valid = sum(1 for r in qc_rows if r.get("pairing_available"))
    logger.info("  [%s/%s] OK: %d sensor-axes with valid features (session=%s, condition=%s)", patient_id, task_name, n_valid, session, condition)
    return pd.DataFrame(rows), pd.DataFrame(qc_rows)


def _fix_qc_typo(qc_df: pd.DataFrame) -> pd.DataFrame:
    """Fix ' pairing_available' -> 'pairing_available' typo."""
    if " pairing_available" in qc_df.columns:
        qc_df = qc_df.rename(columns={" pairing_available": "pairing_available"})
    return qc_df


# -----------------------------------------------------------------------------
# Pairing (includes task_name for safety)
# -----------------------------------------------------------------------------

PAIRING_KEY = ["patient_id", "task_name", "condition", "sensor", "axis"]


def build_paired_df_per_task(task_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build paired T0/T1 dataframe PER TASK.
    Key: patient_id, task_name, condition, sensor, axis.
    No cross-task merging.
    """
    if task_df.empty:
        return pd.DataFrame()
    index_cols = PAIRING_KEY
    t0 = task_df[task_df["session"] == "T0"][index_cols + IMU_FEATURES].copy()
    t1 = task_df[task_df["session"] == "T1"][index_cols + IMU_FEATURES].copy()
    t0 = t0.rename(columns={c: f"{c}_T0" for c in IMU_FEATURES})
    t1 = t1.rename(columns={c: f"{c}_T1" for c in IMU_FEATURES})
    merged = t0.merge(t1, on=index_cols, how="inner")
    for c in IMU_FEATURES:
        merged[f"{c}_delta"] = merged[f"{c}_T1"] - merged[f"{c}_T0"]
    return merged


def build_paired_df_aggregated(paired_per_task: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate paired data to patient-condition level.
    Mean across sensor-axis within each patient-condition.
    SECONDARY analysis layer.
    """
    if paired_per_task.empty:
        return pd.DataFrame()
    agg_cols = ["patient_id", "condition"]
    agg_dict = {}
    for c in IMU_FEATURES:
        t0_col, t1_col, d_col = f"{c}_T0", f"{c}_T1", f"{c}_delta"
        if t0_col in paired_per_task.columns:
            agg_dict[t0_col] = "mean"
        if t1_col in paired_per_task.columns:
            agg_dict[t1_col] = "mean"
        if d_col in paired_per_task.columns:
            agg_dict[d_col] = "mean"
    return paired_per_task.groupby(agg_cols, as_index=False).agg(agg_dict)


# -----------------------------------------------------------------------------
# Statistics (rich paired stats, compatible with pipeline)
# -----------------------------------------------------------------------------


def paired_ttest_full(v0: np.ndarray, v1: np.ndarray) -> Dict[str, float]:
    """Full paired t-test: n, means, sds, medians, change, CI, t, df, p, dz."""
    mask = np.isfinite(v0) & np.isfinite(v1)
    v0 = np.asarray(v0, dtype=float)[mask]
    v1 = np.asarray(v1, dtype=float)[mask]
    diff = v1 - v0
    n = len(v0)
    out = {
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
        "effect_size_dz": np.nan,
    }
    if n < 2:
        return out
    out["mean_T0"] = float(np.mean(v0))
    out["sd_T0"] = float(np.std(v0, ddof=1))
    out["median_T0"] = float(np.median(v0))
    out["mean_T1"] = float(np.mean(v1))
    out["sd_T1"] = float(np.std(v1, ddof=1))
    out["median_T1"] = float(np.median(v1))
    out["mean_change"] = float(np.mean(diff))
    out["sd_change"] = float(np.std(diff, ddof=1))
    if n >= 2 and out["sd_change"] > EPS:
        t_stat, p_val = scipy_stats.ttest_rel(v0, v1)
        out["t_stat"] = float(t_stat)
        out["df"] = float(n - 1)
        out["p_value"] = float(p_val)
        out["effect_size_dz"] = float(out["mean_change"] / out["sd_change"])
        se = out["sd_change"] / np.sqrt(n)
        out["ci95_low"] = float(out["mean_change"] - 1.96 * se)
        out["ci95_high"] = float(out["mean_change"] + 1.96 * se)
    return out


def apply_fdr(stats_df: pd.DataFrame, group_cols: List[str], p_col: str = "p_value") -> pd.DataFrame:
    if stats_df.empty or p_col not in stats_df.columns:
        return stats_df
    out_parts = []
    for _, group in stats_df.groupby(group_cols, dropna=False):
        group = group.copy()
        p = group[p_col].to_numpy(dtype=float)
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
        group["p_value_fdr"] = q
        out_parts.append(group)
    return pd.concat(out_parts, ignore_index=True)


def compute_paired_stats(
    paired_df: pd.DataFrame,
    group_cols: List[str],
    feature_cols: List[str],
) -> pd.DataFrame:
    """Compute rich paired stats per feature per group. Uses complete-case n per feature."""
    rows = []
    for feat in feature_cols:
        t0_col, t1_col = f"{feat}_T0", f"{feat}_T1"
        if t0_col not in paired_df.columns or t1_col not in paired_df.columns:
            continue
        for keys, subset in paired_df.groupby(group_cols, dropna=False):
            if isinstance(keys, tuple):
                key_vals = dict(zip(group_cols, keys))
            else:
                key_vals = {group_cols[0]: keys}
            v0 = subset[t0_col].to_numpy(dtype=float)
            v1 = subset[t1_col].to_numpy(dtype=float)
            res = paired_ttest_full(v0, v1)
            rows.append({
                "feature": feat,
                **key_vals,
                **res,
            })
    if not rows:
        return pd.DataFrame()
    stats_df = pd.DataFrame(rows)
    return apply_fdr(stats_df, group_cols=["feature"] + group_cols[:-1] if len(group_cols) > 1 else ["feature"])


def compute_sensor_axis_stats(paired_per_task: pd.DataFrame) -> pd.DataFrame:
    """Per sensor/axis stats (for heatmaps). Primary sensor-axis level."""
    return compute_paired_stats(
        paired_per_task,
        group_cols=["condition", "sensor", "axis"],
        feature_cols=IMU_FEATURES,
    )


# -----------------------------------------------------------------------------
# Storage for cycle profiles (visualization)
# -----------------------------------------------------------------------------


def save_cycle_profiles(
    task_df: pd.DataFrame,
    rec: EMGRecord,
    patient_id: str,
    task_name: str,
    data_dir: Path,
    segments: List[Tuple[float, float]],
    sensor_groups: Dict[str, Tuple[str, str, str]],
    fs: float,
    lowpass_hz: float,
    n_pts: int,
    out_path: Path,
) -> None:
    """
    Save normalized cycle profiles (T0 vs T1 mean) per sensor-axis-condition.
    FOR VISUALIZATION / SHAPE COMPARISON ONLY.
    """
    session, condition = parse_session_condition(task_name)
    if not session:
        return
    profiles = []
    for sensor_label, (ch_x, ch_y, ch_z) in sensor_groups.items():
        for axis, ch in [("X", ch_x), ("Y", ch_y), ("Z", ch_z)]:
            t = np.asarray(rec.data[ch]["times"], dtype=np.float64)
            v = np.asarray(rec.data[ch]["values"], dtype=np.float64)
            v_filt = filter_lowpass(v, fs, lowpass_hz)
            cycles = extract_phase_normalized_cycles(t[: len(v_filt)], v_filt, segments, n_pts)
            if cycles.size == 0:
                continue
            # Normalize each cycle (demean, optional zscore) for shape
            cycles_norm = np.array([normalize_segment(c, "zscore") for c in cycles])
            mean_profile = np.mean(cycles_norm, axis=0)
            profiles.append({
                "patient_id": patient_id,
                "task_name": task_name,
                "session": session,
                "condition": condition,
                "sensor": sensor_label,
                "axis": axis,
                "phase": np.linspace(0, 1, n_pts).tolist(),
                "mean_profile": mean_profile.tolist(),
            })
    if profiles:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(profiles, f, indent=2)


# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def format_pvalue(p: float) -> str:
    if not np.isfinite(p):
        return "p=n/a"
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"


def format_qvalue(q: float) -> str:
    if not np.isfinite(q):
        return "q=n/a"
    return "q<0.001" if q < 0.001 else f"q={q:.3f}"


def annotate_stats(ax: plt.Axes, stats_row: Optional[pd.Series], use_fdr: bool = True) -> None:
    if stats_row is None:
        return
    n = int(stats_row.get("n_pairs", 0))
    delta = stats_row.get("mean_change", np.nan)
    dz = stats_row.get("effect_size_dz", np.nan)
    p = stats_row.get("p_value", np.nan)
    q = stats_row.get("p_value_fdr", np.nan) if use_fdr else p
    delta_str = f"Δ={delta:+.3f}" if np.isfinite(delta) else "Δ=n/a"
    dz_str = f"dz={dz:.2f}" if np.isfinite(dz) else "dz=n/a"
    parts = [f"n={n}", delta_str, dz_str, format_pvalue(p), format_qvalue(q)]
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
    sems = [scipy_stats.sem(v0) if n > 1 else 0.0, scipy_stats.sem(v1) if n > 1 else 0.0]
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
    sem_delta = scipy_stats.sem(delta) if len(delta) > 1 else 0.0
    ax.errorbar([0], [mean_delta], yerr=[sem_delta], color="black", marker="D", markersize=5, capsize=4, zorder=4)
    ax.set_xlim(-0.2, 0.2)
    ax.set_xticks([0])
    ax.set_xticklabels(["T1 - T0"])
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    annotate_stats(ax, stats_row)


def plot_feature_family(
    paired_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_dir: Path,
    conditions: List[str],
    dpi: int = 150,
    level_label: str = "",
) -> None:
    """Paired slope + delta per feature per condition."""
    figs_dir = out_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)
    for feat in IMU_FEATURES:
        t0_col, t1_col = f"{feat}_T0", f"{feat}_T1"
        if t0_col not in paired_df.columns:
            continue
        ylabel = FEATURE_LABELS.get(feat, feat)
        for condition in conditions:
            subset = paired_df[paired_df["condition"] == condition]
            if subset.empty:
                continue
            stats_row = None
            if "condition" in stats_df.columns:
                ss = stats_df[(stats_df["feature"] == feat) & (stats_df["condition"] == condition)]
            else:
                ss = stats_df[stats_df["feature"] == feat]
            if not ss.empty:
                stats_row = ss.iloc[0]
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            plot_paired_slope(
                axes[0],
                subset[t0_col].to_numpy(dtype=float),
                subset[t1_col].to_numpy(dtype=float),
                ylabel,
                f"{ylabel} ({condition}){level_label}",
                stats_row,
            )
            delta_col = f"{feat}_delta"
            if delta_col in subset.columns:
                plot_delta_strip(
                    axes[1],
                    subset[delta_col].to_numpy(dtype=float),
                    f"Delta {ylabel}",
                    f"{ylabel} delta ({condition}){level_label}",
                    stats_row,
                )
            fig.tight_layout()
            suffix = f"_{level_label.strip()}" if level_label else ""
            fig.savefig(figs_dir / f"paired_{feat}_{condition}{suffix}.png", dpi=dpi, bbox_inches="tight")
            plt.close(fig)


def plot_feature_grid(
    paired_df: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_dir: Path,
    conditions: List[str],
    dpi: int = 150,
    level_label: str = "",
) -> None:
    available = [f for f in IMU_FEATURES if f"{f}_T0" in paired_df.columns]
    if not available or not conditions:
        return
    fig, axes = plt.subplots(len(available), len(conditions), figsize=(5 * len(conditions), 4 * len(available)), squeeze=False)
    for i, feat in enumerate(available):
        for j, condition in enumerate(conditions):
            subset = paired_df[paired_df["condition"] == condition]
            stats_row = None
            if "condition" in stats_df.columns:
                ss = stats_df[(stats_df["feature"] == feat) & (stats_df["condition"] == condition)]
            else:
                ss = stats_df[stats_df["feature"] == feat]
            if not ss.empty:
                stats_row = ss.iloc[0]
            plot_paired_slope(
                axes[i, j],
                subset.get(f"{feat}_T0", pd.Series(dtype=float)).to_numpy(dtype=float),
                subset.get(f"{feat}_T1", pd.Series(dtype=float)).to_numpy(dtype=float),
                FEATURE_LABELS.get(feat, feat),
                f"{feat} ({condition})",
                stats_row,
            )
    fig.suptitle(f"IMU timing/smoothness/complexity features T0 vs T1{level_label}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    suffix = f"_{level_label.strip().replace(' ', '_')}" if level_label else ""
    fig.savefig(out_dir / "figures" / f"paired_imu_features_grid{suffix}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_delta_heatmap(
    sensor_stats_df: pd.DataFrame,
    out_dir: Path,
    feature: str,
    condition: str,
    dpi: int = 150,
) -> None:
    if sensor_stats_df.empty or "feature" not in sensor_stats_df.columns:
        return
    sub = sensor_stats_df[
        (sensor_stats_df["feature"] == feature) & (sensor_stats_df["condition"] == condition)
    ]
    if sub.empty:
        return
    pivot = sub.pivot_table(index="sensor", columns="axis", values="mean_change", aggfunc="mean")
    if pivot.empty:
        return
    fig, ax = plt.subplots(figsize=(6, max(4, len(pivot) * 0.5)))
    vlim = np.nanmax(np.abs(pivot.values)) if pivot.values.size and np.any(np.isfinite(pivot.values)) else 0.0
    vlim = max(vlim, 1e-9)
    im = ax.imshow(pivot.values, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index, fontsize=8)
    ax.set_title(f"{FEATURE_LABELS.get(feature, feature)} mean delta ({condition})")
    plt.colorbar(im, ax=ax, label="Mean delta (T1-T0)")
    fig.tight_layout()
    (out_dir / "figures" / "heatmaps").mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "figures" / "heatmaps" / f"delta_heatmap_{feature}_{condition}.png", dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_normalized_cycle_profiles(
    profile_path: Path,
    out_dir: Path,
    patient_id: str,
    task_name: str,
    condition: str,
    dpi: int = 150,
) -> None:
    """
    Plot T0 vs T1 average normalized cycle profiles.
    SHAPE comparison only - not amplitude.
    """
    if not profile_path.exists():
        return
    try:
        with open(profile_path) as f:
            all_profiles = json.load(f)
    except Exception:
        return
    # Filter this patient/task/condition
    t0_profs = [p for p in all_profiles if p.get("patient_id") == patient_id and p.get("task_name") == task_name and p.get("session") == "T0" and p.get("condition") == condition]
    t1_profs = [p for p in all_profiles if p.get("patient_id") == patient_id and p.get("task_name") == task_name and p.get("session") == "T1" and p.get("condition") == condition]
    # Need matching task names for T0 and T1 - task names differ by session
    # For simplicity, plot what we have per sensor-axis
    for prof in t0_profs + t1_profs:
        sensor = prof.get("sensor", "")
        axis = prof.get("axis", "")
        phase = np.array(prof.get("phase", []))
        mean_profile = np.array(prof.get("mean_profile", []))
        if phase.size == 0 or mean_profile.size == 0:
            continue
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(phase, mean_profile, label=f"{prof.get('session', '')} {sensor} {axis}")
        ax.set_xlabel("Phase (0-1)")
        ax.set_ylabel("Normalized value (shape)")
        ax.set_title(f"Cycle shape: {patient_id} {task_name} {condition} {sensor} {axis}")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        fig.tight_layout()
        safe = f"{patient_id}_{task_name}_{condition}_{sensor}_{axis}".replace(" ", "_").replace(":", "_")
        (out_dir / "figures" / "cycle_profiles").mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / "figures" / "cycle_profiles" / f"{safe}.png", dpi=dpi, bbox_inches="tight")
        plt.close(fig)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser(
        description="IMU feature computation (timing, smoothness, complexity) for non-calibrated sensors"
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data/emg_structured"),
                    help="Input data dir (EMG dirs with *_imu.pkl, *_manual_segments.json)")
    ap.add_argument("--results-dir", type=Path, default=Path("results"),
                    help="Root results dir; IMU output written to {dir}/imu_features")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Override: IMU output dir (default: {results-dir}/imu_features)")
    ap.add_argument("--patients", type=str, nargs="*", help="Limit to these patient IDs")
    ap.add_argument("--conditions", type=str, nargs="*", default=DEFAULT_CONDITIONS)
    ap.add_argument("--lowpass-hz", type=float, default=DEFAULT_LOWPASS_HZ,
                    help="Low-pass cutoff for smoothing (not calibration)")
    ap.add_argument("--n-phase-points", type=int, default=N_PHASE_POINTS,
                    help="Points per phase-normalized cycle (visualization only)")
    ap.add_argument("--min-cycles", type=int, default=DEFAULT_MIN_CYCLES,
                    help="Minimum valid cycles for feature aggregation")
    ap.add_argument("--cycle-normalization", type=str, default="zscore", choices=["demean", "zscore"],
                    help="Normalization for feature computation: demean or zscore")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--save-profiles", action="store_true", help="Save cycle profiles for visualization")
    ap.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging (per sensor-axis details)")
    args = ap.parse_args()

    if args.verbose:
        logging.getLogger(__name__).setLevel(logging.DEBUG)

    out_dir = args.out_dir if args.out_dir is not None else (args.results_dir / "imu_features")

    patients = discover_patients(args.data_dir)
    if args.patients:
        patients = [p for p in patients if p in args.patients]
    if not patients:
        logger.warning("No patients found in %s", args.data_dir)
        return
    logger.info("Processing %d patients: %s", len(patients), patients)

    all_task_rows = []
    all_qc_rows = []
    patients_with_data: List[str] = []
    patients_no_tasks: List[str] = []
    for patient_id in patients:
        tasks = discover_tasks_with_segments(patient_id, args.data_dir)
        if not tasks:
            logger.info("Patient %s: SKIP - no tasks with segments+IMU", patient_id)
            patients_no_tasks.append(patient_id)
            continue
        logger.info("Patient %s: %d task(s) to process: %s", patient_id, len(tasks), tasks)
        for task_name in tasks:
            task_df, qc_df = process_patient_task(
                patient_id,
                task_name,
                args.data_dir,
                lowpass_hz=args.lowpass_hz,
                min_cycles=args.min_cycles,
                norm_mode=args.cycle_normalization,
            )
            if task_df is not None and not task_df.empty:
                all_task_rows.append(task_df)
                if patient_id not in patients_with_data:
                    patients_with_data.append(patient_id)
            if qc_df is not None and not qc_df.empty:
                all_qc_rows.append(qc_df)

    # Final summary: who worked, who got stuck
    patients_all_skipped = [p for p in patients if p not in patients_no_tasks and p not in patients_with_data]
    logger.info("")
    logger.info("--- Processing summary ---")
    logger.info("Patients WITH data: %s", patients_with_data if patients_with_data else "(none)")
    logger.info("Patients with NO tasks (segments+IMU): %s", patients_no_tasks if patients_no_tasks else "(none)")
    if patients_all_skipped:
        logger.info("Patients with tasks but ALL skipped (see SKIP messages above): %s", patients_all_skipped)

    if not all_task_rows:
        logger.warning("No data processed. Check --data-dir and manual_segments.")
        return

    task_df = pd.concat(all_task_rows, ignore_index=True)
    qc_df = pd.concat(all_qc_rows, ignore_index=True)
    qc_df = _fix_qc_typo(qc_df)

    # Per-task pairing (primary)
    paired_per_task = build_paired_df_per_task(task_df)

    # Aggregated pairing (secondary)
    paired_aggregated = build_paired_df_aggregated(paired_per_task) if not paired_per_task.empty else pd.DataFrame()

    # Stats: per-task and aggregated
    if not paired_per_task.empty:
        stats_per_task = compute_paired_stats(
            paired_per_task,
            group_cols=["condition"],
            feature_cols=IMU_FEATURES,
        )
        sensor_axis_stats = compute_sensor_axis_stats(paired_per_task)
    else:
        stats_per_task = pd.DataFrame()
        sensor_axis_stats = pd.DataFrame()

    if not paired_aggregated.empty:
        stats_aggregated = compute_paired_stats(
            paired_aggregated,
            group_cols=["condition"],
            feature_cols=IMU_FEATURES,
        )
    else:
        stats_aggregated = pd.DataFrame()

    out_dir.mkdir(parents=True, exist_ok=True)
    summary_dir = out_dir / "summary_tables"
    summary_dir.mkdir(parents=True, exist_ok=True)

    task_df.to_csv(summary_dir / "imu_features_per_task_sensor_axis.csv", index=False)
    qc_df.to_csv(summary_dir / "imu_features_qc.csv", index=False)
    paired_per_task.to_csv(summary_dir / "imu_features_paired_per_task.csv", index=False)
    paired_aggregated.to_csv(summary_dir / "imu_features_paired_aggregated.csv", index=False)
    if not stats_per_task.empty:
        stats_per_task.to_csv(summary_dir / "imu_feature_stats_per_task.csv", index=False)
    if not stats_aggregated.empty:
        stats_aggregated.to_csv(summary_dir / "imu_feature_stats_aggregated.csv", index=False)
    if not sensor_axis_stats.empty:
        sensor_axis_stats.to_csv(summary_dir / "imu_feature_stats_per_sensor_axis.csv", index=False)

    # Backward compatibility: also write imu_features_per_task.csv (aggregated per task for B7)
    task_agg = task_df.groupby(["patient_id", "task_name", "session", "condition"], as_index=False)[IMU_FEATURES].mean()
    task_agg.to_csv(summary_dir / "imu_features_per_task.csv", index=False)

    # Plots
    plot_feature_family(paired_aggregated, stats_aggregated, out_dir, args.conditions, args.dpi, " (aggregated)")
    plot_feature_family(paired_per_task, stats_per_task, out_dir, args.conditions, args.dpi, " (per-task)")
    plot_feature_grid(paired_aggregated, stats_aggregated, out_dir, args.conditions, args.dpi, " (aggregated)")
    for feat in IMU_FEATURES:
        for cond in args.conditions:
            plot_delta_heatmap(sensor_axis_stats, out_dir, feat, cond, args.dpi)

    logger.info("IMU features written to %s", out_dir)
    logger.info("  Per-task paired rows: %d", len(paired_per_task))
    logger.info("  Aggregated paired rows: %d", len(paired_aggregated))
    logger.info("  QC rows: %d", len(qc_df))
    logger.info("  Figures: %s", out_dir / "figures")


if __name__ == "__main__":
    main()
