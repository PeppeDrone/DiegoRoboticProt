#!/usr/bin/env python3
"""sliding_window_emg_synergy.py

Time-resolved EMG muscle synergy analysis via sliding-window NMF.

Modes: fixed_w (global W, per-window H) | free_window (full NMF per window, Hungarian alignment)
Scopes: whole_task | segments_concat (interpolate segments to common length, concatenate, then NMF)

Example:
  python sliding_window_emg_synergy.py --data-dir data/emg_structured --out-dir results/synergies \\
      --mode fixed_w --analysis-scope segments_concat
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import re
import sys
import traceback
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment, nnls
from scipy.signal import butter, filtfilt, resample_poly
from sklearn.decomposition import NMF

# -----------------------------------------------------------------------------
# Pickle compatibility for EMGRecord (pkl files saved by B0_parse_emg_patient_task)
# -----------------------------------------------------------------------------
try:
    import B0_parse_emg_patient_task as _parse_mod
    sys.modules["__main__"].ChannelInfo = _parse_mod.ChannelInfo
    sys.modules["__main__"].EMGRecord = _parse_mod.EMGRecord
    from B0_parse_emg_patient_task import EMGRecord
except ImportError:
    EMGRecord = Any

# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------


@dataclass
class PreprocessingConfig:
    bandpass_low_hz: float = 20.0
    bandpass_high_hz: float = 450.0
    bandpass_order: int = 4
    envelope_lowpass_hz: float = 6.0
    envelope_order: int = 4
    downsample_to_hz: Optional[float] = None
    scale_percentile: float = 99.0


@dataclass
class SlidingWindowConfig:
    window_length_s: float = 1.0
    window_length_pct: Optional[float] = 0.2
    step_length_s: Optional[float] = None
    window_overlap_fraction: float = 0.5
    min_valid_samples_per_window: int = 20
    min_fraction_nonzero_per_muscle: float = 0.01

    min_active_muscles_per_window: int = 2
    allow_cross_segment_windows: bool = False
    restrict_to_manual_segments: bool = False
    min_overall_activation: float = 1e-8
    max_nan_fraction: float = 0.1

    def get_step_length_s(self, window_length_s: float) -> float:
        """Step length from overlap fraction if step not explicit."""
        if self.step_length_s is not None:
            return self.step_length_s
        return window_length_s * (1.0 - self.window_overlap_fraction)

    def get_effective_window_and_step(
        self, times: np.ndarray, use_pct: bool = False
    ) -> Tuple[float, float]:
        """Return (window_length_s, step_length_s). If use_pct, window = pct of one cycle (1 unit)."""
        if use_pct and self.window_length_pct is not None:
            w_s = float(self.window_length_pct)
        else:
            w_s = self.window_length_s
        step_s = self.get_step_length_s(w_s)
        return w_s, step_s


@dataclass
class NMFConfig:
    n_synergies: int = 3
    init: str = "nndsvda"
    solver: str = "cd"
    beta_loss: float = 2.0
    max_iter: int = 500
    tol: float = 1e-3
    alpha_W: float = 0.0
    alpha_H: float = 0.0
    l1_ratio: float = 0.0
    random_state: Optional[int] = 42
    n_restarts: int = 10


@dataclass
class PlotConfig:
    save_svg: bool = False
    dpi: int = 150
    figsize_heatmap: Tuple[float, float] = (8, 6)
    figsize_timecourse: Tuple[float, float] = (12, 5)
    figsize_vaf: Tuple[float, float] = (10, 4)
    figsize_evolution: Tuple[float, float] = (14, 8)


@dataclass
class AnalysisConfig:
    data_dir: Path = Path("data/emg_structured")
    out_dir: Path = Path("results/synergies")
    mode: str = "fixed_w"
    analysis_scope: str = "segments_concat"
    muscles: List[str] = field(default_factory=list)
    max_patients: Optional[int] = None
    max_tasks: Optional[int] = None
    patients_filter: Optional[List[str]] = None
    tasks_filter: Optional[List[str]] = None
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    sliding_window: SlidingWindowConfig = field(default_factory=SlidingWindowConfig)
    nmf: NMFConfig = field(default_factory=NMFConfig)
    plot: PlotConfig = field(default_factory=PlotConfig)
    verbose: bool = True

    def to_dict(self) -> dict:
        d = {}
        for k, v in self.__dict__.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif hasattr(v, "__dataclass_fields__"):
                d[k] = asdict(v)
            else:
                d[k] = v
        return d


# -----------------------------------------------------------------------------
# Default muscle list
# -----------------------------------------------------------------------------

DEFAULT_MUSCLES = [
    "Brachioradialis",
    "Biceps Brachii Short Head",
    # "Primo Interosseo",
    "Pectoralis Major",
    "Anterior Deltoid",
    "Middle Deltoid",
    "Posterior Deltoid",
    "Triceps Brachii Lateral Head",
    "Triceps Brachii Long Head",
    "Infraspinatus",
    # "Teres Major",
    "Latissimus Dorsi",
    "Biceps Brachii Long Head",
    "Trapezius Middle",
]

EPS = 1e-12
ONSET_THRESHOLD_FRAC = 0.10
CCI_PAIRS = [
    ("Biceps Brachii Long Head", "Triceps Brachii Long Head"),
    ("Biceps Brachii Short Head", "Triceps Brachii Lateral Head"),
    ("Anterior Deltoid", "Posterior Deltoid"),
]


# -----------------------------------------------------------------------------
# Data loading utilities
# -----------------------------------------------------------------------------


def load_emg_record(patient_id: str, task_name: str, data_dir: Path) -> Optional[EMGRecord]:
    """Load EMG record from pickle. Returns None if file not found."""
    pkl_path = data_dir / f"{patient_id}_EMG" / f"{task_name}_emg.pkl"
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.warning(f"Failed to load {pkl_path}: {e}")
        return None


def get_fs_from_record(rec: EMGRecord) -> float:
    """Extract sampling frequency from record."""
    if not rec.channels:
        return 150.0
    return float(rec.channels[0].sampling_freq)


def discover_patients_with_data(data_dir: Path) -> List[str]:
    """Return sorted list of patient IDs that have EMG folders."""
    if not data_dir.exists():
        return []
    out = []
    for pdir in sorted(data_dir.iterdir()):
        if pdir.is_dir() and pdir.name.endswith("_EMG"):
            out.append(pdir.name.replace("_EMG", ""))
    return out


def discover_tasks_for_patient(patient_id: str, data_dir: Path) -> List[str]:
    """Return sorted list of task names (excluding Calibrazione)."""
    pdir = data_dir / f"{patient_id}_EMG"
    if not pdir.exists():
        return []
    out = []
    for p in sorted(pdir.glob("*_emg.pkl")):
        task = p.stem.replace("_emg", "")
        if "Calibrazione" not in task:
            out.append(task)
    return out


def load_manual_segments(
    patient_id: str,
    task_name: str,
    data_dir: Path,
) -> Optional[List[Tuple[float, float]]]:
    """Load manual segment pairs (start_s, end_s). Returns None if file missing."""
    path = data_dir / f"{patient_id}_EMG" / f"{task_name}_manual_segments.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        pairs = [(p["start_s"], p["end_s"]) for p in data.get("pairs", [])]
        return pairs if pairs else None
    except Exception as e:
        logging.warning(f"Failed to load segments {path}: {e}")
        return None


# -----------------------------------------------------------------------------
# Task name canonicalization (for robust synergy lookup)
# -----------------------------------------------------------------------------


def parse_session_condition(task_name: str) -> Tuple[str, str]:
    """Extract (session, condition) from task name. E.g. Task_T0_SN -> (T0, SN)."""
    task_upper = str(task_name).strip().upper()
    session = "T0" if "T0" in task_upper else "T1" if "T1" in task_upper else ""
    condition = "SN" if "SN" in task_upper else "DS" if "DS" in task_upper else ""
    return session, condition


def canonicalize_task_name(task_name: str) -> str:
    """
    Convert task names to a canonical form so equivalent labels map together.
    Example: Task_DS_T0 -> Task_T0_DS  (session before condition)
    Preserves session (T0/T1), condition (DS/SN), and a stable task prefix.
    If parsing fails, returns the original string.
    """
    s = str(task_name).strip()
    if not s:
        return s
    session, condition = parse_session_condition(s)
    if not session or not condition:
        return s
    parts = re.split(r"[_\-\s]+", s)
    prefix_parts = [p for p in parts if p.upper() not in ("T0", "T1", "SN", "DS")]
    prefix = "_".join(prefix_parts) if prefix_parts else "Task"
    return f"{prefix}_{session}_{condition}"


def load_synergy_recommendations(path: Path) -> Dict[str, Any]:
    """
    Load B2 synergy recommendations CSV. Returns structure supporting exact and canonical lookup.
    Return dict keys: "exact" -> {(patient_id, raw_task_name): k}, "canonical" -> {(patient_id, canonical_task_name): k}
    Uses k_recommended_clark2010. Falls back to empty lookups if file missing.
    """
    out: Dict[str, Any] = {"exact": {}, "canonical": {}}
    if not path.exists():
        logging.info("Synergy recommendations not found at %s; using --n-synergies for all tasks", path)
        return out
    try:
        df = pd.read_csv(path)
        if "patient_id" not in df.columns or "task_name" not in df.columns or "k_recommended_clark2010" not in df.columns:
            logging.warning("Synergy recommendations CSV missing required columns; using --n-synergies")
            return out
        exact = {}
        canonical = {}
        for _, row in df.iterrows():
            pid = str(row["patient_id"])
            raw_task = str(row["task_name"])
            k_val = int(row["k_recommended_clark2010"])
            exact[(pid, raw_task)] = k_val
            can_task = canonicalize_task_name(raw_task)
            key_can = (pid, can_task)
            if key_can in canonical and canonical[key_can] != k_val:
                logging.warning(
                    "Canonical conflict: (%s, %s) -> k=%d vs k=%d; keeping first",
                    pid, can_task, canonical[key_can], k_val,
                )
            else:
                canonical[key_can] = k_val
        out["exact"] = exact
        out["canonical"] = canonical
        logging.info("Loaded synergy recommendations: %d exact, %d canonical from %s", len(exact), len(canonical), path)
        return out
    except Exception as e:
        logging.warning("Failed to load synergy recommendations from %s: %s; using --n-synergies", path, e)
        return out


def resolve_task_synergy_number(
    patient_id: str,
    task_name: str,
    default_k: int,
    synergy_lookup: Optional[Dict[str, Any]],
    require_recommendation: bool = False,
    disable_canonical_match: bool = False,
) -> Tuple[int, Dict[str, Any]]:
    """
    Resolve k for a patient/task. Returns (selected_k, metadata dict).
    Matching: exact raw -> canonical (if enabled) -> default.
    """
    meta: Dict[str, Any] = {
        "patient_id": patient_id,
        "task_name_raw": task_name,
        "task_name_canonical": canonicalize_task_name(task_name),
        "k_selected": default_k,
        "k_source": "default_argument",
        "lookup_found": False,
        "matched_lookup_key": None,
        "default_k": default_k,
        "notes": "",
    }
    if synergy_lookup is None or (not synergy_lookup.get("exact") and not synergy_lookup.get("canonical")):
        if require_recommendation:
            meta["k_source"] = "missing_recommendation_error"
            meta["notes"] = "No B2 recommendation and --require-synergy-recommendations set"
            raise ValueError(
                f"Task {patient_id}/{task_name} has no B2 recommendation and --require-synergy-recommendations is set"
            )
        return default_k, meta

    exact_lu = synergy_lookup.get("exact", {})
    canon_lu = synergy_lookup.get("canonical", {})

    key_raw = (patient_id, task_name)
    if key_raw in exact_lu:
        meta["k_selected"] = exact_lu[key_raw]
        meta["k_source"] = "b2_recommendation_exact"
        meta["lookup_found"] = True
        meta["matched_lookup_key"] = str(key_raw)
        return meta["k_selected"], meta

    if not disable_canonical_match:
        can_task = canonicalize_task_name(task_name)
        key_can = (patient_id, can_task)
        if key_can in canon_lu:
            meta["k_selected"] = canon_lu[key_can]
            meta["k_source"] = "b2_recommendation_canonical"
            meta["lookup_found"] = True
            meta["matched_lookup_key"] = str(key_can)
            return meta["k_selected"], meta

    if require_recommendation:
        meta["k_source"] = "missing_recommendation_error"
        meta["notes"] = "No exact/canonical match and --require-synergy-recommendations set"
        raise ValueError(
            f"Task {patient_id}/{task_name} has no matching B2 recommendation and --require-synergy-recommendations is set"
        )
    meta["notes"] = "Fell back to --n-synergies"
    return default_k, meta


# -----------------------------------------------------------------------------
# Channel resolution
# -----------------------------------------------------------------------------


def resolve_emg_channel(muscle_name: str, emg_channels: List[str]) -> Optional[str]:
    """
    Resolve muscle name to full EMG channel. Case-insensitive partial matching.
    Handles formats: "Anterior Deltoid: EMG 5", "EMG 5 Anterior Deltoid", etc.
    """
    muscle_lower = muscle_name.strip().lower()
    if not muscle_lower:
        return None
    matches = []
    for ch in emg_channels:
        ch_lower = ch.lower()
        if "emg" not in ch_lower:
            continue
        if muscle_lower in ch_lower:
            matches.append(ch)
        else:
            parts = muscle_lower.split()
            if all(p in ch_lower for p in parts if len(p) > 1):
                matches.append(ch)
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    return matches[0]


def resolve_muscles_to_channels(
    muscles: List[str],
    emg_channels: List[str],
) -> Tuple[Dict[str, str], List[str]]:
    """
    Resolve all muscles to channel names.
    Returns (muscle -> channel mapping, list of unmatched muscles).
    """
    mapping = {}
    unmatched = []
    for m in muscles:
        ch = resolve_emg_channel(m, emg_channels)
        if ch:
            mapping[m] = ch
        else:
            unmatched.append(m)
    return mapping, unmatched


# -----------------------------------------------------------------------------
# Preprocessing
# -----------------------------------------------------------------------------


def _butter_bandpass(low: float, high: float, fs: float, order: int) -> Tuple[np.ndarray, np.ndarray]:
    """Butterworth bandpass. Clips cutoffs to valid range when fs is low (e.g. IMU-resampled data)."""
    nyq = fs / 2
    if nyq <= 0:
        raise ValueError(f"Invalid sampling frequency fs={fs}; cannot design filter.")
    low_clip = max(1.0, min(low, nyq * 0.99))
    high_clip = max(low_clip + 1.0, min(high, nyq * 0.99))
    b, a = butter(order, [low_clip, high_clip], btype="band", fs=fs)
    return b, a


def preprocess_emg_signal(
    raw: np.ndarray,
    fs: float,
    cfg: PreprocessingConfig,
) -> np.ndarray:
    """
    Preprocess raw EMG: bandpass -> rectify -> lowpass envelope.
    Returns non-negative envelope.
    """
    if len(raw) < 10:
        return raw.astype(np.float64)
    data = np.asarray(raw, dtype=np.float64)
    if np.any(~np.isfinite(data)):
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)

    if cfg.bandpass_low_hz > 0 and cfg.bandpass_high_hz > cfg.bandpass_low_hz:
        b, a = _butter_bandpass(
            cfg.bandpass_low_hz,
            cfg.bandpass_high_hz,
            fs,
            cfg.bandpass_order,
        )
        data = filtfilt(b, a, data)

    data = np.abs(data)
    nyq = fs / 2
    env_hz = min(cfg.envelope_lowpass_hz, nyq * 0.99)
    if env_hz < 0.5:
        env_hz = 0.5
    b, a = butter(cfg.envelope_order, env_hz, btype="low", fs=fs)
    data = filtfilt(b, a, data)
    return np.maximum(data, 0.0).astype(np.float64)


def compute_per_muscle_scale(X: np.ndarray, percentile: float = 99.0) -> np.ndarray:
    """Per-muscle scale from percentile (column-wise). Returns (n_muscles,) vector."""
    X = np.maximum(np.asarray(X, dtype=np.float64), 0.0)
    scale = np.nanpercentile(X, percentile, axis=0)
    scale = np.where(np.isnan(scale) | (scale <= 0), 1.0, scale)
    return np.maximum(scale, 1e-12)


def apply_per_muscle_scale(X: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Apply per-muscle scale and ensure non-negativity for NMF."""
    return np.maximum(np.asarray(X, dtype=np.float64) / np.asarray(scale), 0.0)


# -----------------------------------------------------------------------------
# Segmentation utilities
# -----------------------------------------------------------------------------


def build_emg_matrix(
    rec: EMGRecord,
    channel_order: List[str],
    cfg: PreprocessingConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Build (n_samples, n_muscles) matrix from EMG record.
    Returns (times, matrix, fs).
    """
    fs = get_fs_from_record(rec)
    times_list = []
    rows_list = []
    for ch in channel_order:
        if ch not in rec.data:
            continue
        t = rec.data[ch]["times"]
        v = rec.data[ch]["values"]
        env = preprocess_emg_signal(v, fs, cfg)
        times_list.append(t[: len(env)])
        rows_list.append(env[: len(t)])
    if not times_list:
        return np.array([]), np.array([]).reshape(0, 0), fs
    t_ref = np.asarray(times_list[0], dtype=np.float64)
    n = len(t_ref)
    n_muscles = len(rows_list)
    mat = np.zeros((n, n_muscles), dtype=np.float64)
    for j, row in enumerate(rows_list):
        mat[: min(n, len(row)), j] = row[:n]
    if cfg.downsample_to_hz is not None and cfg.downsample_to_hz < fs:
        n_new = max(2, int(n * cfg.downsample_to_hz / fs))
        t_ref = np.linspace(t_ref[0], t_ref[-1], n_new, dtype=np.float64)
        mat = resample_poly(mat, n_new, n, axis=0).astype(np.float64)
        fs = cfg.downsample_to_hz
    return t_ref, mat, fs


def extract_segment_data(
    times: np.ndarray,
    matrix: np.ndarray,
    start_s: float,
    end_s: float,
) -> Tuple[np.ndarray, np.ndarray]:
    mask = (times >= start_s) & (times <= end_s)
    if not np.any(mask):
        return np.array([]), np.array([]).reshape(0, matrix.shape[1])
    return times[mask], matrix[mask]


def build_segmented_concat(
    times: np.ndarray,
    X: np.ndarray,
    segments: List[Tuple[float, float]],
    n_ref_samples: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Time-normalize each segment to common length via interpolation, then concatenate.
    Returns (t_concat, X_concat, X_cycles, phase). One cycle spans 1.0 in t_concat.
    """
    if not segments or X.size == 0:
        n_muscles = X.shape[1] if X.ndim > 1 else 0
        return (
            np.array([]),
            np.array([]).reshape(0, n_muscles),
            np.array([]).reshape(0, 0, n_muscles),
            np.array([]),
        )
    parts = []
    for s, e in segments:
        t_s, M_s = extract_segment_data(times, X, s, e)
        if M_s.shape[0] < 10:
            continue
        parts.append((t_s, M_s))
    if not parts:
        return (
            np.array([]),
            np.array([]).reshape(0, X.shape[1]),
            np.array([]).reshape(0, 0, X.shape[1]),
            np.array([]),
        )
    if n_ref_samples is None:
        n_ref_samples = int(np.median([M.shape[0] for _, M in parts]))
    n_ref_samples = max(10, n_ref_samples)
    phase = np.linspace(0.0, 1.0, n_ref_samples, dtype=np.float64)
    cycle_stack = []
    for t_s, M_s in parts:
        span = float(t_s[-1] - t_s[0]) if len(t_s) > 1 else 1.0
        if span <= 0:
            continue
        t_phase = (np.asarray(t_s) - t_s[0]) / span
        M_res = np.zeros((n_ref_samples, M_s.shape[1]), dtype=np.float64)
        for j in range(M_s.shape[1]):
            M_res[:, j] = np.interp(phase, t_phase, M_s[:, j])
        cycle_stack.append(M_res)
    if not cycle_stack:
        return (
            np.array([]),
            np.array([]).reshape(0, X.shape[1]),
            np.array([]).reshape(0, 0, X.shape[1]),
            np.array([]),
        )
    X_cycles = np.stack(cycle_stack, axis=0)
    X_concat = X_cycles.reshape(-1, X_cycles.shape[2])
    t_concat = np.arange(X_concat.shape[0], dtype=np.float64) / float(n_ref_samples)
    return t_concat, X_concat, X_cycles, phase


def get_valid_time_mask(
    times: np.ndarray,
    segments: Optional[List[Tuple[float, float]]],
    restrict_to_segments: bool,
) -> np.ndarray:
    """Boolean mask of valid samples for analysis."""
    mask = np.ones(len(times), dtype=bool)
    if segments is None or not restrict_to_segments:
        return mask
    mask[:] = False
    for s, e in segments:
        mask |= (times >= s) & (times <= e)
    return mask


# -----------------------------------------------------------------------------
# Sliding window generation
# -----------------------------------------------------------------------------


@dataclass
class WindowInfo:
    start_s: float
    end_s: float
    center_s: float
    indices: np.ndarray
    n_samples: int
    segment_id: Optional[int]
    valid: bool
    skip_reason: Optional[str] = None


def generate_sliding_windows(
    times: np.ndarray,
    window_length_s: float,
    step_length_s: float,
    valid_mask: Optional[np.ndarray] = None,
    segments: Optional[List[Tuple[float, float]]] = None,
    restrict_to_segments: bool = False,
    allow_cross_segment: bool = False,
    min_valid_samples: int = 20,
) -> List[WindowInfo]:
    """
    Generate sliding windows. valid_mask=True for samples to include.
    If restrict_to_segments and segments given, only windows within segments.
    """
    if valid_mask is None:
        valid_mask = np.ones(len(times), dtype=bool)
    n = len(times)
    if n < 2:
        return []

    dt = float(times[1] - times[0])
    if dt <= 0:
        dt = 1.0 / 150.0
    n_win = max(1, int(window_length_s / dt))
    n_step = max(1, int(step_length_s / dt))

    windows: List[WindowInfo] = []
    i_start = 0
    seg_id = None

    if restrict_to_segments and segments:
        for seg_idx, (s, e) in enumerate(segments):
            i_s = np.searchsorted(times, s, side="left")
            i_e = np.searchsorted(times, e, side="right")
            for i in range(i_s, i_e - n_win + 1, n_step):
                idx = np.arange(i, min(i + n_win, n))
                n_valid = np.sum(valid_mask[idx])
                if n_valid < min_valid_samples:
                    continue
                t_start = times[i]
                t_end = times[min(i + n_win - 1, n - 1)]
                w = WindowInfo(
                    start_s=float(t_start),
                    end_s=float(t_end),
                    center_s=float((t_start + t_end) / 2),
                    indices=idx,
                    n_samples=len(idx),
                    segment_id=seg_idx,
                    valid=True,
                )
                windows.append(w)
    else:
        i = 0
        while i + n_win <= n:
            idx = np.arange(i, i + n_win)
            n_valid = np.sum(valid_mask[idx])
            if n_valid < min_valid_samples:
                i += n_step
                continue
            t_start = times[i]
            t_end = times[i + n_win - 1]
            if segments and not allow_cross_segment:
                in_any = any(
                    (t_start >= s and t_end <= e)
                    for s, e in segments
                )
                if not in_any:
                    i += n_step
                    continue
            w = WindowInfo(
                start_s=float(t_start),
                end_s=float(t_end),
                center_s=float((t_start + t_end) / 2),
                indices=idx,
                n_samples=len(idx),
                segment_id=seg_id,
                valid=True,
            )
            windows.append(w)
            i += n_step

    return windows


# -----------------------------------------------------------------------------
# Window QC
# -----------------------------------------------------------------------------


def check_window_quality(
    X_win: np.ndarray,
    cfg: SlidingWindowConfig,
) -> Tuple[bool, Optional[str]]:
    """Return (is_valid, skip_reason)."""
    if X_win.size == 0:
        return False, "empty"
    n_nan = np.sum(~np.isfinite(X_win))
    if n_nan / X_win.size > cfg.max_nan_fraction:
        return False, "too_many_nans"
    n_samples, n_muscles = X_win.shape
    if n_samples < cfg.min_valid_samples_per_window:
        return False, "too_few_samples"
    X_win = np.nan_to_num(X_win, nan=0.0)
    frac_nonzero = np.mean(X_win > 0, axis=0)
    n_active = np.sum(frac_nonzero >= cfg.min_fraction_nonzero_per_muscle)
    if n_active < cfg.min_active_muscles_per_window:
        return False, "too_few_active_muscles"
    if np.mean(X_win) < cfg.min_overall_activation:
        return False, "too_low_activation"
    var_per_muscle = np.var(X_win, axis=0)
    if np.all(var_per_muscle < 1e-20):
        return False, "zero_variance"
    return True, None


# -----------------------------------------------------------------------------
# NMF fitting
# -----------------------------------------------------------------------------


def _normalize_w_columns(W: np.ndarray, H: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """L2-normalize columns of W and rescale H rows to preserve X = W @ H."""
    for k in range(W.shape[1]):
        nk = np.linalg.norm(W[:, k])
        if nk > 1e-12:
            W[:, k] /= nk
            H[k, :] *= nk
    return W, H


def fit_global_nmf(
    X: np.ndarray,
    cfg: NMFConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Fit NMF to obtain W and H. Returns (W, H, reconstruction_error).
    Normalizes W columns (L2) and rescales H accordingly.
    """
    X = np.maximum(np.asarray(X, dtype=np.float64), 0.0)
    n_samples, n_muscles = X.shape
    n_syn = min(cfg.n_synergies, n_muscles, n_samples)
    if n_syn < 1:
        return np.zeros((n_muscles, 0)), np.zeros((0, n_samples)), np.nan

    best_err = np.inf
    best_W = None
    best_H = None

    for _ in range(cfg.n_restarts):
        nmf = NMF(
            n_components=n_syn,
            init=cfg.init,
            solver=cfg.solver,
            beta_loss=cfg.beta_loss,
            max_iter=cfg.max_iter,
            tol=cfg.tol,
            alpha_W=cfg.alpha_W,
            alpha_H=cfg.alpha_H,
            l1_ratio=cfg.l1_ratio,
            random_state=cfg.random_state + _ if cfg.random_state else None,
        )
        try:
            W = nmf.fit_transform(X.T)
            H = nmf.components_
            W, H = _normalize_w_columns(W.copy(), H.copy())
            X_hat = (W @ H).T
            err = np.mean((X - X_hat) ** 2)
            if err < best_err:
                best_err = err
                best_W = W
                best_H = H
        except Exception:
            continue

    if best_W is None:
        nmf = NMF(n_components=n_syn, init=cfg.init, solver=cfg.solver, random_state=cfg.random_state)
        best_W = nmf.fit_transform(X.T)
        best_H = nmf.components_
        best_W, best_H = _normalize_w_columns(best_W, best_H)
        best_err = np.mean((X - (best_W @ best_H).T) ** 2)

    return best_W, best_H, best_err


def solve_activations_fixed_w(
    X_win: np.ndarray,
    W: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Solve for H with W fixed using NNLS per time sample.
    Returns (H, X_hat, reconstruction_error).
    """
    X_win = np.maximum(np.asarray(X_win, dtype=np.float64), 0.0)
    n_samples, n_muscles = X_win.shape
    n_syn = W.shape[1]
    H = np.zeros((n_syn, n_samples), dtype=np.float64)
    for t in range(n_samples):
        H[:, t], _ = nnls(W, X_win[t, :])
    X_hat = (W @ H).T
    err = np.mean((X_win - X_hat) ** 2)
    return H, X_hat, err


def fit_window_nmf(
    X_win: np.ndarray,
    cfg: NMFConfig,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Full NMF per window. Returns (W, H, reconstruction_error)."""
    return fit_global_nmf(X_win, cfg)


# -----------------------------------------------------------------------------
# Synergy alignment (free-window mode)
# -----------------------------------------------------------------------------


def cosine_similarity_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Compute pairwise cosine similarity between columns of A and B."""
    def norm_cols(M):
        n = np.linalg.norm(M, axis=0, keepdims=True)
        n[n < 1e-12] = 1.0
        return M / n
    A_n = norm_cols(A)
    B_n = norm_cols(B)
    return A_n.T @ B_n


def align_synergies_to_reference(
    W_new: np.ndarray,
    W_ref: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reorder columns of W_new to maximize similarity to W_ref.
    Uses Hungarian assignment on negative cosine similarity (cost minimization).
    Returns (W_aligned, permutation).
    """
    n_muscles, n_syn = W_new.shape
    if W_ref.shape[1] != n_syn or W_ref.shape[0] != n_muscles:
        return W_new, np.arange(n_syn)
    sim = cosine_similarity_matrix(W_new, W_ref)
    cost = -sim
    row_idx, col_idx = linear_sum_assignment(cost)
    order = np.zeros(n_syn, dtype=int)
    order[col_idx] = row_idx
    return W_new[:, order], order


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------


def sanitize_name_for_column(name: str) -> str:
    """Convert free-form muscle/channel names into stable CSV column names."""
    cleaned = re.sub(r"[^0-9a-zA-Z]+", "_", str(name).strip().lower()).strip("_")
    return cleaned or "unnamed"


def safe_pearson_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with near-zero variance protection."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size < 2 or y.size < 2:
        return np.nan
    if np.nanstd(x) <= EPS or np.nanstd(y) <= EPS:
        return np.nan
    r = np.corrcoef(x, y)[0, 1]
    return float(r) if np.isfinite(r) else np.nan


def trapezoid_integral(y: np.ndarray, x: np.ndarray) -> float:
    """Version-safe trapezoidal integration for NumPy 1.x/2.x."""
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(y, x))
    return float(np.trapz(y, x))


def compute_w_similarity(W_a: np.ndarray, W_b: np.ndarray) -> float:
    """
    Align W_b to W_a and return mean diagonal cosine similarity.
    """
    if W_a is None or W_b is None:
        return np.nan
    if W_a.size == 0 or W_b.size == 0 or W_a.shape != W_b.shape:
        return np.nan
    W_b_aligned, _ = align_synergies_to_reference(W_b, W_a)
    sim_mat = cosine_similarity_matrix(W_a, W_b_aligned)
    return float(np.mean(np.diag(sim_mat)))


def compute_vaf(X: np.ndarray, X_hat: np.ndarray) -> float:
    """Uncentered VAF: 1 - sum((X - X_hat)^2) / sum(X^2)."""
    ss_tot = np.sum(X ** 2)
    if ss_tot < EPS:
        return 1.0
    ss_res = np.sum((X - X_hat) ** 2)
    return float(1.0 - ss_res / ss_tot)


def compute_per_muscle_vaf(X: np.ndarray, X_hat: np.ndarray) -> np.ndarray:
    """Per-muscle uncentered VAF (each column)."""
    n_muscles = X.shape[1]
    vaf = np.zeros(n_muscles)
    for j in range(n_muscles):
        ss_tot = np.sum(X[:, j] ** 2)
        if ss_tot < EPS:
            vaf[j] = 1.0
        else:
            vaf[j] = 1.0 - np.sum((X[:, j] - X_hat[:, j]) ** 2) / ss_tot
    return vaf


def synergy_activation_summary(H: np.ndarray) -> Dict[str, float]:
    """Mean, max, auc, centroid, and peak time per synergy."""
    n_syn, n_t = H.shape
    phase = np.linspace(0.0, 1.0, n_t, dtype=np.float64) if n_t > 0 else np.array([])
    out = {}
    for k in range(n_syn):
        h_k = np.asarray(H[k, :], dtype=np.float64)
        total = np.sum(h_k)
        out[f"synergy_{k}_mean"] = float(np.mean(h_k))
        out[f"synergy_{k}_max"] = float(np.max(h_k))
        out[f"synergy_{k}_auc"] = trapezoid_integral(h_k, phase) if n_t > 1 else 0.0
        out[f"synergy_{k}_centroid"] = float(np.sum(phase * h_k) / total) if total > EPS else np.nan
        out[f"synergy_{k}_peak_time"] = float(phase[int(np.argmax(h_k))]) if n_t > 0 else np.nan
    return out


def _compute_signal_timing_metrics(signal: np.ndarray, phase: np.ndarray) -> Dict[str, float]:
    """Onset/offset/duration/centroid from a single normalized cycle signal."""
    signal = np.asarray(signal, dtype=np.float64)
    phase = np.asarray(phase, dtype=np.float64)
    peak = float(np.max(signal)) if signal.size else 0.0
    total = float(np.sum(signal))
    centroid = float(np.sum(phase * signal) / total) if total > EPS else np.nan
    if peak <= EPS:
        return {
            "onset": np.nan,
            "offset": np.nan,
            "duration": np.nan,
            "centroid": centroid,
        }
    thr = ONSET_THRESHOLD_FRAC * peak
    idx = np.flatnonzero(signal >= thr)
    if idx.size == 0:
        return {
            "onset": np.nan,
            "offset": np.nan,
            "duration": np.nan,
            "centroid": centroid,
        }
    onset = float(phase[idx[0]])
    offset = float(phase[idx[-1]])
    return {
        "onset": onset,
        "offset": offset,
        "duration": float(offset - onset),
        "centroid": centroid,
    }


def compute_cycle_muscle_metrics(
    X_cycles: np.ndarray,
    phase: np.ndarray,
    muscle_names: List[str],
    muscle_channels: List[str],
) -> pd.DataFrame:
    """
    Returns one row per (cycle, muscle).
    """
    if X_cycles.size == 0 or len(muscle_names) == 0:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    n_cycles, _, n_muscles = X_cycles.shape
    for cycle_idx in range(n_cycles):
        for muscle_idx in range(n_muscles):
            signal = np.asarray(X_cycles[cycle_idx, :, muscle_idx], dtype=np.float64)
            timing = _compute_signal_timing_metrics(signal, phase)
            rows.append(
                {
                    "cycle_index": cycle_idx,
                    "muscle": muscle_names[muscle_idx],
                    "muscle_channel": muscle_channels[muscle_idx],
                    "mean_amp": float(np.mean(signal)),
                    "peak_amp": float(np.max(signal)),
                    "auc": trapezoid_integral(signal, phase),
                    **timing,
                }
            )
    return pd.DataFrame(rows)


def compute_cycle_cci_metrics(
    X_cycles: np.ndarray,
    phase: np.ndarray,
    muscle_names: List[str],
    muscle_pairs: List[Tuple[str, str]],
) -> pd.DataFrame:
    """
    Returns one row per (cycle, muscle_a, muscle_b).
    """
    if X_cycles.size == 0 or len(muscle_names) == 0:
        return pd.DataFrame()
    idx_by_name = {name: idx for idx, name in enumerate(muscle_names)}
    valid_pairs = [(a, b) for a, b in muscle_pairs if a in idx_by_name and b in idx_by_name]
    if not valid_pairs:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for cycle_idx in range(X_cycles.shape[0]):
        for muscle_a, muscle_b in valid_pairs:
            x = np.asarray(X_cycles[cycle_idx, :, idx_by_name[muscle_a]], dtype=np.float64)
            y = np.asarray(X_cycles[cycle_idx, :, idx_by_name[muscle_b]], dtype=np.float64)
            num = 2.0 * trapezoid_integral(np.minimum(x, y), phase)
            den = trapezoid_integral(x, phase) + trapezoid_integral(y, phase)
            rows.append(
                {
                    "cycle_index": cycle_idx,
                    "muscle_a": muscle_a,
                    "muscle_b": muscle_b,
                    "cci": float(num / den) if den > EPS else np.nan,
                }
            )
    return pd.DataFrame(rows)


def compute_cycle_pairwise_correlations(
    X_cycles: np.ndarray,
    muscle_names: List[str],
) -> pd.DataFrame:
    """
    Returns one row per (cycle, muscle_a, muscle_b).
    """
    if X_cycles.size == 0 or len(muscle_names) < 2:
        return pd.DataFrame()
    rows: List[Dict[str, Any]] = []
    for cycle_idx in range(X_cycles.shape[0]):
        for i in range(len(muscle_names)):
            for j in range(i + 1, len(muscle_names)):
                rows.append(
                    {
                        "cycle_index": cycle_idx,
                        "muscle_a": muscle_names[i],
                        "muscle_b": muscle_names[j],
                        "corr": safe_pearson_correlation(
                            X_cycles[cycle_idx, :, i],
                            X_cycles[cycle_idx, :, j],
                        ),
                    }
                )
    return pd.DataFrame(rows)


def compute_stability_metrics(
    X_cycles: np.ndarray,
    phase: np.ndarray,
    muscle_names: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns:
        cycle_similarity_df: one row per (cycle, muscle)
        stability_summary_df: one row per muscle
    """
    if X_cycles.size == 0 or len(muscle_names) == 0:
        return pd.DataFrame(), pd.DataFrame()

    cycle_similarity_rows: List[Dict[str, Any]] = []
    stability_rows: List[Dict[str, Any]] = []

    for muscle_idx, muscle_name in enumerate(muscle_names):
        muscle_cycles = np.asarray(X_cycles[:, :, muscle_idx], dtype=np.float64)
        mean_amp = np.mean(muscle_cycles, axis=1)
        peak_amp = np.max(muscle_cycles, axis=1)
        auc = np.array([trapezoid_integral(sig, phase) for sig in muscle_cycles], dtype=np.float64)

        mean_cycle = np.mean(muscle_cycles, axis=0)
        sims = np.array(
            [safe_pearson_correlation(sig, mean_cycle) for sig in muscle_cycles],
            dtype=np.float64,
        )
        for cycle_idx, sim in enumerate(sims):
            cycle_similarity_rows.append(
                {
                    "cycle_index": cycle_idx,
                    "muscle": muscle_name,
                    "similarity_to_mean_cycle": sim,
                }
            )

        def _cv(values: np.ndarray) -> float:
            mu = float(np.nanmean(values))
            if not np.isfinite(mu) or mu <= EPS:
                return np.nan
            sigma = float(np.nanstd(values))
            return sigma / mu

        stability_rows.append(
            {
                "muscle": muscle_name,
                "cv_mean_amp": _cv(mean_amp),
                "cv_peak_amp": _cv(peak_amp),
                "cv_auc": _cv(auc),
                "mean_similarity_to_mean_cycle": float(np.nanmean(sims)) if np.any(np.isfinite(sims)) else np.nan,
                "std_similarity_to_mean_cycle": float(np.nanstd(sims)) if np.any(np.isfinite(sims)) else np.nan,
            }
        )

    return pd.DataFrame(cycle_similarity_rows), pd.DataFrame(stability_rows)


def add_task_identifiers(df: pd.DataFrame, patient_id: str, task_name: str) -> pd.DataFrame:
    """Prepend patient/task identifiers to a metrics dataframe."""
    if df is None or df.empty:
        return pd.DataFrame() if df is None else df
    out = df.copy()
    out.insert(0, "task_name", task_name)
    out.insert(0, "patient_id", patient_id)
    return out


def build_task_summary_metrics(
    patient_id: str,
    task_name: str,
    cfg: AnalysisConfig,
    window_df: pd.DataFrame,
    cycle_muscle_df: pd.DataFrame,
    stability_df: pd.DataFrame,
    k_source: str = "",
) -> pd.DataFrame:
    """Build a compact one-row task summary from available window and cycle metrics."""
    session, condition = parse_session_condition(task_name)
    row: Dict[str, Any] = {
        "patient_id": patient_id,
        "task_name": task_name,
        "session": session,
        "condition": condition,
        "mode": cfg.mode,
        "analysis_scope": cfg.analysis_scope,
        "n_synergies": cfg.nmf.n_synergies,
        "k_source": k_source,
        "n_valid_windows": 0,
        "mean_global_vaf": np.nan,
        "mean_reconstruction_error": np.nan,
        "mean_w_similarity_to_global": np.nan,
    }

    if window_df is not None and not window_df.empty:
        valid = window_df[window_df["skip_flag"] == False] if "skip_flag" in window_df.columns else window_df
        row["n_valid_windows"] = int(len(valid))
        if not valid.empty:
            for col in [
                "global_vaf",
                "reconstruction_error",
                "w_similarity_to_global",
                "mean_per_muscle_vaf",
                "min_per_muscle_vaf",
            ]:
                if col in valid.columns:
                    row[f"mean_{col}" if col not in ("global_vaf", "reconstruction_error") else f"mean_{col}"] = float(valid[col].mean())
            row["mean_global_vaf"] = float(valid["global_vaf"].mean()) if "global_vaf" in valid.columns else np.nan
            row["mean_reconstruction_error"] = float(valid["reconstruction_error"].mean()) if "reconstruction_error" in valid.columns else np.nan
            row["mean_w_similarity_to_global"] = float(valid["w_similarity_to_global"].mean()) if "w_similarity_to_global" in valid.columns else np.nan
            for col in valid.columns:
                if re.match(r"synergy_\d+_(mean|max|auc|centroid|peak_time)$", col):
                    row[f"mean_{col}"] = float(valid[col].mean())

    if cycle_muscle_df is not None and not cycle_muscle_df.empty:
        for src, dst in [
            ("mean_amp", "mean_cycle_mean_amp"),
            ("peak_amp", "mean_cycle_peak_amp"),
            ("auc", "mean_cycle_auc"),
            ("duration", "mean_cycle_duration"),
            ("centroid", "mean_cycle_centroid"),
        ]:
            if src in cycle_muscle_df.columns:
                row[dst] = float(cycle_muscle_df[src].mean())

    if stability_df is not None and not stability_df.empty:
        col_map = {
            "cv_mean_amp": "mean_cv_mean_amp",
            "cv_peak_amp": "mean_cv_peak_amp",
            "cv_auc": "mean_cv_auc",
            "mean_similarity_to_mean_cycle": "mean_similarity_to_mean_cycle",
            "std_similarity_to_mean_cycle": "std_similarity_to_mean_cycle",
        }
        for src, dst in col_map.items():
            if src in stability_df.columns:
                row[dst] = float(stability_df[src].mean())

    return pd.DataFrame([row])


# -----------------------------------------------------------------------------
# Main processing
# -----------------------------------------------------------------------------


def process_patient_task(
    patient_id: str,
    task_name: str,
    muscles: List[str],
    cfg: AnalysisConfig,
    synergy_lookup: Optional[Dict[str, Any]] = None,
    require_recommendation: bool = False,
    disable_canonical_match: bool = False,
) -> Tuple[Optional[Dict], Optional[pd.DataFrame], Optional[Dict], Optional[Dict[str, Any]]]:
    """
    Process one patient/task. Returns (results_dict, window_metrics_df, channel_mapping, k_meta).
    k_meta is the synergy resolution metadata for audit.
    """
    rec = load_emg_record(patient_id, task_name, cfg.data_dir)
    if rec is None:
        return None, None, None, None

    emg_chans = list(rec.data.keys())
    emg_chans = [c for c in emg_chans if "EMG" in c.upper() and "ACC" not in c.upper()]
    mapping, unmatched = resolve_muscles_to_channels(muscles, emg_chans)
    if not mapping:
        return None, None, {"matched": {}, "unmatched": muscles}, None

    muscle_names = [m for m in muscles if m in mapping]
    channel_order = [mapping[m] for m in muscle_names]
    times, X_full, fs = build_emg_matrix(rec, channel_order, cfg.preprocessing)
    if X_full.size == 0:
        return None, None, {"matched": mapping, "unmatched": unmatched}, None

    segments = load_manual_segments(patient_id, task_name, cfg.data_dir)
    X_cycles = np.array([]).reshape(0, 0, X_full.shape[1])
    phase = np.array([])

    scale_vector = np.ones(X_full.shape[1], dtype=np.float64)
    if cfg.analysis_scope == "segments_concat" and segments:
        parent_data = []
        for s, e in segments:
            _, M_s = extract_segment_data(times, X_full, s, e)
            if M_s.size > 0:
                parent_data.append(M_s)
        if parent_data:
            scale_vector = compute_per_muscle_scale(
                np.vstack(parent_data), cfg.preprocessing.scale_percentile
            )
    else:
        valid_mask_temp = get_valid_time_mask(
            times, segments, cfg.sliding_window.restrict_to_manual_segments
        )
        X_parent = X_full[valid_mask_temp] if np.any(valid_mask_temp) else X_full
        if X_parent.size > 0:
            scale_vector = compute_per_muscle_scale(
                X_parent, cfg.preprocessing.scale_percentile
            )

    X_norm_full = apply_per_muscle_scale(X_full, scale_vector)

    if cfg.analysis_scope == "segments_concat" and segments:
        times, X_norm, X_cycles, phase = build_segmented_concat(times, X_norm_full, segments)
        if X_norm.size == 0:
            X_norm = X_norm_full
            valid_mask = get_valid_time_mask(
                times, segments, cfg.sliding_window.restrict_to_manual_segments
            )
            segments_for_windows = segments
        else:
            valid_mask = np.ones(len(times), dtype=bool)
            segments_for_windows = None
    else:
        valid_mask = get_valid_time_mask(
            times, segments, cfg.sliding_window.restrict_to_manual_segments
        )
        X_norm = X_norm_full
        segments_for_windows = segments
        X_cycles = np.array([]).reshape(0, 0, X_full.shape[1])
        phase = np.array([])

    sw_cfg = cfg.sliding_window
    use_pct = cfg.analysis_scope == "segments_concat" and X_cycles.size > 0
    window_s, step_s = sw_cfg.get_effective_window_and_step(times, use_pct=use_pct)
    windows = generate_sliding_windows(
        times,
        window_s,
        step_s,
        valid_mask=valid_mask,
        segments=segments_for_windows,
        restrict_to_segments=sw_cfg.restrict_to_manual_segments,
        allow_cross_segment=sw_cfg.allow_cross_segment_windows,
        min_valid_samples=sw_cfg.min_valid_samples_per_window,
    )

    n_syn, k_meta = resolve_task_synergy_number(
        patient_id,
        task_name,
        cfg.nmf.n_synergies,
        synergy_lookup,
        require_recommendation=require_recommendation,
        disable_canonical_match=disable_canonical_match,
    )
    nmf_cfg = replace(cfg.nmf, n_synergies=n_syn)
    logging.info(
        "Task %s (%s): using k=%d from %s",
        task_name, k_meta.get("task_name_canonical", ""), n_syn, k_meta.get("k_source", ""),
    )

    X_parent = X_norm[valid_mask] if np.any(valid_mask) else X_norm
    W_reference_global = None
    if X_parent.shape[0] >= nmf_cfg.n_synergies:
        W_reference_global, _, _ = fit_global_nmf(X_parent, nmf_cfg)
    W_global = W_reference_global

    rows = []
    H_windows = []
    W_windows = []
    valid_window_infos: List[WindowInfo] = []

    for wi, win in enumerate(windows):
        X_win = X_norm[win.indices]
        ok, reason = check_window_quality(X_win, sw_cfg)
        if not ok:
            rows.append({
                "patient_id": patient_id,
                "task_name": task_name,
                "segment_id": win.segment_id,
                "window_index": wi,
                "start_s": win.start_s,
                "end_s": win.end_s,
                "center_s": win.center_s,
                "n_samples": win.n_samples,
                "reconstruction_error": np.nan,
                "global_vaf": np.nan,
                "w_similarity_to_global": np.nan,
                "mean_per_muscle_vaf": np.nan,
                "min_per_muscle_vaf": np.nan,
                "skip_flag": True,
                "skip_reason": reason,
                "mode": cfg.mode,
                "n_synergies": nmf_cfg.n_synergies,
                "k_source": k_meta.get("k_source", ""),
            })
            continue

        if cfg.mode == "fixed_w" and W_global is not None:
            H_win, X_hat, err = solve_activations_fixed_w(X_win, W_global)
            W_win = W_global
        else:
            W_win, H_win, err = fit_window_nmf(X_win, nmf_cfg)
            if wi > 0 and W_windows and cfg.mode == "free_window":
                W_prev = W_windows[-1]
                W_win, perm = align_synergies_to_reference(W_win, W_prev)
                H_win = H_win[perm, :]
            X_hat = (W_win @ H_win).T
        W_windows.append(W_win)
        H_windows.append(H_win)
        valid_window_infos.append(win)

        vaf = compute_vaf(X_win, X_hat)
        per_muscle_vaf = compute_per_muscle_vaf(X_win, X_hat)
        w_similarity = compute_w_similarity(W_reference_global, W_win)
        summary = synergy_activation_summary(H_win)
        vaf_cols = {
            f"vaf_{sanitize_name_for_column(muscle_name)}": float(per_muscle_vaf[idx])
            for idx, muscle_name in enumerate(muscle_names[: len(per_muscle_vaf)])
        }

        row = {
            "patient_id": patient_id,
            "task_name": task_name,
            "segment_id": win.segment_id,
            "window_index": wi,
            "start_s": win.start_s,
            "end_s": win.end_s,
            "center_s": win.center_s,
            "n_samples": win.n_samples,
            "reconstruction_error": err,
            "global_vaf": vaf,
            "w_similarity_to_global": w_similarity,
            "mean_per_muscle_vaf": float(np.nanmean(per_muscle_vaf)) if per_muscle_vaf.size else np.nan,
            "min_per_muscle_vaf": float(np.nanmin(per_muscle_vaf)) if per_muscle_vaf.size else np.nan,
            "skip_flag": False,
            "skip_reason": None,
            "mode": cfg.mode,
            "n_synergies": nmf_cfg.n_synergies,
            "k_source": k_meta.get("k_source", ""),
            **summary,
            **vaf_cols,
        }
        rows.append(row)

    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    cycle_muscle_metrics = pd.DataFrame()
    cycle_cci_metrics = pd.DataFrame()
    cycle_corr_metrics = pd.DataFrame()
    cycle_similarity_metrics = pd.DataFrame()
    stability_summary_metrics = pd.DataFrame()

    if cfg.analysis_scope == "segments_concat" and X_cycles.size > 0 and phase.size > 0:
        cycle_muscle_metrics = add_task_identifiers(
            compute_cycle_muscle_metrics(X_cycles, phase, muscle_names, channel_order),
            patient_id,
            task_name,
        )
        cycle_cci_metrics = add_task_identifiers(
            compute_cycle_cci_metrics(X_cycles, phase, muscle_names, CCI_PAIRS),
            patient_id,
            task_name,
        )
        cycle_corr_metrics = add_task_identifiers(
            compute_cycle_pairwise_correlations(X_cycles, muscle_names),
            patient_id,
            task_name,
        )
        cycle_similarity_metrics, stability_summary_metrics = compute_stability_metrics(
            X_cycles,
            phase,
            muscle_names,
        )
        cycle_similarity_metrics = add_task_identifiers(cycle_similarity_metrics, patient_id, task_name)
        stability_summary_metrics = add_task_identifiers(stability_summary_metrics, patient_id, task_name)

    cfg_eff = replace(cfg, nmf=nmf_cfg)
    task_summary_metrics = build_task_summary_metrics(
        patient_id,
        task_name,
        cfg_eff,
        df,
        cycle_muscle_metrics,
        stability_summary_metrics,
        k_source=k_meta.get("k_source", ""),
    )

    results = {
        "patient_id": patient_id,
        "task_name": task_name,
        "k_meta": k_meta,
        "W_global": W_global,
        "W_reference_global": W_reference_global,
        "W_windows": W_windows,
        "H_windows": H_windows,
        "valid_windows": valid_window_infos,
        "times": times,
        "phase": phase,
        "X_cycles": X_cycles,
        "muscle_names": muscle_names,
        "channel_order": channel_order,
        "segments": segments,
        "windows": windows,
        "X_full": X_norm,
        "scale_vector": scale_vector,
        "cycle_muscle_metrics": cycle_muscle_metrics,
        "cycle_cci_metrics": cycle_cci_metrics,
        "cycle_corr_metrics": cycle_corr_metrics,
        "cycle_similarity_metrics": cycle_similarity_metrics,
        "stability_summary_metrics": stability_summary_metrics,
        "task_summary_metrics": task_summary_metrics,
    }
    ch_map = {"matched": mapping, "unmatched": unmatched}
    return results, df, ch_map, k_meta



# -----------------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------------


def plot_global_w_heatmap(
    W: np.ndarray,
    muscle_names: List[str],
    out_path: Path,
    cfg: PlotConfig,
) -> None:
    """Plot W heatmap (muscles x synergies)."""
    if W.size == 0:
        return
    fig, ax = plt.subplots(figsize=cfg.figsize_heatmap)
    im = ax.imshow(W, aspect="auto", cmap="viridis", interpolation="nearest")
    ax.set_yticks(range(len(muscle_names)))
    ax.set_yticklabels(muscle_names, fontsize=8)
    ax.set_xticks(range(W.shape[1]))
    ax.set_xticklabels([f"S{k}" for k in range(W.shape[1])])
    ax.set_xlabel("Synergy")
    ax.set_ylabel("Muscle")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()
    if cfg.save_svg:
        fig, ax = plt.subplots(figsize=cfg.figsize_heatmap)
        im = ax.imshow(W, aspect="auto", cmap="viridis", interpolation="nearest")
        ax.set_yticks(range(len(muscle_names)))
        ax.set_yticklabels(muscle_names, fontsize=8)
        ax.set_xticks(range(W.shape[1]))
        ax.set_xticklabels([f"S{k}" for k in range(W.shape[1])])
        plt.colorbar(im, ax=ax)
        plt.tight_layout()
        plt.savefig(out_path.with_suffix(".svg"), bbox_inches="tight")
        plt.close()


def plot_h_timecourse(
    times: np.ndarray,
    H_windows: List[np.ndarray],
    window_infos: List[WindowInfo],
    segments: Optional[List[Tuple[float, float]]],
    out_path: Path,
    cfg: PlotConfig,
) -> None:
    """Plot activation H over time (one line per synergy). X-axis: 0-100% cycle phase."""
    if not H_windows or not window_infos:
        return
    centers = np.array([w.center_s for w in window_infos])
    t_min, t_max = float(times[0]), float(times[-1])
    span = t_max - t_min
    if span <= 0:
        span = 1.0
    x_pct = (centers - t_min) / span * 100.0
    n_syn = H_windows[0].shape[0]
    H_centers = np.array([np.mean(H_windows[i], axis=1) for i in range(len(H_windows))])
    fig, ax = plt.subplots(figsize=cfg.figsize_timecourse)
    for k in range(n_syn):
        ax.plot(x_pct, H_centers[:, k], label=f"Synergy {k}", linewidth=1.5)
    ax.set_xlabel("Cycle phase (%)")
    ax.set_ylabel("Activation (mean per window)")
    ax.legend(loc="upper right", fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()


def plot_window_vaf(
    df: pd.DataFrame,
    out_path: Path,
    cfg: PlotConfig,
) -> None:
    """Plot global VAF / reconstruction error per window over time."""
    if df.empty or "center_s" not in df.columns:
        return
    valid = df[df["skip_flag"] == False]
    if valid.empty:
        return
    fig, ax = plt.subplots(figsize=cfg.figsize_vaf)
    ax.plot(valid["center_s"], valid["global_vaf"], "o-", markersize=4, linewidth=1)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Global VAF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()


def plot_free_window_evolution(
    W_windows: List[np.ndarray],
    window_centers: List[float],
    muscle_names: List[str],
    out_path: Path,
    cfg: PlotConfig,
) -> None:
    """Plot W evolution over windows (heatmap per synergy)."""
    if not W_windows or not muscle_names:
        return
    n_syn = W_windows[0].shape[1]
    fig, axes = plt.subplots(1, n_syn, figsize=cfg.figsize_evolution, sharey=True)
    if n_syn == 1:
        axes = [axes]
    for k in range(n_syn):
        W_k = np.array([W_windows[i][:, k] for i in range(len(W_windows))]).T
        im = axes[k].imshow(W_k, aspect="auto", cmap="viridis")
        axes[k].set_title(f"Synergy {k}")
        axes[k].set_yticks(range(len(muscle_names)))
        axes[k].set_yticklabels(muscle_names, fontsize=7)
        step = max(1, len(window_centers) // 10)
        tick_idx = list(range(0, len(window_centers), step))
        axes[k].set_xticks(tick_idx)
        axes[k].set_xticklabels([f"{window_centers[i]:.1f}" for i in tick_idx], rotation=45)
    axes[0].set_ylabel("Muscle")
    for ax in axes:
        ax.set_xlabel("Window (center s)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=cfg.dpi, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Cohort aggregation (one number per patient/task, paired T0 vs T1)
# -----------------------------------------------------------------------------


def _compute_w_similarity(W_a: np.ndarray, W_b: np.ndarray) -> float:
    """Mean cosine similarity between matching synergy columns. Aligns by similarity."""
    return compute_w_similarity(W_a, W_b)


def _compute_w_sparsity(W: np.ndarray, threshold_frac: float = 0.1) -> float:
    """Mean fraction of muscles above threshold per synergy (0-1, higher=less sparse)."""
    if W.size == 0:
        return np.nan
    thr = np.max(W, axis=0, keepdims=True) * threshold_frac
    active = (W >= thr).astype(float)
    return float(np.mean(np.sum(active, axis=0) / W.shape[0]))


def _simplified_h_profile_similarity(H_a: np.ndarray, H_b: np.ndarray) -> float:
    """Correlation between activation profiles. H from npz: (n_windows, n_syn, n_samples)."""
    if H_a.size == 0 or H_b.size == 0:
        return np.nan
    # H_windows npz: (n_windows, n_syn, n_samples)
    if H_a.ndim == 3 and H_b.ndim == 3:
        n_win = min(H_a.shape[0], H_b.shape[0])
        n_syn = min(H_a.shape[1], H_b.shape[1])
        if n_win < 2:
            return np.nan
        ha = np.mean(H_a[:n_win, :n_syn, :], axis=(1, 2))  # (n_win,)
        hb = np.mean(H_b[:n_win, :n_syn, :], axis=(1, 2))
        r = np.corrcoef(ha, hb)[0, 1]
        return float(r) if not np.isnan(r) else np.nan
    # Fallback: (n_syn, n_windows)
    if H_a.ndim == 2 and H_b.ndim == 2:
        n_syn = min(H_a.shape[0], H_b.shape[0])
        n_win = min(H_a.shape[1], H_b.shape[1])
        if n_win < 2:
            return np.nan
        ha = np.mean(H_a[:n_syn, :n_win], axis=0)
        hb = np.mean(H_b[:n_syn, :n_win], axis=0)
        r = np.corrcoef(ha, hb)[0, 1]
        return float(r) if not np.isnan(r) else np.nan
    return np.nan


def compute_cohort_aggregation(out_dir: Path) -> None:
    """
    Aggregate metrics per patient/task and paired T0 vs T1.
    Saves patient_task_metrics.csv and patient_paired_metrics.csv.
    """
    out_dir = Path(out_dir)
    if not out_dir.exists():
        return
    task_rows = []
    paired_rows = []
    for pdir in sorted(out_dir.iterdir()):
        if not pdir.is_dir():
            continue
        patient_id = pdir.name
        t0_data = {}
        t1_data = {}
        for tdir in sorted(pdir.iterdir()):
            if not tdir.is_dir():
                continue
            task_name = tdir.name
            session, condition = parse_session_condition(task_name)
            if not session or not condition:
                continue
            if session == "T0":
                if condition not in t0_data:
                    t0_data[condition] = {"W": None, "H": None, "row": {}}
            else:
                if condition not in t1_data:
                    t1_data[condition] = {"W": None, "H": None, "row": {}}
            df_path = tdir / "window_metrics.csv"
            summary_path = tdir / "task_summary_metrics.csv"
            W_path = tdir / "W_global.csv"
            H_path = tdir / "H_windows.npz"
            row = {"patient_id": patient_id, "task_name": task_name, "session": session, "condition": condition}
            if summary_path.exists():
                try:
                    summary_df = pd.read_csv(summary_path)
                    if not summary_df.empty:
                        row.update(summary_df.iloc[0].to_dict())
                except Exception:
                    pass
            elif df_path.exists():
                df = pd.read_csv(df_path)
                valid = df[df["skip_flag"] == False] if "skip_flag" in df.columns else df
                if not valid.empty:
                    row["mean_global_vaf"] = valid["global_vaf"].mean()
                    row["mean_reconstruction_error"] = valid["reconstruction_error"].mean()
                    row["n_windows"] = len(valid)
                    for k in range(5):
                        c = f"synergy_{k}_mean"
                        if c in valid.columns:
                            row[f"agg_synergy_{k}_mean"] = valid[c].mean()
                        c = f"synergy_{k}_auc"
                        if c in valid.columns:
                            row[f"agg_synergy_{k}_auc"] = valid[c].mean()
            if W_path.exists():
                W_df = pd.read_csv(W_path, index_col=0)
                W = W_df.values
                row["w_sparsity"] = _compute_w_sparsity(W)
                row["n_synergies"] = W.shape[1]
                if session == "T0":
                    t0_data[condition]["W"] = W
                else:
                    t1_data[condition]["W"] = W
            if H_path.exists():
                try:
                    npz = np.load(H_path, allow_pickle=True)
                    H = npz["H"]
                    if session == "T0":
                        t0_data[condition]["H"] = H
                    else:
                        t1_data[condition]["H"] = H
                except Exception:
                    pass
            if session == "T0":
                t0_data[condition]["row"].update(row)
            else:
                t1_data[condition]["row"].update(row)
            task_rows.append(row)
        for condition in set(t0_data) | set(t1_data):
            t0 = t0_data.get(condition)
            t1 = t1_data.get(condition)
            if not t0 or not t1:
                continue
            prow = {"patient_id": patient_id, "condition": condition}
            r0, r1 = t0.get("row", {}), t1.get("row", {})
            if r0.get("mean_global_vaf") is not None and r1.get("mean_global_vaf") is not None:
                prow["vaf_T0"] = r0["mean_global_vaf"]
                prow["vaf_T1"] = r1["mean_global_vaf"]
                prow["vaf_delta"] = r1["mean_global_vaf"] - r0["mean_global_vaf"]
            if r0.get("mean_reconstruction_error") is not None and r1.get("mean_reconstruction_error") is not None:
                prow["recon_err_T0"] = r0["mean_reconstruction_error"]
                prow["recon_err_T1"] = r1["mean_reconstruction_error"]
                prow["recon_err_ratio"] = r1["mean_reconstruction_error"] / max(1e-12, r0["mean_reconstruction_error"])
            if t0.get("W") is not None and t1.get("W") is not None:
                W0, W1 = t0["W"], t1["W"]
                if W0.shape == W1.shape:
                    prow["w_similarity"] = _compute_w_similarity(W0, W1)
            if t0.get("H") is not None and t1.get("H") is not None:
                H0, H1 = t0["H"], t1["H"]
                prow["h_profile_similarity"] = _simplified_h_profile_similarity(H0, H1)
            for col in [
                "mean_cycle_mean_amp",
                "mean_cycle_peak_amp",
                "mean_cycle_auc",
                "mean_cv_mean_amp",
                "mean_cv_peak_amp",
                "mean_cv_auc",
                "mean_similarity_to_mean_cycle",
            ]:
                if r0.get(col) is not None and r1.get(col) is not None:
                    prow[f"{col}_T0"] = r0[col]
                    prow[f"{col}_T1"] = r1[col]
                    prow[f"{col}_delta"] = r1[col] - r0[col]
            paired_rows.append(prow)
    if task_rows:
        pd.DataFrame(task_rows).to_csv(out_dir / "patient_task_metrics.csv", index=False)
    if paired_rows:
        pd.DataFrame(paired_rows).to_csv(out_dir / "patient_paired_metrics.csv", index=False)


# -----------------------------------------------------------------------------
# Save outputs
# -----------------------------------------------------------------------------


def save_outputs(
    results: Dict,
    df: pd.DataFrame,
    ch_mapping: Dict,
    cfg: AnalysisConfig,
    patient_id: str,
    task_name: str,
) -> None:
    out_base = cfg.out_dir / patient_id / task_name
    out_base.mkdir(parents=True, exist_ok=True)
    plots_dir = out_base / "plots"
    plots_dir.mkdir(exist_ok=True)

    if df is not None and not df.empty:
        df.to_csv(out_base / "window_metrics.csv", index=False)

    with open(out_base / "channel_mapping.json", "w") as f:
        json.dump(ch_mapping, f, indent=2)

    W_global = results.get("W_global")
    muscle_names = results.get("muscle_names", results.get("channel_order", []))
    if W_global is not None:
        pd.DataFrame(W_global, index=muscle_names, columns=[f"S{k}" for k in range(W_global.shape[1])]).to_csv(out_base / "W_global.csv")
        np.savez_compressed(out_base / "W_global.npz", W=W_global, muscles=muscle_names)

    H_windows = results.get("H_windows", [])
    valid_wins = results.get("valid_windows", [])
    if H_windows:
        H_stack = np.array([h for h in H_windows])
        window_centers = np.array([w.center_s for w in valid_wins]) if valid_wins else np.arange(len(H_windows), dtype=np.float64)
        window_start_s = np.array([w.start_s for w in valid_wins]) if valid_wins else np.zeros(len(H_windows), dtype=np.float64)
        window_end_s = np.array([w.end_s for w in valid_wins]) if valid_wins else np.zeros(len(H_windows), dtype=np.float64)
        n_synergies = H_stack.shape[1] if H_stack.ndim >= 2 else 0
        muscle_names_list = results.get("muscle_names", results.get("channel_order", []))
        np.savez_compressed(
            out_base / "H_windows.npz",
            H=H_stack,
            window_centers=window_centers,
            window_start_s=window_start_s,
            window_end_s=window_end_s,
            n_synergies=np.array(n_synergies),
            muscle_names=np.array(muscle_names_list, dtype=object),
            task_name=np.array(task_name, dtype=object),
            patient_id=np.array(patient_id, dtype=object),
        )

    scale_vec = results.get("scale_vector")
    if scale_vec is not None:
        np.save(out_base / "scale_vector.npy", scale_vec)

    phase = results.get("phase")
    if phase is not None and len(phase) > 0:
        np.save(out_base / "phase.npy", phase)

    X_cycles = results.get("X_cycles")
    if X_cycles is not None and X_cycles.size > 0:
        muscle_names_cycles = results.get("muscle_names", results.get("channel_order", []))
        np.savez_compressed(
            out_base / "X_cycles.npz",
            X_cycles=X_cycles,
            phase=phase if phase is not None else np.array([]),
            muscle_names=np.array(muscle_names_cycles, dtype=object),
            patient_id=np.array(patient_id, dtype=object),
            task_name=np.array(task_name, dtype=object),
        )

    # synergy_task_metadata.json for B5 robustness/debugging
    session, condition = parse_session_condition(task_name)
    tsm = results.get("task_summary_metrics")
    n_syn_val = 0
    if tsm is not None and not tsm.empty and "n_synergies" in tsm.columns:
        n_syn_val = int(tsm.iloc[0]["n_synergies"])
    elif H_windows:
        _h = np.array([h for h in H_windows])
        n_syn_val = _h.shape[1] if _h.ndim >= 2 else 0
    k_meta = results.get("k_meta") or {}
    synergy_meta = {
        "patient_id": patient_id,
        "task_name": task_name,
        "session": session,
        "condition": condition,
        "n_synergies": n_syn_val,
        "k_source": str(k_meta.get("k_source", "")),
        "muscle_names": results.get("muscle_names", results.get("channel_order", [])),
        "channel_order": results.get("channel_order", []),
        "analysis_scope": cfg.analysis_scope,
        "mode": cfg.mode,
    }
    with open(out_base / "synergy_task_metadata.json", "w") as f:
        json.dump(synergy_meta, f, indent=2)

    for key, filename in [
        ("cycle_muscle_metrics", "cycle_muscle_metrics.csv"),
        ("cycle_cci_metrics", "cycle_cci_metrics.csv"),
        ("cycle_corr_metrics", "cycle_pairwise_correlations.csv"),
        ("cycle_similarity_metrics", "cycle_similarity_to_mean.csv"),
        ("stability_summary_metrics", "stability_summary_metrics.csv"),
        ("task_summary_metrics", "task_summary_metrics.csv"),
    ]:
        metric_df = results.get(key)
        if metric_df is not None and not metric_df.empty:
            metric_df.to_csv(out_base / filename, index=False)

    pc = cfg.plot
    if W_global is not None:
        plot_global_w_heatmap(W_global, muscle_names, plots_dir / "global_W_heatmap.png", pc)
    valid_wins = results.get("valid_windows", [])
    if H_windows and valid_wins:
        plot_h_timecourse(
            results["times"],
            H_windows,
            valid_wins,
            results.get("segments"),
            plots_dir / "H_timecourse.png",
            pc,
        )
    if df is not None and not df.empty:
        plot_window_vaf(df, plots_dir / "window_vaf.png", pc)
    if cfg.mode == "free_window" and results.get("W_windows") and valid_wins:
        centers = [w.center_s for w in valid_wins]
        plot_free_window_evolution(
            results["W_windows"],
            centers,
            muscle_names,
            plots_dir / "W_evolution.png",
            pc,
        )


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def load_muscles_from_args(
    muscles_str: Optional[str],
    muscles_file: Optional[Path],
) -> List[str]:
    """Load muscle list. For txt/csv, split on comma or newline only (preserves multi-word names)."""
    if muscles_file and muscles_file.exists():
        suffix = muscles_file.suffix.lower()
        if suffix == ".json":
            with open(muscles_file) as f:
                data = json.load(f)
            return data.get("muscles", data) if isinstance(data, dict) else data
        if suffix in (".txt", ".csv"):
            text = muscles_file.read_text().strip()
            return [m.strip() for m in re.split(r"[,\n]+", text) if m.strip()]
    if muscles_str:
        return [m.strip() for m in muscles_str.split(",") if m.strip()]
    return DEFAULT_MUSCLES.copy()


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Sliding-window EMG muscle synergy analysis via NMF",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data/emg_structured"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/synergies"))
    ap.add_argument("--patients", type=str, nargs="*", help="Limit to these patient IDs")
    ap.add_argument("--tasks", type=str, nargs="*", help="Limit to these task names")
    ap.add_argument("--muscles", type=str, help="Comma-separated muscle names")
    ap.add_argument("--muscles-file", type=Path, help="JSON or text file with muscle list")
    ap.add_argument("--analysis-scope", choices=["whole_task", "segments_concat"], default="segments_concat")
    ap.add_argument("--mode", choices=["fixed_w", "free_window"], default="fixed_w")
    ap.add_argument("--window-length-s", type=float, default=1.0, help="Window length in s (for whole_task/segments_concat)")
    ap.add_argument("--window-length-pct", type=float, default=0.20, help="Window length as fraction of cycle (segments_concat only)")
    ap.add_argument("--step-length-s", type=float, default=None)
    ap.add_argument("--window-overlap-fraction", type=float, default=0.5)
    ap.add_argument("--n-synergies", type=int, default=2, help="Fixed k when not using B2 recommendations")
    ap.add_argument("--synergy-recommendations", type=Path, default=None, help="B2 CSV: per-task k; default results/synergy_estimation/summary_tables/synergy_recommendations.csv")
    ap.add_argument("--require-synergy-recommendations", action="store_true", help="Fail when a task has no B2 recommendation instead of falling back to --n-synergies")
    ap.add_argument("--disable-canonical-task-match", action="store_true", help="Use exact task-name matching only (no Task_DS_T0 -> Task_T0_DS)")
    ap.add_argument("--n-restarts", type=int, default=10)
    ap.add_argument("--bandpass-low-hz", type=float, default=20.0)
    ap.add_argument("--bandpass-high-hz", type=float, default=450.0)
    ap.add_argument("--bandpass-order", type=int, default=4)
    ap.add_argument("--envelope-lowpass-hz", type=float, default=6.0)
    ap.add_argument("--envelope-order", type=int, default=4)
    ap.add_argument("--downsample-to-hz", type=float, default=None)
    ap.add_argument("--scale-percentile", type=float, default=99.0, help="Percentile for per-muscle scaling")
    ap.add_argument("--restrict-to-manual-segments", action="store_true")
    ap.add_argument("--allow-cross-segment-windows", action="store_true")
    ap.add_argument("--save-svg", action="store_true")
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--verbose", action="store_true", default=True)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    muscles = load_muscles_from_args(getattr(args, "muscles", None), getattr(args, "muscles_file", None))
    cfg = AnalysisConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        mode=args.mode,
        analysis_scope=args.analysis_scope,
        muscles=muscles,
        max_patients=args.max_patients,
        max_tasks=args.max_tasks,
        patients_filter=args.patients if args.patients else None,
        tasks_filter=args.tasks if args.tasks else None,
        preprocessing=PreprocessingConfig(
            bandpass_low_hz=args.bandpass_low_hz,
            bandpass_high_hz=args.bandpass_high_hz,
            bandpass_order=args.bandpass_order,
            envelope_lowpass_hz=args.envelope_lowpass_hz,
            envelope_order=args.envelope_order,
            downsample_to_hz=args.downsample_to_hz,
            scale_percentile=getattr(args, "scale_percentile", 99.0),
        ),
        sliding_window=SlidingWindowConfig(
            window_length_s=args.window_length_s,
            window_length_pct=args.window_length_pct,
            step_length_s=args.step_length_s,
            window_overlap_fraction=args.window_overlap_fraction,
            restrict_to_manual_segments=args.restrict_to_manual_segments,
            allow_cross_segment_windows=args.allow_cross_segment_windows,
        ),
        nmf=NMFConfig(
            n_synergies=args.n_synergies,
            n_restarts=args.n_restarts,
            random_state=args.random_state,
        ),
        plot=PlotConfig(
            save_svg=args.save_svg,
            dpi=args.dpi,
        ),
        verbose=args.verbose,
    )

    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    with open(cfg.out_dir / "config.json", "w") as f:
        json.dump(cfg.to_dict(), f, indent=2)

    synergy_rec_path = getattr(args, "synergy_recommendations", None)
    if synergy_rec_path is None:
        synergy_rec_path = Path("results/synergy_estimation/summary_tables/synergy_recommendations.csv")
    synergy_lookup = load_synergy_recommendations(synergy_rec_path)

    patients = discover_patients_with_data(cfg.data_dir)
    if cfg.patients_filter:
        patients = [p for p in patients if p in cfg.patients_filter]
    if cfg.max_patients:
        patients = patients[: cfg.max_patients]

    all_dfs = []
    unmatched_report = []
    k_resolution_rows: List[Dict[str, Any]] = []

    for patient_id in patients:
        print(f"Patient {patient_id}")
        tasks = discover_tasks_for_patient(patient_id, cfg.data_dir)
        if cfg.tasks_filter:
            tasks = [t for t in tasks if t in cfg.tasks_filter]
        if cfg.max_tasks:
            tasks = tasks[: cfg.max_tasks]
        for task_name in tasks:
            try:
                print(f"  {task_name}: loading EMG, preprocessing...")
                results, df, ch_map, k_meta = process_patient_task(
                    patient_id,
                    task_name,
                    muscles,
                    cfg,
                    synergy_lookup=synergy_lookup,
                    require_recommendation=getattr(args, "require_synergy_recommendations", False),
                    disable_canonical_match=getattr(args, "disable_canonical_task_match", False),
                )
                if results is None and ch_map is not None:
                    print(f"  {task_name}: skipped (unmatched channels)")
                    unmatched_report.append({"patient_id": patient_id, "task_name": task_name, **ch_map})
                    continue
                if results is None:
                    print(f"  {task_name}: skipped (no EMG data)")
                    continue
                if df is not None and not df.empty:
                    all_dfs.append(df)
                if results and ch_map:
                    n_win = len(df) if df is not None and not df.empty else 0
                    n_valid = n_win - (df["skip_flag"].sum() if "skip_flag" in df.columns else 0)
                    n_syn = int(df["n_synergies"].iloc[0]) if df is not None and not df.empty and "n_synergies" in df.columns else cfg.nmf.n_synergies
                    k_src = k_meta.get("k_source", "") if k_meta else ""
                    print(f"  {task_name}: done ({int(n_valid)}/{n_win} windows, k={n_syn} from {k_src}, mode={cfg.mode})")
                    save_outputs(results, df, ch_map, cfg, patient_id, task_name)
                    if k_meta:
                        k_resolution_rows.append({
                            "patient_id": k_meta.get("patient_id", patient_id),
                            "task_name_raw": k_meta.get("task_name_raw", task_name),
                            "task_name_canonical": k_meta.get("task_name_canonical", ""),
                            "k_selected": k_meta.get("k_selected", n_syn),
                            "k_source": k_meta.get("k_source", ""),
                            "lookup_found": k_meta.get("lookup_found", False),
                            "matched_lookup_key": k_meta.get("matched_lookup_key"),
                            "default_k": k_meta.get("default_k", cfg.nmf.n_synergies),
                            "notes": k_meta.get("notes", ""),
                        })
                    if ch_map.get("unmatched"):
                        unmatched_report.append({"patient_id": patient_id, "task_name": task_name, **ch_map})
            except Exception as e:
                logging.error(f"Error {patient_id}/{task_name}: {e}")
                if args.verbose:
                    traceback.print_exc()

    if all_dfs:
        cohort = pd.concat(all_dfs, ignore_index=True)
        cohort.to_csv(cfg.out_dir / "cohort_summary.csv", index=False)

    if unmatched_report:
        pd.DataFrame(unmatched_report).to_csv(cfg.out_dir / "unmatched_channels.csv", index=False)

    if k_resolution_rows:
        pd.DataFrame(k_resolution_rows).to_csv(cfg.out_dir / "synergy_k_resolution.csv", index=False)

    compute_cohort_aggregation(cfg.out_dir)
    print(f"Done. Outputs in {cfg.out_dir}")


if __name__ == "__main__":
    main()
