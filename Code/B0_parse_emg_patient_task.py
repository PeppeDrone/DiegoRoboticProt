#!/usr/bin/env python3
"""parse_emg_patient_task.py

Parses EMG CSV files and creates structured data structures for each patient_task.

Supported formats:
  - Original Delsys/EMGWorks: Label lines, comma delimiter, scientific notation
  - Trigno Discover: Application:; Trigno Discover..., semicolon delimiter, comma decimals

Input layout (data/emg/):
  CROSS_001_EMG/
    CROSS_001_Task_T0_SN.csv
    CROSS_001_Task_T0_DS.csv
    CROSS_001_Task_T1_SN.csv
    CROSS_001_Task_T1_DS.csv
    CROSS_001_Calibrazione_T0_SN.csv
    CROSS_001_Calibrazione_T0_DS.csv
    ...

Output layout (--out or data/emg_structured/):
  CROSS_001_EMG/
    Task_T0_SN_emg.pkl        (resampled to TARGET_FS: imu fs or fixed Hz)
    Task_T0_SN_emg_meta.json  (lightweight, for quality checks)
    Task_T0_SN_imu.pkl        (resampled to same fs as EMG)
    Task_T0_SN_imu_meta.json
    Task_T0_SN_resample_check.png  (before vs after resampling)
    ...                       (gyro not saved)

Each parsed record contains:
  - patient_id, task_name, task_type (Task|Calibrazione), session (T0|T1), condition (SN|DS)
  - channels: list of channel descriptors (name, unit, sampling_freq, n_points)
  - metadata: acquisition settings
  - data: dict mapping channel_name -> {times: ndarray, values: ndarray}
  - raw_matrix: optional full (n_samples, n_cols) array for downstream use

Usage:
  python parse_emg_patient_task.py [--emg-dir data/emg] [--out data/emg_structured] [--format pkl]
"""

from __future__ import annotations

import argparse
import json
import pickle
import random
import re
import warnings
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.signal import resample

try:
    import matplotlib
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


# -----------------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------------
# Target sampling frequency for saved files. Use "imu" to resample all at IMU fs
# (so EMG and IMU share same fs, length, n_samples); or a float (e.g. 1000.0) for fixed Hz.
TARGET_FS: Union[float, str] = "imu"


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class ChannelInfo:
    """Descriptor for one EMG/Acc/Gyro channel."""
    name: str
    unit: str
    sampling_freq: float
    n_points: int
    x_start: float
    domain_unit: str = "s"


@dataclass
class EMGRecord:
    """Structured record for one patient_task file."""
    patient_id: str
    task_name: str       # e.g. "Task_T0_SN"
    task_type: str       # Task | Calibrazione
    session: str         # T0 | T1
    condition: str       # SN | DS
    channels: List[ChannelInfo]
    metadata: Dict[str, str]
    channel_settings: List[Dict[str, float]]  # per-channel System Gain, etc.
    data: Dict[str, Dict[str, np.ndarray]]   # channel_name -> {times, values}
    raw_matrix: Optional[np.ndarray] = None  # full (n_rows, n_cols) if keep_raw
    column_header: Optional[str] = None      # original header line for reference

    def get_channel_data(self, name: str) -> Tuple[np.ndarray, np.ndarray]:
        """Return (times, values) for a channel by name."""
        d = self.data.get(name)
        if d is None:
            raise KeyError(f"Channel '{name}' not found. Available: {list(self.data.keys())}")
        return d["times"], d["values"]

    def get_emg_channels(self) -> List[str]:
        """Return names of EMG channels (excludes Acc, GYRO)."""
        return [c.name for c in self.channels if "EMG" in c.name and "Acc" not in c.name and "GYRO" not in c.name]

    def get_acc_channels(self) -> List[str]:
        """Return names of accelerometer channels."""
        return [c.name for c in self.channels if "Acc" in c.name or "ACC" in c.name]

    def get_gyro_channels(self) -> List[str]:
        """Return names of gyroscope channels."""
        return [c.name for c in self.channels if "GYRO" in c.name or "Gyro" in c.name]


def _get_imu_sampling_freq(rec: EMGRecord) -> float:
    """Return the sampling frequency of the first IMU (Acc) channel."""
    imu_names = rec.get_acc_channels()
    if not imu_names:
        raise ValueError("No IMU (Acc) channels found; cannot use TARGET_FS='imu'.")
    first_imu = imu_names[0]
    for c in rec.channels:
        if c.name == first_imu:
            return c.sampling_freq
    raise ValueError(f"IMU channel '{first_imu}' not in channel list.")


# -----------------------------------------------------------------------------
# Parser
# -----------------------------------------------------------------------------

LABEL_RE = re.compile(
    r"Label:\s*(.+?)\s+"
    r"Sampling frequency:\s*([\d.eE+-]+)\s+"
    r"Number of points:\s*(\d+)\s+"
    r"X start:\s*([\d.eE+-]+)\s+"
    r"Unit:\s*(\S+)\s+"
    r"Domain Unit:\s*(\w+)",
    re.IGNORECASE
)

# Match column header like "X [s],Brachioradialis: EMG 1 [Volts],..."
COL_HEADER_RE = re.compile(r"^X\s*\[s\],")  # starts with "X [s],"

# Match a numeric data row (scientific notation)
DATA_ROW_RE = re.compile(r"^-?[\d.]+e[+-]\d+")


def _parse_float(x: str) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


def _parse_float_comma(x: str) -> float:
    """Parse float with comma as decimal separator (e.g. '1259,2593' or ' 1259,2593 Hz')."""
    try:
        s = str(x).strip()
        m = re.search(r"[\d,]+(?:\.\d+)?", s)
        if m:
            s = m.group(0)
        return float(s.replace(",", "."))
    except (ValueError, TypeError):
        return np.nan


def _normalize_task_name(task_name: str) -> str:
    """Normalize Task_SN_T0 -> Task_T0_SN, Task_DS_T1 -> Task_T1_DS for consistency."""
    m = re.match(r"^Task_(SN|DS)_(T[01])$", task_name, re.IGNORECASE)
    if m:
        return f"Task_{m.group(2)}_{m.group(1)}"
    return task_name


def _is_trigno_discover_format(lines: List[str]) -> bool:
    """Check if file is Trigno Discover export (Application:; Trigno Discover...)."""
    if not lines:
        return False
    first = lines[0].strip()
    return "Trigno Discover" in first or ("Application" in first and ";" in first and "Trigno" in first)


# Italian / mixed -> canonical English muscle names
_MUSCLE_IT_TO_EN: Dict[str, str] = {
    "deltoide anteriore": "Anterior Deltoid",
    "deltoide medio": "Middle Deltoid",
    "deltoide posteriore": "Posterior Deltoid",
    "posterior deltoid": "Posterior Deltoid",
    "trapezio medio": "Trapezius Middle",
    "bicipite brachiale capo lungo": "Biceps brachii long head",
    "bicipite capo corto": "Biceps brachii short head",
    "biceps brachiale short head": "Biceps brachii short head",
    "tricipite brachiale capo laterale": "Triceps brachii lateral head",
    "tricipite brchiale capo lungo": "Triceps brachii long head",
    "triceps brachii lateral head": "Triceps brachii lateral head",
    "triceps brachii long head": "Triceps brachii long head",
    "latissimus dorsi": "Latissimus dorsi",
    "infraspinato": "Infraspinatus",
    "pettorale": "Pectoralis Major",
    "pectoralis major": "Pectoralis Major",
    "primo interosseo": "Primo Interosseo",
    "teres major": "Teres Major",
    "brachioradialis": "Brachioradialis",
    "anterior deltoid": "Anterior Deltoid",
    "middle deltoid": "Middle Deltoid",
    "trapezius middle": "Trapezius Middle",
    "biceps brachii long head": "Biceps brachii long head",
    "biceps brachii short head": "Biceps brachii short head",
    "infraspinatus": "Infraspinatus",
}

# Fixed 14-channel canonical order (same for ALL patients). Muscle name -> slot 1-14.
CANONICAL_EMG_CHANNELS: List[str] = [
    "Anterior Deltoid",
    "Teres Major",
    "Primo Interosseo",
    "Biceps brachii long head",
    "Triceps brachii lateral head",
    "Triceps brachii long head",
    "Trapezius Middle",
    "Infraspinatus",
    "Middle Deltoid",
    "Brachioradialis",
    "Posterior Deltoid",
    "Pectoralis Major",
    "Latissimus dorsi",
    "Biceps brachii short head",
]
_MUSCLE_TO_SLOT: Dict[str, int] = {
    name.lower(): i + 1 for i, name in enumerate(CANONICAL_EMG_CHANNELS)
}
# Also map common variants to slot
for it, en in _MUSCLE_IT_TO_EN.items():
    slot = _MUSCLE_TO_SLOT.get(en.lower())
    if slot is not None and it not in _MUSCLE_TO_SLOT:
        _MUSCLE_TO_SLOT[it] = slot


def _normalize_muscle_name(name: str) -> str:
    """Normalize muscle name to canonical English."""
    key = name.strip().lower()
    return _MUSCLE_IT_TO_EN.get(key, name.strip())


def _muscle_to_canonical_channel(
    muscle: str, is_emg: bool = True, axis: Optional[str] = None
) -> Optional[str]:
    """Map muscle to canonical channel name. Returns e.g. 'Anterior Deltoid: EMG 1' or None if unknown."""
    norm = _normalize_muscle_name(muscle)
    slot = _MUSCLE_TO_SLOT.get(norm.lower())
    if slot is None:
        return None
    canon = CANONICAL_EMG_CHANNELS[slot - 1]
    if is_emg:
        return f"{canon}: EMG {slot}"
    return f"{canon}: Acc {slot}.{axis or 'X'}"


def parse_trigno_discover_csv(
    filepath: Path,
    *,
    keep_raw: bool = False,
    encoding: str = "utf-8",
    max_rows: Optional[int] = None,
) -> EMGRecord:
    """Parse Trigno Discover export format (semicolon delimiter, comma decimals)."""
    print(f"\n--- Parsing {filepath.name} (Trigno Discover format) ---")

    with open(filepath, "r", encoding=encoding, errors="replace") as f:
        lines = f.readlines()

    print(f"  1. File read: {len(lines):,} lines")

    # Header: lines 1-3 metadata, 4=sensor names, 5=sensor mode, 6=col headers, 7=fs, 8=resolution, 9+=data
    def split_row(line: str) -> List[str]:
        return [p.strip() for p in line.strip().split(";")]

    line4 = split_row(lines[3]) if len(lines) > 3 else []
    line6 = split_row(lines[5]) if len(lines) > 5 else []
    line7 = split_row(lines[6]) if len(lines) > 6 else []

    # Extract sensor names: "Anterior Deltoid (85281)" or "Deltoide anteriore (85281)" -> canonical English
    sensor_names: List[str] = []
    for p in line4:
        if p and "(" in p and ")" in p:
            name = re.sub(r"\s*\(\d+\)\s*$", "", p).strip()
            sensor_names.append(_normalize_muscle_name(name))

    COLS_PER_SENSOR = 14  # EMG(t,v) + ACC X,Y,Z(t,v) + GYRO X,Y,Z(t,v)
    n_sensors = len(line6) // COLS_PER_SENSOR
    if n_sensors != len(sensor_names):
        n_sensors = min(n_sensors, len(sensor_names))
        sensor_names = sensor_names[:n_sensors]

    print(f"  2. Sensors: {n_sensors} ({sensor_names[:3]}...)")

    # Build channel list using canonical names (same 14 channels for all patients)
    channel_specs: List[Tuple[str, int, int, float]] = []
    channel_infos: List[ChannelInfo] = []
    acc_axes = {"ACC X": "X", "ACC Y": "Y", "ACC Z": "Z"}
    seen_slots: set = set()  # avoid duplicate canonical channels

    for s_idx in range(n_sensors):
        raw_muscle = sensor_names[s_idx]
        base_col = s_idx * COLS_PER_SENSOR
        emg_ch = _muscle_to_canonical_channel(raw_muscle, is_emg=True)
        if emg_ch is None:
            print(f"     [skip] unknown muscle '{raw_muscle}' at sensor {s_idx + 1}")
            continue
        slot = _MUSCLE_TO_SLOT.get(_normalize_muscle_name(raw_muscle).lower())
        if slot is not None and slot in seen_slots:
            continue  # same muscle from different sensor position
        if slot is not None:
            seen_slots.add(slot)
        emg_t, emg_v = base_col, base_col + 1
        fs_emg = 1259.26
        if base_col + 1 < len(line7):
            fs_emg = _parse_float_comma(line7[emg_v]) or fs_emg
        channel_specs.append((emg_ch, emg_t, emg_v, fs_emg))
        channel_infos.append(ChannelInfo(
            name=emg_ch,
            unit="mV",
            sampling_freq=fs_emg,
            n_points=0,
            x_start=0.0,
            domain_unit="s",
        ))
        for acc_label, axis in acc_axes.items():
            acc_ch = _muscle_to_canonical_channel(raw_muscle, is_emg=False, axis=axis)
            if acc_ch is None:
                continue
            t_col = base_col + 2 + list(acc_axes.keys()).index(acc_label) * 2
            v_col = t_col + 1
            fs_acc = 148.15
            if v_col < len(line7):
                fs_acc = _parse_float_comma(line7[v_col]) or fs_acc
            channel_specs.append((acc_ch, t_col, v_col, fs_acc))
            channel_infos.append(ChannelInfo(
                name=acc_ch,
                unit="g",
                sampling_freq=fs_acc,
                n_points=0,
                x_start=0.0,
                domain_unit="s",
            ))

    print(f"  3. Channels: {len(channel_specs)} (EMG + Acc X,Y,Z per sensor)")

    # Parse data (line 9+), semicolon delimiter, comma decimal
    data_start = 8
    data_lines = []
    for l in lines[data_start:]:
        s = l.strip()
        if not s:
            continue
        parts = split_row(s)
        if len(parts) < 2:
            continue
        try:
            _ = _parse_float_comma(parts[0])
            data_lines.append(s)
        except (ValueError, TypeError):
            continue
        if max_rows and len(data_lines) >= max_rows:
            break

    print(f"  4. Data rows: {len(data_lines):,}")

    # Load data: replace comma with dot, then use genfromtxt (some rows may have fewer cols)
    decoded = "\n".join(row.replace(",", ".") for row in data_lines)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        raw = np.genfromtxt(
            StringIO(decoded),
            delimiter=";",
            dtype=np.float64,
            invalid_raise=False,
            filling_values=np.nan,
        )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    print(f"  5. Raw matrix: {raw.shape[0]:,} x {raw.shape[1]}")

    # Extract per-channel data
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for ch_name, t_col, v_col, _fs in channel_specs:
        if t_col >= raw.shape[1] or v_col >= raw.shape[1]:
            continue
        times = raw[:, t_col].copy()
        values = raw[:, v_col].copy()
        valid = np.isfinite(times) & np.isfinite(values)
        times = times[valid]
        values = values[valid]
        if len(times) > 0:
            data[ch_name] = {"times": times, "values": values}

    # Truncate all channels to common time span (handles multi-rate EMG vs Acc)
    if data:
        t_starts = [d["times"][0] for d in data.values() if len(d["times"]) >= 2]
        t_ends = [d["times"][-1] for d in data.values() if len(d["times"]) >= 2]
        if t_starts and t_ends:
            t_min = max(t_starts)
            t_max = min(t_ends)
            if t_min < t_max:
                for ch_name in list(data.keys()):
                    d = data[ch_name]
                    t, v = d["times"], d["values"]
                    mask = (t >= t_min) & (t <= t_max)
                    n_keep = int(np.sum(mask))
                    if n_keep >= 2:
                        data[ch_name] = {"times": t[mask].copy(), "values": v[mask].copy()}
                    elif n_keep == 0:
                        del data[ch_name]

    # Update n_points in channel_infos
    channel_infos = [
        ChannelInfo(
            name=c.name,
            unit=c.unit,
            sampling_freq=c.sampling_freq,
            n_points=len(data[c.name]["times"]) if c.name in data else 0,
            x_start=c.x_start,
            domain_unit=c.domain_unit,
        )
        for c in channel_infos
    ]

    # Patient/task from filename
    stem = filepath.stem
    parts = stem.split("_")
    patient_id = "_".join(parts[:2]) if len(parts) >= 2 else stem
    task_name = "_".join(parts[2:]) if len(parts) > 2 else stem
    task_name = _normalize_task_name(task_name)
    task_type = "Task"
    session = "T0" if "T0" in task_name else "T1" if "T1" in task_name else ""
    condition = "SN" if "SN" in task_name.upper() else "DS" if "DS" in task_name.upper() else ""

    metadata = {}
    if len(lines) > 0:
        for line in lines[:5]:
            if ":" in line and ";" in line:
                kv = line.strip().split(":", 1)
                if len(kv) == 2:
                    metadata[kv[0].strip().rstrip(":")] = kv[1].strip().split(";")[0].strip()

    print(f"  6. Record: {patient_id} | {task_name}")
    return EMGRecord(
        patient_id=patient_id,
        task_name=task_name,
        task_type=task_type,
        session=session,
        condition=condition,
        channels=channel_infos,
        metadata=metadata,
        channel_settings=[],
        data=data,
        raw_matrix=raw if keep_raw else None,
        column_header=None,
    )


def parse_emg_csv(
    filepath: Path,
    *,
    keep_raw: bool = False,
    encoding: str = "utf-8",
    max_rows: Optional[int] = None,
) -> EMGRecord:
    """Parse a single EMG CSV file into an EMGRecord. Auto-detects Trigno Discover format."""
    with open(filepath, "r", encoding=encoding, errors="replace") as f:
        lines = f.readlines()
    if _is_trigno_discover_format(lines):
        return parse_trigno_discover_csv(filepath, keep_raw=keep_raw, encoding=encoding, max_rows=max_rows)

    print(f"\n--- Parsing {filepath.name} ---")
    print(f"  1. File read: {len(lines):,} lines")

    # 1. Parse channel labels (lines starting with "Label:")
    channels: List[ChannelInfo] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if not line.strip().startswith("Label:"):
            break
        m = LABEL_RE.match(line.strip())
        if m:
            channels.append(ChannelInfo(
                name=m.group(1).strip(),
                sampling_freq=float(m.group(2)),
                n_points=int(m.group(3)),
                x_start=float(m.group(4)),
                unit=m.group(5).strip(),
                domain_unit=m.group(6).strip(),
            ))
        i += 1

    print(f"  2. Channel labels: {len(channels)} channels (lines 1-{i})")
    if channels:
        fs_emg = [c.sampling_freq for c in channels if "EMG" in c.name and "Acc" not in c.name]
        fs_acc = [c.sampling_freq for c in channels if "Acc" in c.name or "ACC" in c.name]
        fs_gyro = [c.sampling_freq for c in channels if "GYRO" in c.name]
        print(f"     EMG fs: {fs_emg[0]:.1f} Hz x{len(fs_emg)} ch" if fs_emg else "     EMG: 0 ch")
        print(f"     Acc fs: {fs_acc[0]:.1f} Hz x{len(fs_acc)} ch" if fs_acc else "     Acc: 0 ch")
        print(f"     Gyro fs: {fs_gyro[0]:.1f} Hz x{len(fs_gyro)} ch" if fs_gyro else "     Gyro: 0 ch")

    # 2. Parse metadata (until "System Gain")
    metadata: Dict[str, str] = {}
    while i < len(lines) and "System Gain" not in lines[i]:
        line = lines[i]
        if ":" in line:
            parts = line.strip().split(":", 1)
            if len(parts) == 2:
                metadata[parts[0].strip()] = parts[1].strip()
        i += 1

    print(f"  3. Metadata: {len(metadata)} keys")
    for k, v in metadata.items():
        print(f"     {k}: {v}")

    # 3. Parse per-channel settings (System Gain, A/D card Gain, ...)
    channel_settings: List[Dict[str, float]] = []
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if stripped.startswith("System Gain"):
            ch_set: Dict[str, float] = {}
            for _ in range(6):
                if i < len(lines) and ":" in lines[i]:
                    k, v = lines[i].strip().split(":", 1)
                    ch_set[k.strip()] = _parse_float(v.strip())
                    i += 1
            channel_settings.append(ch_set)
        else:
            i += 1
            if i >= len(lines):
                break
            # Next line could be column header or data
            break

    print(f"  4. Channel settings: {len(channel_settings)} blocks (6 params each)")

    # 4. Find column header and data start (header is typically right after channel settings)
    col_header: Optional[str] = None
    data_start = None
    search_start = max(0, i - 2)
    for j in range(search_start, min(i + 20, len(lines))):
        row = lines[j].strip()
        if not row:
            continue
        parts = [p.strip() for p in row.split(",")]
        if len(parts) < 10:
            continue
        first = parts[0]
        if "X [s]" in first or (COL_HEADER_RE.search(row) and "[" in row):
            col_header = row
            data_start = j + 1
            break
        if DATA_ROW_RE.match(first):
            data_start = j
            break

    if data_start is None:
        raise ValueError(f"Could not find data section in {filepath}")
    print(f"  5. Data section: starts line {data_start} (1-based)")

    # 5. Parse column names from header if present
    channel_col_pairs: List[Tuple[str, int, int]] = []  # (name, t_col, v_col)
    if col_header and "[" in col_header:
        parts = col_header.split(",")
        col_idx = 0
        while col_idx + 1 < len(parts):
            # expect "X [s]" or time, then "ChannelName [Unit]"
            if col_idx < len(parts) and col_idx + 1 < len(parts):
                time_col = col_idx
                value_col = col_idx + 1
                val_part = parts[value_col].strip()
                if "[" in val_part:
                    name = val_part.split("[")[0].strip()
                    channel_col_pairs.append((name, time_col, value_col))
                col_idx += 2
            else:
                break
    else:
        # Infer from channels (same order as labels)
        for k, ch in enumerate(channels):
            t_col = 2 * k
            v_col = 2 * k + 1
            channel_col_pairs.append((ch.name, t_col, v_col))

    print(f"  6. Column mapping: {len(channel_col_pairs)} channel pairs (t_col, v_col)")

    # 6. Load numeric data (lines starting with scientific-notation number)
    data_lines = []
    for l in lines[data_start:]:
        s = l.strip()
        if not s:
            continue
        first = s.split(",")[0].strip()
        if DATA_ROW_RE.match(first):
            data_lines.append(s)

    if not data_lines:
        raise ValueError(f"No data rows found in {filepath}")

    if max_rows is not None:
        data_lines = data_lines[:max_rows]
        print(f"  7. Data rows: {len(data_lines):,} (limited from max_rows={max_rows})")
    else:
        print(f"  7. Data rows: {len(data_lines):,}")

    raw = np.genfromtxt(
        data_lines,
        delimiter=",",
        dtype=float,
        invalid_raise=False,
        filling_values=np.nan,
    )
    if raw.ndim == 1:
        raw = raw.reshape(1, -1)
    print(f"  8. Raw matrix: shape {raw.shape} = ({raw.shape[0]:,} rows x {raw.shape[1]} cols)")
    print(f"     dtype={raw.dtype}, min={np.nanmin(raw):.2e}, max={np.nanmax(raw):.2e}")

    # 7. Build per-channel (times, values) - extract unique samples for variable-rate channels
    data: Dict[str, Dict[str, np.ndarray]] = {}
    for ch_name, t_col, v_col in channel_col_pairs:
        if t_col >= raw.shape[1] or v_col >= raw.shape[1]:
            continue
        times = raw[:, t_col].copy()
        values = raw[:, v_col].copy()
        # Remove NaN rows
        valid = np.isfinite(times) & np.isfinite(values)
        times = times[valid]
        values = values[valid]
        # For variable-rate channels, keep only rows where time advances (optional)
        # Here we keep all; user can downsample if needed
        data[ch_name] = {"times": times, "values": values}

    print(f"  9. Per-channel data: {len(data)} channels extracted")
    for ki, (ch_name, d) in enumerate(list(data.items())[:3]):
        n = len(d["times"])
        print(f"     [{ki}] {ch_name}: {n:,} samples")
    if len(data) > 3:
        print(f"     ... +{len(data)-3} more channels")

    # 8. Extract patient/task from filename (e.g. CROSS_001_Task_T0_SN.csv)
    stem = filepath.stem
    parts = stem.split("_")
    patient_id = "_".join(parts[:2]) if len(parts) >= 2 else stem  # CROSS_001
    task_name = "_".join(parts[2:]) if len(parts) > 2 else stem   # Task_T0_SN
    task_type = parts[2] if len(parts) > 2 else ""                # Task | Calibrazione
    session = parts[3] if len(parts) > 3 else ""                  # T0 | T1
    condition = parts[4] if len(parts) > 4 else ""               # SN | DS

    print(f"  10. Record: patient={patient_id}, task={task_name} ({task_type} {session} {condition})")

    return EMGRecord(
        patient_id=patient_id,
        task_name=task_name,
        task_type=task_type,
        session=session,
        condition=condition,
        channels=channels,
        metadata=metadata,
        channel_settings=channel_settings,
        data=data,
        raw_matrix=raw if keep_raw else None,
        column_header=col_header,
    )


def _check_duration_consistency(rec: EMGRecord, tolerance_s: float = 0.01) -> None:
    """Verify all EMG and IMU channels have the same duration in seconds. Raises if not."""
    emg_ch = rec.get_emg_channels()
    imu_ch = rec.get_acc_channels()
    check_names = emg_ch + imu_ch
    if not check_names:
        return
    durations: List[Tuple[str, float]] = []
    for ch in check_names:
        d = rec.data.get(ch)
        if d is None:
            continue
        t = d["times"]
        if len(t) < 2:
            continue
        dur = float(t[-1] - t[0])
        durations.append((ch, dur))
    if not durations:
        return
    ref_dur = durations[0][1]
    bad = [(name, d) for name, d in durations if abs(d - ref_dur) > tolerance_s]
    if bad:
        lines = [f"  {name}: {d:.3f} s" for name, d in durations[:5]]
        if len(durations) > 5:
            lines.append(f"  ... (+{len(durations) - 5} more)")
        details = "\n".join(lines)
        raise ValueError(
            f"EMG and IMU channels have inconsistent durations (tolerance={tolerance_s}s). "
            f"Reference={ref_dur:.3f}s. Mismatched: {[(n, f'{d:.3f}s') for n, d in bad]}.\n"
            f"All channel durations:\n{details}"
        )


def _build_quality_meta(
    sub: EMGRecord,
    modality: str,
    source_file: str,
    sub_original: Optional[EMGRecord] = None,
    target_fs: Optional[float] = None,
) -> Dict:
    """Build lightweight metadata for quality checks (no data arrays).
    If sub_original is provided, includes duration, sampling_freq and n_samples before resampling."""
    sub_orig = sub_original if sub_original is not None else sub
    channels_info = []
    duration_s: Optional[float] = None
    n_samples_before: Optional[int] = None
    sampling_freqs_before: List[float] = []

    for c in sub.channels:
        d = sub.data.get(c.name)
        if d is None:
            continue
        t, v = d["times"], d["values"]
        n = len(t)
        duration_s_ch = float(t[-1] - t[0]) if n > 1 else 0.0
        if duration_s is None:
            duration_s = duration_s_ch
        v_fin = v[np.isfinite(v)]
        n_valid = len(v_fin)
        ch_info: Dict = {
            "name": c.name,
            "unit": c.unit,
            "sampling_freq_hz": round(c.sampling_freq, 2),
            "n_points_expected": c.n_points,
            "n_samples_actual": n,
            "duration_s": round(duration_s_ch, 3),
            "min": float(np.min(v_fin)) if n_valid else None,
            "max": float(np.max(v_fin)) if n_valid else None,
            "mean": float(np.mean(v_fin)) if n_valid else None,
            "std": float(np.std(v_fin)) if n_valid else None,
            "nan_ratio": round(float(np.sum(~np.isfinite(v)) / len(v)), 4) if len(v) > 0 else 0,
        }
        if sub_orig is not sub and sub_orig.data.get(c.name) is not None:
            d_orig = sub_orig.data[c.name]
            t_orig = d_orig["times"]
            n_b = len(t_orig)
            ch_info["sampling_freq_hz_before_resample"] = round(
                next(cc.sampling_freq for cc in sub_orig.channels if cc.name == c.name),
                2,
            )
            ch_info["n_samples_before_resample"] = n_b
            sampling_freqs_before.append(ch_info["sampling_freq_hz_before_resample"])
            if n_samples_before is None:
                n_samples_before = n_b
        channels_info.append(ch_info)

    out: Dict = {
        "patient_id": sub.patient_id,
        "task_name": sub.task_name,
        "task_type": sub.task_type,
        "session": sub.session,
        "condition": sub.condition,
        "modality": modality,
        "source_file": source_file,
        "n_channels": len(channels_info),
        "channels": channels_info,
        "acquisition_metadata": sub.metadata,
        "channel_settings_count": len(sub.channel_settings),
    }
    if duration_s is not None:
        out["duration_s"] = round(duration_s, 3)
    if sub_orig is not sub and n_samples_before is not None:
        out["n_samples_before_resample"] = n_samples_before
        out["sampling_freq_hz_before_resample"] = list(dict.fromkeys(sampling_freqs_before))
        out["sampling_freq_hz_after_resample"] = (
            round(target_fs, 2) if target_fs is not None
            else round(next(iter(sub.channels)).sampling_freq, 2)
        )
        first_ch = next(iter(sub.data.values()), None)
        if first_ch is not None:
            out["n_samples_after_resample"] = len(first_ch["times"])
    return out


def _resample_channel(times: np.ndarray, values: np.ndarray, orig_fs: float, target_fs: float) -> Tuple[np.ndarray, np.ndarray]:
    """Resample (times, values) to target_fs Hz. Returns (new_times, new_values)."""
    if len(times) < 2:
        return times.copy(), values.copy()
    duration = float(times[-1] - times[0])
    n_target = max(2, int(round(duration * target_fs)))
    new_values = resample(values, n_target)
    new_times = np.linspace(times[0], times[-1], n_target)
    return new_times.astype(np.float64), new_values.astype(np.float64)


def _resample_record(rec: EMGRecord, target_fs: float) -> EMGRecord:
    """Resample all channels to target_fs Hz. Updates channel metadata."""
    new_data: Dict[str, Dict[str, np.ndarray]] = {}
    new_channels: List[ChannelInfo] = []

    for ch in rec.channels:
        d = rec.data.get(ch.name)
        if d is None:
            continue
        t, v = d["times"], d["values"]
        t_new, v_new = _resample_channel(t, v, ch.sampling_freq, target_fs)
        new_data[ch.name] = {"times": t_new, "values": v_new}
        new_channels.append(ChannelInfo(
            name=ch.name,
            unit=ch.unit,
            sampling_freq=target_fs,
            n_points=len(t_new),
            x_start=ch.x_start,
            domain_unit=ch.domain_unit,
        ))

    return EMGRecord(
        patient_id=rec.patient_id,
        task_name=rec.task_name,
        task_type=rec.task_type,
        session=rec.session,
        condition=rec.condition,
        channels=new_channels,
        metadata=rec.metadata.copy(),
        channel_settings=rec.channel_settings,
        data=new_data,
        raw_matrix=None,
        column_header=rec.column_header,
    )


def _extract_subset(rec: EMGRecord, channel_names: List[str]) -> Optional[EMGRecord]:
    """Extract a subset of rec containing only the given channels. Returns None if empty."""
    if not channel_names:
        return None
    subset_data = {n: rec.data[n] for n in channel_names if n in rec.data}
    if not subset_data:
        return None
    subset_channels = [c for c in rec.channels if c.name in channel_names]
    idx = {c.name: i for i, c in enumerate(rec.channels)}
    subset_settings = [rec.channel_settings[idx[n]] for n in channel_names if n in idx and idx[n] < len(rec.channel_settings)]
    return EMGRecord(
        patient_id=rec.patient_id,
        task_name=rec.task_name,
        task_type=rec.task_type,
        session=rec.session,
        condition=rec.condition,
        channels=subset_channels,
        metadata=rec.metadata.copy(),
        channel_settings=subset_settings,
        data=subset_data,
        raw_matrix=None,
        column_header=None,
    )


def _plot_resample_check(
    rec: EMGRecord,
    sub_emg: Optional[EMGRecord],
    sub_imu: Optional[EMGRecord],
    out_path: Path,
    target_fs: float,
    n_emg: int = 3,
    n_imu: int = 3,
    seed: int = 42,
) -> None:
    """Plot 3 random EMG + 3 random IMU channels, before (left) and after (right) resampling."""
    if not HAS_MATPLOTLIB:
        return
    matplotlib.use("Agg")  # Non-interactive backend for saving; set here so import doesn't affect ginput in other scripts
    emg_names = rec.get_emg_channels()
    imu_names = rec.get_acc_channels()
    rng = random.Random(seed)
    sel_emg = rng.sample(emg_names, min(n_emg, len(emg_names))) if emg_names else []
    sel_imu = rng.sample(imu_names, min(n_imu, len(imu_names))) if imu_names else []
    if not sel_emg and not sel_imu:
        return
    channels = [(n, "emg") for n in sel_emg] + [(n, "imu") for n in sel_imu]
    if not channels:
        return
    if sub_emg is None and sub_imu is None:
        return

    n_rows = len(channels)
    fig, axes = plt.subplots(n_rows, 2, figsize=(12, 1.8 * n_rows), sharex="col")
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle(f"{rec.patient_id} | {rec.task_name}\nBefore (left) vs After (right) resampling", fontsize=10)

    for i, (ch_name, mod) in enumerate(channels):
        if mod == "emg":
            sub_resampled = sub_emg
        else:
            sub_resampled = sub_imu
        if sub_resampled is None or ch_name not in rec.data or ch_name not in sub_resampled.data:
            continue
        t_bef = rec.data[ch_name]["times"]
        v_bef = rec.data[ch_name]["values"]
        t_aft = sub_resampled.data[ch_name]["times"]
        v_aft = sub_resampled.data[ch_name]["values"]
        axes[i, 0].plot(t_bef, v_bef, linewidth=0.5, color="steelblue")
        axes[i, 0].set_ylabel(ch_name, fontsize=7)
        axes[i, 0].tick_params(labelsize=6)
        axes[i, 1].plot(t_aft, v_aft, linewidth=0.5, color="darkgreen")
        axes[i, 1].set_ylabel(ch_name, fontsize=7)
        axes[i, 1].tick_params(labelsize=6)

    axes[0, 0].set_title("Before resampling", fontsize=9)
    axes[0, 1].set_title(f"After resampling ({target_fs:.0f} Hz)", fontsize=9)
    for i in range(n_rows):
        axes[i, 0].set_xlabel("Time (s)" if i == n_rows - 1 else "")
        axes[i, 1].set_xlabel("Time (s)" if i == n_rows - 1 else "")
    plt.tight_layout()
    plt.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"    Saved resample check plot: {out_path.name}")


def discover_emg_files(emg_dir: Path, tasks_only: bool = True) -> List[Tuple[Path, str, str]]:
    """Discover EMG CSV files. Returns [(filepath, patient_id, task_name), ...].
    If tasks_only=True (default), skip Calibrazione files."""
    results: List[Tuple[Path, str, str]] = []
    emg_dir = Path(emg_dir)
    if not emg_dir.exists():
        return results
    for patient_folder in sorted(emg_dir.iterdir()):
        if not patient_folder.is_dir():
            continue
        for csv_file in sorted(patient_folder.glob("*.csv")):
            stem = csv_file.stem
            if tasks_only and "Calibrazione" in stem:
                continue
            parts = stem.split("_")
            patient_id = "_".join(parts[:2]) if len(parts) >= 2 else stem
            task_name = "_".join(parts[2:]) if len(parts) > 2 else stem
            results.append((csv_file, patient_id, task_name))
    return results


def run_parser(
    emg_dir: Path,
    out_dir: Path,
    *,
    format: str = "pkl",
    keep_raw: bool = False,
    max_rows: Optional[int] = None,
) -> Dict[str, Dict[str, EMGRecord]]:
    """
    Parse all EMG files and save structured records.
    Returns {patient_id: {task_name: EMGRecord}}.
    """
    files = discover_emg_files(emg_dir)
    print(f"\n=== Discovery: {len(files)} CSV files in {emg_dir} ===")
    all_records: Dict[str, Dict[str, EMGRecord]] = {}

    for fi, (filepath, patient_id, _task_from_file) in enumerate(files):
        print(f"\n--- File {fi+1}/{len(files)} ---")
        try:
            rec = parse_emg_csv(filepath, keep_raw=keep_raw, max_rows=max_rows)
        except Exception as e:
            print(f"WARN: Failed to parse {filepath}: {e}")
            continue
        _check_duration_consistency(rec)
        patient_id = rec.patient_id
        task_name = rec.task_name
        if patient_id not in all_records:
            all_records[patient_id] = {}
        all_records[patient_id][task_name] = rec

        # Write split output: emg, imu (acc), gyro separately
        out_patient = out_dir / f"{patient_id}_EMG"
        out_patient.mkdir(parents=True, exist_ok=True)

        emg_names = rec.get_emg_channels()
        imu_names = rec.get_acc_channels()

        if TARGET_FS == "imu":
            target_fs = _get_imu_sampling_freq(rec)
        else:
            target_fs = float(TARGET_FS)
        print(f"  Splitting: EMG={len(emg_names)}, IMU={len(imu_names)} channels (gyro ignored, resampling to {target_fs:.1f} Hz)")

        sub_emg_resampled: Optional[EMGRecord] = None
        sub_imu_resampled: Optional[EMGRecord] = None
        for suffix, names in [("emg", emg_names), ("imu", imu_names)]:
            sub_orig = _extract_subset(rec, names)
            if sub_orig is None:
                continue
            sub = _resample_record(sub_orig, target_fs=target_fs)
            if suffix == "emg":
                sub_emg_resampled = sub
            else:
                sub_imu_resampled = sub
            out_path = out_patient / f"{task_name}_{suffix}.{format}"
            meta_path = out_patient / f"{task_name}_{suffix}_meta.json"

            if format == "pkl":
                with open(out_path, "wb") as f:
                    pickle.dump(sub, f, protocol=pickle.HIGHEST_PROTOCOL)
            elif format == "npz":
                save_dict = {}
                for ch_name, d in sub.data.items():
                    safe_name = ch_name.replace(" ", "_").replace(":", "_")[:50]
                    save_dict[f"t_{safe_name}"] = d["times"]
                    save_dict[f"v_{safe_name}"] = d["values"]
                np.savez_compressed(out_path, **save_dict)
                meta_path = out_path.with_suffix(".meta.pkl")
                with open(meta_path, "wb") as f:
                    pickle.dump(
                        {
                            "patient_id": sub.patient_id,
                            "task_name": sub.task_name,
                            "channels": [(c.name, c.unit, c.sampling_freq) for c in sub.channels],
                            "metadata": sub.metadata,
                        },
                        f,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            else:
                raise ValueError(f"Unsupported format: {format}")

            # Save lightweight metadata for quality checks (no data)
            quality_meta = _build_quality_meta(
                sub, modality=suffix, source_file=filepath.name,
                sub_original=sub_orig, target_fs=target_fs,
            )
            with open(meta_path, "w", encoding="utf-8") as f:
                json.dump(quality_meta, f, indent=2, ensure_ascii=False)

            n_samples = len(next(iter(sub.data.values()))["times"]) if sub.data else 0
            print(f"    Saved {suffix}: {out_path.name} + {meta_path.name} ({len(names)} ch, {n_samples:,} samples)")

        plot_path = out_patient / f"{task_name}_resample_check.png"
        _plot_resample_check(
            rec, sub_emg_resampled, sub_imu_resampled, plot_path,
            target_fs=target_fs, n_emg=3, n_imu=3,
        )

    return all_records


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Parse EMG CSV files into structured patient_task data")
    ap.add_argument(
        "--emg-dir",
        type=Path,
        default=Path("data/emg"),
        help="Root directory containing patient EMG folders (e.g. data/emg/CROSS_001_EMG/)",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("data/emg_structured"),
        help="Output directory for structured files",
    )
    ap.add_argument(
        "--format",
        choices=["pkl", "npz"],
        default="pkl",
        help="Output format: pkl (full record) or npz (numpy arrays + separate meta)",
    )
    ap.add_argument(
        "--keep-raw",
        action="store_true",
        help="Keep full raw matrix in each record (increases size)",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="Limit data rows per file (for testing; default: all)",
    )
    args = ap.parse_args()

    print(f"Input dir:  {args.emg_dir.resolve()}")
    print(f"Output dir: {args.out.resolve()}")
    print(f"Format: {args.format}")
    if args.limit:
        print(f"Limit: {args.limit} rows per file")
    print()

    args.out.mkdir(parents=True, exist_ok=True)
    records = run_parser(
        args.emg_dir,
        args.out,
        format=args.format,
        keep_raw=args.keep_raw,
        max_rows=args.limit,
    )

    n_patients = len(records)
    n_tasks = sum(len(t) for t in records.values())
    print(f"\n=== Done ===")
    print(f"  Patients: {n_patients}")
    print(f"  Tasks: {n_tasks}")
    fs_desc = "IMU fs" if TARGET_FS == "imu" else f"{TARGET_FS} Hz"
    print(f"  Output files: ~{n_tasks * 2} (emg + imu per task, resampled to {fs_desc})")


if __name__ == "__main__":
    main()
