#!/usr/bin/env python3
"""plot_imu_smv.py

Load IMU data, align with manual segments, filter, compute SMV (Signal Magnitude Vector),
and plot the averaged SMV across cycles. Iterates over all patients and tasks that have
manual segments.

SMV = sqrt(Acc_X^2 + Acc_Y^2 + Acc_Z^2) per IMU sensor.

Usage:
  python B1.3_inspect_imu_smv.py --data-dir data/emg_structured
  python B1.3_inspect_imu_smv.py --data-dir data/emg_structured --out-dir results/figures/imu_smv --patients CROSS_001 --lowpass-hz 10
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

# Pickle compatibility (use B2 module to match pipeline)
import B0_parse_emg_patient_task
sys.modules["__main__"].ChannelInfo = B0_parse_emg_patient_task.ChannelInfo
sys.modules["__main__"].EMGRecord = B0_parse_emg_patient_task.EMGRecord
from B0_parse_emg_patient_task import EMGRecord


# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

DEFAULT_LOWPASS_HZ = 10.0
DEFAULT_LOWPASS_ORDER = 4
N_PTS = 101


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------


def load_imu_record(patient_id: str, task_name: str, data_dir: Path) -> Optional[EMGRecord]:
    pkl_path = data_dir / f"{patient_id}_EMG" / f"{task_name}_imu.pkl"
    if not pkl_path.exists():
        return None
    try:
        with open(pkl_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(f"  Failed to load {pkl_path}: {e}")
        return None


def load_manual_segments(patient_id: str, task_name: str, data_dir: Path) -> Optional[List[Tuple[float, float]]]:
    path = data_dir / f"{patient_id}_EMG" / f"{task_name}_manual_segments.json"
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        pairs = data.get("pairs", [])
        return [(p["start_s"], p["end_s"]) for p in pairs]
    except Exception as e:
        print(f"  Failed to load segments {path}: {e}")
        return None


def get_fs_from_record(rec: EMGRecord) -> float:
    if not rec.channels:
        return 150.0
    return float(rec.channels[0].sampling_freq)


def discover_patients_with_data(data_dir: Path) -> List[str]:
    if not data_dir.exists():
        return []
    return sorted(
        pdir.name.replace("_EMG", "")
        for pdir in data_dir.iterdir()
        if pdir.is_dir() and pdir.name.endswith("_EMG")
    )


def discover_tasks_with_segments(patient_id: str, data_dir: Path) -> List[str]:
    """Return task names that have both _imu.pkl and _manual_segments.json."""
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


# -----------------------------------------------------------------------------
# IMU sensor grouping
# -----------------------------------------------------------------------------


def group_acc_channels_by_sensor(acc_channels: List[str]) -> Dict[str, Tuple[str, str, str]]:
    """
    Group Acc channels by sensor. Returns {sensor_key: (ch_x, ch_y, ch_z)}.
    sensor_key = e.g. "Anterior Deltoid: Acc 5"
    """
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
# Filtering and SMV
# -----------------------------------------------------------------------------


def filter_lowpass(
    data: np.ndarray,
    fs: float,
    cutoff_hz: float = DEFAULT_LOWPASS_HZ,
    order: int = DEFAULT_LOWPASS_ORDER,
) -> np.ndarray:
    """Butterworth lowpass filter."""
    if len(data) < 10 or cutoff_hz <= 0:
        return np.asarray(data, dtype=np.float64)
    nyq = fs / 2
    cutoff = min(cutoff_hz, nyq * 0.99)
    if cutoff < 0.5:
        cutoff = 0.5
    b, a = butter(order, cutoff, btype="low", fs=fs)
    return filtfilt(b, a, np.asarray(data, dtype=np.float64))


def compute_smv(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> np.ndarray:
    """SMV = sqrt(X^2 + Y^2 + Z^2)."""
    n = min(len(x), len(y), len(z))
    return np.sqrt(x[:n] ** 2 + y[:n] ** 2 + z[:n] ** 2)


def extract_and_align(
    t: np.ndarray,
    v: np.ndarray,
    pairs: List[Tuple[float, float]],
    n_pts: int = N_PTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract segments, interpolate to 0-100% cycle phase, return (pct, mean, std)."""
    pct = np.linspace(0, 100, n_pts)
    phase_new = pct / 100.0
    stacked = []
    for start_s, end_s in pairs:
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
    if not stacked:
        raise ValueError("No valid segments.")
    stacked = np.asarray(stacked)
    return pct, np.mean(stacked, axis=0), np.std(stacked, axis=0)


# -----------------------------------------------------------------------------
# Process and plot
# -----------------------------------------------------------------------------


def process_patient_task_smv(
    patient_id: str,
    task_name: str,
    data_dir: Path,
    out_path: Path,
    lowpass_hz: float = DEFAULT_LOWPASS_HZ,
    lowpass_order: int = DEFAULT_LOWPASS_ORDER,
    dpi: int = 150,
) -> bool:
    """
    Load IMU, align with manual segments, filter, compute SMV, plot.
    Returns True if successful.
    """
    rec = load_imu_record(patient_id, task_name, data_dir)
    if rec is None:
        return False

    pairs = load_manual_segments(patient_id, task_name, data_dir)
    if not pairs:
        print(f"  {task_name}: no manual segments")
        return False

    acc_chans = rec.get_acc_channels()
    if not acc_chans:
        print(f"  {task_name}: no Acc channels")
        return False

    sensor_groups = group_acc_channels_by_sensor(acc_chans)
    if not sensor_groups:
        print(f"  {task_name}: could not group Acc X,Y,Z by sensor")
        return False

    fs = get_fs_from_record(rec)

    n_sensors = len(sensor_groups)
    n_cols = min(3, n_sensors)
    n_rows = (n_sensors + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharex=True)
    axes = np.atleast_2d(axes)
    if n_sensors < n_rows * n_cols:
        for j in range(n_sensors, n_rows * n_cols):
            axes.flat[j].set_visible(False)

    for idx, (sensor_label, (ch_x, ch_y, ch_z)) in enumerate(sensor_groups.items()):
        ax = axes.flat[idx]
        tx = rec.data[ch_x]["times"]
        vx = rec.data[ch_x]["values"]
        ty = rec.data[ch_y]["times"]
        vy = rec.data[ch_y]["values"]
        tz = rec.data[ch_z]["times"]
        vz = rec.data[ch_z]["values"]

        vx_f = filter_lowpass(vx, fs, lowpass_hz, lowpass_order)
        vy_f = filter_lowpass(vy, fs, lowpass_hz, lowpass_order)
        vz_f = filter_lowpass(vz, fs, lowpass_hz, lowpass_order)

        t_ref = tx
        smv = compute_smv(vx_f, vy_f, vz_f)

        try:
            pct, mean_smv, std_smv = extract_and_align(t_ref[: len(smv)], smv, pairs)
        except ValueError:
            ax.text(0.5, 0.5, "No valid segments", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(sensor_label, fontsize=9)
            continue

        ax.plot(pct, mean_smv, color="steelblue", linewidth=1.5)
        ax.fill_between(pct, mean_smv - std_smv, mean_smv + std_smv, alpha=0.4, color="steelblue")
        ax.axhline(1.0, linestyle="--", color="gray", alpha=0.5, linewidth=0.8)
        ax.set_ylabel("SMV (g)")
        ax.set_title(sensor_label, fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    for ax in axes.flat[:n_sensors]:
        if ax.get_xlabel() == "" and ax in axes[-1, :] if n_rows > 1 else True:
            ax.set_xlabel("Cycle (%)")

    fig.suptitle(f"{patient_id} | {task_name} | SMV (lowpass {lowpass_hz} Hz, n={len(pairs)} reps)", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(
        description="Plot IMU SMV aligned with manual segments for all patients"
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data/emg_structured"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures/imu_smv"), help="Output directory for SMV figures")
    ap.add_argument("--patients", type=str, nargs="*", help="Limit to these patient IDs")
    ap.add_argument("--lowpass-hz", type=float, default=DEFAULT_LOWPASS_HZ)
    ap.add_argument("--lowpass-order", type=int, default=DEFAULT_LOWPASS_ORDER)
    ap.add_argument("--dpi", type=int, default=150)
    ap.add_argument("--max-patients", type=int, default=None)
    ap.add_argument("--max-tasks", type=int, default=None)
    args = ap.parse_args()

    patients = discover_patients_with_data(args.data_dir)
    if args.patients:
        patients = [p for p in patients if p in args.patients]
    if args.max_patients:
        patients = patients[: args.max_patients]

    args.out_dir.mkdir(parents=True, exist_ok=True)
    n_ok, n_fail = 0, 0

    for patient_id in patients:
        print(f"Patient {patient_id}")
        tasks = discover_tasks_with_segments(patient_id, args.data_dir)
        if args.max_tasks:
            tasks = tasks[: args.max_tasks]
        for task_name in tasks:
            out_path = args.out_dir / patient_id / f"{task_name}_smv.png"
            try:
                ok = process_patient_task_smv(
                    patient_id,
                    task_name,
                    args.data_dir,
                    out_path,
                    lowpass_hz=args.lowpass_hz,
                    lowpass_order=args.lowpass_order,
                    dpi=args.dpi,
                )
                if ok:
                    print(f"  {task_name}: saved {out_path}")
                    n_ok += 1
                else:
                    n_fail += 1
            except Exception as e:
                print(f"  {task_name}: error {e}")
                import traceback
                traceback.print_exc()
                n_fail += 1

    print(f"\nDone. {n_ok} figures saved, {n_fail} failed. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
