#!/usr/bin/env python3
"""batch_label_imu_segments.py

Loop over all patient/task IMU files, run manual segmentation for each that isn't done yet.
Saves segments and aligned-average figure in proper folders. Resume-friendly.

Usage:
  python batch_label_imu_segments.py
  python batch_label_imu_segments.py --data-dir data/emg_structured
  python batch_label_imu_segments.py --channel "Anterior Deltoid: Acc 5.X" "Brachioradialis: Acc 1.X"
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # Non-interactive for saving figures
import matplotlib.pyplot as plt
import numpy as np

# For pickle unpickling (pickles saved by parse_emg_patient_task run as __main__)
from B0_parse_emg_patient_task import ChannelInfo, EMGRecord
import sys
sys.modules["__main__"].ChannelInfo = ChannelInfo
sys.modules["__main__"].EMGRecord = EMGRecord


def discover_imu_files(data_dir: Path) -> List[tuple]:
    """Return list of (patient_id, task_name) for each *_imu.pkl found."""
    out: List[tuple] = []
    if not data_dir.exists():
        return out
    for pdir in sorted(data_dir.iterdir()):
        if not pdir.is_dir() or not pdir.name.endswith("_EMG"):
            continue
        patient_id = pdir.name.replace("_EMG", "")
        for pkl in sorted(pdir.glob("*_imu.pkl")):
            task_name = pkl.stem.replace("_imu", "")
            # Skip calibration
            if "Calibrazione" in task_name or "Calibrazione" in pkl.name:
                continue
            out.append((patient_id, task_name))
    return out


def is_already_done(data_dir: Path, patient_id: str, task_name: str) -> bool:
    """Check if manual segments exist for this patient/task."""
    seg_path = data_dir / f"{patient_id}_EMG" / f"{task_name}_manual_segments.json"
    return seg_path.exists()


def load_imu_record(patient_id: str, task_name: str, data_dir: Path) -> EMGRecord:
    pkl_path = data_dir / f"{patient_id}_EMG" / f"{task_name}_imu.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


MUSCLE_EN_IT = {
    "anterior deltoid": "deltoide anteriore",
    "middle deltoid": "deltoide medio",
    "posterior deltoid": "deltoide posteriore",
    "trapezius middle": "trapezio medio",
    "biceps long head": "bicipite brachiale capo lungo",
    "biceps short head": "bicipite capo corto",
    "triceps lateral": "tricipite brachiale capo laterale",
    "triceps long head": "tricipite brchiale capo lungo",
    "latissimus dorsi": "latissimus dorsi",
    "infraspinatus": "infraspinato",
    "pectoralis": "pettorale",
    "first dorsal interosseous": "primo interosseo",
    "teres major": "teres major",
    "brachioradialis": "brachioradialis",
}


def resolve_channel(spec: str, channels: List[str]) -> str:
    spec_lower = spec.lower()
    matches = [c for c in channels if spec_lower in c.lower()]
    if not matches:
        muscle = spec.split(":")[0].strip().lower()
        if muscle:
            matches = [c for c in channels if muscle in c.lower() and ("acc" in c.lower() or "emg" in c.lower())]
        if not matches and muscle:
            it_muscle = MUSCLE_EN_IT.get(muscle)
            if it_muscle:
                matches = [c for c in channels if it_muscle in c.lower() and ("acc" in c.lower() or "emg" in c.lower())]
    if not matches:
        raise ValueError(f"Channel '{spec}' not found. Available: {channels}")
    if len(matches) == 1:
        return matches[0]
    exact = next((c for c in matches if c == spec), None)
    if exact:
        return exact
    x_match = next((c for c in matches if ".x" in c.lower()), None)
    return x_match if x_match else matches[0]


def extract_and_align_segments(t, v, pairs, n_pts=101):
    pct = np.linspace(0, 100, n_pts)
    stacked = []
    for start_s, end_s in pairs:
        mask = (t >= start_s) & (t <= end_s)
        if not np.any(mask):
            continue
        t_seg, v_seg = t[mask], v[mask]
        if len(t_seg) < 2:
            continue
        t_norm = (t_seg - t_seg[0]) / (t_seg[-1] - t_seg[0]) * 100
        stacked.append(np.interp(pct, t_norm, v_seg))
    if not stacked:
        raise ValueError("No valid segments.")
    stacked = np.array(stacked)
    return pct, stacked, np.mean(stacked, axis=0), np.std(stacked, axis=0)


def save_aligned_average_figure(
    data_dir: Path,
    patient_id: str,
    task_name: str,
    channels_spec: List[str],
    pairs: List[tuple],
    n_pts: int = 101,
) -> Path:
    """Generate and save aligned average PNG. Returns path to saved file."""
    rec = load_imu_record(patient_id, task_name, data_dir)
    all_channels = sorted(rec.data.keys())
    out_dir = data_dir / f"{patient_id}_EMG"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{task_name}_aligned_average.png"

    n_ch = len(channels_spec)
    fig, axes = plt.subplots(n_ch, 1, figsize=(10, 4 * n_ch), sharex=True, squeeze=False)
    axes = axes[:, 0]
    for ax, spec in zip(axes, channels_spec):
        ch_name = resolve_channel(spec, all_channels)
        t = rec.data[ch_name]["times"]
        v = rec.data[ch_name]["values"]
        pct, stacked, mean, std = extract_and_align_segments(t, v, pairs, n_pts)
        ax.plot(pct, mean, color="steelblue", linewidth=2, label="Mean")
        ax.fill_between(pct, mean - std, mean + std, alpha=0.3, color="steelblue")
        ax.set_ylabel(ch_name)
        ax.legend(loc="upper right")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{patient_id} | {task_name} | {ch_name} (n={stacked.shape[0]} reps)")
    axes[-1].set_xlabel("Cycle (%)")
    fig.suptitle(f"{patient_id} | {task_name} | Aligned average (0-100%)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


def run_manual_for(data_dir: Path, patient_id: str, task_name: str, channels: List[str]) -> bool:
    """Run plot_imu_peaks --manual. Returns True if completed successfully."""
    script = Path(__file__).parent / "plot_imu_peaks.py"
    cmd = [
        sys.executable,
        str(script),
        "--manual",
        "--patient", patient_id,
        "--task", task_name,
        "--data-dir", str(data_dir),
        "--channel",
    ] + channels
    ret = subprocess.run(cmd)
    return ret.returncode == 0


def main():
    ap = argparse.ArgumentParser(
        description="Batch manual segmentation for all IMU files. Skips already done."
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/emg_structured"),
        help="Structured data root (default: data/emg_structured)",
    )
    ap.add_argument(
        "--channel",
        type=str,
        nargs="*",
        default=["Anterior Deltoid: Acc 5.X", "Brachioradialis: Acc 1.X"],
        help="Two channels for manual mode (default: Anterior Deltoid, Brachioradialis)",
    )
    ap.add_argument(
        "--no-save-fig",
        action="store_true",
        help="Do not save aligned average PNG after each label",
    )
    ap.add_argument(
        "--force",
        action="store_true",
        help="Re-label even if already done",
    )
    args = ap.parse_args()

    if len(args.channel) != 2:
        raise ValueError("--channel must specify exactly 2 channels (top=starts, bottom=ends).")

    files = discover_imu_files(args.data_dir)
    if not files:
        print("No IMU files found.")
        return

    todo = []
    for patient_id, task_name in files:
        if is_already_done(args.data_dir, patient_id, task_name) and not args.force:
            print(f"[SKIP] {patient_id} | {task_name} (already done)")
        else:
            todo.append((patient_id, task_name))

    print(f"\n{len(todo)} to label, {len(files) - len(todo)} already done.\n")

    for i, (patient_id, task_name) in enumerate(todo):
        print(f"\n--- [{i+1}/{len(todo)}] {patient_id} | {task_name} ---")
        ok = run_manual_for(args.data_dir, patient_id, task_name, args.channel)
        if not ok:
            print("Process exited with error. Stopping.")
            return
        if not args.no_save_fig:
            seg_path = args.data_dir / f"{patient_id}_EMG" / f"{task_name}_manual_segments.json"
            if seg_path.exists():
                with open(seg_path) as f:
                    seg = json.load(f)
                pairs = [(p["start_s"], p["end_s"]) for p in seg["pairs"]]
                ch = seg.get("channels", args.channel)
                png = save_aligned_average_figure(args.data_dir, patient_id, task_name, ch, pairs)
                print(f"Saved {png}")

    print("\nDone.")


if __name__ == "__main__":
    main()
