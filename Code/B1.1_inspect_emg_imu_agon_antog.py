#!/usr/bin/env python3
"""plot_patient_deltoid_summary.py

Produce figures per patient with aligned average (± std) across the four
repetitions: IMU X, Y, Z (left column) and EMG envelope (right column).

Figure types:
  1. Deltoid: Anterior Deltoid, Posterior Deltoid
  2. Arm: Biceps Brachii Long Head, Triceps Brachii Lateral Head
  3. Shoulder/chest: Middle Deltoid, Pectoralis Major

Usage:
  python B1.1_inspect_emg_imu_agon_antog.py --data-dir data/emg_structured
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
from scipy.signal import butter, filtfilt

# For pickle unpickling (use B2 module to match pipeline)
import B0_parse_emg_patient_task as parse_emg_patient_task
sys.modules["__main__"].ChannelInfo = parse_emg_patient_task.ChannelInfo
sys.modules["__main__"].EMGRecord = parse_emg_patient_task.EMGRecord

from B0_parse_emg_patient_task import EMGRecord

# Preferred figure types shown first: (muscle_specs, title_suffix, filename_suffix)
PREFERRED_MUSCLE_GROUPS = [
    (
        [("Anterior Deltoid", [".X", ".Y", ".Z"]), ("Posterior Deltoid", [".X", ".Y", ".Z"])],
        "Anterior & Posterior Deltoid",
        "deltoid",
    ),
    (
        [("Biceps Brachii Long Head", [".X", ".Y", ".Z"]), ("Triceps Brachii Lateral Head", [".X", ".Y", ".Z"])],
        "Biceps & Triceps Brachii",
        "biceps_triceps",
    ),
    (
        [("Middle Deltoid", [".X", ".Y", ".Z"]), ("Pectoralis Major", [".X", ".Y", ".Z"])],
        "Middle Deltoid & Pectoralis Major",
        "middle_deltoid_pectoralis",
    ),
]
N_PTS = 101
ENVELOPE_LOWPASS_HZ = 6
ENVELOPE_ORDER = 4
EMG_SCALE_TO_UV = 1e6
EMG_YLIM_UV = (0.0, 50.0)  # 0 to 5e-5 V expressed in microvolts


def load_imu_record(patient_id: str, task_name: str, data_dir: Path) -> EMGRecord:
    pkl_path = data_dir / f"{patient_id}_EMG" / f"{task_name}_imu.pkl"
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_emg_record(patient_id: str, task_name: str, data_dir: Path) -> Optional[EMGRecord]:
    pkl_path = data_dir / f"{patient_id}_EMG" / f"{task_name}_emg.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def resolve_channel(spec: str, channels: List[str], prefer_x: bool = True) -> Optional[str]:
    """Resolve partial spec to full channel name."""
    spec_lower = spec.lower()
    matches = [c for c in channels if spec_lower in c.lower()]
    if not matches:
        muscle = spec.split(":")[0].strip().lower()
        if muscle:
            matches = [c for c in channels if muscle in c.lower() and ("acc" in c.lower() or "emg" in c.lower())]
    if not matches:
        return None
    if len(matches) == 1:
        return matches[0]
    if prefer_x:
        x_match = next((c for c in matches if ".x" in c.lower() or "acc.x" in c.lower()), None)
        if x_match:
            return x_match
    return matches[0]


def emg_envelope(emg: np.ndarray, fs: float, lowpass_hz: float = ENVELOPE_LOWPASS_HZ, order: int = ENVELOPE_ORDER) -> np.ndarray:
    """Rectify and low-pass filter EMG to get envelope."""
    rectified = np.abs(emg)
    nyq = fs / 2
    low = lowpass_hz / nyq
    b, a = butter(order, low, btype="low")
    return filtfilt(b, a, rectified)


def extract_and_align(
    t: np.ndarray,
    v: np.ndarray,
    pairs: List[Tuple[float, float]],
    n_pts: int = N_PTS,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Extract segments, resample to 0-100%, return (pct, mean, std)."""
    pct = np.linspace(0, 100, n_pts)
    stacked = []
    for start_s, end_s in pairs:
        mask = (t >= start_s) & (t <= end_s)
        if not np.any(mask):
            continue
        t_seg = t[mask]
        v_seg = v[mask]
        if len(t_seg) < 2:
            continue
        t_norm = (t_seg - t_seg[0]) / (t_seg[-1] - t_seg[0]) * 100
        idx = np.clip(np.searchsorted(t_norm, pct, side="right") - 1, 0, len(v_seg) - 1)
        stacked.append(v_seg[idx])
    if not stacked:
        raise ValueError("No valid segments.")
    stacked = np.array(stacked)
    return pct, np.mean(stacked, axis=0), np.std(stacked, axis=0)


def imu_channel_to_emg(imu_ch: str, emg_channels: List[str]) -> Optional[str]:
    """Map IMU channel to EMG channel by muscle/sensor number."""
    import re
    # Match "Anterior Deltoid: Acc 5.X" or "Trapezius Middle: ACC.X 14"
    m = re.search(r"^(.+?):\s*Acc[\.\s]*(\d+)", imu_ch, re.IGNORECASE)
    if not m:
        m = re.search(r"^(.+?):\s*Acc\.X\s*(\d+)", imu_ch, re.IGNORECASE)
    if not m:
        return None
    muscle, num = m.group(1).strip(), m.group(2)
    pattern = f"{muscle}: EMG {num}"
    for ch in emg_channels:
        if pattern.lower() in ch.lower() or ch == pattern:
            return ch
    return None


def get_fs_from_record(rec: EMGRecord) -> float:
    """Get sampling frequency from first channel."""
    if not rec.channels:
        return 150.0
    return rec.channels[0].sampling_freq


def discover_patients_with_segments(data_dir: Path) -> List[Tuple[str, List[str]]]:
    """Return [(patient_id, [task_name, ...]), ...] for patients that have manual segments."""
    out: List[Tuple[str, List[str]]] = []
    if not data_dir.exists():
        return out
    for pdir in sorted(data_dir.iterdir()):
        if not pdir.is_dir() or not pdir.name.endswith("_EMG"):
            continue
        patient_id = pdir.name.replace("_EMG", "")
        tasks = []
        for seg in sorted(pdir.glob("*_manual_segments.json")):
            task_name = seg.stem.replace("_manual_segments", "")
            if "Calibrazione" in task_name:
                continue
            tasks.append(task_name)
        if tasks:
            out.append((patient_id, tasks))
    return out


def _resolve_imu_axis(imu_chans: List[str], muscle: str, suf: str) -> Optional[str]:
    """Resolve IMU channel for muscle and axis (.X, .Y, .Z)."""
    matches = [c for c in imu_chans if muscle.lower() in c.lower()]
    if suf == ".X":
        matches = [c for c in matches if ".x" in c.lower() or "acc.x" in c.lower()]
    elif suf == ".Y":
        matches = [c for c in matches if ".y" in c.lower()]
    elif suf == ".Z":
        matches = [c for c in matches if ".z" in c.lower()]
    return matches[0] if matches else None


def slugify(text: str) -> str:
    """Filesystem-safe slug."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", text.strip().lower()).strip("_")


def chunk_list(items: List[str], chunk_size: int) -> List[List[str]]:
    """Split list into fixed-size chunks."""
    return [items[i: i + chunk_size] for i in range(0, len(items), chunk_size)]


def discover_available_muscles(
    data_dir: Path,
    patient_id: str,
    tasks: List[str],
) -> List[str]:
    """Discover muscles that have IMU+EMG or EMG-only channels."""
    imu_emg_muscles: set[str] = set()
    emg_only_muscles: set[str] = set()
    for task_name in tasks:
        try:
            imu_rec = load_imu_record(patient_id, task_name, data_dir)
            emg_rec = load_emg_record(patient_id, task_name, data_dir)
        except Exception:
            continue
        if emg_rec is None:
            continue
        imu_chans = sorted(imu_rec.data.keys()) if imu_rec else []
        emg_chans = sorted(emg_rec.data.keys())
        for ch in emg_chans:
            if ":" not in ch or "emg" not in ch.lower():
                continue
            muscle = ch.split(":")[0].strip()
            if not muscle:
                continue
            has_imu_axes = (
                imu_rec is not None
                and all(_resolve_imu_axis(imu_chans, muscle, suf) for suf in [".X", ".Y", ".Z"])
            )
            if has_imu_axes:
                imu_emg_muscles.add(muscle)
            else:
                emg_only_muscles.add(muscle)
    combined = sorted(imu_emg_muscles | emg_only_muscles)
    return combined


def _muscle_matches(muscle: str, available: List[str]) -> Optional[str]:
    """Case-insensitive match: return the available name if muscle matches, else None."""
    ml = muscle.strip().lower()
    for a in available:
        if a.strip().lower() == ml:
            return a
    return None


def build_muscle_groups(
    data_dir: Path,
    patient_id: str,
    tasks: List[str],
) -> List[Tuple[List[Tuple[str, List[str]]], str, str]]:
    """Use preferred figure groups first, then add remaining muscles in pairs."""
    available = discover_available_muscles(data_dir, patient_id, tasks)
    if not available:
        return PREFERRED_MUSCLE_GROUPS

    groups: List[Tuple[List[Tuple[str, List[str]]], str, str]] = []
    used: set[str] = set()

    for muscle_specs, title_suffix, filename_suffix in PREFERRED_MUSCLE_GROUPS:
        present_specs = []
        for muscle, axes in muscle_specs:
            matched = _muscle_matches(muscle, available)
            if matched:
                present_specs.append((matched, axes))  # use actual channel name for plotting
        if present_specs:
            present_names = [muscle for muscle, _ in present_specs]
            dynamic_title = title_suffix if len(present_names) == len(muscle_specs) else " & ".join(present_names)
            dynamic_slug = filename_suffix if len(present_names) == len(muscle_specs) else slugify("_".join(present_names))
            groups.append((present_specs, dynamic_title, dynamic_slug))
            used.update(muscle for muscle, _ in present_specs)

    remaining = [muscle for muscle in available if muscle not in used]
    for chunk in chunk_list(remaining, 2):
        muscle_specs = [(muscle, [".X", ".Y", ".Z"]) for muscle in chunk]
        title_suffix = " & ".join(chunk)
        filename_suffix = slugify("_".join(chunk))
        groups.append((muscle_specs, title_suffix, filename_suffix))

    return groups


def plot_patient_figure(
    data_dir: Path,
    patient_id: str,
    tasks: List[str],
    out_path: Path,
    muscle_specs: List[Tuple[str, List[str]]],
    title_suffix: str,
) -> Path:
    """Create one figure: IMU X,Y,Z combined (left) and EMG envelope (right) per muscle."""
    n_tasks = len(tasks)
    n_rows = len(muscle_specs)
    n_cols = 2  # IMU, EMG
    std_alpha = 0.55  # stronger std shading

    fig = plt.figure(figsize=(12, 4 * n_tasks * n_rows))
    gs = GridSpec(n_tasks * n_rows, n_cols, hspace=0.4, wspace=0.25)

    colors_xyz = ["#e41a1c", "#377eb8", "#4daf4a"]  # X, Y, Z
    # Anchor for shared EMG y-axis: all panels use 0–50 µV
    anchor_emg = fig.add_axes([0, 0, 0.01, 0.01])
    anchor_emg.set_ylim(*EMG_YLIM_UV)
    anchor_emg.set_visible(False)
    shared_emg_ax = anchor_emg

    for ti, task_name in enumerate(tasks):
        seg_path = data_dir / f"{patient_id}_EMG" / f"{task_name}_manual_segments.json"
        if not seg_path.exists():
            continue
        with open(seg_path) as f:
            seg = json.load(f)
        pairs = [(p["start_s"], p["end_s"]) for p in seg["pairs"]]

        try:
            imu_rec = load_imu_record(patient_id, task_name, data_dir)
            imu_chans = sorted(imu_rec.data.keys())
        except Exception:
            imu_rec = None
            imu_chans = []
        emg_rec = load_emg_record(patient_id, task_name, data_dir)
        emg_chans = sorted(emg_rec.data.keys()) if emg_rec else []
        fs_emg = get_fs_from_record(emg_rec) if emg_rec else 150.0

        for ri, (muscle, axes_suffix) in enumerate(muscle_specs):
            row = ti * n_rows + ri
            ax_imu = fig.add_subplot(gs[row, 0])
            # Plot X, Y, Z in same panel
            has_imu = False
            for ai, suf in enumerate(axes_suffix):
                imu_ch = _resolve_imu_axis(imu_chans, muscle, suf)
                if not imu_ch:
                    continue
                has_imu = True
                t = imu_rec.data[imu_ch]["times"]
                v = imu_rec.data[imu_ch]["values"]
                pct, mean, std = extract_and_align(t, v, pairs)
                lbl = suf[1]  # "X", "Y", "Z"
                ax_imu.plot(pct, mean, color=colors_xyz[ai], linewidth=1.5, label=lbl)
                ax_imu.fill_between(pct, mean - std, mean + std, alpha=std_alpha, color=colors_xyz[ai])
            if not has_imu:
                ax_imu.set_visible(False)

            # EMG envelope in right column: resolve EMG channel (IMU mapping or muscle name)
            imu_ch_x = _resolve_imu_axis(imu_chans, muscle, ".X")
            emg_ch = imu_channel_to_emg(imu_ch_x, emg_chans) if imu_ch_x and emg_chans else None
            if not emg_ch and emg_chans:
                emg_ch = next((c for c in emg_chans if muscle.lower() in c.lower() and "emg" in c.lower()), None)
            if emg_ch and emg_rec:
                ax_emg = fig.add_subplot(gs[row, 1], sharey=shared_emg_ax)
                te = emg_rec.data[emg_ch]["times"]
                ve = emg_rec.data[emg_ch]["values"]
                env = emg_envelope(ve, fs_emg) * EMG_SCALE_TO_UV  # V -> µV
                pct_e, mean_e, std_e = extract_and_align(te, env, pairs)
                ax_emg.plot(pct_e, mean_e, color="darkgreen", linewidth=1.5)
                ax_emg.fill_between(pct_e, mean_e - std_e, mean_e + std_e, alpha=std_alpha, color="darkgreen")
                ax_emg.set_ylabel(f"{muscle}\nEMG (µV)", fontsize=9)
                ax_emg.tick_params(labelsize=7)
                ax_emg.set_xticks([])
                ax_emg.spines["top"].set_visible(False)
                ax_emg.spines["right"].set_visible(False)
                if ri == 0 and not has_imu:
                    ax_emg.set_title(f"{task_name}", fontsize=9)
            else:
                ax_emg = fig.add_subplot(gs[row, 1])
                ax_emg.set_visible(False)

            if has_imu:
                ax_imu.set_ylabel(f"{muscle}\nIMU", fontsize=9)
                ax_imu.tick_params(labelsize=7)
                if ri == 0:
                    ax_imu.set_title(f"{task_name}", fontsize=9)
                ax_imu.set_xticks([])
                ax_imu.spines["top"].set_visible(False)
                ax_imu.spines["right"].set_visible(False)
                ax_imu.legend(loc="upper right", fontsize=7)

    # Set xlabel on bottom row axes (last row of last task)
    for ax in fig.get_axes():
        try:
            ss = ax.get_subplotspec()
            if ss.rowspan.stop == n_tasks * n_rows:
                ax.set_xlabel("Cycle (%)")
        except Exception:
            pass

    n_reps = 4  # reps per task (from manual labeling)
    fig.suptitle(f"{patient_id} | {title_suffix} IMU + EMG envelope (n={n_reps} reps per task)", fontsize=11)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Plot patient muscle summaries: IMU X,Y,Z + EMG envelope")
    ap.add_argument("--data-dir", type=Path, default=Path("data/emg_structured"))
    ap.add_argument("--out-dir", type=Path, default=Path("results/figures/emg_imu_agon_antog"), help="Output directory for IMU+EMG summary figures")
    args = ap.parse_args()

    patients = discover_patients_with_segments(args.data_dir)
    if not patients:
        print("No patients with manual segments found.")
        return

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for patient_id, tasks in patients:
        muscle_groups = build_muscle_groups(args.data_dir, patient_id, tasks)
        for muscle_specs, title_suffix, filename_suffix in muscle_groups:
            out_path = args.out_dir / f"{patient_id}_{filename_suffix}_summary.png"
            try:
                plot_patient_figure(
                    args.data_dir,
                    patient_id,
                    tasks,
                    out_path,
                    muscle_specs=muscle_specs,
                    title_suffix=title_suffix,
                )
                print(f"Saved {out_path}")
            except Exception as e:
                print(f"Error {patient_id} {filename_suffix}: {e}")
                import traceback
                traceback.print_exc()

    n_total_figures = sum(len(build_muscle_groups(args.data_dir, patient_id, tasks)) for patient_id, tasks in patients)
    print(f"\nDone. {len(patients)} patients, {n_total_figures} figures in {args.out_dir}")


if __name__ == "__main__":
    main()
