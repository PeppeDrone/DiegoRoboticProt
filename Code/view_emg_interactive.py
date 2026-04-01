#!/usr/bin/env python3
"""view_emg_interactive.py

Load EMG and IMU data for a specified patient and task, and display an
interactive figure with both. EMG: 7x2 layout. IMU: 7x6 layout (triple-width
columns for X, Y, Z per sensor). Use the matplotlib toolbar to pan, zoom, save.

Usage:
  python view_emg_interactive.py --patient CROSS_001 --task Task_T0_SN
  python view_emg_interactive.py --patient CROSS_001 --task Task_T0_SN --t-start 10 --t-end 30

If structured pkl files don't exist, run parse_emg_patient_task.py first.

Optional: pip install neurokit2 for muscle activation panel in per-muscle figures.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import butter, filtfilt

try:
    import neurokit2 as nk
    _HAS_NEUROKIT = True
except ImportError:
    _HAS_NEUROKIT = False

# Import for fallback CSV parsing, crop_to_time_window, and pickle unpickling
# (pkl files reference ChannelInfo, EMGRecord from parse_emg_patient_task)
from parse_emg_patient_task import ChannelInfo, EMGRecord, _extract_subset, parse_emg_csv


def find_and_load_record(
    patient_id: str,
    task_name: str,
    modality: str,
    data_dir: Path,
    emg_dir: Path,
):
    """Load EMG record from pkl or parse CSV. Returns record with .data dict."""
    data_dir = Path(data_dir)
    emg_dir = Path(emg_dir)

    # Try structured pkl first
    pkl_path = data_dir / f"{patient_id}_EMG" / f"{task_name}_{modality}.pkl"
    if pkl_path.exists():
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    # Fallback: parse CSV on the fly (requires parse_emg_patient_task)
    patient_folder = emg_dir / f"{patient_id}_EMG"
    if not patient_folder.exists():
        raise FileNotFoundError(
            f"Neither {pkl_path} nor {patient_folder} found. "
            "Run: python parse_emg_patient_task.py --emg-dir data/emg --out data/emg_structured"
        )

    # Find CSV matching task (task_name in our output = parts[2:] from stem)
    # e.g. Task_SN_T0 matches CROSS_018_Task_SN_T0.csv
    candidates = list(patient_folder.glob("*.csv"))
    csv_path = None
    for c in candidates:
        stem = c.stem
        parts = stem.split("_")
        if len(parts) >= 3:
            task_part = "_".join(parts[2:])
            if task_part == task_name and "Calibrazione" not in stem:
                csv_path = c
                break

    if csv_path is None:
        available = [c.stem for c in candidates if "Calibrazione" not in c.stem]
        raise FileNotFoundError(
            f"No CSV found for task '{task_name}' in {patient_folder}. "
            f"Available: {available}"
        )

    rec = parse_emg_csv(csv_path)
    if modality == "emg":
        names = rec.get_emg_channels()
    elif modality == "imu":
        names = rec.get_acc_channels()
    else:
        names = rec.get_gyro_channels()

    sub = _extract_subset(rec, names)
    if sub is None:
        raise ValueError(f"No {modality} channels in {csv_path.name}")
    return sub


def crop_to_time_window(
    rec: EMGRecord, t_start: Optional[float], t_end: Optional[float]
) -> EMGRecord:
    """Return record with data cropped to [t_start, t_end] in seconds."""
    if t_start is None and t_end is None:
        return rec

    new_data = {}
    for ch_name, d in rec.data.items():
        t = d["times"]
        v = d["values"]
        mask = np.ones(len(t), dtype=bool)
        if t_start is not None:
            mask &= t >= t_start
        if t_end is not None:
            mask &= t <= t_end
        t_crop, v_crop = t[mask], v[mask]
        new_data[ch_name] = {"times": t_crop, "values": v_crop}

    n_kept = len(next(iter(new_data.values()))["times"])
    if n_kept == 0:
        raise ValueError(
            f"No data in time window [{t_start or 0}, {t_end or 'inf'}] s. "
            "Check t-start and t-end."
        )

    return EMGRecord(
        patient_id=rec.patient_id,
        task_name=rec.task_name,
        task_type=rec.task_type,
        session=rec.session,
        condition=rec.condition,
        channels=rec.channels,
        metadata=rec.metadata.copy(),
        channel_settings=rec.channel_settings,
        data=new_data,
        raw_matrix=None,
        column_header=rec.column_header,
    )


def _channel_display_name(ch_name: str) -> str:
    """Extract muscle name only, strip channel index (e.g. 'EMG 1', 'Acc 1.X')."""
    return ch_name.split(":")[0].strip()


def _group_imu_channels(channel_names: list) -> List[Tuple[str, List[str]]]:
    """Group IMU channels by sensor; each sensor has X, Y, Z. Returns [(base_name, [ch_x, ch_y, ch_z]), ...]."""
    groups: dict[str, list[str]] = {}
    for ch in channel_names:
        # Handle "Acc 1.X" and "GYRO.X 3" style names
        if ".X" in ch or ".Y" in ch or ".Z" in ch:
            base = ch.replace(".X", "").replace(".Y", "").replace(".Z", "").strip()
        else:
            base = ch
        groups.setdefault(base, []).append(ch)
    for base in groups:
        def _axis_order(c: str) -> int:
            for i, s in enumerate([".X", ".Y", ".Z"]):
                if s in c:
                    return i
            return 99

        groups[base].sort(key=_axis_order)
    return [(b, groups[b]) for b in sorted(groups.keys(), key=lambda x: groups[x][0])]


def _filter_emg_bandpass(values: np.ndarray, fs: float, low: float = 20.0, high: float = 450.0, order: int = 4) -> np.ndarray:
    """Apply bandpass filter to EMG signal. Typical EMG bandwidth 20-450 Hz."""
    nyq = 0.5 * fs
    low_n = low / nyq
    high_n = min(high / nyq, 0.99)
    b, a = butter(order, [low_n, high_n], btype="band")
    return filtfilt(b, a, values)


def _build_muscle_mapping(
    rec_emg: Optional[EMGRecord],
    rec_imu: Optional[EMGRecord],
    rec_gyro: Optional[EMGRecord],
) -> List[Tuple[str, Optional[str], List[str], List[str]]]:
    """Build list of (muscle_name, emg_ch, acc_chs [x,y,z], gyro_chs [x,y,z]). Uses EMG muscles as primary."""
    muscles: dict[str, dict] = {}

    if rec_emg:
        for ch in rec_emg.data:
            muscle = _channel_display_name(ch)
            muscles.setdefault(muscle, {"emg": None, "acc": [], "gyro": []})
            muscles[muscle]["emg"] = ch

    def add_imu(rec: Optional[EMGRecord], key: str) -> None:
        if not rec:
            return
        groups = _group_imu_channels(list(rec.data.keys()))
        for base, ch_list in groups:
            muscle = _channel_display_name(ch_list[0]) if ch_list else ""
            if muscle:
                muscles.setdefault(muscle, {"emg": None, "acc": [], "gyro": []})
                muscles[muscle][key] = ch_list

    add_imu(rec_imu, "acc")
    add_imu(rec_gyro, "gyro")

    return [
        (m, data["emg"], data["acc"], data["gyro"])
        for m, data in sorted(muscles.items())
    ]


def _style_ax(ax, spine_width: float = 1.5) -> None:
    ax.grid(False)
    ax.set_xticks([])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(spine_width)
    ax.spines["bottom"].set_linewidth(spine_width)


def plot_interactive(
    rec_emg: Optional[EMGRecord],
    rec_imu: Optional[EMGRecord],
    patient_id: str,
    task_name: str,
) -> None:
    """Plot EMG (7x2) and IMU (7x6) in separate figures. Interactive toolbar enabled."""
    spine_width = 1.5
    n_rows = 7

    # EMG: 7x2
    n_emg_cols = 2
    emg_channels = list(rec_emg.data.keys()) if rec_emg else []
    n_emg = len(emg_channels)

    # IMU: 7x8 (2 sensors per row, 4 cols each: X, Y, Z, SMV)
    n_imu_cols = 8
    imu_groups = _group_imu_channels(list(rec_imu.data.keys())) if rec_imu else []
    n_imu_slots = n_rows * 2  # 2 sensor groups per row

    # --- Figure 1: EMG only ---
    if rec_emg:
        fig_emg = plt.figure(figsize=(14, 10))
        gs_emg = fig_emg.add_gridspec(n_rows, n_emg_cols, hspace=0.15, wspace=0.1)
        axes_emg = gs_emg.subplots(sharex=True, sharey=True)
        if n_emg == 0:
            for ax in axes_emg.flat:
                ax.set_visible(False)
        else:
            for idx, ch_name in enumerate(emg_channels):
                if idx >= n_rows * n_emg_cols:
                    break
                row, col = idx // n_emg_cols, idx % n_emg_cols
                ax = axes_emg[row, col]
                t, v = rec_emg.data[ch_name]["times"], rec_emg.data[ch_name]["values"]
                ax.plot(t, v, linewidth=0.8, color="steelblue")
                label = _channel_display_name(ch_name)
                ax.text(0.02, 0.95, label, transform=ax.transAxes, fontsize=8,
                        verticalalignment="top", weight="bold")

            for idx in range(n_rows * n_emg_cols):
                ax = axes_emg.flat[idx]
                if idx >= len(emg_channels):
                    ax.set_visible(False)
                    continue
                _style_ax(ax, spine_width)

        fig_emg.suptitle(f"{patient_id} | {task_name} | EMG", y=1.02, fontsize=12)
        fig_emg.tight_layout()
        plt.show()

    # --- Figure 2: IMU only ---
    if not rec_imu:
        return
    fig_imu = plt.figure(figsize=(20, 10))
    gs_imu = fig_imu.add_gridspec(n_rows, n_imu_cols, hspace=0.15, wspace=0.08)
    # X,Y,Z share y; SMV have their own shared y (centered on 1, fixed -1.05 to 1.05)
    axes_imu = gs_imu.subplots(sharex=True, sharey=False)
    used_imu_axes = set()
    smv_axes: List = []
    if not imu_groups:
        for ax in axes_imu.flat:
            ax.set_visible(False)
    else:
        axis_labels = ("X", "Y", "Z")
        for group_idx, (base_name, ch_list) in enumerate(imu_groups):
            if group_idx >= n_imu_slots:
                break
            row = group_idx // 2
            col_base = (group_idx % 2) * 4  # 0 or 4 (cols 0-3 or 4-7)
            muscle = _channel_display_name(ch_list[0]) if ch_list else ""
            vals_xyz = []
            t_ref = None
            for i, ch_name in enumerate(ch_list):
                if i >= 3:
                    break
                ax = axes_imu[row, col_base + i]
                t, v = rec_imu.data[ch_name]["times"], rec_imu.data[ch_name]["values"]
                if t_ref is None:
                    t_ref = t
                vals_xyz.append(v)
                ax.plot(t, v, linewidth=0.8, color="steelblue")
                lbl = f"{muscle} {axis_labels[i]}" if muscle else axis_labels[i]
                ax.text(0.02, 0.95, lbl, transform=ax.transAxes, fontsize=7,
                        verticalalignment="top", weight="bold")
                used_imu_axes.add((row, col_base + i))
            # SMV at column col_base + 3 (position 4 or 8)
            if len(vals_xyz) == 3:
                vx, vy, vz = vals_xyz[0], vals_xyz[1], vals_xyz[2]
                min_len = min(len(vx), len(vy), len(vz))
                smv = np.sqrt(vx[:min_len] ** 2 + vy[:min_len] ** 2 + vz[:min_len] ** 2)
                t_smv = t_ref[:min_len] if len(t_ref) >= min_len else t_ref
                ax_smv = axes_imu[row, col_base + 3]
                ax_smv.plot(t_smv, smv, linewidth=0.8, color="darkorange")
                ax_smv.axhline(1.0, linestyle="--", color="gray", alpha=0.5, linewidth=0.8)
                ax_smv.set_ylim(1.0 - 1.05, 1.0 + 1.05)  # centrato su 1, range ±1.05
                ax_smv.text(0.02, 0.95, f"{muscle} SMV" if muscle else "SMV",
                            transform=ax_smv.transAxes, fontsize=7,
                            verticalalignment="top", weight="bold")
                used_imu_axes.add((row, col_base + 3))
                smv_axes.append(ax_smv)

        # Share y among SMV axes only
        if len(smv_axes) >= 2:
            for ax in smv_axes[1:]:
                ax.sharey(smv_axes[0])

        for row in range(n_rows):
            for col in range(n_imu_cols):
                ax = axes_imu[row, col]
                if (row, col) not in used_imu_axes:
                    ax.set_visible(False)
                else:
                    _style_ax(ax, spine_width)

    fig_imu.suptitle(f"{patient_id} | {task_name} | IMU", y=1.02, fontsize=12)
    fig_imu.tight_layout()
    plt.show()


def _get_sampling_freq(rec: EMGRecord, ch_name: str) -> float:
    """Get sampling frequency for a channel from record metadata."""
    for c in rec.channels:
        if c.name == ch_name:
            return c.sampling_freq
    return 2000.0  # fallback


def plot_per_muscle(
    muscle_name: str,
    rec_emg: Optional[EMGRecord],
    rec_imu: Optional[EMGRecord],
    rec_gyro: Optional[EMGRecord],
    emg_ch: Optional[str],
    acc_chs: List[str],
    gyro_chs: List[str],
    patient_id: str,
    task_name: str,
) -> None:
    """Plot one figure per muscle: left (raw EMG, clean EMG, Acc); right (Muscle Activation, SMV)."""
    fig, axes = plt.subplots(3, 2, figsize=(14, 8), gridspec_kw={"width_ratios": [1, 1]})
    spine_width = 1.5

    ax_raw = axes[0, 0]
    ax_clean = axes[1, 0]
    ax_acc = axes[2, 0]
    ax_activation = axes[1, 1]
    ax_smv = axes[2, 1]
    axes[0, 1].set_visible(False)

    colors = {"X": "#e41a1c", "Y": "#377eb8", "Z": "#4daf4a"}

    # Panel 1: Raw EMG
    if rec_emg and emg_ch and emg_ch in rec_emg.data:
        t, v = rec_emg.data[emg_ch]["times"], rec_emg.data[emg_ch]["values"]
        ax_raw.plot(t, v, linewidth=0.8, color="steelblue")
        ax_raw.set_ylabel("Raw EMG (V)")
    _style_ax(ax_raw, spine_width)

    # Panel 2: Filtered EMG
    emg_for_nk = None
    if rec_emg and emg_ch and emg_ch in rec_emg.data:
        t, v = rec_emg.data[emg_ch]["times"], rec_emg.data[emg_ch]["values"]
        fs = _get_sampling_freq(rec_emg, emg_ch)
        v_filt = _filter_emg_bandpass(v, fs)
        emg_for_nk = (t, v, fs)
        ax_clean.plot(t, v_filt, linewidth=0.8, color="darkgreen")
        ax_clean.set_ylabel("Clean EMG (V)")
    _style_ax(ax_clean, spine_width)

    # Panel 2 right: Muscle Activation (neurokit emg_process - amplitude + onsets/offsets)
    if _HAS_NEUROKIT and emg_for_nk:
        t_emg, v_emg, fs = emg_for_nk
        try:
            signals, info = nk.emg_process(v_emg, sampling_rate=int(fs))
            amp = signals["EMG_Amplitude"].values
            n_amp = min(len(amp), len(t_emg))
            t_plot = t_emg[:n_amp]
            amp_plot = amp[:n_amp]
            ax_activation.plot(t_plot, amp_plot, linewidth=0.8, color="darkorange", label="Amplitude")
            # Shade activation regions
            if "EMG_Activity" in signals:
                activity = signals["EMG_Activity"].values[:n_amp]
                in_act = activity > 0.5
                if np.any(in_act):
                    ax_activation.fill_between(t_plot, 0, amp_plot, where=in_act, alpha=0.3, color="darkorange")
            # Onset/offset: vertical dashed lines and markers
            for key in ["EMG_Onsets", "EMG_Offsets"]:
                if key in signals:
                    idx = np.where(signals[key].values[:n_amp] > 0.5)[0]
                    for i in idx:
                        ax_activation.axvline(t_plot[i], color="black", linestyle="--", linewidth=0.6, alpha=0.7)
                    if len(idx) > 0:
                        ax_activation.scatter(t_plot[idx], np.zeros(len(idx)), c="red", s=15, zorder=5)
            ax_activation.set_ylabel("Amplitude")
        except Exception:
            ax_activation.text(0.5, 0.5, "neurokit failed", transform=ax_activation.transAxes, ha="center")
    if not _HAS_NEUROKIT:
        ax_activation.text(0.5, 0.5, "pip install neurokit2", transform=ax_activation.transAxes,
                           ha="center", va="center", fontsize=9)
    ax_activation.set_title("Muscle Activation")
    _style_ax(ax_activation, spine_width)

    # Panel 3: Accelerations X, Y, Z
    vals_xyz_acc = []
    t_acc_ref = None
    if rec_imu and acc_chs:
        for ch in acc_chs:
            if ch in rec_imu.data:
                t, v = rec_imu.data[ch]["times"], rec_imu.data[ch]["values"]
                if t_acc_ref is None:
                    t_acc_ref = t
                vals_xyz_acc.append(v)
                label = "Z"
                for s in (".X", ".Y", ".Z"):
                    if s in ch:
                        label = s[1]
                        break
                ax_acc.plot(t, v, linewidth=0.8, label=label, color=colors.get(label, "gray"))
        ax_acc.legend(loc="upper right", fontsize=8)
        ax_acc.set_ylabel("Acc (g)")
    ax_acc.set_title("Acceleration")
    _style_ax(ax_acc, spine_width)

    # Panel 3 right: SMV of this muscle's IMU
    if rec_imu and len(vals_xyz_acc) == 3 and t_acc_ref is not None:
        vx, vy, vz = vals_xyz_acc[0], vals_xyz_acc[1], vals_xyz_acc[2]
        min_len = min(len(vx), len(vy), len(vz), len(t_acc_ref))
        smv = np.sqrt(vx[:min_len] ** 2 + vy[:min_len] ** 2 + vz[:min_len] ** 2)
        t_smv = t_acc_ref[:min_len]
        ax_smv.plot(t_smv, smv, linewidth=0.8, color="darkorange")
        ax_smv.axhline(1.0, linestyle="--", color="gray", alpha=0.5, linewidth=0.8)
        ax_smv.set_ylim(1.0 - 1.05, 1.0 + 1.05)
        ax_smv.set_ylabel("SMV (g)")
    ax_smv.set_title("Acc SMV")
    _style_ax(ax_smv, spine_width)
    ax_acc.set_xlabel("Time (s)")

    fig.suptitle(f"{patient_id} | {task_name} | {muscle_name}", y=0.98, fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def main():
    ap = argparse.ArgumentParser(
        description="Interactive EMG/IMU/Gyro signal viewer"
    )
    ap.add_argument(
        "--patient",
        type=str,
        default="CROSS_001",
        help="Patient ID (default: CROSS_001)",
    )
    ap.add_argument(
        "--task",
        type=str,
        default="Task_T0_SN",
        help="Task name (default: Task_T0_SN)",
    )
    ap.add_argument(
        "--t-start",
        type=float,
        default=None,
        help="Start time in seconds (optional)",
    )
    ap.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="End time in seconds (optional)",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/emg_structured"),
        help="Structured data root (default: data/emg_structured)",
    )
    ap.add_argument(
        "--emg-dir",
        type=Path,
        default=Path("data/emg"),
        help="Raw EMG CSV root for fallback parsing (default: data/emg)",
    )
    args = ap.parse_args()

    rec_emg = None
    rec_imu = None
    rec_gyro = None  # gyro not saved by parser
    for modality in ("emg", "imu"):
        try:
            rec = find_and_load_record(
                args.patient,
                args.task,
                modality,
                args.data_dir,
                args.emg_dir,
            )
            if args.t_start is not None or args.t_end is not None:
                rec = crop_to_time_window(rec, args.t_start, args.t_end)
            if modality == "emg":
                rec_emg = rec
            elif modality == "imu":
                rec_imu = rec
            else:
                rec_gyro = rec
            n_ch = len(rec.data)
            n_samp = len(next(iter(rec.data.values()))["times"]) if rec.data else 0
            print(f"Loaded {modality}: {n_ch} channels, {n_samp:,} samples")
        except (FileNotFoundError, ValueError) as e:
            print(f"  Skipped {modality}: {e}")

    if rec_emg is None and rec_imu is None:
        raise SystemExit("No EMG or IMU data found. Check patient/task/dirs.")

    plot_interactive(
        rec_emg=rec_emg,
        rec_imu=rec_imu,
        patient_id=args.patient,
        task_name=args.task,
    )

    # Per-muscle figures (only for muscles with EMG)
    muscle_list = _build_muscle_mapping(rec_emg, rec_imu, rec_gyro)
    for muscle_name, emg_ch, acc_chs, gyro_chs in muscle_list:
        if emg_ch:
            print(f"  Per-muscle figure: {muscle_name}")
            plot_per_muscle(
                muscle_name=muscle_name,
                rec_emg=rec_emg,
                rec_imu=rec_imu,
                rec_gyro=rec_gyro,
                emg_ch=emg_ch,
                acc_chs=acc_chs,
                gyro_chs=gyro_chs,
                patient_id=args.patient,
                task_name=args.task,
            )


if __name__ == "__main__":
    main()
