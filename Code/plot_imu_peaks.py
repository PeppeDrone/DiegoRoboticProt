#!/usr/bin/env python3
"""plot_imu_peaks.py

Load IMU data for a patient/task, select a channel, find peaks with scipy.find_peaks,
and plot the signal with peaks marked.

Usage:
  python plot_imu_peaks.py --patient CROSS_001 --task Task_T0_SN
  python plot_imu_peaks.py --channel "Brachioradialis: Acc 1.X" "Anterior Deltoid: Acc 5.X"
  python plot_imu_peaks.py --manual --channel "Ch1" "Ch2" --out segments.json
  python plot_imu_peaks.py --align-average
  python plot_imu_peaks.py --channel "Brachioradialis: Acc 1.X" --distance 100 --prominence 0.1

Run parse_emg_patient_task.py first to create structured pkl files.
"""

from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path
from typing import List, Optional

import matplotlib
matplotlib.use("TkAgg")  # Interactive backend for plt.ginput(); must be before pyplot import
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.widgets import Button
import numpy as np
from scipy.signal import find_peaks

# For pickle unpickling
from B0_parse_emg_patient_task import ChannelInfo, EMGRecord


def load_imu_record(
    patient_id: str,
    task_name: str,
    data_dir: Path,
) -> EMGRecord:
    """Load IMU record from pkl."""
    pkl_path = data_dir / f"{patient_id}_EMG" / f"{task_name}_imu.pkl"
    if not pkl_path.exists():
        raise FileNotFoundError(
            f"Not found: {pkl_path}\n"
            "Run: python parse_emg_patient_task.py --emg-dir data/emg --out data/emg_structured"
        )
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def load_emg_record(
    patient_id: str,
    task_name: str,
    data_dir: Path,
) -> Optional[EMGRecord]:
    """Load EMG record from pkl. Returns None if not found."""
    pkl_path = data_dir / f"{patient_id}_EMG" / f"{task_name}_emg.pkl"
    if not pkl_path.exists():
        return None
    with open(pkl_path, "rb") as f:
        return pickle.load(f)


def imu_channel_to_emg(imu_ch: str, emg_channels: List[str]) -> Optional[str]:
    """Map IMU channel (e.g. 'Brachioradialis: Acc 1.X') to EMG channel (e.g. 'Brachioradialis: EMG 1')."""
    m = re.search(r"^(.+?):\s*Acc\s*(\d+)\.", imu_ch, re.IGNORECASE)
    if not m:
        return None
    muscle, num = m.group(1).strip(), m.group(2)
    pattern = f"{muscle}: EMG {num}"
    for ch in emg_channels:
        if pattern.lower() in ch.lower() or ch == pattern:
            return ch
    return None


def list_channels(rec: EMGRecord) -> List[str]:
    """Return sorted list of channel names."""
    return sorted(rec.data.keys())


def muscle_name(ch_name: str) -> str:
    """Extract muscle name from channel (e.g. 'Brachioradialis: Acc 1.X' -> 'Brachioradialis')."""
    return ch_name.split(":")[0].strip()


def _style_ax(ax, hide_xticks: bool = False):
    """Remove top/right spines; optionally hide x-ticks."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if hide_xticks:
        ax.set_xticks([])


# Manual mode: labeling always on these two (top=starts, bottom=ends)
MANUAL_LABEL_CHANNELS = ["Anterior Deltoid", "Brachioradialis"]
# Additional panels shown in manual mode (peaks drawn on all, but no clicking)
MANUAL_VIZ_EXTRA_CHANNELS = ["Posterior Deltoid", "Trapezius Middle"]

# English -> Italian muscle name variants (for files with Italian channel labels)
MUSCLE_EN_IT = {
    "anterior deltoid": "deltoide anteriore",
    "middle deltoid": "deltoide medio",
    "posterior deltoid": "deltoide posteriore",
    "trapezius middle": "trapezio medio",
    "biceps long head": "bicipite brachiale capo lungo",
    "biceps short head": "bicipite capo corto",
    "triceps lateral": "tricipite brachiale capo laterale",
    "triceps long head": "tricipite brchiale capo lungo",  # matches typo in data
    "latissimus dorsi": "latissimus dorsi",
    "infraspinatus": "infraspinato",
    "pectoralis": "pettorale",
    "first dorsal interosseous": "primo interosseo",
    "teres major": "teres major",
    "brachioradialis": "brachioradialis",
}


def resolve_channel(spec: str, channels: List[str], prefer_x: bool = True) -> str:
    """Resolve partial channel spec to full name. Handles both formats: 'Acc N.X' and 'ACC.X N'.
    Also maps English muscle names to Italian variants when files use Italian labels."""
    spec_lower = spec.lower()
    matches = [c for c in channels if spec_lower in c.lower()]
    if not matches:
        # Fallback: match on muscle name only (e.g. "Trapezius Middle" matches "Trapezius Middle: ACC.X 14")
        muscle = spec.split(":")[0].strip().lower()
        if muscle:
            matches = [c for c in channels if muscle in c.lower() and ("acc" in c.lower() or "emg" in c.lower())]
    if not matches and muscle:
        # Fallback: try Italian variant (e.g. "Anterior Deltoid" -> "Deltoide anteriore")
        it_muscle = MUSCLE_EN_IT.get(muscle)
        if it_muscle:
            matches = [c for c in channels if it_muscle in c.lower() and ("acc" in c.lower() or "emg" in c.lower())]
        if not matches:
            # Reverse: maybe channel is Italian, spec is English; match if any Italian term contains English word
            for en, it in MUSCLE_EN_IT.items():
                if en in muscle or muscle in en:
                    matches = [c for c in channels if it in c.lower() and ("acc" in c.lower() or "emg" in c.lower())]
                    if matches:
                        break
    if not matches:
        raise ValueError(f"Channel '{spec}' not found. Available: {channels}")
    if len(matches) == 1:
        return matches[0]
    exact = next((c for c in matches if c == spec), None)
    if exact:
        return exact
    if prefer_x:
        x_match = next((c for c in matches if ".x" in c.lower() or "acc.x" in c.lower()), None)
        if x_match:
            return x_match
    return matches[0]


def extract_and_align_segments(
    t: np.ndarray,
    v: np.ndarray,
    pairs: List[tuple],
    n_pts: int = 101,
) -> tuple:
    """Extract segments, resample to 0-100% via nearest-neighbor (no interpolation)."""
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
        # Nearest-neighbor: index into v_seg for each pct (no interpolation)
        idx = np.clip(np.searchsorted(t_norm, pct, side="right") - 1, 0, len(v_seg) - 1)
        stacked.append(v_seg[idx])
    if not stacked:
        raise ValueError("No valid segments extracted.")
    stacked = np.array(stacked)
    mean = np.mean(stacked, axis=0)
    std = np.std(stacked, axis=0)
    return pct, stacked, mean, std


def get_smv_for_sensor(rec: EMGRecord, sensor_name: str) -> Optional[tuple]:
    """Get (times, smv_values, label) for sensor. sensor_name is muscle prefix, e.g. 'Brachioradialis'."""
    sensor_lower = sensor_name.strip().lower()
    # Group channels by sensor id (stem before .X/.Y/.Z)
    groups = {}
    for ch in rec.data:
        if ".X" not in ch and ".Y" not in ch and ".Z" not in ch:
            continue
        stem = ch.replace(".X", "").replace(".Y", "").replace(".Z", "").strip()
        muscle = stem.split(":")[0].strip().lower()
        if sensor_lower not in muscle and muscle != sensor_lower:
            continue
        idx = 0 if ".X" in ch else 1 if ".Y" in ch else 2
        groups.setdefault(stem, {})[idx] = ch
    for stem, ch_dict in groups.items():
        if len(ch_dict) == 3:
            ch_x = ch_dict[0]
            ch_y = ch_dict[1]
            ch_z = ch_dict[2]
            break
    else:
        return None
    tx, vx = rec.data[ch_x]["times"], rec.data[ch_x]["values"]
    _, vy = rec.data[ch_y]["times"], rec.data[ch_y]["values"]
    _, vz = rec.data[ch_z]["times"], rec.data[ch_z]["values"]
    n = min(len(tx), len(vx), len(vy), len(vz))
    smv = np.sqrt(vx[:n] ** 2 + vy[:n] ** 2 + vz[:n] ** 2)
    return tx[:n], smv, f"{sensor_name} SMV"


def main():
    ap = argparse.ArgumentParser(
        description="Find peaks on IMU channel and plot with markers"
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
        default="Task_T1_SN",
        help="Task name (default: Task_T0_SN)",
    )
    ap.add_argument(
        "--channel",
        type=str,
        nargs="*",
        default=["Posterior Deltoid: Acc 7.X", "Trapezius Middle: Acc 14.X"],
        help="IMU channel name(s). Pass multiple for multiple channels. EMG shown below each IMU.",
    )
    ap.add_argument(
        "--smv",
        type=str,
        default=None,
        help="Use SMV (magnitude) of sensor instead of single channel. Mutually exclusive with --channel (default: None)",
    )
    ap.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/emg_structured"),
        help="Structured data root (default: data/emg_structured)",
    )
    ap.add_argument(
        "--t-start",
        type=float,
        default=None,
        help="Start time in seconds (default: None)",
    )
    ap.add_argument(
        "--t-end",
        type=float,
        default=None,
        help="End time in seconds (default: None)",
    )
    # find_peaks parameters
    ap.add_argument(
        "--height",
        type=float,
        default=None,
        help="Minimum peak height (default: None)",
    )
    ap.add_argument(
        "--distance",
        type=int,
        default=50,
        help="Minimum distance between peaks in samples (default: 50)",
    )
    ap.add_argument(
        "--prominence",
        type=float,
        default=0.1,
        help="Minimum peak prominence (default: 0.1)",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=None,
        help="Minimum peak width in samples (default: None)",
    )
    ap.add_argument(
        "--manual",
        action="store_true",
        help="Manual peak selection: click 4 starts on top panel, 4 ends on bottom. No auto peaks.",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Save manual (start, end) pairs to JSON file (default: None)",
    )
    ap.add_argument(
        "--align-average",
        action="store_true",
        help="Load saved segments, align to 0-100%%, plot mean ± std. Requires prior --manual save.",
    )
    ap.add_argument(
        "--segments",
        type=Path,
        default=None,
        help="Segments JSON for --align-average (default: {data_dir}/{patient}_EMG/{task}_manual_segments.json)",
    )
    ap.add_argument(
        "--n-pts",
        type=int,
        default=101,
        help="Points for 0-100%% alignment when --align-average (default: 101)",
    )
    args = ap.parse_args()

    # --align-average: load segments, plot mean±std, exit
    if args.align_average:
        segments_path = args.segments or (
            args.data_dir / f"{args.patient}_EMG" / f"{args.task}_manual_segments.json"
        )
        if not segments_path.exists():
            raise FileNotFoundError(
                f"Segments file not found: {segments_path}\n"
                "Run with --manual first and click Save & Close."
            )
        with open(segments_path) as f:
            seg_data = json.load(f)
        pairs = [(p["start_s"], p["end_s"]) for p in seg_data["pairs"]]
        channels_spec = seg_data.get("channels", [])
        patient = seg_data.get("patient", args.patient)
        task = seg_data.get("task", args.task)
        if not channels_spec:
            raise ValueError("Segments file has no 'channels' key.")
        rec = load_imu_record(patient, task, args.data_dir)
        all_channels = sorted(rec.data.keys())
        n_ch = len(channels_spec)
        fig, axes = plt.subplots(n_ch, 1, figsize=(10, 4 * n_ch), sharex=True, squeeze=False)
        axes = axes[:, 0]
        for i, (ax, spec) in enumerate(zip(axes, channels_spec)):
            ch_name = resolve_channel(spec, all_channels)
            t = rec.data[ch_name]["times"]
            v = rec.data[ch_name]["values"]
            pct, stacked, mean, std = extract_and_align_segments(
                t, v, pairs, n_pts=args.n_pts
            )
            ax.plot(pct, mean, color="steelblue", linewidth=2)
            ax.fill_between(pct, mean - std, mean + std, alpha=0.3, color="steelblue")
            ax.set_ylabel(muscle_name(ch_name))
            _style_ax(ax)
        axes[-1].set_xlabel("Cycle (%)")
        fig.suptitle(f"{patient} | {task}")
        fig.tight_layout()
        plt.show()
        return

    rec = load_imu_record(args.patient, args.task, args.data_dir)
    channels = list_channels(rec)

    if not args.channel and args.smv is None:
        print("Available IMU channels:")
        for ch in channels:
            print(f"  {ch}")
        print("\nUse --channel <name> [<name> ...] or --smv <sensor> to plot and find peaks.")
        return

    if args.channel and args.smv:
        raise ValueError("Use either --channel or --smv, not both.")

    if args.manual and args.smv:
        raise ValueError("--manual requires --channel (not --smv). Labeling uses Anterior Deltoid + Brachioradialis.")

    # Collect (t, v, ch_name, peaks) for each channel
    items: List[tuple] = []
    if args.manual:
        channel_specs = MANUAL_LABEL_CHANNELS + MANUAL_VIZ_EXTRA_CHANNELS
    elif args.channel:
        channel_specs = args.channel
    else:
        channel_specs = []

    if args.smv and not args.manual:
        result = get_smv_for_sensor(rec, args.smv)
        if result is None:
            raise ValueError(f"Could not find X,Y,Z channels for sensor '{args.smv}'.")
        t, v = result[0], result[1]
        ch_name = result[2]
        items.append((t, v, ch_name))
    elif channel_specs:
        for spec in channel_specs:
            ch_name = resolve_channel(spec, channels)
            t, v = rec.data[ch_name]["times"], rec.data[ch_name]["values"]
            items.append((t, v, ch_name))

    # Optional time crop and find peaks for each
    pk_kw = {"distance": args.distance, "prominence": args.prominence}
    if args.height is not None:
        pk_kw["height"] = args.height
    if args.width is not None:
        pk_kw["width"] = args.width

    plot_data: List[tuple] = []
    for t, v, ch_name in items:
        if args.t_start is not None or args.t_end is not None:
            mask = np.ones(len(t), dtype=bool)
            if args.t_start is not None:
                mask &= t >= args.t_start
            if args.t_end is not None:
                mask &= t <= args.t_end
            t, v = t[mask], v[mask]
        if args.manual:
            peaks = np.array([], dtype=np.intp)
        else:
            peaks, _ = find_peaks(v, **pk_kw)
            print(f"Found {len(peaks)} peaks on {ch_name}")
        plot_data.append((t, v, ch_name, peaks))

    # Plot
    n = len(plot_data)
    if args.manual:
        # Manual: 8 panels = 4 channel pairs (IMU+EMG each). Labeling on Anterior Deltoid (top) and Brachioradialis (bottom).
        # Extra panels: Posterior Deltoid, Trapezius Middle. Peaks drawn on all panels.
        n_pairs = len(plot_data)  # 4: Anterior Deltoid, Brachioradialis, Posterior Deltoid, Trapezius Middle
        emg_rec = load_emg_record(args.patient, args.task, args.data_dir)
        emg_channels = sorted(emg_rec.data.keys()) if emg_rec else []
        has_emg = [
            bool(imu_channel_to_emg(plot_data[i][2], emg_channels) and emg_rec)
            for i in range(n_pairs)
        ]
        n_axes = n_pairs * 2  # IMU + EMG per channel
        fig = plt.figure(figsize=(24, 4.5 * n_axes))
        height_ratios = []
        for _ in range(n_pairs):
            height_ratios.extend([2, 0.8])
        gs = GridSpec(n_axes, 1, height_ratios=height_ratios, hspace=0.15)
        axes = [fig.add_subplot(gs[0])]
        for i in range(1, n_axes):
            axes.append(fig.add_subplot(gs[i], sharex=axes[0]))
        for i in range(n_axes):
            pair_idx = i // 2
            if i % 2 == 0:
                t_imu, v_imu, ch_name, _ = plot_data[pair_idx]
                axes[i].plot(t_imu, v_imu, linewidth=0.8, color="steelblue")
                axes[i].set_ylabel(muscle_name(ch_name), fontsize=9)
            else:
                imu_ch = plot_data[pair_idx][2]
                emg_ch = imu_channel_to_emg(imu_ch, emg_channels) if emg_channels else None
                if emg_ch and emg_rec:
                    te, ve = emg_rec.data[emg_ch]["times"], emg_rec.data[emg_ch]["values"]
                    if args.t_start is not None or args.t_end is not None:
                        mask = np.ones(len(te), dtype=bool)
                        if args.t_start is not None:
                            mask &= te >= args.t_start
                        if args.t_end is not None:
                            mask &= te <= args.t_end
                        te, ve = te[mask], ve[mask]
                    if len(te) > 0:
                        axes[i].plot(te, ve, linewidth=0.6, color="darkgreen", alpha=0.9)
                    axes[i].set_ylabel(muscle_name(emg_ch), fontsize=8)
                else:
                    axes[i].set_visible(False)
            _style_ax(axes[i])
        for ax in axes:
            ax.xaxis.set_tick_params(labelbottom=True)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{args.patient} | {args.task} — Click on Anterior Deltoid (top), then Brachioradialis")
    else:
        fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True, squeeze=False)
        axes = axes[:, 0]
        for i, (ax, (t, v, ch_name, peaks)) in enumerate(zip(axes, plot_data)):
            ax.plot(t, v, linewidth=0.8, color="steelblue")
            if len(peaks) > 0:
                ax.scatter(t[peaks], v[peaks], color="red", s=40, zorder=5)
            ax.set_ylabel(muscle_name(ch_name))
            _style_ax(ax)
        for ax in axes:
            ax.xaxis.set_tick_params(labelbottom=True)
        axes[-1].set_xlabel("Time (s)")
        fig.suptitle(f"{args.patient} | {args.task}")

    fig.tight_layout()

    if args.manual:
        # Labeling: Anterior Deltoid (panel 0), Brachioradialis (panel 2). Peaks drawn on all_axes.
        ax_imu_top = axes[0]   # Anterior Deltoid
        ax_imu_bot = axes[2]   # Brachioradialis
        all_axes = axes
        fig.subplots_adjust(bottom=0.12)
        plt.show(block=False)
        # Position on the screen where the cursor is (e.g. where Cursor IDE is)
        try:
            mng = plt.get_current_fig_manager()
            if hasattr(mng, "window"):
                cx = mng.window.winfo_pointerx()
                cy = mng.window.winfo_pointery()
                mng.window.geometry(f"+{max(0, cx - 500)}+{max(0, cy - 150)}")
        except Exception:
            pass
        plt.pause(0.2)
        colors = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
        pairs: List[tuple] = []

        def draw_start(i: int, t_start: float, y_start: float):
            c = colors[i]
            for ax in all_axes:
                ax.axvline(t_start, color=c, linestyle="--", linewidth=1.2, alpha=0.8)
            ax_imu_top.scatter([t_start], [y_start], color=c, s=60, zorder=6, marker="o", edgecolors="black")
            fig.canvas.draw_idle()

        def draw_end(i: int, t_end: float, y_end: float):
            c = colors[i]
            for ax in all_axes:
                ax.axvline(t_end, color=c, linestyle="--", linewidth=1.2, alpha=0.8)
            ax_imu_bot.scatter([t_end], [y_end], color=c, s=60, zorder=6, marker="o", edgecolors="black")
            fig.canvas.draw_idle()

        print("Click 4 times on the TOP IMU panel (start times).")
        plt.sca(ax_imu_top)
        starts = plt.ginput(4, timeout=0, show_clicks=True)
        if len(starts) < 4:
            plt.close()
            raise ValueError("Need 4 clicks on top IMU panel.")
        for i, (t_s, y_s) in enumerate(starts):
            draw_start(i, float(t_s), float(y_s))

        print("Click 4 times on the BOTTOM IMU panel (end times).")
        plt.sca(ax_imu_bot)
        ends = plt.ginput(4, timeout=0, show_clicks=True)
        if len(ends) < 4:
            plt.close()
            raise ValueError("Need 4 clicks on bottom IMU panel.")
        for i, (t_e, y_e) in enumerate(ends):
            draw_end(i, float(t_e), float(y_e))
            pairs.append((float(starts[i][0]), float(t_e)))

        def save_and_close(_event):
            out_path = args.out
            if out_path is None:
                out_dir = args.data_dir / f"{args.patient}_EMG"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"{args.task}_manual_segments.json"
            # Save labeling channels (Anterior Deltoid, Brachioradialis)
            label_channels = [plot_data[0][2], plot_data[1][2]]
            out_data = {
                "patient": args.patient,
                "task": args.task,
                "channels": label_channels,
                "pairs": [{"start_s": s, "end_s": e} for s, e in pairs],
            }
            out_path.write_text(json.dumps(out_data, indent=2))
            print("Pairs (start_s, end_s):")
            for i, (s, e) in enumerate(pairs):
                print(f"  {i+1}: ({s:.4f}, {e:.4f})")
            print(f"Saved to {out_path}")
            plt.close(fig)
            # Show aligned average for labeling channels only
            n_pts = args.n_pts
            label_plot_data = [plot_data[0], plot_data[1]]
            n_ch = len(label_plot_data)
            fig_avg, axes_avg = plt.subplots(
                n_ch, 1, figsize=(10, 4 * n_ch), sharex=True, squeeze=False
            )
            axes_avg = axes_avg[:, 0]
            for i, (ax_a, (t, v, ch_name, _)) in enumerate(zip(axes_avg, label_plot_data)):
                pct, stacked, mean, std = extract_and_align_segments(
                    t, v, pairs, n_pts=n_pts
                )
                ax_a.plot(pct, mean, color="steelblue", linewidth=2)
                ax_a.fill_between(pct, mean - std, mean + std, alpha=0.3, color="steelblue")
                ax_a.set_ylabel(muscle_name(ch_name))
                _style_ax(ax_a)
            axes_avg[-1].set_xlabel("Cycle (%)")
            fig_avg.suptitle(f"{args.patient} | {args.task}")
            fig_avg.tight_layout()
            plt.show()

        ax_btn = fig.add_axes([0.4, 0.02, 0.2, 0.04])
        btn = Button(ax_btn, "Save & Close")
        btn.on_clicked(save_and_close)
        fig.canvas.draw_idle()

    plt.show()


if __name__ == "__main__":
    main()
