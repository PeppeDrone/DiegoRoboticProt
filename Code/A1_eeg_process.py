#!/usr/bin/env python3
"""run_rest_preprocess_egi_erp_main.py

Resting-state preprocessing runner that:
  1) loads EDF + EGI MFF montage exactly like eeg_preprocess_standalone.py
  2) applies the preprocessing functions implemented in erp-main/src

Directory layout supported (as in your screenshots):

ROOT/
  CROSS_001_F/
    CROSS_001_F_T0_20250214_113906.mff/   (or .mff zip)
    CROSS_001_F_T1_20250310_103254.mff/
    CROSS_001_T0_20250214_113906_fil.edf
    CROSS_001_T1_20250310_103254_fil.edf

Outputs:
  OUT/
    CROSS_001_F/
      T0/
        raw_clean.fif
        epochs_clean-epo.fif
        payloads.json
      T1/
        ...

Notes
-----
- This script is for RESTING-STATE: it uses fixed-length epochs (no ERP event locking).
- If you want different window length/overlap or different filters/ICA thresholds,
  change the DEFAULTS section.

"""

from __future__ import annotations

# -----------------------------------------------------------------------------
# Optional workaround: numba can crash on import if an incompatible 'coverage'
# version is installed (AttributeError: coverage.types.Tracer). This patch
# defines missing symbols so MNE/Numba can import cleanly.
# -----------------------------------------------------------------------------
import os
os.environ.setdefault('NUMBA_DISABLE_COVERAGE', '1')
try:
    import coverage  # type: ignore
    import types as _types
    if not hasattr(coverage, 'types'):
        coverage.types = _types.SimpleNamespace()
    if not hasattr(coverage.types, 'Tracer'):
        class _DummyTracer:  # pragma: no cover
            pass
        coverage.types.Tracer = _DummyTracer
    # Provide other typing aliases that some numba versions expect
    for _name in ('TTraceData','TShouldTraceFn','TFileDisposition','TShouldStartContextFn'):
        if not hasattr(coverage.types, _name):
            setattr(coverage.types, _name, object)
except Exception:
    pass

import argparse
import json
import re
import sys
import zipfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import mne
from mne.channels import make_dig_montage


# -----------------------------------------------------------------------------
# DEFAULTS (chosen to match typical erp-main configs; adjust if desired)
# -----------------------------------------------------------------------------

DEFAULTS = {
    # Filtering
    "notch_base_hz": 50,      # erp-main expands to harmonics up to Nyquist
    "l_freq": 0.5,
    "h_freq": None,          # keep None here; erp-main later low-passes
    "l_trans_bandwidth": 0.5,
    "h_trans_bandwidth": 20.0,
    "filter_method": "fir",

    # Resampling
    "sfreq": 256,            # set None to keep original

    # PyPREP robust reference
    "pyprep_max_iterations": 20,
    "pyprep_channel_wise": True,
    "pyprep_max_chunk_size": None,
    "pyprep_seed": 42,

    # ICA
    "ica_hp_cutoff": 1.0,
    "ica_method": "infomax",
    "ica_n_components": 0.99,
    "ica_random_state": 42,
    "ica_criteria": {
        "thr_brain_min": 0.0,
        "thr_muscle_artifact": 0.5,
        "thr_eye_blink": 0.5,
        "thr_heart_beat": 0.5,
        "thr_line_noise": 0.5,
        "thr_channel_noise": 0.5,
        "thr_other": 0.5,
    },

    # Post-ICA lowpass
    "post_ica_lowpass_hz": 30.0,

    # Epoching (resting-state)
    "epoch_duration_s": 4.0,
    "epoch_overlap_fraction": 0.5,  # 0..1
    "epoch_baseline": None,

    # AutoReject
    "use_autoreject": True,
    "ar_n_interpolate": (1, 4, 8, 16),
    "ar_consensus": (0.2, 0.4, 0.6, 0.8),
    "ar_cv": None,
    "ar_random_state": 42,
    "ar_n_jobs": -1,
}
    

# -----------------------------------------------------------------------------
# Loading utilities (replicate eeg_preprocess_standalone.py behavior)
# -----------------------------------------------------------------------------

def _local(tag: str) -> str:
    return tag.split('}')[-1].lower()


def _find_child(parent, name: str):
    for el in parent:
        if _local(el.tag) == name.lower():
            return el
    return None


def parse_sensor_layout_from_mff(mff_path: Path, unit: str = "cm") -> Tuple[Dict[str, np.ndarray], Dict[str, int]]:
    """Parse sensorLayout.xml from an MFF folder or zip archive.

    Returns
    -------
    sensor_positions : dict mapping channel name -> (x,y,z) in meters
    sensor_mapping   : dict mapping integer sensor index -> channel name
    """
    mff_path = Path(mff_path)
    if not mff_path.exists():
        raise FileNotFoundError(f"MFF path not found: {mff_path}")

    if mff_path.is_dir():
        xml_file = mff_path / "sensorLayout.xml"
        if not xml_file.exists():
            raise ValueError(f"sensorLayout.xml not found in MFF folder: {mff_path}")
        root = ET.parse(xml_file).getroot()
    else:
        # zip archive
        with zipfile.ZipFile(mff_path, "r") as zf:
            xml_candidates = [n for n in zf.namelist() if "sensorlayout.xml" in n.lower()]
            if not xml_candidates:
                raise ValueError(f"sensorLayout.xml not found in {mff_path}")
            xml_path = xml_candidates[0]
            with zf.open(xml_path) as f:
                root = ET.parse(f).getroot()

    # unit conversion
    unit = unit.lower()
    if unit == "m":
        scale = 1.0
    elif unit == "cm":
        scale = 0.01
    elif unit == "mm":
        scale = 0.001
    else:
        raise ValueError(f"Unsupported unit '{unit}'. Use m/cm/mm.")

    sensors = []
    for el in root.iter():
        if _local(el.tag) == "sensor":
            sensors.append(el)

    if not sensors:
        raise ValueError("No <sensor> elements found in sensorLayout.xml")

    # EGI convention: 1..256 are EEG, 257 is VREF
    sensor_positions: Dict[str, np.ndarray] = {}
    sensor_mapping: Dict[str, int] = {}

    for s in sensors:
        idx_el = _find_child(s, "number")
        loc_el = _find_child(s, "x")
        if idx_el is None:
            continue
        idx = int(idx_el.text)

        # read x/y/z (some files may store as <x><y><z>)
        x_el = _find_child(s, "x")
        y_el = _find_child(s, "y")
        z_el = _find_child(s, "z")
        if x_el is None or y_el is None or z_el is None:
            continue

        x = float(x_el.text) * scale
        y = float(y_el.text) * scale
        z = float(z_el.text) * scale

        if idx == 257:
            ch_name = "EEG VREF"
        else:
            ch_name = f"EEG {idx}"

        sensor_positions[ch_name] = np.array([x, y, z], dtype=float)
        sensor_mapping[ch_name] = idx

    # sanity: require positions for EEG 1..256
    missing = [f"EEG {i}" for i in range(1, 257) if f"EEG {i}" not in sensor_positions]
    if missing:
        raise ValueError(
            f"Missing positions for {len(missing)} EEG channels (examples: {missing[:5]}). "
            "Check the MFF sensorLayout.xml or the channel naming."
        )

    return sensor_positions, sensor_mapping


def load_edf_with_egi_montage(edf_path: Path, mff_path: Path, *, ref_channel: str = "EEG VREF", coordinate_unit: str = "cm") -> mne.io.Raw:
    """Load EDF and attach EGI montage parsed from MFF sensorLayout.xml.

    Mirrors eeg_preprocess_standalone.py behavior:
      - EDF preload
      - channel type mapping (EEG channels are those starting with 'EEG')
      - VREF as misc
      - montage attached with on_missing='raise'
    """
    edf_path = Path(edf_path)
    mff_path = Path(mff_path)

    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")

    # channel type mapping as in standalone
    tmap = {}
    for ch in raw.ch_names:
        if ch == "EDF Annotations":
            tmap[ch] = "stim"
        elif ch == ref_channel:
            tmap[ch] = "misc"
        elif ch.upper().startswith("EEG"):
            tmap[ch] = "eeg"
        else:
            tmap[ch] = "misc"
    raw.set_channel_types(tmap)

    # montage from MFF
    pos, _ = parse_sensor_layout_from_mff(mff_path, unit=coordinate_unit)
    montage = make_dig_montage(ch_pos=pos, coord_frame="head")
    raw.set_montage(montage, on_missing="raise", match_case=False)

    return raw


# -----------------------------------------------------------------------------
# File discovery / pairing utilities
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class RecordingKey:
    visit: str          # e.g. "T0" or "T1"
    stamp: str          # YYYYMMDD_HHMMSS


_EDF_RE = re.compile(
    r"(?P<sub>CROSS_\d+)_T(?P<visit>\d)_(?P<date>\d{8})_(?P<time>\d{6})(?:_fil)?\.edf$",
    re.IGNORECASE,
)

_MFF_RE = re.compile(
    r"(?P<sub>CROSS_\d+)(?:_(?P<sex>[FM]))?_T(?P<visit>\d)_(?P<date>\d{8})_(?P<time>\d{6})\.mff$",
    re.IGNORECASE,
)


def _parse_edf_name(p: Path) -> Optional[RecordingKey]:
    m = _EDF_RE.match(p.name)
    if not m:
        return None
    visit = f"T{m.group('visit')}"
    stamp = f"{m.group('date')}_{m.group('time')}"
    return RecordingKey(visit=visit, stamp=stamp)


def _parse_mff_name(p: Path) -> Optional[RecordingKey]:
    m = _MFF_RE.match(p.name)
    if not m:
        return None
    visit = f"T{m.group('visit')}"
    stamp = f"{m.group('date')}_{m.group('time')}"
    return RecordingKey(visit=visit, stamp=stamp)


def find_patient_recordings(patient_dir: Path) -> Dict[RecordingKey, Tuple[Path, Path]]:
    """Return mapping key -> (edf_path, mff_path).

    Pairs EDF and MFF by (visit, timestamp). If an exact timestamp match isn't
    available, falls back to pairing by visit if there is only one candidate.
    """
    patient_dir = Path(patient_dir)

    edfs: Dict[RecordingKey, Path] = {}
    mffs: Dict[RecordingKey, Path] = {}

    for p in patient_dir.iterdir():
        if p.is_file() and p.suffix.lower() == ".edf":
            k = _parse_edf_name(p)
            if k is not None:
                edfs[k] = p
        if p.suffix.lower() == ".mff" and p.exists():
            # .mff can be a directory (EGI) or a zip file with .mff extension
            k = _parse_mff_name(p)
            if k is not None:
                mffs[k] = p

    pairs: Dict[RecordingKey, Tuple[Path, Path]] = {}

    # first try exact key match
    for k, edf in edfs.items():
        if k in mffs:
            pairs[k] = (edf, mffs[k])

    # fallback: if no exact timestamp match, match by visit when unique
    if not pairs:
        # group by visit
        by_visit_edf: Dict[str, List[Tuple[RecordingKey, Path]]] = {}
        by_visit_mff: Dict[str, List[Tuple[RecordingKey, Path]]] = {}
        for k, p in edfs.items():
            by_visit_edf.setdefault(k.visit, []).append((k, p))
        for k, p in mffs.items():
            by_visit_mff.setdefault(k.visit, []).append((k, p))

        for visit, edf_list in by_visit_edf.items():
            mff_list = by_visit_mff.get(visit, [])
            if len(edf_list) == 1 and len(mff_list) == 1:
                pairs[edf_list[0][0]] = (edf_list[0][1], mff_list[0][1])

    return pairs


# -----------------------------------------------------------------------------
# Main preprocessing run (erp-main functions)
# -----------------------------------------------------------------------------

def run_preprocess_one(
    *,
    edf_path: Path,
    mff_path: Path,
    out_dir: Path,
    erp_main_root: Path,
    coordinate_unit: str = "cm",
    ref_channel: str = "EEG VREF",
    cfg: dict = None,
) -> None:
    """Process a single EDF+MFF pair and write outputs."""

    cfg = dict(DEFAULTS) if cfg is None else cfg

    # Make erp-main importable
    erp_main_root = Path(erp_main_root)
    if not (erp_main_root / "src").exists():
        raise ValueError(
            f"erp-main root must contain 'src/'. Got: {erp_main_root}. "
            "Point this to the extracted 'erp-main/erp-main' folder."
        )
    if str(erp_main_root) not in sys.path:
        sys.path.insert(0, str(erp_main_root))

    # Import erp-main functions (now that sys.path is patched)
    from src.raw.preprocessing import (
        notch_filter,
        bandpass_filter,
        resample_eeg,
        pyprep_reference,
        interpolate_bads,
    )
    from src.raw.ica import run_and_apply_ica
    from src.epochs.epochs import create_epochs_of_fixed_length
    from src.epochs.autoreject import run_autoreject_eeg

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Load (standalone loader)
    raw = load_edf_with_egi_montage(
        edf_path,
        mff_path,
        ref_channel=ref_channel,
        coordinate_unit=coordinate_unit,
    )

    payloads = {}

    # 2) erp-main preprocessing steps (resting-state)
    raw = notch_filter(raw, picks="eeg", freqs=cfg["notch_base_hz"], verbose="ERROR")
    raw = bandpass_filter(
        raw,
        l_freq=cfg["l_freq"],
        h_freq=cfg["h_freq"],
        l_trans_bandwidth=cfg["l_trans_bandwidth"],
        h_trans_bandwidth=cfg["h_trans_bandwidth"],
        method=cfg["filter_method"],
    )

    raw = resample_eeg(raw, sfreq=cfg["sfreq"])

    raw = pyprep_reference(
        raw,
        max_iterations=cfg["pyprep_max_iterations"],
        channel_wise=cfg["pyprep_channel_wise"],
        max_chunk_size=cfg["pyprep_max_chunk_size"],
        seed=cfg["pyprep_seed"],
    )

    raw = interpolate_bads(raw, mode="spline")

    raw, ica_payload = run_and_apply_ica(
        raw,
        hp_cutoff=cfg["ica_hp_cutoff"],
        method=cfg["ica_method"],
        n_components=cfg["ica_n_components"],
        random_state=cfg["ica_random_state"],
        ica_criteria=cfg["ica_criteria"],
    )
    payloads["ica"] = _json_safe_payload(ica_payload["ica_bundle"])

    raw = bandpass_filter(
        raw,
        l_freq=None,
        h_freq=cfg["post_ica_lowpass_hz"],
        h_trans_bandwidth=cfg["h_trans_bandwidth"],
        method=cfg["filter_method"],
    )

    # 3) Resting-state epoching
    epochs, ep_payload = create_epochs_of_fixed_length(
        raw,
        duration=float(cfg["epoch_duration_s"]),
        overlap=float(cfg["epoch_overlap_fraction"]),
        baseline=cfg["epoch_baseline"],
        picks="eeg",
        preload=True,
        verbose="ERROR",
    )
    payloads["epoching"] = _json_safe_payload(ep_payload)

    # 4) AutoReject
    if cfg["use_autoreject"]:
        epochs_clean, ar_payload = run_autoreject_eeg(
            epochs,
            n_interpolate=cfg["ar_n_interpolate"],
            consensus=cfg["ar_consensus"],
            cv=cfg["ar_cv"],
            random_state=cfg["ar_random_state"],
            n_jobs=cfg["ar_n_jobs"],
            return_log=True,
            return_payload=True,
        )
        epochs = epochs_clean
        payloads["autoreject"] = _json_safe_payload(ar_payload["autoreject_bundle"])
        payloads["autoreject_dropped_epochs"] = int(ar_payload["autoreject_reject_log"].bad_epochs.sum())

    # 5) Save
    raw_out = out_dir / "raw_clean.fif"
    epo_out = out_dir / "epochs_clean-epo.fif"
    meta_out = out_dir / "payloads.json"

    raw.save(raw_out, overwrite=True)
    epochs.save(epo_out, overwrite=True)

    with meta_out.open("w", encoding="utf-8") as f:
        json.dump(payloads, f, indent=2)


def _json_safe_payload(obj):
    """Convert numpy types to JSON-safe python types."""
    if isinstance(obj, dict):
        return {k: _json_safe_payload(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe_payload(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Resting-state EEG preprocessing using erp-main pipeline",
        epilog="""
Examples:
  python run_rest_preprocess_egi_erp_main.py
  python run_rest_preprocess_egi_erp_main.py --root data/eeg --out output --erp_main_root erp-main/erp-main
  python run_rest_preprocess_egi_erp_main.py --patients CROSS_001_F CROSS_003_F
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--root",
        type=Path,
        default=Path("data/eeg"),
        help="Root folder containing patient folders (default: data/eeg).",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("output"),
        help="Output root folder (default: output).",
    )
    ap.add_argument(
        "--erp_main_root",
        type=Path,
        default=Path("erp-main/erp-main"),
        help="Path to extracted erp-main repo root that contains src/ (default: erp-main/erp-main).",
    )
    ap.add_argument(
        "--coordinate_unit",
        default="cm",
        choices=["m", "cm", "mm"],
        help="Unit used in sensorLayout.xml (default: cm).",
    )
    ap.add_argument(
        "--ref_channel",
        default="EEG VREF",
        help="Reference channel name in EDF (default: EEG VREF).",
    )
    ap.add_argument(
        "--patients",
        nargs="*",
        default=None,
        help="Optional list of specific patient folder names to process (e.g., CROSS_001_F CROSS_003_F).",
    )

    args = ap.parse_args()

    root = args.root
    out_root = args.out
    erp_main_root = args.erp_main_root

    # Validate paths and provide helpful error messages
    if not root.exists():
        print(f"[ERROR] Root folder not found: {root}")
        print(f"        Please create the folder or specify --root /path/to/data")
        return 1
    
    if not erp_main_root.exists():
        print(f"[ERROR] erp-main root folder not found: {erp_main_root}")
        print(f"        Please extract erp-main repo or specify --erp_main_root /path/to/erp-main")
        return 1
    
    erp_main_src = erp_main_root / "src"
    if not erp_main_src.exists():
        print(f"[ERROR] erp-main 'src' folder not found at: {erp_main_src}")
        print(f"        Make sure the path points to the extracted erp-main root that contains src/")
        return 1

    patient_dirs = [p for p in root.iterdir() if p.is_dir()]
    if args.patients:
        wanted = set(args.patients)
        patient_dirs = [p for p in patient_dirs if p.name in wanted]

    if not patient_dirs:
        print("No patient folders found.")
        return 1

    for patient_dir in sorted(patient_dirs):
        pairs = find_patient_recordings(patient_dir)
        if not pairs:
            print(f"[WARN] No valid EDF/MFF pairs found in {patient_dir}")
            continue

        for key, (edf_path, mff_path) in sorted(pairs.items(), key=lambda kv: (kv[0].visit, kv[0].stamp)):
            visit_out = out_root / patient_dir.name / key.visit
            print(f"[INFO] {patient_dir.name} {key.visit} {key.stamp}")
            print(f"       EDF: {edf_path.name}")
            print(f"       MFF: {mff_path.name}")

            run_preprocess_one(
                edf_path=edf_path,
                mff_path=mff_path,
                out_dir=visit_out,
                erp_main_root=erp_main_root,
                coordinate_unit=args.coordinate_unit,
                ref_channel=args.ref_channel,
            )

    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
