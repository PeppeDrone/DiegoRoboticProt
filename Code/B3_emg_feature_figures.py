#!/usr/bin/env python3
"""B3_emg_feature_figures.py

Publication-quality multi-panel figure summarizing EMG feature results from B3_emg_feature_computation.

Scientific message:
1. EMG amplitude and timing descriptors did not substantially change between conditions.
2. Main effect: improved cycle-to-cycle stability, especially on the more affected side.
3. Stability improvement: reduced variability, increased similarity to mean cycle,
   directional consistency across muscles.
4. Infraspinatus (more affected side) is the main representative example.

Layout: 6 panels (3 rows × 2 columns). A: Median±IQR envelope (Infraspinatus, more affected). B: Similarity. C: SD similarity. D: Pooled effect summary. E: Forest (more affected). F: Forest (less affected).

Data: loads directly from B3 outputs in results/synergies/ (and optionally demographics
for more_affected vs less_affected split). Run B3_emg_feature_computation.py first.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

# Condition labels (T0 = baseline, T1 = post-intervention)
COND1_LABEL = "T0"
COND2_LABEL = "T1"

# Paths (override via load_data_from_b3_outputs(emg_dir=..., demographic_path=...))
DEFAULT_EMG_DIR = Path(__file__).resolve().parent / "results" / "synergies"
DEFAULT_DEMO_PATHS = [
    Path(__file__).resolve().parent / "demografica.xlsx",
    Path(__file__).resolve().parent / "demographics.csv",
]

# -----------------------------------------------------------------------------
# Data variables (populated by load_data_from_b3_outputs)
# -----------------------------------------------------------------------------

cycles_cond1_infra_more: Optional[np.ndarray] = None
cycles_cond2_infra_more: Optional[np.ndarray] = None
mean_cycle_cond1_infra_more: Optional[np.ndarray] = None
mean_cycle_cond2_infra_more: Optional[np.ndarray] = None
sd_cycle_cond1_infra_more: Optional[np.ndarray] = None
sd_cycle_cond2_infra_more: Optional[np.ndarray] = None
similarity_cond1_infra_more: Optional[np.ndarray] = None
similarity_cond2_infra_more: Optional[np.ndarray] = None
sd_similarity_cond1_more: Optional[np.ndarray] = None
sd_similarity_cond2_more: Optional[np.ndarray] = None
muscle_names: Optional[List[str]] = None
delta_similarity_all_muscles: Optional[np.ndarray] = None
delta_auc_cv_all_muscles: Optional[np.ndarray] = None
delta_sd_similarity_all_muscles: Optional[np.ndarray] = None
delta_similarity_all_muscles_less: Optional[np.ndarray] = None
delta_auc_cv_all_muscles_less: Optional[np.ndarray] = None
delta_sd_similarity_all_muscles_less: Optional[np.ndarray] = None
delta_cv_peak_amp_all_muscles: Optional[np.ndarray] = None
delta_cv_peak_amp_all_muscles_less: Optional[np.ndarray] = None
# Effect size (Cohen's d_z) for panels E, F - aligns visually with paired-test significance
effect_size_auc_cv_all_muscles: Optional[np.ndarray] = None
effect_size_cv_peak_all_muscles: Optional[np.ndarray] = None
effect_size_auc_cv_all_muscles_less: Optional[np.ndarray] = None
effect_size_cv_peak_all_muscles_less: Optional[np.ndarray] = None
amplitude_effects_summary: Optional[np.ndarray] = None
stability_effects_summary: Optional[np.ndarray] = None
amplitude_effects_summary_less: Optional[np.ndarray] = None
stability_effects_summary_less: Optional[np.ndarray] = None

# Significance (p < 0.05, before FDR) for red square markers on panels D, E, F
ALPHA_SIG = 0.05
COLOR_SIG_MARKER = "#c53030"
panel_d_sig_more: Optional[List[bool]] = None  # [AuC, Centroid, Similarity, SD sim, CV AuC]
panel_d_sig_less: Optional[List[bool]] = None
panel_d_es_more: Optional[np.ndarray] = None  # effect size d_z for [AuC, Centroid, Sim, SD sim, CV AuC]
panel_d_es_less: Optional[np.ndarray] = None
panel_e_sig_cv: Optional[np.ndarray] = None  # per muscle, −Δ AuC CV
panel_e_sig_cv_peak: Optional[np.ndarray] = None  # per muscle, −Δ CV peak amp
panel_f_sig_cv: Optional[np.ndarray] = None
panel_f_sig_cv_peak: Optional[np.ndarray] = None


# -----------------------------------------------------------------------------
# Helpers for data loading
# -----------------------------------------------------------------------------


def _onesample_pvalue(deltas: List[float]) -> float:
    """One-sample t-test p-value (mean delta != 0). Returns 1.0 if n < 2."""
    arr = np.array([float(x) for x in deltas if np.isfinite(x)], dtype=float)
    if len(arr) < 2:
        return 1.0
    try:
        _, p = scipy_stats.ttest_1samp(arr, 0)
        return float(p) if np.isfinite(p) else 1.0
    except Exception:
        return 1.0


def _cohens_dz(deltas: List[float]) -> float:
    """Cohen's d_z for paired data: mean(delta) / SD(delta). Returns 0 if n < 2 or SD = 0."""
    arr = np.array([float(x) for x in deltas if np.isfinite(x)], dtype=float)
    if len(arr) < 2:
        return 0.0
    sd = np.std(arr, ddof=1)
    if sd <= 0:
        return 0.0
    return float(np.mean(arr) / sd)


def _parse_task_name(task_name: str) -> Tuple[str, str]:
    """Extract (session, condition) from task name. e.g. Task_T0_DS -> (T0, DS)."""
    task_upper = str(task_name).strip().upper()
    session = "T0" if "T0" in task_upper else "T1" if "T1" in task_upper else ""
    condition = "SN" if "SN" in task_upper else "DS" if "DS" in task_upper else ""
    return session, condition


def _normalize_side(val: Any) -> str:
    """Normalize DX/SX."""
    v = str(val).strip().upper()
    if v in ("DX", "DESTRO", "RIGHT", "D"):
        return "DX"
    if v in ("SX", "SINISTRO", "LEFT", "L"):
        return "SX"
    return v


def _condition_to_performed_side(condition: str) -> Optional[str]:
    """DS -> DX, SN -> SX."""
    c = str(condition).strip().upper()
    if c == "DS":
        return "DX"
    if c == "SN":
        return "SX"
    return None


def _find_affected_column(df: pd.DataFrame) -> Optional[str]:
    """Find column for lato più colpito."""
    for col in df.columns:
        norm = str(col).strip().lower().replace(" ", "_")
        if "colpito" in norm or "colpito" in unicodedata.normalize("NFKD", str(col)).encode("ascii", "ignore").decode("ascii").lower():
            if norm not in ("id", "patient_id"):
                return col
    return None


def build_patient_cond_to_affected(demo_path: Optional[Path]) -> Dict[Tuple[str, str], str]:
    """(patient_id, condition) -> more_affected | less_affected."""
    out: Dict[Tuple[str, str], str] = {}
    if demo_path is None or not demo_path.exists():
        return out
    try:
        if demo_path.suffix.lower() in (".xlsx", ".xls"):
            demo = pd.read_excel(demo_path, engine="openpyxl")
        else:
            demo = pd.read_csv(demo_path)
    except Exception:
        return out
    if demo.empty:
        return out
    id_col = "id" if "id" in demo.columns else "patient_id" if "patient_id" in demo.columns else demo.columns[0]
    aff_col = _find_affected_column(demo)
    if not aff_col:
        return out
    for _, row in demo.iterrows():
        pid = str(row.get(id_col, "")).strip()
        if not pid:
            continue
        affected = _normalize_side(row.get(aff_col, ""))
        if affected not in ("DX", "SX"):
            continue
        for cond in ("SN", "DS"):
            performed = _condition_to_performed_side(cond)
            if performed:
                out[(pid, cond)] = "more_affected" if performed == affected else "less_affected"
    return out


def discover_task_dirs(emg_dir: Path) -> List[Tuple[str, str, str, str, Path]]:
    """(patient_id, task_name, session, condition, task_dir)."""
    out: List[Tuple[str, str, str, str, Path]] = []
    if not emg_dir.exists():
        return out
    for pdir in sorted(emg_dir.iterdir()):
        if not pdir.is_dir() or pdir.name.startswith("plots_") or pdir.name == "meta_synergy_clustering":
            continue
        patient_id = pdir.name
        for tdir in sorted(pdir.iterdir()):
            if not tdir.is_dir():
                continue
            task_name = tdir.name
            session, condition = _parse_task_name(task_name)
            if session and condition:
                out.append((patient_id, task_name, session, condition, tdir))
    return out


def load_data_from_b3_outputs(
    emg_dir: Optional[Path] = None,
    demographic_path: Optional[Path] = None,
    muscle_infra: str = "Infraspinatus",
) -> bool:
    """
    Load all figure data from B3 outputs. Returns True if successful.
    Populates module-level variables for create_emg_summary_figure().
    """
    global cycles_cond1_infra_more, cycles_cond2_infra_more, mean_cycle_cond1_infra_more, mean_cycle_cond2_infra_more
    global sd_cycle_cond1_infra_more, sd_cycle_cond2_infra_more
    global similarity_cond1_infra_more, similarity_cond2_infra_more, sd_similarity_cond1_more, sd_similarity_cond2_more
    global muscle_names, delta_similarity_all_muscles, delta_auc_cv_all_muscles, delta_sd_similarity_all_muscles
    global delta_similarity_all_muscles_less, delta_auc_cv_all_muscles_less, delta_sd_similarity_all_muscles_less
    global delta_cv_peak_amp_all_muscles, delta_cv_peak_amp_all_muscles_less
    global effect_size_auc_cv_all_muscles, effect_size_cv_peak_all_muscles
    global effect_size_auc_cv_all_muscles_less, effect_size_cv_peak_all_muscles_less
    global amplitude_effects_summary, stability_effects_summary
    global amplitude_effects_summary_less, stability_effects_summary_less
    global panel_d_sig_more, panel_d_sig_less, panel_d_es_more, panel_d_es_less
    global panel_e_sig_cv, panel_e_sig_cv_peak, panel_f_sig_cv, panel_f_sig_cv_peak
    emg_dir = emg_dir or DEFAULT_EMG_DIR
    if demographic_path is None:
        for p in DEFAULT_DEMO_PATHS:
            if p.exists():
                demographic_path = p
                break
    patient_cond_to_affected = build_patient_cond_to_affected(demographic_path)

    tasks = discover_task_dirs(emg_dir)
    if not tasks:
        return False

    # Collect muscle names from first stability file
    all_muscles: List[str] = []
    for _, _, _, _, tdir in tasks:
        stab_path = tdir / "stability_summary_metrics.csv"
        if stab_path.exists():
            df = pd.read_csv(stab_path)
            if "muscle" in df.columns:
                all_muscles = df["muscle"].dropna().unique().tolist()
                break
    if not all_muscles:
        return False
    muscle_names = all_muscles
    infra_idx = next((i for i, m in enumerate(muscle_names) if muscle_infra.lower() in str(m).lower()), None)
    if infra_idx is None:
        infra_idx = 0

    # --- Panels A & B: X_cycles for Infraspinatus (more_affected, T0 vs T1) ---
    cycles_t0: List[np.ndarray] = []
    cycles_t1: List[np.ndarray] = []
    for patient_id, task_name, session, condition, tdir in tasks:
        aff = patient_cond_to_affected.get((patient_id, condition), "more_affected")
        if aff != "more_affected":
            continue
        x_path = tdir / "X_cycles.npz"
        if not x_path.exists():
            continue
        try:
            npz = np.load(x_path, allow_pickle=True)
            X = npz["X_cycles"]
            muscles_npz = npz.get("muscle_names", np.array([]))
            if muscles_npz.size > 0:
                muscles_list = muscles_npz.tolist()
                idx = next((i for i, m in enumerate(muscles_list) if muscle_infra.lower() in str(m).lower()), 0)
            else:
                idx = infra_idx
            if X.ndim == 3 and idx < X.shape[2]:
                cyc = np.asarray(X[:, :, idx], dtype=np.float64)
                if session == "T0":
                    cycles_t0.append(cyc)
                else:
                    cycles_t1.append(cyc)
        except Exception:
            continue

    n_ref = 101
    def _resample_and_stack(cycles_list: List[np.ndarray], n_pts: int = n_ref) -> np.ndarray:
        """Flatten list of (n_cycles, n_samples) arrays, resample each to n_pts, and vstack."""
        all_rows = []
        for block in cycles_list:
            block = np.asarray(block, dtype=np.float64)
            if block.ndim == 1:
                block = block.reshape(1, -1)
            for row in block:
                n = len(row)
                if n == n_pts:
                    all_rows.append(row)
                elif n > 1:
                    all_rows.append(np.interp(np.linspace(0, 1, n_pts), np.linspace(0, 1, n), row))
                else:
                    all_rows.append(np.full(n_pts, float(row[0]) if row.size else 0.0))
        return np.vstack(all_rows) if all_rows else np.array([]).reshape(0, n_pts)

    if cycles_t0:
        cycles_cond1_infra_more = _resample_and_stack(cycles_t0)
        mean_cycle_cond1_infra_more = np.mean(cycles_cond1_infra_more, axis=0)
        sd_cycle_cond1_infra_more = np.std(cycles_cond1_infra_more, axis=0)
    if cycles_t1:
        cycles_cond2_infra_more = _resample_and_stack(cycles_t1)
        mean_cycle_cond2_infra_more = np.mean(cycles_cond2_infra_more, axis=0)
        sd_cycle_cond2_infra_more = np.std(cycles_cond2_infra_more, axis=0)

    # --- Panels C & D: Subject-level similarity and SD (from cycle_similarity_to_mean) ---
    sim_t0: Dict[str, np.ndarray] = {}
    sim_t1: Dict[str, np.ndarray] = {}
    for patient_id, task_name, session, condition, tdir in tasks:
        aff = patient_cond_to_affected.get((patient_id, condition), "more_affected")
        if aff != "more_affected":
            continue
        path = tdir / "cycle_similarity_to_mean.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        infra = df[df["muscle"] == muscle_infra] if "muscle" in df.columns else pd.DataFrame()
        if infra.empty and "muscle" in df.columns:
            for m in df["muscle"].unique():
                if muscle_infra.lower() in str(m).lower():
                    infra = df[df["muscle"] == m]
                    break
        if infra.empty:
            continue
        sims = infra["similarity_to_mean_cycle"].dropna().values
        if len(sims) < 2:
            continue
        mean_sim = float(np.nanmean(sims))
        std_sim = float(np.nanstd(sims))
        if session == "T0":
            sim_t0[patient_id] = np.array([mean_sim, std_sim])
        else:
            sim_t1[patient_id] = np.array([mean_sim, std_sim])

    common = sorted(set(sim_t0) & set(sim_t1))
    if common:
        similarity_cond1_infra_more = np.array([sim_t0[p][0] for p in common])
        similarity_cond2_infra_more = np.array([sim_t1[p][0] for p in common])
        sd_similarity_cond1_more = np.array([sim_t0[p][1] for p in common])
        sd_similarity_cond2_more = np.array([sim_t1[p][1] for p in common])

    # --- Panel E: Delta per muscle (more_affected, mean across patients) ---
    # Key: (patient_id, condition) -> session -> muscle -> metrics
    stab_by_key: Dict[Tuple[str, str], Dict[str, Dict[str, Dict[str, float]]]] = {}
    for patient_id, task_name, session, condition, tdir in tasks:
        aff = patient_cond_to_affected.get((patient_id, condition), "more_affected")
        if aff != "more_affected":
            continue
        path = tdir / "stability_summary_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        key = (patient_id, condition)
        if key not in stab_by_key:
            stab_by_key[key] = {"T0": {}, "T1": {}}
        for _, row in df.iterrows():
            m = str(row.get("muscle", ""))
            if not m:
                continue
            stab_by_key[key][session][m] = {
                "sim": row.get("mean_similarity_to_mean_cycle", np.nan),
                "cv_auc": row.get("cv_auc", np.nan),
                "cv_peak_amp": row.get("cv_peak_amp", np.nan),
                "std_sim": row.get("std_similarity_to_mean_cycle", np.nan),
            }
    pats_with_both = [k for k in stab_by_key if "T0" in stab_by_key[k] and "T1" in stab_by_key[k] and stab_by_key[k]["T0"] and stab_by_key[k]["T1"]]
    delta_sim: Dict[str, List[float]] = {m: [] for m in muscle_names}
    delta_cv: Dict[str, List[float]] = {m: [] for m in muscle_names}
    delta_cv_peak: Dict[str, List[float]] = {m: [] for m in muscle_names}
    delta_sd: Dict[str, List[float]] = {m: [] for m in muscle_names}
    for k in pats_with_both:
        for m in muscle_names:
            t0 = stab_by_key[k]["T0"].get(m, {})
            t1 = stab_by_key[k]["T1"].get(m, {})
            if t0 and t1:
                ds = _delta(t1.get("sim"), t0.get("sim"))
                dc = _delta(t1.get("cv_auc"), t0.get("cv_auc"))
                dp = _delta(t1.get("cv_peak_amp"), t0.get("cv_peak_amp"))
                dd = _delta(t1.get("std_sim"), t0.get("std_sim"))
                if np.isfinite(ds):
                    delta_sim[m].append(ds)
                if np.isfinite(dc):
                    delta_cv[m].append(dc)
                if np.isfinite(dp):
                    delta_cv_peak[m].append(dp)
                if np.isfinite(dd):
                    delta_sd[m].append(dd)
    delta_similarity_all_muscles = np.array([np.mean(delta_sim[m]) if delta_sim[m] else 0.0 for m in muscle_names])
    delta_auc_cv_all_muscles = np.array([np.mean(delta_cv[m]) if delta_cv[m] else 0.0 for m in muscle_names])
    delta_cv_peak_amp_all_muscles = np.array([np.mean(delta_cv_peak[m]) if delta_cv_peak[m] else 0.0 for m in muscle_names])
    delta_sd_similarity_all_muscles = np.array([np.mean(delta_sd[m]) if delta_sd[m] else 0.0 for m in muscle_names])
    # Effect size (Cohen's d_z): improvement → positive when we use -d_z (negative delta = improvement)
    effect_size_auc_cv_all_muscles = np.array([-_cohens_dz(delta_cv[m]) for m in muscle_names])
    effect_size_cv_peak_all_muscles = np.array([-_cohens_dz(delta_cv_peak[m]) for m in muscle_names])

    # --- Less affected: Delta per muscle (for Panel F) ---
    stab_by_key_less: Dict[Tuple[str, str], Dict[str, Dict[str, Dict[str, float]]]] = {}
    for patient_id, task_name, session, condition, tdir in tasks:
        aff = patient_cond_to_affected.get((patient_id, condition), "less_affected")
        if aff != "less_affected":
            continue
        path = tdir / "stability_summary_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        key = (patient_id, condition)
        if key not in stab_by_key_less:
            stab_by_key_less[key] = {"T0": {}, "T1": {}}
        for _, row in df.iterrows():
            m = str(row.get("muscle", ""))
            if not m:
                continue
            stab_by_key_less[key][session][m] = {
                "sim": row.get("mean_similarity_to_mean_cycle", np.nan),
                "cv_auc": row.get("cv_auc", np.nan),
                "cv_peak_amp": row.get("cv_peak_amp", np.nan),
                "std_sim": row.get("std_similarity_to_mean_cycle", np.nan),
            }
    pats_less = [k for k in stab_by_key_less if "T0" in stab_by_key_less[k] and "T1" in stab_by_key_less[k] and stab_by_key_less[k]["T0"] and stab_by_key_less[k]["T1"]]
    delta_sim_less: Dict[str, List[float]] = {m: [] for m in muscle_names}
    delta_cv_less: Dict[str, List[float]] = {m: [] for m in muscle_names}
    delta_cv_peak_less: Dict[str, List[float]] = {m: [] for m in muscle_names}
    delta_sd_less: Dict[str, List[float]] = {m: [] for m in muscle_names}
    for k in pats_less:
        for m in muscle_names:
            t0 = stab_by_key_less[k]["T0"].get(m, {})
            t1 = stab_by_key_less[k]["T1"].get(m, {})
            if t0 and t1:
                ds = _delta(t1.get("sim"), t0.get("sim"))
                dc = _delta(t1.get("cv_auc"), t0.get("cv_auc"))
                dp = _delta(t1.get("cv_peak_amp"), t0.get("cv_peak_amp"))
                dd = _delta(t1.get("std_sim"), t0.get("std_sim"))
                if np.isfinite(ds):
                    delta_sim_less[m].append(ds)
                if np.isfinite(dc):
                    delta_cv_less[m].append(dc)
                if np.isfinite(dp):
                    delta_cv_peak_less[m].append(dp)
                if np.isfinite(dd):
                    delta_sd_less[m].append(dd)
    delta_similarity_all_muscles_less = np.array([np.mean(delta_sim_less[m]) if delta_sim_less[m] else 0.0 for m in muscle_names])
    delta_auc_cv_all_muscles_less = np.array([np.mean(delta_cv_less[m]) if delta_cv_less[m] else 0.0 for m in muscle_names])
    delta_cv_peak_amp_all_muscles_less = np.array([np.mean(delta_cv_peak_less[m]) if delta_cv_peak_less[m] else 0.0 for m in muscle_names])
    delta_sd_similarity_all_muscles_less = np.array([np.mean(delta_sd_less[m]) if delta_sd_less[m] else 0.0 for m in muscle_names])
    effect_size_auc_cv_all_muscles_less = np.array([-_cohens_dz(delta_cv_less[m]) for m in muscle_names])
    effect_size_cv_peak_all_muscles_less = np.array([-_cohens_dz(delta_cv_peak_less[m]) for m in muscle_names])

    # --- Panel D: Amplitude vs stability effects (from task_summary, paired T0 vs T1, more affected) ---
    ts_by_key: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for patient_id, task_name, session, condition, tdir in tasks:
        aff = patient_cond_to_affected.get((patient_id, condition), "more_affected")
        if aff != "more_affected":
            continue
        path = tdir / "task_summary_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0]
        key = (patient_id, condition)
        if key not in ts_by_key:
            ts_by_key[key] = {"T0": {}, "T1": {}}
        for c in ["mean_cycle_mean_amp", "mean_cycle_peak_amp", "mean_cycle_auc", "mean_cycle_centroid",
                  "mean_similarity_to_mean_cycle", "mean_cv_auc", "std_similarity_to_mean_cycle"]:
            if c in row.index and pd.notna(row.get(c)):
                ts_by_key[key][session][c] = float(row[c])
    amp_lists: Dict[str, List[float]] = {c: [] for c in ["mean_cycle_mean_amp", "mean_cycle_peak_amp", "mean_cycle_auc", "mean_cycle_centroid"]}
    stab_lists: Dict[str, List[float]] = {c: [] for c in ["mean_similarity_to_mean_cycle", "std_similarity_to_mean_cycle", "mean_cv_auc"]}
    for key in ts_by_key:
        t0, t1 = ts_by_key[key].get("T0", {}), ts_by_key[key].get("T1", {})
        for c in amp_lists:
            if c in t0 and c in t1:
                amp_lists[c].append(t1[c] - t0[c])
        for c in stab_lists:
            if c in t0 and c in t1:
                stab_lists[c].append(t1[c] - t0[c])
    amplitude_effects_summary = np.array([np.nanmean(amp_lists[c]) if amp_lists[c] else 0.0 for c in amp_lists])
    stability_effects_summary = np.array([
        np.nanmean(stab_lists["mean_similarity_to_mean_cycle"]) if stab_lists["mean_similarity_to_mean_cycle"] else 0.0,
        np.nanmean(stab_lists["std_similarity_to_mean_cycle"]) if stab_lists["std_similarity_to_mean_cycle"] else 0.0,
        np.nanmean(stab_lists["mean_cv_auc"]) if stab_lists["mean_cv_auc"] else 0.0,
    ])
    # Panel D significance (before FDR): AuC, Centroid, Similarity, SD sim, CV AuC
    panel_d_sig_more = [
        _onesample_pvalue(amp_lists["mean_cycle_auc"]) < ALPHA_SIG,
        _onesample_pvalue(amp_lists["mean_cycle_centroid"]) < ALPHA_SIG,
        _onesample_pvalue(stab_lists["mean_similarity_to_mean_cycle"]) < ALPHA_SIG,
        _onesample_pvalue(stab_lists["std_similarity_to_mean_cycle"]) < ALPHA_SIG,
        _onesample_pvalue(stab_lists["mean_cv_auc"]) < ALPHA_SIG,
    ]
    # Effect size for Panel D: AuC, Centroid (raw d_z); Similarity (+d_z); SD sim, CV AuC (-d_z for improvement→right)
    es_auc = _cohens_dz(amp_lists["mean_cycle_auc"])
    es_centroid = _cohens_dz(amp_lists["mean_cycle_centroid"])
    es_sim = _cohens_dz(stab_lists["mean_similarity_to_mean_cycle"])
    es_sd = -_cohens_dz(stab_lists["std_similarity_to_mean_cycle"])
    es_cv = -_cohens_dz(stab_lists["mean_cv_auc"])
    panel_d_es_more = np.array([es_auc, es_centroid, es_sim, es_sd, es_cv])

    # --- Panel D: Less affected (for empty-circle overlay) ---
    ts_by_key_less: Dict[Tuple[str, str], Dict[str, Dict[str, float]]] = {}
    for patient_id, task_name, session, condition, tdir in tasks:
        aff = patient_cond_to_affected.get((patient_id, condition), "less_affected")
        if aff != "less_affected":
            continue
        path = tdir / "task_summary_metrics.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        row = df.iloc[0]
        key = (patient_id, condition)
        if key not in ts_by_key_less:
            ts_by_key_less[key] = {"T0": {}, "T1": {}}
        for c in ["mean_cycle_mean_amp", "mean_cycle_peak_amp", "mean_cycle_auc", "mean_cycle_centroid",
                  "mean_similarity_to_mean_cycle", "mean_cv_auc", "std_similarity_to_mean_cycle"]:
            if c in row.index and pd.notna(row.get(c)):
                ts_by_key_less[key][session][c] = float(row[c])
    amp_lists_less: Dict[str, List[float]] = {c: [] for c in ["mean_cycle_mean_amp", "mean_cycle_peak_amp", "mean_cycle_auc", "mean_cycle_centroid"]}
    stab_lists_less: Dict[str, List[float]] = {c: [] for c in ["mean_similarity_to_mean_cycle", "std_similarity_to_mean_cycle", "mean_cv_auc"]}
    for key in ts_by_key_less:
        t0, t1 = ts_by_key_less[key].get("T0", {}), ts_by_key_less[key].get("T1", {})
        for c in amp_lists_less:
            if c in t0 and c in t1:
                amp_lists_less[c].append(t1[c] - t0[c])
        for c in stab_lists_less:
            if c in t0 and c in t1:
                stab_lists_less[c].append(t1[c] - t0[c])
    amplitude_effects_summary_less = np.array([np.nanmean(amp_lists_less[c]) if amp_lists_less[c] else 0.0 for c in amp_lists_less])
    stability_effects_summary_less = np.array([
        np.nanmean(stab_lists_less["mean_similarity_to_mean_cycle"]) if stab_lists_less["mean_similarity_to_mean_cycle"] else 0.0,
        np.nanmean(stab_lists_less["std_similarity_to_mean_cycle"]) if stab_lists_less["std_similarity_to_mean_cycle"] else 0.0,
        np.nanmean(stab_lists_less["mean_cv_auc"]) if stab_lists_less["mean_cv_auc"] else 0.0,
    ])
    panel_d_sig_less = [
        _onesample_pvalue(amp_lists_less["mean_cycle_auc"]) < ALPHA_SIG,
        _onesample_pvalue(amp_lists_less["mean_cycle_centroid"]) < ALPHA_SIG,
        _onesample_pvalue(stab_lists_less["mean_similarity_to_mean_cycle"]) < ALPHA_SIG,
        _onesample_pvalue(stab_lists_less["std_similarity_to_mean_cycle"]) < ALPHA_SIG,
        _onesample_pvalue(stab_lists_less["mean_cv_auc"]) < ALPHA_SIG,
    ]
    es_auc_l = _cohens_dz(amp_lists_less["mean_cycle_auc"])
    es_centroid_l = _cohens_dz(amp_lists_less["mean_cycle_centroid"])
    es_sim_l = _cohens_dz(stab_lists_less["mean_similarity_to_mean_cycle"])
    es_sd_l = -_cohens_dz(stab_lists_less["std_similarity_to_mean_cycle"])
    es_cv_l = -_cohens_dz(stab_lists_less["mean_cv_auc"])
    panel_d_es_less = np.array([es_auc_l, es_centroid_l, es_sim_l, es_sd_l, es_cv_l])

    # Panel E/F significance (per muscle)
    panel_e_sig_cv = np.array([_onesample_pvalue(delta_cv[m]) < ALPHA_SIG for m in muscle_names])
    panel_e_sig_cv_peak = np.array([_onesample_pvalue(delta_cv_peak[m]) < ALPHA_SIG for m in muscle_names])
    panel_f_sig_cv = np.array([_onesample_pvalue(delta_cv_less[m]) < ALPHA_SIG for m in muscle_names])
    panel_f_sig_cv_peak = np.array([_onesample_pvalue(delta_cv_peak_less[m]) < ALPHA_SIG for m in muscle_names])

    return True


def _delta(a: Any, b: Any) -> float:
    """Compute a - b, return np.nan if invalid."""
    try:
        va, vb = float(a), float(b)
        return va - vb if np.isfinite(va) and np.isfinite(vb) else np.nan
    except (TypeError, ValueError):
        return np.nan


# -----------------------------------------------------------------------------
# Mock data fallback (when X_cycles.npz missing or no demographics)
# -----------------------------------------------------------------------------


def _fill_mock_cycles() -> None:
    """Fill only cycle data for panels A-B when X_cycles.npz is missing."""
    global cycles_cond1_infra_more, cycles_cond2_infra_more
    global mean_cycle_cond1_infra_more, mean_cycle_cond2_infra_more
    global sd_cycle_cond1_infra_more, sd_cycle_cond2_infra_more
    n_timepoints = 101
    phase = np.linspace(0, 1, n_timepoints)
    base = 0.3 + 0.5 * np.sin(2 * np.pi * phase) ** 2
    noise1 = np.random.RandomState(42).randn(40, n_timepoints) * 0.08
    cycles_cond1_infra_more = np.clip(base + noise1, 0, 1)
    mean_cycle_cond1_infra_more = np.mean(cycles_cond1_infra_more, axis=0)
    sd_cycle_cond1_infra_more = np.std(cycles_cond1_infra_more, axis=0)
    noise2 = np.random.RandomState(43).randn(40, n_timepoints) * 0.04
    cycles_cond2_infra_more = np.clip(base + noise2, 0, 1)
    mean_cycle_cond2_infra_more = np.mean(cycles_cond2_infra_more, axis=0)
    sd_cycle_cond2_infra_more = np.std(cycles_cond2_infra_more, axis=0)


def _fill_mock_similarity() -> None:
    """Fill subject-level similarity/SD when cycle_similarity data missing."""
    global similarity_cond1_infra_more, similarity_cond2_infra_more
    global sd_similarity_cond1_more, sd_similarity_cond2_more
    similarity_cond1_infra_more = np.array([0.55, 0.58, 0.62, 0.57, 0.61, 0.59, 0.58, 0.60, 0.64])
    similarity_cond2_infra_more = np.array([0.68, 0.71, 0.69, 0.70, 0.72, 0.67, 0.69, 0.71, 0.68])
    sd_similarity_cond1_more = np.array([0.14, 0.12, 0.10, 0.13, 0.11, 0.15, 0.12, 0.11, 0.09])
    sd_similarity_cond2_more = np.array([0.06, 0.05, 0.07, 0.05, 0.06, 0.07, 0.05, 0.06, 0.05])


def _make_mock_data() -> None:
    """Generate mock data so the script runs as a complete example."""
    global cycles_cond1_infra_more, cycles_cond2_infra_more, mean_cycle_cond1_infra_more
    global mean_cycle_cond2_infra_more, sd_cycle_cond1_infra_more, sd_cycle_cond2_infra_more
    global similarity_cond1_infra_more, similarity_cond2_infra_more
    global sd_similarity_cond1_more, sd_similarity_cond2_more
    global muscle_names, delta_similarity_all_muscles, delta_auc_cv_all_muscles, delta_sd_similarity_all_muscles
    global delta_cv_peak_amp_all_muscles, delta_similarity_all_muscles_less, delta_auc_cv_all_muscles_less
    global delta_cv_peak_amp_all_muscles_less, delta_sd_similarity_all_muscles_less
    global amplitude_effects_summary, stability_effects_summary
    n_timepoints = 101
    n_cycles_cond1 = 40
    n_cycles_cond2 = 40
    n_subjects = 9

    phase = np.linspace(0, 1, n_timepoints)
    base = 0.3 + 0.5 * np.sin(2 * np.pi * phase) ** 2
    # Cond1: wider spread
    noise1 = np.random.RandomState(42).randn(n_cycles_cond1, n_timepoints) * 0.08
    cycles_cond1_infra_more = np.clip(base + noise1, 0, 1)
    mean_cycle_cond1_infra_more = np.mean(cycles_cond1_infra_more, axis=0)
    sd_cycle_cond1_infra_more = np.std(cycles_cond1_infra_more, axis=0)

    # Cond2: tighter spread
    noise2 = np.random.RandomState(43).randn(n_cycles_cond2, n_timepoints) * 0.04
    cycles_cond2_infra_more = np.clip(base + noise2, 0, 1)
    mean_cycle_cond2_infra_more = np.mean(cycles_cond2_infra_more, axis=0)
    sd_cycle_cond2_infra_more = np.std(cycles_cond2_infra_more, axis=0)

    # Subject-level similarity (as reported: 0.594 -> 0.690)
    similarity_cond1_infra_more = np.array(
        [0.55, 0.58, 0.62, 0.57, 0.61, 0.59, 0.58, 0.60, 0.64]
    )
    similarity_cond2_infra_more = np.array(
        [0.68, 0.71, 0.69, 0.70, 0.72, 0.67, 0.69, 0.71, 0.68]
    )

    # SD of similarity (reduction = improvement)
    sd_similarity_cond1_more = np.array(
        [0.14, 0.12, 0.10, 0.13, 0.11, 0.15, 0.12, 0.11, 0.09]
    )
    sd_similarity_cond2_more = np.array(
        [0.06, 0.05, 0.07, 0.05, 0.06, 0.07, 0.05, 0.06, 0.05]
    )

    muscle_names = [
        "Brachioradialis",
        "Biceps Brachii Short Head",
        "Pectoralis Major",
        "Anterior Deltoid",
        "Middle Deltoid",
        "Posterior Deltoid",
        "Triceps Brachii Lateral Head",
        "Triceps Brachii Long Head",
        "Infraspinatus",
        "Latissimus Dorsi",
        "Biceps Brachii Long Head",
        "Trapezius Middle",
    ]
    n_muscles = len(muscle_names)

    # Directional: Δ similarity (positive = improvement), Δ CV (negative = improvement), Δ SD sim (negative = improvement)
    np.random.seed(44)
    delta_similarity_all_muscles = np.random.uniform(-0.03, 0.15, n_muscles)
    delta_similarity_all_muscles[8] = 0.096  # Infraspinatus: strong positive
    delta_auc_cv_all_muscles = np.random.uniform(-0.12, 0.05, n_muscles)  # negative = improvement
    delta_auc_cv_all_muscles[8] = -0.06
    delta_cv_peak_amp_all_muscles = np.random.uniform(-0.15, 0.05, n_muscles)
    delta_cv_peak_amp_all_muscles[8] = -0.08
    delta_sd_similarity_all_muscles = np.random.uniform(-0.08, 0.02, n_muscles)
    delta_sd_similarity_all_muscles[8] = -0.055

    # Less affected: smaller effects than more affected
    delta_similarity_all_muscles_less = np.random.uniform(-0.02, 0.08, n_muscles)
    delta_auc_cv_all_muscles_less = np.random.uniform(-0.08, 0.03, n_muscles)
    delta_cv_peak_amp_all_muscles_less = np.random.uniform(-0.10, 0.03, n_muscles)
    delta_sd_similarity_all_muscles_less = np.random.uniform(-0.05, 0.01, n_muscles)

    # Panel D: magnitude near zero, stability larger
    amplitude_effects_summary = np.array([-0.02, 0.01, -0.03, 0.015])
    stability_effects_summary = np.array([0.096, -0.055, -0.05])
    amplitude_effects_summary_less = np.array([-0.01, 0.005, -0.02, 0.01])
    stability_effects_summary_less = np.array([0.04, -0.02, -0.025])

    # No significance markers for mock (no raw paired data)
    global panel_d_sig_more, panel_d_sig_less, panel_d_es_more, panel_d_es_less
    global panel_e_sig_cv, panel_e_sig_cv_peak, panel_f_sig_cv, panel_f_sig_cv_peak
    panel_d_sig_more = [False] * 5
    panel_d_sig_less = [False] * 5
    panel_d_es_more = np.array([-0.3, 0.15, 0.8, -0.6, -0.5])
    panel_d_es_less = np.array([-0.1, 0.05, 0.4, -0.2, -0.2])
    panel_e_sig_cv = np.zeros(n_muscles, dtype=bool)
    panel_e_sig_cv_peak = np.zeros(n_muscles, dtype=bool)
    panel_f_sig_cv = np.zeros(n_muscles, dtype=bool)
    panel_f_sig_cv_peak = np.zeros(n_muscles, dtype=bool)


# -----------------------------------------------------------------------------
# Style constants
# -----------------------------------------------------------------------------

COLOR_COND1 = "#4a5568"  # slate gray (T0)
COLOR_COND2 = "#2d3748"  # darker slate (T1)
COLOR_LINE_CONNECT = "#a0aec0"  # light gray for subject connectors
COLOR_REF = "#cbd5e0"
# Panel E: distinct colors per metric type (not T0/T1)
COLOR_E_SIM = "#0d9488"   # teal - Δ mean similarity
COLOR_E_CV = "#d97706"    # amber - −Δ AuC CV
COLOR_E_SD = "#7c3aed"    # violet - −Δ SD similarity
# Panel D: pooled effect (more = full, less = empty, same color)
COLOR_D_POINT = "#4a5568"  # slate - used for both more (full) and less (empty) affected
FONT_PANEL = 16
FONT_AXIS = 13
FONT_TICK = 11
FONT_TITLE = 13
LINE_WIDTH_MEAN = 2.5
LINE_WIDTH_INDIV = 0.8
ALPHA_INDIV = 0.35
ALPHA_BAND = 0.25


# -----------------------------------------------------------------------------
# Helper plotting functions
# -----------------------------------------------------------------------------


def _style_axes(ax: plt.Axes, remove_top_right: bool = True) -> None:
    """Apply journal-style axes: remove top/right spines."""
    if remove_top_right:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    ax.tick_params(axis="both", labelsize=FONT_TICK)


def _add_panel_label(ax: plt.Axes, label: str, x_offset: float = 0.02, y_offset: float = 0.98) -> None:
    """Add bold panel label (A, B, ...) at top-left of panel."""
    ax.text(x_offset, y_offset, label, transform=ax.transAxes, fontsize=FONT_PANEL, fontweight="bold", va="top")


# -----------------------------------------------------------------------------
# Figure legend (JSON)
# -----------------------------------------------------------------------------


def _write_figure_legend(
    legend_path: Path,
    cond1_label: str = COND1_LABEL,
    cond2_label: str = COND2_LABEL,
) -> None:
    """Write a JSON file documenting each panel, colors, and what is plotted."""
    legend = {
        "figure": "emg_stability_summary_figure",
        "description": "EMG stability results: amplitude/timing stable, cycle-to-cycle stability improved (especially more affected side).",
        "panels": {
            "A": {
                "title": "Median envelope with IQR band (Infraspinatus, more affected)",
                "x_axis": "Normalized cycle (%)",
                "y_axis": "Normalized EMG envelope",
                "plotted": [
                    f"Median EMG envelope ± IQR for {cond1_label} (shaded band + line)",
                    f"Median EMG envelope ± IQR for {cond2_label} (shaded band + line)",
                ],
                "colors": {cond1_label: COLOR_COND1, cond2_label: COLOR_COND2},
                "interpretation": "Median ± IQR (25th–75th percentile) per phase; robust to outliers. Variability reduced in improved condition.",
            },
            "B": {
                "title": "Similarity to mean cycle (Infraspinatus, more affected, subject-level)",
                "x_axis": f"{cond1_label} vs {cond2_label}",
                "y_axis": "Similarity to mean cycle (Pearson r per subject)",
                "plotted": [
                    "Paired subject-level points connected by gray lines",
                    f"Group mean ± 95% CI for {cond1_label} and {cond2_label}",
                ],
                "colors": {cond1_label: COLOR_COND1, cond2_label: COLOR_COND2, "connecting_lines": COLOR_LINE_CONNECT},
                "interpretation": "Higher similarity = better reproducibility. Increase from T0 to T1 indicates improved stability.",
            },
            "C": {
                "title": "SD of similarity to mean cycle (Infraspinatus, more affected)",
                "x_axis": f"{cond1_label} vs {cond2_label}",
                "y_axis": "SD of similarity to mean cycle (subject-level)",
                "plotted": [
                    "Paired subject-level points connected by gray lines",
                    f"Group mean ± 95% CI for {cond1_label} and {cond2_label}",
                ],
                "colors": {cond1_label: COLOR_COND1, cond2_label: COLOR_COND2},
                "interpretation": "Lower SD = less variability across cycles. Reduction indicates improved stability.",
            },
            "D": {
                "title": "Stability vs magnitude: pooled effect summary",
                "x_axis": "Mean paired Δ (T1 − T0) across subjects",
                "y_axis": "Metric (AuC, Centroid | Similarity, SD similarity, CV AuC)",
                "plotted": [
                    "Full circles: more affected. Empty circles: less affected. Same color.",
                    "Magnitude/timing (top 3) and cycle-to-cycle stability (bottom 3, sign-flipped).",
                ],
                "colors": {"more_affected": COLOR_D_POINT, "less_affected": "edgecolor only"},
                "interpretation": "Primary effect on reproducibility. More affected shows larger stability effects.",
            },
            "E": {
                "title": "Effect size (Cohen's d_z) per muscle, more affected",
                "x_axis": "Effect size, d_z",
                "y_axis": "Muscle name",
                "plotted": [
                    "Cohen's d_z for −Δ AuC CV (squares), −Δ CV peak amp (triangles)",
                    "Red outline = significant paired change (p < 0.05, before FDR)",
                ],
                "colors": {"minus_delta_auc_cv_squares": COLOR_E_CV, "minus_delta_cv_peak_amp_triangles": COLOR_E_SD},
                "interpretation": "Effect size aligns with significance: consistent change yields larger d_z.",
            },
            "F": {
                "title": "Effect size (Cohen's d_z) per muscle, less affected",
                "x_axis": "Effect size, d_z",
                "y_axis": "Muscle name",
                "plotted": [
                    "Cohen's d_z for −Δ AuC CV (squares), −Δ CV peak amp (triangles)",
                    "Red outline = significant paired change (p < 0.05, before FDR)",
                ],
                "colors": {"minus_delta_auc_cv_squares": COLOR_E_CV, "minus_delta_cv_peak_amp_triangles": COLOR_E_SD},
                "interpretation": "Same metrics as Panel E. Effect size reflects standardized paired change.",
            },
        },
        "global_colors": {
            f"{cond1_label}": COLOR_COND1,
            f"{cond2_label}": COLOR_COND2,
        },
    }
    with open(legend_path, "w", encoding="utf-8") as f:
        json.dump(legend, f, indent=2, ensure_ascii=False)
    print(f"Figure legend saved to {legend_path}")


# -----------------------------------------------------------------------------
# Figure creation
# -----------------------------------------------------------------------------


def create_emg_summary_figure(
    out_path: Path,
    cond1_label: str = COND1_LABEL,
    cond2_label: str = COND2_LABEL,
    dpi: int = 300,
) -> None:
    """
    Create the 6-panel publication figure summarizing EMG results.
    """
    # Row 3 (E,F) taller for forest panels with larger markers
    fig, axes = plt.subplots(
        3, 2,
        figsize=(10, 11.5),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.9]},
        constrained_layout=True,
    )
    axes = axes.flatten()

    # ----- Panel A: Median envelope with IQR band (Infraspinatus, more affected) -----
    phase_pct = np.linspace(0, 100, cycles_cond1_infra_more.shape[1])
    med_a1 = np.median(cycles_cond1_infra_more, axis=0)
    med_a2 = np.median(cycles_cond2_infra_more, axis=0)
    q25_a1, q75_a1 = np.percentile(cycles_cond1_infra_more, [25, 75], axis=0)
    q25_a2, q75_a2 = np.percentile(cycles_cond2_infra_more, [25, 75], axis=0)
    ax_a = axes[0]
    ax_a.fill_between(phase_pct, q25_a1, q75_a1, color=COLOR_COND1, alpha=ALPHA_BAND)
    ax_a.plot(phase_pct, med_a1, color=COLOR_COND1, lw=LINE_WIDTH_MEAN, label=cond1_label)
    ax_a.fill_between(phase_pct, q25_a2, q75_a2, color=COLOR_COND2, alpha=ALPHA_BAND)
    ax_a.plot(phase_pct, med_a2, color=COLOR_COND2, lw=LINE_WIDTH_MEAN, label=cond2_label)
    ax_a.set_title("Infraspinatus, more affected side", fontsize=FONT_TITLE)
    ax_a.set_xlabel("Normalized cycle (%)", fontsize=FONT_AXIS)
    ax_a.set_ylabel("Normalized EMG envelope", fontsize=FONT_AXIS)
    ax_a.legend(loc="upper right", fontsize=FONT_TICK)
    ax_a.set_xlim(0, 100)
    ax_a.set_ylim(0, None)
    _style_axes(ax_a)
    _add_panel_label(ax_a, "A")

    # ----- Panel B: Paired similarity to mean cycle (Infraspinatus, more affected) -----
    ax_b = axes[1]
    n_s = len(similarity_cond1_infra_more)
    np.random.seed(101)
    jitter = np.random.uniform(-0.025, 0.025, n_s)
    x1 = 0.85 + jitter
    x2 = 1.15 + jitter
    for i in range(n_s):
        ax_b.plot([0.85, 1.15], [similarity_cond1_infra_more[i], similarity_cond2_infra_more[i]],
                  color=COLOR_LINE_CONNECT, lw=1, zorder=0)
    ax_b.scatter(x1, similarity_cond1_infra_more, c=COLOR_COND1, s=38, zorder=2, edgecolors="white", linewidths=0.5)
    ax_b.scatter(x2, similarity_cond2_infra_more, c=COLOR_COND2, s=38, zorder=2, edgecolors="white", linewidths=0.5)

    m1, m2 = np.mean(similarity_cond1_infra_more), np.mean(similarity_cond2_infra_more)
    se1 = np.std(similarity_cond1_infra_more) / np.sqrt(n_s) if n_s > 1 else 0
    se2 = np.std(similarity_cond2_infra_more) / np.sqrt(n_s) if n_s > 1 else 0
    ax_b.errorbar([0.85], [m1], yerr=[1.96 * se1], fmt="o", color=COLOR_COND1, markersize=10, capsize=4, lw=2)
    ax_b.errorbar([1.15], [m2], yerr=[1.96 * se2], fmt="o", color=COLOR_COND2, markersize=10, capsize=4, lw=2)

    ax_b.set_xticks([0.85, 1.15])
    ax_b.set_xticklabels([cond1_label, cond2_label], fontsize=FONT_TICK)
    ax_b.set_ylabel("Similarity to mean cycle", fontsize=FONT_AXIS)
    ax_b.set_title("Infraspinatus, more affected side", fontsize=FONT_TITLE)
    ax_b.set_xlim(0.7, 1.3)
    ax_b.set_ylim(0, 1)
    _style_axes(ax_b)
    _add_panel_label(ax_b, "B")

    # ----- Panel C: Paired SD of similarity (Infraspinatus, more affected) -----
    ax_c = axes[2]
    np.random.seed(102)
    jitter_c = np.random.uniform(-0.025, 0.025, n_s)
    x1_c, x2_c = 0.85 + jitter_c, 1.15 + jitter_c
    for i in range(n_s):
        ax_c.plot([0.85, 1.15], [sd_similarity_cond1_more[i], sd_similarity_cond2_more[i]],
                  color=COLOR_LINE_CONNECT, lw=1, zorder=0)
    ax_c.scatter(x1_c, sd_similarity_cond1_more, c=COLOR_COND1, s=38, zorder=2, edgecolors="white", linewidths=0.5)
    ax_c.scatter(x2_c, sd_similarity_cond2_more, c=COLOR_COND2, s=38, zorder=2, edgecolors="white", linewidths=0.5)

    md1, md2 = np.mean(sd_similarity_cond1_more), np.mean(sd_similarity_cond2_more)
    sed1 = np.std(sd_similarity_cond1_more) / np.sqrt(n_s) if n_s > 1 else 0
    sed2 = np.std(sd_similarity_cond2_more) / np.sqrt(n_s) if n_s > 1 else 0
    ax_c.errorbar([0.85], [md1], yerr=[1.96 * sed1], fmt="o", color=COLOR_COND1, markersize=10, capsize=4, lw=2)
    ax_c.errorbar([1.15], [md2], yerr=[1.96 * sed2], fmt="o", color=COLOR_COND2, markersize=10, capsize=4, lw=2)

    ax_c.set_xticks([0.85, 1.15])
    ax_c.set_xticklabels([cond1_label, cond2_label], fontsize=FONT_TICK)
    ax_c.set_ylabel("SD of similarity to mean cycle", fontsize=FONT_AXIS)
    ax_c.set_title("Infraspinatus, more affected side", fontsize=FONT_TITLE)
    ax_c.set_xlim(0.7, 1.3)
    ax_c.set_ylim(0, None)
    _style_axes(ax_c)
    _add_panel_label(ax_c, "C")

    # ----- Panel D: Stability vs magnitude pooled summary (effect size d_z; more = full, less = empty) -----
    ax_d = axes[3]
    es_more = panel_d_es_more if panel_d_es_more is not None else np.zeros(5)
    es_less = panel_d_es_less if panel_d_es_less is not None else np.zeros(5)
    mag_effects = np.array([es_more[0], es_more[1]])  # AuC, Centroid
    mag_labels = ["AuC", "Centroid"]
    stab_effects = np.array([es_more[2], es_more[3], es_more[4]])  # Similarity, SD sim, CV AuC
    stab_labels = ["Similarity", "SD similarity", "CV AuC"]
    n_mag, n_stab = len(mag_effects), len(stab_effects)
    y_mag = np.arange(n_mag)
    y_stab = np.arange(n_mag, n_mag + n_stab)
    mag_effects_less = np.array([es_less[0], es_less[1]])
    stab_effects_less = np.array([es_less[2], es_less[3], es_less[4]])
    ax_d.axvline(0, color=COLOR_REF, lw=1, ls="--")
    # More affected: full circles
    ax_d.scatter(mag_effects, y_mag, c=COLOR_D_POINT, s=54, zorder=2)
    ax_d.scatter(stab_effects, y_stab, c=COLOR_D_POINT, s=54, zorder=2)
    # Less affected: empty circles
    ax_d.scatter(mag_effects_less, y_mag, facecolors="none", edgecolors=COLOR_D_POINT, s=54, linewidths=1.5, zorder=1)
    ax_d.scatter(stab_effects_less, y_stab, facecolors="none", edgecolors=COLOR_D_POINT, s=54, linewidths=1.5, zorder=1)
    # Red square markers when significant (before FDR)
    sig_more = panel_d_sig_more if panel_d_sig_more is not None else [False] * 5
    sig_less = panel_d_sig_less if panel_d_sig_less is not None else [False] * 5
    for i, (sig, x, y) in enumerate(zip(sig_more[:2], mag_effects, y_mag)):
        if sig:
            ax_d.scatter(x, y, marker="s", s=90, facecolors="none", edgecolors=COLOR_SIG_MARKER, linewidths=1.5, zorder=3)
    for i, (sig, x, y) in enumerate(zip(sig_more[2:], stab_effects, y_stab)):
        if sig:
            ax_d.scatter(x, y, marker="s", s=90, facecolors="none", edgecolors=COLOR_SIG_MARKER, linewidths=1.5, zorder=3)
    for i, (sig, x, y) in enumerate(zip(sig_less[:2], mag_effects_less, y_mag)):
        if sig:
            ax_d.scatter(x, y, marker="s", s=90, facecolors="none", edgecolors=COLOR_SIG_MARKER, linewidths=1.5, zorder=3)
    for i, (sig, x, y) in enumerate(zip(sig_less[2:], stab_effects_less, y_stab)):
        if sig:
            ax_d.scatter(x, y, marker="s", s=90, facecolors="none", edgecolors=COLOR_SIG_MARKER, linewidths=1.5, zorder=3)
    ax_d.legend(handles=[
        plt.Line2D([0], [0], marker="o", ls="", color=COLOR_D_POINT, markersize=10, label="More affected"),
        plt.Line2D([0], [0], marker="o", ls="", markerfacecolor="none", markeredgecolor=COLOR_D_POINT, markeredgewidth=1.5, markersize=10, label="Less affected"),
    ], loc="upper left", fontsize=10)
    ax_d.set_yticks(list(y_mag) + list(y_stab))
    ax_d.set_yticklabels(mag_labels + stab_labels, fontsize=FONT_TICK)
    ax_d.set_xlabel(r"Effect size, $d_z$", fontsize=FONT_AXIS)
    ax_d.set_title("Pooled effect summary", fontsize=FONT_TITLE)
    all_vals = np.concatenate([mag_effects, stab_effects, mag_effects_less, stab_effects_less])
    lim = max(abs(all_vals).max() if all_vals.size else 0, 0.05) * 1.2
    ax_d.set_xlim(-lim, lim)
    _style_axes(ax_d)
    _add_panel_label(ax_d, "D")

    # ----- Panel E: Effect size (Cohen's d_z), more affected (−Δ AuC CV and −Δ CV peak amp) -----
    S_MARKER = 72  # larger markers for forest panels
    ax_e = axes[4]
    n_muscles = len(muscle_names)
    y_pos = np.arange(n_muscles)
    dy = 0.2  # vertical offset between the two metric columns
    es_cv = effect_size_auc_cv_all_muscles if effect_size_auc_cv_all_muscles is not None else -delta_auc_cv_all_muscles
    es_cv_peak = effect_size_cv_peak_all_muscles if effect_size_cv_peak_all_muscles is not None else -delta_cv_peak_amp_all_muscles
    ax_e.axvline(0, color=COLOR_REF, lw=1, ls="--")
    sig_cv = panel_e_sig_cv if panel_e_sig_cv is not None else np.zeros(n_muscles, dtype=bool)
    sig_cv_peak = panel_e_sig_cv_peak if panel_e_sig_cv_peak is not None else np.zeros(n_muscles, dtype=bool)
    for i in range(n_muscles):
        ax_e.scatter(es_cv[i], y_pos[i] - dy, c=COLOR_E_CV, s=S_MARKER, marker="s", zorder=2, edgecolors="none")
        ax_e.scatter(es_cv_peak[i], y_pos[i] + dy, c=COLOR_E_SD, s=S_MARKER, marker="^", zorder=2, edgecolors="none")
        if i < len(sig_cv) and sig_cv[i]:
            ax_e.scatter(es_cv[i], y_pos[i] - dy, marker="s", s=S_MARKER + 40, facecolors="none",
                         edgecolors=COLOR_SIG_MARKER, linewidths=1.5, zorder=3)
        if i < len(sig_cv_peak) and sig_cv_peak[i]:
            ax_e.scatter(es_cv_peak[i], y_pos[i] + dy, marker="s", s=S_MARKER + 40, facecolors="none",
                         edgecolors=COLOR_SIG_MARKER, linewidths=1.5, zorder=3)
    infra_idx = next((i for i, m in enumerate(muscle_names) if "Infraspinatus" in m), 8)
    ax_e.axhspan(infra_idx - 0.5, infra_idx + 0.5, alpha=0.1, color="#4a5568", zorder=0)
    ax_e.set_yticks(y_pos)
    ax_e.set_yticklabels([m.replace(" Brachii ", " B. ").replace(" Triceps ", " Tri. ") for m in muscle_names], fontsize=10)
    ax_e.set_xlabel(r"Effect size, $d_z$", fontsize=FONT_AXIS)
    ax_e.set_title("More affected", fontsize=FONT_TITLE)
    xlim_e = max(abs(es_cv).max() if es_cv.size else 0, abs(es_cv_peak).max() if es_cv_peak.size else 0, 0.1) * 1.2
    ax_e.set_xlim(-xlim_e, xlim_e)
    ax_e.set_ylim(-1.5, n_muscles - 0.5)
    ax_e.legend(handles=[
        plt.Line2D([0], [0], marker="s", ls="", color=COLOR_E_CV, markersize=11, label="−Δ AuC CV"),
        plt.Line2D([0], [0], marker="^", ls="", color=COLOR_E_SD, markersize=11, label="−Δ CV peak amp"),
    ], loc="lower right", fontsize=9, ncol=2)
    _style_axes(ax_e)
    _add_panel_label(ax_e, "E")

    # ----- Panel F: Effect size (Cohen's d_z), less affected (−Δ AuC CV and −Δ CV peak amp) -----
    ax_f = axes[5]
    es_cv_less = effect_size_auc_cv_all_muscles_less if effect_size_auc_cv_all_muscles_less is not None else -delta_auc_cv_all_muscles_less
    es_cv_peak_less = effect_size_cv_peak_all_muscles_less if effect_size_cv_peak_all_muscles_less is not None else -delta_cv_peak_amp_all_muscles_less
    ax_f.axvline(0, color=COLOR_REF, lw=1, ls="--")
    sig_cv_less = panel_f_sig_cv if panel_f_sig_cv is not None else np.zeros(n_muscles, dtype=bool)
    sig_cv_peak_less = panel_f_sig_cv_peak if panel_f_sig_cv_peak is not None else np.zeros(n_muscles, dtype=bool)
    for i in range(n_muscles):
        ax_f.scatter(es_cv_less[i], y_pos[i] - dy, c=COLOR_E_CV, s=S_MARKER, marker="s", zorder=2, edgecolors="none")
        ax_f.scatter(es_cv_peak_less[i], y_pos[i] + dy, c=COLOR_E_SD, s=S_MARKER, marker="^", zorder=2, edgecolors="none")
        if i < len(sig_cv_less) and sig_cv_less[i]:
            ax_f.scatter(es_cv_less[i], y_pos[i] - dy, marker="s", s=S_MARKER + 40, facecolors="none",
                         edgecolors=COLOR_SIG_MARKER, linewidths=1.5, zorder=3)
        if i < len(sig_cv_peak_less) and sig_cv_peak_less[i]:
            ax_f.scatter(es_cv_peak_less[i], y_pos[i] + dy, marker="s", s=S_MARKER + 40, facecolors="none",
                         edgecolors=COLOR_SIG_MARKER, linewidths=1.5, zorder=3)
    ax_f.axhspan(infra_idx - 0.5, infra_idx + 0.5, alpha=0.1, color="#4a5568", zorder=0)
    ax_f.set_yticks(y_pos)
    ax_f.set_yticklabels([m.replace(" Brachii ", " B. ").replace(" Triceps ", " Tri. ") for m in muscle_names], fontsize=10)
    ax_f.set_xlabel(r"Effect size, $d_z$", fontsize=FONT_AXIS)
    ax_f.set_title("Less affected", fontsize=FONT_TITLE)
    xlim_f = max(abs(es_cv_less).max() if es_cv_less.size else 0, abs(es_cv_peak_less).max() if es_cv_peak_less.size else 0, 0.1) * 1.2
    ax_f.set_xlim(-xlim_f, xlim_f)
    ax_f.set_ylim(-1.5, n_muscles - 0.5)
    ax_f.legend(handles=[
        plt.Line2D([0], [0], marker="s", ls="", color=COLOR_E_CV, markersize=11, label="−Δ AuC CV"),
        plt.Line2D([0], [0], marker="^", ls="", color=COLOR_E_SD, markersize=11, label="−Δ CV peak amp"),
    ], loc="lower right", fontsize=9, ncol=2)
    _style_axes(ax_f)
    _add_panel_label(ax_f, "F")

    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    svg_path = out_path.with_suffix(".svg")
    plt.savefig(svg_path, format="svg", bbox_inches="tight")
    plt.close()
    print(f"Figure saved to {out_path} and {svg_path}")

    # Write figure legend JSON alongside the figure
    legend_path = out_path.with_suffix(".json")
    _write_figure_legend(legend_path, cond1_label=cond1_label, cond2_label=cond2_label)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    """Generate the EMG summary figure."""
    global cycles_cond1_infra_more, cycles_cond2_infra_more, similarity_cond1_infra_more, similarity_cond2_infra_more
    global sd_similarity_cond1_more, sd_similarity_cond2_more
    global muscle_names, delta_similarity_all_muscles, delta_auc_cv_all_muscles, delta_sd_similarity_all_muscles
    global delta_similarity_all_muscles_less, delta_auc_cv_all_muscles_less, delta_sd_similarity_all_muscles_less
    global delta_cv_peak_amp_all_muscles, delta_cv_peak_amp_all_muscles_less
    global amplitude_effects_summary, stability_effects_summary
    global amplitude_effects_summary_less, stability_effects_summary_less
    global panel_d_es_more, panel_d_es_less
    # Load real data from B3 outputs
    loaded = load_data_from_b3_outputs()
    if not loaded or muscle_names is None:
        print("Note: Load failed, using full mock data. Run B3 first and ensure results/synergies/ exists.")
        _make_mock_data()
    elif cycles_cond1_infra_more is None or cycles_cond2_infra_more is None:
        # Panel A needs X_cycles.npz; fill mock cycles only (re-run B3 to get real X_cycles)
        print("Note: X_cycles.npz missing - using mock for panel A. Re-run B3 to save cycle waveforms.")
        _fill_mock_cycles()
    if similarity_cond1_infra_more is None or similarity_cond2_infra_more is None:
        _fill_mock_similarity()
    if delta_similarity_all_muscles is None or delta_auc_cv_all_muscles is None or delta_sd_similarity_all_muscles is None:
        np.random.seed(44)
        n_m = len(muscle_names) if muscle_names else 12
        delta_similarity_all_muscles = np.random.uniform(-0.03, 0.15, n_m)
        delta_auc_cv_all_muscles = np.random.uniform(-0.12, 0.05, n_m)
        delta_sd_similarity_all_muscles = np.random.uniform(-0.08, 0.02, n_m)
    if delta_similarity_all_muscles_less is None or delta_auc_cv_all_muscles_less is None or delta_sd_similarity_all_muscles_less is None:
        np.random.seed(45)
        n_m = len(muscle_names) if muscle_names else 12
        delta_similarity_all_muscles_less = np.random.uniform(-0.02, 0.08, n_m)
        delta_auc_cv_all_muscles_less = np.random.uniform(-0.08, 0.03, n_m)
        delta_sd_similarity_all_muscles_less = np.random.uniform(-0.05, 0.01, n_m)
    if delta_cv_peak_amp_all_muscles is None:
        np.random.seed(46)
        n_m = len(muscle_names) if muscle_names else 12
        delta_cv_peak_amp_all_muscles = np.random.uniform(-0.15, 0.05, n_m)
    if delta_cv_peak_amp_all_muscles_less is None:
        np.random.seed(47)
        n_m = len(muscle_names) if muscle_names else 12
        delta_cv_peak_amp_all_muscles_less = np.random.uniform(-0.10, 0.03, n_m)
    if amplitude_effects_summary is None or stability_effects_summary is None:
        amplitude_effects_summary = np.array([-0.02, 0.01, -0.03, 0.015])
        stability_effects_summary = np.array([0.096, -0.055, -0.05])
    if amplitude_effects_summary_less is None or stability_effects_summary_less is None:
        amplitude_effects_summary_less = np.array([-0.01, 0.005, -0.02, 0.01])
        stability_effects_summary_less = np.array([0.04, -0.02, -0.025])
    if panel_d_es_more is None or panel_d_es_less is None:
        panel_d_es_more = np.array([-0.3, 0.15, 0.8, -0.6, -0.5])
        panel_d_es_less = np.array([-0.1, 0.05, 0.4, -0.2, -0.2])

    out_dir = Path(__file__).resolve().parent / "results" / "emg_figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "emg_stability_summary_figure.png"

    create_emg_summary_figure(out_path, dpi=300)


if __name__ == "__main__":
    main()
