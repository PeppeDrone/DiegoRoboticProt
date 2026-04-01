#!/usr/bin/env python3
"""
Verify that emg_meta_C*_mean in Excel matches the meta_synergy activation time course.

Compares:
1. Cohort mean from meta_task_summary_metrics (source of Excel) per (condition, session, cluster)
2. Mean of activation curves from H_meta_windows (source of time course plot)

These should match - both derive from the same H_meta data.
"""

from pathlib import Path
import numpy as np
import pandas as pd


def read_npz_array(path: Path, key: str):
    if not path.exists():
        return None
    try:
        npz = np.load(path, allow_pickle=True)
        return npz[key] if key in npz.files else None
    except Exception:
        return None


def discover_task_dirs(results_dir: Path):
    discovered = []
    for patient_dir in sorted(results_dir.iterdir()):
        if not patient_dir.is_dir() or patient_dir.name.startswith("plots_") or patient_dir.name == "meta_synergy_clustering":
            continue
        for task_dir in sorted(patient_dir.iterdir()):
            if not task_dir.is_dir():
                continue
            task_upper = task_dir.name.upper()
            session = "T0" if "T0" in task_upper else "T1" if "T1" in task_upper else ""
            condition = "SN" if "SN" in task_upper else "DS" if "DS" in task_upper else ""
            if session and condition:
                discovered.append((patient_dir.name, task_dir.name, session, condition, task_dir))
    return discovered


def main():
    results_dir = Path("results/synergies")
    meta_dir = results_dir / "meta_synergy_clustering"

    # 1. Load meta_task_summary_metrics (source of Excel emg_meta_C*_mean)
    meta_csv = meta_dir / "meta_task_summary_metrics.csv"
    if not meta_csv.exists():
        print(f"Missing {meta_csv}")
        return
    meta_df = pd.read_csv(meta_csv)
    n_clusters = int(meta_df["n_meta_clusters"].iloc[0]) if "n_meta_clusters" in meta_df.columns else 4

    print("=" * 70)
    print("VERIFICATION: Excel emg_meta_C*_mean vs meta_synergy time course")
    print("=" * 70)

    # 2. Cohort means from meta_task_summary_metrics (what goes into Excel)
    print("\n1. COHORT MEANS from meta_task_summary_metrics (source of Excel):")
    print("   (mean of emg_meta_C{k}_mean across all tasks per condition/session)")
    for cond in ["SN", "DS"]:
        for sess in ["T0", "T1"]:
            subset = meta_df[(meta_df["condition"] == cond) & (meta_df["session"] == sess)]
            if subset.empty:
                continue
            row = []
            for k in range(n_clusters):
                col = f"mean_meta_synergy_{k}_mean"
                if col in subset.columns:
                    vals = subset[col].dropna()
                    m = float(vals.mean()) if len(vals) > 0 else np.nan
                    row.append(f"C{k}={m:.4f}")
            print(f"   {cond} {sess}: " + ", ".join(row))

    # 3. Mean of curves from H_meta_windows (what the time course plot shows)
    print("\n2. MEAN OF CURVES from H_meta_windows (source of time course plot):")
    print("   (mean over phase of cohort-averaged activation curve)")
    curves_by_key = {}
    for patient_id, task_name, session, condition, task_dir in discover_task_dirs(results_dir):
        H_meta = read_npz_array(task_dir / "H_meta_windows.npz", "H_meta")
        if H_meta is None or H_meta.ndim != 3:
            continue
        n_win, n_clust, n_samp = H_meta.shape
        for win in range(n_win):
            for meta_c in range(n_clust):
                key = (condition, session, meta_c)
                h = np.asarray(H_meta[win, meta_c, :], dtype=np.float64)
                scalar_mean = float(np.mean(h)) if h.size > 0 else np.nan
                curves_by_key.setdefault(key, []).append(scalar_mean)

    for cond in ["SN", "DS"]:
        for sess in ["T0", "T1"]:
            row = []
            for k in range(n_clusters):
                key = (cond, sess, k)
                vals = curves_by_key.get(key, [])
                m = float(np.mean(vals)) if vals else np.nan
                row.append(f"C{k}={m:.4f}")
            print(f"   {cond} {sess}: " + ", ".join(row))

    # 4. Direct comparison
    print("\n3. DIFFERENCE (Excel source - time course source):")
    print("   Should be ~0 if ordering/labeling is consistent.")
    all_ok = True
    for cond in ["SN", "DS"]:
        for sess in ["T0", "T1"]:
            subset = meta_df[(meta_df["condition"] == cond) & (meta_df["session"] == sess)]
            if subset.empty:
                continue
            for k in range(n_clusters):
                col = f"mean_meta_synergy_{k}_mean"
                if col not in subset.columns:
                    continue
                excel_mean = subset[col].dropna().mean()
                key = (cond, sess, k)
                curve_means = curves_by_key.get(key, [])
                plot_mean = np.mean(curve_means) if curve_means else np.nan
                diff = excel_mean - plot_mean
                status = "OK" if abs(diff) < 0.01 else "MISMATCH"
                if abs(diff) >= 0.01:
                    all_ok = False
                print(f"   {cond} {sess} C{k}: diff={diff:+.4f} [{status}]")

    if all_ok:
        print("\n*** All values match. Ordering and labeling are consistent. ***")
    else:
        print("\n*** MISMATCH DETECTED. Check for ordering/labeling bugs. ***")

    # 5. Combined T0 vs T1 (what the combined plot shows - pools SN+DS)
    print("\n4. COMBINED T0 vs T1 (pools SN+DS, like meta_synergy_timecourse_T0_vs_T1.png):")
    for sess in ["T0", "T1"]:
        row = []
        for k in range(n_clusters):
            vals = []
            for cond in ["SN", "DS"]:
                key = (cond, sess, k)
                vals.extend(curves_by_key.get(key, []))
            m = float(np.mean(vals)) if vals else np.nan
            row.append(f"C{k}={m:.4f}")
        print(f"   {sess}: " + ", ".join(row))


if __name__ == "__main__":
    main()
