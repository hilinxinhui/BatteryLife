"""Voltage relaxation feature extraction for BatteryLife-format pkl files."""

from __future__ import annotations

import argparse
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SKIP_DATASET_DIRS = {"READMEs", "Life labels", "seen_unseen_labels"}
DEFAULT_WINDOWS_S = (300.0, 600.0, 1800.0)


@dataclass(frozen=True)
class Segment:
    start: int
    end: int

    @property
    def length(self) -> int:
        return self.end - self.start


def finite_array(values) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=float)
    return arr[np.isfinite(arr)] if arr.ndim == 1 else np.asarray([], dtype=float)


def as_array(values) -> np.ndarray:
    arr = np.asarray(values if values is not None else [], dtype=float)
    if arr.ndim != 1:
        return np.asarray([], dtype=float)
    return arr


def dataset_from_path(path: Path) -> str:
    return path.parent.name


def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r") as f:
        return json.load(f)


def load_dataset_intervals(repo_root: Path) -> dict:
    path = repo_root / "tutorials" / "feature_extraction" / "configs" / "dataset_intervals.json"
    return load_json(path)


def load_life_labels(repo_root: Path, dataset_name: str) -> dict:
    labels_dir = repo_root / "dataset" / "Life labels"
    label_name = "ISU-ILCC" if dataset_name == "ISU_ILCC" else dataset_name
    return load_json(labels_dir / f"{label_name}_labels.json")


def life_label_key(dataset_name: str, file_name: str) -> str:
    if dataset_name == "Tongji":
        return file_name.replace("--", "-#")
    return file_name


def infer_cutoff_voltage(cell: dict, dataset_name: str, configs: dict) -> float:
    value = cell.get("max_voltage_limit_in_V")
    if value is not None and np.isfinite(float(value)) and float(value) > 0:
        return float(value)
    cfg = configs.get(dataset_name, {})
    if cfg.get("cutoff_voltage") is not None:
        return float(cfg["cutoff_voltage"])
    vmax = []
    for cycle in cell.get("cycle_data", [])[:20]:
        voltage = finite_array(cycle.get("voltage_in_V"))
        if len(voltage):
            vmax.append(float(np.nanmax(voltage)))
    return float(np.nanmedian(vmax)) if vmax else math.nan


def compute_soh(cycle: dict, nominal_capacity: float, soc_interval: list | None) -> float:
    current = as_array(cycle.get("current_in_A"))
    discharge_capacity = as_array(cycle.get("discharge_capacity_in_Ah"))
    if len(current) == 0 or len(discharge_capacity) == 0 or len(current) != len(discharge_capacity):
        return math.nan
    mask = current < 0
    if not np.any(mask) or nominal_capacity <= 0:
        return math.nan
    width = 1.0
    if soc_interval and len(soc_interval) == 2:
        width = float(soc_interval[1]) - float(soc_interval[0])
    if width <= 0:
        return math.nan
    return float(np.nanmax(discharge_capacity[mask]) / nominal_capacity / width)


def compute_rul(cycle_number: int, eol: int | None, total_cycles: int) -> float:
    end = eol if eol is not None else total_cycles
    return float(end - cycle_number)


def max_discharge_capacity(cycle: dict) -> float:
    current = as_array(cycle.get("current_in_A"))
    discharge_capacity = as_array(cycle.get("discharge_capacity_in_Ah"))
    if len(current) == 0 or len(discharge_capacity) == 0 or len(current) != len(discharge_capacity):
        return math.nan
    mask = current < 0
    if not np.any(mask):
        return math.nan
    return float(np.nanmax(discharge_capacity[mask]))


def contiguous_true_segments(mask: np.ndarray, min_points: int) -> list[Segment]:
    segments: list[Segment] = []
    start = None
    for idx, flag in enumerate(mask):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            if idx - start >= min_points:
                segments.append(Segment(start, idx))
            start = None
    if start is not None and len(mask) - start >= min_points:
        segments.append(Segment(start, len(mask)))
    return segments


def nearest_non_rest_current(current: np.ndarray, rest_mask: np.ndarray, idx: int, direction: int) -> float:
    j = idx + direction
    while 0 <= j < len(current):
        if not rest_mask[j] and np.isfinite(current[j]):
            return float(current[j])
        j += direction
    return math.nan


def segment_duration(time: np.ndarray, segment: Segment) -> float:
    if segment.length < 2:
        return 0.0
    return float(time[segment.end - 1] - time[segment.start])


def find_rest_segments(
    voltage: np.ndarray,
    current: np.ndarray,
    time: np.ndarray,
    cutoff_voltage: float,
    nominal_capacity: float,
    current_epsilon_c: float,
    current_epsilon_abs: float,
    voltage_tolerance: float,
    min_points: int,
    min_duration_s: float,
) -> list[dict]:
    n = min(len(voltage), len(current), len(time))
    if n < min_points:
        return []
    voltage = voltage[:n]
    current = current[:n]
    time = time[:n]
    finite = np.isfinite(voltage) & np.isfinite(current) & np.isfinite(time)
    threshold = max(current_epsilon_abs, current_epsilon_c * max(float(nominal_capacity), 1e-9))
    rest_mask = finite & (np.abs(current) <= threshold)
    raw_segments = contiguous_true_segments(rest_mask, min_points=min_points)

    rows = []
    for seg in raw_segments:
        duration = segment_duration(time, seg)
        if duration < min_duration_s:
            continue
        v_seg = voltage[seg.start : seg.end]
        t_seg = time[seg.start : seg.end]
        pre_i = nearest_non_rest_current(current, rest_mask, seg.start, -1)
        post_i = nearest_non_rest_current(current, rest_mask, seg.end - 1, 1)
        start_v = float(v_seg[0])
        end_v = float(v_seg[-1])
        near_full = bool(np.isfinite(cutoff_voltage) and np.nanmax(v_seg) >= cutoff_voltage - voltage_tolerance)
        after_charge = bool(np.isfinite(pre_i) and pre_i > threshold)
        before_discharge = bool(np.isfinite(post_i) and post_i < -threshold)
        rows.append(
            {
                "start_idx": seg.start,
                "end_idx": seg.end,
                "n_points": seg.length,
                "duration_s": duration,
                "start_time_s": float(t_seg[0]),
                "end_time_s": float(t_seg[-1]),
                "start_voltage": start_v,
                "end_voltage": end_v,
                "voltage_delta": end_v - start_v,
                "near_full_charge_voltage": near_full,
                "after_charge": after_charge,
                "before_discharge": before_discharge,
                "paper_compatible": bool(near_full and after_charge and before_discharge),
                "pre_current_A": pre_i,
                "post_current_A": post_i,
                "current_threshold_A": threshold,
            }
        )
    return rows


def skewness(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 3:
        return 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if std == 0.0 or np.allclose(values, values[0]):
        return 0.0
    centered = (values - mean) / std
    n = len(values)
    return float((n / ((n - 1) * (n - 2))) * np.sum(centered**3))


def excess_kurtosis(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        return 0.0
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    if std == 0.0 or np.allclose(values, values[0]):
        return 0.0
    z = (values - mean) / std
    n = len(values)
    term1 = (n * (n + 1) / ((n - 1) * (n - 2) * (n - 3))) * np.sum(z**4)
    term2 = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    return float(term1 - term2)


def relaxation_stats(values: np.ndarray) -> dict:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return {
            "relax_var": math.nan,
            "relax_ske": math.nan,
            "relax_max": math.nan,
            "relax_min": math.nan,
            "relax_mean": math.nan,
            "relax_kur": math.nan,
        }
    return {
        "relax_var": float(np.var(values, ddof=1)) if len(values) > 1 else 0.0,
        "relax_ske": skewness(values),
        "relax_max": float(np.max(values)),
        "relax_min": float(np.min(values)),
        "relax_mean": float(np.mean(values)),
        "relax_kur": excess_kurtosis(values),
    }


def extract_window_features(
    voltage: np.ndarray,
    time: np.ndarray,
    segment: dict,
    windows_s: Iterable[float],
    min_points: int,
) -> list[dict]:
    start = int(segment["start_idx"])
    end = int(segment["end_idx"])
    v_seg = voltage[start:end]
    t_seg = time[start:end]
    rows = []
    for window_s in list(windows_s) + [None]:
        if window_s is None:
            mask = np.ones(len(t_seg), dtype=bool)
            window_name = "full"
        else:
            mask = t_seg - t_seg[0] <= float(window_s)
            window_name = f"{int(window_s)}s"
        if np.count_nonzero(mask) < min_points:
            continue
        if window_s is not None and float(t_seg[mask][-1] - t_seg[0]) + 1e-9 < float(window_s):
            continue
        stats = relaxation_stats(v_seg[mask])
        rows.append(
            {
                "window": window_name,
                "window_s": float(window_s) if window_s is not None else float(segment["duration_s"]),
                "window_points": int(np.count_nonzero(mask)),
                **stats,
            }
        )
    return rows


def iter_pkl_files(dataset_root: Path, datasets: list[str] | None) -> list[Path]:
    if datasets:
        dirs = [dataset_root / name for name in datasets]
    else:
        dirs = [
            p
            for p in sorted(dataset_root.iterdir())
            if p.is_dir() and p.name not in SKIP_DATASET_DIRS
        ]
    files: list[Path] = []
    for directory in dirs:
        files.extend(sorted(directory.glob("*.pkl")))
        files.extend(sorted(directory.glob("*.pickle")))
    return files


def process_file(
    path: Path,
    repo_root: Path,
    configs: dict,
    windows_s: Iterable[float],
    current_epsilon_c: float,
    current_epsilon_abs: float,
    voltage_tolerance: float,
    min_points: int,
    min_duration_s: float,
) -> tuple[list[dict], list[dict]]:
    dataset_name = dataset_from_path(path)
    with path.open("rb") as f:
        cell = pickle.load(f)
    labels = load_life_labels(repo_root, dataset_name)
    eol = labels.get(life_label_key(dataset_name, path.name))
    cell_id = cell.get("cell_id", path.stem)
    nominal_capacity = float(cell.get("nominal_capacity_in_Ah") or 1.0)
    soc_interval = cell.get("SOC_interval", [0.0, 1.0])
    cutoff_voltage = infer_cutoff_voltage(cell, dataset_name, configs)
    cycles = cell.get("cycle_data", [])
    total_cycles = len(cycles)

    audit_rows = []
    feature_rows = []
    for cycle_position, cycle in enumerate(cycles, start=1):
        cycle_number = int(cycle.get("cycle_number", cycle_position))
        voltage = as_array(cycle.get("voltage_in_V"))
        current = as_array(cycle.get("current_in_A"))
        time = as_array(cycle.get("time_in_s"))
        n = min(len(voltage), len(current), len(time))
        voltage = voltage[:n]
        current = current[:n]
        time = time[:n]
        if n == 0:
            continue

        rest_segments = find_rest_segments(
            voltage=voltage,
            current=current,
            time=time,
            cutoff_voltage=cutoff_voltage,
            nominal_capacity=nominal_capacity,
            current_epsilon_c=current_epsilon_c,
            current_epsilon_abs=current_epsilon_abs,
            voltage_tolerance=voltage_tolerance,
            min_points=min_points,
            min_duration_s=min_duration_s,
        )
        compatible = [s for s in rest_segments if s["paper_compatible"]]
        selected = max(compatible, key=lambda s: s["duration_s"], default=None)
        soh = compute_soh(cycle, nominal_capacity, soc_interval)
        discharge_capacity = max_discharge_capacity(cycle)
        rul = compute_rul(cycle_number, eol, total_cycles)

        audit_base = {
            "dataset": dataset_name,
            "file_name": path.name,
            "cell_id": cell_id,
            "cycle_number": cycle_number,
            "cycle_position": cycle_position,
            "nominal_capacity_in_Ah": nominal_capacity,
            "cutoff_voltage": cutoff_voltage,
            "SOH": soh,
            "RUL": rul,
            "discharge_capacity_in_Ah": discharge_capacity,
            "n_points": n,
            "rest_segment_count": len(rest_segments),
            "paper_compatible_segment_count": len(compatible),
            "has_rest": bool(rest_segments),
            "has_paper_compatible_relaxation": bool(compatible),
        }
        if selected:
            audit_base.update({f"selected_{k}": v for k, v in selected.items() if k not in {"paper_compatible"}})
            for feature in extract_window_features(voltage, time, selected, windows_s, min_points):
                feature_rows.append(
                    {
                        "dataset": dataset_name,
                        "file_name": path.name,
                        "cell_id": cell_id,
                        "cycle_number": cycle_number,
                        "cycle_position": cycle_position,
                        "nominal_capacity_in_Ah": nominal_capacity,
                        "cutoff_voltage": cutoff_voltage,
                        "SOH": soh,
                        "RUL": rul,
                        "discharge_capacity_in_Ah": discharge_capacity,
                        "relax_duration_s": selected["duration_s"],
                        "relax_points": selected["n_points"],
                        "relax_start_voltage": selected["start_voltage"],
                        "relax_end_voltage": selected["end_voltage"],
                        "relax_voltage_delta": selected["voltage_delta"],
                        **feature,
                    }
                )
        audit_rows.append(audit_base)
    return audit_rows, feature_rows


def compute_coverage(audit_df: pd.DataFrame, features_df: pd.DataFrame) -> pd.DataFrame:
    if audit_df.empty:
        return pd.DataFrame()
    grouped = audit_df.groupby("dataset", dropna=False)
    rows = []
    for dataset, g in grouped:
        feature_g = features_df[features_df["dataset"] == dataset] if not features_df.empty else pd.DataFrame()
        selected_duration = g["selected_duration_s"].dropna() if "selected_duration_s" in g else pd.Series(dtype=float)
        selected_points = g["selected_n_points"].dropna() if "selected_n_points" in g else pd.Series(dtype=float)
        rows.append(
            {
                "dataset": dataset,
                "cells": int(g["file_name"].nunique()),
                "cycles": int(len(g)),
                "cycles_with_any_rest": int(g["has_rest"].sum()),
                "cycles_with_paper_compatible_relaxation": int(g["has_paper_compatible_relaxation"].sum()),
                "paper_compatible_cycle_ratio": float(g["has_paper_compatible_relaxation"].mean()),
                "median_selected_duration_s": float(selected_duration.median()) if len(selected_duration) else math.nan,
                "median_selected_points": float(selected_points.median()) if len(selected_points) else math.nan,
                "feature_rows": int(len(feature_g)),
                "available_windows": ",".join(sorted(feature_g["window"].dropna().unique())) if not feature_g.empty else "",
            }
        )
    return pd.DataFrame(rows).sort_values("dataset")


def compute_correlations(features_df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = ["relax_var", "relax_ske", "relax_max", "relax_min", "relax_mean", "relax_kur"]
    target_cols = ["SOH", "RUL", "discharge_capacity_in_Ah"]
    rows = []
    if features_df.empty:
        return pd.DataFrame()
    for (dataset, window), group in features_df.groupby(["dataset", "window"], dropna=False):
        for feature in feature_cols:
            for target in target_cols:
                sub = group[[feature, target]].replace([np.inf, -np.inf], np.nan).dropna()
                if len(sub) < 3 or sub[feature].nunique() < 2 or sub[target].nunique() < 2:
                    pearson = math.nan
                    spearman = math.nan
                else:
                    pearson = float(sub[feature].corr(sub[target], method="pearson"))
                    feature_rank = sub[feature].rank(method="average")
                    target_rank = sub[target].rank(method="average")
                    spearman = float(feature_rank.corr(target_rank, method="pearson"))
                rows.append(
                    {
                        "dataset": dataset,
                        "window": window,
                        "feature": feature,
                        "target": target,
                        "n": int(len(sub)),
                        "pearson": pearson,
                        "spearman": spearman,
                        "abs_pearson": abs(pearson) if np.isfinite(pearson) else math.nan,
                        "abs_spearman": abs(spearman) if np.isfinite(spearman) else math.nan,
                    }
                )
    return pd.DataFrame(rows).sort_values(["dataset", "window", "target", "feature"])


def write_report(output_dir: Path, coverage: pd.DataFrame, correlations: pd.DataFrame) -> None:
    lines = ["# Relaxation Feature Report", ""]
    if coverage.empty:
        lines.append("No cycle data was processed.")
    else:
        total_cells = int(coverage["cells"].sum())
        total_cycles = int(coverage["cycles"].sum())
        usable_cycles = int(coverage["cycles_with_paper_compatible_relaxation"].sum())
        lines.extend(
            [
                "## Coverage",
                "",
                f"- Datasets processed: {coverage['dataset'].nunique()}",
                f"- Cells processed: {total_cells}",
                f"- Cycles audited: {total_cycles}",
                f"- Paper-compatible full-charge relaxation cycles: {usable_cycles}",
                "",
                "Datasets with the highest compatible-cycle ratios:",
                "",
            ]
        )
        top = coverage.sort_values("paper_compatible_cycle_ratio", ascending=False).head(10)
        lines.append(dataframe_to_markdown(top))
    if not correlations.empty:
        lines.extend(["", "## Strongest SOH Correlations", ""])
        soh = correlations[correlations["target"] == "SOH"].copy()
        soh = soh[soh["n"] >= 100]
        soh = soh.sort_values("abs_spearman", ascending=False).head(20)
        lines.append("Filtered to correlation rows with `n >= 100`.")
        lines.append("")
        lines.append(dataframe_to_markdown(soh))
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "relaxation_report.md").write_text("\n".join(lines) + "\n")


def write_output_layout(
    output_dir: Path,
    audit_df: pd.DataFrame,
    features_df: pd.DataFrame,
    coverage_df: pd.DataFrame,
    corr_df: pd.DataFrame,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_df.to_csv(audit_dir / "relaxation_cycle_audit.csv", index=False)
    coverage_df.to_csv(audit_dir / "relaxation_dataset_coverage.csv", index=False)

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(combined_dir / "relaxation_cycle_features.csv", index=False)
    corr_df.to_csv(combined_dir / "relaxation_feature_correlations.csv", index=False)

    full_df = features_df[features_df["window"] == "full"].copy() if not features_df.empty else features_df
    fixed_df = features_df[features_df["window"] != "full"].copy() if not features_df.empty else features_df

    full_dir = output_dir / "full_window"
    full_dir.mkdir(parents=True, exist_ok=True)
    full_corr = compute_correlations(full_df)
    full_df.to_csv(full_dir / "relaxation_cycle_features.csv", index=False)
    full_corr.to_csv(full_dir / "relaxation_feature_correlations.csv", index=False)
    write_report(full_dir, coverage_df, full_corr)

    fixed_dir = output_dir / "fixed_windows"
    fixed_dir.mkdir(parents=True, exist_ok=True)
    fixed_corr = compute_correlations(fixed_df)
    fixed_df.to_csv(fixed_dir / "relaxation_cycle_features.csv", index=False)
    fixed_corr.to_csv(fixed_dir / "relaxation_feature_correlations.csv", index=False)
    write_report(fixed_dir, coverage_df, fixed_corr)

    by_window_root = fixed_dir / "by_window"
    if not fixed_df.empty:
        for window, group in fixed_df.groupby("window", dropna=False):
            window_dir = by_window_root / str(window).replace("/", "_")
            window_dir.mkdir(parents=True, exist_ok=True)
            window_corr = compute_correlations(group)
            group.to_csv(window_dir / "relaxation_cycle_features.csv", index=False)
            window_corr.to_csv(window_dir / "relaxation_feature_correlations.csv", index=False)


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    try:
        return df.to_markdown(index=False)
    except ImportError:
        return df.to_string(index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--datasets", nargs="*", default=None, help="Optional dataset names to process.")
    parser.add_argument("--windows-s", nargs="*", type=float, default=list(DEFAULT_WINDOWS_S))
    parser.add_argument("--current-epsilon-c", type=float, default=0.02)
    parser.add_argument("--current-epsilon-abs", type=float, default=0.05)
    parser.add_argument("--voltage-tolerance", type=float, default=0.03)
    parser.add_argument("--min-points", type=int, default=3)
    parser.add_argument("--min-duration-s", type=float, default=30.0)
    parser.add_argument("--limit-files", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    dataset_root = args.dataset_root.resolve() if args.dataset_root else repo_root / "dataset"
    output_dir = args.output_dir.resolve()
    configs = load_dataset_intervals(repo_root)
    files = iter_pkl_files(dataset_root, args.datasets)
    if args.limit_files:
        files = files[: args.limit_files]

    all_audit = []
    all_features = []
    for idx, path in enumerate(files, start=1):
        if idx == 1 or idx % 50 == 0:
            print(f"[{idx}/{len(files)}] {path.relative_to(repo_root)}", flush=True)
        audit_rows, feature_rows = process_file(
            path=path,
            repo_root=repo_root,
            configs=configs,
            windows_s=args.windows_s,
            current_epsilon_c=args.current_epsilon_c,
            current_epsilon_abs=args.current_epsilon_abs,
            voltage_tolerance=args.voltage_tolerance,
            min_points=args.min_points,
            min_duration_s=args.min_duration_s,
        )
        all_audit.extend(audit_rows)
        all_features.extend(feature_rows)

    output_dir.mkdir(parents=True, exist_ok=True)
    audit_df = pd.DataFrame(all_audit)
    features_df = pd.DataFrame(all_features)
    coverage_df = compute_coverage(audit_df, features_df)
    corr_df = compute_correlations(features_df)

    write_output_layout(output_dir, audit_df, features_df, coverage_df, corr_df)

    print(f"Wrote outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
