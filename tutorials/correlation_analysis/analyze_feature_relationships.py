"""Analyze feature/label/model-performance relationships across two repos.

Inputs
------
- BatteryLife feature CSVs, default:
  /Users/lxh/Desktop/codes/SOHbenchmark/dataset/BatteryLife
- SOHbenchmark prediction/per-battery metric outputs, default:
  /Users/lxh/Desktop/codes/SOHbenchmark/results

Outputs are written under tutorials/correlation_analysis/outputs by default.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "voltage_mean",
    "voltage_std",
    "voltage_kurtosis",
    "voltage_skewness",
    "CC_Q",
    "CC_charge_time",
    "voltage_slope",
    "voltage_entropy",
    "current_mean",
    "current_std",
    "current_kurtosis",
    "current_skewness",
    "CV_Q",
    "CV_charge_time",
    "current_slope",
    "current_entropy",
]
TARGET_COLUMNS = ["SOH", "RUL"]
METRIC_COLUMNS = [
    "mae",
    "mape",
    "mse",
    "rmse",
    "max_ae",
    "bias",
    "r2",
    "pearson_r",
    "endpoint_ae",
    "endpoint_ape",
]
MODEL_NAMES = {"MLP", "LSTM", "GRU", "Attention", "CNN"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--features-root",
        default="/Users/lxh/Desktop/codes/SOHbenchmark/dataset/BatteryLife",
        help="Root containing extracted feature CSV subdirectories.",
    )
    parser.add_argument(
        "--results-root",
        default="/Users/lxh/Desktop/codes/SOHbenchmark/results",
        help="SOHbenchmark results root.",
    )
    parser.add_argument(
        "--output-dir",
        default="tutorials/correlation_analysis/outputs",
        help="Directory for CSV/Markdown outputs.",
    )
    parser.add_argument(
        "--max-rows-per-dataset",
        type=int,
        default=250000,
        help="Deterministic sample cap for dataset-level correlations.",
    )
    parser.add_argument(
        "--exclude-tasks",
        default="MIX_large",
        help="Comma-separated SOHbenchmark tasks to exclude from metric/performance mapping.",
    )
    return parser.parse_args()


def safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    pair = pd.concat([x, y], axis=1).replace([np.inf, -np.inf], np.nan).dropna()
    if len(pair) < 3:
        return np.nan
    if pair.iloc[:, 0].nunique(dropna=True) < 2 or pair.iloc[:, 1].nunique(dropna=True) < 2:
        return np.nan
    if method == "spearman":
        left = pair.iloc[:, 0].rank(method="average")
        right = pair.iloc[:, 1].rank(method="average")
        value = left.corr(right, method="pearson")
    else:
        value = pair.iloc[:, 0].corr(pair.iloc[:, 1], method=method)
    return float(value) if pd.notna(value) else np.nan


def finite_float(value) -> float:
    try:
        out = float(value)
    except Exception:
        return np.nan
    return out if math.isfinite(out) else np.nan


def iter_feature_csvs(features_root: Path) -> Iterable[tuple[str, Path]]:
    for dataset_dir in sorted(p for p in features_root.iterdir() if p.is_dir()):
        for csv_path in sorted(dataset_dir.glob("*.csv")):
            yield dataset_dir.name, csv_path


def deterministic_sample(df: pd.DataFrame, max_rows: int) -> pd.DataFrame:
    if max_rows <= 0 or len(df) <= max_rows:
        return df
    # Evenly spaced rows preserve the life trajectory better than random sampling.
    idx = np.linspace(0, len(df) - 1, num=max_rows, dtype=np.int64)
    return df.iloc[idx].reset_index(drop=True)


def load_dataset_frame(features_root: Path, dataset: str, max_rows: int) -> pd.DataFrame:
    frames = []
    for csv_path in sorted((features_root / dataset).glob("*.csv")):
        frame = pd.read_csv(csv_path, usecols=["cycle_number", *FEATURE_COLUMNS, *TARGET_COLUMNS])
        frame.insert(0, "battery_name", csv_path.stem)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return deterministic_sample(pd.concat(frames, ignore_index=True), max_rows)


def compute_feature_truth_correlations(features_root: Path, output_dir: Path, max_rows: int) -> pd.DataFrame:
    rows = []
    for dataset_dir in sorted(p for p in features_root.iterdir() if p.is_dir()):
        dataset = dataset_dir.name
        frame = load_dataset_frame(features_root, dataset, max_rows)
        if frame.empty:
            continue
        for target in TARGET_COLUMNS:
            for feature in FEATURE_COLUMNS:
                rows.append(
                    {
                        "dataset": dataset,
                        "target": target,
                        "feature": feature,
                        "n_rows_used": len(frame),
                        "pearson": safe_corr(frame[feature], frame[target], "pearson"),
                        "spearman": safe_corr(frame[feature], frame[target], "spearman"),
                    }
                )
    out = pd.DataFrame(rows)
    out["abs_spearman"] = out["spearman"].abs()
    out["abs_pearson"] = out["pearson"].abs()
    out.to_csv(output_dir / "feature_truth_correlations_by_dataset.csv", index=False)
    return out


def feature_corr_structure(frame: pd.DataFrame) -> dict:
    feat = frame[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    out = {
        "feature_rows": len(feat),
        "mean_abs_feature_corr": np.nan,
        "median_abs_feature_corr": np.nan,
        "max_abs_feature_corr": np.nan,
        "effective_feature_rank": np.nan,
        "zero_cv_fraction": np.nan,
    }
    if len(feat) >= 3:
        corr = feat.rank(method="average").corr(method="pearson").to_numpy(dtype=float)
        tri = np.triu_indices_from(corr, k=1)
        vals = np.abs(corr[tri])
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out["mean_abs_feature_corr"] = float(np.mean(vals))
            out["median_abs_feature_corr"] = float(np.median(vals))
            out["max_abs_feature_corr"] = float(np.max(vals))
        pearson_corr = feat.corr(method="pearson").fillna(0.0).to_numpy(dtype=float)
        eigvals = np.linalg.eigvalsh(pearson_corr)
        eigvals = np.clip(eigvals, 0, None)
        if eigvals.sum() > 0:
            p = eigvals / eigvals.sum()
            p = p[p > 0]
            out["effective_feature_rank"] = float(np.exp(-np.sum(p * np.log(p))))
    cv_cols = ["current_mean", "current_std", "CV_Q", "CV_charge_time", "current_slope", "current_entropy"]
    if all(col in frame.columns for col in cv_cols):
        out["zero_cv_fraction"] = float((frame[cv_cols].abs().sum(axis=1) == 0).mean())
    return out


def compute_battery_feature_profiles(features_root: Path, output_dir: Path) -> pd.DataFrame:
    rows = []
    for dataset, csv_path in iter_feature_csvs(features_root):
        frame = pd.read_csv(csv_path, usecols=["cycle_number", *FEATURE_COLUMNS, *TARGET_COLUMNS])
        row = {
            "dataset_folder": dataset,
            "battery_name": csv_path.stem,
            "n_cycles": len(frame),
            "cycle_min": finite_float(frame["cycle_number"].min()),
            "cycle_max": finite_float(frame["cycle_number"].max()),
            "SOH_min": finite_float(frame["SOH"].min()),
            "SOH_max": finite_float(frame["SOH"].max()),
            "SOH_range": finite_float(frame["SOH"].max() - frame["SOH"].min()),
            "RUL_min": finite_float(frame["RUL"].min()),
            "RUL_max": finite_float(frame["RUL"].max()),
            "RUL_range": finite_float(frame["RUL"].max() - frame["RUL"].min()),
        }
        row.update(feature_corr_structure(frame))
        for feature in FEATURE_COLUMNS:
            series = frame[feature].replace([np.inf, -np.inf], np.nan)
            row[f"{feature}__mean"] = finite_float(series.mean())
            row[f"{feature}__std"] = finite_float(series.std(ddof=1))
            row[f"{feature}__spearman_cycle"] = safe_corr(frame["cycle_number"], series, "spearman")
            for target in TARGET_COLUMNS:
                row[f"{feature}__spearman_{target}"] = safe_corr(series, frame[target], "spearman")
        rows.append(row)
    out = pd.DataFrame(rows)
    out.to_csv(output_dir / "battery_feature_profiles.csv", index=False)
    return out


def parse_metric_path(path: Path, results_root: Path) -> dict:
    rel = path.relative_to(results_root)
    parts = rel.parts
    if len(parts) >= 5 and parts[0] in {"SOH", "RUL"}:
        # Legacy layout: target/task/model/experiment/battery_metrics.csv
        return {
            "target": parts[0],
            "task": parts[1],
            "model": parts[2],
            "experiment": parts[3],
            "source_layout": "nested",
        }

    # TSLib layout: soh_XJTU_SOH_CNN/battery_metrics.csv
    setting = parts[0]
    split = setting.split("_")
    lower_target = split[0].upper()
    target_idx = None
    for idx, token in enumerate(split):
        if token in {"SOH", "RUL"}:
            target_idx = idx
            break
    if lower_target not in {"SOH", "RUL"} or target_idx is None:
        raise ValueError(f"Cannot parse result setting: {setting}")
    model_idx = None
    for idx in range(target_idx + 1, len(split)):
        if split[idx] in MODEL_NAMES:
            model_idx = idx
            break
    if model_idx is None:
        raise ValueError(f"Cannot parse model from result setting: {setting}")
    return {
        "target": split[target_idx],
        "task": "_".join(split[1:target_idx]),
        "model": split[model_idx],
        "experiment": "_".join(split[model_idx + 1 :]) or "experiment1",
        "source_layout": "flat",
    }


def load_battery_metrics(results_root: Path, output_dir: Path, exclude_tasks: set[str]) -> pd.DataFrame:
    rows = []
    metric_paths = sorted(results_root.glob("*/battery_metrics.csv"))
    metric_paths += sorted(results_root.glob("*/*/*/*/battery_metrics.csv"))
    for path in metric_paths:
        meta = parse_metric_path(path, results_root)
        frame = pd.read_csv(path)
        for key, value in meta.items():
            frame[key] = value
        rows.append(frame)
    out = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if not out.empty:
        # If both flat TSLib and nested legacy results exist, prefer the flat
        # result for identical target/task/model/battery rows.
        out["_layout_rank"] = out["source_layout"].map({"flat": 0, "nested": 1}).fillna(2)
        out = out.sort_values("_layout_rank").drop_duplicates(
            ["target", "task", "model", "battery_name"], keep="first"
        )
        out = out.drop(columns=["_layout_rank"])
        if exclude_tasks:
            out = out[~out["task"].isin(exclude_tasks)].copy()
        front = ["target", "task", "model", "experiment", "source_layout", "battery_name"]
        out = out[front + [col for col in out.columns if col not in front]]
    out.to_csv(output_dir / "sohbenchmark_battery_metrics_long.csv", index=False)
    return out


def join_profiles_and_metrics(profiles: pd.DataFrame, metrics: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if profiles.empty or metrics.empty:
        return pd.DataFrame()

    task_to_folder = {
        "CALB42": "CALB",
        "CALB2024": "CALB",
        "NAion": "NA-ion",
        "NAion42": "NA-ion",
        "NAion2024": "NA-ion",
        "ZN-coin42": "ZN-coin",
        "ZN-coin2024": "ZN-coin",
    }
    metrics = metrics.copy()
    metrics["dataset_folder"] = metrics["task"].map(task_to_folder).fillna(metrics["task"])

    non_mix_metrics = metrics[metrics["task"] != "MIX_large"].copy()
    mix_metrics = metrics[metrics["task"] == "MIX_large"].copy()

    joined_parts = []
    if not non_mix_metrics.empty:
        joined_parts.append(non_mix_metrics.merge(
        profiles,
        on=["dataset_folder", "battery_name"],
        how="left",
        validate="many_to_one",
        ))

    if not mix_metrics.empty:
        # MIX_large spans several physical folders. Use battery_name only after
        # removing duplicate names that cannot be resolved safely (currently
        # Stanford vs Stanford_2).
        unique_profiles = profiles.drop_duplicates("battery_name", keep=False)
        mix_joined = mix_metrics.drop(columns=["dataset_folder"]).merge(
            unique_profiles,
            on="battery_name",
            how="left",
            validate="many_to_one",
        )
        joined_parts.append(mix_joined)

    joined = pd.concat(joined_parts, ignore_index=True) if joined_parts else pd.DataFrame()

    joined.to_csv(output_dir / "battery_metrics_with_feature_profiles.csv", index=False)
    return joined


def compute_performance_mapping(joined: pd.DataFrame, output_dir: Path) -> pd.DataFrame:
    if joined.empty:
        return pd.DataFrame()
    descriptor_cols = [
        col
        for col in joined.columns
        if col
        not in {
            "target",
            "task",
            "model",
            "experiment",
            "battery_name",
            "dataset_folder",
            *METRIC_COLUMNS,
        }
        and pd.api.types.is_numeric_dtype(joined[col])
    ]
    rows = []
    group_keys = [
        ("all", ["target"]),
        ("by_model", ["target", "model"]),
        ("by_task", ["target", "task"]),
    ]
    for scope, keys in group_keys:
        for key_values, group in joined.groupby(keys, dropna=False):
            if not isinstance(key_values, tuple):
                key_values = (key_values,)
            base = {"scope": scope, "n_battery_model_rows": len(group)}
            base.update(dict(zip(keys, key_values)))
            for metric in ["mae", "mape", "rmse", "r2", "pearson_r"]:
                if metric not in group.columns:
                    continue
                for desc in descriptor_cols:
                    corr = safe_corr(group[desc], group[metric], "spearman")
                    if pd.notna(corr):
                        rows.append({**base, "metric": metric, "descriptor": desc, "spearman": corr})
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_spearman"] = out["spearman"].abs()
        out = out.sort_values(["scope", "target", "metric", "abs_spearman"], ascending=[True, True, True, False])
    out.to_csv(output_dir / "feature_profile_performance_mapping.csv", index=False)
    return out


def load_prediction_npz_relationships(features_root: Path, results_root: Path, output_dir: Path) -> pd.DataFrame:
    rows = []
    npz_paths = sorted(results_root.glob("*/*/res.npz"))
    npz_paths += sorted(results_root.glob("*/*/*/*/test_predictions/*/res.npz"))
    for npz_path in npz_paths:
        metric_path = nearest_metric_file(npz_path, results_root)
        if metric_path is None:
            continue
        meta = parse_metric_path(metric_path, results_root)
        battery_name = npz_path.parent.name
        feature_path = find_feature_csv(features_root, battery_name, meta.get("task"))
        if feature_path is None:
            continue
        data = np.load(npz_path)
        pred = pd.Series(data["pred_label"].reshape(-1), name="pred")
        true = pd.Series(data["true_label"].reshape(-1), name="true")
        feat = pd.read_csv(feature_path, usecols=["cycle_number", *FEATURE_COLUMNS, *TARGET_COLUMNS])
        n = min(len(feat), len(pred), len(true))
        if n < 3:
            continue
        feat = feat.iloc[:n].reset_index(drop=True)
        pred = pred.iloc[:n].reset_index(drop=True)
        true = true.iloc[:n].reset_index(drop=True)
        abs_error = (pred - true).abs()
        rel_error = abs_error / true.replace(0, np.nan).abs()
        for feature in FEATURE_COLUMNS:
            rows.append(
                {
                    **meta,
                    "battery_name": battery_name,
                    "feature": feature,
                    "n_rows": n,
                    "feature_pred_spearman": safe_corr(feat[feature], pred, "spearman"),
                    "feature_abs_error_spearman": safe_corr(feat[feature], abs_error, "spearman"),
                    "feature_rel_error_spearman": safe_corr(feat[feature], rel_error, "spearman"),
                }
            )
    out = pd.DataFrame(rows)
    if not out.empty:
        out["abs_feature_abs_error_spearman"] = out["feature_abs_error_spearman"].abs()
    out.to_csv(output_dir / "feature_prediction_error_relationships_available_npz.csv", index=False)
    return out


def nearest_metric_file(path: Path, results_root: Path) -> Path | None:
    for parent in [path.parent, *path.parents]:
        if parent == results_root.parent:
            break
        candidate = parent / "battery_metrics.csv"
        if candidate.exists():
            return candidate
    return None


def find_feature_csv(features_root: Path, battery_name: str, task: str | None = None) -> Path | None:
    folder = None
    if task:
        folder = {
            "CALB42": "CALB",
            "CALB2024": "CALB",
            "NAion": "NA-ion",
            "NAion42": "NA-ion",
            "NAion2024": "NA-ion",
            "ZN-coin42": "ZN-coin",
            "ZN-coin2024": "ZN-coin",
        }.get(task, task)
    if folder and (features_root / folder / f"{battery_name}.csv").exists():
        return features_root / folder / f"{battery_name}.csv"
    matches = list(features_root.glob(f"*{os.sep}{battery_name}.csv"))
    if matches:
        if len(matches) > 1:
            return None
        return matches[0]
    # Sanitized result folder names currently preserve all known BatteryLife names,
    # but keep a fallback for future outputs with punctuation replaced.
    for path in features_root.glob("*/*.csv"):
        if path.stem == battery_name:
            return path
    return None


def top_records(df: pd.DataFrame, n: int, by: str) -> list[dict]:
    if df.empty or by not in df.columns:
        return []
    return df.sort_values(by, ascending=False).head(n).replace({np.nan: None}).to_dict("records")


def write_summary(
    output_dir: Path,
    truth_corr: pd.DataFrame,
    profiles: pd.DataFrame,
    metrics: pd.DataFrame,
    joined: pd.DataFrame,
    mapping: pd.DataFrame,
    pred_rel: pd.DataFrame,
) -> None:
    lines = []
    lines.append("# Feature Relationship Analysis\n")
    lines.append("## Data Coverage\n")
    lines.append(f"- Battery feature profiles: {len(profiles):,} batteries/cells")
    lines.append(f"- SOHbenchmark battery metric rows: {len(metrics):,}")
    lines.append(f"- Joined metric/profile rows: {len(joined):,}")
    lines.append(f"- Available per-sample prediction relationship rows: {len(pred_rel):,}")
    if not metrics.empty:
        lines.append(f"- Targets: {', '.join(sorted(metrics['target'].dropna().unique()))}")
        lines.append(f"- Models: {', '.join(sorted(metrics['model'].dropna().unique()))}")
        lines.append(f"- Tasks: {len(metrics['task'].dropna().unique())}")
    lines.append("")

    lines.append("## Strongest Feature-Truth Links\n")
    for target in TARGET_COLUMNS:
        subset = truth_corr[truth_corr["target"] == target]
        lines.append(f"### {target}")
        for row in top_records(subset, 10, "abs_spearman"):
            lines.append(
                f"- {row['dataset']} / {row['feature']}: "
                f"Spearman={row['spearman']:.4f}, Pearson={row['pearson']:.4f}, n={row['n_rows_used']}"
            )
    lines.append("")

    lines.append("## Feature Profile to Model Performance\n")
    if not mapping.empty:
        for target in sorted(mapping["target"].dropna().unique()):
            subset = mapping[(mapping["target"] == target) & (mapping["scope"] == "all")]
            lines.append(f"### {target}")
            for metric in ["mae", "mape", "rmse", "r2", "pearson_r"]:
                metric_subset = subset[subset["metric"] == metric]
                if metric_subset.empty:
                    continue
                best = metric_subset.sort_values("abs_spearman", ascending=False).iloc[0]
                lines.append(
                    f"- {metric}: strongest descriptor `{best['descriptor']}` "
                    f"Spearman={best['spearman']:.4f} over {int(best['n_battery_model_rows'])} rows"
                )
    lines.append("")

    lines.append("## Prediction/Output Relationship Coverage\n")
    if pred_rel.empty:
        lines.append("- No usable `res.npz` prediction files were found.")
    else:
        meta_cols = ["target", "task", "model", "battery_name"]
        available = pred_rel[meta_cols].drop_duplicates()
        lines.append(f"- Usable prediction files: {len(available)}")
        for _, row in available.iterrows():
            lines.append(f"- {row['target']} / {row['task']} / {row['model']} / {row['battery_name']}")
        strongest = top_records(pred_rel, 10, "abs_feature_abs_error_spearman")
        lines.append("- Strongest feature-error links among available predictions:")
        for row in strongest:
            lines.append(
                f"  - {row['target']} / {row['task']} / {row['model']} / "
                f"{row['battery_name']} / {row['feature']}: "
                f"abs-error Spearman={row['feature_abs_error_spearman']:.4f}"
            )
    lines.append("")

    lines.append("## Output Files\n")
    for name in [
        "feature_truth_correlations_by_dataset.csv",
        "battery_feature_profiles.csv",
        "sohbenchmark_battery_metrics_long.csv",
        "battery_metrics_with_feature_profiles.csv",
        "feature_profile_performance_mapping.csv",
        "feature_prediction_error_relationships_available_npz.csv",
    ]:
        lines.append(f"- `{name}`")

    (output_dir / "analysis_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manifest(output_dir: Path, args: argparse.Namespace) -> None:
    manifest = {
        "features_root": str(Path(args.features_root).resolve()),
        "results_root": str(Path(args.results_root).resolve()),
        "max_rows_per_dataset": args.max_rows_per_dataset,
        "exclude_tasks": [task.strip() for task in args.exclude_tasks.split(",") if task.strip()],
        "feature_columns": FEATURE_COLUMNS,
        "target_columns": TARGET_COLUMNS,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    features_root = Path(args.features_root).expanduser().resolve()
    results_root = Path(args.results_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not features_root.exists():
        raise FileNotFoundError(features_root)
    if not results_root.exists():
        raise FileNotFoundError(results_root)

    print("[1/6] feature-truth correlations", flush=True)
    truth_corr = compute_feature_truth_correlations(features_root, output_dir, args.max_rows_per_dataset)
    print("[2/6] battery feature profiles", flush=True)
    profiles = compute_battery_feature_profiles(features_root, output_dir)
    print("[3/6] SOHbenchmark battery metrics", flush=True)
    exclude_tasks = {task.strip() for task in args.exclude_tasks.split(",") if task.strip()}
    metrics = load_battery_metrics(results_root, output_dir, exclude_tasks)
    print("[4/6] join profiles and metrics", flush=True)
    joined = join_profiles_and_metrics(profiles, metrics, output_dir)
    print("[5/6] feature profile/performance mapping", flush=True)
    mapping = compute_performance_mapping(joined, output_dir)
    print("[6/6] available prediction-output/error relationships", flush=True)
    pred_rel = load_prediction_npz_relationships(features_root, results_root, output_dir)
    write_summary(output_dir, truth_corr, profiles, metrics, joined, mapping, pred_rel)
    write_manifest(output_dir, args)
    print(f"done: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
