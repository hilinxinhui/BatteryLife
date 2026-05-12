"""Split flat relaxation outputs into audit, full-window, and fixed-window outputs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from relaxation_features import compute_correlations, write_report


FEATURE_FILE = "relaxation_cycle_features.csv"
AUDIT_FILE = "relaxation_cycle_audit.csv"
COVERAGE_FILE = "relaxation_dataset_coverage.csv"
CORRELATION_FILE = "relaxation_feature_correlations.csv"


def copy_if_exists(src: Path, dst: Path) -> None:
    if src.exists():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)


def write_partition(df: pd.DataFrame, out_dir: Path, coverage: pd.DataFrame) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / FEATURE_FILE, index=False)
    corr = compute_correlations(df)
    corr.to_csv(out_dir / CORRELATION_FILE, index=False)
    write_report(out_dir, coverage, corr)


def organize(input_dir: Path, output_dir: Path) -> None:
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    coverage_path = input_dir / COVERAGE_FILE
    if not coverage_path.exists():
        raise FileNotFoundError(f"Missing {coverage_path}")
    coverage = pd.read_csv(coverage_path)

    copy_if_exists(input_dir / AUDIT_FILE, output_dir / "audit" / AUDIT_FILE)
    copy_if_exists(input_dir / COVERAGE_FILE, output_dir / "audit" / COVERAGE_FILE)

    features_path = input_dir / FEATURE_FILE
    if not features_path.exists():
        raise FileNotFoundError(f"Missing {features_path}")
    features = pd.read_csv(features_path)

    combined_dir = output_dir / "combined"
    combined_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(combined_dir / FEATURE_FILE, index=False)
    combined_corr = compute_correlations(features)
    combined_corr.to_csv(combined_dir / CORRELATION_FILE, index=False)

    full = features[features["window"] == "full"].copy()
    fixed = features[features["window"] != "full"].copy()
    write_partition(full, output_dir / "full_window", coverage)
    write_partition(fixed, output_dir / "fixed_windows", coverage)

    by_window_root = output_dir / "fixed_windows" / "by_window"
    for window, group in fixed.groupby("window", dropna=False):
        safe_window = str(window).replace("/", "_")
        window_dir = by_window_root / safe_window
        window_dir.mkdir(parents=True, exist_ok=True)
        group.to_csv(window_dir / FEATURE_FILE, index=False)
        corr = compute_correlations(group)
        corr.to_csv(window_dir / CORRELATION_FILE, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs_split")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    organize(args.input_dir, args.output_dir)
    print(f"Wrote split outputs to {args.output_dir.resolve()}", flush=True)


if __name__ == "__main__":
    main()
