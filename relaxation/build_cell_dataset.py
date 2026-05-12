"""Build per-cell relaxation feature CSVs from split pipeline outputs."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "cycle_number",
    "relax_var",
    "relax_ske",
    "relax_max",
    "relax_min",
    "relax_mean",
    "relax_kur",
    "relax_duration_s",
    "relax_points",
    "relax_start_voltage",
    "relax_end_voltage",
    "relax_voltage_delta",
    "window_s",
    "window_points",
    "SOH",
    "RUL",
    "discharge_capacity_in_Ah",
]

SUMMARY_FILES = {
    "audit/relaxation_dataset_coverage.csv": "relaxation_dataset_coverage.csv",
    "full_window/relaxation_feature_correlations.csv": "full_window_correlations.csv",
    "fixed_windows/relaxation_feature_correlations.csv": "fixed_window_correlations.csv",
    "full_window/relaxation_report.md": "full_window_report.md",
    "fixed_windows/relaxation_report.md": "fixed_window_report.md",
}


def safe_cell_name(file_name: str) -> str:
    return file_name[:-4] if file_name.endswith(".pkl") else Path(file_name).stem


def write_groups(df: pd.DataFrame, output_root: Path, fixed_window: bool) -> None:
    for (dataset, file_name), group in df.groupby(["dataset", "file_name"], sort=False):
        cell_name = safe_cell_name(str(file_name))
        out_dir = output_root / str(dataset)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{cell_name}.csv"
        cols = [c for c in FEATURE_COLUMNS if c in group.columns]
        group = group.sort_values("cycle_number")
        if fixed_window and "window" in group.columns:
            cols = ["window"] + cols
        group[cols].to_csv(out_path, index=False)


def build_full(input_dir: Path, output_dir: Path) -> None:
    src = input_dir / "full_window" / "relaxation_cycle_features.csv"
    if not src.exists():
        raise FileNotFoundError(f"Missing {src}")
    df = pd.read_csv(src)
    write_groups(df, output_dir / "full", fixed_window=False)


def build_fixed(input_dir: Path, output_dir: Path) -> None:
    by_window = input_dir / "fixed_windows" / "by_window"
    if not by_window.exists():
        raise FileNotFoundError(f"Missing {by_window}")
    for window_dir in sorted(p for p in by_window.iterdir() if p.is_dir()):
        src = window_dir / "relaxation_cycle_features.csv"
        if not src.exists():
            continue
        df = pd.read_csv(src)
        write_groups(df, output_dir / "fixed" / window_dir.name, fixed_window=False)


def copy_summary(input_dir: Path, output_dir: Path) -> None:
    summary_dir = output_dir / "summary"
    summary_dir.mkdir(parents=True, exist_ok=True)
    for relative_src, dst_name in SUMMARY_FILES.items():
        src = input_dir / relative_src
        if src.exists():
            shutil.copy2(src, summary_dir / dst_name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "outputs_split",
        help="Split output directory produced by organize_outputs.py.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent,
        help="Directory where full/, fixed/, and summary/ will be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    build_full(input_dir, output_dir)
    build_fixed(input_dir, output_dir)
    copy_summary(input_dir, output_dir)
    print(f"Wrote per-cell relaxation datasets to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
