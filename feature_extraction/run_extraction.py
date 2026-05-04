"""CLI entry point for feature extraction across all BatteryLife datasets."""

import os
import sys
import json
import argparse
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from feature_extraction.extractor_factory import get_extractor
from feature_extraction.utils import load_battery_data, list_dataset_cells


# Valid dataset subdirectories (exclude non-data dirs)
VALID_DATASETS = [
    "CALB", "CALCE", "HNEI", "HUST", "ISU_ILCC", "MATR",
    "MICH", "MICH_EXP", "NA-ion", "RWTH", "SDU", "SNL",
    "Stanford", "Stanford_2", "Tongji", "UL_PUR", "XJTU", "ZN-coin",
]


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_dataset(dataset_name: str, dataset_dir: str, output_dir: str, config: dict):
    """Extract features for all cells in one dataset."""
    print(f"\n{'='*60}", flush=True)
    print(f"Extracting dataset: {dataset_name}", flush=True)
    print(f"{'='*60}", flush=True)

    cells = list_dataset_cells(dataset_dir, dataset_name)
    if not cells:
        print(f"  No .pkl files found for {dataset_name}, skipping.", flush=True)
        return

    ds_config = config.get(dataset_name)
    if ds_config is None:
        print(f"  WARNING: No config found for {dataset_name}, skipping.", flush=True)
        return

    extractor = get_extractor(dataset_name, ds_config)
    out_ds_dir = Path(output_dir) / dataset_name
    out_ds_dir.mkdir(parents=True, exist_ok=True)

    # Progress bar for cells within this dataset
    pbar = tqdm(
        total=len(cells),
        desc=f"  {dataset_name}",
        position=0,
        leave=True,
        file=sys.stdout,
        ncols=80,
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
    )

    for pkl_path in cells:
        try:
            cell_data = load_battery_data(pkl_path)
            df = extractor.extract_cell(cell_data)

            cell_id = cell_data.get("cell_id", Path(pkl_path).stem)
            csv_path = out_ds_dir / f"{cell_id}.csv"
            df.to_csv(csv_path, index=False)
        except Exception as e:
            tqdm.write(f"    ERROR processing {Path(pkl_path).name}: {e}", file=sys.stdout)
        finally:
            pbar.update(1)

    pbar.close()
    print(f"  Done: {len(cells)} cells → {out_ds_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Extract PINN4SOH-style 16 features + SOH + RUL for BatteryLife datasets."
    )
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset",
        help="Root directory containing dataset subdirectories (default: dataset)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="extracted_features",
        help="Output directory for extracted CSV files (default: extracted_features)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="feature_extraction/configs/dataset_intervals.json",
        help="Path to dataset interval config JSON",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated list of dataset names to process (default: all)",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    if args.datasets:
        datasets = [d.strip() for d in args.datasets.split(",")]
    else:
        datasets = VALID_DATASETS

    print(f"\n{'#'*60}", flush=True)
    print(f"# Feature Extraction Pipeline", flush=True)
    print(f"# Datasets to process: {len(datasets)}", flush=True)
    print(f"# Output directory: {Path(args.output_dir).resolve()}", flush=True)
    print(f"{'#'*60}", flush=True)

    for dataset_name in datasets:
        extract_dataset(dataset_name, args.dataset_dir, args.output_dir, config)

    print(f"\n{'='*60}", flush=True)
    print("Feature extraction complete!", flush=True)
    print(f"Output directory: {Path(args.output_dir).resolve()}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    main()
