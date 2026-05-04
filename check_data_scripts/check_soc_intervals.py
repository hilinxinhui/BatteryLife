"""
Check SOC intervals for BatteryLife dataset cells.

This script loads all .pkl battery files from a given dataset directory,
extracts the SOC_interval field for each cell, and logs the values.
A summary of unique SOC intervals across the dataset is also provided.

Results are saved to logs only (no plots).
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

from tqdm import tqdm


SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_DATASET = SCRIPT_DIR / '../dataset/CALB'
DEFAULT_LOG_DIR = SCRIPT_DIR / 'logs'


def setup_logging(log_dir: Path, script_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_name}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def main():
    parser = argparse.ArgumentParser(
        description="Check SOC intervals for BatteryLife dataset."
    )
    parser.add_argument(
        '--dataset', type=str,
        default=str(DEFAULT_DATASET),
        help='Path to the dataset directory containing .pkl files.'
    )
    parser.add_argument(
        '--log_dir', type=str,
        default=str(DEFAULT_LOG_DIR),
        help='Directory to save log files.'
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    log_dir = Path(args.log_dir)

    logger = setup_logging(log_dir, 'check_soc_intervals')
    logger.info("Starting SOC interval check.")
    logger.info("Dataset path: %s", dataset_path.resolve())

    if not dataset_path.exists():
        logger.error("Dataset path does not exist: %s", dataset_path)
        sys.exit(1)

    files = [f for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    files.sort()
    logger.info("Found %d .pkl files.", len(files))

    if not files:
        logger.warning("No .pkl files found in %s", dataset_path)
        return

    soc_records = []
    missing_soc = 0

    for file in tqdm(files, desc="Processing cells"):
        pkl_path = dataset_path / file
        try:
            with open(pkl_path, 'rb') as f:
                cell_data = pickle.load(f)
        except Exception as exc:
            logger.error("Failed to load %s: %s", file, exc)
            continue

        filename = file.replace('.pkl', '')
        nominal_capacity = cell_data.get('nominal_capacity_in_Ah', None)
        soc_interval = cell_data.get('SOC_interval', None)

        if soc_interval is None:
            missing_soc += 1
            logger.warning("Cell %s: SOC_interval is missing.", filename)
        else:
            try:
                interval_width = soc_interval[1] - soc_interval[0]
            except Exception:
                interval_width = None

            logger.info(
                "Cell %s: SOC_interval=%s, width=%s, nominal_capacity=%s Ah",
                filename, soc_interval, interval_width, nominal_capacity
            )

        soc_records.append({
            'cell': filename,
            'soc_interval': soc_interval,
            'nominal_capacity': nominal_capacity
        })

    # Summary
    logger.info("=" * 60)
    logger.info("SOC INTERVAL SUMMARY")
    logger.info("=" * 60)
    logger.info("Total cells processed: %d", len(soc_records))
    logger.info("Cells with missing SOC_interval: %d", missing_soc)

    # Unique SOC intervals
    valid_socs = [r['soc_interval'] for r in soc_records if r['soc_interval'] is not None]
    if valid_socs:
        try:
            unique_socs = sorted(set(tuple(s) for s in valid_socs))
            logger.info("Unique SOC intervals found (%d distinct):", len(unique_socs))
            for us in unique_socs:
                count = sum(1 for r in soc_records if r['soc_interval'] is not None and tuple(r['soc_interval']) == us)
                logger.info("  %s -> %d cells", list(us), count)
        except Exception as exc:
            logger.warning("Could not compute unique SOC intervals: %s", exc)
            logger.info("Raw SOC intervals: %s", valid_socs)
    else:
        logger.info("No valid SOC intervals found.")

    logger.info("=" * 60)
    logger.info("SOC interval check completed.")


if __name__ == '__main__':
    main()
