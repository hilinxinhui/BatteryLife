"""
Check operating conditions (cycle stages) for BatteryLife dataset cells.

This script loads all .pkl battery files from a given dataset directory,
analyzes the current profile of each cycle, compresses consecutive identical
stages into a stage sequence, and flags cycles with anomalous multi-stage
charge or discharge behavior.

Original notebook: check_operating_conditions.ipynb
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_DATASET = SCRIPT_DIR / '../dataset/RWTH'
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


def compress_stages(stage_list):
    """
    Remove consecutive duplicates from a stage list.
    Example:
        ['rest', 'rest', 'discharge', 'discharge', 'charge']
        -> ['rest', 'discharge', 'charge']
    """
    if len(stage_list) == 0:
        return []

    compressed = [stage_list[0]]
    for s in stage_list[1:]:
        if s != compressed[-1]:
            compressed.append(s)
    return compressed


def get_cycle_stage_description(
    current_series,
    nominal_capacity,
    zero_c_rate_threshold=0.02,
    remove_rest=True
):
    """
    Convert a current series into a compressed stage description.
    Stages: 'rest', 'charge', 'discharge'.
    """
    stage_seq = []
    for current in current_series:
        c_rate = current / nominal_capacity
        if abs(c_rate) <= zero_c_rate_threshold:
            stage = 'rest'
        elif c_rate > 0:
            stage = 'charge'
        else:
            stage = 'discharge'
        stage_seq.append(stage)

    compressed_stages = compress_stages(stage_seq)

    if remove_rest:
        compressed_stages = [s for s in compressed_stages]

    return compressed_stages


def main():
    parser = argparse.ArgumentParser(
        description="Check operating conditions (cycle stages) for BatteryLife dataset."
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

    logger = setup_logging(log_dir, 'check_operating_conditions')
    logger.info("Starting operating conditions check.")
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

    anomaly_count = 0

    for file in tqdm(files, desc="Processing cells"):
        pkl_path = dataset_path / file
        try:
            with open(pkl_path, 'rb') as f:
                cell_data = pickle.load(f)
        except Exception as exc:
            logger.error("Failed to load %s: %s", file, exc)
            continue

        filename = file.replace('.pkl', '')
        length = len(cell_data['cycle_data'])
        capacity = cell_data.get('nominal_capacity_in_Ah', None)

        if capacity is None:
            logger.warning("Missing nominal_capacity_in_Ah for %s, skipping.", file)
            continue

        for i in range(length):
            cycle_data = cell_data['cycle_data'][i]
            cycle_df = pd.DataFrame()
            cycle_df['current'] = cycle_data['current_in_A']
            cycle_df['voltage'] = cycle_data['voltage_in_V']
            cycle_df['charge_capacity'] = cycle_data['charge_capacity_in_Ah']
            cycle_df['discharge_capacity'] = cycle_data['discharge_capacity_in_Ah']
            cycle_df['test_time'] = cycle_data['time_in_s']
            cycle_df['cycle_number'] = cycle_data['cycle_number']

            current = cycle_df['current'].values.tolist()

            stage_desc = get_cycle_stage_description(
                current_series=current,
                nominal_capacity=capacity,
                zero_c_rate_threshold=0.02,
                remove_rest=True
            )

            discharge_count = sum(1 for s in stage_desc if s == 'discharge')
            charge_count = sum(1 for s in stage_desc if s == 'charge')

            if discharge_count > 1 or charge_count > 1:
                msg = f"Cell {file}, Cycle {i + 1} stages: {stage_desc}"
                logger.info(msg)
                anomaly_count += 1

    logger.info(
        "Operating conditions check completed. %d anomalous cycles found across %d cells.",
        anomaly_count, len(files)
    )


if __name__ == '__main__':
    main()
