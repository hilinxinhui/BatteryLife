"""
Check SOH (State of Health) curves for BatteryLife dataset cells.

This script loads all .pkl battery files from a given dataset directory,
computes the SOH for each cycle, and plots SOH vs cycle number.

SOH is calculated as:
    SOH = discharge_capacity_max / nominal_capacity / SOC_interval

Special cases:
    - Files starting with 'CALB_-10' use the minimum discharge capacity.
    - Files containing 'DefaultGroup' use the overall max discharge capacity.

Original notebook: check_soh_curves.ipynb
"""

import argparse
import logging
import os
import pickle
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_DATASET = SCRIPT_DIR / '../dataset/CALB'
DEFAULT_LOG_DIR = SCRIPT_DIR / 'logs'
DEFAULT_PLOT_DIR = SCRIPT_DIR / 'plots' / 'soh'


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
        description="Check SOH curves for BatteryLife dataset."
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
    parser.add_argument(
        '--plot_dir', type=str,
        default=str(DEFAULT_PLOT_DIR),
        help='Directory to save plot images.'
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    log_dir = Path(args.log_dir)
    plot_dir = Path(args.plot_dir)

    logger = setup_logging(log_dir, 'check_soh_curves')
    logger.info("Starting SOH curves check.")
    logger.info("Dataset path: %s", dataset_path.resolve())
    logger.info("Plot output directory: %s", plot_dir.resolve())

    if not dataset_path.exists():
        logger.error("Dataset path does not exist: %s", dataset_path)
        sys.exit(1)

    plot_dir.mkdir(parents=True, exist_ok=True)

    files = [f for f in os.listdir(dataset_path) if f.endswith('.pkl')]
    files.sort()
    logger.info("Found %d .pkl files.", len(files))

    if not files:
        logger.warning("No .pkl files found in %s", dataset_path)
        return

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
        nominal_capacity = cell_data.get('nominal_capacity_in_Ah', None)
        soc_interval = cell_data.get('SOC_interval', [0, 1])

        if nominal_capacity is None:
            logger.warning("Missing nominal_capacity_in_Ah for %s, skipping.", file)
            continue

        soc_interval_val = soc_interval[1] - soc_interval[0]
        if soc_interval_val == 0:
            soc_interval_val = 1.0
            logger.warning("SOC_interval is zero for %s, using 1.0 as fallback.", file)

        soh_values = []
        cycles = []

        for i in range(length):
            cycle_data = cell_data['cycle_data'][i]
            cycle_df = pd.DataFrame()
            cycle_df['current'] = cycle_data['current_in_A']
            cycle_df['voltage'] = cycle_data['voltage_in_V']
            cycle_df['charge_capacity'] = cycle_data['charge_capacity_in_Ah']
            cycle_df['discharge_capacity'] = cycle_data['discharge_capacity_in_Ah']
            cycle_df['test_time_s'] = cycle_data['time_in_s']
            cycle_df['cycle_number'] = cycle_data['cycle_number']
            cycles.append(i + 1)

            if file.startswith('CALB_-10'):
                soh_value = abs(cycle_df.loc[cycle_df['current'] < 0, 'discharge_capacity'].min())
            elif 'DefaultGroup' in file:
                soh_value = float(cycle_df['discharge_capacity'].max())
            else:
                soh_value = float(cycle_df.loc[cycle_df['current'] < 0, 'discharge_capacity'].max())

            soh_values.append(soh_value / nominal_capacity / soc_interval_val)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(cycles, soh_values, marker='o', markersize=2, linewidth=1)
        ax.set_xlabel('Cycle number')
        ax.set_ylabel('SOH')
        ax.grid(alpha=0.3)
        ax.set_title(f'{filename}')
        fig.tight_layout()

        plot_path = plot_dir / f"soh_curves_{filename}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)

        logger.info("Saved SOH plot for %s -> %s", filename, plot_path.name)

    logger.info("SOH curves check completed. %d plots saved to %s", len(files), plot_dir)


if __name__ == '__main__':
    main()
