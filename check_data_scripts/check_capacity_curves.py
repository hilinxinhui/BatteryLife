"""
Check capacity curves for BatteryLife dataset cells.

This script loads all .pkl battery files from a given dataset directory,
aggregates charge/discharge capacity across all cycles for each cell,
and plots them against time.

Original notebook: check_capacity_curves.ipynb
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
import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_DATASET = SCRIPT_DIR / '../dataset/NA-ion'
DEFAULT_LOG_DIR = SCRIPT_DIR / 'logs'
DEFAULT_PLOT_DIR = SCRIPT_DIR / 'plots' / 'capacity'


def setup_logging(log_dir: Path, script_name: str) -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / f"{script_name}.log"

    logger = logging.getLogger(script_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # clear existing handlers

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
        description="Check charge/discharge capacity curves for BatteryLife dataset."
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

    logger = setup_logging(log_dir, 'check_capacity_curves')
    logger.info("Starting capacity curves check.")
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
        capacity = cell_data.get('nominal_capacity_in_Ah', None)

        charge_caps = []
        discharge_caps = []
        times = []

        for i in range(length):
            cycle_data = cell_data['cycle_data'][i]
            cycle_df = pd.DataFrame()
            cycle_df['current'] = cycle_data['current_in_A']
            cycle_df['voltage'] = cycle_data['voltage_in_V']
            cycle_df['charge_capacity'] = cycle_data['charge_capacity_in_Ah']
            cycle_df['discharge_capacity'] = cycle_data['discharge_capacity_in_Ah']
            cycle_df['test_time'] = cycle_data['time_in_s']
            cycle_df['cycle_number'] = cycle_data['cycle_number']

            charge_cap = cycle_df['charge_capacity'].values.tolist()
            discharge_cap = cycle_df['discharge_capacity'].values.tolist()
            time = cycle_df['test_time'].values.tolist()

            charge_caps += charge_cap
            discharge_caps += discharge_cap
            times += time

        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(times, charge_caps, 'b-', label='Charge Capacity')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Charge Capacity (Ah)', color='b')
        ax1.tick_params(axis='y', labelcolor='b')

        ax2 = ax1.twinx()
        ax2.plot(times, discharge_caps, 'r-', label='Discharge Capacity')
        ax2.set_ylabel('Discharge Capacity (Ah)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')

        plt.title(f'Charge/Discharge Capacity vs Time\nFile: {filename}')
        fig.tight_layout()

        plot_path = plot_dir / f"capacity_curves_{filename}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)

        logger.info("Saved plot for %s -> %s", filename, plot_path.name)

    logger.info("Capacity curves check completed. %d plots saved to %s", len(files), plot_dir)


if __name__ == '__main__':
    main()
