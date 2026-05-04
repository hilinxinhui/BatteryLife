"""
Check voltage-current curves for BatteryLife dataset cells.

This script loads all .pkl battery files from a given dataset directory,
aggregates voltage and current data across the first N cycles (default 21)
for each cell, and plots them against time in **separate** figures.

Voltage plots are saved under plots/voltage/ and current plots under plots/current/.

Original notebook: check_voltage_current_curves.ipynb
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
DEFAULT_DATASET = SCRIPT_DIR / '../dataset/SNL'
DEFAULT_LOG_DIR = SCRIPT_DIR / 'logs'
DEFAULT_PLOT_DIR = SCRIPT_DIR / 'plots'


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
        description="Check voltage-current curves for BatteryLife dataset."
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
        help='Root directory to save plot images (voltage/ and current/ subdirs will be created).'
    )
    parser.add_argument(
        '--max_cycles', type=int,
        default=21,
        help='Maximum number of cycles to aggregate per cell (default: 21).'
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    log_dir = Path(args.log_dir)
    plot_dir = Path(args.plot_dir)
    max_cycles = args.max_cycles

    logger = setup_logging(log_dir, 'check_voltage_current_curves')
    logger.info("Starting voltage-current curves check.")
    logger.info("Dataset path: %s", dataset_path.resolve())
    logger.info("Plot root directory: %s", plot_dir.resolve())
    logger.info("Max cycles per cell: %d", max_cycles)

    if not dataset_path.exists():
        logger.error("Dataset path does not exist: %s", dataset_path)
        sys.exit(1)

    voltage_plot_dir = plot_dir / 'voltage'
    current_plot_dir = plot_dir / 'current'
    voltage_plot_dir.mkdir(parents=True, exist_ok=True)
    current_plot_dir.mkdir(parents=True, exist_ok=True)

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

        currents = []
        voltages = []
        times = []

        for i in range(min(length, max_cycles)):
            cycle_data = cell_data['cycle_data'][i]
            cycle_df = pd.DataFrame()
            cycle_df['current'] = cycle_data['current_in_A']
            cycle_df['voltage'] = cycle_data['voltage_in_V']
            cycle_df['charge_capacity'] = cycle_data['charge_capacity_in_Ah']
            cycle_df['discharge_capacity'] = cycle_data['discharge_capacity_in_Ah']
            cycle_df['test_time'] = cycle_data['time_in_s']
            cycle_df['cycle_number'] = cycle_data['cycle_number']

            voltage = cycle_df['voltage'].values.tolist()
            current = cycle_df['current'].values.tolist()
            time = cycle_df['test_time'].values.tolist()

            currents += current
            voltages += voltage
            times += time

        # Voltage plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, voltages, 'b-', label='Voltage')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f'Voltage vs Time\nFile: {filename}\nCapacity: {capacity} Ah')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        voltage_plot_path = voltage_plot_dir / f"voltage_curves_{filename}.png"
        plt.savefig(voltage_plot_path, dpi=300)
        plt.close(fig)

        # Current plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, currents, 'r-', label='Current')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Current (A)')
        ax.set_title(f'Current vs Time\nFile: {filename}\nCapacity: {capacity} Ah')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        current_plot_path = current_plot_dir / f"current_curves_{filename}.png"
        plt.savefig(current_plot_path, dpi=300)
        plt.close(fig)

        logger.info(
            "Saved voltage plot -> %s, current plot -> %s",
            voltage_plot_path.name, current_plot_path.name
        )

    logger.info(
        "Voltage-current curves check completed. %d cells processed. "
        "Voltage plots: %s, Current plots: %s",
        len(files), voltage_plot_dir, current_plot_dir
    )


if __name__ == '__main__':
    main()
