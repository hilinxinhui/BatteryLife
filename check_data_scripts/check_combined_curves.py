"""
Combined visualization of capacity and SOH curves for BatteryLife dataset cells.

This script generates a single large figure with a 2x2 subplot layout for each battery:
    - Row 0, Col 0: Charge & Discharge capacity vs time (dual-y)
    - Row 0, Col 1: SOH vs cycle number
    - Row 1, Col 0: Charge capacity vs time
    - Row 1, Col 1: Discharge capacity vs time

Plots are saved under plots/combined/ by default.
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
DEFAULT_PLOT_DIR = SCRIPT_DIR / 'plots' / 'combined'


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
        description="Combined capacity and SOH curves visualization for BatteryLife dataset."
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
        help='Directory to save plot images (default: plots/combined).'
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    log_dir = Path(args.log_dir)
    plot_dir = Path(args.plot_dir)

    logger = setup_logging(log_dir, 'check_combined_curves')
    logger.info("Starting combined curves visualization.")
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

        # Aggregate capacity data across all cycles
        charge_caps = []
        discharge_caps = []
        times = []

        # SOH data per cycle
        soh_values = []
        cycles = []

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
            cycles.append(i + 1)

            # Compute SOH
            if file.startswith('CALB_-10'):
                soh_value = abs(cycle_df.loc[cycle_df['current'] < 0, 'discharge_capacity'].min())
            elif 'DefaultGroup' in file:
                soh_value = float(cycle_df['discharge_capacity'].max())
            else:
                soh_value = float(cycle_df.loc[cycle_df['current'] < 0, 'discharge_capacity'].max())

            soh_values.append(soh_value / nominal_capacity / soc_interval_val)

        # Create 2x2 combined figure
        fig, axes = plt.subplots(2, 2, figsize=(16, 10))

        # (0,0): Charge & Discharge capacity vs time (dual-y)
        ax = axes[0, 0]
        ax.plot(times, charge_caps, 'b-', linewidth=0.8, label='Charge Capacity')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Charge Capacity (Ah)', color='b')
        ax.tick_params(axis='y', labelcolor='b')
        ax2 = ax.twinx()
        ax2.plot(times, discharge_caps, 'r-', linewidth=0.8, label='Discharge Capacity')
        ax2.set_ylabel('Discharge Capacity (Ah)', color='r')
        ax2.tick_params(axis='y', labelcolor='r')
        ax.set_title('Charge & Discharge Capacity vs Time')

        # (0,1): SOH vs cycle number
        ax = axes[0, 1]
        ax.plot(cycles, soh_values, 'g-', marker='o', markersize=2, linewidth=1)
        ax.set_xlabel('Cycle number')
        ax.set_ylabel('SOH')
        ax.grid(alpha=0.3)
        ax.set_title('SOH vs Cycle Number')

        # (1,0): Charge capacity vs time
        ax = axes[1, 0]
        ax.plot(times, charge_caps, 'b-', linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Charge Capacity (Ah)')
        ax.set_title('Charge Capacity vs Time')
        ax.grid(alpha=0.3)

        # (1,1): Discharge capacity vs time
        ax = axes[1, 1]
        ax.plot(times, discharge_caps, 'r-', linewidth=0.8)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Discharge Capacity (Ah)')
        ax.set_title('Discharge Capacity vs Time')
        ax.grid(alpha=0.3)

        fig.suptitle(f'Battery Comprehensive Curves - {filename}', fontsize=14, fontweight='bold')
        fig.tight_layout(rect=[0, 0, 1, 0.96])

        plot_path = plot_dir / f"combined_curves_{filename}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)

        logger.info("Saved combined plot for %s -> %s", filename, plot_path.name)

    logger.info(
        "Combined curves visualization completed. %d plots saved to %s",
        len(files), plot_dir
    )


if __name__ == '__main__':
    main()
