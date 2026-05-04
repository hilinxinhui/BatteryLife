"""
Check time monotonicity for BatteryLife dataset cells.

This script loads all .pkl battery files from a given dataset directory,
iterates through every cycle, and verifies that the 'time_in_s' sequence
is strictly monotonically increasing. Any non-monotonic points are logged
and marked on the generated time-sequence plots.

Plots are saved under plots/times/ by default.

Original notebook: check_times.ipynb
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
DEFAULT_DATASET = SCRIPT_DIR / '../dataset/ISU_ILCC'
DEFAULT_LOG_DIR = SCRIPT_DIR / 'logs'
DEFAULT_PLOT_DIR = SCRIPT_DIR / 'plots' / 'times'


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
        description="Check time monotonicity for BatteryLife dataset."
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
        help='Directory to save plot images (default: plots/times).'
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    log_dir = Path(args.log_dir)
    plot_dir = Path(args.plot_dir)

    logger = setup_logging(log_dir, 'check_times')
    logger.info("Starting time monotonicity check.")
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

    total_anomalies = 0

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

        times_all = []
        bad_points = []  # list of (global_index, time_value) for plotting

        for i in range(length):
            cycle_data = cell_data['cycle_data'][i]
            cycle_df = pd.DataFrame()
            cycle_df['current'] = cycle_data['current_in_A']
            cycle_df['voltage'] = cycle_data['voltage_in_V']
            cycle_df['charge_capacity'] = cycle_data['charge_capacity_in_Ah']
            cycle_df['discharge_capacity'] = cycle_data['discharge_capacity_in_Ah']
            cycle_df['test_time_s'] = cycle_data['time_in_s']
            cycle_df['cycle_number'] = cycle_data['cycle_number']

            time = cycle_df['test_time_s'].tolist()
            time = [round(t, 6) for t in time]
            diff = np.diff(time)

            bad_idx = np.where(diff < 0)[0]

            if len(bad_idx) != 0:
                total_anomalies += len(bad_idx)
                logger.info(
                    "Time of cycle %d in cell %s: %d non-monotonic point(s) found.",
                    i + 1, filename, len(bad_idx)
                )
                for idx in bad_idx:
                    logger.info(
                        "  arr[%d]=%s -> arr[%d]=%s",
                        idx, time[idx], idx + 1, time[idx + 1]
                    )
                    # mark both points for visualization
                    bad_points.append((len(times_all) + idx, time[idx]))
                    bad_points.append((len(times_all) + idx + 1, time[idx + 1]))

            times_all += time

        # Plot time sequence with anomalies highlighted
        fig, ax = plt.subplots(figsize=(12, 5))
        ax.plot(range(len(times_all)), times_all, 'b-', linewidth=0.5, label='Time')

        if bad_points:
            # Deduplicate points that may have been added multiple times
            bad_points_dict = {x: y for x, y in bad_points}
            bad_x = list(bad_points_dict.keys())
            bad_y = list(bad_points_dict.values())
            ax.scatter(bad_x, bad_y, c='red', s=15, zorder=5, label='Non-monotonic points')

        ax.set_xlabel('Global data point index')
        ax.set_ylabel('Time (s)')
        ax.set_title(f'Time sequence monotonicity check\n{filename}')
        ax.legend()
        ax.grid(alpha=0.3)
        fig.tight_layout()

        plot_path = plot_dir / f"times_{filename}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)

        logger.info("Saved time plot for %s -> %s", filename, plot_path.name)

    logger.info(
        "Time monotonicity check completed. Total non-monotonic points: %d",
        total_anomalies
    )


if __name__ == '__main__':
    main()
