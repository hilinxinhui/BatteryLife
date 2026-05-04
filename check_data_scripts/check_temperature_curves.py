"""
Check temperature curves for BatteryLife dataset cells.

This script loads all .pkl battery files from a given dataset directory,
aggregates temperature data across all cycles for each cell,
and plots temperature vs time.

Basic temperature statistics (min, max, mean) are also logged.
No anomaly criteria are applied at this stage.

Plots are saved under plots/temperature/ by default.

Original inspection scope: temperature profiles
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
DEFAULT_PLOT_DIR = SCRIPT_DIR / 'plots' / 'temperature'


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
        description="Check temperature curves for BatteryLife dataset."
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
        help='Directory to save plot images (default: plots/temperature).'
    )
    args = parser.parse_args()

    dataset_path = Path(args.dataset)
    log_dir = Path(args.log_dir)
    plot_dir = Path(args.plot_dir)

    logger = setup_logging(log_dir, 'check_temperature_curves')
    logger.info("Starting temperature curves check.")
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

        temperatures = []
        times = []

        for i in range(length):
            cycle_data = cell_data['cycle_data'][i]
            cycle_df = pd.DataFrame()
            cycle_df['temperature'] = cycle_data['temperature_in_C']
            cycle_df['test_time'] = cycle_data['time_in_s']

            temp = cycle_df['temperature'].values.tolist()
            time = cycle_df['test_time'].values.tolist()

            temperatures += temp
            times += time

        if not temperatures:
            logger.warning("No temperature data for %s, skipping plot.", filename)
            continue

        # Plot temperature curve
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(times, temperatures, color='darkorange', linewidth=0.8, label='Temperature')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Temperature vs Time\n{filename}')
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()

        plot_path = plot_dir / f"temperature_curves_{filename}.png"
        plt.savefig(plot_path, dpi=300)
        plt.close(fig)

        # Log basic statistics
        logger.info(
            "Cell %s: temperature min=%.2f°C, max=%.2f°C, mean=%.2f°C, std=%.2f°C",
            filename,
            np.min(temperatures),
            np.max(temperatures),
            np.mean(temperatures),
            np.std(temperatures)
        )

    logger.info(
        "Temperature curves check completed. %d plots saved to %s",
        len(files), plot_dir
    )


if __name__ == '__main__':
    main()
