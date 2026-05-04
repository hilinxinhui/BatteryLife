#!/usr/bin/env python3
"""
Batch runner for BatteryLife check scripts.

Runs specified check scripts across ALL sub-datasets under the dataset root.
Each dataset is processed sequentially, with plots organized into subdirectories
by check type and dataset name.

Usage:
    python3 batch_runner.py                          # Run all checks on all datasets
    python3 batch_runner.py --checks combined        # Run only combined curves
    python3 batch_runner.py --checks temperature soc # Run temperature + SOC checks
"""

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.resolve()
DEFAULT_DATASET_ROOT = SCRIPT_DIR / '../dataset'

# (script_name, has_plot_output, plot_subdir_name)
CHECKS = {
    'capacity': ('check_capacity_curves.py', True, 'capacity'),
    'soh': ('check_soh_curves.py', True, 'soh'),
    'combined': ('check_combined_curves.py', True, 'combined'),
    'temperature': ('check_temperature_curves.py', True, 'temperature'),
    'times': ('check_times.py', True, 'times'),
    'voltage_current': ('check_voltage_current_curves.py', True, None),  # special: voltage/ + current/
    'operating': ('check_operating_conditions.py', False, None),
    'soc': ('check_soc_intervals.py', False, None),
}


def get_dataset_dirs(dataset_root: Path):
    """Find all subdirectories containing at least one .pkl file."""
    dirs = []
    skip_names = {'Life labels', 'READMEs', 'seen_unseen_labels', 'README.md'}
    if dataset_root.exists():
        for item in sorted(dataset_root.iterdir()):
            if not item.is_dir():
                continue
            if item.name in skip_names:
                continue
            if list(item.glob('*.pkl')):
                dirs.append(item)
    return dirs


def main():
    parser = argparse.ArgumentParser(
        description='Batch run BatteryLife data check scripts across all datasets.'
    )
    parser.add_argument(
        '--checks', nargs='+',
        choices=list(CHECKS.keys()) + ['all'],
        default=['all'],
        help='Which checks to run (default: all)'
    )
    parser.add_argument(
        '--log_dir', type=str,
        default=str(SCRIPT_DIR / 'logs'),
        help='Directory to save log files'
    )
    parser.add_argument(
        '--plot_dir', type=str,
        default=str(SCRIPT_DIR / 'plots'),
        help='Root directory to save plot images'
    )
    parser.add_argument(
        '--dataset_root', type=str,
        default=str(DEFAULT_DATASET_ROOT),
        help='Root directory containing all dataset subdirectories'
    )
    args = parser.parse_args()

    dataset_root = Path(args.dataset_root)
    log_dir = Path(args.log_dir)
    plot_dir = Path(args.plot_dir)

    checks = list(CHECKS.keys()) if 'all' in args.checks else args.checks
    dataset_dirs = get_dataset_dirs(dataset_root)

    if not dataset_dirs:
        print(f"No datasets found under {dataset_root}")
        sys.exit(1)

    print(f"Found {len(dataset_dirs)} dataset(s):")
    for d in dataset_dirs:
        n = len(list(d.glob('*.pkl')))
        print(f"  - {d.name} ({n} cells)")
    print(f"Checks to run: {checks}")

    for check_key in checks:
        script_name, has_plot, plot_subdir = CHECKS[check_key]
        script_path = SCRIPT_DIR / script_name

        if not script_path.exists():
            print(f"\n[SKIP] Script not found: {script_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Running: {script_name}")
        print(f"{'='*60}")

        for ds in dataset_dirs:
            cmd = [
                sys.executable, str(script_path),
                '--dataset', str(ds),
                '--log_dir', str(log_dir),
            ]

            if has_plot:
                if plot_subdir:
                    # e.g. plots/combined/CALB/
                    target_plot_dir = plot_dir / plot_subdir / ds.name
                else:
                    # voltage_current: plots/voltage_current/CALB/
                    # (script internally creates voltage/ and current/)
                    target_plot_dir = plot_dir / 'voltage_current' / ds.name
                cmd.extend(['--plot_dir', str(target_plot_dir)])

            print(f"  -> {ds.name} ...", end='', flush=True)
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode == 0:
                print(" OK")
            else:
                print(f" FAILED (code {result.returncode})")
                if result.stderr:
                    # Print last few lines of stderr
                    err_lines = result.stderr.strip().split('\n')[-3:]
                    for line in err_lines:
                        print(f"     {line}")

    print(f"\n{'='*60}")
    print("Batch run completed.")
    print(f"Logs:   {log_dir}")
    print(f"Plots:  {plot_dir}")


if __name__ == '__main__':
    main()
