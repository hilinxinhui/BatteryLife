"""Utility functions for loading battery data and computing SOH/RUL."""

import os
import pickle
import numpy as np


def load_battery_data(pkl_path: str) -> dict:
    """Load a BatteryLife .pkl file and return the raw dict.

    The returned dict contains keys such as:
    - cell_id
    - cycle_data (list of cycle dicts)
    - nominal_capacity_in_Ah
    - SOC_interval (list of [min, max])
    - ... (other metadata)
    """
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data


def compute_soh(cycle: dict, nominal_capacity: float, soc_interval: list) -> float:
    """Compute SOH for a single cycle.

    SOH = max(discharge_capacity during current < 0) / nominal_capacity / SOC_interval_width

    Parameters
    ----------
    cycle : dict
        A single cycle dict with keys 'current_in_A', 'discharge_capacity_in_Ah'.
    nominal_capacity : float
        The cell's nominal capacity in Ah.
    soc_interval : list
        [SOC_min, SOC_max], e.g. [0.0, 1.0].

    Returns
    -------
    float
        SOH value. Returns 0.0 if no discharge segment found.
    """
    current = np.asarray(cycle["current_in_A"])
    discharge_cap = np.asarray(cycle["discharge_capacity_in_Ah"])

    # Find discharge segment (current < 0)
    discharge_mask = current < 0
    if not np.any(discharge_mask):
        return 0.0

    max_discharge_cap = float(np.max(discharge_cap[discharge_mask]))
    soc_width = soc_interval[1] - soc_interval[0]
    if nominal_capacity <= 0 or soc_width <= 0:
        return 0.0

    soh = max_discharge_cap / nominal_capacity / soc_width
    return soh


def compute_rul(cycle_number: int, total_cycles: int) -> int:
    """Compute RUL for a single cycle.

    RUL = total_cycles - cycle_number

    Parameters
    ----------
    cycle_number : int
        The current cycle number (1-indexed).
    total_cycles : int
        Total number of cycles in the cell's life.

    Returns
    -------
    int
        Remaining useful life in cycle count.
    """
    return total_cycles - cycle_number


def get_cycle_start_voltage(cycle: dict) -> float:
    """Get the starting voltage of a cycle (first point)."""
    voltage = cycle.get("voltage_in_V", [])
    if len(voltage) > 0:
        return float(voltage[0])
    return 0.0


def list_dataset_cells(dataset_dir: str, dataset_name: str) -> list:
    """List all .pkl files for a given dataset.

    Returns list of absolute file paths.
    """
    ds_path = os.path.join(dataset_dir, dataset_name)
    if not os.path.isdir(ds_path):
        return []
    files = sorted([
        os.path.join(ds_path, f)
        for f in os.listdir(ds_path)
        if f.endswith(".pkl")
    ])
    return files
