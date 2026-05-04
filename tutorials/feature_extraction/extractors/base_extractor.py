"""Base feature extractor class."""

from abc import ABC, abstractmethod
import numpy as np
import pandas as pd

from feature_extraction.core.cc_cv_splitter import CCVSplitter
from feature_extraction.core.stats import compute_stats, compute_slope, compute_entropy
from feature_extraction.utils import compute_soh, compute_rul


class BaseFeatureExtractor(ABC):
    """Abstract base class for PINN4SOH-style 16-feature extraction.

    Each subclass handles one (or a family of) dataset(s) with specific
    pre-filtering and CC/CV splitting logic.
    """

    FEATURE_ORDER = [
        "cycle_number",
        "voltage_mean",
        "voltage_std",
        "voltage_kurtosis",
        "voltage_skewness",
        "CC_Q",
        "CC_charge_time",
        "voltage_slope",
        "voltage_entropy",
        "current_mean",
        "current_std",
        "current_kurtosis",
        "current_skewness",
        "CV_Q",
        "CV_charge_time",
        "current_slope",
        "current_entropy",
        "SOH",
        "RUL",
    ]

    def __init__(self, dataset_name: str, config: dict):
        self.dataset_name = dataset_name
        self.config = config
        self.splitter = CCVSplitter(config)

    def extract_cell(self, cell_data: dict) -> pd.DataFrame:
        """Extract features for all cycles in a cell.

        Parameters
        ----------
        cell_data : dict
            Loaded BatteryLife pkl dict.

        Returns
        -------
        pd.DataFrame
            One row per cycle, columns = FEATURE_ORDER.
        """
        cycles = cell_data["cycle_data"]
        nominal_capacity = cell_data.get("nominal_capacity_in_Ah", 1.0)
        soc_interval = cell_data.get("SOC_interval", [0.0, 1.0])
        total_cycles = len(cycles)

        rows = []
        for cycle in cycles:
            features = self.extract_cycle_features(
                cycle, nominal_capacity, soc_interval, total_cycles
            )
            rows.append(features)

        df = pd.DataFrame(rows, columns=self.FEATURE_ORDER)
        return df

    def extract_cycle_features(
        self, cycle: dict, nominal_capacity: float, soc_interval: list, total_cycles: int
    ) -> dict:
        """Extract 16 features + SOH + RUL for a single cycle.

        Parameters
        ----------
        cycle : dict
            Single cycle dict from BatteryLife.
        nominal_capacity : float
        soc_interval : list
        total_cycles : int

        Returns
        -------
        dict
            Keys aligned with FEATURE_ORDER.
        """
        voltage = np.asarray(cycle.get("voltage_in_V", []))
        current = np.asarray(cycle.get("current_in_A", []))
        time = np.asarray(cycle.get("time_in_s", []))
        charge_capacity = np.asarray(cycle.get("charge_capacity_in_Ah", []))
        cycle_number = int(cycle.get("cycle_number", 0))

        # Apply dynamic cc_window min if configured
        cc_window_cfg = self.config.get("cc_window", {})
        if cc_window_cfg and cc_window_cfg.get("dynamic_min") is not None:
            voltage = np.asarray(voltage, dtype=float)
            # Compute dynamic min based on first charge point voltage
            charge_mask = current > 0
            if np.any(charge_mask):
                v_start = float(voltage[np.where(charge_mask)[0][0]])
                dynamic_expr = cc_window_cfg["dynamic_min"]
                if dynamic_expr == "max(3.6, V_start + 0.1)":
                    dynamic_min = max(3.6, v_start + 0.1)
                elif dynamic_expr == "V_start + 0.1":
                    dynamic_min = v_start + 0.1
                else:
                    dynamic_min = v_start + 0.1
                # Temporarily override voltage_min for this cycle
                cc_window_cfg = dict(cc_window_cfg)
                cc_window_cfg["voltage_min"] = dynamic_min
                # Create a temporary splitter with modified config
                tmp_config = dict(self.config)
                tmp_config["cc_window"] = cc_window_cfg
                splitter = CCVSplitter(tmp_config)
            else:
                splitter = self.splitter
        else:
            splitter = self.splitter

        cc_mask, cv_mask = splitter.split(voltage, current, time, charge_capacity)

        # Extract CC features
        cc_features = self._extract_cc_features(
            voltage, current, time, charge_capacity, cc_mask
        )

        # Extract CV features
        cv_features = self._extract_cv_features(
            voltage, current, time, charge_capacity, cv_mask
        )

        # SOH and RUL
        soh = compute_soh(cycle, nominal_capacity, soc_interval)
        rul = compute_rul(cycle_number, total_cycles)

        result = {
            "cycle_number": cycle_number,
            **cc_features,
            **cv_features,
            "SOH": soh,
            "RUL": rul,
        }
        return result

    def _extract_cc_features(self, voltage, current, time, charge_capacity, cc_mask):
        """Extract 8 CC-phase features."""
        if not np.any(cc_mask):
            return {
                "voltage_mean": 0.0,
                "voltage_std": 0.0,
                "voltage_kurtosis": 0.0,
                "voltage_skewness": 0.0,
                "CC_Q": 0.0,
                "CC_charge_time": 0.0,
                "voltage_slope": 0.0,
                "voltage_entropy": 0.0,
            }

        v_cc = voltage[cc_mask]
        t_cc = time[cc_mask]
        q_cc = charge_capacity[cc_mask]

        stats = compute_stats(v_cc)
        cc_q = float(q_cc[-1] - q_cc[0]) if len(q_cc) > 1 else 0.0
        cc_time = float(t_cc[-1] - t_cc[0]) if len(t_cc) > 1 else 0.0
        v_slope = compute_slope(t_cc, v_cc)
        v_entropy = compute_entropy(v_cc)

        return {
            "voltage_mean": stats["mean"],
            "voltage_std": stats["std"],
            "voltage_kurtosis": stats["kurtosis"],
            "voltage_skewness": stats["skewness"],
            "CC_Q": cc_q,
            "CC_charge_time": cc_time,
            "voltage_slope": v_slope,
            "voltage_entropy": v_entropy,
        }

    def _extract_cv_features(self, voltage, current, time, charge_capacity, cv_mask):
        """Extract 8 CV-phase features."""
        if not np.any(cv_mask):
            return {
                "current_mean": 0.0,
                "current_std": 0.0,
                "current_kurtosis": 0.0,
                "current_skewness": 0.0,
                "CV_Q": 0.0,
                "CV_charge_time": 0.0,
                "current_slope": 0.0,
                "current_entropy": 0.0,
            }

        i_cv = current[cv_mask]
        t_cv = time[cv_mask]
        q_cv = charge_capacity[cv_mask]

        stats = compute_stats(i_cv)
        cv_q = float(q_cv[-1] - q_cv[0]) if len(q_cv) > 1 else 0.0
        cv_time = float(t_cv[-1] - t_cv[0]) if len(t_cv) > 1 else 0.0
        i_slope = compute_slope(t_cv, i_cv)
        i_entropy = compute_entropy(i_cv)

        return {
            "current_mean": stats["mean"],
            "current_std": stats["std"],
            "current_kurtosis": stats["kurtosis"],
            "current_skewness": stats["skewness"],
            "CV_Q": cv_q,
            "CV_charge_time": cv_time,
            "current_slope": i_slope,
            "current_entropy": i_entropy,
        }
