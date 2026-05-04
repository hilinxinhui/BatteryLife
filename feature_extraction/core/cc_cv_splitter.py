"""CC/CV phase detection and splitting logic."""

import numpy as np


class CCVSplitter:
    """Splits a single cycle's charge data into CC and CV phases.

    Supports multiple split strategies:
    - voltage_threshold:  V >= threshold → CV
    - current_threshold:  used for HUST pre-filter (exclude I > threshold)
    - charge_capacity_ratio: used for MATR pre-filter (keep Q >= ratio * max_Q)
    """

    def __init__(self, config: dict):
        """
        Parameters
        ----------
        config : dict
            Dataset config entry from dataset_intervals.json.
        """
        self.config = config
        self.has_cv = config.get("has_cv", True)
        self.cc_cv_split_cfg = config.get("cc_cv_split")
        self.pre_filter_cfg = config.get("pre_filter")
        self.cc_window_cfg = config.get("cc_window")
        self.cv_current_window_cfg = config.get("cv_current_window")

    def split(self, voltage, current, time, charge_capacity) -> tuple:
        """Split charge data into CC and CV masks.

        Parameters
        ----------
        voltage, current, time, charge_capacity : array-like
            Full-cycle time series (may include discharge).

        Returns
        -------
        cc_mask : np.ndarray (bool)
            True for points belonging to CC phase (within cc_window).
        cv_mask : np.ndarray (bool)
            True for points belonging to CV phase (within cv_window).
        """
        voltage = np.asarray(voltage)
        current = np.asarray(current)
        time = np.asarray(time)
        charge_capacity = np.asarray(charge_capacity)

        n = len(voltage)
        if n == 0:
            return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

        # Step 1: isolate charge segment (current > 0)
        charge_mask = current > 0
        if not np.any(charge_mask):
            return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

        charge_idx = np.where(charge_mask)[0]

        # Step 2: apply pre-filter if configured
        valid_mask = self._apply_pre_filter(charge_mask, voltage, current, time, charge_capacity)
        valid_idx = np.where(valid_mask)[0]
        if len(valid_idx) == 0:
            return np.zeros(n, dtype=bool), np.zeros(n, dtype=bool)

        # Step 3: split valid charge points into CC and CV
        if not self.has_cv or self.cc_cv_split_cfg is None:
            # Pure CC dataset
            cc_full_mask = np.zeros(n, dtype=bool)
            cc_full_mask[valid_idx] = True
            return cc_full_mask, np.zeros(n, dtype=bool)

        split_mode = self.cc_cv_split_cfg.get("mode", "voltage_threshold")

        if split_mode == "voltage_threshold":
            threshold = self.cc_cv_split_cfg["threshold"]
            cc_points = valid_idx[voltage[valid_idx] < threshold]
            cv_points = valid_idx[voltage[valid_idx] >= threshold]
        elif split_mode == "current_threshold":
            # For HUST: after pre-filter, split by voltage
            threshold = self.cc_cv_split_cfg.get("threshold", 3.595)
            cc_points = valid_idx[voltage[valid_idx] < threshold]
            cv_points = valid_idx[voltage[valid_idx] >= threshold]
        elif split_mode == "control_mode":
            # BatteryLife does not have control/mA or control/V fields.
            # Fallback to voltage_threshold.
            threshold = self.cc_cv_split_cfg.get("fallback_threshold", 4.199)
            cc_points = valid_idx[voltage[valid_idx] < threshold]
            cv_points = valid_idx[voltage[valid_idx] >= threshold]
        else:
            raise ValueError(f"Unknown CC/CV split mode: {split_mode}")

        # Step 4: apply cc_window and cv_current_window sub-windows
        cc_mask = np.zeros(n, dtype=bool)
        cv_mask = np.zeros(n, dtype=bool)

        if len(cc_points) > 0 and self.cc_window_cfg is not None:
            v_min = self.cc_window_cfg.get("voltage_min")
            v_max = self.cc_window_cfg.get("voltage_max")
            if v_min is not None and v_max is not None:
                win_mask = (voltage[cc_points] >= v_min) & (voltage[cc_points] <= v_max)
                cc_mask[cc_points[win_mask]] = True
            else:
                cc_mask[cc_points] = True
        elif len(cc_points) > 0:
            cc_mask[cc_points] = True

        if len(cv_points) > 0 and self.cv_current_window_cfg is not None:
            c_min = self.cv_current_window_cfg.get("current_min")
            c_max = self.cv_current_window_cfg.get("current_max")
            if c_min is not None and c_max is not None:
                win_mask = (current[cv_points] >= c_min) & (current[cv_points] <= c_max)
                cv_mask[cv_points[win_mask]] = True
            else:
                cv_mask[cv_points] = True
        elif len(cv_points) > 0:
            cv_mask[cv_points] = True

        return cc_mask, cv_mask

    def _apply_pre_filter(self, charge_mask, voltage, current, time, charge_capacity):
        """Apply pre-filtering to charge points. Returns boolean mask over full array."""
        if self.pre_filter_cfg is None:
            return charge_mask.copy()

        mode = self.pre_filter_cfg.get("mode")
        valid = charge_mask.copy()

        if mode == "current_threshold":
            # HUST: exclude points where current > threshold (5C stage)
            current_max = self.pre_filter_cfg["current_max"]
            valid = valid & (current <= current_max)

        elif mode == "charge_capacity_ratio":
            # MATR: keep only points where charge_Q >= ratio * max(charge_Q)
            ratio = self.pre_filter_cfg["ratio_threshold"]
            charge_idx = np.where(charge_mask)[0]
            if len(charge_idx) > 0:
                max_q = np.max(charge_capacity[charge_idx])
                threshold_q = ratio * max_q
                valid = valid & (charge_capacity >= threshold_q)

        return valid
