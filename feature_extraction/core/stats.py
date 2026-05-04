"""Statistical feature computation helpers for CC/CV phase features."""

import numpy as np
from scipy import stats as scipy_stats


def compute_stats(values: np.ndarray) -> dict:
    """Compute mean, std, kurtosis, skewness for a 1D array.

    Returns
    -------
    dict with keys: mean, std, kurtosis, skewness
    """
    if len(values) == 0:
        return {"mean": 0.0, "std": 0.0, "kurtosis": 0.0, "skewness": 0.0}

    mean_val = float(np.mean(values))
    std_val = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

    # Guard against constant arrays (kurtosis/skew return NaN)
    if std_val == 0.0 or np.allclose(values, values[0]):
        kurt_val = 0.0
        skew_val = 0.0
    else:
        # scipy uses Fisher's definition (normal => 0.0)
        kurt_val = float(scipy_stats.kurtosis(values, fisher=True, bias=False))
        skew_val = float(scipy_stats.skew(values, bias=False))
        # Replace any remaining NaN/Inf with 0.0
        if not np.isfinite(kurt_val):
            kurt_val = 0.0
        if not np.isfinite(skew_val):
            skew_val = 0.0

    return {
        "mean": mean_val,
        "std": std_val,
        "kurtosis": kurt_val,
        "skewness": skew_val,
    }


def compute_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Linear regression slope of y vs x using least squares.

    Returns 0.0 if insufficient points or zero variance in x.
    """
    if len(x) < 2 or len(y) < 2:
        return 0.0
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if np.allclose(x, x[0]):
        return 0.0
    # np.polyfit(x, y, 1) returns [slope, intercept]
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def compute_entropy(values: np.ndarray, bins: int = 50) -> float:
    """Shannon entropy of the empirical distribution of values.

    Uses histogram binning (equal-width) and normalizes by max possible entropy
    so that a uniform distribution returns 1.0.

    Returns 0.0 for empty or constant arrays.
    """
    values = np.asarray(values, dtype=float)
    if len(values) == 0:
        return 0.0
    if np.allclose(values, values[0]):
        return 0.0

    # Use min-max range for binning
    hist, _ = np.histogram(values, bins=bins, range=(values.min(), values.max()))
    probs = hist / hist.sum()
    probs = probs[probs > 0]  # remove empty bins
    if len(probs) == 0:
        return 0.0

    entropy = -np.sum(probs * np.log2(probs))
    # normalize by log2(bins) for comparability
    max_entropy = np.log2(bins)
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    return float(normalized_entropy)
