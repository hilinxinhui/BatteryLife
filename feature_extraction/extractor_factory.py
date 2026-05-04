"""Factory mapping dataset names to extractor classes."""

from feature_extraction.extractors.generic_extractor import GenericExtractor
from feature_extraction.extractors.special_extractors import (
    MATRExtractor,
    HUSTExtractor,
    TongjiExtractor,
    PureCCExtractor,
)

# Map dataset name → extractor class
DATASET_EXTRACTOR_MAP = {
    "XJTU": GenericExtractor,
    "MATR": MATRExtractor,
    "HUST": HUSTExtractor,
    "Tongji": TongjiExtractor,
    "NA-ion": PureCCExtractor,
    "ZN-coin": PureCCExtractor,
    # All others default to GenericExtractor
}


def get_extractor(dataset_name: str, config: dict):
    """Get the appropriate extractor for a dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset (must match keys in dataset_intervals.json).
    config : dict
        Config entry for this dataset.

    Returns
    -------
    BaseFeatureExtractor instance
    """
    extractor_cls = DATASET_EXTRACTOR_MAP.get(dataset_name, GenericExtractor)
    return extractor_cls(dataset_name, config)
