"""Specialized feature extractors for datasets with non-standard charging."""

from feature_extraction.extractors.base_extractor import BaseFeatureExtractor


class MATRExtractor(BaseFeatureExtractor):
    """Extractor for MATR (MIT) dataset.

    Applies SOC > 80% pre-filter via charge_capacity_ratio >= 0.79.
    The base splitter already handles this via the config pre_filter.
    """

    def __init__(self, dataset_name: str, config: dict):
        super().__init__(dataset_name, config)


class HUSTExtractor(BaseFeatureExtractor):
    """Extractor for HUST dataset.

    Applies current <= 1.5A pre-filter to exclude 5C fast-charge stage.
    The base splitter already handles this via the config pre_filter.
    """

    def __init__(self, dataset_name: str, config: dict):
        super().__init__(dataset_name, config)


class TongjiExtractor(BaseFeatureExtractor):
    """Extractor for Tongji (TJU) dataset.

    BatteryLife data lacks control/mA and control/V fields, so we fall back
    to voltage_threshold splitting. The base splitter handles this fallback.
    """

    def __init__(self, dataset_name: str, config: dict):
        super().__init__(dataset_name, config)


class PureCCExtractor(BaseFeatureExtractor):
    """Extractor for pure CC datasets without CV phase.

    Works for: NA-ion, ZN-coin.
    has_cv=false in config ensures all CV features are zero.
    """

    def __init__(self, dataset_name: str, config: dict):
        super().__init__(dataset_name, config)
