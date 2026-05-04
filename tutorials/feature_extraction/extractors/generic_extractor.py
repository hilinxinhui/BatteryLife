"""Generic feature extractor for standard CC-CV datasets."""

from feature_extraction.extractors.base_extractor import BaseFeatureExtractor


class GenericExtractor(BaseFeatureExtractor):
    """Default extractor for datasets with standard CC-CV charging.

    Works for: XJTU, CALCE, CALB, HNEI, MICH, MICH_EXP, SDU, UL_PUR,
    Stanford, Stanford_2, SNL, RWTH, ISU_ILCC.
    """

    def __init__(self, dataset_name: str, config: dict):
        super().__init__(dataset_name, config)
