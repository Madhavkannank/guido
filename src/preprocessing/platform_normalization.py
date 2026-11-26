"""
Cross-Platform Genomic Normalization & Invariant Feature Extraction
===================================================================
Eliminates technical batch effects across sequencing/microarray platforms
(Illumina HiSeq, NovaSeq, Affymetrix U133, Agilent, NanoString nCounter).

Techniques:
  1. Within-sample percentile rank transformation (scale- and gain-invariant).
  2. Housekeeping control gene ratio normalization (geometric mean subtraction).
  3. Zero-leakage missing gene imputation using training reference medians.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

logger = logging.getLogger(__name__)

DEFAULT_HOUSEKEEPING_GENES = ["GAPDH", "ACTB", "PGK1", "B2M", "HPRT1", "TBP"]


def within_sample_rank_transform(
    data: Union[pd.Series, pd.DataFrame]
) -> Union[pd.Series, pd.DataFrame]:
    """
    Computes within-sample percentile rank transformation.
    For each sample (row or series), values are converted to relative ranks in [0.0, 1.0].
    This transformation is invariant to monotonic platform gain shifts and library size scaling.
    """
    if isinstance(data, pd.Series):
        if len(data) <= 1:
            return pd.Series(0.5, index=data.index)
        # Percentile rank [0, 1]
        ranks = data.rank(method="average", pct=True)
        return ranks

    elif isinstance(data, pd.DataFrame):
        # Rank along axis=1 (across genes for each sample)
        return data.rank(axis=1, method="average", pct=True)
    else:
        raise TypeError(f"Expected pd.Series or pd.DataFrame, got {type(data)}")


def housekeeping_ratio_normalize(
    data: Union[pd.Series, pd.DataFrame],
    housekeeping_genes: Optional[List[str]] = None,
) -> Union[pd.Series, pd.DataFrame]:
    """
    Normalizes gene expression by subtracting the mean of reference housekeeping genes
    in log2 space, equivalent to computing log2(Gene / GeometricMean(HK)).
    """
    hk_genes = housekeeping_genes or DEFAULT_HOUSEKEEPING_GENES

    if isinstance(data, pd.Series):
        available_hk = [g for g in hk_genes if g in data.index and not np.isnan(data[g])]
        if not available_hk:
            logger.warning("No reference housekeeping genes found in sample; returning unnormalized.")
            return data.copy()
        hk_mean = float(data[available_hk].mean())
        return data - hk_mean

    elif isinstance(data, pd.DataFrame):
        available_hk = [g for g in hk_genes if g in data.columns]
        if not available_hk:
            logger.warning("No reference housekeeping genes found in dataframe; returning unnormalized.")
            return data.copy()
        hk_mean = data[available_hk].mean(axis=1)
        return data.sub(hk_mean, axis=0)
    else:
        raise TypeError(f"Expected pd.Series or pd.DataFrame, got {type(data)}")


class CrossPlatformNormalizer(BaseEstimator, TransformerMixin):
    """
    Scikit-learn compatible transformer ensuring cross-platform compatibility
    without test-set leakage.
    """

    def __init__(
        self,
        method: str = "rank",
        housekeeping_genes: Optional[List[str]] = None,
        impute_strategy: str = "median",
    ):
        self.method = method
        self.housekeeping_genes = housekeeping_genes or DEFAULT_HOUSEKEEPING_GENES
        self.impute_strategy = impute_strategy
        self.reference_features_: List[str] = []
        self.reference_medians_: Dict[str, float] = {}

    def fit(self, X: pd.DataFrame, y: Optional[Any] = None) -> "CrossPlatformNormalizer":
        """Learns feature list and reference medians from training split."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        self.reference_features_ = list(X.columns)
        self.reference_medians_ = X.median().to_dict()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Applies normalization and handles missing genes."""
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_out = X.copy()

        # Handle missing genes from training reference
        missing_genes = [g for g in self.reference_features_ if g not in X_out.columns]
        for g in missing_genes:
            med_val = self.reference_medians_.get(g, 0.0)
            X_out[g] = med_val

        # Align column order to training reference
        X_out = X_out[self.reference_features_]

        # Impute any internal NaNs
        X_out = X_out.fillna(pd.Series(self.reference_medians_))

        if self.method == "rank":
            return within_sample_rank_transform(X_out)
        elif self.method == "housekeeping":
            return housekeeping_ratio_normalize(X_out, self.housekeeping_genes)
        elif self.method == "raw":
            return X_out
        else:
            raise ValueError(f"Unknown normalization method: {self.method}")
