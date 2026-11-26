"""
Clinical Sample Quality Control (QC) & FFPE Artifact Gating
===========================================================
Enforces pre-analytical sample viability standards for FFPE biopsy specimens,
core needle biopsies, and surgical resections prior to clinical risk inference.

Thresholds:
  - DV200 (Percentage of RNA fragments > 200 nucleotides):
      * PASS: >= 50% (High/Medium quality RNA)
      * WARN: 30% <= DV200 < 50% (Degraded FFPE, flag risk caution)
      * FAIL: < 30% (Severely degraded; explicit rejection)
  - Zero/Dropout Rate: Max 40% zero expression.
  - Reference Housekeeping Detectability: Minimum 3 housekeeping controls detected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import logging
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_HOUSEKEEPING = ["GAPDH", "ACTB", "PGK1", "B2M", "HPRT1", "TBP"]


class QCStatus(str, Enum):
    PASS = "PASS"
    WARN = "WARN"
    FAIL = "FAIL"


@dataclass
class QCResult:
    """Structured pre-analytical quality control decision."""
    status: QCStatus
    quality_score: float  # 0.0 to 100.0
    dv200: Optional[float]
    zero_fraction: float
    warnings: List[str] = field(default_factory=list)
    rejection_reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def is_actionable(self) -> bool:
        """Returns True if sample is safe for clinical risk computation."""
        return self.status in (QCStatus.PASS, QCStatus.WARN)


class ClinicalSampleQualityGate:
    """
    Evaluates incoming patient genomic profiles against pre-analytical QC standards.
    """

    def __init__(
        self,
        min_dv200: float = 50.0,
        warn_dv200: float = 30.0,
        max_zero_fraction: float = 0.40,
        min_housekeeping_count: int = 3,
        min_library_depth: float = 5_000_000,
    ):
        self.min_dv200 = min_dv200
        self.warn_dv200 = warn_dv200
        self.max_zero_fraction = max_zero_fraction
        self.min_housekeeping_count = min_housekeeping_count
        self.min_library_depth = min_library_depth

    def evaluate_sample_quality(
        self,
        sample: Union[pd.Series, Dict[str, float]],
        dv200: Optional[float] = None,
        library_depth: Optional[float] = None,
    ) -> QCResult:
        """
        Executes pre-analytical quality checks on a single patient sample.
        """
        if isinstance(sample, dict):
            sample = pd.Series(sample)

        warnings: List[str] = []
        rejection_reasons: List[str] = []

        # 1. Zero / Dropout Fraction
        vals = sample.values.astype(float)
        n_features = len(vals)
        if n_features == 0:
            return QCResult(
                status=QCStatus.FAIL,
                quality_score=0.0,
                dv200=dv200,
                zero_fraction=1.0,
                rejection_reasons=["Empty sample profile with zero gene measurements."],
            )

        zero_count = int(np.sum(vals == 0.0) + np.sum(np.isnan(vals)))
        zero_frac = float(zero_count / n_features)

        if zero_frac > self.max_zero_fraction:
            rejection_reasons.append(
                f"Excessive dropout rate ({zero_frac:.1%} of genes unexpressed or missing; threshold is {self.max_zero_fraction:.1%})."
            )

        # 2. Housekeeping Control Detectability
        hk_detected = [g for g in DEFAULT_HOUSEKEEPING if g in sample.index and sample[g] > 0.0]
        # For full transcriptomic profiles (> 20 genes), enforce minimum housekeeping controls
        if n_features >= 20 and len(hk_detected) < self.min_housekeeping_count:
            rejection_reasons.append(
                f"Insufficient housekeeping control genes detected ({len(hk_detected)} / {self.min_housekeeping_count} required: {hk_detected})."
            )
        elif len(hk_detected) < self.min_housekeeping_count:
            warnings.append(
                f"Targeted minimal panel payload: {len(hk_detected)} standard housekeeping controls present."
            )

        # 3. DV200 RNA Quality Evaluation
        if dv200 is not None:
            if dv200 < self.warn_dv200:
                rejection_reasons.append(
                    f"Severe FFPE RNA fragmentation (DV200 score {dv200:.1f}% is below minimum rejection threshold of {self.warn_dv200:.1f}%)."
                )
            elif dv200 < self.min_dv200:
                warnings.append(
                    f"Moderate FFPE RNA degradation (DV200 score {dv200:.1f}% between {self.warn_dv200:.1f}% and {self.min_dv200:.1f}%). Interpret risk score with caution."
                )

        # 4. Sequencing Library Depth
        if library_depth is not None and library_depth < self.min_library_depth:
            warnings.append(
                f"Sub-optimal library depth ({library_depth:,.0f} reads < {self.min_library_depth:,.0f} recommended)."
            )

        # Compute Quality Score (0 - 100)
        base_score = 100.0
        # Deduct for dropout
        base_score -= min(40.0, (zero_frac / self.max_zero_fraction) * 40.0)
        # Deduct for DV200
        if dv200 is not None:
            if dv200 < self.warn_dv200:
                base_score -= 50.0
            elif dv200 < self.min_dv200:
                base_score -= 20.0
        # Deduct for housekeeping
        if len(hk_detected) < len(DEFAULT_HOUSEKEEPING):
            base_score -= (len(DEFAULT_HOUSEKEEPING) - len(hk_detected)) * 5.0

        quality_score = float(np.clip(base_score, 0.0, 100.0))

        # Determine Final Status
        if len(rejection_reasons) > 0:
            status = QCStatus.FAIL
        elif len(warnings) > 0:
            status = QCStatus.WARN
        else:
            status = QCStatus.PASS

        return QCResult(
            status=status,
            quality_score=round(quality_score, 1),
            dv200=dv200,
            zero_fraction=round(zero_frac, 4),
            warnings=warnings,
            rejection_reasons=rejection_reasons,
            metrics={
                "genes_total": n_features,
                "zero_genes": zero_count,
                "housekeeping_detected": len(hk_detected),
                "library_depth": library_depth,
            },
        )


def simulate_ffpe_degradation(
    sample: pd.Series,
    dv200: float = 35.0,
    seed: int = 42,
) -> pd.Series:
    """
    Simulates the biochemical effects of formalin-fixation paraffin-embedding (FFPE)
    and RNA fragmentation on gene expression profiles:
      - Signal attenuation proportional to degradation (lower DV200)
      - Non-linear loss of low-abundance transcripts
      - Technical noise injection
    """
    rng = np.random.RandomState(seed)
    degraded = sample.copy().astype(float)

    # Fragmentation factor: DV200=100 -> factor=1.0; DV200=20 -> factor=0.5
    deg_factor = float(np.clip(dv200 / 100.0, 0.2, 1.0))

    # Attenuate expression values
    attenuation = deg_factor + rng.normal(0, 0.05, size=len(degraded))
    degraded = degraded * np.clip(attenuation, 0.1, 1.2)

    # Dropout effect for low-abundance transcripts
    dropout_prob = (1.0 - deg_factor) * 0.4
    dropout_mask = rng.uniform(0, 1, size=len(degraded)) < dropout_prob
    degraded[dropout_mask] = 0.0

    return degraded
