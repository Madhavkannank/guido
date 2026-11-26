"""
Deployable Organ-Conditioned Clinical Risk Scoring Engine
=========================================================
Implements deterministic, batch-independent organ-conditioned risk scoring using
frozen reference distributions computed strictly from training cohorts.

Key Design Principles:
  1. Zero Transductive Leakage: Patients are evaluated strictly against frozen training
     quantiles, not against other test samples in the current batch.
  2. N=1 Inference Parity: Single-patient scoring (batch_size=1) produces mathematically
     identical scores to batch evaluation (batch_size=N).
  3. Robust Fallback Mechanics: Explicit handling and telemetry for unknown or rare organ types,
     with transparent fallback to global pan-cancer reference distributions.
  4. Serializable Artifacts: Reference distributions exportable to JSON/joblib for production serving.
"""
from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class OrganReferenceDistribution:
    """Stores frozen summary statistics and quantiles for a single organ cohort."""
    organ_cohort: str
    n_training_samples: int
    median: float
    iqr: float
    min_score: float
    max_score: float
    quantiles: Dict[str, float]  # e.g. "0.01", "0.05", "0.10", ..., "0.95", "0.99"
    mean: float
    std: float

    def score_to_percentile(self, score: float) -> float:
        """
        Maps a continuous risk score to an empirical percentile (0.0 to 1.0)
        using monotonic piecewise-linear interpolation over frozen training quantiles.
        """
        pct_keys = sorted(float(k) for k in self.quantiles.keys())
        q_vals = [self.quantiles[f"{p:.2f}" if f"{p:.2f}" in self.quantiles else str(p)] for p in pct_keys]

        # Monotonic linear interpolation
        pct = float(np.interp(score, q_vals, pct_keys, left=0.001, right=0.999))
        return round(pct, 4)

    def score_to_zscore(self, score: float) -> float:
        """Standardizes a continuous risk score using median and IQR."""
        denominator = max(self.iqr, 1e-5)
        return float((score - self.median) / denominator)


@dataclass
class DeployableOrganScorer:
    """
    Production-ready organ-conditioned risk scorer.
    Computes and freezes reference distributions per organ during model training,
    enabling deterministic single-patient (N=1) scoring in clinical deployment.
    """
    method: str = "percentile"  # 'percentile' or 'median_center'
    organ_references: Dict[str, OrganReferenceDistribution] = field(default_factory=dict)
    global_reference: Optional[OrganReferenceDistribution] = None
    is_fitted: bool = False

    def fit(
        self,
        df: pd.DataFrame,
        cohort_col: str = "cohort",
        score_col: str = "risk_score",
    ) -> DeployableOrganScorer:
        """
        Fits and freezes reference distributions across all cohorts in the training data,
        as well as a global pan-cancer reference for unknown organ fallback.
        """
        clean_df = df[[cohort_col, score_col]].dropna().copy()
        clean_df[score_col] = pd.to_numeric(clean_df[score_col], errors="coerce")
        clean_df = clean_df.dropna()

        if len(clean_df) == 0:
            raise ValueError("Training DataFrame contains zero valid risk score rows.")

        self.organ_references = {}

        # 1. Fit individual organ references
        for cohort_name, group in clean_df.groupby(cohort_col):
            scores = group[score_col].values.astype(float)
            if len(scores) < 3:
                logger.warning("Cohort '%s' has fewer than 3 samples; skipping reference fitting.", cohort_name)
                continue
            ref = self._compute_distribution(str(cohort_name), scores)
            self.organ_references[str(cohort_name)] = ref

        # 2. Fit global pan-cancer reference distribution (for fallback)
        all_scores = clean_df[score_col].values.astype(float)
        self.global_reference = self._compute_distribution("GLOBAL_PAN_CANCER", all_scores)

        self.is_fitted = True
        return self

    @staticmethod
    def _compute_distribution(name: str, scores: np.ndarray) -> OrganReferenceDistribution:
        """Computes summary statistics and quantiles from an array of scores."""
        percentile_points = [
            0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40,
            0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90,
            0.95, 0.98, 0.99
        ]
        q_dict = {
            f"{p:.2f}": float(np.percentile(scores, p * 100.0))
            for p in percentile_points
        }
        med = float(np.median(scores))
        q25 = float(np.percentile(scores, 25.0))
        q75 = float(np.percentile(scores, 75.0))
        iqr = max(q75 - q25, 1e-5)

        return OrganReferenceDistribution(
            organ_cohort=name,
            n_training_samples=len(scores),
            median=round(med, 4),
            iqr=round(iqr, 4),
            min_score=round(float(np.min(scores)), 4),
            max_score=round(float(np.max(scores)), 4),
            quantiles=q_dict,
            mean=round(float(np.mean(scores)), 4),
            std=round(float(np.std(scores)), 4),
        )

    def score_single(
        self,
        raw_score: float,
        organ_cohort: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Scores a single patient against their organ's frozen reference distribution.

        Parameters:
          raw_score: Model's raw uncalibrated or logistic risk score.
          organ_cohort: Patient's cancer type / organ identifier (e.g. 'TCGA-BRCA' or 'BRCA').

        Returns:
          Dictionary with raw_score, organ_conditioned_score, organ_percentile, is_fallback, fallback_reason.
        """
        if not self.is_fitted:
            raise ValueError("DeployableOrganScorer must be fitted on training data before scoring.")

        raw_score = float(raw_score)
        norm_organ = str(organ_cohort).strip().upper() if organ_cohort is not None else "UNKNOWN"
        # Try both with and without "TCGA-" prefix
        matched_ref: Optional[OrganReferenceDistribution] = None
        for candidate in [norm_organ, f"TCGA-{norm_organ}", norm_organ.replace("TCGA-", "")]:
            if candidate in self.organ_references:
                matched_ref = self.organ_references[candidate]
                break

        is_fallback = False
        fallback_reason = None

        if matched_ref is None:
            is_fallback = True
            if self.global_reference is not None:
                matched_ref = self.global_reference
                fallback_reason = (
                    f"Unknown organ '{organ_cohort}'. Evaluated against frozen Pan-Cancer Global reference."
                )
            else:
                # Mark as unavailable rather than fabricating in-sample reference or leaking
                return {
                    "raw_score": raw_score,
                    "organ_cohort": organ_cohort,
                    "organ_conditioned_score": None,
                    "organ_percentile": None,
                    "is_fallback": True,
                    "status": "unavailable_no_independent_reference",
                    "fallback_reason": f"Unknown organ '{organ_cohort}' and no independent reference distribution available.",
                    "reference_used": None,
                    "reference_n_samples": 0,
                    "clinical_interpretation_notice": (
                        "Percentile normalization provides within-organ relative ranking, NOT calibrated absolute clinical risk. "
                        "When no independent reference distribution exists, organ conditioning is unavailable."
                    ),
                }

        # Compute conditioned score based on configured method
        if self.method == "median_center":
            cond_score = matched_ref.score_to_zscore(raw_score)
            pct = matched_ref.score_to_percentile(raw_score)
        else:
            pct = matched_ref.score_to_percentile(raw_score)
            cond_score = pct

        return {
            "raw_score": round(raw_score, 4),
            "organ_cohort": organ_cohort,
            "organ_conditioned_score": round(cond_score, 4),
            "organ_percentile": round(pct * 100.0, 2),  # 0.0 to 100.0%
            "is_fallback": is_fallback,
            "fallback_reason": fallback_reason,
            "reference_used": matched_ref.organ_cohort,
            "reference_n_samples": matched_ref.n_training_samples,
            "clinical_interpretation_notice": (
                "Percentile normalization provides within-organ relative ranking, NOT calibrated absolute clinical risk. "
                "Patients are evaluated strictly against frozen training reference quantiles; individual evaluations do "
                "not alter cohort statistics."
            ),
        }

    def score_batch(
        self,
        df: pd.DataFrame,
        cohort_col: str = "cohort",
        score_col: str = "risk_score",
    ) -> pd.DataFrame:
        """
        Scores an evaluation DataFrame row-by-row deterministically.
        Guarantees that each row's score is independent of other rows in the batch.
        """
        results = []
        for _, row in df.iterrows():
            s = float(row[score_col])
            c = row[cohort_col] if cohort_col in row else None
            res = self.score_single(raw_score=s, organ_cohort=c)
            results.append(res)

        res_df = pd.DataFrame(results)
        return res_df

    def to_dict(self) -> Dict[str, Any]:
        """Serializes scorer state to dictionary."""
        return {
            "method": self.method,
            "is_fitted": self.is_fitted,
            "organ_references": {
                k: asdict(v) for k, v in self.organ_references.items()
            },
            "global_reference": asdict(self.global_reference) if self.global_reference else None,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> DeployableOrganScorer:
        """Restores scorer state from serialized dictionary."""
        scorer = cls(method=data.get("method", "percentile"))
        refs = {}
        for k, v in data.get("organ_references", {}).items():
            refs[k] = OrganReferenceDistribution(**v)
        scorer.organ_references = refs
        if data.get("global_reference"):
            scorer.global_reference = OrganReferenceDistribution(**data["global_reference"])
        scorer.is_fitted = data.get("is_fitted", False)
        return scorer

    def save_json(self, path: Union[str, Path]) -> None:
        """Saves frozen reference distributions to a JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load_json(cls, path: Union[str, Path]) -> DeployableOrganScorer:
        """Loads frozen reference distributions from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls.from_dict(data)
