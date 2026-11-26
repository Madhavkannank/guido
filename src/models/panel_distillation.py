"""
Minimal Gene Panel Distillation Engine (Penalized Cox & Stability Selection)
=============================================================================
Distills high-dimensional transcriptomic models (20,000+ genes) down to minimal,
clinically assayable candidate panels (15, 25, 35, 50 genes) using genuine
survival analysis (ElasticNet/L1-penalized Cox Proportional Hazards regression).

Scientific Safeguards & Methodology:
  1. True Survival Likelihood: Preserves continuous survival times and right-censoring
     status. Does NOT binarize survival times or substitute a binary classifier.
  2. Zero-Leakage Nested CV: Preprocessing (scaling), feature screening, stability selection,
     and model fitting are executed strictly within outer training folds.
  3. Out-of-Fold Evaluation: Retention curves and candidate panel performance metrics
     are computed exclusively on out-of-fold held-out validation predictions.
  4. Stability Selection (Meinshausen & Bühlmann): Resampling within training partitions
     measures empirical feature selection probability and coefficient direction.
  5. Provenance Discipline: Explicitly records whether input data is synthetic or real TCGA,
     preserves endpoint definitions (PFI/OS), and flags in-silico validation status.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.validation.organ_stratified_evaluation import (
    compute_harrell_c_index_with_ci,
    compute_unos_c_index,
)

logger = logging.getLogger(__name__)

# Canonical Hallmark Gene Sets (Functional Biological Annotation)
HALLMARK_ANNOTATIONS: Dict[str, List[str]] = {
    "Sustained Proliferation": ["MKI67", "CCND1", "CDK4", "MYC", "EGFR", "ERBB2", "E2F1", "AURKA", "PLK1"],
    "Evading Growth Suppressors": ["TP53", "RB1", "CDKN2A", "PTEN", "SMAD4", "BRCA1", "BRCA2", "ATM"],
    "Resisting Cell Death": ["BCL2", "BCL2L1", "BAX", "MCL1", "CASP3", "FAS", "BAD", "BIRC5"],
    "Enabling Replicative Immortality": ["TERT", "TERC", "DKC1", "TINF2", "POT1", "ACD"],
    "Inducing Angiogenesis": ["VEGFA", "VEGFB", "KDR", "FLT1", "ANGPT1", "ANGPT2", "HIF1A"],
    "Activating Invasion & Metastasis": ["CDH1", "CDH2", "VIM", "SNAI1", "SNAI2", "TWIST1", "ZEB1", "MMP2", "MMP9"],
    "Reprogramming Energy Metabolism": ["SLC2A1", "HK2", "LDHA", "PKM", "IDH1", "GLS", "FASN"],
    "Evading Immune Destruction": ["CD274", "PDCD1", "CTLA4", "LAG3", "HAVCR2", "B2M", "HLA-A", "CXCL9"],
    "Tumor-Promoting Inflammation": ["IL6", "TNF", "CXCL8", "STAT3", "NFKB1", "PTGS2", "CSF1"],
    "Genome Instability & Mutation": ["MLH1", "MSH2", "MSH6", "PMS2", "PARP1", "RAD51", "ERCC1"],
}

# Standard Housekeeping Candidate Genes (Reference Controls)
# Empirically qualified across TCGA cohorts in src/data/cross_platform_normalizer.py
QUALIFIED_HOUSEKEEPING_CONTROLS: List[str] = ["RPL13A", "ACTB", "PPIA", "TBP"]
TISSUE_VARIABLE_HOUSEKEEPING_CONTROLS: List[str] = ["HPRT1", "B2M", "PGK1"]
HOUSEKEEPING_CONTROLS: List[str] = QUALIFIED_HOUSEKEEPING_CONTROLS


@dataclass
class CandidatePanel:
    """Represents a distilled candidate clinical gene panel."""
    panel_size: int
    genes: List[str]
    weights: Dict[str, float]
    c_index: float  # Genuine out-of-fold Harrell's C-index
    c_index_ci: List[float]  # Out-of-fold 95% bootstrap confidence interval
    unos_c_index: float  # Out-of-fold Uno's IPCW C-index
    retention_ratio: float  # Out-of-fold retention relative to full model
    hallmark_representation: Dict[str, int]
    biotypes: Dict[str, int]
    assay_type_recommendation: str
    validation_status: str = "in_silico_candidate"
    evidence_level: str = "in_silico_unvalidated"
    sample_count: int = 0
    event_count: int = 0


@dataclass
class DistillationReport:
    """Comprehensive report detailing the minimal gene panel distillation process."""
    data_source: str
    is_synthetic: bool
    endpoint: str
    evaluation_method: str
    n_samples: int
    n_events: int
    censoring_rate: float
    full_model_gene_count: int
    full_model_c_index: float  # Out-of-fold
    full_model_c_index_ci: List[float]
    full_model_unos_c_index: float
    candidate_panels: Dict[int, CandidatePanel]
    retention_curve: List[Dict[str, Any]]
    stability_rankings: List[Dict[str, Any]]
    limitations: List[str] = field(default_factory=list)


def _compute_univariate_survival_concordance(
    X: np.ndarray,
    times: np.ndarray,
    events: np.ndarray,
) -> np.ndarray:
    """
    Computes univariate survival prognostic strength |C - 0.5| for each feature.
    Used for pre-screening high-dimensional spaces strictly within training folds.
    """
    n, p = X.shape
    scores = np.zeros(p, dtype=float)
    event_idx = np.where(events == 1)[0]
    if len(event_idx) == 0:
        return scores

    for j in range(p):
        xj = X[:, j]
        concordant = 0.0
        pairs = 0
        for i in event_idx:
            ti = times[i]
            xi = xj[i]
            valid_k = times > ti
            if not np.any(valid_k):
                continue
            xk = xj[valid_k]
            pairs += len(xk)
            concordant += float(np.sum(xi > xk)) + 0.5 * float(np.sum(xi == xk))
        c_val = concordant / pairs if pairs > 0 else 0.5
        scores[j] = abs(c_val - 0.5)
    return scores


class MinimalGenePanelDistiller:
    """
    Distills transcriptomic risk models into minimal panels using regularized
    Cox Proportional Hazards regression and cross-validated stability selection.
    """

    def __init__(
        self,
        panel_sizes: Optional[List[int]] = None,
        n_folds: int = 4,
        penalizer: float = 0.1,
        l1_ratio: float = 0.7,
        n_subsamples: int = 3,
        max_screen_features: int = 60,
        random_state: int = 42,
    ):
        self.panel_sizes = panel_sizes or [15, 25, 35, 50]
        self.n_folds = n_folds
        self.penalizer = penalizer
        self.l1_ratio = l1_ratio
        self.n_subsamples = n_subsamples
        self.max_screen_features = max(max_screen_features, max(self.panel_sizes))
        self.random_state = random_state

        self.feature_stability_scores_: Dict[str, float] = {}
        self.feature_weights_: Dict[str, float] = {}
        self.distillation_report_: Optional[DistillationReport] = None

    def fit(
        self,
        df_X: pd.DataFrame,
        times: np.ndarray,
        events: np.ndarray,
        data_source: str = "synthetic",
        is_synthetic: bool = True,
        endpoint: str = "PFI",
    ) -> DistillationReport:
        """
        Runs stability selection across CV folds to rank genes and extract candidate panels.
        All validation metrics are computed OUT-OF-FOLD.
        """
        feature_names = list(df_X.columns)
        n_features = len(feature_names)
        n_samples = len(df_X)

        X = df_X.values.astype(float)
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=int)

        # Scientific Validation: Check survival constraints
        valid_mask = (times > 0) & (~np.isnan(times)) & (~np.isnan(events))
        X = X[valid_mask]
        times = times[valid_mask]
        events = events[valid_mask]
        n_samples = len(times)
        n_events = int(np.sum(events == 1))

        if n_samples < 20 or n_events < 5:
            raise ValueError(
                f"Insufficient survival data for Cox distillation: N={n_samples}, events={n_events}. "
                "At least 20 samples and 5 events are required for survival modeling."
            )

        censoring_rate = round(float(1.0 - (n_events / n_samples)), 4)
        logger.info(
            "Fitting MinimalGenePanelDistiller (N=%d, Events=%d, Censoring=%.1f%%, P=%d features)",
            n_samples, n_events, censoring_rate * 100, n_features
        )

        # ─── Step 1: Out-of-Fold Nested Cross-Validation ─────────────────────
        kf = KFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        oof_full_scores = np.zeros(n_samples, dtype=float)

        curve_sizes = sorted(list(set(self.panel_sizes + [5, 10, min(50, n_features)])))
        oof_panel_scores: Dict[int, np.ndarray] = {
            s: np.zeros(n_samples, dtype=float) for s in curve_sizes if s <= n_features
        }

        selection_counts = np.zeros(n_features, dtype=float)
        summed_coefs = np.zeros(n_features, dtype=float)

        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_tr, X_val = X[train_idx], X[val_idx]
            t_tr, t_val = times[train_idx], times[val_idx]
            e_tr, e_val = events[train_idx], events[val_idx]

            if np.sum(e_tr == 1) < 3:
                logger.warning("Fold %d has < 3 events in training set; skipping.", fold_idx)
                continue

            # Scale strictly on training split
            scaler = StandardScaler()
            X_tr_sc = scaler.fit_transform(X_tr)
            X_val_sc = scaler.transform(X_val)

            # Feature screening if p > max_screen_features
            if n_features > self.max_screen_features:
                univ_scores = _compute_univariate_survival_concordance(X_tr_sc, t_tr, e_tr)
                screened_indices = np.argsort(-univ_scores)[:self.max_screen_features]
            else:
                screened_indices = np.arange(n_features)

            # Within-fold stability selection on screened features
            n_screened = len(screened_indices)
            fold_counts = np.zeros(n_screened, dtype=float)
            fold_coefs = np.zeros(n_screened, dtype=float)

            rng = np.random.RandomState(self.random_state + fold_idx * 17)
            n_tr = len(X_tr)

            for b in range(self.n_subsamples):
                sub_idx = rng.choice(n_tr, size=int(0.85 * n_tr), replace=False)
                if np.sum(e_tr[sub_idx] == 1) < 2:
                    continue

                df_sub = pd.DataFrame(
                    X_tr_sc[sub_idx][:, screened_indices],
                    columns=[f"F_{k}" for k in range(n_screened)]
                )
                df_sub["time"] = t_tr[sub_idx]
                df_sub["event"] = e_tr[sub_idx]

                try:
                    cph_sub = CoxPHFitter(penalizer=self.penalizer, l1_ratio=self.l1_ratio)
                    cph_sub.fit(df_sub, duration_col="time", event_col="event")
                    b_coefs = cph_sub.params_.values
                    active = np.abs(b_coefs) > 1e-4
                    fold_counts += active.astype(float)
                    fold_coefs += b_coefs
                except Exception:
                    # In case of numerical issues with small subsamples
                    continue

            # Record selection counts for global stability
            effective_subs = max(1.0, float(self.n_subsamples))
            for local_k, orig_k in enumerate(screened_indices):
                selection_counts[orig_k] += fold_counts[local_k] / effective_subs
                summed_coefs[orig_k] += fold_coefs[local_k] / effective_subs

            # Fold ranking of features
            local_scores = fold_counts / effective_subs + 0.3 * np.abs(fold_coefs / effective_subs)
            local_ranked = screened_indices[np.argsort(-local_scores)]

            # 1. Full Model out-of-fold prediction
            try:
                # Use ridge penalty on screened features for baseline stability
                df_tr_full = pd.DataFrame(
                    X_tr_sc[:, screened_indices],
                    columns=[f"F_{k}" for k in range(n_screened)]
                )
                df_tr_full["time"] = t_tr
                df_tr_full["event"] = e_tr
                cph_full = CoxPHFitter(penalizer=0.05, l1_ratio=0.1)
                cph_full.fit(df_tr_full, duration_col="time", event_col="event")
                full_b = cph_full.params_.values
                oof_full_scores[val_idx] = np.dot(X_val_sc[:, screened_indices], full_b)
            except Exception as e:
                logger.warning("Full model fold %d fit exception: %s", fold_idx, e)

            # 2. Candidate Panel out-of-fold predictions
            for size in oof_panel_scores.keys():
                top_k = local_ranked[:size]
                try:
                    df_tr_panel = pd.DataFrame(
                        X_tr_sc[:, top_k],
                        columns=[f"P_{k}" for k in range(size)]
                    )
                    df_tr_panel["time"] = t_tr
                    df_tr_panel["event"] = e_tr
                    cph_panel = CoxPHFitter(penalizer=0.02, l1_ratio=0.1)
                    cph_panel.fit(df_tr_panel, duration_col="time", event_col="event")
                    panel_b = cph_panel.params_.values
                    oof_panel_scores[size][val_idx] = np.dot(X_val_sc[:, top_k], panel_b)
                except Exception as e:
                    logger.warning("Panel size %d fold %d fit exception: %s", size, fold_idx, e)

        # ─── Step 2: Compute Genuine Out-of-Fold Concordance ─────────────────
        full_res = compute_harrell_c_index_with_ci(times, events, oof_full_scores, n_bootstrap=100)
        full_c = full_res["c_index"]
        full_ci = [full_res["ci_lower"], full_res["ci_upper"]]
        full_uno = compute_unos_c_index(times, events, oof_full_scores)

        retention_curve: List[Dict[str, Any]] = []
        for size in sorted(oof_panel_scores.keys()):
            sub_res = compute_harrell_c_index_with_ci(times, events, oof_panel_scores[size], n_bootstrap=100)
            sub_c = sub_res["c_index"]
            sub_ci = [sub_res["ci_lower"], sub_res["ci_upper"]]
            sub_uno = compute_unos_c_index(times, events, oof_panel_scores[size])
            retention = round(float(sub_c / max(full_c, 0.01)), 4) if full_c > 0.0 else 0.0

            retention_curve.append({
                "panel_size": size,
                "c_index": sub_c,
                "ci_95": sub_ci,
                "unos_c_index": sub_uno,
                "retention_ratio": retention,
            })

        # ─── Step 3: Global Consensus Stability Ranking on Full Data ─────────
        total_folds = max(1.0, float(self.n_folds))
        stability_prob = selection_counts / total_folds
        mean_coefs = summed_coefs / total_folds

        self.feature_stability_scores_ = {
            feature_names[i]: round(float(stability_prob[i]), 4) for i in range(n_features)
        }
        self.feature_weights_ = {
            feature_names[i]: round(float(mean_coefs[i]), 4) for i in range(n_features)
        }

        composite_score = stability_prob * 0.7 + (np.abs(mean_coefs) / (np.max(np.abs(mean_coefs)) + 1e-8)) * 0.3
        ranked_indices = np.argsort(-composite_score)

        stability_rankings = [
            {
                "rank": rank + 1,
                "gene": feature_names[idx],
                "stability_score": round(float(stability_prob[idx]), 4),
                "mean_coefficient": round(float(mean_coefs[idx]), 4),
            }
            for rank, idx in enumerate(ranked_indices[:min(100, n_features)])
        ]

        # ─── Step 4: Final Candidate Panels Annotation ───────────────────────
        candidate_panels: Dict[int, CandidatePanel] = {}

        # Scale full dataset for consensus panel weight calculation
        sc_full = StandardScaler()
        X_full_sc = sc_full.fit_transform(X)

        for size in self.panel_sizes:
            if size > n_features:
                continue
            top_idx = ranked_indices[:size]
            sub_genes = [feature_names[i] for i in top_idx]

            # Fit consensus weights
            sub_weights: Dict[str, float] = {}
            try:
                df_consensus = pd.DataFrame(
                    X_full_sc[:, top_idx],
                    columns=[f"G_{k}" for k in range(size)]
                )
                df_consensus["time"] = times
                df_consensus["event"] = events
                cph_cons = CoxPHFitter(penalizer=0.02, l1_ratio=0.1)
                cph_cons.fit(df_consensus, duration_col="time", event_col="event")
                sub_weights = {
                    sub_genes[i]: round(float(cph_cons.params_.values[i]), 4)
                    for i in range(len(sub_genes))
                }
            except Exception:
                sub_weights = {
                    sub_genes[i]: round(float(mean_coefs[top_idx[i]]), 4)
                    for i in range(len(sub_genes))
                }

            # Extract corresponding out-of-fold metrics from retention curve
            curve_entry = next((entry for entry in retention_curve if entry["panel_size"] == size), None)
            if curve_entry:
                panel_c = curve_entry["c_index"]
                panel_ci = curve_entry["ci_95"]
                panel_uno = curve_entry["unos_c_index"]
                retention = curve_entry["retention_ratio"]
            else:
                panel_c, panel_ci, panel_uno, retention = 0.5, [0.5, 0.5], 0.5, 0.5

            hallmarks = self._annotate_hallmarks(sub_genes)
            biotypes = self._annotate_biotypes(sub_genes)
            assay_rec = self._recommend_assay(size)

            candidate_panels[size] = CandidatePanel(
                panel_size=size,
                genes=sub_genes,
                weights=sub_weights,
                c_index=panel_c,
                c_index_ci=panel_ci,
                unos_c_index=panel_uno,
                retention_ratio=retention,
                hallmark_representation=hallmarks,
                biotypes=biotypes,
                assay_type_recommendation=assay_rec,
                validation_status="in_silico_candidate",
                evidence_level="in_silico_unvalidated",
                sample_count=n_samples,
                event_count=n_events,
            )

        limitations = [
            "In-silico candidate gene panels require prospective analytical and clinical validation.",
            "Primer and probe efficiency has not been measured experimentally.",
            "Performance retention is evaluated via out-of-fold cross-validation on available cohorts.",
        ]
        if is_synthetic:
            limitations.append("EVALUATED ON SYNTHETIC BENCHMARK DATASET - NOT REAL PATIENT BIOLOGY.")

        report = DistillationReport(
            data_source=data_source,
            is_synthetic=is_synthetic,
            endpoint=endpoint,
            evaluation_method="Nested K-Fold Cross-Validation (Out-of-Fold Predictions)",
            n_samples=n_samples,
            n_events=n_events,
            censoring_rate=censoring_rate,
            full_model_gene_count=n_features,
            full_model_c_index=full_c,
            full_model_c_index_ci=full_ci,
            full_model_unos_c_index=full_uno,
            candidate_panels=candidate_panels,
            retention_curve=retention_curve,
            stability_rankings=stability_rankings,
            limitations=limitations,
        )
        self.distillation_report_ = report
        return report

    def _annotate_hallmarks(self, genes: List[str]) -> Dict[str, int]:
        """Maps panel genes to biological cancer hallmark pathways."""
        hallmark_counts: Dict[str, int] = {}
        gene_set = set(genes)
        for hallmark, hallmark_genes in HALLMARK_ANNOTATIONS.items():
            overlap = len(gene_set.intersection(hallmark_genes))
            if overlap > 0:
                hallmark_counts[hallmark] = overlap
        if not hallmark_counts:
            hallmark_counts["Multi-Pathway Prognosis"] = len(genes)
        return hallmark_counts

    def _annotate_biotypes(self, genes: List[str]) -> Dict[str, int]:
        """Annotates genes with genomic biotypes (Protein-coding, lncRNA, pseudogene)."""
        biotypes = {"protein_coding": 0, "lncRNA": 0, "other": 0}
        for g in genes:
            if g.startswith("LINC") or g.endswith("-AS1") or g.endswith("-IT1"):
                biotypes["lncRNA"] += 1
            elif g.startswith("MIR") or g.startswith("SNORD"):
                biotypes["other"] += 1
            else:
                biotypes["protein_coding"] += 1
        return biotypes

    def _recommend_assay(self, panel_size: int) -> str:
        """Recommends targeted diagnostic assay technology candidate based on gene count."""
        if panel_size <= 20:
            return "RT-qPCR TaqMan Low-Density Array (TLDA) / Multiplex PCR (Candidate Protocol - Unvalidated)"
        elif panel_size <= 40:
            return "NanoString nCounter Elements / Digital Spatial Profiling (Candidate Protocol - Unvalidated)"
        else:
            return "Illumina AmpliSeq Targeted RNA Library Prep (Candidate Protocol - Unvalidated)"

    def export_assay_specifications(
        self,
        panel_size: int,
        output_path: Union[str, Path],
    ) -> Dict[str, Any]:
        """
        Exports in-silico assay design specifications to disk with explicit validation status.
        """
        if not self.distillation_report_ or panel_size not in self.distillation_report_.candidate_panels:
            raise ValueError(f"Panel size {panel_size} not available in fitted distillation report.")

        panel = self.distillation_report_.candidate_panels[panel_size]
        report = self.distillation_report_
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)

        spec = {
            "title": f"GUIDO In-Silico Assay Design Specification — {panel_size}-Gene Candidate Panel",
            "validation_status": "in_silico_candidate",
            "regulatory_status": "Research Use Only (RUO) - Unvalidated Assay Design",
            "is_synthetic_training": report.is_synthetic,
            "training_data_source": report.data_source,
            "survival_endpoint": report.endpoint,
            "panel_size": panel_size,
            "target_genes": panel.genes,
            "gene_weights": panel.weights,
            "housekeeping_candidate_controls": {
                "qualified_stable_controls": QUALIFIED_HOUSEKEEPING_CONTROLS,
                "tissue_variable_controls_to_avoid": TISSUE_VARIABLE_HOUSEKEEPING_CONTROLS,
                "empirical_basis": "Empirically evaluated across real TCGA cohorts in cross_platform_normalizer.py; RPL13A, ACTB, PPIA, and TBP demonstrate cross-tissue CV < 0.05, while HPRT1, B2M, and PGK1 exhibit significant tissue-specific variation (>0.9 log2 shift)."
            },
            "recommended_platform_candidate": panel.assay_type_recommendation,
            "out_of_fold_metrics": {
                "c_index": panel.c_index,
                "c_index_95_ci": panel.c_index_ci,
                "unos_c_index": panel.unos_c_index,
                "retention_ratio": panel.retention_ratio,
                "n_samples": panel.sample_count,
                "n_events": panel.event_count,
            },
            "hallmark_coverage": panel.hallmark_representation,
            "biotype_distribution": panel.biotypes,
            "in_silico_design_guidelines": {
                "amplicon_length_bp": "60-120 bp (recommended for FFPE tissue RNA)",
                "primer_melting_temp_c": "58-62 deg C",
                "normalization_protocol": "Geometric mean of qualified housekeeping controls (RPL13A, ACTB, PPIA, TBP); avoid solitary reliance on HPRT1/B2M.",
            },
            "missing_evidence": [
                "Primer/probe amplification efficiency (must be verified between 90% and 110% with R^2 >= 0.98).",
                "Limit of Detection (LoD) and Limit of Blank (LoB) on titrated FFPE RNA reference materials.",
                "Analytical specificity and cross-hybridization assessment across homologous transcript isoforms.",
                "Linear dynamic range across >= 4 orders of magnitude.",
                "Intra-assay and inter-assay repeatability/reproducibility (CV < 8% across technical replicates).",
                "Turnaround time (24-48h target) and reagent costs (<$200 target) are aspirational estimates and have not been validated with an accredited laboratory partner.",
                "Prospective clinical utility (impact on adjuvant treatment decisions and patient outcome) must be demonstrated in multi-center clinical trials prior to diagnostic use.",
            ],
            "notice": "This document is an in-silico research specification and must not be used as an approved clinical assay protocol."
        }

        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(spec, f, indent=2)

        logger.info("Exported assay design specification to: %s", out_p)
        return spec
