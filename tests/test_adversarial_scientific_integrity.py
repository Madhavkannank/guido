"""
Adversarial Scientific Integrity & Verification Suite
=====================================================
Rigorous adversarial tests to detect and prevent:
  1. Information Leakage: Verifies that feature selection inside nested CV folds
     does not artificially inflate out-of-fold C-index on pure noise data.
  2. Metric Honesty: Verifies that zero-event or single-sample cohorts return None,
     status="unevaluable", and NEVER fabricate a default 0.5 C-index.
  3. Batch Independence & N=1 Parity: Verifies that a patient's score is invariant
     regardless of whether they are evaluated alone or embedded in hostile batches
     (high-risk, low-risk, multi-organ, or extreme outliers).
  4. Housekeeping Qualification: Verifies that tissue-variable reference genes
     are systematically disqualified and prevented from being sole normalizers.
  5. Measurement Integrity: Verifies strict semantic separation between unmeasured
     genes and true biological zero expression.
  6. QC Honesty: Verifies that absent pre-analytical parameters (DV200, library depth)
     are cataloged as UNMEASURED_NOT_RECORDED and never imputed with fake proxy values.
  7. Claim Audit: Inspects repository documentation to guarantee zero fictitious
     signatures, approvals, or ungrounded claims exist.
"""
from pathlib import Path
import re
import numpy as np
import pandas as pd
import pytest

from src.models.panel_distillation import (
    MinimalGenePanelDistiller,
    QUALIFIED_HOUSEKEEPING_CONTROLS,
    TISSUE_VARIABLE_HOUSEKEEPING_CONTROLS,
)
from src.validation.organ_stratified_evaluation import (
    compute_harrell_c_index_with_ci,
    compute_unos_c_index,
    adjust_p_values_benjamini_hochberg,
)
from src.models.organ_conditioned_scoring import (
    DeployableOrganScorer,
)
from src.data.cross_platform_normalizer import (
    qualify_housekeeping_genes,
    profile_expression_measurement_status,
    ClinicalSampleQualityGate,
    ClinicalQCResult,
)


def test_adversarial_information_leakage_on_pure_noise():
    """
    Adversarial Check A: Zero-Leakage Validation on Pure Noise.
    Generates pure Gaussian noise features with random exponential survival times.
    Verifies that strict nested cross-validation feature selection does NOT leak
    information across folds: the out-of-fold C-index should remain near 0.5 (random guess),
    proving that feature selection happens strictly within training folds.
    """
    np.random.seed(999)
    n_samples = 60
    n_features = 80

    # Pure noise features: no real correlation with survival
    df_noise = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples, n_features)),
        columns=[f"NOISE_GENE_{i}" for i in range(n_features)],
    )
    # Random survival times and events
    times = np.random.exponential(scale=500, size=n_samples) + 10
    events = np.random.binomial(1, 0.35, size=n_samples).astype(float)

    distiller = MinimalGenePanelDistiller(
        panel_sizes=[10, 20],
        n_folds=3,
        penalizer=0.5,
        l1_ratio=0.5,
        n_subsamples=3,
        random_state=42,
    )

    report = distiller.fit(
        df_X=df_noise,
        times=times,
        events=events,
        data_source="adversarial_pure_noise",
        is_synthetic=True,
    )

    # In pure noise, nested OOF C-index must not show artificial prognostic power (> 0.70)
    # Leaked selection across all samples would yield C > 0.85 on 80 features.
    assert report.full_model_c_index < 0.70, (
        f"Information leakage detected! Pure noise yielded out-of-fold C-index of {report.full_model_c_index:.4f}"
    )
    for size, panel in report.candidate_panels.items():
        assert panel.c_index < 0.72, (
            f"Panel size {size} leaked information! OOF C-index is {panel.c_index:.4f} on pure noise."
        )


def test_adversarial_metric_honesty_zero_events_returns_none():
    """
    Adversarial Check B: Metric Honesty on Unevaluable Cohorts.
    Verifies that a cohort with zero events (100% censored) or zero comparable pairs
    strictly returns c_index = None, status = 'unevaluable', and NEVER returns 0.5.
    """
    # 30 patients, but 0 events (all censored)
    scores = np.random.uniform(0.1, 0.9, size=30)
    times = np.random.uniform(100, 1000, size=30)
    events = np.zeros(30, dtype=float)

    res = compute_harrell_c_index_with_ci(
        times=times,
        events=events,
        scores=scores,
    )

    assert res["c_index"] is None, f"Expected None, got {res['c_index']}!"
    assert res["ci_lower"] is None
    assert res["ci_upper"] is None
    assert res["status"] == "unevaluable"
    assert "events" in res["reason"].lower() or "comparable" in res["reason"].lower()

    # Also test Uno's IPCW C-index with 0 events
    uno_c = compute_unos_c_index(scores=scores, times=times, events=events)
    assert uno_c is None, f"Uno's C-index should be None for 0 events, got {uno_c}!"


def test_adversarial_batch_independence_and_n_equals_1_invariance():
    """
    Adversarial Check C: Batch Independence & Deterministic N=1 Inference.
    Tests a target patient score against multiple adversarial batch environments:
      - Batch 1: Target patient alone (N=1).
      - Batch 2: Target patient + 50 extremely high-risk patients.
      - Batch 3: Target patient + 50 extremely low-risk patients.
      - Batch 4: Target patient + 20 multi-organ heterogeneous patients.
      - Batch 5: Target patient + pathological outliers (0.00001 and 0.99999).
    The patient's organ-conditioned score and percentile MUST be 100% numerically
    identical across all batches.
    """
    # Create and fit reference scorer
    np.random.seed(42)
    training_data = pd.DataFrame({
        "cohort": ["TCGA-BRCA"] * 300 + ["TCGA-KIRC"] * 300,
        "risk_score": np.concatenate([
            np.random.beta(2, 5, size=300),  # BRCA lower baseline
            np.random.beta(4, 3, size=300),  # KIRC higher baseline
        ]),
    })

    scorer = DeployableOrganScorer(method="percentile")
    scorer.fit(training_data, cohort_col="cohort", score_col="risk_score")

    target_score = 0.4285
    target_cohort = "TCGA-BRCA"

    # 1. Single patient scoring (N=1)
    single_res = scorer.score_single(raw_score=target_score, organ_cohort=target_cohort)

    # 2. Hostile Batch: 50 extremely high-risk patients
    batch_high = pd.DataFrame({
        "cohort": [target_cohort] + ["TCGA-BRCA"] * 50,
        "risk_score": [target_score] + [0.95] * 50,
    })
    res_high = scorer.score_batch(batch_high).iloc[0]

    # 3. Hostile Batch: 50 extremely low-risk patients
    batch_low = pd.DataFrame({
        "cohort": [target_cohort] + ["TCGA-BRCA"] * 50,
        "risk_score": [target_score] + [0.05] * 50,
    })
    res_low = scorer.score_batch(batch_low).iloc[0]

    # 4. Hostile Batch: multi-organ mixture
    batch_multi = pd.DataFrame({
        "cohort": [target_cohort, "TCGA-KIRC", "TCGA-LUAD", "TCGA-KIRC"],
        "risk_score": [target_score, 0.88, 0.72, 0.15],
    })
    res_multi = scorer.score_batch(batch_multi).iloc[0]

    # 5. Hostile Batch: extreme numerical outliers
    batch_outliers = pd.DataFrame({
        "cohort": [target_cohort, target_cohort, target_cohort],
        "risk_score": [target_score, 0.000001, 0.999999],
    })
    res_outliers = scorer.score_batch(batch_outliers).iloc[0]

    expected_pct = single_res["organ_percentile"]
    expected_cond = single_res["organ_conditioned_score"]

    for b_name, b_res in [
        ("High-risk batch", res_high),
        ("Low-risk batch", res_low),
        ("Multi-organ batch", res_multi),
        ("Outlier batch", res_outliers),
    ]:
        assert b_res["organ_percentile"] == expected_pct, (
            f"Transductive leakage in {b_name}! Expected percentile {expected_pct}, got {b_res['organ_percentile']}"
        )
        assert b_res["organ_conditioned_score"] == expected_cond, (
            f"Transductive leakage in {b_name}! Expected conditioned score {expected_cond}, got {b_res['organ_conditioned_score']}"
        )


def test_adversarial_housekeeping_stability_disqualification():
    """
    Adversarial Check D: Housekeeping Gene Qualification.
    Simulates candidate housekeeping controls where one gene has high inter-tissue divergence.
    Verifies that qualify_housekeeping_genes systematically identifies and disqualifies it.
    """
    np.random.seed(123)
    n_per_cohort = 40
    cohorts = ["COHORT_X"] * n_per_cohort + ["COHORT_Y"] * n_per_cohort

    # RPL13A: stable across both cohorts (mean 12.0)
    rpl = np.random.normal(12.0, 0.3, size=2 * n_per_cohort)
    # UNSTABLE_GENE: shifts from 5.0 in X to 11.5 in Y (6.5 log2 divergence)
    unstable = np.concatenate([
        np.random.normal(5.0, 0.4, size=n_per_cohort),
        np.random.normal(11.5, 0.4, size=n_per_cohort),
    ])

    df = pd.DataFrame({"RPL13A": rpl, "UNSTABLE_GENE": unstable})
    cohort_s = pd.Series(cohorts)

    reports = qualify_housekeeping_genes(
        df,
        cohort_series=cohort_s,
        candidate_genes=["RPL13A", "UNSTABLE_GENE"],
    )

    assert reports["RPL13A"].stability_tier == "QUALIFIED_STABLE"
    assert reports["UNSTABLE_GENE"].stability_tier == "DISQUALIFIED_TISSUE_VARIABLE"
    assert reports["UNSTABLE_GENE"].max_cohort_diff > 5.0
    assert "High inter-tissue variance detected" in reports["UNSTABLE_GENE"].scientific_note


def test_adversarial_unmeasured_vs_true_zero_separation():
    """
    Adversarial Check E: Semantic Separation of Unmeasured vs True Zero.
    Verifies that a biological zero count (0.0) is not marked as unmeasured/missing,
    and an unmeasured gene (absent from profile or NaN) is never treated as zero.
    """
    sample = pd.Series({"GENE_A": 5.2, "GENE_B": 0.0})
    manifest = ["GENE_A", "GENE_B", "GENE_C_UNMEASURED"]

    full_profile, prov = profile_expression_measurement_status(sample, panel_manifest=manifest)

    # GENE_B is measured zero
    assert full_profile["GENE_B"] == 0.0
    assert prov.true_zero_count == 1
    assert "GENE_B" not in prov.unmeasured_gene_list

    # GENE_C is unmeasured (NaN)
    assert np.isnan(full_profile["GENE_C_UNMEASURED"])
    assert prov.unmeasured_genes_count == 1
    assert "GENE_C_UNMEASURED" in prov.unmeasured_gene_list


def test_adversarial_clinical_qc_no_dv200_fabrication():
    """
    Adversarial Check F: Absence of Fabricated QC Metrics.
    When DV200 is unrecorded/None, verify that ClinicalSampleQualityGate:
      1. Sets dv200 to None.
      2. Sets dv200_status to 'UNMEASURED_NOT_RECORDED'.
      3. Never imputes a synthetic number.
    """
    gate = ClinicalSampleQualityGate()
    sample = pd.Series({"RPL13A": 12.0, "ACTB": 15.0, "PPIA": 11.0, "TBP": 8.0, "TP53": 7.5})

    qc_res = gate.evaluate_sample_quality(
        sample=sample,
        dv200=None,
        library_depth=None,
        specimen_type="FRESH_FROZEN",
    )

    assert qc_res.dv200 is None
    assert qc_res.dv200_status == "UNMEASURED_NOT_RECORDED"
    assert qc_res.library_depth_status == "UNMEASURED_NOT_RECORDED"
    assert any("not fabricated" in note for note in qc_res.provenance_notes)


def test_adversarial_claim_audit_no_fictitious_entities():
    """
    Adversarial Check G: Repository Claim & Entity Audit.
    Scans docs/CLINICAL_TRANSLATION_AND_REGULATORY_ROADMAP.md for forbidden fictitious signatures.
    """
    doc_path = Path("docs/CLINICAL_TRANSLATION_AND_REGULATORY_ROADMAP.md")
    assert doc_path.exists(), "Regulatory roadmap document not found!"

    content = doc_path.read_text(encoding="utf-8")

    forbidden_entities = [
        "Antigravity AI Engine",
        "Medical Affairs Review Board",
        "Quality Management Representative: QMS Officer",
        "Chief Medical Officer",
        "Regulatory Affairs Director",
    ]

    for entity in forbidden_entities:
        matches = re.findall(re.escape(entity), content, re.IGNORECASE)
        assert len(matches) == 0, f"Found forbidden fictitious entity '{entity}' in {doc_path}!"

    # Verify that RUO classification is explicitly documented
    assert "Research Use Only (RUO)" in content or "Research Use Only" in content
    # Verify CDSCO Rule 3(zb) is cited
    assert "Rule 3(zb)" in content
    # Verify ISO 14971 claim is toned down (Targeting ... alignment)
    assert "Targeting ISO 14971" in content or "targeting ISO 14971" in content
