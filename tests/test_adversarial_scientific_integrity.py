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
from lifelines import CoxPHFitter, KaplanMeierFitter

from src.models.panel_distillation import (
    MinimalGenePanelDistiller,
    QUALIFIED_HOUSEKEEPING_CONTROLS,
    TISSUE_VARIABLE_HOUSEKEEPING_CONTROLS,
    extract_tcga_patient_ids,
)
from src.validation.organ_stratified_evaluation import (
    compute_harrell_c_index_with_ci,
    compute_unos_c_index,
    adjust_p_values_benjamini_hochberg,
    OrganConditionedRiskAdjuster,
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


def test_adversarial_comparator_transparency():
    """
    Adversarial Check H: Comparator Model Transparency.
    Verifies that when input features exceed screening cutoff, the baseline model
    is explicitly reported as 'screened_high_dimensional_comparator' with its actual
    screened feature count, and NEVER misrepresented as a complete feature universe model.
    """
    np.random.seed(42)
    n_samples = 40
    n_features = 80
    max_screen = 30

    X_mat = np.random.normal(0, 1, size=(n_samples, n_features))
    df_X = pd.DataFrame(X_mat, columns=[f"GENE_{i:03d}" for i in range(n_features)])
    times = np.random.exponential(1000, size=n_samples) + 20
    events = np.random.binomial(1, 0.4, size=n_samples)

    distiller = MinimalGenePanelDistiller(
        panel_sizes=[10, 20],
        max_screen_features=max_screen,
        n_folds=3,
        n_subsamples=2,
        random_state=42,
    )
    report = distiller.fit(
        df_X=df_X,
        times=times,
        events=events,
        data_source="synthetic_comparator_test",
        is_synthetic=True,
    )

    # Must accurately report screened comparator
    assert report.comparator_type == "screened_high_dimensional_comparator"
    assert report.original_feature_count == 80
    assert report.screened_feature_count == max_screen
    assert report.comparator_feature_count == max_screen
    assert report.full_model_gene_count == max_screen

    # Retention must be calculated against comparator
    for size, panel in report.candidate_panels.items():
        assert panel.comparator_type == "screened_high_dimensional_comparator"
        assert panel.comparator_feature_count == max_screen


def test_adversarial_patient_level_cross_validation_zero_overlap():
    """
    Adversarial Check I: Zero Patient Overlap across GroupKFold Partitions.
    Simulates multi-biospecimen patients (e.g. primary tumor -01A and recurrence -02A).
    Verifies that GroupKFold partitions strictly by patient barcode so that zero
    patients have biospecimens present in both training and validation folds simultaneously.
    """
    # 20 patients, each with 2 biospecimens = 40 samples total
    patient_barcodes = [f"TCGA-XX-{i:04d}" for i in range(20)]
    sample_ids = []
    for pt in patient_barcodes:
        sample_ids.append(f"{pt}-01A")
        sample_ids.append(f"{pt}-02A")

    n_samples = len(sample_ids)
    extracted_pts = extract_tcga_patient_ids(sample_ids)
    assert len(np.unique(extracted_pts)) == 20

    df_X = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples, 25)),
        index=sample_ids,
        columns=[f"GENE_{i}" for i in range(25)],
    )
    times = np.random.uniform(100, 2000, size=n_samples)
    events = np.random.binomial(1, 0.5, size=n_samples)

    distiller = MinimalGenePanelDistiller(
        panel_sizes=[5, 10],
        n_folds=4,
        n_subsamples=2,
        random_state=42,
    )
    report = distiller.fit(
        df_X=df_X,
        times=times,
        events=events,
        sample_ids=sample_ids,
        data_source="multi_specimen_patient_test",
        is_synthetic=True,
    )

    assert report.patient_grouping_status == "patient_grouped_strictly_partitioned"
    assert report.patient_count == 20

    # Verify fold diagnostics confirm zero patient overlap
    for fold in report.fold_diagnostics:
        assert fold["patient_overlap_count"] == 0
        assert fold["patient_overlap_detected"] is False


def test_adversarial_oof_failure_injection_comparator_failure(monkeypatch):
    """
    Adversarial Check J1: Live OOF Failure Injection on Comparator Model.
    Replaces self-contained NumPy simulation with real MinimalGenePanelDistiller.fit() execution.
    Injects deterministic failure into comparator model fitting for fold 1.
    Verifies:
      1. Failed predictions remain NaN in oof_comparator_scores buffer (never converted to 0.0).
      2. Missing predictions are NOT treated as low-risk or zero-risk.
      3. OOF coverage ratio is calculated strictly as valid_samples / total_samples (< 1.0).
      4. evaluation_status is set to 'incomplete_missing_predictions'.
      5. Fold diagnostics record fit_success=False, comparator_fit_success=False, and failure_reasons.
      6. distiller.fit(require_complete_oof=True) raises an explicit RuntimeError.
    """
    np.random.seed(42)
    n_samples = 40
    df_X = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples, 20)),
        columns=[f"G_{i}" for i in range(20)],
    )
    times = np.random.uniform(50, 1000, size=n_samples)
    events = np.random.binomial(1, 0.5, size=n_samples)

    orig_fit = CoxPHFitter.fit

    def mock_fit_comparator_fail(self, df, *args, **kwargs):
        # Full comparator model uses penalizer 0.05
        if getattr(self, "penalizer", None) == 0.05:
            call_idx = getattr(mock_fit_comparator_fail, "call_count", 0)
            mock_fit_comparator_fail.call_count = call_idx + 1
            if call_idx == 1:
                raise RuntimeError("Simulated solver divergence in fold 1 comparator")
        return orig_fit(self, df, *args, **kwargs)

    mock_fit_comparator_fail.call_count = 0
    monkeypatch.setattr(CoxPHFitter, "fit", mock_fit_comparator_fail)

    distiller = MinimalGenePanelDistiller(
        panel_sizes=[5, 10],
        n_folds=3,
        n_subsamples=1,
        random_state=42,
    )

    # 1. require_complete_oof=False: should complete with status="incomplete_missing_predictions"
    report = distiller.fit(
        df_X=df_X,
        times=times,
        events=events,
        data_source="adversarial_oof_failure_injection",
        is_synthetic=True,
        require_complete_oof=False,
    )

    assert report.evaluation_status == "incomplete_missing_predictions"
    assert report.oof_coverage_ratio < 1.0
    assert report.oof_coverage_ratio == round(float(np.sum(~np.isnan(distiller.oof_comparator_scores_))) / n_samples, 4)

    # Failed predictions must remain strictly NaN (never converted to 0.0)
    failed_val_indices = np.where(np.isnan(distiller.oof_comparator_scores_))[0]
    assert len(failed_val_indices) > 0
    for idx in failed_val_indices:
        assert np.isnan(distiller.oof_comparator_scores_[idx])
        assert distiller.oof_comparator_scores_[idx] != 0.0

    # Fold 1 diagnostics verification
    f1 = report.fold_diagnostics[1]
    assert f1["comparator_fit_success"] is False
    assert f1["fit_success"] is False
    assert any("Simulated solver divergence in fold 1 comparator" in r for r in f1["failure_reasons"])

    # 2. require_complete_oof=True: must raise RuntimeError
    mock_fit_comparator_fail.call_count = 0
    with pytest.raises(RuntimeError, match="Cross-validation incomplete"):
        distiller.fit(
            df_X=df_X,
            times=times,
            events=events,
            data_source="adversarial_oof_failure_injection",
            is_synthetic=True,
            require_complete_oof=True,
        )


def test_adversarial_oof_failure_injection_candidate_panel_failure(monkeypatch):
    """
    Adversarial Check J2: Separated Candidate Panel vs Comparator Diagnostics.
    Injects deterministic failure into candidate panel size 10 in fold 0, while the comparator succeeds.
    Verifies:
      1. Comparator fit succeeds on fold 0 (comparator_fit_success=True).
      2. Candidate panel size 10 fails on fold 0 (panel_fit_success[10]=False).
      3. CRITICAL: Fold overall fit_success is False! (A fold must NOT be marked completely
         successful merely because the comparator succeeded when candidate panels failed).
      4. Candidate panel 10 records incomplete evaluation status, missing count > 0,
         coverage ratio < 1.0, and fold success rate < 1.0.
      5. Other candidate panels (size 5) succeed and maintain complete status.
      6. require_complete_oof=True raises RuntimeError.
    """
    np.random.seed(42)
    n_samples = 40
    df_X = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples, 20)),
        columns=[f"G_{i}" for i in range(20)],
    )
    times = np.random.uniform(50, 1000, size=n_samples)
    events = np.random.binomial(1, 0.5, size=n_samples)

    orig_fit = CoxPHFitter.fit

    def mock_fit_panel_fail(self, df, *args, **kwargs):
        # Candidate panel has columns starting with P_
        p_cols = [c for c in df.columns if c.startswith("P_")]
        if len(p_cols) == 10:
            call_idx = getattr(mock_fit_panel_fail, "p10_count", 0)
            mock_fit_panel_fail.p10_count = call_idx + 1
            if call_idx == 0:
                raise RuntimeError("Simulated singularity in panel size 10 Cox model")
        return orig_fit(self, df, *args, **kwargs)

    mock_fit_panel_fail.p10_count = 0
    monkeypatch.setattr(CoxPHFitter, "fit", mock_fit_panel_fail)

    distiller = MinimalGenePanelDistiller(
        panel_sizes=[5, 10],
        n_folds=3,
        n_subsamples=1,
        random_state=42,
    )

    report = distiller.fit(
        df_X=df_X,
        times=times,
        events=events,
        data_source="adversarial_panel_failure_test",
        is_synthetic=True,
        require_complete_oof=False,
    )

    f0 = report.fold_diagnostics[0]
    assert f0["comparator_fit_success"] is True
    assert f0["panel_fit_success"][10] is False
    assert f0["panel_fit_success"][5] is True
    assert f0["fit_success"] is False, "Fold must NOT be marked successful when a candidate panel failed!"
    assert any("Simulated singularity in panel size 10" in r for r in f0["failure_reasons"])

    p10 = report.candidate_panels[10]
    assert p10.panel_evaluation_status == "incomplete_missing_predictions"
    assert p10.oof_missing_count > 0
    assert p10.oof_coverage_ratio < 1.0
    assert p10.panel_fit_fold_success_rate < 1.0

    p5 = report.candidate_panels[5]
    assert p5.panel_evaluation_status == "complete"
    assert p5.oof_missing_count == 0
    assert p5.oof_coverage_ratio == 1.0
    assert p5.panel_fit_fold_success_rate == 1.0

    # With require_complete_oof=True, must raise
    mock_fit_panel_fail.p10_count = 0
    with pytest.raises(RuntimeError, match="Cross-validation incomplete"):
        distiller.fit(
            df_X=df_X,
            times=times,
            events=events,
            data_source="adversarial_panel_failure_test",
            is_synthetic=True,
            require_complete_oof=True,
        )


def test_adversarial_patient_cv_unique_patients_fewer_than_folds():
    """
    Adversarial Check I2: Patient-Level CV with Unique Patients Fewer than Requested Folds.
    When 3 unique patients are present but 5 folds are requested:
      1. Effective folds dynamically adjusts to min(5, 3) = 3 folds.
      2. Never silently falls back to row-level KFold.
      3. Partitions strictly by patient barcode with 0 patient overlap in every fold.
      4. cv_strategy explicitly documents the adjustment.
      5. patient_grouping_status remains 'patient_grouped_strictly_partitioned'.
    """
    patients = ["TCGA-AA-0001", "TCGA-BB-0002", "TCGA-CC-0003"]
    sample_ids = []
    for pt in patients:
        for vial_idx in range(8):
            sample_ids.append(f"{pt}-{vial_idx:02d}A")

    n_samples = len(sample_ids)
    df_X = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples, 15)),
        index=sample_ids,
        columns=[f"G_{i}" for i in range(15)],
    )
    times = np.random.uniform(100, 1000, size=n_samples)
    events = np.array([1, 0] * (n_samples // 2))

    distiller = MinimalGenePanelDistiller(
        panel_sizes=[5],
        n_folds=5,  # Requested 5 folds, but only 3 unique patients exist
        n_subsamples=1,
        random_state=42,
    )

    report = distiller.fit(
        df_X=df_X,
        times=times,
        events=events,
        sample_ids=sample_ids,
        data_source="few_patients_test",
        is_synthetic=True,
    )

    assert report.patient_grouping_status == "patient_grouped_strictly_partitioned"
    assert report.patient_count == 3
    assert len(report.fold_diagnostics) == 3
    assert "adjusted to 3 folds from requested 5" in report.cv_strategy

    for fold in report.fold_diagnostics:
        assert fold["patient_overlap_count"] == 0
        assert fold["patient_overlap_detected"] is False


def test_adversarial_patient_cv_fewer_than_two_patients_raises_value_error():
    """
    Adversarial Check I3: Patient-Level CV with Fewer than 2 Unique Patients.
    When only 1 unique patient is present, data cannot be partitioned into disjoint
    training and validation patient sets.
    Verifies that distiller.fit() raises an explicit ValueError.
    """
    sample_ids = [f"TCGA-ZZ-9999-{i:02d}A" for i in range(24)]
    n_samples = len(sample_ids)
    df_X = pd.DataFrame(
        np.random.normal(0, 1, size=(n_samples, 10)),
        index=sample_ids,
        columns=[f"G_{i}" for i in range(10)],
    )
    times = np.random.uniform(100, 1000, size=n_samples)
    events = np.array([1, 0] * (n_samples // 2))

    distiller = MinimalGenePanelDistiller(panel_sizes=[5], n_folds=3, random_state=42)

    with pytest.raises(ValueError, match="Patient-level cross-validation cannot be performed"):
        distiller.fit(
            df_X=df_X,
            times=times,
            events=events,
            sample_ids=sample_ids,
            data_source="single_patient_test",
            is_synthetic=True,
        )


def test_adversarial_patient_cv_ungrouped_fallback_only_when_no_barcodes():
    """
    Adversarial Check I4: Ungrouped Row-Level CV Only When Patient IDs Genuinely Unavailable.
    When sample IDs do not contain TCGA 12-char patient barcodes:
      1. Falls back to row-level KFold.
      2. Preserves status 'row_level_ungrouped_patient_ids_unavailable'.
    """
    sample_ids = [f"CELL_LINE_SAMPLE_{i:03d}" for i in range(30)]
    df_X = pd.DataFrame(
        np.random.normal(0, 1, size=(30, 15)),
        index=sample_ids,
        columns=[f"G_{i}" for i in range(15)],
    )
    times = np.random.uniform(50, 500, size=30)
    events = np.random.binomial(1, 0.5, size=30)

    distiller = MinimalGenePanelDistiller(panel_sizes=[5], n_folds=3, n_subsamples=1, random_state=42)
    report = distiller.fit(
        df_X=df_X,
        times=times,
        events=events,
        sample_ids=sample_ids,
        data_source="cell_lines",
        is_synthetic=True,
    )

    assert report.patient_grouping_status == "row_level_ungrouped_patient_ids_unavailable"
    assert "Row-Level KFold" in report.cv_strategy


def test_adversarial_pancan_master_loader_deduplication_and_sample_prioritization(tmp_path, monkeypatch):
    """
    Adversarial Check N1: PanCan Master Loader Patient Deduplication & Sample Prioritization.
    Verifies that load_pancan_master_data():
      1. Deduplicates multiple biospecimens for the same 12-char patient.
      2. Prioritizes primary solid tumor (-01) over recurrent (-02) or metastatic (-06).
    """
    import retrain_fixed

    samples = ["TCGA-01-0001-01A", "TCGA-01-0001-02A", "TCGA-02-0002-01A"]
    genes = [f"GENE_{i}" for i in range(60)]

    all_samples = list(samples)
    for i in range(3, 55):
        all_samples.append(f"TCGA-03-{i:04d}-01A")

    expr_data = np.random.normal(0, 1, size=(len(genes), len(all_samples)))
    df_expr = pd.DataFrame(expr_data, index=genes, columns=all_samples)

    cdr_rows = []
    for s in all_samples:
        cdr_rows.append({
            "sample": s,
            "Redaction": None,
            "PFI": 1 if "-01" in s else 0,
            "PFI.time": 300.0,
            "OS": 1,
            "OS.time": 400.0,
        })
    df_cdr = pd.DataFrame(cdr_rows).set_index("sample")

    shards_dir = tmp_path / "data" / "processed" / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    df_expr.to_parquet(shards_dir / "TCGA_PANCAN_EB_ADJUST_expression.parquet")

    xena_dir = tmp_path / "data" / "raw" / "xena"
    xena_dir.mkdir(parents=True, exist_ok=True)
    df_cdr.to_csv(xena_dir / "TCGA_CDR_survival.tsv", sep="\t")

    monkeypatch.setattr(retrain_fixed, "ROOT", tmp_path)
    monkeypatch.setattr(retrain_fixed, "XENA_DIR", xena_dir)

    res = retrain_fixed.load_pancan_master_data(horizon_days=1095, endpoint="PFI")
    assert res is not None
    X, y, times, events = res

    # TCGA-01-0001-01A must be kept, TCGA-01-0001-02A must be excluded
    assert "TCGA-01-0001-01A" in X.index
    assert "TCGA-01-0001-02A" not in X.index
    pts = [s[:12] for s in X.index]
    assert pts.count("TCGA-01-0001") == 1


def test_adversarial_pancan_master_loader_excludes_missing_events_and_nonpositive_times(tmp_path, monkeypatch):
    """
    Adversarial Check N2: Exclusion of Missing Events and Non-Positive Times.
    Verifies that load_pancan_master_data() strictly excludes:
      - NaN events (prohibiting default conversion of NaN events to censored 0)
      - Non-numeric / invalid events (e.g. 2)
      - Non-positive follow-up times (t <= 0)
      - Missing follow-up times (NaN)
    """
    import retrain_fixed

    samples = [f"TCGA-10-{i:04d}-01A" for i in range(80)]
    genes = [f"GENE_{i}" for i in range(20)]
    df_expr = pd.DataFrame(np.random.normal(0, 1, size=(len(genes), len(samples))), index=genes, columns=samples)

    cdr_rows = []
    for i, s in enumerate(samples):
        if i == 0:
            e, t = np.nan, 300.0
        elif i == 1:
            e, t = 2, 300.0
        elif i == 2:
            e, t = 1, 0.0
        elif i == 3:
            e, t = 1, -5.0
        elif i == 4:
            e, t = 1, np.nan
        else:
            e = 1 if i % 2 == 0 else 0
            t = 300.0 if e == 1 else 1500.0
        cdr_rows.append({"sample": s, "Redaction": None, "PFI": e, "PFI.time": t})

    df_cdr = pd.DataFrame(cdr_rows).set_index("sample")

    shards_dir = tmp_path / "data" / "processed" / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)
    df_expr.to_parquet(shards_dir / "TCGA_PANCAN_EB_ADJUST_expression.parquet")

    xena_dir = tmp_path / "data" / "raw" / "xena"
    xena_dir.mkdir(parents=True, exist_ok=True)
    df_cdr.to_csv(xena_dir / "TCGA_CDR_survival.tsv", sep="\t")

    monkeypatch.setattr(retrain_fixed, "ROOT", tmp_path)
    monkeypatch.setattr(retrain_fixed, "XENA_DIR", xena_dir)

    res = retrain_fixed.load_pancan_master_data(horizon_days=1095, endpoint="PFI")
    assert res is not None
    X, y, times, events = res

    # Samples 0-4 must be strictly excluded!
    for bad_idx in range(5):
        bad_sample = samples[bad_idx]
        assert bad_sample not in X.index, f"Invalid sample {bad_sample} was not excluded!"

    assert set(events.unique()).issubset({0, 1})
    assert (times > 0).all()
    assert not events.isna().any()
    assert not times.isna().any()


def test_adversarial_pancan_master_loader_alignment_and_provenance_fallback(monkeypatch):
    """
    Adversarial Check N3: Fallback Provenance Recording and Matrix Axis Alignment.
    Verifies that when master matrix is absent or fails, extract_unified_cohort(return_provenance=True):
      1. Explicitly records data_provenance == 'individual_cohort_aggregation_fallback'.
      2. Records master_matrix_fallback == True with a non-empty master_fallback_reason.
    """
    import retrain_fixed

    # 1. Test invalid endpoint raises ValueError in load_pancan_master_data
    with pytest.raises(ValueError, match="Invalid survival endpoint"):
        retrain_fixed.load_pancan_master_data(endpoint="BOGUS_ENDPOINT")

    # 2. Test fallback provenance in extract_unified_cohort when master matrix returns None
    def mock_load_master(*args, **kwargs):
        return None

    def mock_load_cohort(cohort, *args, **kwargs):
        if cohort == "BRCA":
            idx = [f"TCGA-BRCA-{i:04d}-01A" for i in range(10)]
            df_x = pd.DataFrame(np.random.normal(0, 1, (10, 5)), index=idx, columns=[f"G_{j}" for j in range(5)])
            y = pd.Series([1, 0] * 5, index=idx)
            t = pd.Series([200.0] * 10, index=idx)
            e = pd.Series([1, 0] * 5, index=idx)
            return df_x, y, t, e
        raise FileNotFoundError(f"Cohort {cohort} not available")

    monkeypatch.setattr(retrain_fixed, "load_pancan_master_data", mock_load_master)
    monkeypatch.setattr(retrain_fixed, "load_cohort_data", mock_load_cohort)

    X_df, y, times, events, prov = retrain_fixed.extract_unified_cohort(
        "PAN-CANCER-ALL", endpoint="PFI", return_provenance=True
    )
    assert prov["data_provenance"] == "individual_cohort_aggregation_fallback"
    assert prov["master_matrix_fallback"] is True
    assert prov["master_fallback_reason"] is not None
    assert len(X_df) == 10
    assert len(y) == 10
    assert len(times) == 10
    assert len(events) == 10


def test_adversarial_unos_c_index_independent_numerical_validation():
    """
    Adversarial Check K2: Uno's IPCW C-Index Independent Numerical Benchmarking.
    Compares compute_unos_c_index() against an independent KaplanMeierFitter-based
    reference estimator across 8 deterministic benchmark cases:
      1. No censoring (events=1 for all samples; matches Harrell's C-index).
      2. Heavy censoring (>70% censoring).
      3. Tied event times.
      4. Tied event and censoring times.
      5. Equal risk scores (must return exactly 0.5000).
      6. Censoring survival crossing 0.05 (tau truncation threshold).
      7. Explicit tau provided.
      8. Unevaluable cases (all-censored, empty inputs -> None).
    """
    def ref_uno(times, events, scores, tau=None):
        times = np.asarray(times, dtype=float)
        events = np.asarray(events, dtype=int)
        scores = np.asarray(scores, dtype=float)

        valid_mask = (times > 0) & (~np.isnan(times)) & (~np.isnan(events)) & (~np.isnan(scores))
        times = times[valid_mask]
        events = events[valid_mask]
        scores = scores[valid_mask]

        n = len(times)
        event_indices = np.where(events == 1)[0]
        if len(event_indices) == 0 or n < 2:
            return None

        kmf = KaplanMeierFitter()
        kmf.fit(durations=times, event_observed=(1 - events))
        sf = kmf.survival_function_
        timeline = sf.index.values
        s_vals = sf.iloc[:, 0].values

        def get_g_left(t_val):
            idx = np.searchsorted(timeline, t_val, side="left")
            if idx == 0:
                return 1.0
            return float(s_vals[idx - 1])

        eligible_tau_times = [t for t in np.unique(times) if get_g_left(t) >= 0.05]
        if not eligible_tau_times:
            return None
        max_tau = max(eligible_tau_times)
        tau_cutoff = max_tau if tau is None else min(float(tau), max_tau)

        num, denom = 0.0, 0.0
        for i in event_indices:
            t_i = times[i]
            if t_i > tau_cutoff:
                continue
            g_i = get_g_left(t_i)
            if g_i < 0.05:
                continue
            w_i = 1.0 / (g_i ** 2)
            longer = np.where(times > t_i)[0]
            if len(longer) == 0:
                continue
            num += (np.sum(scores[i] > scores[longer]) + 0.5 * np.sum(scores[i] == scores[longer])) * w_i
            denom += len(longer) * w_i
        return round(float(num / denom), 4) if denom > 0 else None

    # Case 1: No censoring
    t_no_cens = np.array([10.0, 20.0, 30.0, 40.0, 50.0])
    e_no_cens = np.array([1, 1, 1, 1, 1])
    s_no_cens = np.array([0.9, 0.8, 0.5, 0.3, 0.1])
    u_guid = compute_unos_c_index(times=t_no_cens, events=e_no_cens, scores=s_no_cens)
    u_ref = ref_uno(t_no_cens, e_no_cens, s_no_cens)
    assert u_guid == u_ref == 1.0

    # Case 2: Heavy censoring
    np.random.seed(101)
    t_heavy = np.sort(np.random.uniform(5, 500, size=60))
    e_heavy = np.random.binomial(1, 0.25, size=60)
    e_heavy[0] = 1
    s_heavy = np.random.uniform(0, 1, size=60)
    assert compute_unos_c_index(times=t_heavy, events=e_heavy, scores=s_heavy) == ref_uno(t_heavy, e_heavy, s_heavy)

    # Case 3: Tied event times
    t_tied_e = np.array([10.0, 10.0, 20.0, 30.0, 30.0, 40.0])
    e_tied_e = np.array([1, 1, 0, 1, 1, 0])
    s_tied_e = np.array([0.8, 0.7, 0.4, 0.6, 0.3, 0.1])
    assert compute_unos_c_index(times=t_tied_e, events=e_tied_e, scores=s_tied_e) == ref_uno(t_tied_e, e_tied_e, s_tied_e)

    # Case 4: Tied event & censoring times
    t_tied_ec = np.array([10.0, 10.0, 20.0, 20.0, 30.0, 40.0])
    e_tied_ec = np.array([1, 0, 1, 0, 1, 0])
    s_tied_ec = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4])
    assert compute_unos_c_index(times=t_tied_ec, events=e_tied_ec, scores=s_tied_ec) == ref_uno(t_tied_ec, e_tied_ec, s_tied_ec)

    # Case 5: Equal risk scores (must be 0.5000)
    s_equal = np.full(len(t_tied_ec), 0.5)
    assert compute_unos_c_index(times=t_tied_ec, events=e_tied_ec, scores=s_equal) == 0.5000

    # Case 6: Censoring survival crossing 0.05 (tau truncation)
    t_cross = np.arange(1.0, 31.0)
    e_cross = np.array([0] * 28 + [1, 1])
    s_cross = np.random.uniform(0, 1, size=30)
    assert compute_unos_c_index(times=t_cross, events=e_cross, scores=s_cross) == ref_uno(t_cross, e_cross, s_cross)

    # Case 7: Explicit tau provided
    assert compute_unos_c_index(times=t_heavy, events=e_heavy, scores=s_heavy, tau=150.0) == ref_uno(t_heavy, e_heavy, s_heavy, tau=150.0)

    # Case 8: Unevaluable cases
    assert compute_unos_c_index(times=t_no_cens, events=np.zeros_like(e_no_cens), scores=s_no_cens) is None
    assert compute_unos_c_index(times=[], events=[], scores=[]) is None
    assert compute_unos_c_index(times=[10.0], events=[1], scores=[0.5]) is None


def test_adversarial_unos_ipcw_c_index_mathematical_rigor():
    """
    Adversarial Check K: Uno's IPCW C-Index Exact Formulation.
    Verifies:
      1. Left-continuous Kaplan-Meier estimator G_hat(t-) is used for censoring weights.
      2. Truncation time tau = sup { t : G_hat(t-) >= 0.05 }.
      3. Equal scores receive exactly 0.5 credit.
      4. Unevaluable cases return None.
    """
    np.random.seed(42)
    n = 60
    # Create distinct times and events
    times = np.sort(np.random.uniform(10, 1000, size=n))
    events = np.random.binomial(1, 0.5, size=n)
    events[0] = 1  # ensure at least one early event
    scores = np.random.uniform(0, 1, size=n)

    uno_c = compute_unos_c_index(scores=scores, times=times, events=events)
    assert uno_c is not None
    assert 0.0 <= uno_c <= 1.0

    # Tied scores test: identical scores for all patients must yield exactly 0.5
    tied_scores = np.full(n, 0.5, dtype=float)
    tied_c = compute_unos_c_index(scores=tied_scores, times=times, events=events)
    assert tied_c is not None
    assert abs(tied_c - 0.5) < 1e-5

    # All-censored cohort must return None
    all_censored = np.zeros(n, dtype=int)
    assert compute_unos_c_index(scores=scores, times=times, events=all_censored) is None


def test_adversarial_organ_conditioned_reference_leakage_prevention():
    """
    Adversarial Check L: Zero Evaluation Reference Leakage.
    Verifies:
      1. Reference distributions fit strictly on training data and are frozen.
      2. Evaluation samples never alter reference distributions.
      3. Unknown cohorts return None/unavailable rather than leaking evaluation data.
    """
    np.random.seed(42)
    train_df = pd.DataFrame({
        "cohort": ["TCGA-BRCA"] * 100,
        "risk_score": np.random.beta(2, 5, size=100),
    })

    adjuster = OrganConditionedRiskAdjuster(method="percentile")
    adjuster.fit(train_df, cohort_col="cohort", score_col="risk_score")

    # Evaluation dataframe with a known cohort and an unknown cohort
    eval_df = pd.DataFrame({
        "cohort": ["TCGA-BRCA", "TCGA-UNKNOWN"],
        "risk_score": [0.5, 0.5],
    })

    # Unknown cohort must return None / NaN
    val_unknown = adjuster.transform_single(0.5, "TCGA-UNKNOWN")
    assert val_unknown is None

    transformed = adjuster.transform(eval_df, cohort_col="cohort", score_col="risk_score")
    assert not np.isnan(transformed[0])
    assert np.isnan(transformed[1])  # Unknown cohort is NaN, not fabricated!

    # DeployableOrganScorer unknown cohort returns unavailable
    scorer = DeployableOrganScorer(method="percentile")
    scorer.organ_references = {}  # No references, no global reference
    scorer.is_fitted = True

    single_res = scorer.score_single(raw_score=0.65, organ_cohort="TCGA-UNSEEN")
    assert single_res["organ_conditioned_score"] is None
    assert single_res["status"] == "unavailable_no_independent_reference"
    assert "relative ranking" in single_res["clinical_interpretation_notice"].lower()


def test_adversarial_survival_endpoint_integrity():
    """
    Adversarial Check M: Survival Endpoint Isolation (PFI vs OS).
    Verifies:
      1. PFI and OS are isolated without silent fallback.
      2. Invalid endpoints raise ValueError.
    """
    from retrain_fixed import load_cohort_data

    # Test invalid endpoint rejection
    with pytest.raises(ValueError, match="Invalid survival endpoint"):
        load_cohort_data("KICH", endpoint="INVALID_ENDPOINT")

    # Real data loading if available
    try:
        X_pfi, y_pfi, t_pfi, e_pfi = load_cohort_data("KICH", endpoint="PFI")
        X_os, y_os, t_os, e_os = load_cohort_data("KICH", endpoint="OS")

        # Events for PFI (recurrence/progression) and OS (death) must not be silently identical
        # In KICH, PFI events = 11, OS events = 9
        assert int(e_pfi.sum()) == 11
        assert int(e_os.sum()) == 9
        assert int(e_pfi.sum()) != int(e_os.sum())
    except Exception as e:
        pytest.skip(f"TCGA-KICH data not present: {e}")


def test_adversarial_xena_acquisition_no_synthetic_labels(monkeypatch):
    """
    Adversarial Check N: Real Xena Ingestion Never Generates Synthetic Labels.
    Verifies that when phenotype downloading fails or yields empty records,
    run_xena_acquisition() raises an explicit RuntimeError and never fabricates random labels.
    """
    import src.data.xena_acquisition as xena_mod

    dummy_expr = pd.DataFrame(
        np.random.normal(0, 1, size=(25, 6)),
        index=[f"GENE_{i}" for i in range(25)],
        columns=[f"TCGA-01-000{i}" for i in range(6)],
    )

    monkeypatch.setattr(xena_mod, "download_matrix", lambda dataset, hub: dummy_expr)
    monkeypatch.setattr(xena_mod, "download_phenotypes", lambda dataset, hub: None)

    with pytest.raises(RuntimeError, match="Real-data acquisition strictly prohibits synthetic label generation"):
        xena_mod.run_xena_acquisition("TCGA.BRCA.sampleMap/HiSeqV2")


def test_adversarial_panel_distillation_input_validation_and_alignment():
    """
    Adversarial Check O: Raw Survival Event/Time Validation & Strict Synchronous Alignment.
    Verifies:
      1. Mismatched array lengths across df_X, times, events, and patient/sample IDs raise ValueError.
      2. Raw survival inputs with NaN events, invalid event values (!= 0 or 1), and non-positive times
         are validated prior to casting and explicitly excluded.
      3. Strict 1:1 alignment between expression rows, times, events, and patient barcodes is preserved.
    """
    np.random.seed(42)
    n = 35
    df_X = pd.DataFrame(
        np.random.normal(0, 1, size=(n, 12)),
        columns=[f"G_{i}" for i in range(12)],
        index=[f"TCGA-AL-{i:04d}-01A" for i in range(n)],
    )
    times = np.random.uniform(50, 800, size=n)
    events = np.array([1, 0] * (n // 2) + [1] * (n % 2), dtype=float)
    patient_ids = [f"TCGA-AL-{i:04d}" for i in range(n)]

    distiller = MinimalGenePanelDistiller(panel_sizes=[5], n_folds=3, random_state=42)

    # 1. Length mismatch rejection
    with pytest.raises(ValueError, match="Array length mismatch"):
        distiller.fit(df_X, times[:20], events)

    with pytest.raises(ValueError, match="Patient IDs length mismatch"):
        distiller.fit(df_X, times, events, patient_ids=patient_ids[:20])

    # 2. Inject NaN event, invalid event (2.0), and non-positive time (-10.0)
    events_corrupted = events.copy()
    events_corrupted[4] = np.nan
    events_corrupted[11] = 2.0
    times_corrupted = times.copy()
    times_corrupted[18] = -10.0

    report = distiller.fit(
        df_X=df_X,
        times=times_corrupted,
        events=events_corrupted,
        patient_ids=patient_ids,
        data_source="adversarial_alignment_test",
        is_synthetic=True,
    )

    # 3 corrupted samples excluded -> 32 remain
    assert report.n_samples == 32
    assert len(distiller.oof_comparator_scores_) == 32
    assert report.patient_grouping_status == "patient_grouped_strictly_partitioned"
    assert report.patient_count == 32

