# GUIDO Clinical Cancer Risk Engine: Clinical Translation & Regulatory Roadmap

**Document ID:** GUIDO-REG-TR-001  
**Version:** 1.1.0 (In-Silico Research & Engineering Specification)  
**Date:** 2026-11-26  
**Status:** In-Silico Research Specification (Research Use Only — RUO)  
**Target Regulatory Jurisdictions:** India (CDSCO), United States (US FDA), European Union (EU IVDR)  

---

## 1. Intended Use Statement & Indications for Use (IFU)

### 1.1 Intended Use
> **"GUIDO (Genomic Unified In Silico Disease Oncology Risk Engine) is an investigational bioinformatics software framework intended exclusively for Research Use Only (RUO). GUIDO quantitatively analyzes tumor transcriptomic expression profiles (bulk RNA-seq or targeted RT-qPCR / NanoString panel candidate designs) to evaluate in-silico associations with 3-year progression-free interval (PFI) and recurrence risk in adult solid tumor oncology research cohorts."**

### 1.2 Indications for Use & Target Research Population
* **Intended Users:** Clinical oncology researchers, cancer computational biologists, molecular pathologists, and biostatisticians conducting investigational oncology biomarker studies.
* **Target Specimen Types:** Formalin-Fixed Paraffin-Embedded (FFPE) tumor tissue biopsies, core needle biopsies, or fresh-frozen surgical resections with verified pre-analytical quality criteria ($DV200 \ge 30\%$ when measured, $< 40\%$ gene dropout).
* **Research Indications:** Exploratory evaluation of recurrence risk, cross-platform normalization assessment, and minimal prognostic panel distillation.

### 1.3 Absolute Contraindications & Clinical Limitations
* **RESEARCH USE ONLY (RUO):** GUIDO is NOT cleared, approved, or certified for clinical diagnosis, patient management, or treatment decision-making by CDSCO, US FDA, EMA, or any national regulatory authority.
* **NOT FOR PRIMARY DIAGNOSIS:** GUIDO does not differentiate benign from malignant tissue. Histopathological diagnosis by a certified pathologist is mandatory.
* **NO AUTONOMOUS THERAPEUTIC SELECTION:** GUIDO risk scores must NEVER be used to initiate, alter, withhold, or discontinue chemotherapy, targeted therapy, radiation, or immunotherapy.
* **RETROSPECTIVE IN-SILICO NATURE:** All performance metrics reported herein derive from retrospective analyses of public observational cohorts (TCGA, METABRIC, GEO). Prospective clinical validity and utility have not been established.

---

## 2. Global Regulatory Classification Analysis

| Regulatory Body | Jurisdiction | Classification Framework | Determined Class | Regulatory Pathway | Quality Management System (QMS) |
|---|---|---|---|---|---|
| **CDSCO** | India | Medical Device Rules 2017 (Rule 3(zb) for SaMD) | **Class C** (Moderate-High Risk SaMD) | Form MD-14 (Import) / MD-7 (Mfg); Clinical investigation dossier under Chapter VII | Targeting ISO 13485:2016 alignment + Schedule M-III |
| **US FDA** | United States | 21 CFR 866 (Immunology & Microbiology) | **Class II / De Novo** | De Novo Request (establishing Special Controls) or 510(k) | Targeting 21 CFR Part 820 (QMSR) + ISO 14971:2019 alignment |
| **EU IVDR** | European Union | Regulation (EU) 2017/746 | **Class C** (Rule 3(k): Cancer Staging/Prognosis) | Notified Body Conformity Assessment (Annex IX/X) | Targeting ISO 13485:2016 + IVDR Article 10 alignment |

### 2.1 India CDSCO Analysis (Medical Device Rules 2017)
* **SaMD Definition:** Governed under Rule 3(zb) of the Medical Device Rules 2017, which defines software designed to analyze genomic data for prognostic or diagnostic purposes as Software as a Medical Device (SaMD).
* **Classification:** Oncology multi-gene prognostic algorithms are categorized as **Class C (Moderate-High Risk)** due to the potential clinical impact of misclassified recurrence risk.
* **Regulatory Prerequisites:** Formal deployment requires establishing an ISO 13485:2016 certified QMS, conducting local clinical performance evaluation (or obtaining an authorized waiver under Rule 101/102 with local clinical trial data), and securing a Form MD-15 import/manufacturing license.

### 2.2 United States FDA Analysis
* **Regulatory Classification:** Software analyzing multi-gene expression signatures to predict oncology outcomes is classified as an In Vitro Diagnostic Multivariate Index Assay (IVDMIA) under 21 CFR 866.
* **Pathway Strategy:** A De Novo classification petition will be required for multi-organ indications lacking an identical predicate. Once special controls are published, secondary indications can follow the 510(k) pathway.
* **Standards Alignment:** Software lifecycle processes must be structured to comply with IEC 62304 (Medical device software lifecycle), FDA Guidance on Content of Premarket Submissions for Device Software Functions (2023), and 21 CFR Part 820.

### 2.3 European Union IVDR (Regulation 2017/746) Analysis
* **Classification Rule:** Annex VIII, Rule 3(k) explicitly assigns software intended for cancer prognosis or disease staging to **Class C**.
* **Performance Evaluation:** Requires a comprehensive Performance Evaluation Report (PER) encompassing Scientific Validity, Analytical Performance (conducted in an accredited laboratory), and Clinical Performance demonstrated across relevant target populations.

---

## 3. Data Provenance & Scientific Evidence Base

### 3.1 Retrospective In-Silico Training Cohorts
* **Discovery Consortia:** National Cancer Institute (NCI) / NHGRI The Cancer Genome Atlas (TCGA) Pan-Cancer Atlas.
* **Outcomes Grounding:** NIH TCGA Pan-Cancer Clinical Data Resource (TCGA-CDR, Liu et al., *Cell* 2018). Endpoint: Progression-Free Interval (PFI, days to progression, recurrence, or metastasis).
* **Local Data Transparency:**
  - Evaluated on available local TCGA cohorts (TCGA-BRCA, TCGA-KICH, TCGA-KIRC, TCGA-KIRP).
  - The remaining 29 TCGA cohorts require streaming ingestion from cloud repositories.
  - Claims of pan-cancer validity reflect method architecture and must not be conflated with full completed empirical validation across all 33 cancer types.

### 3.2 External Retrospective Hospital Benchmarks
* **Breast Cancer (METABRIC):** Retrospective analysis of cBioPortal METABRIC cohort ($N = 1,980$ patients) for out-of-distribution evaluation.
* **Lung Adenocarcinoma (NCBI GEO GSE31210):** Retrospective analysis of Okayama et al. Stage I/II surgical cohort ($N = 226$ patients).
* **Partition Discipline:** Strictly blind, out-of-distribution evaluation without retraining or parameter fine-tuning.

### 3.3 Clinical Pre-Analytical Quality Control (QC) Gates
Pre-analytical quality checks are enforced before risk scoring:
1. **DV200 Score (When Available):**
   - $\ge 50\%$: Optimal FFPE RNA integrity.
   - $30\% - 49\%$: Moderate degradation; flag elevated risk uncertainty on report.
   - $< 30\%$: Hard rejection; specimen rejected with guidance for re-extraction.
   - *Honest Missingness Notice:* When DV200 is unrecorded in public datasets or fresh-frozen specimens, it is explicitly cataloged as `UNMEASURED_NOT_RECORDED` rather than imputed with fabricated proxy values.
2. **Gene Dropout / Missingness:** Hard rejection if $> 40\%$ of target assay genes are unmeasured or unexpressed.
3. **Qualified Housekeeping Controls:** Verified detectability of empirically stable cross-tissue normalizers ($RPL13A, ACTB, PPIA, TBP$). Tissue-variable genes ($B2M, HPRT1, PGK1$) are explicitly excluded as solitary normalizers.

---

## 4. Phased Clinical Translation Roadmap

```
┌─────────────────────────────────┐     ┌──────────────────────────────────┐     ┌──────────────────────────────────┐
│             PHASE A             │     │             PHASE B              │     │             PHASE C              │
│    Discovery & Retrospective    │ ──> │       Wet-Lab Analytical         │ ──> │        Prospective Multi-Center  │
│       In-Silico Research        │     │       Assay Validation           │     │          Clinical Trials         │
│     [COMPLETED IN-SILICO]       │     │         [MONTHS 1 - 12]          │     │         [MONTHS 12 - 24]         │
└─────────────────────────────────┘     └──────────────────────────────────┘     └──────────────────────────────────┘
  • Real TCGA cohort evaluation           • Primer/probe wet-lab synthesis         • Multi-center clinical protocol
  • Organ-stratified C-index              • NABL/CAP-CLIA lab partnership          • Oncologist decision-impact study
  • Stage-adjusted Cox regression         • Wet-lab LoD, dynamic range, LoB        • CDSCO Class C dossier submission
  • In-silico panel distillation          • Formal ISO 13485 QMS audit             • Health economic outcomes (HEOR)
```

### Phase A: Biomarker Discovery & Retrospective In-Silico Research (Current Status)
* Formulated penalized Cox Proportional Hazards feature selection with nested cross-validation out-of-fold scoring.
* Evaluated organ-stratified prognostic discrimination within individual tumor types using Harrell's and Uno's IPCW C-index with Benjamini-Hochberg FDR correction.
* Modeled stage- and age-adjusted Cox proportional hazards with Grambsch-Therneau proportional hazards diagnostics.
* Distilled high-dimensional transcriptomic models into candidate minimal panels ($15, 25, 35, 50$ genes) as in-silico candidate specifications.
* Implemented deployable organ-conditioned scoring using frozen training reference distributions for deterministic $N=1$ patient evaluation.

### Phase B: Analytical Assay Development & Wet-Lab Validation (Target Months 1–12)
1. **Wet-Lab Assay Synthesis:** Synthesize custom probe/primer pairs for distilled candidate panels on target diagnostic platforms (RT-qPCR TaqMan arrays or NanoString nCounter).
2. **Accredited Laboratory Partnership:** Execute formal collaborative research agreements with accredited diagnostic laboratories (NABL ISO 15189 / CAP-CLIA).
3. **Analytical Validation Protocol:**
   - **Limit of Detection (LoD) & Limit of Blank (LoB):** Establish input titration curves across serial dilutions of reference RNA.
   - **Amplification Efficiency:** Verify primer amplification efficiencies strictly within 90%–110% ($R^2 \ge 0.98$).
   - **Analytical Specificity:** Confirm absence of cross-hybridization with homologous isoforms.
   - **Analytical Precision:** Verify intra-run and inter-run repeatability ($CV < 8\%$) across technical replicates.
4. **Target Specifications (Aspirational Goals):**
   - *Target Turnaround Time:* Aspirational goal of 24–48 hours from specimen receipt to report generation. (Actual commercial turnaround will depend on laboratory operational schedules and batching frequencies).
   - *Target Cost:* Aspirational reagent target ($< \$200$ USD / ₹16,000 INR) to ensure accessible healthcare adoption. (Actual costs depend on third-party commercial synthesis and licensing agreements).

### Phase C: Prospective Clinical Validation & Regulatory Submission (Target Months 12–24)
1. **Archival Institutional Cohort Study:** Validate candidate assays on independent institutional FFPE cohorts with verified 5-year longitudinal follow-up.
2. **Clinical Decision Impact Study:** Measure concordance and therapeutic recommendation shifts among medical oncologists presented with standard clinical staging versus staging plus GUIDO genomic recurrence risk.
3. **Regulatory Submission Dossier:** Compile and submit formal medical device registration dossiers to CDSCO under MDR 2017 Rule 3(zb).

---

## 5. Software Risk Management Architecture (Targeting ISO 14971 Alignment)

The software architecture is engineered to align with medical device risk management principles (ISO 14971:2019 / IEC 62304):

| Hazard ID | Potential Failure Mode | Clinical Harm | Engineering Risk Mitigation in GUIDO |
|---|---|---|---|
| **HAZ-001** | Organ baseline survival confounding | False positive high risk in aggressive organ types (e.g. GBM) | Organ-stratified baseline hazards and frozen training reference percentile distributions |
| **HAZ-002** | Severe FFPE RNA degradation | Inaccurate gene expression quantification | Automated pre-analytical QC gate ($DV200 < 30\%$ triggers hard rejection) |
| **HAZ-003** | Inter-platform laboratory gain shifts | Systematic risk score drift across platforms | Within-sample rank transformation and qualified housekeeping ratio normalization |
| **HAZ-004** | Misinterpretation as diagnostic tool | Inappropriate clinical decision-making | Mandatory RUO watermark, limitation disclosures, and requirement for pathologist confirmation |
| **HAZ-005** | Transductive batch-size leakage | Altered individual patient score depending on test batch composition | Deterministic single-patient ($N=1$) scoring using frozen reference distributions |

*Regulatory Notice:* The risk controls above reflect software architectural implementations. Formal certification of ISO 14971 compliance requires an institutional audit under an accredited ISO 13485 Quality Management System, which remains to be performed during Phase B.

---

## 6. Document Governance & Research Sign-Off

This document constitutes an engineering and regulatory planning specification for investigational software. It records computational methodologies and translation roadmaps developed during research.

* **Document Owner:** GUIDO Oncology Research Consortium
* **Status:** In-Silico Research & Engineering Specification (RUO)
* **Institutional Governance Notice:** Formal institutional approvals, clinical trial authorizations, and regulatory submissions will be executed by designated clinical investigators, principal investigators, and regulatory sponsors upon initiation of Phase B wet-lab translation.
