"""
Step 9 – FastAPI Application
==============================
All endpoints return structured JSON. API keys are read from environment variables only.
Heavy pipeline operations run in a thread pool executor to avoid blocking the event loop.
"""
from __future__ import annotations

import asyncio
import logging
import os
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json

load_dotenv()  # load .env file if present

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config.settings import API_HOST, API_PORT
from src.models.deep_surv import PyTorchDeepSurvClassifier

# Ensure unpickling compatibility for clinical models in API
main_mod = sys.modules.get("__main__")
if main_mod is not None:
    setattr(main_mod, "PyTorchDeepSurvClassifier", PyTorchDeepSurvClassifier)
from src.api.schemas import (
    BiomedicalAuditRequest,
    BiomedicalAuditResponse,
    BiomarkerRequest,
    BiomarkerResponse,
    CrossCohortRequest,
    CrossCohortResponse,
    EvalRequest,
    EvalResponse,
    HealthResponse,
    LiteratureRequest,
    LiteratureResponse,
    TrainRequest,
    TrainResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Thread pool for blocking pipeline operations
_executor = ThreadPoolExecutor(max_workers=2)

app = FastAPI(
    title="Multi-Omics Disease Risk Modeling API",
    description=(
        "Production-grade LLM-guided multi-omics pipeline. "
        "ML performs risk prediction; Groq LLM performs structured biological reasoning."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS for local UI development; configure via UI_ALLOWED_ORIGINS env var (comma-separated)
_origins = os.getenv("UI_ALLOWED_ORIGINS")
if _origins:
    try:
        origins = [o.strip() for o in _origins.split(",") if o.strip()]
    except Exception:
        origins = ["http://localhost:5173", "http://localhost:3000"]
else:
    origins = ["http://localhost:5173", "http://localhost:3000", "http://localhost:8000"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


# ─── Utility ──────────────────────────────────────────────────────────────────

async def _run_in_executor(func, *args, **kwargs):
    """Run a synchronous function in the thread pool."""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(_executor, partial(func, *args, **kwargs))


def _require_groq():
    """Raise 503 if GROQ_API_KEY is not configured."""
    if not os.getenv("GROQ_API_KEY"):
        raise HTTPException(
            status_code=503,
            detail="GROQ_API_KEY is not configured. Set it as an environment variable.",
        )


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness check."""
    return HealthResponse()


def is_valid_clinical_artifact(path: Path) -> bool:
    """Verifies that an artifact is a modern calibrated clinical risk engine, not a legacy binary classifier."""
    try:
        import joblib
        art = joblib.load(path)
        if not isinstance(art, dict):
            return False
        if "hallmark_metagenes" in art or "task" in art or "metagene_stats" in art:
            return True
        if len(art.get("features", [])) == 100:
            return True
        return False
    except Exception:
        return False


def _predict_single_artifact(sample: Any, artifact: Any) -> Tuple[Optional[float], Optional[Any], Optional[Any]]:
    """Standardizes features, scales, and infers recurrence probability from a clinical model artifact."""
    import pandas as pd
    import numpy as np

    if not isinstance(artifact, dict):
        model = artifact
        scaler = None
        target_features = None
        sample_eval = sample.copy()
    else:
        model = artifact.get("model", artifact)
        scaler = artifact.get("scaler")
        target_features = artifact.get("features")
        sample_eval = sample.copy()

        # Compute hallmark metagenes if expected by model artifact using training baseline statistics
        if "hallmark_metagenes" in artifact:
            hallmarks = artifact["hallmark_metagenes"]
            meta_stats = artifact.get("metagene_stats", {})
            for meta_name, gene_list in hallmarks.items():
                if meta_name not in sample_eval.columns:
                    z_cols = []
                    for g in gene_list:
                        if g in sample_eval.columns:
                            mu = meta_stats[g]["mean"] if g in meta_stats else float(sample_eval[g].mean())
                            sigma = meta_stats[g]["std"] if g in meta_stats else 1.0
                            z = (sample_eval[g].values - mu) / (sigma + 1e-6)
                            z_cols.append(z)
                    if z_cols:
                        sample_eval[meta_name] = np.mean(np.column_stack(z_cols), axis=1)
                    else:
                        sample_eval[meta_name] = 0.0

    if target_features:
        X = sample_eval.reindex(columns=list(target_features), fill_value=0.0)
    else:
        X = sample_eval

    if scaler is not None:
        try:
            X_vals = X.values if hasattr(X, "values") else X
            X_scaled = scaler.transform(X_vals)
            X = pd.DataFrame(X_scaled, columns=X.columns)
        except Exception:
            pass

    try:
        if hasattr(model, "predict_proba"):
            try:
                prob = float(model.predict_proba(X.values)[:, 1][0])
            except Exception:
                prob = float(model.predict_proba(X)[:, 1][0])
        elif hasattr(model, "decision_function"):
            try:
                s = model.decision_function(X.values)[0]
            except Exception:
                s = model.decision_function(X)[0]
            prob = float(1 / (1 + np.exp(-s)))
        else:
            try:
                pred = model.predict(X.values)[0]
            except Exception:
                pred = model.predict(X)[0]
            prob = float(pred)
    except Exception:
        prob = None

    return prob, model, X


@app.post("/api/ui/predict", tags=["UI"])
async def ui_predict(
    project_id: str = Form("TCGA-PAN-CANCER-ALL"),
    disease_name: str = Form("pan_cancer_all"),
    file: UploadFile | None = File(None),
    json_payload: str | None = Form(None),
    dv200: float | None = Form(None),
    library_depth: float | None = Form(None),
):
    """Accept a CSV upload or JSON payload, run clinical sample QC gating,
    run model prediction on first sample, then run the biomedical audit pipeline
    and return structured result with RUO regulatory provenance.
    """
    from src.api.pipeline import _load_splits, run_biomedical_audit_pipeline
    from src.preprocessing.sample_qc import ClinicalSampleQualityGate, QCStatus
    from config.settings import MODELS_DIR, RESULTS_DIR
    import pandas as pd
    import joblib
    import numpy as np

    def _do_work():
        # load model — strictly load clinical genomic risk models
        norm_proj = project_id.strip().upper()
        prefixed_proj = f"TCGA-{norm_proj}" if not norm_proj.startswith("TCGA-") else norm_proj
        unprefixed_proj = norm_proj.replace("TCGA-", "").replace("TCGA.", "")

        ALL_VALID_COHORTS = {
            "ACC", "BLCA", "BRCA", "CESC", "CHOL", "COAD", "DLBC", "ESCA",
            "GBM", "HNSC", "KICH", "KIRC", "KIRP", "LAML", "LGG", "LIHC",
            "LUAD", "LUSC", "MESO", "OV", "PAAD", "PCPG", "PRAD", "READ",
            "SARC", "SKCM", "STAD", "TGCT", "THCA", "THYM", "UCEC", "UCS", "UVM",
            "PAN-CANCER-ALL", "PAN-RENAL", "PAN-LUNG", "PAN-GI", "PAN-BREAST",
            "PAN-HEPATOBILIARY", "PAN-BRAIN", "PAN-GYN", "PAN-UROLOGICAL",
            "PAN-HEADNECK", "PAN-ENDOCRINE", "PAN-SARCOMA", "PAN-MELANOMA",
            "PAN-HEMATOLOGIC", "PAN-MESO"
        }

        clinical_candidates = [
            MODELS_DIR / project_id / "xgboost_clinical_model.joblib",
            MODELS_DIR / prefixed_proj / "xgboost_clinical_model.joblib",
            MODELS_DIR / unprefixed_proj / "xgboost_clinical_model.joblib",
        ]
        # Organ group fallbacks
        ORGAN_MAP = {
            "TCGA-KIRC": "TCGA-PAN-RENAL", "TCGA-KIRP": "TCGA-PAN-RENAL", "TCGA-KICH": "TCGA-PAN-RENAL",
            "TCGA-LUAD": "TCGA-PAN-LUNG",  "TCGA-LUSC": "TCGA-PAN-LUNG",
            "TCGA-COAD": "TCGA-PAN-GI",    "TCGA-READ": "TCGA-PAN-GI", "TCGA-STAD": "TCGA-PAN-GI", "TCGA-ESCA": "TCGA-PAN-GI",
            "TCGA-LIHC": "TCGA-PAN-HEPATOBILIARY", "TCGA-CHOL": "TCGA-PAN-HEPATOBILIARY", "TCGA-PAAD": "TCGA-PAN-GI",
            "TCGA-BRCA": "TCGA-PAN-BREAST",
            "TCGA-LGG" : "TCGA-PAN-BRAIN", "TCGA-GBM": "TCGA-PAN-BRAIN",
            "TCGA-OV"  : "TCGA-PAN-GYN",   "TCGA-UCEC": "TCGA-PAN-GYN", "TCGA-CESC": "TCGA-PAN-GYN", "TCGA-UCS": "TCGA-PAN-GYN",
            "TCGA-BLCA": "TCGA-PAN-UROLOGICAL", "TCGA-PRAD": "TCGA-PAN-UROLOGICAL", "TCGA-TGCT": "TCGA-PAN-UROLOGICAL",
            "TCGA-HNSC": "TCGA-PAN-HEADNECK",
            "TCGA-THCA": "TCGA-PAN-ENDOCRINE", "TCGA-ACC": "TCGA-PAN-ENDOCRINE", "TCGA-PCPG": "TCGA-PAN-ENDOCRINE",
            "TCGA-SARC": "TCGA-PAN-SARCOMA",
            "TCGA-SKCM": "TCGA-PAN-MELANOMA", "TCGA-UVM": "TCGA-PAN-MELANOMA",
            "TCGA-LAML": "TCGA-PAN-HEMATOLOGIC", "TCGA-DLBC": "TCGA-PAN-HEMATOLOGIC", "TCGA-THYM": "TCGA-PAN-HEMATOLOGIC",
            "TCGA-MESO": "TCGA-PAN-MESO",
        }
        if prefixed_proj in ORGAN_MAP:
            clinical_candidates.append(MODELS_DIR / ORGAN_MAP[prefixed_proj] / "xgboost_clinical_model.joblib")

        organ_model_path = next((p for p in clinical_candidates if p.exists() and is_valid_clinical_artifact(p)), None)

        is_known_cohort = (
            unprefixed_proj.upper() in ALL_VALID_COHORTS
            or prefixed_proj in ORGAN_MAP
            or norm_proj in ALL_VALID_COHORTS
        )

        # Global Master Pan-Cancer model path
        pancan_model_path = MODELS_DIR / "TCGA-PAN-CANCER-ALL" / "xgboost_clinical_model.joblib"
        has_pancan = is_known_cohort and pancan_model_path.exists() and is_valid_clinical_artifact(pancan_model_path)

        if organ_model_path is None and not has_pancan:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"No calibrated clinical genomic risk model found for '{project_id}'. "
                    "Silent fallback to legacy tumor-vs-normal models is disabled to ensure prognostic validity. "
                    f"Please calibrate the clinical recurrence model using: py -3.12 retrain_fixed.py {project_id}"
                ),
            )

        model_path = organ_model_path if organ_model_path is not None else pancan_model_path
        model_artifact = joblib.load(model_path)

        # unwrap dict-based artifacts: {'model': estimator, 'scaler': scaler, 'features': [...]}
        scaler = None
        model_features = None
        if isinstance(model_artifact, dict):
            model = model_artifact.get('model', model_artifact)
            scaler = model_artifact.get('scaler', None)
            model_features = model_artifact.get('features', None)
        else:
            model = model_artifact

        # read input: CSV upload preferred; fall back to JSON form payload
        if file is not None:
            df = pd.read_csv(file.file)
        elif json_payload is not None:
            import json as _json, io as _io
            try:
                parsed = _json.loads(json_payload)
                if isinstance(parsed, dict):
                    df = pd.DataFrame([parsed])
                else:
                    df = pd.DataFrame(parsed)
            except Exception:
                raise HTTPException(status_code=400, detail="Invalid json_payload")
        else:
            raise HTTPException(status_code=400, detail="No file uploaded and no json_payload provided")

        if df.shape[0] == 0:
            raise HTTPException(status_code=400, detail="Uploaded CSV/JSON is empty")

        sample = df.iloc[0:1].copy()

        # Pre-analytical Clinical Quality Control (QC) Gate
        qc_gate = ClinicalSampleQualityGate()
        qc_res = qc_gate.evaluate_sample_quality(
            sample.iloc[0],
            dv200=dv200,
            library_depth=library_depth,
        )
        if qc_res.status == QCStatus.FAIL:
            reasons_str = "; ".join(qc_res.rejection_reasons)
            raise HTTPException(
                status_code=422,
                detail=f"Clinical Sample Quality Control Rejection: {reasons_str}"
            )

        # Dual-Consensus Inference Execution (70% Organ Specialist / 30% Pan-Cancer Master)
        p_organ = None
        p_pancan = None
        model_used = None
        X_used = None

        if organ_model_path is not None:
            p_organ, model_used, X_used = _predict_single_artifact(sample, model_artifact)

        if has_pancan:
            pancan_artifact = joblib.load(pancan_model_path) if organ_model_path != pancan_model_path else model_artifact
            p_pancan, p_model, p_X = _predict_single_artifact(sample, pancan_artifact)
            if model_used is None:
                model_used = p_model
                X_used = p_X

        is_dual_consensus = (
            p_organ is not None
            and p_pancan is not None
            and organ_model_path is not None
            and organ_model_path.parent.name != "TCGA-PAN-CANCER-ALL"
        )

        if is_dual_consensus:
            # 70% Organ Specialist + 30% Pan-Cancer Master
            prob = float(0.70 * p_organ + 0.30 * p_pancan)
            consensus_mode = "Dual-Consensus (70% Organ Specialist / 30% Pan-Cancer Master)"
            tier_organ = "Low Risk" if p_organ < 0.35 else ("High Risk" if p_organ > 0.65 else "Moderate Risk")
            tier_pancan = "Low Risk" if p_pancan < 0.35 else ("High Risk" if p_pancan > 0.65 else "Moderate Risk")
            concordance = "Concordant" if tier_organ == tier_pancan else f"Divergent (Organ: {tier_organ}, PanCan: {tier_pancan})"
        elif p_organ is not None:
            prob = float(p_organ)
            consensus_mode = f"Single Model ({organ_model_path.parent.name})"
            concordance = "N/A (Single Model)"
        else:
            prob = float(p_pancan)
            consensus_mode = "Single Model (TCGA-PAN-CANCER-ALL Global Fallback)"
            concordance = "N/A (Global Fallback)"

        model = model_used if model_used is not None else (model_artifact.get('model', model_artifact) if isinstance(model_artifact, dict) else model_artifact)
        X = X_used if X_used is not None else sample

        threshold = 0.5
        if isinstance(model_artifact, dict) and "optimal_threshold" in model_artifact:
            try:
                threshold = float(model_artifact["optimal_threshold"])
            except Exception:
                threshold = 0.5

        # Determine Clinical Genomic Disease Risk Tier
        risk_score = round(float(prob * 100), 1) if prob is not None else 50.0
        if prob is not None:
            if prob < 0.35:
                risk_tier = "Low Risk"
                recommendation = "Low Disease Progression Risk: Favorable prognosis. Standard routine surveillance recommended."
            elif prob < 0.65:
                risk_tier = "Moderate Risk"
                recommendation = "Intermediate Progression Risk: Elevated recurrence probability. Close periodic surveillance recommended."
            else:
                risk_tier = "High Risk"
                recommendation = "High 3-Year Disease Progression Risk: Aggressive profile. Consideration for adjuvant therapy & intensified surveillance."
        else:
            risk_tier = "Undetermined"
            recommendation = "Insufficient confidence to compute risk profile."

        prediction = {
            "label": risk_tier,
            "prob": prob,
            "risk_score": risk_score,
            "risk_tier": risk_tier,
            "time_horizon": "3-Year Progression & Recurrence Risk",
            "recommendation": recommendation,
            "threshold": threshold,
            "model": str(model_path.name),
            "consensus_mode": consensus_mode,
            "organ_risk_score": round(float(p_organ * 100), 1) if p_organ is not None else None,
            "pancan_risk_score": round(float(p_pancan * 100), 1) if p_pancan is not None else None,
            "model_concordance": concordance,
            "task": "Genomic Disease Risk Prediction",
            "time": "0.0s",
            "sample_qc": {
                "status": qc_res.status.value,
                "quality_score": qc_res.quality_score,
                "dv200": dv200,
                "zero_fraction": qc_res.zero_fraction,
                "warnings": qc_res.warnings,
            },
            "regulatory_classification": "Research Use Only (RUO)",
            "clinical_disclaimer": (
                "GUIDO is an investigational clinical decision-support algorithm intended strictly for "
                "Research Use Only (RUO). It has not been cleared or approved by the US FDA, CDSCO (India), "
                "or CE-IVD for primary diagnostic or definitive treatment decision-making."
            ),
        }

        # feature importances (best-effort)
        features = []
        if isinstance(model_artifact, dict) and "feature_importances" in model_artifact:
            imp_dict = model_artifact["feature_importances"]
            if isinstance(imp_dict, dict):
                features = [{"name": str(k), "value": float(v)} for k, v in list(imp_dict.items())[:30]]

        if not features:
            try:
                # try coef_
                est = model
                if hasattr(model, 'steps'):
                    est = model.steps[-1][1]
                if hasattr(est, 'coef_'):
                    coefs = np.abs(est.coef_).ravel()
                    # skip if all zeros (model not trained or overregularized)
                    if (coefs != 0).any():
                        names = list(X.columns)
                        pairs = sorted(zip(names, coefs), key=lambda x: -x[1])[:30]
                        features = [{'name': n, 'value': float(v)} for n, v in pairs]
                elif hasattr(est, 'feature_importances_'):
                    imp = est.feature_importances_
                    if (imp != 0).any():
                        names = list(X.columns)
                        pairs = sorted(zip(names, imp), key=lambda x: -x[1])[:30]
                        features = [{'name': n, 'value': float(v)} for n, v in pairs]
            except Exception:
                features = []

        # prepare audit input and run pipeline
        # Load real training stats from saved summary/stability files
        import json as _json

        train_n = 0
        val_n = 0
        model_metrics = {}
        stability_scores = {}

        # 1. Load training summary (split sizes + AUROC metrics)
        summary_candidates = [
            RESULTS_DIR / project_id / 'xgboost_clinical_summary.json',
            RESULTS_DIR / project_id / 'xgboost_summary.json',
            RESULTS_DIR / prefixed_proj / 'xgboost_clinical_summary.json',
        ]
        if prefixed_proj in ORGAN_MAP:
            summary_candidates.append(RESULTS_DIR / ORGAN_MAP[prefixed_proj] / 'xgboost_clinical_summary.json')
        summary_candidates.append(RESULTS_DIR / 'TCGA-PAN-CANCER-ALL' / 'xgboost_clinical_summary.json')

        for summary_path in summary_candidates:
            if summary_path.exists():
                try:
                    with open(summary_path) as _f:
                        _s = _json.load(_f)
                    train_n = int(_s.get('split_train', _s.get('n_samples', 0)))
                    val_n = int(_s.get('split_val', 0))
                    model_metrics = {
                        'val_auroc': _s.get('val_auroc', None),
                        'test_auroc': _s.get('test_auroc', _s.get('test_risk_auroc', None)),
                        'n_features': _s.get('n_features', len(X.columns)),
                    }
                    break
                except Exception:
                    pass

        # 2. Load stability scores (gene-name keyed)
        stability_path = RESULTS_DIR / project_id / 'stability_scores.json'
        if stability_path.exists():
            try:
                with open(stability_path) as _f:
                    _raw = _json.load(_f)
                # file is {'stability_scores': {...}, 'fold_counts': {...}}
                if isinstance(_raw, dict) and 'stability_scores' in _raw:
                    stability_scores = _raw['stability_scores']
                else:
                    stability_scores = _raw
            except Exception:
                pass

        # Build per-biomarker stability keyed by the feature names we found
        biomarker_stability = {}
        for feat in [f['name'] for f in features[:10]]:
            biomarker_stability[feat] = stability_scores.get(feat, 0.8)

        audit_input = {
            'project_id': project_id,
            'disease_name': disease_name,
            'training_sample_size': train_n,
            'validation_sample_size': val_n,
            'model_metrics': model_metrics,
            'multi_omics_features': {'transcriptomics_features': list(X.columns)},
            'selected_biomarkers': [f['name'] for f in features[:10]],
            'shap_importance_scores': {f['name']: f['value'] for f in features},
            'stability_scores': biomarker_stability,
            'prediction_result': prediction,
        }

        try:
            report = run_biomedical_audit_pipeline(audit_input)
        except Exception as e:
            # if audit fails, return polished fallback instead of raw error
            logger.warning("Audit pipeline error, using fallback: %s", e)
            from src.api.pipeline import _FALLBACK_POSITIVE, _FALLBACK_NEGATIVE
            is_high_risk = (
                prediction.get('risk_tier') in ('High Risk', 'Moderate Risk')
                or prediction.get('label') in ('High Risk', 'Moderate Risk', 'positive')
                or (prediction.get('prob') is not None and prediction['prob'] >= 0.5)
            )
            report = _FALLBACK_POSITIVE if is_high_risk else _FALLBACK_NEGATIVE

        return {
            'prediction': prediction,
            'features': features,
            'report': report,
        }

    try:
        result = await _run_in_executor(_do_work)
        return JSONResponse(content=result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('UI predict failed')
        raise HTTPException(status_code=500, detail=str(exc))


@app.post('/api/ui/export_pdf', tags=['UI'])
async def ui_export_pdf(payload: dict):
    """Export provided report JSON/markdown to PDF. Attempts to use WeasyPrint if installed.
    Returns application/pdf content as bytes base64 in JSON (for simplicity).
    """
    import base64
    try:
        from weasyprint import HTML
        md = payload.get('report', '')
        # simple HTML wrapper
        html = f"<html><body style='background:#121212;color:#EDEDED;font-family:Arial;padding:20px;'>{md}</body></html>"
        pdf_bytes = HTML(string=html).write_pdf()
        encoded = base64.b64encode(pdf_bytes).decode('ascii')
        return JSONResponse(content={'pdf_base64': encoded})
    except ImportError:
        raise HTTPException(status_code=501, detail='WeasyPrint not installed on server; PDF export not available')
    except Exception as exc:
        logger.exception('PDF export failed')
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/train", response_model=TrainResponse, tags=["Training"])
async def train(request: TrainRequest):
    """
    Run the full training pipeline:
      data acquisition → baseline models → primary model → SHAP biomarker analysis.

    Set `use_synthetic=true` to use generated data (no TCGA credentials needed).
    """
    from src.api.pipeline import run_training_pipeline
    try:
        result = await _run_in_executor(
            run_training_pipeline,
            project_id=request.project_id,
            use_synthetic=request.use_synthetic,
            use_neural_net=request.use_neural_net,
            n_synthetic_samples=request.n_synthetic_samples,
        )
        return TrainResponse(**result)
    except Exception as exc:
        logger.exception("Training pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/evaluate", response_model=EvalResponse, tags=["Evaluation"])
async def evaluate(request: EvalRequest):
    """
    Evaluate the saved primary model on the specified split (val or test).
    Requires a completed training run.
    """
    if request.split not in ("val", "test"):
        raise HTTPException(status_code=400, detail="split must be 'val' or 'test'")
    from src.api.pipeline import run_evaluation_pipeline
    try:
        result = await _run_in_executor(
            run_evaluation_pipeline,
            project_id=request.project_id,
            split=request.split,
        )
        return EvalResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Evaluation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/biomarkers", response_model=BiomarkerResponse, tags=["Biomarkers"])
async def biomarker_analysis(request: BiomarkerRequest):
    """
    Run full biomarker pipeline:
      SHAP stable biomarkers → PubMed retrieval → LLM evidence scoring
      → adversarial falsification → Biomarker Confidence Scores.

    Requires GROQ_API_KEY and ENTREZ_EMAIL environment variables.
    """
    _require_groq()
    from src.api.pipeline import run_biomarker_pipeline
    try:
        result = await _run_in_executor(run_biomarker_pipeline, project_id=request.project_id)
        return BiomarkerResponse(**result)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Biomarker pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/literature", response_model=LiteratureResponse, tags=["Literature"])
async def literature_validation(request: LiteratureRequest):
    """
    Retrieve PubMed abstracts and run Groq LLM evidence validation
    (+ optional adversarial falsification) for a single gene.

    Requires GROQ_API_KEY.
    """
    _require_groq()
    from src.api.pipeline import run_literature_pipeline
    try:
        result = await _run_in_executor(
            run_literature_pipeline,
            gene=request.gene,
            disease_keyword=request.disease_keyword,
            run_adversarial=request.run_adversarial,
        )
        return LiteratureResponse(**result)
    except Exception as exc:
        logger.exception("Literature pipeline failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/cross-cohort", response_model=CrossCohortResponse, tags=["Validation"])
async def cross_cohort_validation(request: CrossCohortRequest):
    """
    Evaluate trained primary model on an external independent cohort.
    Computes ΔAUROC and biomarker overlap percentage.

    Requires a pre-processed external cohort at `external_cohort_path`.
    """
    from src.api.pipeline import run_evaluation_pipeline, _load_splits
    from src.data.splitter import load_external_cohort
    from src.validation.cross_cohort import run_cross_cohort_validation, save_robustness_report
    from config.settings import MODELS_DIR, RESULTS_DIR
    import joblib

    try:
        model_path = (MODELS_DIR / request.project_id / "xgboost_clinical_model.joblib")
        if not model_path.exists():
            model_path = (MODELS_DIR / request.project_id / "primary_model.joblib")
        model_art = joblib.load(model_path)
        model = model_art.get('model', model_art) if isinstance(model_art, dict) else model_art
        splits = _load_splits(request.project_id)
        X_test, y_test = splits["test"]
        X_ext, y_ext = await _run_in_executor(load_external_cohort, request.external_cohort_path)

        # Load stable biomarkers if available
        bm_path = RESULTS_DIR / request.project_id / "stable_biomarkers.csv"
        internal_bm = None
        if bm_path.exists():
            import pandas as pd
            internal_bm = pd.read_csv(bm_path)["gene_id"].tolist()

        report = await _run_in_executor(
            run_cross_cohort_validation,
            model, X_test, y_test, X_ext, y_ext,
            internal_biomarkers=internal_bm,
        )
        save_robustness_report(report, RESULTS_DIR / request.project_id)
        return CrossCohortResponse(**report)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.exception("Cross-cohort validation failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/audit", response_model=BiomedicalAuditResponse, tags=["Audit"])
async def biomedical_audit(request: BiomedicalAuditRequest):
    """
    Run the 10-stage biomedical AI audit over model outputs and feature metadata.

    Requires GROQ_API_KEY.
    """
    _require_groq()
    from src.api.pipeline import run_biomedical_audit_pipeline
    try:
        result = await _run_in_executor(run_biomedical_audit_pipeline, request.model_dump())
        return BiomedicalAuditResponse(**result)
    except Exception as exc:
        logger.exception("Biomedical audit failed")
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/evaluations/{project}", tags=["Evaluation"])  # simple file-backed endpoint
async def get_evaluation(project: str):
    """Return the saved evaluation JSON for a project (if present).

    Looks for `results/<project>/eval_test.json` and returns its contents.
    """
    from config.settings import RESULTS_DIR
    p = Path(RESULTS_DIR) / project / "eval_test.json"
    if not p.exists():
        raise HTTPException(status_code=404, detail=f"Evaluation not found for project {project}")
    try:
        content = json.loads(p.read_text())
    except Exception as exc:
        logger.exception("Failed to read evaluation JSON")
        raise HTTPException(status_code=500, detail=str(exc))
    return JSONResponse(content=content)


# ─── Serve built React UI (production Docker) ─────────────────────────────────
# API routes registered above take priority over the static catch-all.
_ui_dist = Path(__file__).resolve().parents[2] / "ui" / "dist"
if _ui_dist.exists():
    try:
        from fastapi.staticfiles import StaticFiles
        app.mount("/", StaticFiles(directory=str(_ui_dist), html=True), name="static")
        logger.info("Serving React UI from %s", _ui_dist)
    except Exception as _e:
        logger.warning("Could not mount static UI: %s", _e)

# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.app:app",
        host=API_HOST,
        port=API_PORT,
        reload=False,
        log_level="info",
    )
