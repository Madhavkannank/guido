---
title: GUIDO
emoji: 🧬
colorFrom: green
colorTo: gray
sdk: docker
app_port: 7860
pinned: false
---

#  GUIDO — Multi-Omics Disease Risk Modeling System

A **production-ready, LLM-guided multi-omics disease risk pipeline** with a full React UI.

-  XGBoost performs statistical risk prediction across 13 TCGA cancer cohorts
-  Groq LLM performs structured, abstract-grounded biological reasoning
-  SHAP biomarker analysis with stability scoring across CV folds
-  React + Vite frontend served directly from the FastAPI container
-  Single-container Docker deployment (API + UI on one port)
-  No hallucinations — LLM receives only provided PubMed abstracts
-  No statistical shortcuts — gate check enforces model vs. baseline comparison

---

##  Trained Model Performance

All models trained on real TCGA data via UCSC Xena:

| Cancer | TCGA Project | Val AUROC | Test AUROC | Train N |
|--------|-------------|-----------|------------|---------|
| Breast | TCGA-BRCA | 1.000 | 1.000 | 852 |
| Lung Adenocarcinoma | TCGA-LUAD | 1.000 | 1.000 | 403 |
| Lung Squamous | TCGA-LUSC | 1.000 | 1.000 | 387 |
| Kidney Clear Cell | TCGA-KIRC | 1.000 | 1.000 | 424 |
| Kidney Papillary | TCGA-KIRP | 1.000 | 1.000 | 226 |
| Colon Adenocarcinoma | TCGA-COAD | 0.996 | 1.000 | 230 |
| Glioblastoma | TCGA-GBM | 1.000 | 1.000 | 120 |
| Kidney Chromophobe | TCGA-KICH | 1.000 | 1.000 | 63 |
| Liver | TCGA-LIHC | 1.000 | 0.993 | 296 |
| LGG, SKCM, PAAD, PRAD | (see results/) | — | — | — |

> Models stored as `models_artifacts/TCGA-{PROJECT}/xgboost_clinical_model.joblib`  
> Each artifact is a dict: `{model: XGBClassifier, scaler: StandardScaler, features: [...]}`

---

##  Architecture Overview

```
TCGA Data (UCSC Xena)
        
        
Preprocessing (log2-TPM, top HVGs)
        
        
XGBoost Clinical Model (5-fold CV)
        
         StandardScaler
         Feature alignment (embedded feature list)
         SHAP biomarker stability scoring
        
        
FastAPI  (/api/ui/predict)
        
         Prediction + probability
         Top-30 feature importances
         10-stage Biomedical AI Audit (Groq LLM)
                
                 PubMed abstract retrieval (NCBI Entrez)
                 LLM evidence scoring
                 Adversarial falsification
                 Biomarker Confidence Score (BCS)
        
        
React UI  (served at / from same container)
         Prediction form (CSV or JSON upload)
         Feature importance chart (Recharts)
         AI audit report viewer
         Export (PDF via browser print / JSON download)
        
        
MLflow tracking (optional, port 5000)
```

---

##  Project Structure

```
guido/
 src/
    api/
       app.py               # FastAPI + StaticFiles (serves React UI at /)
       pipeline.py          # Training, audit, biomarker pipelines
       schemas.py           # Pydantic request/response models
    data/
       acquisition.py       # GDC download
       xena_acquisition.py  # UCSC Xena download
       splitter.py
       synthetic_data.py
    models/
       baselines.py
       primary_model.py
       confidence_scoring.py
    shap_analysis/shap_runner.py
    literature/pubmed_retrieval.py
    llm/groq_validator.py
    validation/cross_cohort.py

 ui/                           # React + Vite frontend
    src/
       App.jsx               # Root + ErrorBoundary + footer
       components/
           PredictionForm.jsx
           PredictionResult.jsx
           FeatureChart.jsx
           ReportViewer.jsx
           ExportPanel.jsx   # Client-side PDF + JSON export
           Dashboard.jsx
           Navbar.jsx
           TypingTitle.jsx
    public/                   # logo.svg, quad-bl.jpg, fonts/
    .env                      # VITE_API_BASE=http://127.0.0.1:8000 (dev)
    .env.production           # VITE_API_BASE= (empty  relative URLs in Docker)
    .env.example

 models_artifacts/             # Trained .joblib files (gitignored)
    TCGA-{PROJECT}/
        xgboost_clinical_model.joblib
        xgboost_clinical_summary.json

 results/                      # Per-project results (gitignored)
    TCGA-{PROJECT}/
        stability_scores.json
        stable_biomarkers.csv
        patient_reports/

 config/settings.py
 Dockerfile                    # 3-stage: node UI build  python deps  runtime
 docker-compose.yml            # api + mlflow services
 .dockerignore
 .env.example
 requirements.txt
 run_pipeline.py
```

---

##  Local Development

### 1. Clone & set up environment

```bash
git clone https://github.com/YOUR_USERNAME/guido.git
cd guido
cp .env.example .env
```

Edit `.env`:
```
GROQ_API_KEY=gsk_...           # from console.groq.com
ENTREZ_EMAIL=your@email.com    # for PubMed retrieval
```

### 2. Install Python dependencies

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
```

### 3. Start the backend

```bash
python -m uvicorn src.api.app:app --host 127.0.0.1 --port 8000
```

### 4. Start the frontend (separate terminal)

```bash
cd ui
npm install
npm run dev
#  http://localhost:5173
```

> The Vite dev server proxies `/api/*` to `http://127.0.0.1:8000` automatically.  
> No CORS issues in local dev.

### 5. API docs

```
http://localhost:8000/docs
```

---

##  Docker (Production)

Single command — builds the React UI, bundles it into the FastAPI container, serves everything on port 8000:

```bash
docker compose up --build
```

- **App (UI + API)**  http://localhost:8000
- **MLflow**  http://localhost:5000

Secrets are injected at runtime via `env_file: .env` — **never baked into the image**.

### Environment variables

```
GROQ_API_KEY=gsk_...
GROQ_MODEL=llama3-70b-8192
ENTREZ_EMAIL=your@email.com
API_HOST=0.0.0.0
API_PORT=8000
UI_ALLOWED_ORIGINS=https://your-domain.com
N_TOP_GENES=500
RANDOM_SEED=42
```

---

##  Free Deployment (Hugging Face Spaces)

Free forever — 16GB RAM, Docker supported, no credit card.

1. Create Space at [huggingface.co/spaces](https://huggingface.co/spaces)  Docker  Public
2. Set Space secrets: `GROQ_API_KEY`, `ENTREZ_EMAIL`, `PORT=7860`
3. Push:
   ```bash
   git remote add hf https://huggingface.co/spaces/YOUR_USERNAME/guido
   git push hf main
   ```

### Render (~$7/month)

1. New Web Service  Docker  connect GitHub repo
2. Set env vars in dashboard
3. Port: `8000` (the Dockerfile CMD uses `${PORT:-8000}`)

---

##  API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Liveness check |
| POST | `/api/ui/predict` | **Main UI endpoint** — CSV/JSON  prediction + audit |
| POST | `/train` | Full training pipeline |
| POST | `/evaluate` | Evaluate saved model (val or test split) |
| POST | `/biomarkers` | PubMed + LLM scoring + BCS |
| POST | `/literature` | Single-gene literature validation |
| POST | `/cross-cohort` | External cohort robustness check |
| POST | `/audit` | 10-stage biomedical AI audit |
| GET | `/evaluations/{project}` | Saved evaluation results |

---

##  10-Stage Biomedical AI Audit

Runs automatically on every `/api/ui/predict` call:

| Stage | Check | Pass Condition |
|-------|-------|----------------|
| 1 | Data quality | train_n  100 |
| 2 | Feature selection | SHAP-based, stability  3 folds |
| 3 | Model performance | Val AUROC  0.75 |
| 4 | Biomarker stability |  50% of top biomarkers stable |
| 5 | Literature evidence | PubMed abstracts retrieved |
| 6 | LLM evidence scoring | BCS computed |
| 7 | Adversarial check | Vulnerability scored |
| 8 | SHAP consistency | Feature importances non-zero |
| 9 | Reproducibility | Fixed seed + pinned deps |
| 10 | Deployment safety | Non-root Docker, healthcheck |

**Verdict:** `APPROVED` / `CONDITIONALLY_APPROVED` / `REJECTED`

---

##  Biomarker Confidence Score (BCS)

```
BCS = S  ((E_raw - 1) / 4)  (1 - V)
```

- **S** = Stability score  [0, 1] — fraction of CV folds where gene appears
- **E_raw** = Evidence strength (1–5) from LLM-graded PubMed abstracts
- **V** = Adversarial vulnerability  [0, 1]

 BCS  0.30  high-confidence biomarker

---

##  LLM Safety Constraints

- LLM receives **only provided PubMed abstracts** — no prior knowledge used
- System prompt explicitly forbids extrapolation beyond provided text
- Strict JSON output schema (max 3 retries with validation)
- Adversarial mode independently scores falsifiability
- **LLM never makes predictions** — only explains ML model outputs

---

##  Training Pipeline

| Step | Script | Description |
|------|--------|-------------|
| 1 | `xena_acquisition.py` | Download TCGA RNA-seq from UCSC Xena |
| 2 | `splitter.py` | Stratified train/val/test split |
| 3 | `train_with_clinical.py` | XGBoost + StandardScaler + stability scoring |
| 4 | `shap_runner.py` | CV-SHAP  stable biomarkers ( 3 folds) |
| 5 | `pubmed_retrieval.py` | Abstract retrieval via NCBI Entrez |
| 6 | `groq_validator.py` | Evidence scoring + adversarial falsification |
| 7 | `confidence_scoring.py` | BCS computation |
| 8 | `cross_cohort.py` | ΔAUROC + bootstrap CI |
| 9 | `app.py` | FastAPI REST + React UI serving |
| 10 | Dockerfile | Multi-stage build  single container |

---

##  Supported Cancer Cohorts

| Project ID | Cancer Type |
|------------|------------|
| TCGA-BRCA | Breast Cancer |
| TCGA-LUAD | Lung Adenocarcinoma |
| TCGA-LUSC | Lung Squamous Cell Carcinoma |
| TCGA-KIRC | Kidney Clear Cell Carcinoma |
| TCGA-KIRP | Kidney Papillary Carcinoma |
| TCGA-KICH | Kidney Chromophobe |
| TCGA-COAD | Colon Adenocarcinoma |
| TCGA-LGG | Brain Lower Grade Glioma |
| TCGA-GBM | Glioblastoma Multiforme |
| TCGA-LIHC | Liver Hepatocellular Carcinoma |
| TCGA-PAAD | Pancreatic Adenocarcinoma |
| TCGA-SKCM | Skin Cutaneous Melanoma |
| TCGA-PRAD | Prostate Adenocarcinoma |

---

##  Tests

```bash
pytest tests/ -v
```

---

##  Security Notes

- **Never commit `.env`** — it is gitignored
- Copy `.env.example`  `.env` and fill in real values
- In Docker, secrets are injected at runtime via `env_file:` — not baked into image
- `models_artifacts/` and `data/` are gitignored (large files, use Docker volumes)
- Rotate your `GROQ_API_KEY` at [console.groq.com](https://console.groq.com) if ever exposed
#
