"""
Step 5 (Part 1) – PubMed Literature Retrieval
===============================================
Uses Biopython's Entrez module to query PubMed and retrieve up to
PUBMED_MAX_ABSTRACTS abstracts per (gene, disease) keyword pair.

Abstracts are cached locally in JSON format to avoid repeated API calls.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

try:
    from Bio import Entrez, Medline
    _BIO_AVAILABLE = True
except ImportError:
    Entrez = None
    Medline = None
    _BIO_AVAILABLE = False

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from config.settings import (
    ENTREZ_API_KEY,
    ENTREZ_EMAIL,
    PUBMED_MAX_ABSTRACTS,
    RESULTS_DIR,
    TCGA_PROJECT_ID,
)

logger = logging.getLogger(__name__)

if _BIO_AVAILABLE and Entrez is not None:
    Entrez.email = ENTREZ_EMAIL
    if ENTREZ_API_KEY:
        Entrez.api_key = ENTREZ_API_KEY

CACHE_DIR = RESULTS_DIR / "pubmed_cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Disease keyword inferred from project ID (can be overridden)
_PROJECT_DISEASE_MAP = {
    # 33 Official TCGA Cancer Cohorts
    "TCGA-ACC":  "adrenocortical carcinoma",
    "TCGA-BLCA": "bladder urothelial carcinoma",
    "TCGA-BRCA": "breast cancer",
    "TCGA-CESC": "cervical squamous cell carcinoma",
    "TCGA-CHOL": "cholangiocarcinoma",
    "TCGA-COAD": "colorectal cancer colon adenocarcinoma",
    "TCGA-DLBC": "diffuse large B-cell lymphoma",
    "TCGA-ESCA": "esophageal cancer",
    "TCGA-GBM":  "glioblastoma multiforme",
    "TCGA-HNSC": "head and neck squamous cell carcinoma",
    "TCGA-KICH": "kidney chromophobe",
    "TCGA-KIRC": "clear cell renal cell carcinoma",
    "TCGA-KIRP": "papillary renal cell carcinoma",
    "TCGA-LAML": "acute myeloid leukemia",
    "TCGA-LGG":  "lower grade glioma brain tumor",
    "TCGA-LIHC": "liver hepatocellular carcinoma",
    "TCGA-LUAD": "lung adenocarcinoma",
    "TCGA-LUSC": "lung squamous cell carcinoma",
    "TCGA-MESO": "malignant mesothelioma",
    "TCGA-OV":   "ovarian serous cystadenocarcinoma",
    "TCGA-PAAD": "pancreatic ductal adenocarcinoma",
    "TCGA-PCPG": "pheochromocytoma and paraganglioma",
    "TCGA-PRAD": "prostate adenocarcinoma",
    "TCGA-READ": "rectum adenocarcinoma",
    "TCGA-SARC": "sarcoma",
    "TCGA-SKCM": "cutaneous melanoma",
    "TCGA-STAD": "stomach gastric adenocarcinoma",
    "TCGA-TGCT": "testicular germ cell tumor",
    "TCGA-THCA": "thyroid carcinoma",
    "TCGA-THYM": "thymoma",
    "TCGA-UCEC": "uterine corpus endometrial carcinoma",
    "TCGA-UCS":  "uterine carcinosarcoma",
    "TCGA-UVM":  "uveal melanoma",
    # Pan-Organ Groupings
    "PAN-RENAL":        "renal cell carcinoma kidney cancer",
    "PAN-LUNG":         "non-small cell lung cancer",
    "PAN-GI":           "gastrointestinal colorectal gastric cancer",
    "PAN-HEPATOBILIARY": "hepatobiliary liver hepatocellular carcinoma",
    "PAN-BREAST":       "breast cancer",
    "PAN-BRAIN":        "glioma glioblastoma brain tumor",
    "PAN-GYN":          "gynecologic ovarian endometrial cervical cancer",
    "PAN-UROLOGICAL":   "urological bladder prostate cancer",
    "PAN-HEADNECK":     "head and neck cancer",
    "PAN-ENDOCRINE":    "endocrine thyroid adrenocortical carcinoma",
    "PAN-SARCOMA":      "sarcoma soft tissue bone",
    "PAN-MELANOMA":     "cutaneous uveal melanoma",
    "PAN-HEMATOLOGIC":  "hematologic leukemia lymphoma thymoma",
    "PAN-MESO":         "malignant mesothelioma",
}

# Hallmark pathway search terms for composite metagenes
_METAGENE_SEARCH_TERMS = {
    "META_PROLIFERATION"        : ("MKI67 OR PCNA", "cancer cell proliferation mitosis"),
    "META_EMT_INVASION"         : ("VIM OR CDH2", "epithelial mesenchymal transition metastasis"),
    "META_ANGIOGENESIS"         : ("VEGFA OR KDR", "tumor angiogenesis neovascularization"),
    "META_HYPOXIA"              : ("HIF1A OR LDHA", "tumor hypoxia microenvironment"),
    "META_GLYCOLYSIS_WARBURG"   : ("PKM OR GAPDH", "Warburg effect aerobic glycolysis"),
    "META_IMMUNE_EXHAUST"       : ("CD274 OR PDCD1", "T cell immune exhaustion immunotherapy checkpoint"),
    "META_STEMNESS_PLURIPOTENCY": ("SOX2 OR POU5F1", "cancer stem cells pluripotency"),
    "META_DNA_REPAIR"           : ("PARP1 OR BRCA1", "DNA repair homologous recombination deficiency"),
    "META_APOPTOSIS_EVASION"    : ("BCL2 OR MCL1", "apoptosis evasion anti-apoptotic"),
    "META_INFLAMMATION_IL6"     : ("IL6 OR STAT3", "tumor-promoting inflammation IL-6 cytokine"),
    "META_EXTRACELLULAR_MATRIX" : ("COL1A1 OR LOX", "extracellular matrix remodeling desmoplasia"),
    "META_OXIDATIVE_STRESS_NRF2": ("NFE2L2 OR HMOX1", "NRF2 antioxidant oxidative stress response"),
}


def get_disease_keyword(project_id: str = TCGA_PROJECT_ID) -> str:
    proj = project_id.strip().upper()
    if proj in _PROJECT_DISEASE_MAP:
        return _PROJECT_DISEASE_MAP[proj]
    # Check with or without TCGA- prefix
    prefixed = f"TCGA-{proj}" if not proj.startswith("TCGA-") else proj
    unprefixed = proj.replace("TCGA-", "").replace("TCGA.", "")
    return _PROJECT_DISEASE_MAP.get(prefixed, _PROJECT_DISEASE_MAP.get(unprefixed, "cancer"))


def _cache_file(gene: str, disease: str) -> Path:
    safe_gene = gene.replace("/", "_")
    safe_disease = disease.replace(" ", "_")
    return CACHE_DIR / f"{safe_gene}_{safe_disease}.json"


def fetch_pubmed_abstracts(
    gene: str,
    disease_keyword: str,
    max_results: int = PUBMED_MAX_ABSTRACTS,
) -> List[Dict[str, str]]:
    """
    Search PubMed for `gene disease_keyword` and return up to max_results
    structured abstract records. Handles both individual genes and hallmark metagenes.

    Each record contains:
        pmid, title, abstract, authors, journal, year

    Results are cached locally to disk.
    """
    cache = _cache_file(gene, disease_keyword)
    if cache.exists():
        logger.debug("Cache hit: %s", cache)
        return json.loads(cache.read_text(encoding="utf-8"))

    # Resolve metagene queries or standard individual gene queries
    if not _BIO_AVAILABLE:
        logger.warning("Biopython ('Bio') is not installed. Live PubMed queries disabled.")
        return []

    if gene in _METAGENE_SEARCH_TERMS:
        genes_term, pathway_term = _METAGENE_SEARCH_TERMS[gene]
        query = f"({genes_term}[Title/Abstract] OR {pathway_term}[Title/Abstract]) AND {disease_keyword}[Title/Abstract]"
    else:
        query = f"{gene}[Gene Name] AND {disease_keyword}[Title/Abstract]"

    try:
        # Search
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="relevance")
        record = Entrez.read(handle)
        handle.close()
        pmids: List[str] = record.get("IdList", [])

        if not pmids:
            logger.info("No PubMed results for gene=%s, disease=%s", gene, disease_keyword)
            cache.write_text(json.dumps([]), encoding="utf-8")
            return []

        time.sleep(0.35)  # NCBI rate limit: 3 requests/sec without API key

        # Fetch record details
        handle = Entrez.efetch(db="pubmed", id=",".join(pmids), rettype="medline", retmode="text")
        records = list(Medline.parse(handle))
        handle.close()

        abstracts = []
        for rec in records:
            abstracts.append({
                "pmid":     rec.get("PMID", ""),
                "title":    rec.get("TI",   ""),
                "abstract": rec.get("AB",   ""),
                "authors":  ", ".join(rec.get("AU", [])),
                "journal":  rec.get("TA",   ""),
                "year":     rec.get("DP",   "")[:4] if rec.get("DP") else "",
            })

        cache.write_text(json.dumps(abstracts, ensure_ascii=False), encoding="utf-8")
        logger.info("Fetched %d abstracts for gene=%s", len(abstracts), gene)
        return abstracts

    except Exception as exc:
        logger.warning("PubMed fetch failed for gene=%s: %s", gene, exc)
        return []


def fetch_all_biomarkers(
    gene_list: List[str],
    disease_keyword: Optional[str] = None,
    project_id: str = TCGA_PROJECT_ID,
) -> Dict[str, List[Dict[str, str]]]:
    """
    Retrieve PubMed abstracts for all genes in gene_list.

    Returns
    -------
    {gene_id: [abstract_records]}
    """
    if disease_keyword is None:
        disease_keyword = get_disease_keyword(project_id)

    results: Dict[str, List[Dict[str, str]]] = {}
    for gene in gene_list:
        results[gene] = fetch_pubmed_abstracts(gene, disease_keyword)
        time.sleep(0.1)  # gentle throttle

    return results
