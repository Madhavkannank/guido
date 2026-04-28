"""
AI Provider Router
==================
Gemini is the primary biomedical reasoning provider.
GUIDO's existing Groq/LLM pipeline is the automatic fallback.

Fallback triggers:
  - GEMINI_API_KEY missing
  - Gemini request raises any exception
  - Gemini response is not valid JSON
  - Gemini response is missing required fields
  - confidence_score outside [0, 1]
"""
from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Required fields for Gemini biomedical audit responses
_REQUIRED_FIELDS = {
    "verdict",
    "confidence_score",
    "evidence_summary",
    "risk_explanation",
    "limitations",
    "cited_pubmed_ids",
}

_GEMINI_SYSTEM_PROMPT = """You are a biomedical research decision-support system.

CRITICAL RULES:
1. Use ONLY the supplied PubMed abstracts/evidence provided in the user message.
2. Do NOT invent citations or PMIDs not present in the supplied abstracts.
3. If evidence is missing or insufficient, set verdict to "insufficient_evidence".
4. Do NOT give diagnosis or treatment advice. This is research decision-support only.
5. Respond with VALID JSON only — no prose, no markdown fences.

Required JSON output schema:
{
  "verdict": "supported" | "weak" | "contradicted" | "insufficient_evidence",
  "confidence_score": <float 0.0-1.0>,
  "evidence_summary": "<string>",
  "risk_explanation": "<string>",
  "limitations": "<string>",
  "cited_pubmed_ids": ["<pmid>", ...],
  "provider_used": "gemini",
  "fallback_reason": null
}
"""


def _get_gemini_key() -> str:
    return os.getenv("GEMINI_API_KEY", "")


def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    text = re.sub(r"```(?:json)?\s*", "", text).strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group())
    except json.JSONDecodeError:
        return None


def _validate_gemini_output(data: Dict[str, Any]) -> Optional[str]:
    """Return a reason string if invalid, else None."""
    missing = _REQUIRED_FIELDS - data.keys()
    if missing:
        return f"missing fields: {missing}"
    cs = data.get("confidence_score")
    try:
        if not (0.0 <= float(cs) <= 1.0):
            return f"confidence_score out of range: {cs}"
    except (TypeError, ValueError):
        return f"confidence_score not numeric: {cs}"
    return None


def _call_gemini(prompt: str) -> Optional[Dict[str, Any]]:
    """Call Gemini API. Returns parsed dict or None on any failure."""
    api_key = _get_gemini_key()
    if not api_key:
        logger.info("GEMINI_API_KEY not set — skipping Gemini")
        return None

    try:
        from google import genai  # type: ignore
        from google.genai import types  # type: ignore

        client = genai.Client(api_key=api_key)
        try:
            model_name = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash")
            # ensure models/ prefix
            if not model_name.startswith("models/"):
                model_name = "models/" + model_name
            response = client.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=_GEMINI_SYSTEM_PROMPT,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()
            return _extract_json(raw)
        finally:
            # Always close the client to prevent connection leaks
            try:
                client.close()
            except Exception:
                pass
    except Exception as exc:
        logger.warning("Gemini call failed: %s", exc)
        return None


def _wrap_with_metadata(
    result: Dict[str, Any],
    provider: str,
    fallback_reason: Optional[str],
) -> Dict[str, Any]:
    result["provider_used"] = provider
    result["fallback_reason"] = fallback_reason
    return result


# ─── Public router functions ──────────────────────────────────────────────────

def run_biomedical_audit(
    audit_input: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Run biomedical audit via Gemini (primary) → GUIDO Groq pipeline (fallback).

    Returns the audit dict with added metadata:
      provider_used: "gemini" | "guido_fallback"
      fallback_reason: null | reason string
    """
    fallback_reason: Optional[str] = None

    # ── Try Gemini ────────────────────────────────────────────────────────────
    if _get_gemini_key():
        prompt = (
            "Run a biomedical audit for the following input and return JSON "
            "matching the required schema.\n\n"
            f"{json.dumps(audit_input, ensure_ascii=False)}"
        )
        gemini_result = _call_gemini(prompt)
        if gemini_result is not None:
            invalid_reason = _validate_gemini_output(gemini_result)
            if invalid_reason is None:
                return _wrap_with_metadata(gemini_result, "gemini", None)
            fallback_reason = f"gemini_invalid_output: {invalid_reason}"
            logger.warning("Gemini output invalid (%s) — falling back", fallback_reason)
        else:
            fallback_reason = "gemini_call_failed"
    else:
        fallback_reason = "gemini_api_key_missing"

    # ── Fallback: GUIDO existing pipeline ─────────────────────────────────────
    logger.info("Using GUIDO fallback (reason: %s)", fallback_reason)
    from src.llm.groq_validator import run_biomedical_system_audit
    result = run_biomedical_system_audit(audit_input)
    return _wrap_with_metadata(result, "guido_fallback", fallback_reason)


def run_biomarker_confidence_score(
    gene: str,
    disease_keyword: str,
    abstracts: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """
    Score biomarker evidence via Gemini (primary) → Groq validate_biomarker_evidence (fallback).
    """
    fallback_reason: Optional[str] = None

    if _get_gemini_key():
        from src.llm.groq_validator import _format_abstracts
        abstract_text = _format_abstracts(abstracts)
        prompt = (
            f"Gene: {gene}\nDisease: {disease_keyword}\n\n"
            f"PubMed Abstracts:\n{abstract_text}\n\n"
            "Evaluate evidence for this gene using ONLY the abstracts above. "
            "Return JSON matching the required schema."
        )
        gemini_result = _call_gemini(prompt)
        if gemini_result is not None:
            invalid_reason = _validate_gemini_output(gemini_result)
            if invalid_reason is None:
                gemini_result["gene"] = gene
                return _wrap_with_metadata(gemini_result, "gemini", None)
            fallback_reason = f"gemini_invalid_output: {invalid_reason}"
            logger.warning("Gemini output invalid (%s) — falling back", fallback_reason)
        else:
            fallback_reason = "gemini_call_failed"
    else:
        fallback_reason = "gemini_api_key_missing"

    logger.info("Using GUIDO fallback for gene %s (reason: %s)", gene, fallback_reason)
    from src.llm.groq_validator import validate_biomarker_evidence
    result = validate_biomarker_evidence(gene, disease_keyword, abstracts)
    if result is not None:
        result = _wrap_with_metadata(result, "guido_fallback", fallback_reason)
    return result


def run_adversarial_falsification(
    gene: str,
    disease_keyword: str,
    abstracts: List[Dict[str, str]],
) -> Optional[Dict[str, Any]]:
    """
    Adversarial falsification via Gemini (primary) → Groq adversarial_falsify (fallback).
    """
    fallback_reason: Optional[str] = None

    if _get_gemini_key():
        from src.llm.groq_validator import _format_abstracts
        abstract_text = _format_abstracts(abstracts)
        prompt = (
            f"Gene: {gene}\nDisease: {disease_keyword}\n\n"
            f"PubMed Abstracts:\n{abstract_text}\n\n"
            "Perform adversarial analysis — identify every methodological weakness, "
            "contradictory finding, or correlation-only claim. "
            "Return JSON matching the required schema."
        )
        gemini_result = _call_gemini(prompt)
        if gemini_result is not None:
            invalid_reason = _validate_gemini_output(gemini_result)
            if invalid_reason is None:
                gemini_result["gene"] = gene
                return _wrap_with_metadata(gemini_result, "gemini", None)
            fallback_reason = f"gemini_invalid_output: {invalid_reason}"
            logger.warning("Gemini output invalid (%s) — falling back", fallback_reason)
        else:
            fallback_reason = "gemini_call_failed"
    else:
        fallback_reason = "gemini_api_key_missing"

    logger.info("Using GUIDO fallback for adversarial gene %s (reason: %s)", gene, fallback_reason)
    from src.llm.groq_validator import adversarial_falsify
    result = adversarial_falsify(gene, disease_keyword, abstracts)
    if result is not None:
        result = _wrap_with_metadata(result, "guido_fallback", fallback_reason)
    return result
