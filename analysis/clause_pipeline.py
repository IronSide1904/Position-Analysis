from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from config import CLAUSE_ENGINE_CONFIG, DEBUG_MODE
from analysis.clause_classifier import CLAUSE_TOPICS, classify_clause_topic
from analysis.clause_extractor import extract_candidate_clauses
from analysis.clause_model_mapper import assign_evidence_grade, map_clause_to_model_lines
from analysis.filing_section_splitter import split_filing_into_sections
from analysis.filing_text_cleaner import clean_filing_html

ENGINE_VERSION = "clause_engine_v1_1"
FINAL_COLUMNS = [
    "ticker",
    "cik",
    "form",
    "filing_date",
    "accession_number",
    "source_url",
    "section",
    "topic",
    "subtopic",
    "clause_text",
    "evidence_grade",
    "confidence",
    "model_line_affected",
    "direction",
    "timeframe",
    "suggested_assumption_change",
    "dashboard_action",
    "extraction_method",
    "review_status",
    "user_note",
]


def _empty(message: str = "Clause extraction unavailable") -> pd.DataFrame:
    df = pd.DataFrame(columns=FINAL_COLUMNS)
    df.attrs["warnings"] = [message]
    df.attrs["debug"] = {}
    return df


def _safe_name(value: str | None) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "unknown")).strip("_")
    return text or "unknown"


def _metadata_for_form(filing_metadata: dict, form: str, ticker: str, cik: str | None) -> dict:
    metadata = {"ticker": ticker, "cik": cik, "form": form}
    if isinstance(filing_metadata, dict):
        form_meta = filing_metadata.get(form) or filing_metadata.get(form.upper()) or {}
        metadata.update(form_meta if isinstance(form_meta, dict) else {})
        if "latest_filings" in filing_metadata:
            for item in filing_metadata.get("latest_filings") or []:
                if item.get("form") == form:
                    metadata.update(item)
                    break
    metadata["ticker"] = ticker
    metadata["cik"] = cik
    metadata["form"] = form
    metadata.setdefault("source_url", metadata.get("document_url") or metadata.get("filing_url"))
    return metadata


def _cache_path(metadata: dict) -> Path:
    cache_dir = Path(CLAUSE_ENGINE_CONFIG["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    filename = "_".join(
        [
            _safe_name(metadata.get("ticker")),
            _safe_name(metadata.get("form")),
            _safe_name(metadata.get("accession_number") or metadata.get("filing_date")),
            ENGINE_VERSION,
        ]
    )
    return cache_dir / f"{filename}_clauses.pkl"


def _read_cache(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        df = pd.read_pickle(path)
        if isinstance(df, pd.DataFrame):
            df.attrs["cache_hit"] = True
            return df
    except Exception:
        return None
    return None


def _write_cache(path: Path, df: pd.DataFrame) -> None:
    try:
        df.to_pickle(path)
    except Exception:
        pass


def _expand_candidate(candidate: dict) -> list[dict]:
    classification = classify_clause_topic(candidate.get("clause_text"), candidate.get("section"), candidate.get("topic"))
    mappings = map_clause_to_model_lines(classification["topic"], classification["subtopic"], candidate.get("clause_text"))
    evidence = assign_evidence_grade(candidate.get("clause_text"), candidate.get("section"), candidate.get("form"))
    rows = []
    for mapping in mappings:
        rows.append(
            {
                "ticker": candidate.get("ticker"),
                "cik": candidate.get("cik"),
                "form": candidate.get("form"),
                "filing_date": candidate.get("filing_date"),
                "accession_number": candidate.get("accession_number"),
                "source_url": candidate.get("source_url"),
                "section": candidate.get("section"),
                "topic": classification["topic"],
                "subtopic": classification["subtopic"],
                "clause_text": candidate.get("clause_text"),
                "evidence_grade": evidence,
                "confidence": classification["confidence"],
                "model_line_affected": mapping["model_line_affected"],
                "direction": mapping["direction"],
                "timeframe": mapping["timeframe"],
                "suggested_assumption_change": mapping["suggested_assumption_change"],
                "dashboard_action": mapping["dashboard_action"],
                "extraction_method": candidate.get("extraction_method"),
                "review_status": "Unreviewed",
                "user_note": "",
            }
        )
    return rows


def run_clause_extraction_pipeline(
    filing_texts: dict,
    filing_metadata: dict,
    ticker: str,
    cik: str | None = None,
) -> pd.DataFrame:
    """
    Main deterministic clause extraction pipeline.
    """
    if not CLAUSE_ENGINE_CONFIG["enabled"]:
        return _empty("Clause engine is disabled.")
    if not filing_texts:
        return _empty("No filing text available.")

    final_rows = []
    warnings_out = []
    debug = {"forms": {}, "engine_version": ENGINE_VERSION}
    for form, raw_text in (filing_texts or {}).items():
        metadata = _metadata_for_form(filing_metadata or {}, form, ticker, cik)
        path = _cache_path(metadata)
        cached = _read_cache(path)
        if cached is not None:
            final_rows.extend(cached.to_dict("records"))
            debug["forms"][form] = {"cache_hit": True, "final_clauses": len(cached)}
            continue

        cleaned = clean_filing_html(raw_text or "")
        if not cleaned:
            warnings_out.append(f"No filing text available for {form}.")
            continue
        sections = split_filing_into_sections(cleaned)
        warnings_out.extend(sections.get("_warnings", []))
        candidates = extract_candidate_clauses(
            sections,
            CLAUSE_TOPICS,
            metadata,
            max_clauses_per_topic=CLAUSE_ENGINE_CONFIG["max_clauses_per_topic"],
        )
        rows = []
        for candidate in candidates.to_dict("records"):
            rows.extend(_expand_candidate(candidate))
        df = pd.DataFrame(rows, columns=FINAL_COLUMNS)
        if not df.empty:
            df = df.drop_duplicates(subset=["form", "section", "clause_text", "model_line_affected"]).head(CLAUSE_ENGINE_CONFIG["max_total_clauses"])
        _write_cache(path, df)
        final_rows.extend(df.to_dict("records"))
        debug["forms"][form] = {
            "cache_hit": False,
            "filing_text_length": len(cleaned),
            "sections_detected": len([key for key in sections if not key.startswith("_")]),
            "candidate_clauses": len(candidates),
            "final_clauses": len(df),
        }

    if not final_rows:
        df = _empty("No relevant clauses found.")
    else:
        df = pd.DataFrame(final_rows, columns=FINAL_COLUMNS)
        df = df.drop_duplicates(subset=["form", "section", "clause_text", "model_line_affected"]).head(CLAUSE_ENGINE_CONFIG["max_total_clauses"])
        df.attrs["warnings"] = warnings_out
        df.attrs["debug"] = debug if DEBUG_MODE or CLAUSE_ENGINE_CONFIG.get("debug") else {}
    return df
