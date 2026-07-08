from __future__ import annotations

import hashlib
import re

import pandas as pd

from config import CLAUSE_ENGINE_CONFIG
from analysis.filing_text_cleaner import split_into_paragraphs


def find_keyword_hits(section_text: str, topic_keywords: dict) -> list[dict]:
    """
    Find topic keyword hits inside section text.
    """
    hits = []
    text = section_text or ""
    for topic, keywords in topic_keywords.items():
        for keyword in keywords:
            pattern = re.compile(rf"\b{re.escape(keyword)}\b", re.IGNORECASE)
            for match in pattern.finditer(text):
                hits.append({"topic": topic, "keyword": keyword, "start": match.start(), "end": match.end()})
    return sorted(hits, key=lambda item: item["start"])


def extract_clause_window(section_text: str, hit_start: int, hit_end: int, context_window_chars: int = 700) -> str:
    """
    Extract a clause around a keyword hit, preferring paragraph boundaries.
    """
    text = section_text or ""
    if not text:
        return ""
    start = max(0, hit_start - context_window_chars)
    end = min(len(text), hit_end + context_window_chars)
    left = max(text.rfind("\n\n", 0, hit_start), text.rfind(". ", 0, hit_start), start)
    right_candidates = [pos for pos in [text.find("\n\n", hit_end), text.find(". ", hit_end)] if pos != -1]
    right = min(right_candidates) + 1 if right_candidates else end
    if right <= left:
        left, right = start, end
    return re.sub(r"\s+", " ", text[left:right]).strip()


def _too_numeric(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    numeric = sum(1 for char in compact if char.isdigit() or char in "$%,.()-")
    return numeric / max(len(compact), 1) > 0.68


def _metadata_value(metadata: dict, key: str):
    return metadata.get(key) if isinstance(metadata, dict) else None


def extract_candidate_clauses(
    sections: dict,
    topic_keywords: dict,
    metadata: dict,
    max_clauses_per_topic: int = 20,
) -> pd.DataFrame:
    """
    Extract candidate clauses from relevant sections.
    """
    rows = []
    seen = set()
    per_topic: dict[str, int] = {}
    max_chars = CLAUSE_ENGINE_CONFIG["max_clause_chars"]
    min_chars = CLAUSE_ENGINE_CONFIG["min_clause_chars"]
    context = CLAUSE_ENGINE_CONFIG["context_window_chars"]
    for section, section_text in (sections or {}).items():
        if section.startswith("_") or not section_text:
            continue
        hits = find_keyword_hits(section_text, topic_keywords)
        if not hits and section == "Unknown":
            for paragraph in split_into_paragraphs(section_text)[:200]:
                hits.extend(find_keyword_hits(paragraph, topic_keywords))
        for hit in hits:
            if per_topic.get(hit["topic"], 0) >= max_clauses_per_topic:
                continue
            clause = extract_clause_window(section_text, hit["start"], hit["end"], context)
            clause = clause[:max_chars].strip()
            if len(clause) < min_chars or _too_numeric(clause):
                continue
            digest = hashlib.sha1(re.sub(r"\s+", " ", clause.lower()).encode("utf-8")).hexdigest()
            if digest in seen:
                continue
            seen.add(digest)
            per_topic[hit["topic"]] = per_topic.get(hit["topic"], 0) + 1
            rows.append(
                {
                    "ticker": _metadata_value(metadata, "ticker"),
                    "cik": _metadata_value(metadata, "cik"),
                    "form": _metadata_value(metadata, "form"),
                    "filing_date": _metadata_value(metadata, "filing_date"),
                    "accession_number": _metadata_value(metadata, "accession_number"),
                    "source_url": _metadata_value(metadata, "source_url") or _metadata_value(metadata, "document_url"),
                    "section": section,
                    "topic": hit["topic"],
                    "keyword": hit["keyword"],
                    "clause_text": clause,
                    "raw_start": hit["start"],
                    "raw_end": hit["end"],
                    "extraction_method": "keyword_window",
                }
            )
            if len(rows) >= CLAUSE_ENGINE_CONFIG["max_total_clauses"]:
                break
        if len(rows) >= CLAUSE_ENGINE_CONFIG["max_total_clauses"]:
            break
    columns = [
        "ticker",
        "cik",
        "form",
        "filing_date",
        "accession_number",
        "source_url",
        "section",
        "topic",
        "keyword",
        "clause_text",
        "raw_start",
        "raw_end",
        "extraction_method",
    ]
    return pd.DataFrame(rows, columns=columns)
