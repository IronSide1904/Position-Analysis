from __future__ import annotations

import re

from bs4 import BeautifulSoup


def normalize_filing_text(text: str) -> str:
    """
    Normalize SEC filing text while preserving headings and sentence boundaries.
    """
    if not text:
        return ""
    replacements = {
        "\xa0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
    }
    out = str(text)
    for old, new in replacements.items():
        out = out.replace(old, new)
    out = re.sub(r"[ \t]+", " ", out)
    out = re.sub(r" *\n *", "\n", out)
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out.strip()


def clean_filing_html(raw_html_or_text: str) -> str:
    """
    Clean SEC filing HTML/text into readable plain text.
    """
    if not raw_html_or_text:
        return ""
    raw = str(raw_html_or_text)
    if re.search(r"<[a-zA-Z][^>]*>", raw[:5000]):
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n")
    else:
        text = raw
    return normalize_filing_text(text)


def _mostly_numeric(text: str) -> bool:
    compact = re.sub(r"\s+", "", text or "")
    if not compact:
        return True
    numeric = sum(1 for char in compact if char.isdigit() or char in "$%,.()-")
    return numeric / max(len(compact), 1) > 0.72


def split_into_paragraphs(text: str) -> list[str]:
    """
    Split clean text into paragraphs and ignore short/numeric junk.
    """
    clean = normalize_filing_text(text)
    parts = re.split(r"\n{2,}|(?<=[.!?])\s+(?=[A-Z(])", clean)
    paragraphs = []
    for part in parts:
        paragraph = re.sub(r"\s+", " ", part).strip()
        if len(paragraph) < 40 or _mostly_numeric(paragraph):
            continue
        paragraphs.append(paragraph)
    return paragraphs
