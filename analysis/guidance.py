from __future__ import annotations

import pandas as pd


def analyze_guidance_accuracy(filing_texts: dict, historicals: pd.DataFrame) -> dict:
    """
    Extract prior management guidance and compare it to actual results where possible.
    """
    text = " ".join((filing_texts or {}).values()).lower()
    if "guidance" not in text and "outlook" not in text:
        return {"status": "insufficient data", "summary": "No reliable guidance language extracted.", "table": pd.DataFrame()}
    table = pd.DataFrame(
        [
            {
                "period": "Latest",
                "metric": "Revenue / margin guidance",
                "management_guide": "Guidance language found; numeric extraction requires manual review.",
                "actual_result": "See reported financial table.",
                "beat_miss": "manual review",
                "variance": None,
                "explanation": "Automated text extraction is intentionally conservative.",
                "credibility_impact": "Unknown",
            }
        ]
    )
    return {"status": "manual review", "summary": "Guidance language found, but numeric comparison needs manual validation.", "table": table}

