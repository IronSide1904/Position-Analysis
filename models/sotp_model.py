from __future__ import annotations

import pandas as pd


def run_sotp(segment_data: pd.DataFrame, assumptions: dict) -> dict:
    """
    Value segments separately using revenue, margin, NOPAT/OCF proxy, and selected multiple.
    """
    if segment_data is None or segment_data.empty:
        return {
            "available": False,
            "segment_table": pd.DataFrame(),
            "enterprise_value": None,
            "summary": "Manual segment assumptions required; SEC segment data is unavailable.",
        }
    rows = []
    total = 0.0
    for _, row in segment_data.iterrows():
        name = row.get("segment", "Segment")
        revenue = float(row.get("revenue", 0) or 0)
        margin = float(row.get("margin", assumptions.get("default_margin", 0.15)) or 0)
        multiple = float(row.get("multiple", assumptions.get("default_multiple", 12.0)) or 0)
        value = revenue * margin * multiple
        total += value
        rows.append({"segment": name, "revenue": revenue, "margin": margin, "multiple": multiple, "value": value})
    return {"available": True, "segment_table": pd.DataFrame(rows), "enterprise_value": total, "summary": "SOTP based on segment-level assumptions."}

