from __future__ import annotations

import pandas as pd


def _latest(df: pd.DataFrame, col: str, default=0):
    if df is None or df.empty or col not in df:
        return default
    value = df[col].dropna()
    return value.iloc[-1] if not value.empty else default


def analyze_capex_ocf_nopat_quality(historicals: pd.DataFrame, clauses: pd.DataFrame) -> dict:
    """
    Analyze whether reported CAPEX, NOPAT, OCF and FCF reflect economic reality.
    """
    revenue = float(_latest(historicals, "Revenue", 0) or 0)
    ocf = float(_latest(historicals, "OCF", 0) or 0)
    nopat = float(_latest(historicals, "NOPAT", 0) or 0)
    capex = float(_latest(historicals, "Total CAPEX", 0) or 0)
    maint = float(_latest(historicals, "Maintenance CAPEX", 0) or 0)
    fcf = float(_latest(historicals, "FCF", 0) or 0)
    red_flags = []
    implications = []
    adjustments = []

    capex_pct = capex / revenue if revenue else None
    ocf_conversion = ocf / nopat if nopat else None
    if capex_pct and capex_pct > 0.12:
        red_flags.append("CAPEX intensity is high; verify maintenance versus growth split.")
        adjustments.append("Stress test higher maintenance CAPEX.")
    if ocf_conversion is not None and ocf_conversion < 0.7:
        red_flags.append("OCF conversion trails NOPAT; working capital or cash quality may be distorting earnings.")
        implications.append("Lower OCF margin or increase working capital investment.")
    if fcf < 0 and revenue > 0:
        red_flags.append("FCF is negative after CAPEX.")
    if clauses is not None and not clauses.empty and "Working Capital" in set(clauses["topic"]):
        implications.append("Filing clauses mention working-capital drivers; review inventory, receivables, and deferred revenue.")

    score = 7
    score -= min(len(red_flags) * 1.5, 5)
    return {
        "summary": "Cash conversion appears reasonable." if not red_flags else "Manual review required: reinvestment and cash conversion have visible pressure points.",
        "quality_score": max(int(round(score)), 1),
        "red_flags": red_flags,
        "model_implications": implications,
        "suggested_dcf_adjustments": adjustments,
        "metrics": {
            "capex_pct_revenue": capex_pct,
            "maintenance_capex_pct_capex": maint / capex if capex else None,
            "ocf_conversion": ocf_conversion,
        },
    }

