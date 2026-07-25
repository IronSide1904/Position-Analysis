from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _num(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None:
            return default
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _median(frame: pd.DataFrame | None, names: list[str]) -> float | None:
    if frame is None or frame.empty:
        return None
    for name in names:
        if name in frame:
            values = pd.to_numeric(frame[name], errors="coerce").replace([math.inf, -math.inf], pd.NA).dropna()
            if not values.empty:
                return float(values.median())
    return None


def recommend_terminal_multiple(
    selected_metric: str,
    peer_multiples: pd.DataFrame | None,
    sector_multiples: pd.DataFrame | dict | None,
    growth_profile: dict,
    moat_analysis: dict | None,
    ocf_quality: dict | None,
    capital_intensity: dict | None,
    management_score: dict | None,
) -> dict:
    """
    Recommend terminal multiple range and explain why.
    """
    selected_metric = selected_metric or "EV/FCF"
    column_names = [selected_metric, selected_metric.replace("/", "_"), selected_metric.replace("/", "")]
    peer = _median(peer_multiples, column_names)
    if isinstance(sector_multiples, pd.DataFrame):
        sector = _median(sector_multiples, column_names)
    elif isinstance(sector_multiples, dict):
        sector = _num(sector_multiples.get(selected_metric))
    else:
        sector = None
    anchor = peer or sector or {"EV/Revenue": 3.0, "EV/EBITDA": 12.0, "EV/EBIT": 16.0, "EV/OCF": 16.0, "EV/FCF": 18.0, "P/E": 20.0}.get(selected_metric, 15.0)
    growth = _num((growth_profile or {}).get("revenue_cagr"), 0.0) or 0.0
    moat_text = str((moat_analysis or {}).get("classification") or "").lower()
    ocf_margin = _num((ocf_quality or {}).get("ocf_margin"), None)
    capex_intensity = _num((capital_intensity or {}).get("total_capex_pct_revenue"), None)
    management = _num((management_score or {}).get("management_score"), 5.0) or 5.0
    score = 0
    score += 1 if growth > 0.12 else -1 if growth < 0.02 else 0
    score += 1 if "wide" in moat_text or "strong" in moat_text else -1 if "weak" in moat_text else 0
    score += 1 if ocf_margin is not None and ocf_margin > 0.18 else -1 if ocf_margin is not None and ocf_margin < 0.05 else 0
    score += 1 if capex_intensity is not None and capex_intensity < 0.06 else -1 if capex_intensity is not None and capex_intensity > 0.15 else 0
    score += 1 if management >= 7 else -1 if management <= 3 else 0
    midpoint = anchor * (1 + 0.06 * score)
    low = max(midpoint * 0.8, 1.0)
    high = midpoint * 1.2
    if score >= 2:
        classification = "Premium justified"
    elif score <= -2:
        classification = "Conservative / discount required"
    elif peer is None and sector is None:
        classification = "Insufficient data"
    else:
        classification = "In-line"
    warnings = []
    if peer is None:
        warnings.append("Peer multiple unavailable. Add peers or fetch peer data.")
    if capex_intensity is not None and capex_intensity > 0.15:
        warnings.append("High capital intensity may require a discount multiple.")
    return {
        "recommended_low": low,
        "recommended_mid": midpoint,
        "recommended_high": high,
        "selected_multiple": anchor,
        "classification": classification,
        "reason": "Anchored to peer/sector multiple where available, then adjusted for growth, moat, OCF quality, capital intensity, and management credibility.",
        "warnings": warnings,
    }
