from __future__ import annotations

import pandas as pd

from models.dcf_model import run_dcf

try:
    from scipy.optimize import brentq
except Exception:
    def brentq(func, a, b, maxiter=80, xtol=1e-6):
        fa = func(a)
        fb = func(b)
        if fa * fb > 0:
            raise ValueError("Root is not bracketed")
        lo, hi = a, b
        for _ in range(maxiter):
            mid = (lo + hi) / 2
            fm = func(mid)
            if abs(fm) < xtol:
                return mid
            if fa * fm <= 0:
                hi = mid
                fb = fm
            else:
                lo = mid
                fa = fm
        return (lo + hi) / 2


def run_reverse_dcf(market_data: dict, historicals: pd.DataFrame, base_assumptions: dict) -> dict:
    """
    Start from current price / market cap / EV and estimate what growth and margin
    assumptions are implied by the market.
    """
    target_price = market_data.get("price")
    if not target_price:
        return {
            "implied_revenue_cagr": None,
            "implied_nopat_margin": None,
            "implied_ocf_margin": None,
            "implied_terminal_multiple": None,
            "market_case": "Unknown",
            "interpretation": "Price unavailable; reverse DCF cannot be calculated.",
            "gap_vs_user_case": {},
        }

    def price_at_growth(growth: float) -> float:
        assumptions = dict(base_assumptions)
        assumptions["revenue_cagr"] = growth
        value = run_dcf(historicals, market_data, assumptions)["fair_value_per_share"]
        return float(value or 0)

    try:
        implied_growth = brentq(lambda g: price_at_growth(g) - target_price, -0.2, 0.6)
    except Exception:
        implied_growth = None

    implied_nopat_margin = base_assumptions.get("nopat_margin")
    if implied_growth is None and market_data.get("enterprise_value"):
        revenue = float(historicals["Revenue"].iloc[-1]) if "Revenue" in historicals and not historicals.empty else 0
        implied_nopat_margin = market_data["enterprise_value"] / max(revenue * 18, 1) if revenue else None

    market_case = "Unknown"
    if implied_growth is not None:
        if implied_growth < 0.03:
            market_case = "Bear"
        elif implied_growth < 0.12:
            market_case = "Base"
        elif implied_growth < 0.25:
            market_case = "Bull"
        else:
            market_case = "Extreme Bull"

    user_growth = base_assumptions.get("revenue_cagr")
    gap = {"revenue_cagr": implied_growth - user_growth if implied_growth is not None and user_growth is not None else None}
    interpretation = "Positive evidence exists, but market expectations may already be higher than your adjusted case." if gap["revenue_cagr"] and gap["revenue_cagr"] > 0 else "Market expectations appear at or below the user case."

    return {
        "implied_revenue_cagr": implied_growth,
        "implied_nopat_margin": implied_nopat_margin,
        "implied_ocf_margin": base_assumptions.get("ocf_margin"),
        "implied_terminal_multiple": base_assumptions.get("terminal_multiple"),
        "market_case": market_case,
        "interpretation": interpretation,
        "gap_vs_user_case": gap,
    }


def compare_clause_to_reverse_dcf(clause_row: dict, reverse_dcf_output: dict) -> dict:
    """
    Compare a clause implication with market-implied Reverse DCF expectations.
    """
    if not clause_row or not reverse_dcf_output:
        return {"market_already_prices_this": "Unknown", "interpretation": "Reverse DCF output is unavailable."}
    model_line = clause_row.get("model_line_affected")
    direction = clause_row.get("direction")
    implied_growth = reverse_dcf_output.get("implied_revenue_cagr")
    market_case = reverse_dcf_output.get("market_case", "Unknown")
    if model_line in {"revenue_growth", "segment_revenue_growth"} and direction == "Increase":
        if implied_growth is None:
            return {
                "market_already_prices_this": "Unknown",
                "interpretation": "This clause supports higher revenue growth, but Reverse DCF could not estimate market-implied growth.",
            }
        priced = implied_growth >= 0.12
        interpretation = (
            "This clause supports higher revenue growth, but the current price already implies aggressive growth. Treat as confirmation, not automatic upside."
            if priced
            else "This clause supports higher revenue growth, and Reverse DCF does not appear to price an aggressive growth case."
        )
        return {"market_already_prices_this": priced, "interpretation": interpretation}
    if model_line in {"wacc", "terminal_multiple", "scenario_probability"} and direction in {"Increase", "Decrease"}:
        return {
            "market_already_prices_this": "Unknown",
            "interpretation": f"This clause changes risk/thesis quality. Compare it manually against the market case: {market_case}.",
        }
    return {
        "market_already_prices_this": "Unknown",
        "interpretation": "Clause impact is model-specific; review before changing assumptions.",
    }
