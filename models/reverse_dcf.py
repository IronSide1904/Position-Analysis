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


REVERSE_DCF_BOUNDS = {
    "revenue_cagr": (-0.10, 0.35),
    "nopat_margin": (-0.10, 0.35),
    "ocf_margin": (-0.10, 0.40),
    "terminal_growth": (0.00, 0.04),
    "terminal_multiple": (6.0, 25.0),
}


def _solve_one_variable(market_data: dict, historicals: pd.DataFrame, base_assumptions: dict, key: str, bounds: tuple[float, float]) -> dict:
    target_price = market_data.get("price")
    low, high = bounds

    def price_at(value: float) -> float:
        assumptions = dict(base_assumptions)
        assumptions[key] = value
        return float(run_dcf(historicals, market_data, assumptions).get("fair_value_per_share") or 0)

    low_price = price_at(low)
    high_price = price_at(high)
    low_gap = low_price - float(target_price or 0)
    high_gap = high_price - float(target_price or 0)
    if low_gap == 0:
        implied = low
    elif high_gap == 0:
        implied = high
    elif low_gap * high_gap < 0:
        try:
            implied = brentq(lambda value: price_at(value) - target_price, low, high)
        except Exception:
            implied = None
    else:
        implied = None

    if implied is not None:
        status = "Realistic"
        display_value = implied
        conclusion = "Market price can be explained within the configured reasonable range."
    elif low_gap > 0 and high_gap > 0:
        status = "Below Range"
        display_value = f"<{low:.1%}" if key != "terminal_multiple" else f"<{low:.1f}x"
        conclusion = "Market price is below the value implied even at the low end of the range."
    else:
        status = "Outside Range"
        display_value = f">{high:.1%}" if key != "terminal_multiple" else f">{high:.1f}x"
        conclusion = "Current price requires assumptions outside reasonable bounds."

    return {
        "key": key,
        "implied": implied,
        "display_value": display_value,
        "bounds": bounds,
        "status": status,
        "low_price": low_price,
        "high_price": high_price,
        "conclusion": conclusion,
    }


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
            "implied_terminal_growth": None,
            "implied_terminal_multiple": None,
            "market_case": "Unknown",
            "interpretation": "Price unavailable; reverse DCF cannot be calculated.",
            "gap_vs_user_case": {},
            "solves": {},
            "bounds": REVERSE_DCF_BOUNDS,
        }

    solves = {
        key: _solve_one_variable(market_data, historicals, base_assumptions, key, bounds)
        for key, bounds in REVERSE_DCF_BOUNDS.items()
    }
    implied_growth = solves["revenue_cagr"]["implied"]
    implied_nopat_margin = solves["nopat_margin"]["implied"]
    implied_ocf_margin = solves["ocf_margin"]["implied"]
    implied_terminal_growth = solves["terminal_growth"]["implied"]
    implied_terminal_multiple = solves["terminal_multiple"]["implied"]

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
    elif any(item["status"] == "Outside Range" for item in solves.values()):
        market_case = "Outside reasonable range"

    user_growth = base_assumptions.get("revenue_cagr")
    gap = {"revenue_cagr": implied_growth - user_growth if implied_growth is not None and user_growth is not None else None}
    outside = [key for key, item in solves.items() if item["status"] == "Outside Range"]
    if outside:
        interpretation = "Current price requires assumptions outside reasonable bounds under the current base case."
    elif gap["revenue_cagr"] and gap["revenue_cagr"] > 0:
        interpretation = "Positive evidence exists, but market expectations may already be higher than your adjusted case."
    else:
        interpretation = "Market expectations appear at or below the user case."

    return {
        "implied_revenue_cagr": implied_growth,
        "implied_nopat_margin": implied_nopat_margin,
        "implied_ocf_margin": implied_ocf_margin,
        "implied_terminal_growth": implied_terminal_growth,
        "implied_terminal_multiple": implied_terminal_multiple,
        "market_case": market_case,
        "interpretation": interpretation,
        "gap_vs_user_case": gap,
        "solves": solves,
        "bounds": REVERSE_DCF_BOUNDS,
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
