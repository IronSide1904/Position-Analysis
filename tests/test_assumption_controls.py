from ui.dashboard_v2 import get_assumption_range


def test_revenue_cagr_uses_tight_profile_aware_default_range():
    result = get_assumption_range(
        "revenue_cagr",
        "General",
        base_value=0.08,
        bear_value=0.05,
        bull_value=0.13,
        market_implied_value=0.12,
    )

    assert result["min"] >= -0.08
    assert result["max"] <= 0.28
    assert result["step"] == 0.005


def test_revenue_cagr_range_expands_for_market_implied_outlier():
    result = get_assumption_range(
        "revenue_cagr",
        "General",
        base_value=0.08,
        bear_value=0.05,
        bull_value=0.13,
        market_implied_value=0.42,
    )

    assert result["max"] > 0.42
    assert result["warning_level"] == "expanded"
