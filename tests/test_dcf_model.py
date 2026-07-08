import pandas as pd

from models.dcf_model import default_assumptions_from_historicals, run_dcf


def sample_historicals():
    return pd.DataFrame(
        [
            {
                "Period": "Latest",
                "Revenue": 1000.0,
                "Gross Margin": 0.5,
                "EBIT": 200.0,
                "OCF": 220.0,
                "Total CAPEX": 50.0,
                "Diluted Shares": 100.0,
                "Net Debt": 100.0,
            }
        ]
    )


def test_dcf_runs_with_fallback_assumptions():
    historicals = sample_historicals()
    market = {"price": 10.0, "shares_outstanding": 100.0}
    assumptions = default_assumptions_from_historicals(historicals, market)
    result = run_dcf(historicals, market, assumptions)
    assert result["fair_value_per_share"] is not None
    assert not result["forecast_table"].empty


def test_user_adjusted_assumption_updates_fair_value():
    historicals = sample_historicals()
    market = {"price": 10.0, "shares_outstanding": 100.0}
    assumptions = default_assumptions_from_historicals(historicals, market)
    low = run_dcf(historicals, market, {**assumptions, "revenue_cagr": 0.01})["fair_value_per_share"]
    high = run_dcf(historicals, market, {**assumptions, "revenue_cagr": 0.20})["fair_value_per_share"]
    assert high > low

