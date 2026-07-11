import pandas as pd

from models.dcf_model import default_assumptions_from_historicals
from models.reverse_dcf import run_reverse_dcf


def test_reverse_dcf_runs_with_market_price():
    historicals = pd.DataFrame(
        [{"Revenue": 1000.0, "Gross Margin": 0.5, "EBIT": 200.0, "OCF": 200.0, "Total CAPEX": 50.0, "Diluted Shares": 100.0, "Net Debt": 0.0}]
    )
    market = {"price": 10.0, "shares_outstanding": 100.0}
    assumptions = default_assumptions_from_historicals(historicals, market)
    result = run_reverse_dcf(market, historicals, assumptions)
    assert result["market_case"] in {"Bear", "Base", "Bull", "Extreme Bull", "Unknown"}


def test_reverse_dcf_flags_implied_margin_outside_bounds():
    historicals = pd.DataFrame(
        [{"Revenue": 1000.0, "Gross Margin": 0.5, "EBIT": 150.0, "OCF": 150.0, "Total CAPEX": 50.0, "Diluted Shares": 100.0, "Net Debt": 0.0}]
    )
    market = {"price": 1000.0, "shares_outstanding": 100.0}
    assumptions = default_assumptions_from_historicals(historicals, market)

    result = run_reverse_dcf(market, historicals, assumptions)

    assert result["implied_nopat_margin"] is None
    assert result["solves"]["nopat_margin"]["status"] == "Outside Range"
    assert result["solves"]["nopat_margin"]["display_value"] == ">35.0%"
    assert "outside reasonable bounds" in result["interpretation"]
