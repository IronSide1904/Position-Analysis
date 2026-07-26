import pandas as pd

from models.dcf_model import default_assumptions_from_historicals, run_dcf
from models.multiples_model import build_multiples_table, calculate_current_multiples, calculate_scenario_implied_multiples, peer_median_multiples
from models.sotp_model import VALUATION_METHODS, build_default_segment_data, run_reverse_sotp, run_sotp, run_sotp_scenarios, sotp_timeframe_options


def sample_historicals():
    return pd.DataFrame(
        [
            {
                "Period": "Latest",
                "Revenue": 1000.0,
                "Gross Margin": 0.5,
                "EBIT": 180.0,
                "EBITDA": 220.0,
                "NOPAT": 142.0,
                "OCF": 210.0,
                "FCF": 160.0,
                "Total CAPEX": 50.0,
                "Net Income": 120.0,
                "Diluted Shares": 100.0,
                "Net Debt": 100.0,
            }
        ]
    )


def test_sotp_scenarios_and_reverse_sotp_do_not_blank_without_segment_filings():
    historicals = sample_historicals()
    market = {"price": 20.0, "market_cap": 2000.0, "enterprise_value": 2100.0, "shares_outstanding": 100.0}
    assumptions = default_assumptions_from_historicals(historicals, market)
    dcf = run_dcf(historicals, market, assumptions)
    segments = build_default_segment_data(historicals, {"sector": "Technology"}, assumptions)

    result = run_sotp(segments, market, assumptions, dcf_output=dcf, sector="Technology")
    scenarios = run_sotp_scenarios(segments, market, assumptions, dcf, sector="Technology")
    reverse = run_reverse_sotp(market, segments, assumptions)

    assert result["fair_value_per_share"] is not None
    assert set(scenarios) == {"Bear Case", "Base Case", "Bull Case", "User Case", "Market-Implied Case"}
    assert not reverse["segments"].empty
    assert "Market-Implied EV/Revenue" in reverse["segments"].columns


def test_sotp_defaults_use_aapl_product_service_segments():
    historicals = sample_historicals()
    assumptions = default_assumptions_from_historicals(historicals, {"shares_outstanding": 100.0})
    segments = build_default_segment_data(
        historicals,
        {
            "ticker": "AAPL",
            "sector": "Technology",
            "company_description": "Apple reports iPhone, Mac, iPad, Wearables and Services revenue.",
        },
        assumptions,
    )
    names = set(segments["Segment"].astype(str))

    assert {"iPhone", "Mac", "iPad", "Wearables, Home and Accessories", "Services"}.issubset(names)
    assert "Core business" not in names
    assert "Manual Segment Value" in segments.columns


def test_sotp_supports_required_valuation_methods():
    required = {"EV/Revenue", "EV/EBITDA", "EV/EBIT", "EV/NOPAT", "EV/OCF", "EV/FCF", "P/E", "Manual Value"}

    assert required.issubset(set(VALUATION_METHODS))


def test_sotp_timeframe_changes_metric_basis_and_output():
    historicals = sample_historicals()
    historicals.loc[0, "Period"] = "FY2025A"
    market = {"price": 20.0, "market_cap": 2000.0, "enterprise_value": 2100.0, "shares_outstanding": 100.0}
    assumptions = default_assumptions_from_historicals(historicals, market)
    assumptions["revenue_cagr"] = 0.10
    dcf = run_dcf(historicals, market, assumptions)
    segments = build_default_segment_data(historicals, {"sector": "Technology"}, assumptions)

    options = sotp_timeframe_options(historicals, dcf, assumptions)
    latest = run_sotp(segments, market, assumptions, dcf_output=dcf, historicals=historicals, timeframe="Latest / LTM")
    final_year = run_sotp(segments, market, assumptions, dcf_output=dcf, historicals=historicals, timeframe=options[-2])

    assert "FY2026E" in options
    assert "Normalized Year" in options
    assert final_year["segments"]["Revenue"].iloc[0] > latest["segments"]["Revenue"].iloc[0]
    assert final_year["fair_value_per_share"] != latest["fair_value_per_share"]
    assert {"SOTP Timeframe", "Valuation Metric", "Metric Value", "Segment EV"}.issubset(final_year["segments"].columns)


def test_multiples_table_uses_unavailable_safe_values_and_peer_fallbacks():
    historicals = sample_historicals()
    market = {"price": 20.0, "market_cap": 2000.0, "enterprise_value": 2100.0, "shares_outstanding": 100.0}
    assumptions = default_assumptions_from_historicals(historicals, market)
    dcf = run_dcf(historicals, market, assumptions)
    current = calculate_current_multiples(historicals, market)
    peer_medians, warnings = peer_median_multiples(pd.DataFrame(), sector="Technology")
    scenario_multiples = calculate_scenario_implied_multiples({"Base Case": dcf}, historicals, market)
    table = build_multiples_table(current, scenario_multiples, peer_medians, peer_medians)

    assert warnings
    assert table["Metric"].tolist()
    assert "EV/OCF" in set(table["Metric"])
    assert table.loc[table["Metric"] == "EV/OCF", "Current Company"].iloc[0] is not None
