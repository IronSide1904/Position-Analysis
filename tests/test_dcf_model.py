import pandas as pd

from models.dcf_model import build_dcf_output_table, build_scenario_table, default_assumptions_from_historicals, run_dcf
from models.financial_model import build_ev_to_equity_bridge
from ui.dashboard_v2 import (
    _active_assumption_edit_count,
    _assumption_gap_table,
    _build_assumption_scenarios,
    _dcf_forecast_output_table,
    _scenario_valuation_summary,
    recalculate_active_scenario,
    validate_scenario_consistency,
)


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


def test_dcf_uses_last_positive_revenue_before_market_cap_fallback():
    historicals = pd.DataFrame(
        [
            {"Period": "FY 2025", "Revenue": 7200.0, "Gross Margin": 0.5, "EBIT": 1200.0, "OCF": 1000.0, "Total CAPEX": 200.0, "Diluted Shares": 100.0},
            {"Period": "LTM Latest", "Revenue": 0.0, "Gross Margin": 0.5, "EBIT": 1200.0, "OCF": 1000.0, "Total CAPEX": 200.0, "Diluted Shares": 100.0},
        ]
    )
    market = {"price": 10.0, "shares_outstanding": 100.0, "market_cap": 206000.0}
    assumptions = default_assumptions_from_historicals(historicals, market)
    result = run_dcf(historicals, market, assumptions)

    assert result["forecast_table"].iloc[0]["Revenue"] == 7200.0 * (1 + assumptions["revenue_cagr"])


def test_dcf_accepts_year_specific_forecast_assumptions():
    historicals = sample_historicals()
    market = {"price": 10.0, "shares_outstanding": 100.0}
    assumptions = default_assumptions_from_historicals(historicals, market)
    assumptions["forecast_assumptions_by_year"] = {
        "1": {"revenue_cagr": 0.10, "ocf_margin": 0.20},
        "2": {"revenue_cagr": 0.05, "ocf_margin": 0.30},
    }

    forecast = run_dcf(historicals, market, assumptions)["forecast_table"]

    assert forecast.iloc[0]["Revenue Growth"] == 0.10
    assert forecast.iloc[1]["Revenue Growth"] == 0.05
    assert forecast.iloc[0]["OCF Margin"] == 0.20
    assert forecast.iloc[1]["OCF Margin"] == 0.30


def test_dcf_forecast_explicitly_models_reinvestment_lines():
    historicals = sample_historicals()
    market = {"price": 10.0, "shares_outstanding": 100.0}
    assumptions = default_assumptions_from_historicals(historicals, market)
    result = run_dcf(historicals, market, assumptions)
    forecast = result["forecast_table"]

    for column in [
        "Maintenance CAPEX",
        "Growth CAPEX",
        "Total CAPEX",
        "Working Capital Investment",
        "Normalized Cash Earnings",
        "FCF",
        "FCFF",
    ]:
        assert column in forecast.columns

    first_year = forecast.iloc[0]
    assert first_year["Total CAPEX"] == first_year["Maintenance CAPEX"] + first_year["Growth CAPEX"]
    assert first_year["Normalized Cash Earnings"] == first_year["OCF"] - first_year["Maintenance CAPEX"]


def test_scenario_table_includes_capex_assumptions():
    historicals = sample_historicals()
    market = {"price": 10.0, "shares_outstanding": 100.0}
    assumptions = default_assumptions_from_historicals(historicals, market)
    table = build_scenario_table(historicals, market, assumptions)

    required_rows = {
        "Maintenance CAPEX % revenue",
        "Growth CAPEX % revenue",
        "Total CAPEX % revenue",
        "CAPEX Normalization Year",
        "Working Capital % revenue",
        "FCF margin",
        "Fair value per share",
    }
    assert required_rows.issubset(set(table["Line Item"]))
    assert {"Bear", "Base", "Bull", "User", "Market-Implied"}.issubset(table.columns)


def test_dcf_output_table_starts_with_latest_actual_column():
    historicals = sample_historicals()
    historicals.loc[0, "Period"] = "2025"
    market = {"price": 10.0, "shares_outstanding": 100.0}
    assumptions = default_assumptions_from_historicals(historicals, market)
    result = run_dcf(historicals, market, assumptions)

    table = _dcf_forecast_output_table(result, assumptions, historicals)

    assert "2025A" in table.columns
    assert table.columns.tolist().index("2025A") < table.columns.tolist().index("FY1E")
    revenue_row = table[table["Metric"] == "Revenue"].iloc[0]
    assert revenue_row["2025A"] == 1000.0
    assert revenue_row["FY1E"] > revenue_row["2025A"]


def test_default_user_case_equals_base_case_until_edits():
    historicals = sample_historicals()
    market = {"price": 10.0, "shares_outstanding": 100.0}
    base = default_assumptions_from_historicals(historicals, market)

    scenarios = _build_assumption_scenarios(base, None)

    assert _active_assumption_edit_count(scenarios["Base Case"], scenarios["User Case"]) == 0
    edited_user = {**scenarios["User Case"], "revenue_cagr": scenarios["Base Case"]["revenue_cagr"] + 0.02}
    assert _active_assumption_edit_count(scenarios["Base Case"], edited_user) == 1


def test_dcf_detail_and_ev_bridge_are_separate_tables():
    historicals = sample_historicals()
    market = {"price": 10.0, "shares_outstanding": 100.0, "cash": 25.0, "debt": 125.0}
    assumptions = default_assumptions_from_historicals(historicals, market)
    result = run_dcf(historicals, market, assumptions)

    dcf_detail = build_dcf_output_table(result, assumptions, market)
    bridge = build_ev_to_equity_bridge(market, result, assumptions)

    assert "Bridge" not in dcf_detail.columns
    assert "Terminal" in dcf_detail.columns
    assert "Value" in bridge.columns
    assert "Evidence / Source" in bridge.columns
    assert not any(str(column).startswith("Year ") for column in bridge.columns)
    assert {"Enterprise value", "Equity value", "Fair value / share"}.issubset(set(bridge["Metric"]))


def test_active_scenario_state_recalculates_user_case_outputs():
    historicals = sample_historicals()
    market = {"price": 10.0, "shares_outstanding": 100.0}
    base = default_assumptions_from_historicals(historicals, market)
    ctx = {"dataset": {"ticker": "TEST", "market_data": market}, "historicals": historicals, "base_assumptions": base}

    base_state = recalculate_active_scenario(ctx, "User Case", base)
    edited = {**base, "revenue_cagr": base["revenue_cagr"] + 0.10}
    edited_state = recalculate_active_scenario(ctx, "User Case", edited)

    assert edited_state.selected_case == "User Case"
    assert edited_state.active_assumptions["revenue_cagr"] == edited["revenue_cagr"]
    assert edited_state.model_outputs["dcf"]["fair_value_per_share"] != base_state.model_outputs["dcf"]["fair_value_per_share"]
    assert validate_scenario_consistency(ctx, edited_state) == []


def test_market_gap_and_scenario_reference_use_central_state():
    historicals = sample_historicals()
    market = {"price": 1000.0, "shares_outstanding": 100.0}
    base = default_assumptions_from_historicals(historicals, market)
    ctx = {"dataset": {"ticker": "TEST", "market_data": market}, "historicals": historicals, "base_assumptions": base}

    state = recalculate_active_scenario(ctx, "User Case", base)
    gap = _assumption_gap_table(state)
    valuation_reference = _scenario_valuation_summary(state, market)

    nopat_row = gap[gap["Assumption"] == "NOPAT Margin %"].iloc[0]
    assert nopat_row["Market-Implied"] == "Outside range"
    assert "Market Price" in valuation_reference["Scenario"].tolist()
