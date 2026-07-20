import pandas as pd

from models.operating_driver_model import (
    build_business_model_profile,
    build_wacc_table,
    build_valuation_method_results,
    default_driver_matrix,
    driver_assumptions_to_dcf_assumptions,
    driver_result_table,
    integrate_driver_valuation,
    run_driver_model,
    solve_market_implied_driver,
)


def _historicals():
    return pd.DataFrame(
        [
            {
                "Period": "FY 2025",
                "Revenue": 1000.0,
                "Gross Margin": 0.5,
                "EBIT": 120.0,
                "OCF": 160.0,
                "Total CAPEX": 90.0,
                "Diluted Shares": 100.0,
                "Net Debt": 50.0,
            }
        ]
    )


def _market():
    return {"price": 10.0, "shares_outstanding": 100.0, "market_cap": 1000.0, "debt": 50.0, "cash": 25.0, "beta": 1.2}


def _assumptions():
    return {
        "forecast_years": 5,
        "revenue_cagr": 0.08,
        "tax_rate": 0.21,
        "ocf_margin": 0.20,
        "nopat_margin": 0.12,
        "maintenance_capex_pct_revenue": 0.03,
        "growth_capex_pct_revenue": 0.04,
        "working_capital_pct_revenue": 0.01,
        "diluted_shares": 100.0,
        "diluted_share_growth": 0.0,
        "wacc": 0.10,
        "terminal_growth": 0.02,
        "terminal_multiple": 12.0,
        "margin_of_safety": 0.30,
        "net_debt": 50.0,
    }


def _driver_matrix():
    profile = build_business_model_profile("Capacity / Infrastructure")
    matrix = default_driver_matrix(profile, _historicals(), _market(), _assumptions(), years=5)
    periods = ["FY1E", "FY2F", "FY3F", "FY4F", "FY5F"]
    matrix.loc[matrix["row_key"] == "capacity_added", periods] = [1, 1, 1, 1, 1]
    matrix.loc[matrix["row_key"] == "utilization", periods] = 0.80
    matrix.loc[matrix["row_key"] == "revenue_per_unit", periods] = 1000.0
    matrix.loc[matrix["row_key"] == "ebitda_margin", periods] = 0.30
    matrix.loc[matrix["row_key"] == "maintenance_cost_per_unit", periods] = 10.0
    matrix.loc[matrix["row_key"] == "hardware_cost_per_unit", periods] = 100.0
    matrix.loc[matrix["row_key"] == "infrastructure_cost_per_unit", periods] = 200.0
    matrix.loc[matrix["row_key"] == "land_cost_per_unit", periods] = 50.0
    matrix.loc[matrix["row_key"] == "hardware_useful_life", periods] = 5.0
    matrix.loc[matrix["row_key"] == "infrastructure_useful_life", periods] = 10.0
    matrix.loc[matrix["row_key"] == "customer_prepayment_pct", periods] = 0.20
    matrix.loc[matrix["row_key"] == "equity_funding_pct", periods] = 0.10
    matrix.loc[matrix["row_key"] == "equity_issue_price", periods] = 10.0
    matrix.loc[matrix["row_key"] == "sbc_dilution_pct", periods] = 0.01
    matrix.loc[matrix["row_key"] == "cost_of_debt", periods] = 0.08
    matrix.loc[matrix["row_key"] == "risk_free_rate", periods] = 0.04
    matrix.loc[matrix["row_key"] == "beta", periods] = 1.2
    matrix.loc[matrix["row_key"] == "equity_risk_premium", periods] = 0.05
    matrix.loc[matrix["row_key"] == "exit_ebitda_multiple", periods] = 10.0
    matrix.loc[matrix["row_key"] == "exit_ebit_multiple", periods] = 12.0
    return profile, matrix


def test_capacity_additions_update_ending_and_average_capacity():
    profile, matrix = _driver_matrix()
    result = run_driver_model(profile, matrix, _historicals(), _market(), _assumptions())
    drivers = pd.DataFrame(result.driver_forecast)

    assert drivers.loc[0, "Ending Capacity"] == 2.0
    assert drivers.loc[0, "Average Capacity"] == 1.5


def test_utilization_and_revenue_per_unit_update_revenue():
    profile, matrix = _driver_matrix()
    result = run_driver_model(profile, matrix, _historicals(), _market(), _assumptions())
    income = pd.DataFrame(result.income_statement)

    assert income.loc[0, "Revenue"] == 1.5 * 0.80 * 1000.0


def test_build_cost_changes_capex_and_funding_need():
    profile, matrix = _driver_matrix()
    result = run_driver_model(profile, matrix, _historicals(), _market(), _assumptions())
    funding = pd.DataFrame(result.funding_schedule)

    assert funding.loc[0, "Build CAPEX"] == 350.0
    assert funding.loc[0, "Customer Prepayments"] == 70.0
    assert funding.loc[0, "Equity Raised"] == 35.0
    assert funding.loc[0, "Debt Drawn"] >= 0.0


def test_maintenance_treatment_controls_ebit_and_invested_capital():
    profile, matrix = _driver_matrix()
    expensed = run_driver_model(profile, matrix, _historicals(), _market(), _assumptions(), maintenance_treatment="Expensed")
    capitalized = run_driver_model(profile, matrix, _historicals(), _market(), _assumptions(), maintenance_treatment="Capitalized")
    exp_income = pd.DataFrame(expensed.income_statement)
    cap_income = pd.DataFrame(capitalized.income_statement)
    exp_capital = pd.DataFrame(expensed.invested_capital_schedule)
    cap_capital = pd.DataFrame(capitalized.invested_capital_schedule)

    assert exp_income.loc[0, "EBIT"] < cap_income.loc[0, "EBIT"]
    assert cap_capital.loc[0, "Invested Capital"] > exp_capital.loc[0, "Invested Capital"]


def test_depreciation_uses_separate_lives_and_land_is_not_depreciated():
    profile, matrix = _driver_matrix()
    result = run_driver_model(profile, matrix, _historicals(), _market(), _assumptions())
    depreciation = pd.DataFrame(result.depreciation_schedule)

    assert depreciation.loc[0, "Hardware Depreciation"] == 20.0
    assert depreciation.loc[0, "Infrastructure Depreciation"] == 20.0
    assert depreciation.loc[0, "Depreciation"] == 40.0
    assert depreciation.loc[0, "Land CAPEX"] == 50.0


def test_customer_prepayments_do_not_become_revenue_and_equity_dilutes():
    profile, matrix = _driver_matrix()
    result = run_driver_model(profile, matrix, _historicals(), _market(), _assumptions())
    income = pd.DataFrame(result.income_statement)
    funding = pd.DataFrame(result.funding_schedule)
    shares = pd.DataFrame(result.share_schedule)

    assert funding.loc[0, "Customer Prepayments"] > 0
    assert income.loc[0, "Revenue"] != funding.loc[0, "Customer Prepayments"]
    assert shares.loc[0, "Diluted Shares"] > 100.0


def test_debt_funding_uses_average_debt_for_interest():
    profile, matrix = _driver_matrix()
    result = run_driver_model(profile, matrix, _historicals(), _market(), _assumptions())
    debt = pd.DataFrame(result.debt_schedule)

    assert debt.loc[0, "Average Debt"] == (debt.loc[0, "Beginning Debt"] + debt.loc[0, "Ending Debt"]) / 2
    assert debt.loc[0, "Interest Expense"] == debt.loc[0, "Average Debt"] * 0.08


def test_roic_uses_nopat_and_average_invested_capital():
    profile, matrix = _driver_matrix()
    result = run_driver_model(profile, matrix, _historicals(), _market(), _assumptions())
    returns = pd.DataFrame(result.roic_schedule)

    assert returns.loc[0, "ROIC"] == returns.loc[0, "NOPAT"] / returns.loc[0, "Average Invested Capital"]
    assert returns.loc[0, "ROIC Spread"] == returns.loc[0, "ROIC"] - returns.loc[0, "WACC"]


def test_driver_assumptions_feed_dcf_forecast_assumptions():
    profile, matrix = _driver_matrix()
    result = run_driver_model(profile, matrix, _historicals(), _market(), _assumptions())
    dcf_assumptions = driver_assumptions_to_dcf_assumptions(_assumptions(), result)

    assert "forecast_assumptions_by_year" in dcf_assumptions
    assert dcf_assumptions["diluted_shares"] > 100.0
    assert dcf_assumptions["net_debt"] is not None


def test_valuation_methods_do_not_show_negative_equity_as_target_price():
    profile, matrix = _driver_matrix()
    result = run_driver_model(profile, matrix, _historicals(), {**_market(), "debt": 100000.0, "cash": 0.0}, _assumptions())
    methods = build_valuation_method_results(result, {"fair_value_per_share": 1.0, "enterprise_value": 100.0, "equity_value": 10.0}, {**_market(), "debt": 100000.0}, _assumptions(), 10.0, 12.0)
    table = pd.DataFrame([item.__dict__ for item in methods])
    ebit = table[table["method"] == "EBIT Multiple"].iloc[0]

    assert not bool(ebit["applicable"])
    assert pd.isna(ebit["value_per_share"])


def test_market_implied_solver_respects_bounds():
    profile, matrix = _driver_matrix()
    solved = solve_market_implied_driver(profile, matrix, _historicals(), _market(), _assumptions(), "utilization", 0.10, 0.95)

    assert solved["status"] in {"Solved", "Outside reasonable range"}
    assert solved["required_value"] is None or 0.10 <= solved["required_value"] <= 0.95


def test_integrated_driver_result_builds_primary_tables():
    profile, matrix = _driver_matrix()
    result = integrate_driver_valuation("User Case", profile, matrix, _historicals(), _market(), _assumptions())
    table = driver_result_table(result.driver_model)

    assert result.dcf_result["forecast_table"] is not None
    assert "ROIC %" in table["Line Item"].tolist()
    assert "Historical / LTM" in table.columns
    assert table.loc[table["Line Item"] == "Revenue", "Historical / LTM"].iloc[0] == 1000.0
    assert result.economic_interpretation


def test_wacc_build_exposes_calculated_components():
    profile, matrix = _driver_matrix()
    result = run_driver_model(profile, matrix, _historicals(), _market(), _assumptions())
    table = build_wacc_table(result, _market())

    assert "Cost of Equity" in table.columns
    assert "After-Tax Cost of Debt" in table.columns
    assert table.loc[table["Period"] == "FY1E", "Cost of Equity"].iloc[0] == 0.10
    assert table.loc[table["Period"] == "FY1E", "Pretax Cost of Debt"].iloc[0] == 0.08


def test_standard_financial_mode_still_runs_without_capacity_drivers():
    profile = build_business_model_profile("Standard Financial")
    matrix = default_driver_matrix(profile, _historicals(), _market(), _assumptions(), years=5)
    result = run_driver_model(profile, matrix, _historicals(), _market(), _assumptions())
    income = pd.DataFrame(result.income_statement)

    assert not income.empty
    assert income.loc[0, "Revenue"] == 1080.0
