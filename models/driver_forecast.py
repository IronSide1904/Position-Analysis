from __future__ import annotations

import pandas as pd

from models.driver_templates import get_driver_template
from models.operating_driver_model import (
    build_business_model_profile,
    default_driver_matrix,
    driver_result_table,
    integrate_driver_valuation,
    matrix_to_driver_inputs,
    period_labels,
    run_driver_model,
    solve_market_implied_driver,
)


def build_driver_based_forecast(
    profile: str,
    drivers: dict | pd.DataFrame,
    scenario: str,
    historicals: pd.DataFrame | None = None,
    market: dict | None = None,
    assumptions: dict | None = None,
) -> pd.DataFrame:
    """
    Convert business-specific drivers into a financial forecast table.
    """
    market = market or {}
    assumptions = assumptions or {"forecast_years": 5}
    model_profile = build_business_model_profile(profile)
    if isinstance(drivers, pd.DataFrame):
        matrix = drivers
    else:
        matrix = default_driver_matrix(model_profile, historicals, market, assumptions, years=int(assumptions.get("forecast_years", 5)))
        labels = period_labels(int(assumptions.get("forecast_years", 5)))
        for key, values in (drivers or {}).items():
            if isinstance(values, dict):
                for label in labels:
                    if label in values:
                        matrix.loc[matrix["row_key"] == key, label] = values[label]
            else:
                matrix.loc[matrix["row_key"] == key, labels] = values
    result = run_driver_model(model_profile, matrix, historicals, market, assumptions, years=int(assumptions.get("forecast_years", 5)))
    table = driver_result_table(result)
    table.insert(0, "Scenario", scenario)
    return table


def solve_market_implied_drivers(
    profile: str,
    current_market_value: dict,
    base_drivers: pd.DataFrame,
    target_metric: str,
    historicals: pd.DataFrame | None = None,
    assumptions: dict | None = None,
) -> dict:
    """
    Estimate which operating driver is implied by the current share price.
    """
    assumptions = assumptions or {"forecast_years": 5}
    model_profile = build_business_model_profile(profile)
    bounds = {
        "utilization": (0.10, 0.95),
        "revenue_per_blackwell_gw": (250_000_000.0, 5_000_000_000.0),
        "revenue_per_rubin_gw": (250_000_000.0, 6_000_000_000.0),
        "adjusted_ebitda_margin": (-0.20, 0.80),
        "exit_ebit_multiple": (2.0, 35.0),
        "exit_ebitda_multiple": (2.0, 35.0),
        "blackwell_gw_deployed": (0.0, 10.0),
        "rubin_gw_deployed": (0.0, 10.0),
    }
    low, high = bounds.get(target_metric, (0.0, 1.0))
    return solve_market_implied_driver(
        model_profile,
        base_drivers,
        historicals if historicals is not None else pd.DataFrame(),
        current_market_value,
        assumptions,
        target_metric,
        low,
        high,
        years=int(assumptions.get("forecast_years", 5)),
    )


def _transpose_schedule(frame: pd.DataFrame, rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    periods = frame["Period"].tolist() if "Period" in frame else []
    out = []
    for label, column, basis in rows:
        item = {"Line Item": label, "Assumption / basis": basis}
        for period in periods:
            match = frame[frame["Period"] == period]
            item[period] = match.iloc[0].get(column) if not match.empty and column in match else None
        out.append(item)
    return pd.DataFrame(out)


def build_cwb_style_tables(integrated_result, market: dict, assumptions: dict) -> dict[str, pd.DataFrame]:
    """
    Build clean, screenshot-friendly tables for the CWB-style valuation view.
    """
    driver = integrated_result.driver_model
    drivers = pd.DataFrame(driver.driver_forecast)
    income = pd.DataFrame(driver.income_statement)
    cash = pd.DataFrame(driver.cash_flow)
    funding = pd.DataFrame(driver.funding_schedule)
    shares = pd.DataFrame(driver.share_schedule)
    methods = pd.DataFrame([item.__dict__ for item in integrated_result.method_results])
    template = get_driver_template(getattr(integrated_result, "profile", "") or "AI Infrastructure / Data Center")

    key_inputs = pd.DataFrame(
        [
            {"Input": "Shares outstanding", "Value": assumptions.get("diluted_shares"), "Assumption / basis": "Current diluted share count; updated by equity funding and SBC dilution."},
            {"Input": "Discount rate", "Value": assumptions.get("wacc"), "Assumption / basis": "Driver WACC build feeds the DCF discount rate."},
            {"Input": "Years to discount", "Value": assumptions.get("forecast_years"), "Assumption / basis": "Explicit forecast horizon."},
            {"Input": "Current price", "Value": market.get("price"), "Assumption / basis": "Latest market snapshot."},
            {"Input": "Net debt", "Value": assumptions.get("net_debt"), "Assumption / basis": "Current net debt plus driver-based funding schedule."},
            {"Input": "Tax rate", "Value": assumptions.get("tax_rate"), "Assumption / basis": "Normalized tax rate."},
            {"Input": "Terminal / target year", "Value": drivers["Period"].iloc[-1] if not drivers.empty else None, "Assumption / basis": "Last explicit forecast year."},
        ]
    )
    capacity = _transpose_schedule(
        drivers,
        [
            ("Blackwell GW deployed", "Blackwell GW Deployed", "Ending GW by chip generation."),
            ("Rubin GW deployed", "Rubin GW Deployed", "Ending GW by chip generation."),
            ("Other GW deployed", "Other GW Deployed", "Ending non-Blackwell/Rubin GW."),
            ("Total energized GW", "Total Energized GW", "Blackwell + Rubin + Other deployed GW."),
            ("Average Blackwell GW", "Average Blackwell GW", "Average GW = (beginning + ending) / 2."),
            ("Average Rubin GW", "Average Rubin GW", "Average GW = (beginning + ending) / 2."),
            ("Average total GW", "Average Total GW", "Average Blackwell + Rubin + Other GW."),
        ],
    )
    economics = _transpose_schedule(
        drivers,
        [
            ("Revenue per Blackwell GW", "Revenue per Blackwell GW", "Revenue = average GW x revenue/GW x utilization."),
            ("Revenue per Rubin GW", "Revenue per Rubin GW", "Revenue = average GW x revenue/GW x utilization."),
            ("Hardware cost per Blackwell GW", "Hardware Cost per Blackwell GW", "Growth CAPEX input."),
            ("Hardware cost per Rubin GW", "Hardware Cost per Rubin GW", "Growth CAPEX input."),
            ("Land / power / cooling cost per GW", "Land / Power / Cooling Cost per Blackwell GW", "Infrastructure cost bucket."),
            ("Total build cost per Blackwell GW", "Total Build Cost per Blackwell GW", "Hardware + land/power/cooling per GW."),
            ("Total build cost per Rubin GW", "Total Build Cost per Rubin GW", "Hardware + land/power/cooling per GW."),
        ],
    )
    projected = _transpose_schedule(
        income.merge(cash, on="Period", how="outer", suffixes=("", "_cash")).merge(funding, on="Period", how="outer", suffixes=("", "_funding")).merge(shares, on="Period", how="outer"),
        [
            ("Revenue", "Revenue", "Average GW x revenue per GW x utilization."),
            ("Revenue % change", "Revenue Growth", "Current revenue versus prior period."),
            ("Adjusted EBITDA margin", "EBITDA Margin", "EBITDA / revenue."),
            ("Adjusted EBITDA", "Adjusted EBITDA", "Revenue x adjusted EBITDA margin."),
            ("Hardware depreciation", "Hardware Depreciation", "Average GW x hardware cost / useful life."),
            ("EBIT", "EBIT", "EBITDA - depreciation."),
            ("EBIT margin", "EBIT Margin", "EBIT / revenue."),
            ("Interest expense", "Interest Expense", "Average debt x cost of debt."),
            ("Tax", "Tax Expense", "Pretax income x tax rate when positive."),
            ("Net income", "Net Income", "Pretax income - tax."),
            ("NOPAT", "NOPAT", "EBIT x (1 - tax rate)."),
            ("OCF", "Operating Cash Flow", "EBITDA + prepayments - interest - tax - working capital."),
            ("CAPEX", "Total CAPEX", "Growth CAPEX + maintenance CAPEX."),
            ("FCF", "Free Cash Flow Before Financing", "OCF - total CAPEX."),
            ("Equity raised", "Equity Raised", "Equity funding % x growth CAPEX."),
            ("Ending net debt", "Ending Net Debt", "Debt less cash after funding."),
            ("Shares outstanding", "Diluted Shares", "Beginning shares + issued shares + SBC - buybacks."),
        ],
    )
    funding_table = _transpose_schedule(
        funding.merge(shares, on="Period", how="outer"),
        [
            ("Growth CAPEX", "Build CAPEX", "New GW x build cost per GW."),
            ("Maintenance CAPEX", "Capitalized Maintenance CAPEX", "Revenue x maintenance CAPEX %."),
            ("Total CAPEX", "Total CAPEX", "Growth CAPEX + maintenance CAPEX."),
            ("Customer prepayments", "Customer Prepayments", "Prepayments fund CAPEX; not counted as current revenue."),
            ("Operating cash flow", "Operating Cash Flow", "Cash operating output after prepayments, tax, interest, and working capital."),
            ("Free cash flow", "Free Cash Flow Before Financing", "OCF - total CAPEX."),
            ("Debt raised / repaid", "Debt Drawn", "Residual funding gap after prepayments and equity funding."),
            ("Equity raised", "Equity Raised", "Equity funding % x growth CAPEX."),
            ("Ending net debt", "Ending Net Debt", "Prior net debt adjusted for FCF and financing."),
            ("Share dilution", "Cumulative Dilution", "Ending shares / starting shares - 1."),
        ],
    )
    valuation = pd.DataFrame(
        [
            {
                "Method": row.get("method"),
                "Target metric": row.get("key_metric"),
                "Multiple / assumption": row.get("multiple_or_assumption"),
                "Enterprise value": row.get("enterprise_value"),
                "Equity value": row.get("equity_value"),
                "Value per share today": row.get("value_per_share"),
                "Assumption / basis": row.get("warning") or row.get("reason"),
            }
            for _, row in methods.iterrows()
        ]
    )
    return {
        "Key Inputs": key_inputs,
        "Driver Buildout": capacity,
        "Per-Unit Economics": economics,
        "Projected Financials": projected,
        "Cash Flow & Funding": funding_table,
        "Target-Year Valuation": valuation,
        "Template": pd.DataFrame([{"Profile": template["profile"], "Description": template["description"], "Assumption / basis": "Selected business-driver template."}]),
    }
