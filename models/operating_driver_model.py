from __future__ import annotations

from dataclasses import dataclass, field
import math

import pandas as pd

from models.dcf_model import run_dcf


MODEL_TYPES = [
    "Standard Financial",
    "Capacity / Infrastructure",
    "Subscription / SaaS",
    "Marketplace / Transaction",
    "Manufacturing / Unit Volume",
    "Retail / Store",
    "Segment-Based",
    "Custom",
]


@dataclass
class BusinessModelProfile:
    model_type: str = "Standard Financial"
    capacity_unit_name: str | None = None
    revenue_driver_name: str | None = None
    uses_capacity_model: bool = False
    uses_unit_economics: bool = False
    uses_customer_model: bool = False
    uses_segment_model: bool = False
    capital_intensive: bool = False
    debt_funded: bool = False
    maintenance_cost_treatment: str = "Capitalized"
    depreciation_method: str = "Vintage-based"
    driver_definitions: dict = field(default_factory=dict)
    default_driver_rows: list = field(default_factory=list)
    applicable_valuation_methods: list = field(default_factory=list)


@dataclass
class OperatingAssetClass:
    name: str
    capacity_by_period: dict
    utilization_by_period: dict
    revenue_per_unit_by_period: dict
    hardware_cost_per_unit_by_period: dict
    infrastructure_cost_per_unit_by_period: dict
    hardware_useful_life: float
    infrastructure_useful_life: float
    maintenance_cost_per_unit_by_period: dict


@dataclass
class DriverModelResult:
    driver_forecast: dict
    revenue_forecast: dict
    income_statement: dict
    cash_flow: dict
    funding_schedule: dict
    debt_schedule: dict
    share_schedule: dict
    depreciation_schedule: dict
    invested_capital_schedule: dict
    roic_schedule: dict
    warnings: list
    historical_ltm: dict = field(default_factory=dict)


@dataclass
class ValuationMethodResult:
    method: str
    applicable: bool
    relevance_score: float
    value_per_share: float | None
    enterprise_value: float | None
    equity_value: float | None
    reason: str
    warning: str | None = None
    key_metric: str = ""
    multiple_or_assumption: str = ""


@dataclass
class IntegratedValuationResult:
    selected_scenario: str
    driver_model: DriverModelResult
    dcf_result: dict
    method_results: list[ValuationMethodResult]
    market_implied_result: dict
    economic_interpretation: str
    dcf_assumptions: dict


def _num(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        number = float(value)
        return number if math.isfinite(number) else default
    except (TypeError, ValueError):
        return default


def _latest_positive(historicals: pd.DataFrame | None, column: str, default: float = 0.0) -> float:
    if historicals is None or historicals.empty or column not in historicals:
        return default
    values = pd.to_numeric(historicals[column], errors="coerce").dropna()
    values = values[values > 0]
    return float(values.iloc[-1]) if not values.empty else default


def _latest(historicals: pd.DataFrame | None, column: str, default: float = 0.0) -> float:
    if historicals is None or historicals.empty or column not in historicals:
        return default
    values = pd.to_numeric(historicals[column], errors="coerce").dropna()
    return float(values.iloc[-1]) if not values.empty else default


def _latest_any(historicals: pd.DataFrame | None, columns: list[str], default: float = 0.0) -> float:
    for column in columns:
        value = _latest(historicals, column, None)
        if value is not None:
            return value
    return default


def infer_business_model_profile(dataset: dict | None) -> BusinessModelProfile:
    dataset = dataset or {}
    text = " ".join(str(dataset.get(key, "")) for key in ["sector", "industry", "company", "company_description"]).lower()
    if any(token in text for token in ["data center", "infrastructure", "telecom", "tower", "energy", "utility", "power"]):
        model_type = "Capacity / Infrastructure"
    elif any(token in text for token in ["software", "saas", "subscription", "cloud"]):
        model_type = "Subscription / SaaS"
    elif any(token in text for token in ["marketplace", "transaction", "payments", "exchange"]):
        model_type = "Marketplace / Transaction"
    elif any(token in text for token in ["manufacturing", "semiconductor", "hardware", "equipment", "production"]):
        model_type = "Manufacturing / Unit Volume"
    elif any(token in text for token in ["retail", "restaurant", "store"]):
        model_type = "Retail / Store"
    else:
        model_type = "Standard Financial"
    return build_business_model_profile(model_type)


def build_business_model_profile(model_type: str = "Standard Financial") -> BusinessModelProfile:
    model_type = model_type if model_type in MODEL_TYPES else "Standard Financial"
    capacity_names = {
        "Capacity / Infrastructure": ("Capacity", "Revenue per capacity unit"),
        "Subscription / SaaS": ("Customers", "ARPU"),
        "Marketplace / Transaction": ("Transaction volume", "Take rate revenue per unit"),
        "Manufacturing / Unit Volume": ("Production units", "Revenue per unit"),
        "Retail / Store": ("Stores", "Revenue per store"),
        "Segment-Based": ("Segment units", "Revenue per segment unit"),
        "Custom": ("Operating units", "Revenue per unit"),
        "Standard Financial": (None, None),
    }
    capacity_unit, revenue_driver = capacity_names[model_type]
    driver_rows = [
        "capacity_added",
        "utilization",
        "revenue_per_unit",
        "ebitda_margin",
        "maintenance_cost_per_unit",
        "hardware_cost_per_unit",
        "infrastructure_cost_per_unit",
        "land_cost_per_unit",
        "hardware_useful_life",
        "infrastructure_useful_life",
        "customer_prepayment_pct",
        "grant_funding_pct",
        "equity_funding_pct",
        "equity_issue_price",
        "sbc_dilution_pct",
        "share_repurchases",
        "cost_of_debt",
        "risk_free_rate",
        "beta",
        "equity_risk_premium",
        "exit_ebitda_multiple",
        "exit_ebit_multiple",
    ]
    return BusinessModelProfile(
        model_type=model_type,
        capacity_unit_name=capacity_unit,
        revenue_driver_name=revenue_driver,
        uses_capacity_model=model_type != "Standard Financial",
        uses_unit_economics=model_type in {"Capacity / Infrastructure", "Manufacturing / Unit Volume", "Retail / Store", "Custom"},
        uses_customer_model=model_type in {"Subscription / SaaS", "Marketplace / Transaction"},
        uses_segment_model=model_type == "Segment-Based",
        capital_intensive=model_type in {"Capacity / Infrastructure", "Manufacturing / Unit Volume", "Retail / Store"},
        debt_funded=model_type in {"Capacity / Infrastructure", "Energy / Commodity", "Retail / Store"},
        maintenance_cost_treatment="Capitalized" if model_type in {"Capacity / Infrastructure", "Manufacturing / Unit Volume"} else "Expensed",
        depreciation_method="Vintage-based",
        driver_definitions=DRIVER_DEFINITIONS,
        default_driver_rows=driver_rows,
        applicable_valuation_methods=["DCF / FCFF", "EBITDA Multiple", "Revenue Multiple", "ROIC / Economic Value", "Reverse DCF"],
    )


DRIVER_DEFINITIONS = {
    "capacity_added": {"label": "Capacity Added", "unit": "units", "category": "Operating Drivers", "source": "Analyst estimate", "confidence": "Low", "warning": "Sensitivity recommended."},
    "utilization": {"label": "Utilization", "unit": "%", "category": "Operating Drivers", "source": "Company/peer utilization", "confidence": "Low", "warning": "Utilization above 100% is invalid."},
    "revenue_per_unit": {"label": "Revenue per Unit", "unit": "money", "category": "Operating Drivers", "source": "Contracts / peer economics", "confidence": "Low", "warning": "Do not count customer prepayments as revenue."},
    "ebitda_margin": {"label": "EBITDA Margin", "unit": "%", "category": "Economics", "source": "Historical margin / scenario", "confidence": "Medium", "warning": "EBITDA can overstate capital-intensive economics."},
    "maintenance_cost_per_unit": {"label": "Maintenance Cost per Unit", "unit": "money", "category": "Economics", "source": "Management guidance / estimate", "confidence": "Low", "warning": "Treatment must be explicit: expensed vs capitalized."},
    "hardware_cost_per_unit": {"label": "Hardware Cost per Unit", "unit": "money", "category": "Capital", "source": "Company/supplier disclosure", "confidence": "Low", "warning": "Hardware depreciates over its own useful life."},
    "infrastructure_cost_per_unit": {"label": "Infrastructure Cost per Unit", "unit": "money", "category": "Capital", "source": "Company/industry estimate", "confidence": "Low", "warning": "Infrastructure depreciates separately."},
    "land_cost_per_unit": {"label": "Land Cost per Unit", "unit": "money", "category": "Capital", "source": "Analyst estimate", "confidence": "Low", "warning": "Land is not depreciated."},
    "hardware_useful_life": {"label": "Hardware Useful Life", "unit": "years", "category": "Economics", "source": "Accounting policy / peer", "confidence": "Low", "warning": "Short lives raise depreciation burden."},
    "infrastructure_useful_life": {"label": "Infrastructure Useful Life", "unit": "years", "category": "Economics", "source": "Accounting policy / peer", "confidence": "Low", "warning": "Do not depreciate land."},
    "customer_prepayment_pct": {"label": "Customer Prepayment % of Build CAPEX", "unit": "%", "category": "Funding", "source": "Contract terms / estimate", "confidence": "Low", "warning": "Prepayments fund CAPEX but are not current revenue."},
    "grant_funding_pct": {"label": "Grant Funding % of Build CAPEX", "unit": "%", "category": "Funding", "source": "Grant/subsidy disclosure", "confidence": "Low", "warning": "Verify collection timing."},
    "equity_funding_pct": {"label": "Equity Funding % of Build CAPEX", "unit": "%", "category": "Funding", "source": "Scenario", "confidence": "Low", "warning": "Equity funding dilutes per-share value."},
    "equity_issue_price": {"label": "Equity Issue Price", "unit": "per_share", "category": "Funding", "source": "Market price / scenario", "confidence": "Medium", "warning": "Low issue prices amplify dilution."},
    "sbc_dilution_pct": {"label": "SBC Dilution %", "unit": "%", "category": "Funding", "source": "SBC / share trend", "confidence": "Medium", "warning": "Dilution affects fair value per share."},
    "share_repurchases": {"label": "Share Repurchases", "unit": "shares", "category": "Funding", "source": "Buyback plan / scenario", "confidence": "Low", "warning": "Do not assume buybacks offset SBC without evidence."},
    "cost_of_debt": {"label": "Cost of Debt", "unit": "%", "category": "WACC", "source": "Debt schedule / market", "confidence": "Medium", "warning": "Debt growth must increase interest expense."},
    "risk_free_rate": {"label": "Risk-Free Rate", "unit": "%", "category": "WACC", "source": "Market rate", "confidence": "Medium", "warning": "Refresh periodically."},
    "beta": {"label": "Beta", "unit": "x", "category": "WACC", "source": "Market data", "confidence": "Medium", "warning": "Extreme beta can dominate WACC."},
    "equity_risk_premium": {"label": "Equity Risk Premium", "unit": "%", "category": "WACC", "source": "Market assumption", "confidence": "Medium", "warning": "ERP is a judgment input."},
    "exit_ebitda_multiple": {"label": "Exit EBITDA Multiple", "unit": "x", "category": "Valuation", "source": "Peer multiples", "confidence": "Low", "warning": "Use with caution for capital-intensive firms."},
    "exit_ebit_multiple": {"label": "Exit EBIT Multiple", "unit": "x", "category": "Valuation", "source": "Peer multiples", "confidence": "Low", "warning": "Requires positive meaningful EBIT."},
}


def period_labels(years: int = 5) -> list[str]:
    return [f"FY{year}E" if year == 1 else f"FY{year}F" for year in range(1, years + 1)]


def default_driver_matrix(profile: BusinessModelProfile, historicals: pd.DataFrame | None, market: dict, assumptions: dict, years: int = 5) -> pd.DataFrame:
    labels = period_labels(years)
    latest_revenue = _latest_positive(historicals, "Revenue", _num(market.get("market_cap")) * 0.25)
    latest_capacity = 1.0
    revenue_per_unit = latest_revenue / latest_capacity if latest_capacity else latest_revenue
    price = _num(market.get("price"), 1.0) or 1.0
    defaults = {
        "capacity_added": 0.10 if profile.uses_capacity_model else 0.0,
        "utilization": 0.75,
        "revenue_per_unit": revenue_per_unit,
        "ebitda_margin": max(_num(assumptions.get("ocf_margin"), 0.18), _num(assumptions.get("nopat_margin"), 0.12)),
        "maintenance_cost_per_unit": latest_revenue * _num(assumptions.get("maintenance_capex_pct_revenue"), 0.03),
        "hardware_cost_per_unit": latest_revenue * _num(assumptions.get("growth_capex_pct_revenue"), 0.02) * 6,
        "infrastructure_cost_per_unit": latest_revenue * _num(assumptions.get("growth_capex_pct_revenue"), 0.02) * 4,
        "land_cost_per_unit": 0.0,
        "hardware_useful_life": 5.0,
        "infrastructure_useful_life": 15.0,
        "customer_prepayment_pct": 0.0,
        "grant_funding_pct": 0.0,
        "equity_funding_pct": 0.0,
        "equity_issue_price": price,
        "sbc_dilution_pct": _num(assumptions.get("diluted_share_growth"), 0.0),
        "share_repurchases": 0.0,
        "cost_of_debt": max(_num(assumptions.get("pretax_cost_of_debt"), 0.06), 0.01),
        "risk_free_rate": 0.04,
        "beta": _num(market.get("beta"), 1.0) or 1.0,
        "equity_risk_premium": 0.055,
        "exit_ebitda_multiple": max(_num(assumptions.get("terminal_multiple"), 12.0), 1.0),
        "exit_ebit_multiple": max(_num(assumptions.get("terminal_multiple"), 12.0), 1.0),
    }
    rows = []
    for key in profile.default_driver_rows:
        definition = DRIVER_DEFINITIONS[key]
        row = {
            "row_key": key,
            "Category": definition["category"],
            "Driver": definition["label"],
            "Unit": definition["unit"],
            "Method": definition["source"],
            "Evidence Grade": "Estimated",
            "Confidence": definition["confidence"],
            "Warning": definition["warning"],
            "Historical / LTM": defaults[key],
        }
        for idx, label in enumerate(labels, start=1):
            row[label] = defaults[key] if key != "capacity_added" else defaults[key] * idx
        rows.append(row)
    return pd.DataFrame(rows)


def matrix_to_driver_inputs(matrix: pd.DataFrame, years: int = 5) -> dict:
    labels = period_labels(years)
    out = {label: {} for label in labels}
    if matrix is None or matrix.empty:
        return out
    for _, row in matrix.iterrows():
        key = row.get("row_key")
        if not key:
            continue
        for label in labels:
            out[label][key] = _num(row.get(label), None)
    return out


def _depreciation_for_vintages(vintages: list[dict], period_index: int, capex_key: str, life_key: str) -> float:
    total = 0.0
    for vintage in vintages:
        if vintage["period_index"] > period_index:
            continue
        life = max(_num(vintage.get(life_key), 1.0), 1.0)
        age = period_index - vintage["period_index"] + 1
        if age <= life:
            total += _num(vintage.get(capex_key)) / life
    return total


def run_driver_model(
    profile: BusinessModelProfile,
    driver_matrix: pd.DataFrame,
    historicals: pd.DataFrame | None,
    market: dict,
    assumptions: dict,
    *,
    years: int = 5,
    maintenance_treatment: str | None = None,
    capitalized_maintenance_pct: float = 1.0,
) -> DriverModelResult:
    labels = period_labels(years)
    inputs = matrix_to_driver_inputs(driver_matrix, years)
    treatment = maintenance_treatment or profile.maintenance_cost_treatment or "Capitalized"
    beginning_capacity = 1.0
    beginning_debt = max(_num(market.get("debt"), _latest(historicals, "Net Debt")), 0.0)
    beginning_cash = _num(market.get("cash"), 0.0)
    beginning_shares = _num(assumptions.get("diluted_shares"), _latest(historicals, "Diluted Shares", _num(market.get("shares_outstanding"))))
    invested_capital_begin = max(_latest_positive(historicals, "Revenue") * 0.75, 1.0)
    tax_rate_default = _num(assumptions.get("tax_rate"), 0.21)
    working_capital_pct = _num(assumptions.get("working_capital_pct_revenue"), 0.01)
    latest_revenue = _latest_positive(historicals, "Revenue")
    latest_ebitda = _latest_any(
        historicals,
        ["Adjusted EBITDA", "EBITDA", "Operating Income Before Depreciation"],
        latest_revenue * _num(assumptions.get("ocf_margin"), 0.18),
    )
    latest_depreciation = _latest_any(
        historicals,
        ["D&A", "Depreciation", "Depreciation & Amortization", "Depreciation and Amortization"],
        latest_revenue * _num(assumptions.get("depreciation_amortization_pct_revenue"), 0.0),
    )
    latest_ebit = _latest_any(historicals, ["EBIT", "Operating Income"], latest_ebitda - latest_depreciation)
    latest_interest = max(beginning_debt, 0.0) * _num(assumptions.get("pretax_cost_of_debt"), 0.06)
    latest_pretax = latest_ebit - latest_interest
    latest_tax = max(latest_pretax, 0.0) * tax_rate_default
    latest_net_income = _latest_any(historicals, ["Net Income", "Net income"], latest_pretax - latest_tax)
    latest_nopat = latest_ebit * (1 - tax_rate_default) if latest_ebit >= 0 else latest_ebit
    latest_ocf = _latest_any(historicals, ["OCF", "Operating Cash Flow", "Net Cash Provided By Operating Activities"], latest_revenue * _num(assumptions.get("ocf_margin"), 0.0))
    latest_build_capex = abs(_latest_any(historicals, ["Growth CAPEX", "Growth Capex"], latest_revenue * _num(assumptions.get("growth_capex_pct_revenue"), 0.0)))
    latest_maintenance_capex = abs(_latest_any(historicals, ["Maintenance CAPEX", "Maintenance Capex"], latest_revenue * _num(assumptions.get("maintenance_capex_pct_revenue"), 0.0)))
    latest_fcf = _latest_any(historicals, ["FCF", "Free Cash Flow"], latest_ocf - latest_build_capex - latest_maintenance_capex)
    market_equity_ltm = _num(market.get("market_cap"), _num(market.get("price")) * max(beginning_shares, 0.0))
    market_debt_ltm = max(beginning_debt, 0.0)
    equity_weight_ltm = market_equity_ltm / max(market_equity_ltm + market_debt_ltm, 1.0)
    debt_weight_ltm = 1 - equity_weight_ltm
    cost_of_debt_ltm = _num(assumptions.get("pretax_cost_of_debt"), 0.06)
    risk_free_rate_ltm = 0.04
    beta_ltm = _num(market.get("beta"), 1.0) or 1.0
    equity_risk_premium_ltm = 0.055
    cost_of_equity_ltm = risk_free_rate_ltm + beta_ltm * equity_risk_premium_ltm
    after_tax_cost_of_debt_ltm = cost_of_debt_ltm * (1 - tax_rate_default)
    wacc_ltm = _num(assumptions.get("wacc"), equity_weight_ltm * cost_of_equity_ltm + debt_weight_ltm * after_tax_cost_of_debt_ltm)
    roic_ltm = latest_nopat / invested_capital_begin if invested_capital_begin else None
    historical_ltm = {
        "Ending Capacity": 1.0,
        "Utilization": _num(driver_matrix.loc[driver_matrix["row_key"] == "utilization", "Historical / LTM"].iloc[0], 0.75) if driver_matrix is not None and not driver_matrix.empty and "Historical / LTM" in driver_matrix else 0.75,
        "Revenue per Unit": _num(driver_matrix.loc[driver_matrix["row_key"] == "revenue_per_unit", "Historical / LTM"].iloc[0], latest_revenue) if driver_matrix is not None and not driver_matrix.empty and "Historical / LTM" in driver_matrix else latest_revenue,
        "Revenue": latest_revenue,
        "Revenue Growth": _num(assumptions.get("revenue_cagr"), 0.0),
        "Adjusted EBITDA": latest_ebitda,
        "EBITDA Margin": latest_ebitda / latest_revenue if latest_revenue else None,
        "Maintenance Operating Expense": latest_revenue * _num(assumptions.get("maintenance_expense_pct_revenue"), 0.0),
        "Depreciation": latest_depreciation,
        "EBIT": latest_ebit,
        "EBIT Margin": latest_ebit / latest_revenue if latest_revenue else None,
        "Interest Expense": latest_interest,
        "Pretax Income": latest_pretax,
        "Tax Expense": latest_tax,
        "Net Income": latest_net_income,
        "NOPAT": latest_nopat,
        "Operating Cash Flow": latest_ocf,
        "Build CAPEX": latest_build_capex,
        "Capitalized Maintenance CAPEX": latest_maintenance_capex,
        "Free Cash Flow Before Financing": latest_fcf,
        "Customer Prepayments": 0.0,
        "Government Grants / Subsidies": 0.0,
        "Equity Raised": 0.0,
        "Debt Drawn": 0.0,
        "Ending Debt": beginning_debt,
        "Ending Net Debt": beginning_debt - beginning_cash,
        "Diluted Shares": beginning_shares,
        "Cumulative Dilution": 0.0,
        "Average Invested Capital": invested_capital_begin,
        "ROIC": roic_ltm,
        "Risk-Free Rate": risk_free_rate_ltm,
        "Beta": beta_ltm,
        "Equity Risk Premium": equity_risk_premium_ltm,
        "Cost of Equity": cost_of_equity_ltm,
        "Pretax Cost of Debt": cost_of_debt_ltm,
        "After-Tax Cost of Debt": after_tax_cost_of_debt_ltm,
        "Market Value of Equity": market_equity_ltm,
        "Market Value of Debt": market_debt_ltm,
        "Equity Weight": equity_weight_ltm,
        "Debt Weight": debt_weight_ltm,
        "WACC": wacc_ltm,
        "ROIC Spread": roic_ltm - wacc_ltm if roic_ltm is not None else None,
        "Economic Profit": latest_nopat - wacc_ltm * invested_capital_begin if roic_ltm is not None else None,
    }
    rows = []
    warnings = []
    debt = beginning_debt
    cash = beginning_cash
    shares = beginning_shares
    invested_capital = invested_capital_begin
    prior_revenue = _latest_positive(historicals, "Revenue")
    vintages: list[dict] = []
    for idx, label in enumerate(labels, start=1):
        values = inputs.get(label, {})
        added = max(_num(values.get("capacity_added")), 0.0)
        ending_capacity = beginning_capacity + added
        avg_capacity = (beginning_capacity + ending_capacity) / 2
        utilization = _num(values.get("utilization"), 0.75)
        if utilization > 1.0:
            warnings.append(f"{label}: utilization exceeds 100%.")
        utilization = min(max(utilization, 0.0), 1.5)
        revenue_per_unit = _num(values.get("revenue_per_unit"), prior_revenue or 0.0)
        revenue = avg_capacity * utilization * revenue_per_unit if profile.uses_capacity_model else (prior_revenue * (1 + _num(assumptions.get("revenue_cagr"), 0.08)) if prior_revenue else revenue_per_unit)
        ebitda_margin = _num(values.get("ebitda_margin"), _num(assumptions.get("ocf_margin"), 0.18))
        ebitda = revenue * ebitda_margin
        maintenance_amount = avg_capacity * _num(values.get("maintenance_cost_per_unit"))
        if treatment == "Expensed":
            maintenance_expense = maintenance_amount
            capitalized_maintenance = 0.0
        elif treatment == "Mixed":
            cap_pct = min(max(_num(capitalized_maintenance_pct, 0.5), 0.0), 1.0)
            capitalized_maintenance = maintenance_amount * cap_pct
            maintenance_expense = maintenance_amount * (1 - cap_pct)
        else:
            maintenance_expense = 0.0
            capitalized_maintenance = maintenance_amount
        hardware_capex = added * _num(values.get("hardware_cost_per_unit"))
        infrastructure_capex = added * _num(values.get("infrastructure_cost_per_unit"))
        land_capex = added * _num(values.get("land_cost_per_unit"))
        build_capex = hardware_capex + infrastructure_capex + land_capex
        vintages.append(
            {
                "period_index": idx,
                "hardware_capex": hardware_capex,
                "infrastructure_capex": infrastructure_capex,
                "hardware_life": max(_num(values.get("hardware_useful_life"), 5.0), 1.0),
                "infrastructure_life": max(_num(values.get("infrastructure_useful_life"), 15.0), 1.0),
            }
        )
        hardware_depr = _depreciation_for_vintages(vintages, idx, "hardware_capex", "hardware_life")
        infra_depr = _depreciation_for_vintages(vintages, idx, "infrastructure_capex", "infrastructure_life")
        depreciation = hardware_depr + infra_depr
        ebit = ebitda - depreciation - maintenance_expense
        tax_rate = tax_rate_default
        nopat = ebit * (1 - tax_rate) if ebit >= 0 else ebit
        change_wc = revenue * working_capital_pct
        customer_prepayments = build_capex * _num(values.get("customer_prepayment_pct"))
        grant_funding = build_capex * _num(values.get("grant_funding_pct"))
        equity_raised = build_capex * _num(values.get("equity_funding_pct"))
        financing_pct = _num(values.get("customer_prepayment_pct")) + _num(values.get("grant_funding_pct")) + _num(values.get("equity_funding_pct"))
        if financing_pct > 1.0:
            warnings.append(f"{label}: financing percentages exceed 100%.")
        cash_taxes = max(ebit, 0.0) * tax_rate
        pre_interest_ocf = ebitda - maintenance_expense - cash_taxes - change_wc
        pre_interest_fcf = pre_interest_ocf - build_capex - capitalized_maintenance
        funding_gap = max(-pre_interest_fcf, 0.0)
        debt_drawn = max(funding_gap - customer_prepayments - grant_funding - equity_raised, 0.0)
        debt_repaid = min(max(pre_interest_fcf, 0.0) * 0.25, debt)
        ending_debt = debt + debt_drawn - debt_repaid
        average_debt = (debt + ending_debt) / 2
        cost_of_debt = _num(values.get("cost_of_debt"), 0.06)
        interest_expense = average_debt * cost_of_debt
        pretax_income = ebit - interest_expense
        tax_expense = max(pretax_income, 0.0) * tax_rate
        net_income = pretax_income - tax_expense
        operating_cash_flow = pre_interest_ocf - interest_expense
        fcf_before_financing = operating_cash_flow - build_capex - capitalized_maintenance
        new_shares = equity_raised / max(_num(values.get("equity_issue_price"), _num(market.get("price"), 1.0)), 0.01)
        sbc_shares = shares * _num(values.get("sbc_dilution_pct"))
        repurchases = min(_num(values.get("share_repurchases")), shares + new_shares + sbc_shares)
        ending_shares = shares + new_shares + sbc_shares - repurchases
        ending_cash = cash + fcf_before_financing + customer_prepayments + grant_funding + equity_raised + debt_drawn - debt_repaid
        if ending_cash < -1e-6:
            warnings.append(f"{label}: ending cash is negative after financing.")
        ending_invested_capital = invested_capital + build_capex + capitalized_maintenance - depreciation
        average_invested_capital = (invested_capital + ending_invested_capital) / 2
        market_equity = _num(market.get("market_cap"), _num(market.get("price")) * max(shares, 0.0))
        market_debt = max(average_debt, 0.0)
        equity_weight = market_equity / max(market_equity + market_debt, 1.0)
        debt_weight = 1 - equity_weight
        cost_of_equity = _num(values.get("risk_free_rate"), 0.04) + _num(values.get("beta"), 1.0) * _num(values.get("equity_risk_premium"), 0.055)
        after_tax_cost_of_debt = cost_of_debt * (1 - tax_rate)
        wacc = equity_weight * cost_of_equity + debt_weight * after_tax_cost_of_debt
        if wacc < after_tax_cost_of_debt:
            warnings.append(f"{label}: WACC is below after-tax cost of debt.")
        roic = nopat / average_invested_capital if average_invested_capital else None
        roic_spread = roic - wacc if roic is not None else None
        row = {
            "Period": label,
            "Beginning Capacity": beginning_capacity,
            "Capacity Added": added,
            "Ending Capacity": ending_capacity,
            "Average Capacity": avg_capacity,
            "Utilization": utilization,
            "Revenue per Unit": revenue_per_unit,
            "Revenue": revenue,
            "Revenue Growth": revenue / prior_revenue - 1 if prior_revenue else None,
            "Adjusted EBITDA": ebitda,
            "EBITDA Margin": ebitda_margin,
            "Maintenance Operating Expense": maintenance_expense,
            "Capitalized Maintenance CAPEX": capitalized_maintenance,
            "Hardware CAPEX": hardware_capex,
            "Infrastructure CAPEX": infrastructure_capex,
            "Land CAPEX": land_capex,
            "Build CAPEX": build_capex,
            "Hardware Depreciation": hardware_depr,
            "Infrastructure Depreciation": infra_depr,
            "Depreciation": depreciation,
            "EBIT": ebit,
            "EBIT Margin": ebit / revenue if revenue else None,
            "Interest Expense": interest_expense,
            "Pretax Income": pretax_income,
            "Tax Expense": tax_expense,
            "Net Income": net_income,
            "NOPAT": nopat,
            "Operating Cash Flow": operating_cash_flow,
            "Free Cash Flow Before Financing": fcf_before_financing,
            "Customer Prepayments": customer_prepayments,
            "Government Grants / Subsidies": grant_funding,
            "Equity Raised": equity_raised,
            "Debt Drawn": debt_drawn,
            "Debt Repaid": debt_repaid,
            "Ending Cash": ending_cash,
            "Beginning Debt": debt,
            "Ending Debt": ending_debt,
            "Average Debt": average_debt,
            "Ending Net Debt": ending_debt - ending_cash,
            "New Shares Issued": new_shares,
            "SBC Shares": sbc_shares,
            "Share Repurchases": repurchases,
            "Diluted Shares": ending_shares,
            "Cumulative Dilution": ending_shares / beginning_shares - 1 if beginning_shares else None,
            "Invested Capital": ending_invested_capital,
            "Average Invested Capital": average_invested_capital,
            "ROIC": roic,
            "Risk-Free Rate": _num(values.get("risk_free_rate"), 0.04),
            "Beta": _num(values.get("beta"), 1.0),
            "Equity Risk Premium": _num(values.get("equity_risk_premium"), 0.055),
            "Cost of Equity": cost_of_equity,
            "Pretax Cost of Debt": cost_of_debt,
            "After-Tax Cost of Debt": after_tax_cost_of_debt,
            "Market Value of Equity": market_equity,
            "Market Value of Debt": market_debt,
            "Equity Weight": equity_weight,
            "Debt Weight": debt_weight,
            "WACC": wacc,
            "ROIC Spread": roic_spread,
            "Economic Profit": nopat - wacc * average_invested_capital if roic is not None else None,
        }
        rows.append(row)
        beginning_capacity = ending_capacity
        debt = ending_debt
        cash = ending_cash
        shares = ending_shares
        invested_capital = ending_invested_capital
        prior_revenue = revenue
    table = pd.DataFrame(rows)
    return DriverModelResult(
        driver_forecast=table[["Period", "Beginning Capacity", "Capacity Added", "Ending Capacity", "Average Capacity", "Utilization", "Revenue per Unit"]].to_dict("list"),
        revenue_forecast=table[["Period", "Revenue", "Revenue Growth"]].to_dict("list"),
        income_statement=table[["Period", "Revenue", "Adjusted EBITDA", "EBITDA Margin", "Maintenance Operating Expense", "Depreciation", "EBIT", "EBIT Margin", "Interest Expense", "Pretax Income", "Tax Expense", "Net Income", "NOPAT"]].to_dict("list"),
        cash_flow=table[["Period", "Adjusted EBITDA", "Maintenance Operating Expense", "Interest Expense", "Tax Expense", "Operating Cash Flow", "Build CAPEX", "Capitalized Maintenance CAPEX", "Free Cash Flow Before Financing"]].to_dict("list"),
        funding_schedule=table[["Period", "Build CAPEX", "Capitalized Maintenance CAPEX", "Free Cash Flow Before Financing", "Customer Prepayments", "Government Grants / Subsidies", "Equity Raised", "Debt Drawn", "Debt Repaid", "Ending Cash", "Ending Debt", "Ending Net Debt"]].to_dict("list"),
        debt_schedule=table[["Period", "Beginning Debt", "Debt Drawn", "Debt Repaid", "Ending Debt", "Average Debt", "Interest Expense"]].to_dict("list"),
        share_schedule=table[["Period", "New Shares Issued", "SBC Shares", "Share Repurchases", "Diluted Shares", "Cumulative Dilution"]].to_dict("list"),
        depreciation_schedule=table[["Period", "Hardware Depreciation", "Infrastructure Depreciation", "Depreciation", "Land CAPEX"]].to_dict("list"),
        invested_capital_schedule=table[["Period", "Invested Capital", "Average Invested Capital", "Build CAPEX", "Capitalized Maintenance CAPEX"]].to_dict("list"),
        roic_schedule=table[["Period", "NOPAT", "Average Invested Capital", "ROIC", "Risk-Free Rate", "Beta", "Equity Risk Premium", "Cost of Equity", "Pretax Cost of Debt", "After-Tax Cost of Debt", "Market Value of Equity", "Market Value of Debt", "Equity Weight", "Debt Weight", "WACC", "ROIC Spread", "Economic Profit"]].to_dict("list"),
        warnings=warnings,
        historical_ltm=historical_ltm,
    )


def driver_result_table(result: DriverModelResult) -> pd.DataFrame:
    source = pd.DataFrame(result.income_statement)
    cash = pd.DataFrame(result.cash_flow)
    funding = pd.DataFrame(result.funding_schedule)
    drivers = pd.DataFrame(result.driver_forecast)
    returns = pd.DataFrame(result.roic_schedule)
    if source.empty:
        return pd.DataFrame()
    periods = source["Period"].tolist()
    rows = []

    def add(section: str, label: str, frame: pd.DataFrame, column: str):
        row = {"Section": section, "Line Item": label}
        row["Historical / LTM"] = result.historical_ltm.get(column)
        for period in periods:
            match = frame[frame["Period"] == period]
            row[period] = match.iloc[0].get(column) if not match.empty and column in match else None
        rows.append(row)

    for label, column in [
        ("Capacity / Units", "Ending Capacity"),
        ("Utilization %", "Utilization"),
        ("Revenue per Unit", "Revenue per Unit"),
    ]:
        add("Operating Drivers", label, drivers, column)
    for label, column in [
        ("Revenue", "Revenue"),
        ("Adjusted EBITDA", "Adjusted EBITDA"),
        ("EBITDA Margin %", "EBITDA Margin"),
        ("Maintenance Expense", "Maintenance Operating Expense"),
        ("Depreciation", "Depreciation"),
        ("EBIT", "EBIT"),
        ("EBIT Margin %", "EBIT Margin"),
        ("Interest Expense", "Interest Expense"),
        ("Net Income", "Net Income"),
        ("NOPAT", "NOPAT"),
    ]:
        add("Income Statement", label, source, column)
    for label, column in [
        ("Operating Cash Flow", "Operating Cash Flow"),
        ("Build CAPEX", "Build CAPEX"),
        ("Maintenance CAPEX", "Capitalized Maintenance CAPEX"),
        ("Free Cash Flow", "Free Cash Flow Before Financing"),
    ]:
        add("Cash Flow", label, cash, column)
    for label, column in [
        ("Customer Prepayments", "Customer Prepayments"),
        ("Equity Raised", "Equity Raised"),
        ("Debt Drawn", "Debt Drawn"),
        ("Ending Debt", "Ending Debt"),
        ("Ending Net Debt", "Ending Net Debt"),
        ("Diluted Shares", "Diluted Shares"),
    ]:
        add("Funding", label, funding if column != "Diluted Shares" else pd.DataFrame(result.share_schedule), column)
    for label, column in [
        ("Invested Capital", "Average Invested Capital"),
        ("ROIC %", "ROIC"),
        ("WACC %", "WACC"),
        ("ROIC Spread %", "ROIC Spread"),
    ]:
        add("Returns", label, returns, column)
    return pd.DataFrame(rows)


def driver_assumptions_to_dcf_assumptions(base_assumptions: dict, result: DriverModelResult) -> dict:
    income = pd.DataFrame(result.income_statement)
    cash = pd.DataFrame(result.cash_flow)
    funding = pd.DataFrame(result.funding_schedule)
    shares = pd.DataFrame(result.share_schedule)
    returns = pd.DataFrame(result.roic_schedule)
    if income.empty:
        return dict(base_assumptions)
    yearly = {}
    prev_shares = _num(base_assumptions.get("diluted_shares"), None)
    for idx, period in enumerate(income["Period"].tolist(), start=1):
        inc = income[income["Period"] == period].iloc[0]
        cfo = cash[cash["Period"] == period].iloc[0]
        fund = funding[funding["Period"] == period].iloc[0]
        share = shares[shares["Period"] == period].iloc[0] if not shares.empty else {}
        roic = returns[returns["Period"] == period].iloc[0] if not returns.empty else {}
        revenue = _num(inc.get("Revenue"), None)
        diluted_shares = _num(share.get("Diluted Shares"), None)
        yearly[str(idx)] = {
            "revenue_cagr": _num(inc.get("Revenue") / income.iloc[idx - 2].get("Revenue") - 1 if idx > 1 and income.iloc[idx - 2].get("Revenue") else inc.get("Revenue Growth"), 0.0),
            "nopat_margin": _num(inc.get("NOPAT"), 0.0) / revenue if revenue else 0.0,
            "ocf_margin": _num(cfo.get("Operating Cash Flow"), 0.0) / revenue if revenue else 0.0,
            "depreciation_amortization_pct_revenue": _num(inc.get("Depreciation"), 0.0) / revenue if revenue else 0.0,
            "maintenance_capex_pct_revenue": _num(cfo.get("Capitalized Maintenance CAPEX"), 0.0) / revenue if revenue else 0.0,
            "growth_capex_pct_revenue": _num(fund.get("Build CAPEX"), 0.0) / revenue if revenue else 0.0,
            "working_capital_pct_revenue": _num(base_assumptions.get("working_capital_pct_revenue"), 0.01),
            "diluted_share_growth": diluted_shares / prev_shares - 1 if diluted_shares and prev_shares else _num(base_assumptions.get("diluted_share_growth"), 0.0),
        }
        prev_shares = diluted_shares or prev_shares
    latest = income.iloc[-1]
    latest_cash = cash.iloc[-1]
    latest_funding = funding.iloc[-1]
    latest_returns = returns.iloc[-1]
    latest_shares = shares.iloc[-1] if not shares.empty else {}
    revenue = _num(latest.get("Revenue"), None)
    out = dict(base_assumptions)
    out.update(
        {
            "forecast_assumptions_by_year": yearly,
            "revenue_cagr": yearly["1"]["revenue_cagr"],
            "nopat_margin": _num(latest.get("NOPAT"), 0.0) / revenue if revenue else out.get("nopat_margin"),
            "ocf_margin": _num(latest_cash.get("Operating Cash Flow"), 0.0) / revenue if revenue else out.get("ocf_margin"),
            "maintenance_capex_pct_revenue": _num(latest_cash.get("Capitalized Maintenance CAPEX"), 0.0) / revenue if revenue else out.get("maintenance_capex_pct_revenue"),
            "growth_capex_pct_revenue": _num(latest_funding.get("Build CAPEX"), 0.0) / revenue if revenue else out.get("growth_capex_pct_revenue"),
            "depreciation_amortization_pct_revenue": _num(latest.get("Depreciation"), 0.0) / revenue if revenue else out.get("depreciation_amortization_pct_revenue"),
            "diluted_shares": _num(latest_shares.get("Diluted Shares"), out.get("diluted_shares")),
            "net_debt": _num(latest_funding.get("Ending Net Debt"), out.get("net_debt")),
            "wacc": _num(latest_returns.get("WACC"), out.get("wacc")),
            "use_direct_nopat_override": True,
        }
    )
    out["total_capex_pct_revenue"] = _num(out.get("maintenance_capex_pct_revenue")) + _num(out.get("growth_capex_pct_revenue"))
    return out


def _value_per_share(equity_value, shares) -> float | None:
    shares = _num(shares, None)
    equity_value = _num(equity_value, None)
    if shares is None or shares <= 0 or equity_value is None or equity_value <= 0:
        return None
    return equity_value / shares


def build_valuation_method_results(
    driver_result: DriverModelResult,
    dcf_result: dict,
    market: dict,
    assumptions: dict,
    exit_ebitda_multiple: float,
    exit_ebit_multiple: float,
    revenue_multiple: float = 3.0,
    earnings_multiple: float = 20.0,
) -> list[ValuationMethodResult]:
    income = pd.DataFrame(driver_result.income_statement)
    funding = pd.DataFrame(driver_result.funding_schedule)
    shares_df = pd.DataFrame(driver_result.share_schedule)
    returns = pd.DataFrame(driver_result.roic_schedule)
    if income.empty:
        return []
    final = income.iloc[-1]
    final_funding = funding.iloc[-1]
    final_shares = _num(shares_df.iloc[-1].get("Diluted Shares"), _num(assumptions.get("diluted_shares")))
    net_debt = _num(final_funding.get("Ending Net Debt"))
    methods = [
        ValuationMethodResult(
            "DCF / FCFF",
            dcf_result.get("fair_value_per_share") is not None,
            0.90,
            dcf_result.get("fair_value_per_share"),
            dcf_result.get("enterprise_value"),
            dcf_result.get("equity_value"),
            "Primary intrinsic value framework.",
            "Terminal value dominates valuation." if _num(dcf_result.get("terminal_value_weight_pct")) > 0.75 else None,
            "FCFF",
            "Calculated",
        )
    ]
    ebit_ev = _num(final.get("EBIT")) * exit_ebit_multiple
    ebit_equity = ebit_ev - net_debt
    methods.append(
        ValuationMethodResult(
            "EBIT Multiple",
            _num(final.get("EBIT")) > 0 and ebit_equity > 0,
            0.65 if _num(final.get("EBIT")) > 0 else 0.0,
            _value_per_share(ebit_equity, final_shares),
            ebit_ev if _num(final.get("EBIT")) > 0 else None,
            ebit_equity if ebit_equity > 0 else None,
            "Positive EBIT and positive equity value required.",
            "No positive equity value under this method." if _num(final.get("EBIT")) > 0 and ebit_equity <= 0 else "Not applicable: EBIT is negative." if _num(final.get("EBIT")) <= 0 else None,
            "Terminal EBIT",
            f"{exit_ebit_multiple:.1f}x",
        )
    )
    ebitda_ev = _num(final.get("Adjusted EBITDA")) * exit_ebitda_multiple
    ebitda_equity = ebitda_ev - net_debt
    capex_intensity = _num(final_funding.get("Build CAPEX")) / _num(final.get("Revenue"), 1.0)
    methods.append(
        ValuationMethodResult(
            "EBITDA Multiple",
            _num(final.get("Adjusted EBITDA")) > 0 and ebitda_equity > 0,
            0.45 if capex_intensity > 0.20 else 0.70,
            _value_per_share(ebitda_equity, final_shares),
            ebitda_ev,
            ebitda_equity if ebitda_equity > 0 else None,
            "Cross-check valuation using terminal EBITDA.",
            "EBITDA may overstate economics for a capital-intensive company." if capex_intensity > 0.20 else None,
            "Terminal EBITDA",
            f"{exit_ebitda_multiple:.1f}x",
        )
    )
    net_income = _num(final.get("Net Income"))
    earnings_equity = net_income * earnings_multiple
    methods.append(
        ValuationMethodResult(
            "Earnings / P-E",
            net_income > 0,
            0.55 if net_income > 0 else 0.0,
            _value_per_share(earnings_equity, final_shares),
            None,
            earnings_equity if earnings_equity > 0 else None,
            "Useful only with positive normalized net income.",
            None if net_income > 0 else "Not applicable: the scenario produces negative net income.",
            "Terminal Net Income",
            f"{earnings_multiple:.1f}x",
        )
    )
    revenue_ev = _num(final.get("Revenue")) * revenue_multiple
    methods.append(
        ValuationMethodResult(
            "Revenue Multiple",
            revenue_ev - net_debt > 0,
            0.35,
            _value_per_share(revenue_ev - net_debt, final_shares),
            revenue_ev,
            revenue_ev - net_debt if revenue_ev - net_debt > 0 else None,
            "Secondary cross-check when profitability is immature.",
            "Revenue multiples ignore capital intensity and funding burden.",
            "Terminal Revenue",
            f"{revenue_multiple:.1f}x",
        )
    )
    final_return = returns.iloc[-1]
    spread = _num(final_return.get("ROIC Spread"), None)
    relevance = 0.85 if capex_intensity > 0.20 or (spread is not None and spread < 0.03) else 0.55
    methods.append(
        ValuationMethodResult(
            "ROIC / Economic Value",
            True,
            relevance,
            None,
            None,
            None,
            roic_status(spread),
            None,
            "ROIC vs WACC",
            f"{_num(final_return.get('ROIC')):.1%} vs {_num(final_return.get('WACC')):.1%}",
        )
    )
    methods.append(
        ValuationMethodResult(
            "Reverse DCF",
            True,
            0.75,
            _num(market.get("price"), None),
            None,
            None,
            "Compares market price with required operating drivers.",
            None,
            "Market Price",
            "Solver",
        )
    )
    return methods


def roic_status(spread: float | None) -> str:
    if spread is None:
        return "ROIC unavailable."
    if spread > 0.03:
        return "Strong value creation"
    if spread >= 0:
        return "Marginal value creation"
    if spread >= -0.03:
        return "Weak economics"
    return "Value destructive"


def valuation_methods_table(methods: list[ValuationMethodResult]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Method": item.method,
                "Key Metric": item.key_metric,
                "Multiple / Assumption": item.multiple_or_assumption,
                "Enterprise Value": item.enterprise_value,
                "Equity Value": item.equity_value if item.equity_value is not None else "No positive equity value under this method" if item.enterprise_value is not None and not item.applicable else None,
                "Value / Share": item.value_per_share,
                "Relevance": item.relevance_score,
                "Status": "Applicable" if item.applicable else item.warning or "Not decision-useful",
            }
            for item in methods
        ]
    )


def build_wacc_table(driver_result: DriverModelResult, market: dict) -> pd.DataFrame:
    returns = pd.DataFrame(driver_result.roic_schedule)
    debt = pd.DataFrame(driver_result.debt_schedule)
    if returns.empty or debt.empty:
        return pd.DataFrame()
    rows = []
    if driver_result.historical_ltm:
        rows.append(
            {
                "Period": "Historical / LTM",
                "Risk-Free Rate": driver_result.historical_ltm.get("Risk-Free Rate"),
                "Beta": driver_result.historical_ltm.get("Beta"),
                "Equity Risk Premium": driver_result.historical_ltm.get("Equity Risk Premium"),
                "Cost of Equity": driver_result.historical_ltm.get("Cost of Equity"),
                "Pretax Cost of Debt": driver_result.historical_ltm.get("Pretax Cost of Debt"),
                "After-Tax Cost of Debt": driver_result.historical_ltm.get("After-Tax Cost of Debt"),
                "Equity Weight": driver_result.historical_ltm.get("Equity Weight"),
                "Debt Weight": driver_result.historical_ltm.get("Debt Weight"),
                "Average Debt": driver_result.historical_ltm.get("Market Value of Debt"),
                "WACC": driver_result.historical_ltm.get("WACC"),
            }
        )
    for _, row in returns.iterrows():
        debt_row = debt[debt["Period"] == row["Period"]].iloc[0]
        rows.append(
            {
                "Period": row["Period"],
                "Risk-Free Rate": row.get("Risk-Free Rate"),
                "Beta": row.get("Beta"),
                "Equity Risk Premium": row.get("Equity Risk Premium"),
                "Cost of Equity": row.get("Cost of Equity"),
                "Pretax Cost of Debt": row.get("Pretax Cost of Debt"),
                "After-Tax Cost of Debt": row.get("After-Tax Cost of Debt"),
                "Equity Weight": row.get("Equity Weight"),
                "Debt Weight": row.get("Debt Weight"),
                "Average Debt": debt_row.get("Average Debt"),
                "WACC": row.get("WACC"),
            }
        )
    return pd.DataFrame(rows)


def build_economic_interpretation(result: DriverModelResult, methods: list[ValuationMethodResult]) -> str:
    income = pd.DataFrame(result.income_statement)
    funding = pd.DataFrame(result.funding_schedule)
    returns = pd.DataFrame(result.roic_schedule)
    if income.empty or funding.empty or returns.empty:
        return "Driver model unavailable; use the Standard Financial DCF."
    final_income = income.iloc[-1]
    final_funding = funding.iloc[-1]
    final_return = returns.iloc[-1]
    cumulative_growth_capital = pd.to_numeric(funding["Build CAPEX"], errors="coerce").fillna(0).sum()
    primary_methods = ", ".join(item.method for item in methods if item.relevance_score >= 0.75) or "DCF"
    return (
        f"The selected scenario requires {cumulative_growth_capital:,.0f} of cumulative growth capital. "
        f"Projected terminal NOPAT is {final_income.get('NOPAT', 0):,.0f}. "
        f"ROIC is {_num(final_return.get('ROIC')):.1%} versus WACC of {_num(final_return.get('WACC')):.1%}, "
        f"which indicates {roic_status(final_return.get('ROIC Spread')).lower()}. "
        f"The most relevant frameworks are {primary_methods}."
    )


def integrate_driver_valuation(
    selected_scenario: str,
    profile: BusinessModelProfile,
    driver_matrix: pd.DataFrame,
    historicals: pd.DataFrame,
    market: dict,
    base_assumptions: dict,
    *,
    maintenance_treatment: str | None = None,
    capitalized_maintenance_pct: float = 1.0,
) -> IntegratedValuationResult:
    driver_result = run_driver_model(profile, driver_matrix, historicals, market, base_assumptions, years=int(base_assumptions.get("forecast_years", 5)), maintenance_treatment=maintenance_treatment, capitalized_maintenance_pct=capitalized_maintenance_pct)
    dcf_assumptions = driver_assumptions_to_dcf_assumptions(base_assumptions, driver_result)
    dcf_result = run_dcf(historicals, market, dcf_assumptions)
    matrix_inputs = matrix_to_driver_inputs(driver_matrix, int(base_assumptions.get("forecast_years", 5)))
    last_label = period_labels(int(base_assumptions.get("forecast_years", 5)))[-1]
    last_inputs = matrix_inputs.get(last_label, {})
    methods = build_valuation_method_results(
        driver_result,
        dcf_result,
        market,
        dcf_assumptions,
        _num(last_inputs.get("exit_ebitda_multiple"), 12.0),
        _num(last_inputs.get("exit_ebit_multiple"), 12.0),
    )
    return IntegratedValuationResult(
        selected_scenario=selected_scenario,
        driver_model=driver_result,
        dcf_result=dcf_result,
        method_results=methods,
        market_implied_result={},
        economic_interpretation=build_economic_interpretation(driver_result, methods),
        dcf_assumptions=dcf_assumptions,
    )


def solve_market_implied_driver(
    profile: BusinessModelProfile,
    driver_matrix: pd.DataFrame,
    historicals: pd.DataFrame,
    market: dict,
    assumptions: dict,
    driver_key: str,
    low: float,
    high: float,
    *,
    years: int = 5,
) -> dict:
    target_price = _num(market.get("price"), None)
    if target_price is None:
        return {"driver": driver_key, "status": "Unavailable", "required_value": None}
    labels = period_labels(years)

    def price_at(value: float) -> float | None:
        trial = driver_matrix.copy()
        trial.loc[trial["row_key"] == driver_key, labels] = value
        integrated = integrate_driver_valuation("Market-Implied", profile, trial, historicals, market, assumptions)
        return integrated.dcf_result.get("fair_value_per_share")

    low_price = price_at(low)
    high_price = price_at(high)
    if low_price is None or high_price is None or (low_price - target_price) * (high_price - target_price) > 0:
        return {"driver": driver_key, "status": "Outside reasonable range", "required_value": None, "low": low, "high": high}
    lo, hi = low, high
    for _ in range(32):
        mid = (lo + hi) / 2
        mid_price = price_at(mid)
        if mid_price is None:
            break
        if (low_price - target_price) * (mid_price - target_price) <= 0:
            hi = mid
            high_price = mid_price
        else:
            lo = mid
            low_price = mid_price
    return {"driver": driver_key, "status": "Solved", "required_value": (lo + hi) / 2, "low": low, "high": high}
