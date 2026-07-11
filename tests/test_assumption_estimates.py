import pandas as pd

from models.assumption_estimates import (
    estimate_da_pct_revenue,
    estimate_growth_capex,
    estimate_maintenance_capex,
    estimate_nopat_margin,
    estimate_ocf_margin,
    estimate_sbc_pct_revenue,
    estimate_working_capital_pct_revenue,
    run_assumption_sanity_checks,
)
from models.dcf_model import default_assumptions_from_historicals


def test_missing_maintenance_capex_uses_da_proxy_not_zero():
    estimate = estimate_maintenance_capex(1000.0, 80.0, 30.0, {"profile": "Industrial / Hardware"})

    assert estimate.value == 30.0
    assert estimate.evidence_grade == "Proxy-based"
    assert estimate.confidence == "Medium"


def test_growth_capex_is_total_capex_minus_maintenance_capex():
    maintenance = estimate_maintenance_capex(1000.0, 80.0, 30.0, {"profile": "Industrial / Hardware"})
    growth = estimate_growth_capex(1000.0, 80.0, maintenance)

    assert growth.value == 50.0
    assert growth.evidence_grade in {"Calculated", "Estimated"}


def test_growth_capex_zero_warns_when_da_proxy_exceeds_total_capex():
    maintenance = estimate_maintenance_capex(1000.0, 20.0, 30.0, {"profile": "Industrial / Hardware"})
    growth = estimate_growth_capex(1000.0, 20.0, maintenance)

    assert growth.value == 0.0
    assert growth.is_real_zero is True
    assert "requires review" in growth.warning


def test_missing_sbc_uses_historical_average_not_zero():
    estimate = estimate_sbc_pct_revenue(1000.0, historical_sbc_pct=[0.12, 0.16, 0.14], business_profile={"profile": "SaaS / Software"})

    assert estimate.value == 0.14
    assert estimate.evidence_grade == "Estimated"
    assert estimate.value != 0.0


def test_missing_sbc_uses_software_profile_when_no_history():
    estimate = estimate_sbc_pct_revenue(1000.0, business_profile={"profile": "SaaS / Software"})

    assert estimate.value > 0
    assert estimate.evidence_grade == "Business-profile estimate"


def test_working_capital_reconstructs_from_balance_sheet():
    estimate = estimate_working_capital_pct_revenue(
        1000.0,
        balance_sheet={
            "receivables": 180.0,
            "inventory": 50.0,
            "payables": 90.0,
            "deferred_revenue": 40.0,
        },
    )

    assert estimate.value == 0.10
    assert estimate.evidence_grade == "Reconstructed"


def test_missing_working_capital_uses_business_profile_not_zero():
    estimate = estimate_working_capital_pct_revenue(1000.0, business_profile={"profile": "Industrial / Hardware"})

    assert estimate.value == 0.03
    assert estimate.evidence_grade == "Business-profile estimate"


def test_ocf_margin_calculates_from_ocf_and_revenue():
    estimate = estimate_ocf_margin(1000.0, operating_cash_flow=220.0)

    assert estimate.value == 0.22
    assert estimate.evidence_grade == "Calculated"


def test_ocf_margin_reconstructs_from_nopat_da_and_working_capital():
    estimate = estimate_ocf_margin(1000.0, nopat=160.0, depreciation_amortization=40.0, change_in_working_capital=20.0)

    assert estimate.value == 0.18
    assert estimate.evidence_grade == "Reconstructed"


def test_nopat_margin_calculates_from_ebit_and_tax_rate():
    estimate = estimate_nopat_margin(1000.0, ebit=200.0, tax_rate=0.21)

    assert estimate.value == 0.158
    assert estimate.evidence_grade == "Calculated"


def test_da_pct_revenue_uses_historical_average_if_current_missing():
    estimate = estimate_da_pct_revenue(1000.0, historical_da_pct=[0.03, 0.05, 0.04])

    assert estimate.value == 0.04
    assert estimate.evidence_grade == "Estimated"


def test_manual_review_only_after_all_ocf_fallbacks_fail():
    estimate = estimate_ocf_margin(None)

    assert estimate.value is None
    assert estimate.evidence_grade == "Manual review"


def test_crwd_like_software_defaults_do_not_silently_zero_sbc_or_maintenance_capex():
    historicals = pd.DataFrame(
        [
            {
                "Period": "FY 2025",
                "Revenue": 1000.0,
                "Gross Margin": 0.75,
                "EBIT": 80.0,
                "OCF": 260.0,
                "Total CAPEX": 40.0,
                "Diluted Shares": 100.0,
            }
        ]
    )

    assumptions = default_assumptions_from_historicals(historicals, {"stock_profile": "SaaS / Software"})

    assert assumptions["sbc_pct_revenue"] > 0
    assert assumptions["maintenance_capex_pct_revenue"] > 0
    assert assumptions["assumption_estimates"]["sbc_pct_revenue"]["evidence_grade"] == "Business-profile estimate"
    assert assumptions["assumption_estimates"]["maintenance_capex_pct_revenue"]["evidence_grade"] in {"Estimated", "Business-profile estimate"}


def test_sanity_checks_flag_suspicious_zeroes_and_terminal_weight():
    warnings = run_assumption_sanity_checks(
        {
            "maintenance_capex_pct_revenue": 0.0,
            "total_capex_pct_revenue": 0.05,
            "sbc_pct_revenue": 0.0,
            "working_capital_pct_revenue": 0.0,
            "depreciation_amortization_pct_revenue": 0.0,
            "growth_capex_pct_revenue": 0.0,
            "revenue_cagr": 0.12,
        },
        {"profile": "SaaS / Software"},
        {"terminal_value_weight_pct": 0.70},
    )

    metrics = {item["Metric"] for item in warnings}
    assert "Maintenance CAPEX" in metrics
    assert "SBC % Revenue" in metrics
    assert "Terminal Value Weight" in metrics
