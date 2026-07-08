import pandas as pd

from analysis.accounting_interpreter import (
    build_accounting_interpretation,
    build_accounting_interpretation_table,
    infer_business_profile,
    interpret_capex,
    interpret_depreciation_amortization,
    interpret_nopat_quality,
    interpret_ocf_quality,
)


def _historicals(capex_values, da_values, revenue=1000, ocf=180, nopat=150, sbc=20):
    rows = []
    for i, (capex, da) in enumerate(zip(capex_values, da_values), start=1):
        rows.append(
            {
                "Period": f"FY20{i}",
                "Revenue": revenue * (1 + i * 0.05),
                "EBIT": nopat / 0.79,
                "NOPAT": nopat,
                "OCF": ocf,
                "Total CAPEX": capex,
                "D&A": da,
                "FCF": ocf - capex,
                "SBC": sbc,
            }
        )
    return pd.DataFrame(rows)


def _clauses(text: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "topic": "M_AND_A",
                "subtopic": "General",
                "clause_text": text,
                "model_line_affected": "scenario_probability",
                "suggested_assumption_change": "Review manually",
            }
        ]
    )


def test_software_acquisition_amortization_rejects_da_proxy():
    dataset = {"sector": "Technology", "industry": "Software - Application", "company": {"longBusinessSummary": "SaaS cloud subscription platform."}}
    historicals = _historicals([18, 20, 21], [95, 110, 120], sbc=80)
    clauses = _clauses("Amortization of acquired intangibles increased following an acquisition and business combination.")

    profile = infer_business_profile(dataset, clauses)
    result = interpret_depreciation_amortization(historicals, clauses, profile)

    assert profile["business_model"] == "SaaS"
    assert result["da_as_maintenance_capex_proxy"] is False
    assert "D&A may not represent maintenance CAPEX" in " ".join(result["warnings"])


def test_data_center_capacity_expansion_classifies_growth_capex():
    dataset = {"sector": "Technology", "industry": "Data Center Infrastructure"}
    historicals = _historicals([180, 220, 260], [70, 80, 90])
    clauses = _clauses("Capital expenditures increased for data center capacity expansion, new facilities, and infrastructure build-out.")

    profile = infer_business_profile(dataset, clauses)
    result = interpret_capex(historicals, clauses, profile)

    assert profile["business_model"] == "Data Center"
    assert result["classification"] in {"Growth-heavy", "Mixed"}
    assert result["growth_capex_estimate"] > 0
    assert any("growth" in warning.lower() or "CAPEX" in warning for warning in result["warnings"])


def test_asset_heavy_manufacturer_can_use_da_proxy_when_capex_tracks_da():
    dataset = {"sector": "Industrials", "industry": "Manufacturing Machinery"}
    historicals = _historicals([100, 102, 98], [95, 100, 101])
    clauses = pd.DataFrame(columns=["clause_text", "topic"])

    profile = infer_business_profile(dataset, clauses)
    result = interpret_depreciation_amortization(historicals, clauses, profile)

    assert profile["asset_intensity"] == "High"
    assert result["da_as_maintenance_capex_proxy"] is True
    assert result["recommended_maintenance_capex_method"] == "D&A proxy"


def test_inventory_and_receivables_lower_ocf_quality():
    dataset = {"sector": "Consumer Cyclical", "industry": "Hardware Retail"}
    historicals = _historicals([40, 42, 44], [40, 41, 42], ocf=60, nopat=150)
    clauses = _clauses("Inventory increased in anticipation of demand and accounts receivable increased faster than revenue.")
    profile = infer_business_profile(dataset, clauses)

    result = interpret_ocf_quality(historicals, clauses, profile)

    assert result["ocf_quality_score"] < 7
    assert any("Inventory" in flag or "Receivables" in flag for flag in result["red_flags"])


def test_material_sbc_and_acquisition_flag_nopat_quality_and_dashboard_table():
    dataset = {"sector": "Technology", "industry": "Software"}
    historicals = _historicals([20, 22, 24], [100, 115, 130], revenue=1000, sbc=90)
    clauses = _clauses("Stock-based compensation increased and acquisition integration costs included amortization of acquired intangibles.")
    profile = infer_business_profile(dataset, clauses)

    nopat = interpret_nopat_quality(historicals, clauses, profile)
    interpretation = build_accounting_interpretation(dataset, historicals, clauses)
    table = build_accounting_interpretation_table(interpretation, historicals)

    assert nopat["nopat_quality_score"] < 7
    assert any("SBC" in warning for warning in nopat["warnings"])
    assert {"Metric", "Reported Value", "Economic Interpretation", "Business Logic", "Clause Evidence", "Model Impact", "Confidence", "Suggested Action"}.issubset(table.columns)
