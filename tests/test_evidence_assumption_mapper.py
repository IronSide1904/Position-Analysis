import pandas as pd

from analysis.evidence_assumption_mapper import (
    build_assumption_update_from_impact,
    build_evidence_assumption_impacts,
    calculate_implied_assumption,
    extract_numeric_guidance,
    extract_numeric_signals,
    impact_status_summary,
    map_clause_to_multi_line_impacts,
    unique_filter_values,
)


def _financials(revenue=1_000_000_000):
    return pd.DataFrame([{"Revenue": revenue, "Gross Profit": revenue * 0.7, "OCF": revenue * 0.25, "FCF": revenue * 0.2}])


def _assumptions(**overrides):
    base = {
        "revenue_cagr": 0.08,
        "gross_margin": 0.70,
        "nopat_margin": 0.12,
        "ocf_margin": 0.22,
        "wacc": 0.095,
        "terminal_multiple": 15.0,
        "maintenance_capex_pct_revenue": 0.03,
        "growth_capex_pct_revenue": 0.02,
        "total_capex_pct_revenue": 0.05,
        "sbc_pct_revenue": 0.10,
    }
    base.update(overrides)
    return base


def _clause(text, **overrides):
    row = {
        "ticker": "CRWD",
        "form": "10-K",
        "filing_date": "2026-03-15",
        "section": "MD&A",
        "topic": "GUIDANCE_OUTLOOK",
        "subtopic": "Revenue Guidance",
        "clause_text": text,
        "model_line_affected": "revenue_growth",
        "direction": "Mixed",
        "confidence": "Medium",
        "evidence_grade": "Guided",
        "review_status": "Unreviewed",
        "source_url": "https://example.test/filing",
    }
    row.update(overrides)
    return row


def test_numeric_revenue_guidance_produces_implied_revenue_growth():
    guidance = extract_numeric_guidance("Revenue expected to be between $1.2 billion and $1.3 billion in FY2027.")
    implied = calculate_implied_assumption(guidance, "revenue_growth", _financials(1_080_000_000), _assumptions())

    assert guidance["metric"] == "revenue"
    assert guidance["midpoint"] == 1_250_000_000
    assert round(implied["implied_value"], 3) == 0.157


def test_revenue_range_produces_midpoint_and_implied_range():
    impacts = build_evidence_assumption_impacts(
        [_clause("Revenue expected to be between $1.2 billion and $1.3 billion in FY2027.")],
        _assumptions(),
        _financials(1_000_000_000),
    )
    row = impacts.iloc[0]

    assert row["implied_value_display"] == "25.0%"
    assert row["assumption_signal"] == "Implied Range"
    assert row["implied_range_display"] == "20.0% - 30.0%"
    assert row["delta_display"] == "+17.0 pts"


def test_margin_guidance_produces_implied_margin():
    guidance = extract_numeric_guidance("Gross margin expected to be 72% to 74%.")
    implied = calculate_implied_assumption(guidance, "gross_margin", _financials(), _assumptions(gross_margin=0.70))

    assert guidance["metric"] == "gross_margin"
    assert implied["implied_value"] == 0.73
    assert implied["implied_range_low"] == 0.72
    assert implied["implied_range_high"] == 0.74


def test_basis_point_guidance_converts_correctly():
    guidance = extract_numeric_guidance("We expect operating income margin to expand by 150 bps.")
    implied = calculate_implied_assumption(guidance, "nopat_margin", _financials(), _assumptions(nopat_margin=0.12))

    assert guidance["bps_change"] == 150
    assert round(implied["implied_value"], 3) == 0.135


def test_capex_dollar_guidance_converts_to_capex_pct_revenue():
    guidance = extract_numeric_guidance("Capital expenditures expected to be approximately $150 million.")
    implied = calculate_implied_assumption(
        guidance,
        "total_capex_pct_revenue",
        _financials(2_000_000_000),
        _assumptions(revenue_cagr=0.0),
    )

    assert round(implied["implied_value"], 3) == 0.075


def test_sbc_dollar_guidance_converts_to_sbc_pct_revenue():
    guidance = extract_numeric_guidance("Stock-based compensation was $300 million.")
    implied = calculate_implied_assumption(
        guidance,
        "sbc_pct_revenue",
        _financials(1_500_000_000),
        _assumptions(revenue_cagr=0.0),
    )

    assert round(implied["implied_value"], 3) == 0.20


def test_qualitative_guidance_does_not_create_fake_percentage():
    guidance = extract_numeric_guidance("Management expects demand to remain strong and backlog provides visibility.")
    implied = calculate_implied_assumption(guidance, "revenue_growth", _financials(), _assumptions())

    assert guidance["unit"] == "directional"
    assert implied["implied_value"] is None


def test_current_dcf_value_is_compared_correctly():
    impacts = build_evidence_assumption_impacts(
        [_clause("We expect revenue growth of 12% to 14%.")],
        _assumptions(revenue_cagr=0.08),
        _financials(),
    )

    assert impacts.iloc[0]["current_dcf_value"] == 0.08
    assert round(impacts.iloc[0]["delta_vs_current_dcf"], 3) == 0.05


def test_delta_displays_in_percentage_points():
    impacts = build_evidence_assumption_impacts(
        [_clause("We expect revenue growth of 12% to 14%.")],
        _assumptions(revenue_cagr=0.08),
        _financials(),
    )

    assert impacts.iloc[0]["delta_display"] == "+5.0 pts"


def test_duplicate_clause_rows_are_grouped_in_default_view():
    row = _clause("We expect revenue growth of 12% to 14%.")
    impacts = build_evidence_assumption_impacts([row, dict(row)], _assumptions(), _financials())

    assert len(impacts) == 1


def test_applying_evidence_creates_assumption_update_log_entry():
    impacts = build_evidence_assumption_impacts(
        [_clause("We expect revenue growth of 12% to 14%.")],
        _assumptions(),
        _financials(),
    )
    update = build_assumption_update_from_impact(impacts.iloc[0].to_dict(), scenario="User Case", user_note="Use midpoint.")

    assert update["scenario"] == "User Case"
    assert update["model_line"] == "revenue_growth"
    assert update["new_value"] == 0.13
    assert update["status"] == "Pending"


def test_base_case_is_not_changed_unless_explicitly_selected():
    assumptions = _assumptions(revenue_cagr=0.08)
    impacts = build_evidence_assumption_impacts(
        [_clause("We expect revenue growth of 12% to 14%.")],
        assumptions,
        _financials(),
    )
    update = build_assumption_update_from_impact(impacts.iloc[0].to_dict(), scenario="User Case")

    assert assumptions["revenue_cagr"] == 0.08
    assert update["scenario"] == "User Case"


def test_filter_dropdowns_show_unique_values():
    impacts = build_evidence_assumption_impacts(
        [
            _clause("We expect revenue growth of 12% to 14%."),
            _clause("Gross margin expected to be 72% to 74%.", model_line_affected="gross_margin", subtopic="Margin Guidance"),
        ],
        _assumptions(),
        _financials(),
    )

    assert unique_filter_values(impacts, "topic_label") == ["Guidance / Outlook"]
    assert unique_filter_values(impacts, "model_line_label") == ["Gross Margin", "Revenue Growth"]


def test_numeric_revenue_guidance_does_not_return_directional_positive():
    impacts = build_evidence_assumption_impacts(
        [_clause("We expect revenue growth of 12% to 14%.")],
        _assumptions(revenue_cagr=0.08),
        _financials(),
    )

    row = impacts.iloc[0]
    assert row["implied_value_status"] in {"Calculated", "Range"}
    assert row["assumption_signal"] in {"Calculated %", "Implied Range"}
    assert "Directional" not in row["implied_value_display"]
    assert extract_numeric_signals(row["clause_text"])["has_numeric_signal"] is True


def test_revenue_range_returns_range_status():
    impacts = build_evidence_assumption_impacts(
        [_clause("Revenue expected to be between $1.2 billion and $1.3 billion in FY2027.")],
        _assumptions(revenue_cagr=0.08),
        _financials(1_000_000_000),
    )

    assert impacts.iloc[0]["implied_value_status"] == "Range"


def test_margin_basis_point_clause_returns_calculated_margin_impact():
    impacts = build_evidence_assumption_impacts(
        [_clause("Operating income margin is expected to expand by 200 bps.", model_line_affected="nopat_margin")],
        _assumptions(nopat_margin=0.12),
        _financials(),
    )

    row = impacts.iloc[0]
    assert row["implied_value_status"] == "Calculated"
    assert round(row["implied_value"], 3) == 0.14
    assert row["delta_display"] == "+2.0 pts"


def test_capex_clause_with_dollar_amount_returns_capex_pct_revenue():
    impacts = build_evidence_assumption_impacts(
        [
            _clause(
                "Capital expenditures expected to be approximately $150 million.",
                topic="CAPEX",
                subtopic="Growth CAPEX",
                model_line_affected="total_capex_pct_revenue",
            )
        ],
        _assumptions(revenue_cagr=0.0, total_capex_pct_revenue=0.05),
        _financials(2_000_000_000),
    )

    row = impacts.iloc[0]
    assert row["implied_value_status"] == "Calculated"
    assert round(row["implied_value"], 3) == 0.075


def test_capex_qualitative_expansion_clause_returns_estimated_range_not_mixed_only():
    impacts = build_evidence_assumption_impacts(
        [
            _clause(
                "We expect higher capital expenditures for data center infrastructure expansion and manufacturing capacity.",
                topic="CAPEX",
                subtopic="Capacity Expansion",
                model_line_affected="growth_capex_pct_revenue",
            )
        ],
        _assumptions(),
        _financials(),
    )

    statuses = set(impacts["implied_value_status"])
    signals = set(impacts["assumption_signal"])
    assert "Estimated Range" in statuses
    assert set(impacts["model_line"]).issuperset({"growth_capex_pct_revenue", "fcf_margin", "revenue_growth"})
    assert "Directional Only" not in statuses
    assert {"CAPEX Increase Warning", "Near-Term FCF Pressure", "Revenue Visibility Support"}.issubset(signals)


def test_sbc_qualitative_clause_uses_estimated_range_where_possible():
    impacts = build_evidence_assumption_impacts(
        [
            _clause(
                "Stock-based compensation and RSU awards increased as we hired more engineers.",
                topic="SBC_DILUTION_BUYBACKS",
                subtopic="SBC",
                model_line_affected="sbc_pct_revenue",
            )
        ],
        _assumptions(sbc_pct_revenue=0.10),
        _financials(),
    )

    sbc_row = impacts[impacts["model_line"] == "sbc_pct_revenue"].iloc[0]
    assert sbc_row["implied_value_status"] == "Estimated Range"
    assert sbc_row["assumption_signal"] == "SBC / Dilution Warning"
    assert "Directional" not in sbc_row["implied_value_display"]


def test_backlog_clause_returns_estimated_revenue_growth_range_without_direct_conversion_data():
    impacts = build_evidence_assumption_impacts(
        [
            _clause(
                "RPO grew 25% and backlog increased due to strong enterprise bookings.",
                topic="BACKLOG_RPO_BOOKINGS",
                subtopic="RPO / Contracted Revenue",
                model_line_affected="revenue_growth",
            )
        ],
        _assumptions(),
        _financials(),
    )

    revenue_row = impacts[impacts["model_line"] == "revenue_growth"].iloc[0]
    assert revenue_row["implied_value_status"] == "Estimated Range"
    assert revenue_row["assumption_signal"] == "Revenue Visibility Support"
    assert revenue_row["implied_value"] is not None
    assert revenue_row["current_dcf_value"] == 0.08
    assert revenue_row["delta_vs_current_dcf"] > 0
    assert "%" in revenue_row["implied_value_display"]
    assert set(impacts["assumption_signal"]) == {"Revenue Visibility Support", "Bull Case Support"}


def test_assumption_signal_avoids_generic_scenario_and_directional_labels():
    impacts = build_evidence_assumption_impacts(
        [
            _clause("Management expects demand to remain strong.", model_line_affected="revenue_growth"),
            _clause("Lower demand and margin pressure may continue.", model_line_affected="gross_margin", topic="MARGIN_COSTS"),
            _clause("Risk factors increased due to customer concentration.", topic="RISK_FACTORS", model_line_affected="wacc"),
        ],
        _assumptions(),
        _financials(),
    )

    forbidden = {"Scenario Support", "Directional Only", "Directional Positive", "Directional Negative", "Directional Mixed"}
    assert set(impacts["assumption_signal"]).isdisjoint(forbidden)
    assert {"Revenue Visibility Support", "Margin Pressure Warning", "Risk Increase Warning"}.issubset(set(impacts["assumption_signal"]))


def test_m_and_a_clause_returns_multiple_model_line_impacts_instead_of_one_mixed_row():
    impacts = build_evidence_assumption_impacts(
        [
            _clause(
                "The acquisition increased goodwill and intangible assets and will require integration costs.",
                topic="M_AND_A",
                subtopic="Acquisition",
                model_line_affected="scenario_probability",
            )
        ],
        _assumptions(),
        _financials(),
    )

    assert len(impacts) >= 3
    assert set(impacts["model_line"]).issuperset({"revenue_growth", "nopat_margin", "terminal_multiple"})
    assert impacts[impacts["model_line"].isin(["revenue_growth", "nopat_margin", "wacc"])]["implied_value"].notna().all()
    assert impacts[impacts["model_line"].isin(["revenue_growth", "nopat_margin", "wacc"])]["delta_vs_current_dcf"].notna().all()
    assert not (len(impacts) == 1 and impacts.iloc[0]["direction"] == "Mixed")


def test_debt_clause_returns_estimated_wacc_percentage_impact():
    impacts = build_evidence_assumption_impacts(
        [
            _clause(
                "Higher debt and liquidity risk could increase interest expense.",
                topic="DEBT_LIQUIDITY",
                subtopic="Debt Risk",
                model_line_affected="wacc",
            )
        ],
        _assumptions(wacc=0.095),
        _financials(),
    )

    row = impacts.iloc[0]
    assert row["model_line"] == "wacc"
    assert row["implied_value_status"] == "Estimated Range"
    assert row["assumption_signal"] == "Risk Increase Warning"
    assert row["implied_value"] > row["current_dcf_value"]
    assert row["delta_vs_current_dcf"] > 0
    assert "%" in row["implied_value_display"]


def test_mixed_direction_only_used_after_line_level_impacts_conflict():
    impacts = map_clause_to_multi_line_impacts(
        _clause(
            "The acquisition increased goodwill and intangible assets and will require integration costs.",
            topic="M_AND_A",
            subtopic="Acquisition",
            model_line_affected="scenario_probability",
        ),
        _financials(),
        _assumptions(),
    )

    directions = {row["direction"] for row in impacts}
    assert "Mixed" in directions
    assert {"Increase", "Decrease"}.issubset(directions)


def test_directional_only_percentage_is_reported_in_summary():
    impacts = build_evidence_assumption_impacts(
        [
            _clause("Management expects demand to remain strong.", model_line_affected="revenue_growth"),
            _clause("The business may be affected by general economic conditions.", topic="RISK_FACTORS", model_line_affected="wacc"),
        ],
        _assumptions(),
        _financials(),
    )
    summary = impact_status_summary(impacts)

    assert summary["total"] == len(impacts)
    assert "directional_only_share" in summary
