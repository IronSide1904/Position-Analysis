import pandas as pd

from analysis.clause_classifier import classify_clause_topic
from analysis.clause_model_mapper import map_clause_to_model_lines
from analysis.clause_pipeline import run_clause_extraction_pipeline
from analysis.filing_section_splitter import split_filing_into_sections
from models.reverse_dcf import compare_clause_to_reverse_dcf
from ui.dashboard_v2 import (
    ScenarioModelState,
    _apply_evidence_to_user_case,
    _evidence_priority_score,
    _unique_sorted_evidence_values,
    deduplicate_evidence_impacts,
    prepare_evidence_impacts,
)


def test_empty_filing_text_returns_empty_dataframe():
    df = run_clause_extraction_pipeline({}, {}, ticker="ABC", cik="0000000000")
    assert df.empty
    assert "No filing text available" in df.attrs["warnings"][0]


def test_section_splitter_detects_risk_factors_and_mda():
    text = """
    Item 1A. Risk Factors
    Competition may adversely affect our business.

    Item 7. Management's Discussion and Analysis
    Revenue increased due to new customers.
    """
    sections = split_filing_into_sections(text)
    assert "Risk Factors" in sections
    assert "MD&A" in sections


def test_capex_clause_maps_to_growth_capex():
    mappings = map_clause_to_model_lines(
        "CAPEX",
        "Capacity Expansion",
        "We expect higher capital expenditures for facility expansion and manufacturing capacity.",
    )
    assert any(row["model_line_affected"] == "growth_capex_pct_revenue" for row in mappings)


def test_backlog_clause_maps_to_revenue_growth():
    mappings = map_clause_to_model_lines("BACKLOG_RPO_BOOKINGS", "Backlog Increase", "Backlog increased due to orders.")
    assert any(row["model_line_affected"] == "revenue_growth" for row in mappings)


def test_sbc_clause_maps_to_diluted_shares_and_sbc():
    mappings = map_clause_to_model_lines("SBC_DILUTION_BUYBACKS", "SBC", "Stock-based compensation and RSUs increased.")
    lines = {row["model_line_affected"] for row in mappings}
    assert {"sbc", "diluted_shares"}.issubset(lines)


def test_acquisition_clause_maps_to_m_and_a_impacts():
    classified = classify_clause_topic("The acquisition increased goodwill and intangible assets.", "Notes", "M_AND_A")
    mappings = map_clause_to_model_lines(classified["topic"], classified["subtopic"], "The acquisition increased goodwill and intangible assets.")
    lines = {row["model_line_affected"] for row in mappings}
    assert "revenue_growth" in lines or "terminal_multiple" in lines


def test_debt_clause_maps_to_net_debt_and_wacc():
    mappings = map_clause_to_model_lines("DEBT_LIQUIDITY", "Debt Increase", "The company borrowed under its credit facility.")
    lines = {row["model_line_affected"] for row in mappings}
    assert {"net_debt", "wacc"}.issubset(lines)


def test_risk_factor_maps_to_risk_lines():
    mappings = map_clause_to_model_lines("RISK_FACTORS", "Competition Risk", "Competition could adversely affect margins.")
    lines = {row["model_line_affected"] for row in mappings}
    assert {"wacc", "terminal_multiple", "scenario_probability"}.issubset(lines)


def test_clause_pipeline_deduplicates_and_handles_malformed_html():
    text = """
    <html><body>
    <h1>Item 7. Management's Discussion and Analysis</h1>
    <p>We expect higher capital expenditures for facility expansion and manufacturing capacity during the next year.</p>
    <p>We expect higher capital expenditures for facility expansion and manufacturing capacity during the next year.</p>
    <p>Backlog increased due to new orders from customers and supports future revenue growth.</p>
    </body></html>
    """
    df = run_clause_extraction_pipeline(
        {"10-K": text},
        {"10-K": {"filing_date": "2026-01-01", "accession_number": "abc", "source_url": "https://example.test"}},
        ticker="ABC",
        cik="0000000000",
    )
    assert not df.empty
    assert df["clause_text"].nunique() <= len(df)
    assert "growth_capex_pct_revenue" in set(df["model_line_affected"])
    assert "revenue_growth" in set(df["model_line_affected"])


def test_reverse_dcf_clause_comparison_plain_english():
    result = compare_clause_to_reverse_dcf(
        {"model_line_affected": "revenue_growth", "direction": "Increase"},
        {"implied_revenue_cagr": 0.2, "market_case": "Bull"},
    )
    assert result["market_already_prices_this"] is True
    assert "already implies" in result["interpretation"]


def _sample_evidence_rows():
    return [
        {
            "ticker": "ABC",
            "form": "10-K",
            "filing_date": "2026-01-01",
            "accession_number": "abc",
            "section": "MD&A",
            "topic": "GUIDANCE_OUTLOOK",
            "subtopic": "Revenue guidance",
            "clause_text": "Management expects revenue growth of 15.7% next year.",
            "evidence_grade": "Guided",
            "confidence": "High",
            "model_line_affected": "revenue_growth",
            "direction": "Increase",
            "timeframe": "FY2027E",
            "suggested_assumption_change": "Apply 15.7% revenue growth after checking period mapping.",
            "dashboard_action": "Update scenario",
            "review_status": "Unreviewed",
        },
        {
            "ticker": "ABC",
            "form": "10-K",
            "filing_date": "2026-01-01",
            "accession_number": "abc",
            "section": "MD&A",
            "topic": "GUIDANCE_OUTLOOK",
            "subtopic": "Revenue guidance",
            "clause_text": "Management expects revenue growth of 15.7% next year.",
            "evidence_grade": "Guided",
            "confidence": "High",
            "model_line_affected": "revenue_growth",
            "direction": "Increase",
            "timeframe": "FY2027E",
            "suggested_assumption_change": "Apply 15.7% revenue growth after checking period mapping.",
            "dashboard_action": "Update scenario",
            "review_status": "Unreviewed",
        },
        {
            "ticker": "ABC",
            "form": "10-K",
            "filing_date": "2026-01-01",
            "accession_number": "abc",
            "section": "MD&A",
            "topic": "GUIDANCE_OUTLOOK",
            "subtopic": "Revenue guidance",
            "clause_text": "Management expects revenue growth of 15.7% next year.",
            "evidence_grade": "Guided",
            "confidence": "High",
            "model_line_affected": "ocf_margin",
            "direction": "Increase",
            "timeframe": "FY2027E",
            "suggested_assumption_change": "Cash conversion could support OCF margin.",
            "dashboard_action": "Review DCF",
            "review_status": "Unreviewed",
        },
        {
            "ticker": "ABC",
            "form": "10-K",
            "filing_date": "2026-01-01",
            "accession_number": "abc",
            "section": "Item 1A",
            "topic": "RISK_FACTORS",
            "subtopic": "Header",
            "clause_text": "Item 1A. Risk Factors",
            "evidence_grade": "Unknown",
            "confidence": "Low",
            "model_line_affected": "scenario_probability",
            "direction": "Mixed",
            "timeframe": "Unknown",
            "suggested_assumption_change": "Review manually.",
            "dashboard_action": "Manual review",
            "review_status": "Needs Review",
        },
    ]


def test_deduplicate_evidence_impacts_preserves_distinct_model_lines():
    df = pd.DataFrame(_sample_evidence_rows())
    deduped = deduplicate_evidence_impacts(df)

    assert len(deduped) == 3
    assert {"revenue_growth", "ocf_margin"}.issubset(set(deduped["model_line_affected"]))


def test_top_model_impacts_prioritize_calculated_rows_over_boilerplate():
    df = pd.DataFrame(_sample_evidence_rows())
    impacts = prepare_evidence_impacts(
        df,
        user_assumptions={"revenue_cagr": 0.08, "ocf_margin": 0.20},
        market_implied={"revenue_cagr": 0.14},
        limit=1,
    )

    assert impacts.iloc[0]["Model Line"] == "revenue_cagr"
    assert impacts.iloc[0]["Assumption Signal"] == "Calculated %"
    assert "Risk Factors" not in impacts.iloc[0]["Evidence Summary"]


def test_priority_scoring_ranks_numeric_signal_above_qualitative_only():
    numeric = {"Assumption Signal": "Calculated %", "Model Line": "revenue_cagr", "Confidence": "High", "_implied_value": 0.16}
    qualitative = {"Assumption Signal": "Qualitative Support", "Model Line": "revenue_cagr", "Confidence": "High", "_implied_value": None}

    assert _evidence_priority_score(numeric, user_value=0.08, market_value=0.14) > _evidence_priority_score(qualitative, user_value=0.08, market_value=0.14)


def test_evidence_filters_use_unique_sorted_values():
    frame = pd.DataFrame({"topic": ["B", "A", "B", None, "A"]})

    assert _unique_sorted_evidence_values(frame, "topic") == ["A", "B"]


def test_prepare_evidence_impacts_formats_user_and_market_values():
    df = pd.DataFrame(_sample_evidence_rows())
    impacts = prepare_evidence_impacts(
        df,
        user_assumptions={"revenue_cagr": 0.08, "ocf_margin": 0.25},
        market_implied={"revenue_cagr": 0.145},
        limit=10,
    )

    revenue = impacts[impacts["Model Line"] == "revenue_cagr"].iloc[0]
    assert revenue["Current User Case"] == "8.0%"
    assert revenue["Market-Implied"] == "14.5%"
    assert revenue["Delta vs User"] == "+7.7 pts"


def test_applying_evidence_updates_user_case_without_changing_base(monkeypatch):
    session = {}
    monkeypatch.setattr("ui.dashboard_v2.st.session_state", session)
    base = {"revenue_cagr": 0.08, "gross_margin": 0.45, "opex_pct_revenue": 0.25, "tax_rate": 0.21}
    state = ScenarioModelState(
        ticker="ABC",
        selected_case="User Case",
        base_assumptions=dict(base),
        user_assumptions=dict(base),
        bull_assumptions={},
        bear_assumptions={},
        market_implied_assumptions={"revenue_cagr": 0.14},
        active_assumptions=dict(base),
        model_outputs={},
        reverse_dcf_outputs={},
        assumption_change_log=[],
        evidence_links={},
        warnings=[],
    )
    row = {
        "_implied_value": 0.157,
        "_evidence_id": "abc-evidence",
        "_raw": {"timeframe": "FY2027E", "suggested_assumption_change": "Guidance midpoint / base revenue - 1"},
        "Model Line": "revenue_cagr",
        "Timeframe": "FY2027E",
        "Source": "Guidance / Outlook",
        "Confidence": "Medium",
        "Evidence Summary": "Revenue guidance implies stronger growth than current User Case.",
    }

    assert _apply_evidence_to_user_case({"dataset": {"ticker": "ABC"}}, state, row)
    assert state.base_assumptions["revenue_cagr"] == 0.08
    assert session["assumption_user_case_ABC"]["revenue_cagr"] == 0.157
    assert session["assumption_update_log"][0]["case"] == "User Case"
    assert session["assumption_update_log"][0]["status"] == "Active"
