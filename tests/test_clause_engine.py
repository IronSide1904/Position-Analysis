import pandas as pd

from analysis.clause_classifier import classify_clause_topic
from analysis.clause_model_mapper import map_clause_to_model_lines
from analysis.clause_pipeline import run_clause_extraction_pipeline
from analysis.filing_section_splitter import split_filing_into_sections
from models.reverse_dcf import compare_clause_to_reverse_dcf


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
