import pandas as pd

from models.company_story import build_company_story_summary
from models.financial_derivations import add_percentage_change_rows, derive_financial_rows, derive_revenue
from models.financial_model import build_time_axis_financial_model
from ui.components import format_dataframe_for_display
from ui.dashboard_v2 import _build_assumption_matrix


def test_derive_revenue_uses_sign_aware_cogs():
    assert derive_revenue(645.0, -645.0) == 1290.0
    assert derive_revenue(645.0, 645.0) == 1290.0


def test_derive_financial_rows_repairs_zero_revenue_and_adds_change_rows():
    table = pd.DataFrame(
        [
            {"Line Item": "Revenue", "FY2024A": 0.0, "FY2025A": 1500.0},
            {"Line Item": "COGS / Cost of sales", "FY2024A": -645.0, "FY2025A": 700.0},
            {"Line Item": "Gross profit", "FY2024A": 645.0, "FY2025A": 800.0},
            {"Line Item": "EBIT", "FY2024A": 200.0, "FY2025A": 250.0},
            {"Line Item": "Operating cash flow", "FY2024A": 260.0, "FY2025A": 300.0},
            {"Line Item": "Total CAPEX", "FY2024A": 40.0, "FY2025A": 50.0},
        ]
    )

    derived, log = derive_financial_rows(table)

    revenue_2024 = derived.loc[derived["Line Item"] == "Revenue", "FY2024A"].iloc[0]
    cogs_pct = derived.loc[derived["Line Item"] == "COGS % revenue", "FY2024A"].iloc[0]
    fcf = derived.loc[derived["Line Item"] == "FCF", "FY2024A"].iloc[0]
    cogs_2025 = derived.loc[derived["Line Item"] == "COGS / Cost of sales", "FY2025A"].iloc[0]

    assert revenue_2024 == 1290.0
    assert round(cogs_pct, 3) == 0.5
    assert cogs_2025 == -700.0
    assert fcf == 220.0
    assert "Revenue % change" in derived["Line Item"].tolist()
    assert any(row["Line item"] == "Revenue" and row["Period"] == "FY2024A" for row in log)


def test_percentage_change_rows_skip_margin_rows():
    table = pd.DataFrame(
        [
            {"Line Item": "Revenue", "FY2024A": 100.0, "FY2025A": 125.0},
            {"Line Item": "Gross margin %", "FY2024A": 0.4, "FY2025A": 0.42},
        ]
    )

    with_changes = add_percentage_change_rows(table)

    assert "Revenue % change" in with_changes["Line Item"].tolist()
    assert "Gross margin % % change" not in with_changes["Line Item"].tolist()
    assert with_changes.loc[with_changes["Line Item"] == "Revenue % change", "FY2025A"].iloc[0] == 0.25
    assert pd.isna(with_changes.loc[with_changes["Line Item"] == "Revenue % change", "FY2024A"].iloc[0])


def test_time_axis_model_formats_da_as_money_not_raw_number():
    historicals = pd.DataFrame(
        [
            {
                "Period": "FY 2025",
                "Revenue": 10_000_000_000.0,
                "Gross Profit": 5_000_000_000.0,
                "Gross Margin": 0.5,
                "OPEX": 2_000_000_000.0,
                "EBITDA": 1_200_000_000.0,
                "EBIT": 1_000_000_000.0,
                "NOPAT": 790_000_000.0,
                "OCF": 1_100_000_000.0,
                "Adjusted OCF": 1_100_000_000.0,
                "Maintenance CAPEX": 100_000_000.0,
                "Growth CAPEX": 100_000_000.0,
                "Total CAPEX": 200_000_000.0,
                "FCF": 900_000_000.0,
                "Adjusted FCF": 1_000_000_000.0,
                "SBC": 50_000_000.0,
                "Diluted Shares": 100_000_000.0,
                "Net Debt": 0.0,
            }
        ]
    )
    forecast = pd.DataFrame(
        [{"Year": 1, "Revenue": 11_000_000_000.0, "D&A": 1_550_151_594.0, "NOPAT": 1_000_000_000.0, "OCF": 1_200_000_000.0, "Maintenance CAPEX": 200_000_000.0, "Growth CAPEX": 150_000_000.0, "CAPEX": 350_000_000.0, "FCF": 850_000_000.0, "Diluted Shares": 100_000_000.0}]
    )
    assumptions = {"tax_rate": 0.21, "gross_margin": 0.5}

    model = build_time_axis_financial_model(historicals, forecast, assumptions)
    display = format_dataframe_for_display(model)
    da_display = display.loc[display["Line Item"] == "D&A", "FY2026E"].iloc[0]
    cogs_actual = model.loc[model["Line Item"] == "COGS / Cost of sales", "FY2025A"].iloc[0]
    cogs_forecast = model.loc[model["Line Item"] == "COGS / Cost of sales", "FY2026E"].iloc[0]

    assert da_display == "$1.6B"
    assert cogs_actual < 0
    assert cogs_forecast < 0


def test_time_axis_first_visible_change_uses_hidden_prior_period():
    historicals = pd.DataFrame(
        [
            {"Period": "FY 2020", "Revenue": 80.0, "Gross Profit": 40.0, "Gross Margin": 0.5, "OPEX": 10.0, "EBIT": 30.0, "EBITDA": 32.0, "NOPAT": 24.0, "OCF": 25.0, "Total CAPEX": 5.0, "FCF": 20.0},
            {"Period": "FY 2021", "Revenue": 100.0, "Gross Profit": 50.0, "Gross Margin": 0.5, "OPEX": 12.0, "EBIT": 38.0, "EBITDA": 40.0, "NOPAT": 30.0, "OCF": 31.0, "Total CAPEX": 6.0, "FCF": 25.0},
            {"Period": "FY 2022", "Revenue": 110.0, "Gross Profit": 55.0, "Gross Margin": 0.5, "OPEX": 13.0, "EBIT": 42.0, "EBITDA": 44.0, "NOPAT": 33.0, "OCF": 34.0, "Total CAPEX": 7.0, "FCF": 27.0},
            {"Period": "FY 2023", "Revenue": 121.0, "Gross Profit": 60.5, "Gross Margin": 0.5, "OPEX": 14.0, "EBIT": 46.5, "EBITDA": 49.0, "NOPAT": 36.0, "OCF": 37.0, "Total CAPEX": 7.5, "FCF": 29.5},
            {"Period": "FY 2024", "Revenue": 133.1, "Gross Profit": 66.55, "Gross Margin": 0.5, "OPEX": 15.0, "EBIT": 51.55, "EBITDA": 54.0, "NOPAT": 40.0, "OCF": 41.0, "Total CAPEX": 8.0, "FCF": 33.0},
            {"Period": "FY 2025", "Revenue": 146.41, "Gross Profit": 73.205, "Gross Margin": 0.5, "OPEX": 16.0, "EBIT": 57.205, "EBITDA": 60.0, "NOPAT": 45.0, "OCF": 46.0, "Total CAPEX": 9.0, "FCF": 37.0},
        ]
    )

    model = build_time_axis_financial_model(historicals, pd.DataFrame(), {"tax_rate": 0.21})
    revenue_change = model.loc[model["Line Item"] == "Revenue % change", "FY2021A"].iloc[0]
    gross_profit_change = model.loc[model["Line Item"] == "Gross profit % change", "FY2021A"].iloc[0]

    assert "__prior_period_for_change" not in model.columns
    assert revenue_change == 0.25
    assert gross_profit_change == 0.25


def test_mixed_object_columns_keep_numeric_formatting():
    frame = pd.DataFrame(
        [
            {"Case": "Base Case", "CAPEX % Revenue": 0.030553079216937677, "Fair Value / Share": 121.7, "Revenue": 312_120_750_000.0},
            {"Case": "Market-Implied Case", "CAPEX % Revenue": "Not solved", "Fair Value / Share": 315.32, "Revenue": "Not solved"},
        ]
    )

    display = format_dataframe_for_display(frame)

    assert display.loc[0, "CAPEX % Revenue"] == "3.1%"
    assert display.loc[0, "Fair Value / Share"] == "$121.70"
    assert display.loc[0, "Revenue"] == "$312.1B"


def test_assumption_matrix_fills_historical_proxy_rows():
    model_table = pd.DataFrame(
        [
            {"Line Item": "Revenue % change", "FY2025A": 0.12, "LTM Latest": 0.0},
            {"Line Item": "COGS % revenue", "FY2025A": 0.53, "LTM Latest": 0.53},
            {"Line Item": "OPEX % revenue", "FY2025A": 0.15, "LTM Latest": 0.15},
            {"Line Item": "Tax rate", "FY2025A": 0.21, "LTM Latest": 0.21},
            {"Line Item": "NOPAT margin %", "FY2025A": 0.12, "LTM Latest": 0.12},
            {"Line Item": "OCF margin %", "FY2025A": 0.16, "LTM Latest": 0.16},
            {"Line Item": "D&A % revenue", "FY2025A": 0.03, "LTM Latest": 0.03},
            {"Line Item": "Total CAPEX % revenue", "FY2025A": 0.05, "LTM Latest": 0.05},
        ]
    )
    assumptions = {
        "forecast_years": 1,
        "revenue_cagr": 0.08,
        "gross_margin": 0.47,
        "opex_pct_revenue": 0.15,
        "tax_rate": 0.21,
        "nopat_margin": 0.12,
        "ocf_margin": 0.16,
        "depreciation_amortization_pct_revenue": 0.03,
        "maintenance_capex_pct_revenue": 0.03,
        "growth_capex_pct_revenue": 0.02,
        "working_capital_pct_revenue": 0.01,
        "sbc_pct_revenue": 0.0,
        "diluted_share_growth": 0.02,
    }

    matrix, _, actual_labels = _build_assumption_matrix(assumptions, pd.DataFrame([{"Period": "FY 2025"}]), model_table)
    actual_cells = matrix[actual_labels]

    assert not actual_cells.isna().any().any()
    assert "None" not in actual_cells.astype(str).to_string()
    assert matrix.loc[matrix["Row Key"] == "revenue_cagr", "FY2025A"].iloc[0] == 12.0
    assert matrix.loc[matrix["Row Key"] == "maintenance_capex_pct_revenue", "FY2025A"].iloc[0] == 3.0
    assert matrix.loc[matrix["Row Key"] == "growth_capex_pct_revenue", "FY2025A"].iloc[0] == 2.0
    assert matrix.loc[matrix["Row Key"] == "working_capital_pct_revenue", "FY2025A"].iloc[0] == 1.0
    assert matrix.loc[matrix["Row Key"] == "sbc_pct_revenue", "FY2025A"].iloc[0] == 0.0
    assert matrix.loc[matrix["Row Key"] == "diluted_share_growth", "LTM Latest"].iloc[0] == 2.0


def test_company_story_does_not_hallucinate_missing_buzz():
    story = build_company_story_summary(
        {"ticker": "TEST", "company": "Test Corp", "sector": "Technology", "industry": "Software", "company_description": "Test Corp sells workflow software to enterprises."},
        filing_texts={},
        peers=pd.DataFrame(),
    )

    assert story["buzz_context"] == "Social/news buzz unavailable."
    assert story["manual_review_questions"]
    assert any(item["assumption"] == "Revenue CAGR" for item in story["assumption_implications"])


def test_company_story_product_story_prefers_company_description_over_filing_boilerplate():
    story = build_company_story_summary(
        {
            "ticker": "TEST",
            "company": "Test Corp",
            "sector": "Technology",
            "industry": "Data Services",
            "company_description": "Test Corp provides AI data engineering services and workflow software for enterprise customers.",
        },
        filing_texts={"10-K": "UNITED STATES SECURITIES AND EXCHANGE COMMISSION. Cover page and exchange act boilerplate."},
        peers=pd.DataFrame(),
    )

    assert "AI data engineering services" in story["product_story"]
    assert "SECURITIES AND EXCHANGE COMMISSION" not in story["product_story"]


def test_company_story_uses_sec_business_section_when_description_missing():
    filing = """
    UNITED STATES SECURITIES AND EXCHANGE COMMISSION

    Item 1. Business
    Test Corp sells labeled data, model evaluation, and enterprise workflow services to large technology customers.

    Item 1A. Risk Factors
    Demand can change.
    """
    story = build_company_story_summary(
        {"ticker": "TEST", "company": "Test Corp", "sector": "Technology", "industry": "Data Services"},
        filing_texts={"10-K": filing},
        peers=pd.DataFrame(),
    )

    assert "labeled data" in story["product_story"]
    assert "SECURITIES AND EXCHANGE COMMISSION" not in story["product_story"]


def test_company_story_summary_is_compact():
    long_description = " ".join(["This company provides a broad enterprise AI data platform with services and products."] * 30)
    story = build_company_story_summary(
        {
            "ticker": "TEST",
            "company": "Test Corp",
            "sector": "Technology",
            "industry": "Software",
            "company_description": long_description,
        },
        filing_texts={"10-K": " ".join(["The company sells software, data services, and managed workflows."] * 30)},
        peers=pd.DataFrame(),
    )

    assert len(story["business_summary"]) <= 323
    assert len(story["product_story"]) <= 363
    assert len(story["industry_positioning"]) <= 363
