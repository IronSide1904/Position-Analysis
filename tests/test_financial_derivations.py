import pandas as pd

from models.company_story import build_company_story_summary
from models.financial_derivations import add_percentage_change_rows, derive_financial_rows, derive_revenue
from models.financial_model import build_time_axis_financial_model
from ui.components import format_dataframe_for_display


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

    assert revenue_2024 == 1290.0
    assert round(cogs_pct, 3) == 0.5
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

    assert da_display == "$1.6B"


def test_company_story_does_not_hallucinate_missing_buzz():
    story = build_company_story_summary(
        {"ticker": "TEST", "company": "Test Corp", "sector": "Technology", "industry": "Software", "company_description": "Test Corp sells workflow software to enterprises."},
        filing_texts={},
        peers=pd.DataFrame(),
    )

    assert story["buzz_context"] == "Social/news buzz unavailable."
    assert story["manual_review_questions"]
    assert any(item["assumption"] == "Revenue CAGR" for item in story["assumption_implications"])
