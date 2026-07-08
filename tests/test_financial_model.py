import pandas as pd

from models.financial_model import build_historical_financial_table


def _fact(values):
    return pd.DataFrame(
        [
            {"val": value, "fy": year, "fp": "FY", "form": "10-K", "filed": f"{year + 1}-01-31", "end": f"{year}-12-31"}
            for year, value in values.items()
        ]
    )


def test_historical_table_uses_sec_annual_companyfacts():
    dataset = {
        "market_data": {"shares_outstanding": 100.0, "cash": 50.0, "debt": 75.0},
        "financials": {
            "sec": {},
            "yfinance": {},
            "sec_normalized": {
                "metrics": {
                    "revenue": _fact({2023: 1000.0, 2024: 1200.0}),
                    "gross_profit": _fact({2023: 450.0, 2024: 600.0}),
                    "operating_income": _fact({2023: 200.0, 2024: 300.0}),
                    "net_income": _fact({2023: 150.0, 2024: 220.0}),
                    "operating_cash_flow": _fact({2023: 230.0, 2024: 330.0}),
                    "capex": _fact({2023: -50.0, 2024: -70.0}),
                    "depreciation_amortization": _fact({2023: 40.0, 2024: 60.0}),
                    "sbc": _fact({2023: 20.0, 2024: 25.0}),
                    "shares_outstanding": _fact({2023: 110.0, 2024: 105.0}),
                    "cash": _fact({2023: 80.0, 2024: 90.0}),
                    "debt_current": _fact({2023: 10.0, 2024: 20.0}),
                    "debt_noncurrent": _fact({2023: 90.0, 2024: 100.0}),
                }
            },
        },
    }

    table = build_historical_financial_table(dataset)

    assert table["Period"].tolist() == ["FY 2023", "FY 2024"]
    assert table["Revenue"].tolist() == [1000.0, 1200.0]
    assert table["Total CAPEX"].tolist() == [50.0, 70.0]
    assert table["Diluted Shares"].tolist() == [110.0, 105.0]
    assert table["Net Debt"].tolist() == [20.0, 30.0]
