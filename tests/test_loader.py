import pandas as pd

from data_sources import loader


def _market_stub():
    return {
        "ticker": "AAPL",
        "price_history": pd.DataFrame({"Close": [1.0]}),
        "finviz": {"available": False, "company": None, "sector": None, "industry": None},
        "yfinance_snapshot": {"available": True, "company": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics"},
        "market_data": {"price": 1.0, "market_cap": 10.0},
        "warnings": [],
    }


def _sec_stub():
    return {
        "available": True,
        "source": "SEC",
        "ticker": "AAPL",
        "cik": "0000320193",
        "company_name": "Apple Inc.",
        "latest_filings": [{"form": "10-K", "filing_date": "2025-01-01"}],
        "companyfacts": {},
        "submissions": {},
        "financials": {"available": True, "metrics": {}, "warnings": []},
        "warnings": [],
    }


def test_loader_does_not_fetch_deep_sec_by_default(monkeypatch):
    monkeypatch.setattr(loader, "load_market_data", lambda ticker: _market_stub())
    monkeypatch.setattr(loader, "load_sec_data_fast", lambda ticker: _sec_stub())
    monkeypatch.setattr(loader, "load_sec_deep_data", lambda ticker: (_ for _ in ()).throw(AssertionError("deep SEC should be lazy")))
    monkeypatch.setattr(loader, "fetch_yfinance_financials", lambda ticker: {"available": True, "source": "yfinance"})
    monkeypatch.setattr(loader, "extract_core_financials_from_companyfacts", lambda facts: {})

    dataset = loader.load_company_dataset("AAPL")

    assert dataset["filing_texts"] == {}
    assert dataset["cik"] == "0000320193"


def test_loader_fetches_deep_sec_when_requested(monkeypatch):
    monkeypatch.setattr(loader, "load_market_data", lambda ticker: _market_stub())
    monkeypatch.setattr(loader, "load_sec_data_fast", lambda ticker: _sec_stub())
    monkeypatch.setattr(loader, "load_sec_deep_data", lambda ticker: {"filing_texts": {"10-K": "filing text"}, "warnings": []})
    monkeypatch.setattr(loader, "fetch_yfinance_financials", lambda ticker: {"available": True, "source": "yfinance"})
    monkeypatch.setattr(loader, "extract_core_financials_from_companyfacts", lambda facts: {})

    dataset = loader.load_company_dataset("AAPL", include_deep_sec=True)

    assert dataset["filing_texts"]["10-K"] == "filing text"
