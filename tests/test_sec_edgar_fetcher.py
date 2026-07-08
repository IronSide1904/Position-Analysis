import pandas as pd

from data_sources import sec_edgar_fetcher


def test_sec_ticker_to_cik(monkeypatch):
    df = pd.DataFrame([{"ticker": "AAPL", "cik_str": "0000320193", "title": "Apple Inc."}])
    monkeypatch.setattr(sec_edgar_fetcher, "fetch_ticker_cik_map", lambda: df)
    assert sec_edgar_fetcher.get_cik_for_ticker("aapl") == "0000320193"


def test_companyfacts_missing_tags_handled():
    facts = sec_edgar_fetcher.extract_core_financials_from_companyfacts({"facts": {"us-gaap": {}}})
    assert facts["revenue"]["value"] is None
    assert facts["revenue"]["confidence"] == "unavailable"

