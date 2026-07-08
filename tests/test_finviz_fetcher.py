import pandas as pd

from data_sources import finviz_fetcher


def test_missing_finviz_token_returns_empty(monkeypatch):
    monkeypatch.delenv("FINVIZ_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(finviz_fetcher, "_streamlit_secret", lambda name: None)
    monkeypatch.setattr(finviz_fetcher, "prepare_external_data_env", lambda: [])
    monkeypatch.setattr(finviz_fetcher, "load_dotenv", lambda: None)
    df = finviz_fetcher.fetch_finviz_export()
    assert df.empty
    assert "FINVIZ_AUTH_TOKEN" in df.attrs["error"]


def test_finviz_html_login_response_detected(monkeypatch):
    monkeypatch.setattr(finviz_fetcher, "get_finviz_auth_token", lambda: "token")

    class Response:
        status_code = 200
        text = "<html><form>login</form></html>"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(finviz_fetcher.requests, "get", lambda *args, **kwargs: Response())
    df = finviz_fetcher.fetch_finviz_export()
    assert df.empty
    assert "HTML" in df.attrs["error"]


def test_valid_finviz_csv_parsed(monkeypatch):
    monkeypatch.setattr(finviz_fetcher, "get_finviz_auth_token", lambda: "token")

    class Response:
        status_code = 200
        text = "Ticker,Company,Market Cap,Change\nABC,Acme,1.2B,3.4%\n"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(finviz_fetcher.requests, "get", lambda *args, **kwargs: Response())
    df = finviz_fetcher.fetch_finviz_export()
    assert df.iloc[0]["ticker"] == "ABC"
    assert df.iloc[0]["market_cap"] == 1_200_000_000
    assert round(df.iloc[0]["change"], 3) == 0.034


def test_screener_decision_url_uses_ticker_and_hides_auth_in_preview(monkeypatch):
    monkeypatch.setattr(finviz_fetcher, "get_finviz_auth_token", lambda: "secret-token")

    url, params = finviz_fetcher.build_finviz_export_url(ticker="inod")
    preview = finviz_fetcher.build_finviz_preview_url("inod")

    assert url.endswith("/export/screener")
    assert params["v"] == "152"
    assert params["t"] == "INOD"
    assert params["ft"] == "4"
    assert "0,1,2,3,4,5,6" in params["c"]
    assert params["auth"] == "secret-token"
    assert "auth=" not in preview
    assert "t=INOD" in preview


def test_decision_headers_are_normalized(monkeypatch):
    monkeypatch.setattr(finviz_fetcher, "get_finviz_auth_token", lambda: "token")

    class Response:
        status_code = 200
        text = (
            "Ticker,Company,Shs Outstand,Shs Float,Short Float,Rel Volume,"
            "Avg Volume,ATR,Beta,Price,Change,Gross Margin,Earnings Date\n"
            "ABC,Acme,1.5B,950M,12.5%,1.42,850K,2.31,1.2,42.50,-3.4%,48.2%,Jul 30 AMC\n"
        )

        def raise_for_status(self):
            return None

    monkeypatch.setattr(finviz_fetcher.requests, "get", lambda *args, **kwargs: Response())
    df = finviz_fetcher.fetch_finviz_export(ticker="ABC")
    row = df.iloc[0]
    assert row["shares_outstanding"] == 1_500_000_000
    assert row["shares_float"] == 950_000_000
    assert row["short_float"] == 0.125
    assert row["relative_volume"] == 1.42
    assert row["average_volume"] == 850_000
    assert row["change"] == -0.034
    assert round(row["gross_margin"], 3) == 0.482
    assert row["earnings_date"] == "Jul 30 AMC"


def test_finviz_numeric_market_cap_export_is_millions(monkeypatch):
    monkeypatch.setattr(finviz_fetcher, "get_finviz_auth_token", lambda: "token")

    class Response:
        status_code = 200
        text = "Ticker,Company,Market Cap,Price\nABC,Acme,4556400.04,310.23\n"

        def raise_for_status(self):
            return None

    monkeypatch.setattr(finviz_fetcher.requests, "get", lambda *args, **kwargs: Response())
    df = finviz_fetcher.fetch_finviz_export()
    assert df.iloc[0]["market_cap"] == 4_556_400_040_000


def test_snapshot_safe_failure(monkeypatch):
    empty = pd.DataFrame()
    empty.attrs["error"] = "bad token"
    monkeypatch.setattr(finviz_fetcher, "fetch_finviz_export", lambda *args, **kwargs: empty)
    snap = finviz_fetcher.fetch_finviz_ticker_snapshot("abc")
    assert snap["available"] is False
    assert snap["ticker"] == "ABC"
