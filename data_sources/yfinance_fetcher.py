from __future__ import annotations

import pandas as pd

from config import YFINANCE_CONFIG
from data_sources.runtime_env import prepare_external_data_env, yfinance_cache_dir


def _configure_yfinance(yf):
    if hasattr(yf, "set_tz_cache_location"):
        yf.set_tz_cache_location(yfinance_cache_dir())


def _empty_frame(error: str = "yfinance unavailable") -> pd.DataFrame:
    df = pd.DataFrame()
    df.attrs["error"] = error
    return df


def fetch_price_history(ticker: str, period: str = "5y", interval: str = "1d") -> pd.DataFrame:
    """
    Fetch OHLCV price history.
    """
    try:
        prepare_external_data_env()
        import yfinance as yf

        _configure_yfinance(yf)
        period = period or YFINANCE_CONFIG["default_period"]
        interval = interval or YFINANCE_CONFIG["interval"]
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
        if df is None or df.empty:
            return _empty_frame("No yfinance price history")
        return df.reset_index()
    except Exception as exc:
        return _empty_frame(str(exc))


def fetch_yfinance_snapshot(ticker: str) -> dict:
    """
    Fetch price, market cap, EV, sector/industry fallback, beta, shares, cash/debt if available.
    """
    payload = {
        "available": False,
        "source": "yfinance",
        "ticker": ticker.upper(),
        "company": None,
        "sector": None,
        "industry": None,
        "price": None,
        "market_cap": None,
        "enterprise_value": None,
        "float_shares": None,
        "current_price": None,
        "cash": None,
        "debt": None,
        "shares_outstanding": None,
        "beta": None,
        "error": None,
    }
    try:
        prepare_external_data_env()
        import yfinance as yf

        _configure_yfinance(yf)
        info = yf.Ticker(ticker).get_info()
        payload.update(
            {
                "available": True,
                "company": info.get("longName") or info.get("shortName"),
                "sector": info.get("sector"),
                "industry": info.get("industry"),
                "price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "market_cap": info.get("marketCap"),
                "enterprise_value": info.get("enterpriseValue"),
                "cash": info.get("totalCash"),
                "debt": info.get("totalDebt"),
                "shares_outstanding": info.get("sharesOutstanding"),
                "float_shares": info.get("floatShares"),
                "beta": info.get("beta"),
            }
        )
    except Exception as exc:
        payload["error"] = str(exc)
    return payload


def fetch_yfinance_financials(ticker: str) -> dict:
    """
    Fetch financial statements as fallback.
    """
    try:
        prepare_external_data_env()
        import yfinance as yf

        _configure_yfinance(yf)
        stock = yf.Ticker(ticker)
        return {
            "available": True,
            "source": "yfinance",
            "income_stmt": stock.income_stmt,
            "cashflow": stock.cashflow,
            "balance_sheet": stock.balance_sheet,
            "error": None,
        }
    except Exception as exc:
        return {
            "available": False,
            "source": "yfinance",
            "income_stmt": pd.DataFrame(),
            "cashflow": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "error": str(exc),
        }
