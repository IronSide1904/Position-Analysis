from __future__ import annotations

import pandas as pd

from data_sources.yfinance_fetcher import fetch_yfinance_snapshot


OVERRIDES = {
    "AAPL": ["MSFT", "GOOGL", "META"],
    "MSFT": ["AAPL", "GOOGL", "ORCL"],
    "NVDA": ["AMD", "AVGO", "INTC"],
    "TSLA": ["F", "GM", "RIVN"],
    "AMD": ["NVDA", "INTC", "AVGO"],
    "SPY": ["QQQ", "IWM", "DIA"],
    "QQQ": ["SPY", "XLK", "IWM"],
}


def select_peer_candidates(ticker: str, sector: str, industry: str) -> list[str]:
    """
    Select peer candidates using sector/industry and available manual overrides.
    """
    return OVERRIDES.get(ticker.upper(), [])


def build_peer_comparison(ticker: str, peers: list[str]) -> pd.DataFrame:
    """
    Build peer valuation and quality comparison.
    """
    rows = []
    for symbol in [ticker] + list(peers or []):
        snap = fetch_yfinance_snapshot(symbol)
        rows.append(
            {
                "ticker": symbol.upper(),
                "market_cap": snap.get("market_cap"),
                "enterprise_value": snap.get("enterprise_value"),
                "price": snap.get("price"),
                "beta": snap.get("beta"),
                "sector": snap.get("sector"),
                "industry": snap.get("industry"),
            }
        )
    return pd.DataFrame(rows)

