# PA-11R Hybrid Dashboard

PA-11R Hybrid is a Streamlit dashboard for retail traders and investors. It keeps the interface simple while pulling primary-source evidence from SEC/EDGAR where possible, optional Finviz Elite enhancements, and yfinance fallback market data.

## Setup

```bash
pip install -r requirements.txt
python scripts/launch_dashboard.py --port 8504
```

The launcher normalizes duplicate Windows `Path` / `PATH` environment entries before starting Streamlit. This avoids the recurring PowerShell `Start-Process` failure: `Item has already been added. Key in dictionary: 'Path'`.

## Local Secrets

Finviz Elite is optional. Do not hardcode tokens in the codebase.

Create a local `.env` file:

```env
FINVIZ_AUTH_TOKEN=PASTE_NEW_FINVIZ_TOKEN_HERE
SEC_USER_AGENT="Your Name your.email@example.com"
```

For Streamlit Cloud, use `.streamlit/secrets.toml`:

```toml
FINVIZ_AUTH_TOKEN = "PASTE_NEW_FINVIZ_TOKEN_HERE"
SEC_USER_AGENT = "Your Name your.email@example.com"
```

The dashboard still runs without Finviz. SEC and yfinance fallbacks keep the app usable when optional data is unavailable.

## Data Priority

1. SEC/EDGAR official company filings and companyfacts
2. Finviz Elite optional snapshot fields
3. yfinance price, market data, and financial-statement fallback
4. Manual/default unavailable objects

## What The Dashboard Covers

- Company snapshot
- Thesis and scorecard
- Interactive DCF lab
- Reverse DCF
- SOTP starter model
- Filing clause and note map
- CAPEX / NOPAT / OCF quality
- Operating leverage
- M&A strategy
- Management and board
- Guidance tracker
- Compensation, SBC, and alignment
- Moat analyzer and new entrant test
- Peer comparison
- Risks and thesis breakers
- Final Buy / Watchlist / Avoid decision

## Validation

Run:

```bash
pytest tests
```

The tests focus on safe failure behavior and core model math. Network-backed SEC/yfinance behavior is intentionally isolated so the app can be validated offline.
