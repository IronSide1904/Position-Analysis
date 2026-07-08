import math

from ui.formatting import (
    format_market_summary_value,
    fmt_dollar,
    fmt_dollar_billions,
    fmt_dollar_millions,
    fmt_multiple,
    fmt_number,
    fmt_percent,
    fmt_per_share,
    fmt_ratio,
    fmt_score,
    fmt_shares,
    fmt_volume,
)


def test_financial_display_formatters():
    assert fmt_dollar(1_000_000) == "$1,000,000"
    assert fmt_dollar(1_250_000_000, scale="M") == "$1,250M"
    assert fmt_dollar_millions(2_450_000_000) == "$2,450M"
    assert fmt_dollar_billions(2_836_180_000_000) == "$2,836.18B"
    assert fmt_per_share(15.423) == "$15.42"
    assert fmt_percent(0.124) == "12.4%"
    assert fmt_percent(22.4) == "22.4%"
    assert fmt_multiple(15.234) == "15.2x"
    assert fmt_ratio(1.12345) == "1.12"
    assert fmt_score(74.2) == "74/100"
    assert fmt_volume(12_013_668) == "12,013,668"
    assert fmt_volume(40_250_000) == "40.25M"
    assert fmt_volume(7_430_000_000) == "7.43B"
    assert fmt_shares(205_000_000) == "205.00M"
    assert fmt_number(1234.567) == "1,235"


def test_formatters_show_unavailable_for_bad_values():
    for value in [None, math.nan, math.inf, "not a number"]:
        assert fmt_dollar(value) == "Unavailable"
        assert fmt_percent(value) == "Unavailable"
        assert fmt_per_share(value) == "Unavailable"
        assert fmt_ratio(value) == "Unavailable"


def test_market_summary_context_formatting():
    assert format_market_summary_value("Beta", 1.12345) == "1.12"
    assert format_market_summary_value("Rolling Beta", 0.874) == "0.87"
    assert format_market_summary_value("Current Ratio", 1.284) == "1.28"
    assert format_market_summary_value("Quick Ratio", 1.274) == "1.27"
    assert format_market_summary_value("Debt / Equity", 0.304) == "0.30"
    assert format_market_summary_value("Short Ratio", 2.364) == "2.36"
    assert format_market_summary_value("ATR", 12.154) == "12.15"
    assert format_market_summary_value("Relative Volume", 0.524) == "0.52"
    assert format_market_summary_value("Correlation", 0.738492) == "0.74"
    assert format_market_summary_value("PEG", 1.054) == "1.05"
    assert format_market_summary_value("P/B", 6.854) == "6.85"
    assert format_market_summary_value("P/S", 8.914) == "8.91"
    assert format_market_summary_value("P/C", 36.234) == "36.23"
    assert format_market_summary_value("P/FCF", 38.904) == "38.90"
    assert format_market_summary_value("P/E", 22.74) == "22.7x"
    assert format_market_summary_value("Forward P/E", 19.64) == "19.6x"
    assert format_market_summary_value("ROA", 0.199) == "19.9%"
    assert format_market_summary_value("Upside", -0.512) == "-51.2%"
    assert format_market_summary_value("Market Cap", 2_836_180_000_000) == "$2,836.18B"
    assert format_market_summary_value("Price", 381.8) == "$381.80"
    assert format_market_summary_value("Shares Outstanding", 7_430_000_000) == "7.43B"
