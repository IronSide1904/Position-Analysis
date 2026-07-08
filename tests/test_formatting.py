import math

from ui.formatting import fmt_dollar, fmt_multiple, fmt_number, fmt_percent, fmt_per_share, fmt_score, fmt_shares


def test_financial_display_formatters():
    assert fmt_dollar(1_250_000_000) == "$1,250M"
    assert fmt_per_share(15.423) == "$15"
    assert fmt_percent(0.124) == "12.4%"
    assert fmt_multiple(15.234) == "15x"
    assert fmt_score(74.2) == "74/100"
    assert fmt_shares(205_000_000) == "205M"
    assert fmt_number(1234.567) == "1,235"


def test_formatters_show_unavailable_for_bad_values():
    for value in [None, math.nan, math.inf, "not a number"]:
        assert fmt_dollar(value) == "Unavailable"
        assert fmt_percent(value) == "Unavailable"
        assert fmt_per_share(value) == "Unavailable"
