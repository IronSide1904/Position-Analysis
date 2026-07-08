import pandas as pd

from analysis.clauses import extract_relevant_clauses
from analysis.compensation import analyze_compensation_alignment
from analysis.ma_strategy import analyze_ma_strategy
from analysis.moat import analyze_moat


def test_clause_extraction_empty_when_no_text():
    df = extract_relevant_clauses({})
    assert df.empty
    assert "clause_text" in df.columns


def test_compensation_unavailable_clean_message():
    historicals = pd.DataFrame([{"Revenue": 1000.0, "SBC": 0.0, "Diluted Shares": 100.0}])
    result = analyze_compensation_alignment({}, historicals)
    assert result["alignment_score"] >= 1
    assert result["red_flags"]


def test_ma_unavailable_clean_message():
    result = analyze_ma_strategy({}, pd.DataFrame())
    assert result["classification"] == "Insufficient data"


def test_moat_unavailable_clean_message():
    result = analyze_moat({"ticker": "ABC"}, pd.DataFrame(), {})
    assert result["classification"] == "Unknown / insufficient data"


def test_weak_moat_warning_signal():
    result = analyze_moat({"ticker": "ABC"}, pd.DataFrame(), {})
    assert any("terminal" in flag.lower() for flag in result["red_flags"])

