from analysis.risks import analyze_risks_and_thesis_breakers


def test_risk_extractor_filters_xbrl_fragments():
    filing_texts = {
        "10-Q": (
            "INNODATA INC._March 31, 2026 0000903651 --12-31 2026 Q1 false "
            "http://fasb.org/us-gaap/2025#InterestIncomeExpenseNonoperatingNet "
            "P3Y http://fasb.org/us-gaap/2025#SecuredOvernightFinancingRateSofrMember. "
            "Our business faces competitive risks because customer demand can change quickly "
            "and competitors may reduce pricing or win major customers."
        )
    }

    result = analyze_risks_and_thesis_breakers(filing_texts, None, None)

    assert result["top_risks"]
    assert "fasb.org" not in result["top_risks"][0]
    assert "competitive risks" in result["top_risks"][0]
    assert result["risk_rows"][0]["explanation"]
