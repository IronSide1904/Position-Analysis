from streamlit.testing.v1 import AppTest


def test_pa11r_cockpit_tabs_render():
    app = AppTest.from_file("app.py", default_timeout=120)
    app.run(timeout=180)

    assert len(app.exception) == 0
    expected = [
        "DCF Model",
        "Snapshot",
        "SOTP",
        "Financials & Reinvestment",
        "Evidence & Assumptions",
        "Business Quality & Risks",
        "Sources & Review",
    ]
    labels = [tab.label for tab in app.tabs]
    assert all(label in labels for label in expected)
    assert [labels.index(label) for label in expected] == sorted(labels.index(label) for label in expected)


def test_mr1_lite_removed_from_dashboard():
    app = AppTest.from_file("app.py", default_timeout=120)
    app.run(timeout=180)

    assert len(app.exception) == 0
    assert len(app.radio) == 0
    assert "MR-1 Lite" not in app.main.text
