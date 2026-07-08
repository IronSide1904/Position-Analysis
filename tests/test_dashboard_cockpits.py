from streamlit.testing.v1 import AppTest


def test_pa11r_cockpit_tabs_render():
    app = AppTest.from_file("app.py", default_timeout=120)
    app.run(timeout=180)

    assert len(app.exception) == 0
    assert [tab.label for tab in app.tabs] == [
        "Snapshot",
        "Valuation",
        "Evidence & Assumptions",
        "Business Quality",
        "Management & Capital Allocation",
        "Sources & Data Quality",
    ]


def test_mr1_lite_cockpit_tabs_render():
    app = AppTest.from_file("app.py", default_timeout=120)
    app.run(timeout=180)
    app.radio[0].set_value("MR-1 Lite")
    app.run(timeout=180)

    assert len(app.exception) == 0
    assert [tab.label for tab in app.tabs] == [
        "Snapshot",
        "Trading Setup",
        "Regime & Relative Context",
        "Volume & Volatility",
        "Sources & Data Quality",
    ]
