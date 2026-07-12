import inspect

from ui.dashboard_v2 import _clause_impact_filters


def test_clause_impact_filter_columns_match_filter_count():
    source = inspect.getsource(_clause_impact_filters)

    assert "[1] * len(filter_specs)" in source
    assert "cols[idx]" in source
