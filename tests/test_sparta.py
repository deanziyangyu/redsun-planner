from __future__ import annotations

from planner.sparta import FRAMEWORK_NAME, FRAMEWORK_TITLE


def test_sparta_is_a_name_not_an_api_vocabulary() -> None:
    assert FRAMEWORK_NAME == "SPARTA"
    assert FRAMEWORK_TITLE == (
        "Spatiotemporal Pareto Acquisition-Regime Trajectory Allocation"
    )
