from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pytest

from planner.report_html import (
    render_comparison_html,
    render_simulation_html,
    write_comparison_html,
    write_simulation_html,
)
from planner.simulation import (
    FocusedSpec,
    MockTarget,
    SearchImagingSpec,
    SimulationReport,
    StageProfile,
    SurveySpec,
    simulate_search_imaging,
)

if TYPE_CHECKING:
    from pathlib import Path


def _report() -> tuple[SearchImagingSpec, SimulationReport]:
    spec = SearchImagingSpec(
        field_size_um=(400.0, 200.0),
        survey=SurveySpec((200.0, 100.0), 0.1, 0.02, 1, 0.06, 0.05, 0.1, 0.25),
        focused=FocusedSpec(2.0, 1, 1, 1, 0.06, 0.1, 0.1, 0.25, (64.0, 64.0)),
        stage=StageProfile(1000.0, 10000.0, 0.01),
        targets=(
            MockTarget("a", (100.0, 50.0)),
            MockTarget("b", (150.0, 50.0)),
        ),
        bleach_time_constant_s=10.0,
    )
    return spec, simulate_search_imaging(spec)


def _comparison_payload() -> dict[str, Any]:
    return {
        "field_um": [2000.0, 2000.0],
        "fov_um": 133.12,
        "dwell_s": 2.0,
        "rows": 16,
        "geometry": {
            "sample_count": 2,
            "field_um": [2000.0, 2000.0],
            "survey_tile_size_um": [133.12, 133.12],
            "survey_tiles": [[100.0, 100.0], [300.0, 100.0]],
            "samples": [[100.0, 100.0], [300.0, 100.0]],
            "focused_roi_um": [133.12, 133.12],
            "broom_rows_um": [62.5, 187.5],
            "broom_fov_um": 133.12,
            "slow_zones_um": [[33.44, 166.56, 62.5], [233.44, 366.56, 62.5]],
        },
        "operating": {
            "has_envelope": False,
            "thresholds": None,
            "ranges": {
                "velocity": [0.0, 100.0],
                "exposure": [0.0, 2.0],
                "width": [0.0, 133.12],
            },
            "traces": [
                {
                    "name": "adaptive broom",
                    "color": "#d99a2b",
                    "dashed": False,
                    "markers": True,
                    "points": [
                        {
                            "label": "dwell at sample",
                            "v": 66.56,
                            "te": 0.02,
                            "w": 133.12,
                            "corner": None,
                        },
                        {
                            "label": "cruise between samples",
                            "v": 103.7,
                            "te": 0.02,
                            "w": 133.12,
                            "corner": None,
                        },
                    ],
                }
            ],
        },
        "rows_data": [
            {
                "scenario": "measured",
                "samples": 10,
                "broom_cruise_s": 384.0,
                "broom_dwell_s": 561.3,
                "adaptive_broom_s": 392.6,
                "added_vs_cruise_s": 8.6,
                "transition_s": 19.6,
                "dual_snap_s": 786.7,
                "speedup_vs_broom": 1.43,
                "speedup": 2.0,
                "bleach_cost_mean": 0.01,
            },
            {
                "scenario": "measured",
                "samples": 50,
                "broom_cruise_s": 384.0,
                "broom_dwell_s": 561.3,
                "adaptive_broom_s": 422.2,
                "added_vs_cruise_s": 38.2,
                "transition_s": 38.4,
                "dual_snap_s": 974.3,
                "speedup_vs_broom": 1.33,
                "speedup": 2.31,
                "bleach_cost_mean": 0.012,
            },
        ],
    }


def test_simulation_html_is_self_contained_and_has_charts() -> None:
    spec, report = _report()
    page = render_simulation_html(report, spec=spec)

    assert page.startswith("<!doctype html>")
    assert "<svg" in page
    assert "Sub-scan metrics" in page
    assert "Photobleaching from the overview" in page
    assert "Wall time by component" in page
    assert "Survey and focused layout" in page
    assert "Operating-space trajectories" in page
    assert "survey tile" in page
    assert "200.0 x 100.0" in page
    assert "Throughput (mm2/min)" in page
    assert "http://" not in page
    assert "https://" not in page
    assert "<script" not in page


def test_simulation_html_escapes_the_title() -> None:
    _spec, report = _report()

    page = render_simulation_html(report, title="<script>alert(1)</script>")

    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_comparison_html_has_grouped_charts_and_table() -> None:
    page = render_comparison_html(_comparison_payload())

    assert page.startswith("<!doctype html>")
    assert page.count("<svg") >= 1
    assert "broom-fast" in page
    assert "broom-2s" in page
    assert "adaptive broom" in page
    assert "dual snap scan" in page
    assert "bleach cost" in page
    assert "Plan geometry" in page
    assert "Dual snap scan coverage" in page
    assert "Adaptive broom raster" in page
    assert "Survey tile" in page
    assert "133.1 x 133.1" in page
    assert "Broom strip" in page
    assert "Focused ROI" in page
    assert "Operating-space trajectories" in page
    assert "http://" not in page
    assert "<script" not in page


def test_comparison_html_rejects_malformed_rows() -> None:
    with pytest.raises(TypeError, match="rows_data"):
        render_comparison_html({"rows_data": "not a list"})
    with pytest.raises(TypeError, match="mapping"):
        render_comparison_html({"rows_data": ["not a mapping"]})


def test_writers_create_files(tmp_path: Path) -> None:
    spec, report = _report()
    simulation_path = tmp_path / "simulation.html"
    comparison_path = tmp_path / "comparison.html"

    assert write_simulation_html(simulation_path, report, spec=spec) == simulation_path
    assert (
        write_comparison_html(comparison_path, _comparison_payload()) == comparison_path
    )

    assert simulation_path.read_text(encoding="utf-8").startswith("<!doctype html>")
    assert comparison_path.read_text(encoding="utf-8").startswith("<!doctype html>")
