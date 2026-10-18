from __future__ import annotations

import json
from math import hypot
from typing import Any, cast

import pytest

from planner.acquisition_space import (
    AcquisitionEnvelope,
    AxisThreshold,
    MeasurementLimits,
    OperatingCorner,
)
from planner.simulation import (
    FocusedSpec,
    MockTarget,
    RoundReport,
    SearchImagingSpec,
    StageProfile,
    SurveySpec,
    mock_targets,
    simulate_search_imaging,
)


def _survey(
    tile: tuple[float, float] = (200.0, 100.0),
    *,
    overlap: float = 0.1,
    frames: int = 2,
) -> SurveySpec:
    return SurveySpec(
        tile_size_um=tile,
        overlap=overlap,
        exposure_s=0.01,
        frames=frames,
        readout_s=0.005,
        overhead_s=0.02,
        pixel_size_um=0.1,
        resolution_um=0.25,
    )


def _focused(roi: tuple[float, float] = (64.0, 64.0)) -> FocusedSpec:
    return FocusedSpec(
        exposure_s=0.2,
        frames=1,
        channels=2,
        z_planes=1,
        readout_s=0.02,
        overhead_s=0.1,
        pixel_size_um=0.1,
        resolution_um=0.25,
        roi_size_um=roi,
    )


def _stage(
    velocity: float = 1000.0, acceleration: float = 10000.0, settle: float = 0.01
) -> StageProfile:
    return StageProfile(velocity, acceleration, settle)


def test_stage_profile_uses_triangular_and_trapezoidal_moves() -> None:
    stage = _stage()
    assert stage.travel_time_s(0.0) == 0.0
    # Below v^2/a the profile is triangular; above it cruises.
    assert stage.travel_time_s(50.0) == pytest.approx(2 * (50.0 / 10000.0) ** 0.5)
    assert stage.travel_time_s(200.0) == pytest.approx(200.0 / 1000.0 + 0.1)
    assert stage.travel_time_s(-50.0) == pytest.approx(stage.travel_time_s(50.0))


def test_positioning_adds_settle_only_to_a_nonzero_move() -> None:
    stage = _stage(settle=0.04)
    assert stage.positioning_time_s(0.0) == 0.0
    assert stage.positioning_time_s(50.0) == pytest.approx(
        stage.travel_time_s(50.0) + 0.04
    )


def test_stage_profile_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="positive"):
        StageProfile(0.0, 1000.0)
    with pytest.raises(ValueError, match="nonnegative"):
        StageProfile(1000.0, 1000.0, settle_s=-0.1)
    with pytest.raises(TypeError, match="real number"):
        StageProfile(True, 1000.0)


def test_mock_targets_are_deterministic_and_inside_the_field() -> None:
    first = mock_targets((500.0, 400.0), 20, seed=7, margin_um=10.0)
    second = mock_targets((500.0, 400.0), 20, seed=7, margin_um=10.0)

    assert first == second
    assert [target.target_id for target in first][:2] == ["target-000", "target-001"]
    for target in first:
        x, y = target.center_xy_um
        assert 10.0 <= x <= 490.0
        assert 10.0 <= y <= 390.0
    assert mock_targets((10.0, 10.0), 0) == ()


def test_mock_targets_reject_a_margin_that_leaves_no_room() -> None:
    with pytest.raises(ValueError, match="no room"):
        mock_targets((10.0, 10.0), 1, margin_um=5.0)


def test_survey_tiles_cover_the_field_in_serpentine_order() -> None:
    spec = SearchImagingSpec(
        field_size_um=(500.0, 250.0),
        survey=_survey((200.0, 100.0), overlap=0.5),
        focused=_focused(),
        stage=_stage(),
    )
    tiles = spec.survey_tiles()

    columns = {tile.column for tile in tiles}
    rows = {tile.row for tile in tiles}
    assert columns == {0, 1, 2, 3}
    assert rows == {0, 1, 2, 3}
    # Odd rows are traversed right to left.
    first_row = [tile.column for tile in tiles if tile.row == 0]
    second_row = [tile.column for tile in tiles if tile.row == 1]
    assert first_row == [0, 1, 2, 3]
    assert second_row == [3, 2, 1, 0]
    # Every point along both axes is inside at least one tile.
    for coordinate, tile_extent, axis in ((500.0, 200.0, 0), (250.0, 100.0, 1)):
        centers = [t.center_xy_um[axis] for t in tiles]
        covered_from = min(center - tile_extent / 2 for center in centers)
        covered_to = max(center + tile_extent / 2 for center in centers)
        assert covered_from <= 0.0
        assert covered_to >= coordinate


def test_a_tile_larger_than_the_field_is_one_centered_tile() -> None:
    spec = SearchImagingSpec(
        field_size_um=(100.0, 80.0),
        survey=_survey((200.0, 160.0)),
        focused=_focused(),
        stage=_stage(),
    )

    assert spec.survey_centers() == ((50.0, 40.0),)
    assert spec.overscan_um() == (50.0, 40.0)


def test_survey_round_estimates_each_component() -> None:
    spec = SearchImagingSpec(
        field_size_um=(200.0, 100.0),
        survey=_survey(frames=2),
        focused=_focused(),
        stage=_stage(settle=0.02),
    )
    report = simulate_search_imaging(spec)
    distance = hypot(100.0, 50.0)

    assert report.survey.positions == 1
    assert report.survey.distance_um == pytest.approx(distance)
    assert report.survey.travel_s == pytest.approx(spec.stage.travel_time_s(distance))
    assert report.survey.settle_s == pytest.approx(0.02)
    assert report.survey.exposure_s == pytest.approx(0.01 * 2)
    assert report.survey.readout_s == pytest.approx(0.005 * 2)
    assert report.survey.overhead_s == pytest.approx(0.02)


def test_focused_round_visits_nearest_neighbours_and_counts_all_frames() -> None:
    targets = (
        MockTarget("a", (10.0, 10.0)),
        MockTarget("b", (20.0, 10.0)),
    )
    spec = SearchImagingSpec(
        field_size_um=(200.0, 100.0),
        survey=_survey(),
        focused=_focused(),
        stage=_stage(),
        targets=targets,
    )
    report = simulate_search_imaging(spec)

    assert report.focused_order_ids == ("b", "a")
    assert report.focused.positions == 2
    frames = 1 * 2 * 1
    assert report.focused.exposure_s == pytest.approx(0.2 * frames * 2)
    assert report.focused.readout_s == pytest.approx(0.02 * frames * 2)
    expected_travel = spec.stage.travel_time_s(
        hypot(80.0, 40.0)
    ) + spec.stage.travel_time_s(10.0)
    assert report.focused.travel_s == pytest.approx(expected_travel)
    assert report.focused.settle_s == pytest.approx(0.02)


def test_photobleaching_counts_overview_coverage_and_exposure_cost() -> None:
    spec = SearchImagingSpec(
        field_size_um=(400.0, 200.0),
        survey=_survey((200.0, 100.0)),
        focused=FocusedSpec(
            2.0,
            1,
            1,
            1,
            0.02,
            0.1,
            0.1,
            0.25,
            (50.0, 50.0),
        ),
        stage=_stage(),
        targets=(
            MockTarget("a", (100.0, 50.0)),
            MockTarget("b", (190.0, 50.0)),
        ),
        bleach_time_constant_s=10.0,
    )
    report = simulate_search_imaging(spec)
    bleaching = report.photobleaching

    assert bleaching.samples == 2
    assert bleaching.preview_exposure_s == pytest.approx(0.02)
    assert bleaching.mean_cover_count == pytest.approx(1.5)
    assert bleaching.prior_exposure_s_min == pytest.approx(0.02)
    assert bleaching.prior_exposure_s_max == pytest.approx(0.04)
    assert bleaching.prior_exposure_s_mean == pytest.approx(0.03)
    assert bleaching.focused_exposure_s == pytest.approx(2.0)
    assert bleaching.cost_min == pytest.approx(0.01)
    assert bleaching.cost_max == pytest.approx(0.02)
    assert bleaching.cost_mean == pytest.approx(0.015)
    assert bleaching.bleach_time_constant_s == pytest.approx(10.0)
    assert bleaching.retained_min == pytest.approx(0.9960079893, abs=1e-9)
    assert bleaching.retained_max == pytest.approx(0.9980019987, abs=1e-9)
    assert bleaching.loss_fraction_min == pytest.approx(0.0019980013, abs=1e-9)
    assert bleaching.loss_fraction_max == pytest.approx(0.0039920107, abs=1e-9)

    payload = report.to_dict()
    assert payload["photobleaching"]["cost_mean"] == pytest.approx(0.015)
    assert payload["photobleaching"]["retained_mean"] is not None


def test_photobleaching_without_a_decay_constant_reports_cost_only() -> None:
    spec = SearchImagingSpec(
        field_size_um=(400.0, 200.0),
        survey=_survey((200.0, 100.0)),
        focused=_focused(),
        stage=_stage(),
        targets=(MockTarget("a", (100.0, 50.0)),),
    )
    bleaching = simulate_search_imaging(spec).photobleaching

    assert bleaching.cost_mean == pytest.approx(0.1)
    assert bleaching.bleach_time_constant_s is None
    assert bleaching.retained_mean is None
    assert bleaching.loss_fraction_mean is None


def test_photobleaching_with_no_targets_is_empty() -> None:
    spec = SearchImagingSpec(
        field_size_um=(400.0, 200.0),
        survey=_survey((200.0, 100.0)),
        focused=_focused(),
        stage=_stage(),
        bleach_time_constant_s=10.0,
    )
    bleaching = simulate_search_imaging(spec).photobleaching

    assert bleaching.samples == 0
    assert bleaching.mean_cover_count == 0.0
    assert bleaching.prior_exposure_s_mean == 0.0
    assert bleaching.cost_mean == 0.0


def test_sub_scan_metrics_and_averaged_metrics() -> None:
    spec = SearchImagingSpec(
        field_size_um=(400.0, 200.0),
        survey=_survey((200.0, 100.0)),
        focused=_focused(),
        stage=_stage(),
        targets=(
            MockTarget("a", (100.0, 50.0)),
            MockTarget("b", (150.0, 50.0)),
        ),
        review_s=2.0,
    )
    report = simulate_search_imaging(spec)

    survey = report.survey_scan
    focused = report.focused_scan
    assert survey.scan_id == "survey"
    assert survey.area_um2 == 80000.0
    assert survey.positions == report.survey.positions
    assert survey.throughput_um2_s == pytest.approx(survey.area_um2 / survey.duration_s)
    assert survey.seconds_per_position == pytest.approx(
        survey.duration_s / survey.positions
    )
    assert focused.scan_id == "focused"
    assert focused.area_um2 == 2 * 64.0 * 64.0
    assert focused.positions == 2
    assert focused.throughput_um2_s == pytest.approx(
        focused.area_um2 / focused.duration_s
    )
    assert focused.seconds_per_position == pytest.approx(
        focused.duration_s / focused.positions
    )
    assert report.averaged.mean_throughput_um2_s == pytest.approx(
        (survey.throughput_um2_s + focused.throughput_um2_s) / 2
    )
    assert report.averaged.mean_seconds_per_position == pytest.approx(
        (survey.seconds_per_position + focused.seconds_per_position) / 2
    )
    assert report.averaged.mean_travel_fraction == pytest.approx(
        (survey.travel_fraction + focused.travel_fraction) / 2
    )
    assert report.averaged.combined_acquired_area_um2 == pytest.approx(
        80000.0 + 2 * 64.0 * 64.0
    )
    assert report.averaged.combined_throughput_um2_s == pytest.approx(
        (80000.0 + 2 * 64.0 * 64.0) / (survey.duration_s + focused.duration_s)
    )
    assert report.averaged.total_duration_s == pytest.approx(report.total_s)

    payload = report.to_dict()
    assert payload["sub_scans"]["survey"]["scan_id"] == "survey"
    assert payload["sub_scans"]["focused"]["positions"] == 2
    assert payload["averaged"]["combined_acquired_area_um2"] == pytest.approx(
        80000.0 + 2 * 64.0 * 64.0
    )


def test_an_empty_focused_scan_does_not_halve_the_averages() -> None:
    spec = SearchImagingSpec(
        field_size_um=(400.0, 200.0),
        survey=_survey((200.0, 100.0)),
        focused=_focused(),
        stage=_stage(),
    )
    report = simulate_search_imaging(spec)

    assert report.focused_scan.positions == 0
    assert report.focused_scan.throughput_um2_s == 0.0
    assert report.focused_scan.seconds_per_position == 0.0
    assert report.averaged.mean_throughput_um2_s == pytest.approx(
        report.survey_scan.throughput_um2_s
    )
    assert report.averaged.mean_seconds_per_position == pytest.approx(
        report.survey_scan.seconds_per_position
    )
    assert report.averaged.combined_acquired_area_um2 == pytest.approx(
        report.field_area_um2
    )


def test_report_totals_throughput_and_target_rate() -> None:
    spec = SearchImagingSpec(
        field_size_um=(400.0, 200.0),
        survey=_survey((200.0, 100.0)),
        focused=_focused(),
        stage=_stage(),
        targets=(
            MockTarget("a", (50.0, 50.0)),
            MockTarget("b", (150.0, 50.0)),
        ),
        review_s=2.0,
    )
    report = simulate_search_imaging(spec)

    assert report.total_s == pytest.approx(
        report.survey.total_s + 2.0 + report.focused.total_s
    )
    assert report.field_area_um2 == 80000.0
    assert report.survey_throughput_um2_s == pytest.approx(
        80000.0 / report.survey.total_s
    )
    assert report.end_to_end_throughput_um2_s == pytest.approx(80000.0 / report.total_s)
    assert report.targets_per_hour == pytest.approx(2 * 3600.0 / report.total_s)


def test_equivalent_point_reproduces_survey_throughput() -> None:
    spec = SearchImagingSpec(
        field_size_um=(600.0, 300.0),
        survey=_survey((300.0, 150.0)),
        focused=_focused(),
        stage=_stage(),
    )
    report = simulate_search_imaging(spec)

    assert report.survey_acquisition.velocity_um_s == 0.0
    assert report.survey_equivalent.velocity_um_s > 0.0
    assert report.survey_equivalent_metrics.area_throughput_um2_s == pytest.approx(
        report.survey_throughput_um2_s
    )


def test_rounds_may_share_one_magnification() -> None:
    spec = SearchImagingSpec(
        field_size_um=(400.0, 200.0),
        survey=_survey((200.0, 100.0)),
        focused=_focused(),
        stage=_stage(),
        targets=(MockTarget("a", (100.0, 50.0)),),
    )
    report = simulate_search_imaging(spec)

    assert report.survey_acquisition.pixel_size_um == report.focused_point.pixel_size_um
    assert report.survey_acquisition.resolution_um == report.focused_point.resolution_um


def test_corners_distinguish_the_equivalent_survey_and_focused_rounds() -> None:
    envelope = AcquisitionEnvelope(
        velocity_um_s=AxisThreshold(1.0, 100.0),
        exposure_s=AxisThreshold(0.05, 0.1),
        width_um=AxisThreshold(50.0, 150.0),
    )
    spec = SearchImagingSpec(
        field_size_um=(400.0, 200.0),
        survey=_survey((200.0, 100.0)),
        focused=_focused((200.0, 200.0)),
        stage=_stage(),
        targets=(MockTarget("a", (100.0, 50.0)),),
    )
    report = simulate_search_imaging(spec, envelope=envelope)

    assert report.survey_acquisition_corner is OperatingCorner.C001
    assert report.survey_equivalent_corner is OperatingCorner.C101
    assert report.focused_corner is OperatingCorner.C011


def test_limits_produce_verdicts_for_every_point() -> None:
    limits = MeasurementLimits(max_blur_ratio=2.0, max_gamma=1.0)
    spec = SearchImagingSpec(
        field_size_um=(400.0, 200.0),
        survey=_survey((200.0, 100.0)),
        focused=_focused(),
        stage=_stage(),
        targets=(MockTarget("a", (100.0, 50.0)),),
    )
    report = simulate_search_imaging(spec, limits=limits)

    assert report.survey_acquisition_verdict is not None
    assert report.focused_verdict is not None
    assert report.survey_acquisition_verdict.feasible
    assert report.focused_verdict.feasible


def test_report_serializes_to_json_without_nan_or_infinity() -> None:
    spec = SearchImagingSpec(
        field_size_um=(400.0, 200.0),
        survey=_survey((200.0, 100.0)),
        focused=_focused(),
        stage=_stage(),
        targets=mock_targets((400.0, 200.0), 3, seed=2),
    )
    payload = simulate_search_imaging(spec).to_dict()

    encoded = json.dumps(payload, allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    assert payload["survey"]["positions"] == 9


def test_spec_rejects_targets_outside_the_field_or_duplicate_ids() -> None:
    with pytest.raises(ValueError, match="outside the field"):
        SearchImagingSpec(
            field_size_um=(100.0, 100.0),
            survey=_survey(),
            focused=_focused(),
            stage=_stage(),
            targets=(MockTarget("a", (150.0, 50.0)),),
        )
    with pytest.raises(ValueError, match="duplicate"):
        SearchImagingSpec(
            field_size_um=(100.0, 100.0),
            survey=_survey(),
            focused=_focused(),
            stage=_stage(),
            targets=(MockTarget("a", (50.0, 50.0)), MockTarget("a", (60.0, 50.0))),
        )


def test_spec_rejects_a_non_tuple_target_collection() -> None:
    with pytest.raises(TypeError, match="tuple"):
        SearchImagingSpec(
            field_size_um=(100.0, 100.0),
            survey=_survey(),
            focused=_focused(),
            stage=_stage(),
            targets=cast("Any", [MockTarget("a", (50.0, 50.0))]),
        )


def test_round_report_rejects_inconsistent_totals() -> None:
    with pytest.raises(ValueError, match="summed"):
        RoundReport(
            positions=1,
            distance_um=0.0,
            travel_s=0.0,
            settle_s=0.0,
            exposure_s=0.01,
            readout_s=0.0,
            overhead_s=0.0,
            total_s=0.02,
        )
