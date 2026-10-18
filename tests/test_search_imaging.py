from __future__ import annotations

from typing import Any, cast

import pytest

from planner import (
    AcquisitionEnvelope,
    AxisThreshold,
    FocusedSpec,
    FocusedTarget,
    MeasurementLimits,
    OperatingCorner,
    Phase,
    SearchImagingRecipe,
    SearchImagingSpec,
    StageProfile,
    SurveySpec,
    build_focused_recipe,
    build_survey_recipe,
    estimate_search_imaging,
    focused_revisit_plan,
    search_imaging_records,
    stitching_preview,
    survey_overscan_um,
    survey_positions,
    survey_scan_plan,
    survey_tiles,
)
from planner.simulation import MockTarget, simulate_search_imaging


def _survey(**overrides: Any) -> SurveySpec:
    values: dict[str, Any] = {
        "tile_size_um": (200.0, 100.0),
        "overlap": 0.1,
        "exposure_s": 0.01,
        "frames": 1,
        "readout_s": 0.005,
        "overhead_s": 0.02,
        "pixel_size_um": 0.1,
        "resolution_um": 0.25,
    }
    values.update(overrides)
    return SurveySpec(**values)


def _focused(**overrides: Any) -> FocusedSpec:
    values: dict[str, Any] = {
        "exposure_s": 0.2,
        "frames": 1,
        "channels": 1,
        "z_planes": 1,
        "readout_s": 0.02,
        "overhead_s": 0.1,
        "pixel_size_um": 0.1,
        "resolution_um": 0.25,
        "roi_size_um": (64.0, 64.0),
    }
    values.update(overrides)
    return FocusedSpec(**values)


def _stage() -> StageProfile:
    return StageProfile(1000.0, 10000.0, 0.01)


def _recipe(**overrides: Any) -> SearchImagingRecipe:
    values: dict[str, Any] = {
        "recipe_id": "recipe-1",
        "field_size_um": (500.0, 250.0),
        "survey": _survey(),
        "focused": _focused(),
        "stage": _stage(),
        "review_s": 30.0,
    }
    values.update(overrides)
    return SearchImagingRecipe(**values)


def _envelope() -> AcquisitionEnvelope:
    return AcquisitionEnvelope(
        velocity_um_s=AxisThreshold(1.0, 100.0),
        exposure_s=AxisThreshold(0.05, 0.1),
        width_um=AxisThreshold(50.0, 150.0),
    )


def _targets() -> tuple[FocusedTarget, ...]:
    return (
        FocusedTarget("target-000", (50.0, 50.0)),
        FocusedTarget("target-001", (150.0, 50.0), roi_size_um=(32.0, 32.0)),
    )


def test_survey_tiles_cover_the_field_in_serpentine_order() -> None:
    recipe = _recipe(
        field_size_um=(500.0, 250.0),
        survey=_survey(tile_size_um=(200.0, 100.0), overlap=0.5),
    )
    tiles = survey_tiles(recipe)

    assert len(tiles) == 16
    assert {tile.row for tile in tiles} == {0, 1, 2, 3}
    assert {tile.column for tile in tiles} == {0, 1, 2, 3}
    first_row = [tile.column for tile in tiles if tile.row == 0]
    second_row = [tile.column for tile in tiles if tile.row == 1]
    assert first_row == [0, 1, 2, 3]
    assert second_row == [3, 2, 1, 0]
    for axis, extent, tile_extent in ((0, 500.0, 200.0), (1, 250.0, 100.0)):
        centers = [tile.center_xy_um[axis] for tile in tiles]
        assert min(center - tile_extent / 2 for center in centers) <= 0.0
        assert max(center + tile_extent / 2 for center in centers) >= extent


def test_survey_positions_mirror_the_tile_grid() -> None:
    recipe = _recipe(
        field_size_um=(500.0, 250.0),
        survey=_survey(tile_size_um=(200.0, 100.0), overlap=0.5),
    )
    tiles = survey_tiles(recipe)
    positions = survey_positions(recipe)

    assert len(positions) == len(tiles)
    assert [position.position_id for position in positions] == [
        tile.tile_id for tile in tiles
    ]
    assert [(position.row, position.column) for position in positions] == [
        (tile.row, tile.column) for tile in tiles
    ]
    assert [position.x_um for position in positions] == [
        tile.center_xy_um[0] for tile in tiles
    ]
    assert [position.y_um for position in positions] == [
        tile.center_xy_um[1] for tile in tiles
    ]


def test_survey_step_and_overscan_follow_the_grid() -> None:
    recipe = _recipe(
        field_size_um=(500.0, 250.0),
        survey=_survey(tile_size_um=(200.0, 100.0), overlap=0.1),
    )
    preview = stitching_preview(recipe)

    assert preview.step_xy_um == pytest.approx((180.0, 90.0))
    assert preview.overscan_um == pytest.approx((60.0, 30.0))
    assert survey_overscan_um(recipe) == pytest.approx((60.0, 30.0))
    assert recipe.overscan_um() == pytest.approx((60.0, 30.0))
    assert (preview.tile_rows, preview.tile_columns) == (3, 3)


def test_a_tile_larger_than_the_field_is_one_centered_tile() -> None:
    recipe = _recipe(
        field_size_um=(100.0, 80.0),
        survey=_survey(tile_size_um=(200.0, 160.0)),
    )

    assert survey_tiles(recipe) == recipe.survey_tiles()
    assert recipe.survey_positions()[0].x_um == pytest.approx(50.0)
    assert recipe.survey_positions()[0].y_um == pytest.approx(40.0)
    assert recipe.overscan_um() == pytest.approx((50.0, 40.0))


def test_stitching_metadata_includes_the_expected_mosaic_shape() -> None:
    recipe = _recipe(
        field_size_um=(500.0, 250.0),
        survey=_survey(tile_size_um=(200.0, 100.0), overlap=0.1),
        tile_shape_yx=(512, 640),
    )
    preview = stitching_preview(recipe)

    assert preview.tile_shape_yx == (512, 640)
    # Three rows at 0.9 * 512 px and three columns at 0.9 * 640 px.
    assert preview.mosaic_shape_yx == (1434, 1792)
    assert preview.to_dict()["mosaic_shape_yx"] == [1434, 1792]


def test_stitching_metadata_omits_the_mosaic_without_an_image_shape() -> None:
    recipe = _recipe()
    preview = stitching_preview(recipe)

    assert preview.tile_shape_yx is None
    assert preview.mosaic_shape_yx is None
    assert preview.to_dict()["mosaic_shape_yx"] is None


def test_survey_recipe_phase_settings_and_metadata() -> None:
    recipe = _recipe(
        field_size_um=(500.0, 250.0),
        survey=_survey(tile_size_um=(200.0, 100.0), overlap=0.1, frames=2),
        channel_ids=("GFP", "RFP"),
        focused=_focused(channels=2),
        tile_shape_yx=(512, 640),
    )
    survey = build_survey_recipe(recipe, envelope=_envelope())

    assert survey.phase is Phase.SURVEY
    assert len(survey.positions) == len(survey_tiles(recipe))
    assert survey.settings.exposure_ms == pytest.approx(10.0)
    assert survey.settings.frames == 2
    assert survey.settings.channel_id == "GFP"
    assert survey.settings.image_shape_yx == (512, 640)
    metadata = survey.settings.metadata
    assert metadata["planner_search_round"] == "survey"
    assert metadata["planner_survey_overlap"] == pytest.approx(0.1)
    assert metadata["planner_survey_step_xy_um"] == pytest.approx([180.0, 90.0])
    assert metadata["planner_survey_grid_rows"] == 3
    assert metadata["planner_survey_grid_columns"] == 3
    assert metadata["planner_survey_overscan_um"] == pytest.approx([60.0, 30.0])
    assert metadata["planner_acquisition_corner"] == OperatingCorner.C001.value


def test_focused_recipe_phase_settings_and_metadata() -> None:
    recipe = _recipe(
        focused=_focused(frames=2, channels=2, z_planes=3),
        channel_ids=("GFP", "RFP"),
    )
    focused = build_focused_recipe(recipe, _targets(), envelope=_envelope())

    assert focused.phase is Phase.REVISIT
    assert [position.position_id for position in focused.positions] == [
        "target-000",
        "target-001",
    ]
    assert [position.x_um for position in focused.positions] == [50.0, 150.0]
    assert focused.settings.frames == 2 * 2 * 3
    assert focused.settings.exposure_ms == pytest.approx(200.0)
    assert focused.settings.channel_id is None
    metadata = focused.settings.metadata
    assert metadata["planner_search_round"] == "focused"
    assert metadata["planner_focused_target_ids"] == ["target-000", "target-001"]
    assert metadata["planner_focused_frames"] == 2
    assert metadata["planner_focused_channels"] == 2
    assert metadata["planner_focused_z_planes"] == 3
    assert metadata["planner_focused_frames_per_target"] == 12
    assert metadata["planner_focused_channel_ids"] == ["GFP", "RFP"]
    assert metadata["planner_focused_target_roi_size_um"] == [None, [32.0, 32.0]]


def test_single_channel_focused_recipe_selects_the_declared_channel() -> None:
    recipe = _recipe(channel_ids=("GFP",), focused=_focused(channels=1))
    focused = build_focused_recipe(recipe, _targets())

    assert focused.settings.channel_id == "GFP"


def test_search_imaging_records_returns_both_rounds_and_stitching() -> None:
    recipe = _recipe(tile_shape_yx=(512, 640))
    records = search_imaging_records(recipe, _targets(), envelope=_envelope())

    assert records.recipe is recipe
    assert records.survey_recipe.phase is Phase.SURVEY
    assert records.focused_recipe.phase is Phase.REVISIT
    assert len(records.survey_recipe.positions) == len(survey_tiles(recipe))
    assert len(records.focused_recipe.positions) == len(_targets())
    assert records.stitching.mosaic_shape_yx == (1434, 1792)


def test_estimate_agrees_with_the_simulator_and_reports_model_corners() -> None:
    recipe = _recipe(
        field_size_um=(400.0, 200.0),
        survey=_survey(tile_size_um=(200.0, 100.0)),
        focused=_focused(roi_size_um=(200.0, 200.0)),
        review_s=2.0,
    )
    targets = (
        FocusedTarget("target-000", (50.0, 50.0)),
        FocusedTarget("target-001", (150.0, 50.0)),
    )
    envelope = _envelope()
    limits = MeasurementLimits(max_blur_ratio=2.0, max_gamma=1.0)
    expected_spec = SearchImagingSpec(
        field_size_um=recipe.field_size_um,
        survey=recipe.survey,
        focused=recipe.focused,
        stage=recipe.stage,
        targets=(
            MockTarget("target-000", (50.0, 50.0)),
            MockTarget("target-001", (150.0, 50.0)),
        ),
        review_s=recipe.review_s,
    )
    expected = simulate_search_imaging(expected_spec, envelope=envelope, limits=limits)
    actual = estimate_search_imaging(recipe, targets, envelope=envelope, limits=limits)

    assert actual.to_dict() == expected.to_dict()
    assert actual.survey_acquisition_corner is OperatingCorner.C001
    assert actual.survey_equivalent_corner is OperatingCorner.C101
    assert actual.focused_corner is OperatingCorner.C011
    assert actual.survey_acquisition_metrics.area_throughput_um2_s == pytest.approx(0.0)
    assert actual.survey_equivalent_metrics.area_throughput_um2_s == pytest.approx(
        actual.survey_throughput_um2_s
    )
    assert actual.survey_acquisition_verdict is not None
    assert actual.survey_acquisition_verdict.feasible
    assert actual.focused_verdict is not None
    assert actual.focused_verdict.feasible


def test_estimate_rejects_a_target_outside_the_field() -> None:
    recipe = _recipe(field_size_um=(100.0, 100.0))
    with pytest.raises(ValueError, match="outside the field"):
        estimate_search_imaging(recipe, (FocusedTarget("target-000", (150.0, 50.0)),))


def test_recipe_validates_geometry_channels_and_tile_shape() -> None:
    with pytest.raises(ValueError, match="field_size_um"):
        _recipe(field_size_um=(0.0, 100.0))
    with pytest.raises(ValueError, match="review_s"):
        _recipe(review_s=-1.0)
    with pytest.raises(ValueError, match="exactly focused.channels"):
        _recipe(channel_ids=("GFP",), focused=_focused(channels=2))
    with pytest.raises(ValueError, match="repeat"):
        _recipe(channel_ids=("GFP", "GFP"), focused=_focused(channels=2))
    with pytest.raises(TypeError, match="tuple"):
        _recipe(channel_ids=["GFP"])
    with pytest.raises(ValueError, match="tile_shape_yx"):
        _recipe(tile_shape_yx=(0, 640))
    with pytest.raises(TypeError, match="SurveySpec"):
        _recipe(survey=("not", "a", "survey"))


def test_focused_targets_are_validated() -> None:
    with pytest.raises(ValueError, match="target_id"):
        FocusedTarget("", (1.0, 2.0))
    with pytest.raises(ValueError, match="roi_size_um"):
        FocusedTarget("target-000", (1.0, 2.0), roi_size_um=(0.0, 1.0))
    recipe = _recipe()
    with pytest.raises(ValueError, match="duplicate target id"):
        build_focused_recipe(
            recipe,
            (
                FocusedTarget("target-000", (10.0, 10.0)),
                FocusedTarget("target-000", (20.0, 20.0)),
            ),
        )
    with pytest.raises(TypeError, match="FocusedTarget"):
        estimate_search_imaging(
            recipe, cast("Any", (MockTarget("target-000", (1.0, 1.0)),))
        )


def test_estimate_validates_the_optional_model_inputs() -> None:
    recipe = _recipe()
    with pytest.raises(TypeError, match="envelope"):
        estimate_search_imaging(recipe, (), envelope=cast("Any", "not-an-envelope"))
    with pytest.raises(TypeError, match="limits"):
        estimate_search_imaging(recipe, (), limits=cast("Any", "not-limits"))


def test_survey_plan_emits_one_run_and_one_capture_per_tile() -> None:
    bluesky = pytest.importorskip("bluesky")
    ophyd_sim = pytest.importorskip("ophyd.sim")
    np = pytest.importorskip("numpy")
    x = ophyd_sim.SynAxis(name="survey_x", value=0, egu="um")
    y = ophyd_sim.SynAxis(name="survey_y", value=0, egu="um")
    camera = ophyd_sim.SynSignal(
        func=lambda: np.ones((4, 4), dtype=np.uint16), name="survey_camera"
    )
    recipe = _recipe(
        field_size_um=(400.0, 200.0),
        survey=_survey(tile_size_um=(200.0, 100.0)),
    )
    documents: list[tuple[str, dict[str, Any]]] = []
    engine = bluesky.RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))

    engine(
        survey_scan_plan(
            recipe,
            stage_x=x,
            stage_y=y,
            camera=camera,
            readables=(x, y),
        )
    )

    starts = [doc for name, doc in documents if name == "start"]
    events = [doc for name, doc in documents if name == "event"]
    descriptors = [doc for name, doc in documents if name == "descriptor"]
    stop = next(doc for name, doc in documents if name == "stop")
    assert len(starts) == 1
    assert starts[0]["planner_search_round"] == "survey"
    assert starts[0]["planner_search_round_index"] == 1
    assert starts[0]["planner_search_workflow"] == "two-round-search-and-image-v1"
    assert starts[0]["planner_scan_mode"] == "snap"
    assert starts[0]["planner_phase"] == "survey"
    assert starts[0]["planner_survey_overlap"] == pytest.approx(0.1)
    assert len(events) == len(recipe.survey_tiles())
    assert len(descriptors) == 1
    assert descriptors[0]["name"] == "survey"
    assert stop["exit_status"] == "success"


def test_focused_plan_emits_one_run_and_the_declared_frame_count() -> None:
    bluesky = pytest.importorskip("bluesky")
    ophyd_sim = pytest.importorskip("ophyd.sim")
    np = pytest.importorskip("numpy")
    x = ophyd_sim.SynAxis(name="focused_x", value=0, egu="um")
    y = ophyd_sim.SynAxis(name="focused_y", value=0, egu="um")
    camera = ophyd_sim.SynSignal(
        func=lambda: np.ones((4, 4), dtype=np.uint16), name="focused_camera"
    )
    recipe = _recipe(
        field_size_um=(400.0, 200.0),
        focused=_focused(frames=1, channels=2, z_planes=1),
        channel_ids=("GFP", "RFP"),
    )
    targets = (
        FocusedTarget("target-000", (50.0, 50.0)),
        FocusedTarget("target-001", (150.0, 50.0)),
    )
    documents: list[tuple[str, dict[str, Any]]] = []
    engine = bluesky.RunEngine({})
    engine.subscribe(lambda name, doc: documents.append((name, doc)))

    engine(
        focused_revisit_plan(
            recipe,
            targets,
            stage_x=x,
            stage_y=y,
            camera=camera,
            readables=(x, y),
        )
    )

    starts = [doc for name, doc in documents if name == "start"]
    events = [doc for name, doc in documents if name == "event"]
    descriptors = [doc for name, doc in documents if name == "descriptor"]
    assert len(starts) == 1
    assert starts[0]["planner_search_round"] == "focused"
    assert starts[0]["planner_search_round_index"] == 2
    assert starts[0]["planner_search_target_ids"] == [
        "target-000",
        "target-001",
    ]
    assert starts[0]["planner_focused_channels"] == 2
    assert starts[0]["planner_focused_z_planes"] == 1
    assert starts[0]["planner_phase"] == "revisit"
    assert len(events) == len(targets) * 2
    assert [event["data"]["focused_x"] for event in events] == [
        50.0,
        50.0,
        150.0,
        150.0,
    ]
    assert len(descriptors) == 1
    assert descriptors[0]["name"] == "focused"
