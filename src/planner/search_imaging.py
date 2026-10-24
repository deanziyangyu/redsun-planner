"""Two-round search-and-image recipes beside the broom plans.

The workflow is round-one overlapping survey, host-owned detection and review,
then round-two focused revisit. This module owns the inert recipes, the survey
grid and stitching context, the Bluesky plan generators and a deterministic
throughput estimate built on :mod:`planner.simulation`.

It performs no detection, no candidate selection, no image stitching and no
device I/O. A host detects and reviews candidates between the two plan runs,
then supplies the selected :class:`FocusedTarget` records. Round one need not be
lower resolution than round two: the two rounds carry independent pixel size and
resolution fields, so the survey may run at the focused magnification.
"""

from __future__ import annotations

# Host plan widgets resolve these plan annotations at runtime.
from collections.abc import Callable, Iterable, Mapping, Sequence  # noqa: TC003
from dataclasses import dataclass
from math import ceil, isfinite
from typing import Any

from bluesky.utils import MsgGenerator  # noqa: TC002

from .acquisition_space import (
    AcquisitionEnvelope,
    AcquisitionPoint,
    MeasurementLimits,
)
from .contracts import AcquisitionSettings, Phase
from .plans import snap_scan_plan
from .scanning import ScanPosition, StageScanRecipe
from .simulation import (
    FocusedSpec,
    MockTarget,
    SearchImagingSpec,
    SimulationReport,
    StageProfile,
    SurveySpec,
    SurveyTile,
    simulate_search_imaging,
)

DEFAULT_SURVEY_STREAM = "survey"
DEFAULT_FOCUSED_STREAM = "focused"
WORKFLOW_ID = "two-round-search-and-image-v1"


def _finite(value: float, name: str, *, positive: bool) -> float:
    """Return a finite number, rejecting booleans and out-of-range values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _pair(
    value: tuple[float, float], name: str, *, positive: bool
) -> tuple[float, float]:
    """Return a validated 2-tuple of finite numbers."""
    if len(value) != 2:
        raise ValueError(f"{name} must be a pair of numbers")
    return (
        _finite(value[0], f"{name}[0]", positive=positive),
        _finite(value[1], f"{name}[1]", positive=positive),
    )


def _positive_int(value: int, name: str) -> int:
    """Return a positive integer, rejecting booleans and non-integers."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class FocusedTarget:
    """One host-selected target for the round-two focused revisit.

    The center is in field coordinates, matching the survey tile coordinates.
    ``roi_size_um`` is optional; when it is absent the recipe's focused ROI size
    applies. Detection, scoring, deduplication and selection happen in the host
    before these records are built.
    """

    target_id: str
    center_xy_um: tuple[float, float]
    roi_size_um: tuple[float, float] | None = None

    def __post_init__(self) -> None:
        """Check the identifier, center and optional ROI."""
        if not isinstance(self.target_id, str) or not self.target_id.strip():
            raise ValueError("target_id must be a nonempty string")
        object.__setattr__(
            self,
            "center_xy_um",
            _pair(self.center_xy_um, "center_xy_um", positive=False),
        )
        if self.roi_size_um is not None:
            object.__setattr__(
                self,
                "roi_size_um",
                _pair(self.roi_size_um, "roi_size_um", positive=True),
            )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary of the declared values."""
        return {
            "target_id": self.target_id,
            "center_xy_um": list(self.center_xy_um),
            "roi_size_um": (
                None if self.roi_size_um is None else list(self.roi_size_um)
            ),
        }


@dataclass(frozen=True)
class StitchingPreview:
    """Host-stitching context for the overlapping survey mosaic.

    The preview describes the grid a host needs to stitch: tile counts, overlap,
    the physical step, any field overscan and, when the tile image shape is
    declared, the expected mosaic shape in pixels. It is metadata only; this
    package performs no registration, blending or stitching.
    """

    tile_rows: int
    tile_columns: int
    tile_size_um: tuple[float, float]
    overlap: float
    step_xy_um: tuple[float, float]
    overscan_um: tuple[float, float]
    tile_shape_yx: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Check the grid, overlap, step and optional image shape."""
        if (
            isinstance(self.tile_rows, bool)
            or not isinstance(self.tile_rows, int)
            or self.tile_rows < 1
        ):
            raise ValueError("tile_rows must be a positive integer")
        if (
            isinstance(self.tile_columns, bool)
            or not isinstance(self.tile_columns, int)
            or self.tile_columns < 1
        ):
            raise ValueError("tile_columns must be a positive integer")
        object.__setattr__(
            self,
            "tile_size_um",
            _pair(self.tile_size_um, "tile_size_um", positive=True),
        )
        overlap = _finite(self.overlap, "overlap", positive=False)
        if overlap >= 1:
            raise ValueError("overlap must be less than 1")
        object.__setattr__(self, "overlap", overlap)
        object.__setattr__(
            self, "step_xy_um", _pair(self.step_xy_um, "step_xy_um", positive=True)
        )
        object.__setattr__(
            self,
            "overscan_um",
            _pair(self.overscan_um, "overscan_um", positive=False),
        )
        if self.tile_shape_yx is not None:
            shape = self.tile_shape_yx
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or isinstance(shape[0], bool)
                or isinstance(shape[1], bool)
                or not isinstance(shape[0], int)
                or not isinstance(shape[1], int)
                or shape[0] < 1
                or shape[1] < 1
            ):
                raise ValueError(
                    "tile_shape_yx must be a pair of positive integers when supplied"
                )

    @property
    def mosaic_shape_yx(self) -> tuple[int, int] | None:
        """Return the expected stitched mosaic shape in pixels, when derivable."""
        if self.tile_shape_yx is None:
            return None
        tile_height_px, tile_width_px = self.tile_shape_yx
        step_width_px = tile_width_px * (1.0 - self.overlap)
        step_height_px = tile_height_px * (1.0 - self.overlap)
        return (
            ceil(tile_height_px + (self.tile_rows - 1) * step_height_px),
            ceil(tile_width_px + (self.tile_columns - 1) * step_width_px),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe dictionary, including the derived mosaic shape."""
        mosaic = self.mosaic_shape_yx
        return {
            "tile_rows": self.tile_rows,
            "tile_columns": self.tile_columns,
            "tile_size_um": list(self.tile_size_um),
            "overlap": self.overlap,
            "step_xy_um": list(self.step_xy_um),
            "overscan_um": list(self.overscan_um),
            "tile_shape_yx": (
                None if self.tile_shape_yx is None else list(self.tile_shape_yx)
            ),
            "mosaic_shape_yx": None if mosaic is None else list(mosaic),
        }


@dataclass(frozen=True)
class SearchImagingRecipe:
    """Declarative settings for one two-round search-and-image workflow.

    ``survey`` and ``focused`` carry independent pixel size and resolution, so
    the survey may use the same magnification as the focused round. The recipe
    is inert: a host preflights it against device limits and owns the detection
    and review step between the two rounds.
    """

    recipe_id: str
    field_size_um: tuple[float, float]
    survey: SurveySpec
    focused: FocusedSpec
    stage: StageProfile
    review_s: float = 0.0
    channel_ids: tuple[str, ...] = ()
    origin_xy_um: tuple[float, float] = (0.0, 0.0)
    tile_shape_yx: tuple[int, int] | None = None
    specimen_id: str = "unspecified-specimen"
    coordinate_frame: str = "stage-xy"
    calibration_id: str = "uncalibrated-stage-scan"
    version: str = "0.1-local"

    def __post_init__(self) -> None:
        """Check the recipe identity, geometry, channels and image shape."""
        for name in (
            "recipe_id",
            "specimen_id",
            "coordinate_frame",
            "calibration_id",
            "version",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty string")
        object.__setattr__(
            self,
            "field_size_um",
            _pair(self.field_size_um, "field_size_um", positive=True),
        )
        if not isinstance(self.survey, SurveySpec):
            raise TypeError("survey must be a SurveySpec")
        if not isinstance(self.focused, FocusedSpec):
            raise TypeError("focused must be a FocusedSpec")
        if not isinstance(self.stage, StageProfile):
            raise TypeError("stage must be a StageProfile")
        object.__setattr__(
            self, "review_s", _finite(self.review_s, "review_s", positive=False)
        )
        object.__setattr__(
            self,
            "origin_xy_um",
            _pair(self.origin_xy_um, "origin_xy_um", positive=False),
        )
        if not isinstance(self.channel_ids, tuple):
            raise TypeError("channel_ids must be a tuple of channel identifiers")
        if any(
            not isinstance(channel_id, str) or not channel_id.strip()
            for channel_id in self.channel_ids
        ):
            raise ValueError("channel_ids must be nonempty strings")
        if len(set(self.channel_ids)) != len(self.channel_ids):
            raise ValueError("channel_ids must not repeat")
        if self.channel_ids and len(self.channel_ids) != self.focused.channels:
            raise ValueError(
                "channel_ids must name exactly focused.channels identifiers"
            )
        if self.tile_shape_yx is not None:
            shape = self.tile_shape_yx
            if (
                not isinstance(shape, tuple)
                or len(shape) != 2
                or any(
                    isinstance(size, bool) or not isinstance(size, int) or size < 1
                    for size in shape
                )
            ):
                raise ValueError(
                    "tile_shape_yx must be a pair of positive integers when supplied"
                )

    def survey_tiles(self) -> tuple[SurveyTile, ...]:
        """Return the serpentine survey tiles in acquisition order."""
        return _survey_spec(self).survey_tiles()

    def survey_positions(self) -> tuple[ScanPosition, ...]:
        """Return the survey tiles as stage positions with row and column."""
        return survey_positions(self)

    def overscan_um(self) -> tuple[float, float]:
        """Return how far the last tile edge extends past the field per axis."""
        return _survey_spec(self).overscan_um()


@dataclass(frozen=True)
class SearchImagingRecords:
    """Both round recipes plus the stitching context for one workflow."""

    recipe: SearchImagingRecipe
    survey_recipe: StageScanRecipe
    focused_recipe: StageScanRecipe
    stitching: StitchingPreview

    def __post_init__(self) -> None:
        """Check the record types."""
        if not isinstance(self.recipe, SearchImagingRecipe):
            raise TypeError("recipe must be a SearchImagingRecipe")
        if not isinstance(self.survey_recipe, StageScanRecipe):
            raise TypeError("survey_recipe must be a StageScanRecipe")
        if not isinstance(self.focused_recipe, StageScanRecipe):
            raise TypeError("focused_recipe must be a StageScanRecipe")
        if not isinstance(self.stitching, StitchingPreview):
            raise TypeError("stitching must be a StitchingPreview")


def _require_recipe(recipe: SearchImagingRecipe) -> SearchImagingRecipe:
    """Return the recipe after a type check."""
    if not isinstance(recipe, SearchImagingRecipe):
        raise TypeError("recipe must be a SearchImagingRecipe")
    return recipe


def _survey_spec(recipe: SearchImagingRecipe) -> SearchImagingSpec:
    """Build the grid-only simulation specification for a recipe."""
    return SearchImagingSpec(
        field_size_um=recipe.field_size_um,
        survey=recipe.survey,
        focused=recipe.focused,
        stage=recipe.stage,
        origin_xy_um=recipe.origin_xy_um,
        review_s=recipe.review_s,
    )


def _validated_targets(
    recipe: SearchImagingRecipe, targets: Iterable[FocusedTarget]
) -> tuple[FocusedTarget, ...]:
    """Validate target records against each other and the declared field."""
    width, height = recipe.field_size_um
    validated: list[FocusedTarget] = []
    seen: set[str] = set()
    for target in targets:
        if not isinstance(target, FocusedTarget):
            raise TypeError("targets must contain FocusedTarget records")
        if target.target_id in seen:
            raise ValueError(f"duplicate target id {target.target_id!r}")
        seen.add(target.target_id)
        x, y = target.center_xy_um
        if not (0 <= x <= width and 0 <= y <= height):
            raise ValueError(
                f"target {target.target_id} center {target.center_xy_um} is outside the field"
            )
        validated.append(target)
    return tuple(validated)


def survey_tiles(recipe: SearchImagingRecipe) -> tuple[SurveyTile, ...]:
    """Return the overlapping serpentine survey tiles in acquisition order."""
    return _require_recipe(recipe).survey_tiles()


def survey_positions(recipe: SearchImagingRecipe) -> tuple[ScanPosition, ...]:
    """Return the survey tiles as stage positions with their grid row and column."""
    return tuple(
        ScanPosition(
            tile.tile_id,
            tile.center_xy_um[0],
            tile.center_xy_um[1],
            tile.row,
            tile.column,
        )
        for tile in survey_tiles(_require_recipe(recipe))
    )


def survey_overscan_um(recipe: SearchImagingRecipe) -> tuple[float, float]:
    """Return the survey overscan past the field edge on each axis."""
    return _require_recipe(recipe).overscan_um()


def stitching_preview(recipe: SearchImagingRecipe) -> StitchingPreview:
    """Return the stitching context for the survey mosaic."""
    recipe = _require_recipe(recipe)
    tiles = recipe.survey_tiles()
    rows = max(tile.row for tile in tiles) + 1
    columns = max(tile.column for tile in tiles) + 1
    tile_width, tile_height = recipe.survey.tile_size_um
    return StitchingPreview(
        tile_rows=rows,
        tile_columns=columns,
        tile_size_um=recipe.survey.tile_size_um,
        overlap=recipe.survey.overlap,
        step_xy_um=(
            tile_width * (1.0 - recipe.survey.overlap),
            tile_height * (1.0 - recipe.survey.overlap),
        ),
        overscan_um=recipe.overscan_um(),
        tile_shape_yx=recipe.tile_shape_yx,
    )


def _survey_point(recipe: SearchImagingRecipe) -> AcquisitionPoint:
    """Return the configured stationary survey point for corner context."""
    tile_width, tile_height = recipe.survey.tile_size_um
    frames = recipe.survey.frames
    period = frames * (recipe.survey.exposure_s + recipe.survey.readout_s)
    period += recipe.survey.overhead_s
    return AcquisitionPoint(
        velocity_um_s=0.0,
        exposure_s=recipe.survey.exposure_s,
        width_um=tile_width,
        frame_period_s=max(period, recipe.survey.exposure_s),
        pixel_size_um=recipe.survey.pixel_size_um,
        resolution_um=recipe.survey.resolution_um,
        height_um=tile_height,
    )


def _focused_point(recipe: SearchImagingRecipe) -> AcquisitionPoint:
    """Return the configured stationary focused point for corner context."""
    frames = recipe.focused.frames * recipe.focused.channels * recipe.focused.z_planes
    period = frames * (recipe.focused.exposure_s + recipe.focused.readout_s)
    period += recipe.focused.overhead_s
    width, height = recipe.focused.roi_size_um
    return AcquisitionPoint(
        velocity_um_s=0.0,
        exposure_s=recipe.focused.exposure_s,
        width_um=width,
        frame_period_s=max(period, recipe.focused.exposure_s),
        pixel_size_um=recipe.focused.pixel_size_um,
        resolution_um=recipe.focused.resolution_um,
        height_um=height,
    )


def _corner_value(
    envelope: AcquisitionEnvelope | None, point: AcquisitionPoint
) -> str | None:
    """Return the operating corner code for a point, when an envelope is given."""
    if envelope is None:
        return None
    corner = envelope.corner(point)
    return None if corner is None else corner.value


def _round_metadata(
    md: Mapping[str, Any] | None, required: Mapping[str, Any]
) -> dict[str, Any]:
    """Merge round metadata, refusing to override the caller's planning fields."""
    result = dict(md or {})
    reserved = set(required).intersection(result)
    if reserved:
        raise ValueError(
            "metadata cannot override planner fields: " + ", ".join(sorted(reserved))
        )
    result.update(required)
    return result


def build_survey_recipe(
    recipe: SearchImagingRecipe,
    *,
    envelope: AcquisitionEnvelope | None = None,
) -> StageScanRecipe:
    """Build the phase-survey recipe from the overlapping grid.

    ``envelope`` only labels the configured snap operating point with its
    acquisition-space corner; it changes no geometry. The host still owns
    detection and review after this recipe's run.
    """
    recipe = _require_recipe(recipe)
    stitching = stitching_preview(recipe)
    point = _survey_point(recipe)
    survey_metadata = {
        "planner_search_workflow": WORKFLOW_ID,
        "planner_search_imaging_recipe_id": recipe.recipe_id,
        "planner_search_round": "survey",
        "planner_search_round_index": 1,
        "planner_search_detection_owner": "host",
        "planner_survey_overlap": recipe.survey.overlap,
        "planner_survey_tile_size_um": list(recipe.survey.tile_size_um),
        "planner_survey_step_xy_um": list(stitching.step_xy_um),
        "planner_survey_grid_rows": stitching.tile_rows,
        "planner_survey_grid_columns": stitching.tile_columns,
        "planner_survey_overscan_um": list(stitching.overscan_um),
        "planner_acquisition_point": point.to_dict(),
        "planner_acquisition_corner": _corner_value(envelope, point),
    }
    return StageScanRecipe(
        recipe_id=f"{recipe.recipe_id}-survey",
        specimen_id=recipe.specimen_id,
        positions=recipe.survey_positions(),
        settings=AcquisitionSettings(
            exposure_ms=recipe.survey.exposure_s * 1000.0,
            frames=recipe.survey.frames,
            channel_id=recipe.channel_ids[0] if recipe.channel_ids else None,
            image_shape_yx=recipe.tile_shape_yx,
            metadata=survey_metadata,
        ),
        version=recipe.version,
        phase=Phase.SURVEY,
        coordinate_frame=recipe.coordinate_frame,
        calibration_id=recipe.calibration_id,
    )


def build_focused_recipe(
    recipe: SearchImagingRecipe,
    targets: Iterable[FocusedTarget],
    *,
    envelope: AcquisitionEnvelope | None = None,
) -> StageScanRecipe:
    """Build the phase-revisit recipe for the host-selected targets.

    Targets are visited in the order supplied; the host owns their detection,
    review and ordering. ``AcquisitionSettings.frames`` is the product of zoom
    frames, channels and z planes so one capture per position still accounts for
    every focused frame.
    """
    recipe = _require_recipe(recipe)
    validated = _validated_targets(recipe, targets)
    point = _focused_point(recipe)
    frames = recipe.focused.frames * recipe.focused.channels * recipe.focused.z_planes
    single_channel = recipe.focused.channels == 1 and bool(recipe.channel_ids)
    focused_metadata = {
        "planner_search_workflow": WORKFLOW_ID,
        "planner_search_imaging_recipe_id": recipe.recipe_id,
        "planner_search_round": "focused",
        "planner_search_round_index": 2,
        "planner_search_detection_owner": "host",
        "planner_focused_target_ids": [target.target_id for target in validated],
        "planner_focused_frames": recipe.focused.frames,
        "planner_focused_channels": recipe.focused.channels,
        "planner_focused_z_planes": recipe.focused.z_planes,
        "planner_focused_frames_per_target": frames,
        "planner_focused_channel_ids": list(recipe.channel_ids),
        "planner_focused_target_roi_size_um": [
            None if target.roi_size_um is None else list(target.roi_size_um)
            for target in validated
        ],
        "planner_acquisition_point": point.to_dict(),
        "planner_acquisition_corner": _corner_value(envelope, point),
    }
    return StageScanRecipe(
        recipe_id=f"{recipe.recipe_id}-focused",
        specimen_id=recipe.specimen_id,
        positions=tuple(
            ScanPosition(
                target.target_id,
                target.center_xy_um[0],
                target.center_xy_um[1],
            )
            for target in validated
        ),
        settings=AcquisitionSettings(
            exposure_ms=recipe.focused.exposure_s * 1000.0,
            frames=frames,
            channel_id=(recipe.channel_ids[0] if single_channel else None),
            metadata=focused_metadata,
        ),
        version=recipe.version,
        phase=Phase.REVISIT,
        coordinate_frame=recipe.coordinate_frame,
        calibration_id=recipe.calibration_id,
    )


def search_imaging_records(
    recipe: SearchImagingRecipe,
    targets: Iterable[FocusedTarget],
    *,
    envelope: AcquisitionEnvelope | None = None,
) -> SearchImagingRecords:
    """Return both round recipes and the survey stitching context.

    Detection and candidate selection are the host's responsibility; this
    function only lowers the declared settings and the supplied target records
    into inert recipes.
    """
    recipe = _require_recipe(recipe)
    target_records = _validated_targets(recipe, targets)
    return SearchImagingRecords(
        recipe=recipe,
        survey_recipe=build_survey_recipe(recipe, envelope=envelope),
        focused_recipe=build_focused_recipe(recipe, target_records, envelope=envelope),
        stitching=stitching_preview(recipe),
    )


def _search_imaging_spec(
    recipe: SearchImagingRecipe, targets: Iterable[FocusedTarget]
) -> SearchImagingSpec:
    """Convert a recipe and targets into the simulator's inert specification."""
    records = tuple(
        MockTarget(target.target_id, target.center_xy_um)
        for target in _validated_targets(recipe, targets)
    )
    return SearchImagingSpec(
        field_size_um=recipe.field_size_um,
        survey=recipe.survey,
        focused=recipe.focused,
        stage=recipe.stage,
        targets=records,
        origin_xy_um=recipe.origin_xy_um,
        review_s=recipe.review_s,
    )


def estimate_search_imaging(
    recipe: SearchImagingRecipe,
    targets: Iterable[FocusedTarget] = (),
    *,
    envelope: AcquisitionEnvelope | None = None,
    limits: MeasurementLimits | None = None,
) -> SimulationReport:
    """Estimate the two-round run with the deterministic mock simulator.

    Round one is reported as the configured snap point and as the equivalent
    continuous point; round two is the focused point. ``envelope`` supplies the
    model corners and ``limits`` supplies feasibility verdicts. Detection and
    candidate selection are outside this estimate: the targets are supplied
    records, not detections.
    """
    recipe = _require_recipe(recipe)
    if envelope is not None and not isinstance(envelope, AcquisitionEnvelope):
        raise TypeError("envelope must be an AcquisitionEnvelope or None")
    if limits is not None and not isinstance(limits, MeasurementLimits):
        raise TypeError("limits must be a MeasurementLimits or None")
    return simulate_search_imaging(
        _search_imaging_spec(recipe, targets),
        envelope=envelope,
        limits=limits,
    )


def survey_scan_plan(
    recipe: SearchImagingRecipe,
    *,
    stage_xy: Any | None = None,
    stage_x: Any | None = None,
    stage_y: Any | None = None,
    camera: Any,
    readables: Sequence[Any] = (),
    prepare_camera: Callable[[Any], Iterable[Any]] | None = None,
    prepare_capture: Callable[[Any, int], Iterable[Any]] | None = None,
    capture_one: Callable[[Any, bool], Iterable[Any]] | None = None,
    md: Mapping[str, Any] | None = None,
    stream_name: str = DEFAULT_SURVEY_STREAM,
) -> MsgGenerator[Any]:
    """Emit the round-one survey plan: one run over the overlapping tiles.

    Device arguments mirror :func:`planner.plans.snap_scan_plan`, which this
    generator delegates to. The host preflights limits and adapter capabilities.
    """
    survey_recipe = build_survey_recipe(recipe)
    round_md = _round_metadata(
        md,
        {
            **survey_recipe.settings.metadata,
            "planner_search_workflow": WORKFLOW_ID,
            "planner_search_imaging_recipe_id": recipe.recipe_id,
            "planner_search_round": "survey",
            "planner_search_round_index": 1,
            "planner_search_stream": stream_name,
            "planner_search_review_boundary": "host-owned detection and review",
        },
    )
    yield from snap_scan_plan(
        survey_recipe,
        stage_xy=stage_xy,
        stage_x=stage_x,
        stage_y=stage_y,
        camera=camera,
        readables=readables,
        prepare_camera=prepare_camera,
        prepare_capture=prepare_capture,
        capture_one=capture_one,
        md=round_md,
        stream_name=stream_name,
    )


def focused_revisit_plan(
    recipe: SearchImagingRecipe,
    targets: Iterable[FocusedTarget],
    *,
    stage_xy: Any | None = None,
    stage_x: Any | None = None,
    stage_y: Any | None = None,
    camera: Any,
    readables: Sequence[Any] = (),
    prepare_camera: Callable[[Any], Iterable[Any]] | None = None,
    prepare_capture: Callable[[Any, int], Iterable[Any]] | None = None,
    capture_one: Callable[[Any, bool], Iterable[Any]] | None = None,
    md: Mapping[str, Any] | None = None,
    stream_name: str = DEFAULT_FOCUSED_STREAM,
) -> MsgGenerator[Any]:
    """Emit the round-two focused plan: one run over the host-selected targets.

    Device arguments mirror :func:`planner.plans.snap_scan_plan`, which this
    generator delegates to. The target list is the host's detection/review
    output; this plan does not detect or rank anything.
    """
    target_records = _validated_targets(recipe, targets)
    focused_recipe = build_focused_recipe(recipe, target_records)
    round_md = _round_metadata(
        md,
        {
            **focused_recipe.settings.metadata,
            "planner_search_workflow": WORKFLOW_ID,
            "planner_search_imaging_recipe_id": recipe.recipe_id,
            "planner_search_round": "focused",
            "planner_search_round_index": 2,
            "planner_search_stream": stream_name,
            "planner_search_target_ids": [
                target.target_id for target in target_records
            ],
        },
    )
    yield from snap_scan_plan(
        focused_recipe,
        stage_xy=stage_xy,
        stage_x=stage_x,
        stage_y=stage_y,
        camera=camera,
        readables=readables,
        prepare_camera=prepare_camera,
        prepare_capture=prepare_capture,
        capture_one=capture_one,
        md=round_md,
        stream_name=stream_name,
    )


__all__ = [
    "DEFAULT_FOCUSED_STREAM",
    "DEFAULT_SURVEY_STREAM",
    "WORKFLOW_ID",
    "FocusedTarget",
    "SearchImagingRecipe",
    "SearchImagingRecords",
    "StitchingPreview",
    "build_focused_recipe",
    "build_survey_recipe",
    "estimate_search_imaging",
    "focused_revisit_plan",
    "search_imaging_records",
    "stitching_preview",
    "survey_overscan_um",
    "survey_positions",
    "survey_scan_plan",
    "survey_tiles",
]
