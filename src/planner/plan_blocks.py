"""Discoverable contracts for public Bluesky plan generators."""

from __future__ import annotations

from dataclasses import dataclass
from inspect import Parameter, isgeneratorfunction, signature
from math import isfinite
from typing import TYPE_CHECKING, Any, get_origin, get_type_hints

from bluesky.utils import MsgGenerator

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

from .contracts import ResourceRequirement
from .operating_schedule import operating_point_schedule_plan
from .plans import (
    collect_while_waiting_plan,
    hand_eye_calibration_plan,
    live_median_scan,
    live_stream,
    prepare_live_stream,
    push_broom_plan,
    replay_dataset_plan,
    snap_scan_plan,
    teardown_live_stream,
)
from .search_imaging import (
    DEFAULT_FOCUSED_STREAM,
    DEFAULT_SURVEY_STREAM,
    focused_revisit_plan,
    survey_scan_plan,
)


@dataclass(frozen=True)
class PlanParameterContract:
    """Describe one public plan-block argument for a host view."""

    name: str
    type_name: str
    description: str
    units: str = ""
    required: bool = True
    has_default: bool = False
    default: str | int | float | bool | tuple[()] | None = None

    def __post_init__(self) -> None:
        """Validate the host-facing argument description."""
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (self.name, self.type_name, self.description)
        ):
            raise ValueError("parameter name, type_name, and description are required")
        if not isinstance(self.units, str):
            raise ValueError("parameter units must be a string")
        if not isinstance(self.required, bool) or not isinstance(
            self.has_default, bool
        ):
            raise ValueError("parameter required/default flags must be booleans")
        if self.required and self.has_default:
            raise ValueError("a required parameter cannot declare a default")
        scalar_default = isinstance(self.default, (str, int, float, bool, type(None)))
        empty_sequence_default = isinstance(self.default, tuple) and not self.default
        if not scalar_default and not empty_sequence_default:
            raise ValueError("defaults must be scalar values or an empty tuple")
        if isinstance(self.default, float) and not isfinite(self.default):
            raise ValueError("float defaults must be finite")
        if not self.has_default and self.default is not None:
            raise ValueError("a parameter default requires has_default=True")


@dataclass(frozen=True)
class PlanBlockDescriptor:
    """Pair a planner plan stub with its reusable execution contract.

    The descriptor is runtime metadata, not a persisted run record. It does not
    validate the host's devices, limits, resource locks, or execution outcome.
    """

    block_id: str
    contract_version: int
    plan_stub: Callable[..., Iterable[Any]]
    summary: str
    parameters: tuple[PlanParameterContract, ...]
    device_contracts: tuple[str, ...]
    resource_requirements: tuple[ResourceRequirement, ...]
    physical_effects: tuple[str, ...]
    preconditions: tuple[str, ...]
    completion_condition: str
    failure_condition: str
    validation_evidence: tuple[str, ...] = ()

    def to_view_model(self) -> dict[str, Any]:
        """Return host-facing display metadata without the plan callable.

        The result is a convenience for building a view from this in-memory
        descriptor. It is not a persistence format, execution request, or
        substitute for host preflight.
        """
        return {
            "block_id": self.block_id,
            "contract_version": self.contract_version,
            "plan_name": self.plan_stub.__name__,
            "summary": self.summary,
            "parameters": [
                {
                    "name": parameter.name,
                    "type_name": parameter.type_name,
                    "description": parameter.description,
                    "units": parameter.units,
                    "required": parameter.required,
                    "has_default": parameter.has_default,
                    "default": parameter.default,
                }
                for parameter in self.parameters
            ],
            "device_contracts": list(self.device_contracts),
            "resource_requirements": [
                {"name": resource.name, "mode": resource.mode}
                for resource in self.resource_requirements
            ],
            "physical_effects": list(self.physical_effects),
            "preconditions": list(self.preconditions),
            "completion_condition": self.completion_condition,
            "failure_condition": self.failure_condition,
            "validation_evidence": list(self.validation_evidence),
        }

    def __post_init__(self) -> None:
        """Validate identity, argument uniqueness, and declared effects."""
        if (
            not isinstance(self.block_id, str)
            or not self.block_id.startswith("planner.")
            or not self.block_id.removeprefix("planner.").strip()
            or self.block_id != self.block_id.strip()
        ):
            raise ValueError("block_id must use the planner.<name> form")
        if (
            isinstance(self.contract_version, bool)
            or not isinstance(self.contract_version, int)
            or self.contract_version < 1
        ):
            raise ValueError("contract_version must be a positive integer")
        if not callable(self.plan_stub):
            raise TypeError("plan_stub must be callable")
        if not isgeneratorfunction(self.plan_stub):
            raise TypeError("plan_stub must be a Bluesky generator function")
        return_type = get_type_hints(self.plan_stub).get("return")
        if get_origin(return_type) is not get_origin(MsgGenerator[Any]):
            raise TypeError("plan_stub must return bluesky.utils.MsgGenerator")
        if not isinstance(self.parameters, tuple):
            raise TypeError("parameters must be a tuple")
        if any(
            not isinstance(value, str) or not value.strip()
            for value in (
                self.summary,
                self.completion_condition,
                self.failure_condition,
            )
        ):
            raise ValueError("summary and terminal conditions are required")
        if any(
            not isinstance(parameter, PlanParameterContract)
            for parameter in self.parameters
        ):
            raise TypeError("parameters must contain PlanParameterContract values")
        if len({parameter.name for parameter in self.parameters}) != len(
            self.parameters
        ):
            raise ValueError("parameter names must be unique within a plan block")
        plan_parameters = signature(self.plan_stub).parameters
        if any(
            parameter.kind in (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
            for parameter in plan_parameters.values()
        ):
            raise ValueError(
                "plan stubs with variadic arguments need an explicit contract"
            )
        if tuple(plan_parameters) != tuple(
            parameter.name for parameter in self.parameters
        ):
            raise ValueError("descriptor parameters must match the plan stub signature")
        for contract in self.parameters:
            plan_parameter = plan_parameters[contract.name]
            has_default = plan_parameter.default is not Parameter.empty
            if contract.required != (not has_default):
                raise ValueError(
                    f"required flag for {contract.name!r} disagrees with the plan stub"
                )
            if contract.has_default != has_default:
                raise ValueError(
                    f"has_default for {contract.name!r} disagrees with the plan stub"
                )
            if has_default and contract.default != plan_parameter.default:
                raise ValueError(
                    f"default for {contract.name!r} disagrees with the plan stub"
                )
        for name, values in (
            ("device_contracts", self.device_contracts),
            ("physical_effects", self.physical_effects),
            ("preconditions", self.preconditions),
            ("validation_evidence", self.validation_evidence),
        ):
            if not isinstance(values, tuple):
                raise TypeError(f"{name} must be a tuple")
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError(f"{name} entries must be nonempty strings")
            if len(values) != len(set(values)):
                raise ValueError(f"{name} entries must be unique")
        if not isinstance(self.resource_requirements, tuple) or any(
            not isinstance(resource, ResourceRequirement)
            for resource in self.resource_requirements
        ):
            raise TypeError("resource_requirements must contain typed requirements")
        if len({resource.name for resource in self.resource_requirements}) != len(
            self.resource_requirements
        ):
            raise ValueError("resource requirements must be unique")


PLAN_BLOCKS = (
    PlanBlockDescriptor(
        block_id="planner.live_stream",
        contract_version=1,
        plan_stub=live_stream,
        summary="Capture from staged live flyers while collecting stream assets.",
        parameters=(
            PlanParameterContract(
                "detectors",
                "Sequence[LiveFlyer]",
                "Staged and prepared Bluesky flyers to capture.",
            ),
            PlanParameterContract(
                "stream_name", "str", "Name of the Bluesky event stream."
            ),
            PlanParameterContract("parent", "str", "Parent run UID."),
            PlanParameterContract(
                "median_scan",
                "str | None",
                "Optional median-reference run UID.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "until",
                "Iterable[BlueskyMsg] | None",
                "Optional plan run while live flyers continue capturing.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "flush_period_s",
                "float",
                "Maximum requested-sleep interval between asset collections.",
                units="s",
                required=False,
                has_default=True,
                default=0.5,
            ),
            PlanParameterContract(
                "on_complete",
                "Callable[[LiveFlyer, Any], Iterable[Any]] | None",
                "Optional host progress callback for flyer completion.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "on_flush",
                "Callable[[LiveFlyer], Iterable[Any]] | None",
                "Optional host progress callback after asset collection.",
                required=False,
                has_default=True,
            ),
        ),
        device_contracts=("planner.protocols.LiveFlyer",),
        resource_requirements=(
            ResourceRequirement("camera"),
            ResourceRequirement("storage"),
        ),
        physical_effects=("image_acquisition", "stream_asset_write"),
        preconditions=(
            "The host stages and prepares each flyer with its trigger configuration.",
            "The host holds the camera and storage resources for the full capture.",
        ),
        completion_condition=(
            "Flyers complete, pending assets are collected, flyers are unstaged, "
            "and the nested run closes."
        ),
        failure_condition=(
            "An interrupted flyer, callback, or collection leaves final device and "
            "run state for host reconciliation."
        ),
    ),
    PlanBlockDescriptor(
        block_id="planner.live_median_scan",
        contract_version=1,
        plan_stub=live_median_scan,
        summary="Acquire a four-leg XY motion sample for a host temporal median.",
        parameters=(
            PlanParameterContract(
                "detectors",
                "Sequence[LiveFlyer]",
                "Staged flyers that will be triggered at each position.",
            ),
            PlanParameterContract(
                "motor", "XYMotor", "Readable device with separately movable X/Y axes."
            ),
            PlanParameterContract(
                "step_um",
                "float",
                "Commanded distance for each square-motion step.",
                units="um",
            ),
            PlanParameterContract(
                "frames_per_side",
                "int",
                "Number of samples collected on each of the four legs.",
            ),
            PlanParameterContract(
                "parent",
                "str | None",
                "Optional parent run UID.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "purpose",
                "str",
                "Purpose recorded in run metadata.",
                required=False,
                has_default=True,
                default="median_scan",
            ),
            PlanParameterContract(
                "stream_name",
                "str",
                "Bluesky event stream name.",
                required=False,
                has_default=True,
                default="median_scan",
            ),
            PlanParameterContract(
                "on_frame",
                "Callable[[int, int, Any, float], Iterable[Any]] | None",
                "Optional host progress callback after each saved frame.",
                required=False,
                has_default=True,
            ),
        ),
        device_contracts=(
            "planner.protocols.LiveFlyer",
            "planner.protocols.XYMotor",
        ),
        resource_requirements=(
            ResourceRequirement("camera"),
            ResourceRequirement("stage"),
            ResourceRequirement("storage"),
        ),
        physical_effects=("stage_motion", "image_acquisition"),
        preconditions=(
            "The host verifies travel limits and retains stage/camera resources.",
            "The motor exposes X and Y axes and the flyers are staged.",
        ),
        completion_condition=(
            "Four signed legs are commanded, with an image and pre-step stage "
            "readback per sample. Commanded displacements sum to zero; the final "
            "return position is not verified by a post-move readback."
        ),
        failure_condition=(
            "An interruption can leave the stage mid-path; the host must read back "
            "position and reconcile before another plan."
        ),
    ),
    PlanBlockDescriptor(
        block_id="planner.snap_scan",
        contract_version=1,
        plan_stub=snap_scan_plan,
        summary="Visit a validated XY tile sequence and capture one image per point.",
        parameters=(
            PlanParameterContract(
                "recipe",
                "StageScanRecipe | SnapScanPlan | ObservedSnapGridFit",
                "Validated plan positions and optional coverage provenance.",
            ),
            PlanParameterContract(
                "stage_xy",
                "ReadableMovable | None",
                "Combined XY stage binding, if used.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stage_x",
                "Movable | None",
                "Separate X stage axis, when stage_xy is not used.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stage_y",
                "Movable | None",
                "Separate Y stage axis, when stage_xy is not used.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract("camera", "Any", "Bluesky camera detector."),
            PlanParameterContract(
                "readables",
                "Sequence[Readable]",
                "Additional stage readback signals included in each event.",
                required=False,
                has_default=True,
                default=(),
            ),
            PlanParameterContract(
                "prepare_camera",
                "Callable[[Any], Iterable[Any]] | None",
                "Host adapter plan run before each trigger/read pair.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "prepare_capture",
                "Callable[[Any, int], Iterable[Any]] | None",
                "Host plan that prepares one external capture for the scan.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "capture_one",
                "Callable[[Any, bool], Iterable[Any]] | None",
                "Host plan that triggers or reads one frame.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "md",
                "Mapping[str, Any] | None",
                "Additional run metadata; planner-owned keys are reserved.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stream_name",
                "str",
                "Bluesky event stream name.",
                required=False,
                has_default=True,
                default="primary",
            ),
        ),
        device_contracts=(
            "Bluesky Readable/Movable stage",
            "Bluesky Triggerable/Readable camera",
        ),
        resource_requirements=(
            ResourceRequirement("stage"),
            ResourceRequirement("camera"),
            ResourceRequirement("storage"),
        ),
        physical_effects=("stage_motion", "image_acquisition"),
        preconditions=(
            "The host validates the recipe against calibrated travel limits and adapter capabilities.",
            "Exactly one supported combined or separate XY stage binding is supplied.",
        ),
        completion_condition=(
            "An image event is emitted for each validated scan position and the run closes; "
            "the stage remains at the final position."
        ),
        failure_condition=(
            "An interrupted move or capture may leave the stage at an intermediate position; "
            "the host reconciles readbacks and partial data."
        ),
    ),
    PlanBlockDescriptor(
        block_id="planner.push_broom",
        contract_version=1,
        plan_stub=push_broom_plan,
        summary="Prepare, execute, and collect a sequence of push-broom strips.",
        parameters=(
            PlanParameterContract(
                "recipe", "BroomScanPlan", "Validated ordered strip geometry."
            ),
            PlanParameterContract(
                "flyer", "Flyer", "Bluesky strip motion/capture device."
            ),
            PlanParameterContract(
                "detectors",
                "Sequence[Readable]",
                "Inline event detectors, when not using external stream writers.",
                required=False,
                has_default=True,
                default=(),
            ),
            PlanParameterContract(
                "stream_writers",
                "Sequence[WritesStreamAssets]",
                "External image asset writers paired with geometry events.",
                required=False,
                has_default=True,
                default=(),
            ),
            PlanParameterContract(
                "md",
                "Mapping[str, Any] | None",
                "Additional run metadata; planner-owned keys are reserved.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stream_name",
                "str",
                "Bluesky event stream name.",
                required=False,
                has_default=True,
                default="primary",
            ),
            PlanParameterContract(
                "progress_callback",
                "Callable[[Any, int, int], Iterable[Any]] | None",
                "Host callback for per-strip completion/progress.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "validate_strip",
                "Callable[[Any, int, int], None] | None",
                "Host validation after strip geometry and assets are collected.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "post_scan",
                "Callable[[], Iterable[Any]] | None",
                "Host plan for successful-run work after all strips validate.",
                required=False,
                has_default=True,
            ),
        ),
        device_contracts=(
            "Bluesky Flyable/Preparable/Collectable flyer",
            "Bluesky WritesStreamAssets writer (optional)",
        ),
        resource_requirements=(
            ResourceRequirement("stage"),
            ResourceRequirement("camera"),
            ResourceRequirement("storage"),
        ),
        physical_effects=(
            "continuous_stage_motion",
            "image_acquisition",
            "stream_asset_write",
        ),
        preconditions=(
            "The host validates the full strip envelope, camera mode, and resource ownership.",
            "The flyer prepares bounded strips without starting motion.",
            "External stream writers declare data keys compatible with each strip event.",
            "Frame events preserve frame ID, image index, stage pose and exposure clock context.",
        ),
        completion_condition=(
            "Every strip's available geometry and asset documents are collected, "
            "validation succeeds, and optional post_scan work completes."
        ),
        failure_condition=(
            "The active flyer is stopped and available strip records are collected where "
            "possible; the host reconciles stage position and partial assets."
        ),
    ),
    PlanBlockDescriptor(
        block_id="planner.hand_eye_calibration",
        contract_version=2,
        plan_stub=hand_eye_calibration_plan,
        summary="Capture signed stage probes or a configurable XY grid for image-to-stage fitting.",
        parameters=(
            PlanParameterContract(
                "anchor_xy_um",
                "tuple[float, float]",
                "Preflighted stage anchor in physical micrometres.",
                units="um",
            ),
            PlanParameterContract(
                "stage_xy",
                "ReadableMovable | None",
                "Combined XY stage binding, if used.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stage_x",
                "Movable | None",
                "Separate X stage axis, when stage_xy is not used.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stage_y",
                "Movable | None",
                "Separate Y stage axis, when stage_xy is not used.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract("camera", "Any", "Bluesky camera detector."),
            PlanParameterContract(
                "x_readback",
                "Readable | None",
                "Optional explicit X stage readback.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "y_readback",
                "Readable | None",
                "Optional explicit Y stage readback.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "x_probe_um",
                "float",
                "Signed X probe magnitude.",
                units="um",
                required=False,
                has_default=True,
                default=10.0,
            ),
            PlanParameterContract(
                "y_probe_um",
                "float",
                "Signed Y probe magnitude.",
                units="um",
                required=False,
                has_default=True,
                default=10.0,
            ),
            PlanParameterContract(
                "repeats_per_direction",
                "int",
                "Number of signed bidirectional probe repetitions.",
                required=False,
                has_default=True,
                default=1,
            ),
            PlanParameterContract(
                "grid_shape_xy",
                "tuple[int, int] | None",
                "Optional odd column/row counts for a centered snake grid; overrides signed probes.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "grid_spacing_xy_um",
                "tuple[float, float] | None",
                "Optional X/Y grid spacing in reported micrometres; requires grid_shape_xy.",
                units="um",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "mode_context",
                "Mapping[str, str]",
                "Camera/objective mode identity recorded with captures.",
            ),
            PlanParameterContract(
                "readables",
                "Sequence[Readable]",
                "Additional readback signals recorded with each image.",
                required=False,
                has_default=True,
                default=(),
            ),
            PlanParameterContract(
                "coordinate_frame", "str", "Identifier for the stage coordinate frame."
            ),
            PlanParameterContract(
                "registration_roi_yxhw_px",
                "tuple[int, int, int, int] | None",
                "Optional image-registration ROI as y, x, height, width.",
                units="px",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "registration_preprocessing",
                "str",
                "Registration preprocessing mode recorded with the run.",
                required=False,
                has_default=True,
                default="none",
            ),
            PlanParameterContract(
                "capture_one",
                "Callable[[Any, Sequence[Any], str], Iterable[Any]] | None",
                "Optional host plan around each camera capture.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "md",
                "Mapping[str, Any] | None",
                "Additional run metadata; planner-owned keys are reserved.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stream_name",
                "str",
                "Bluesky event stream name.",
                required=False,
                has_default=True,
                default="calibration",
            ),
        ),
        device_contracts=(
            "Bluesky Movable stage axes",
            "Bluesky Triggerable/Readable camera",
        ),
        resource_requirements=(
            ResourceRequirement("stage"),
            ResourceRequirement("camera"),
            ResourceRequirement("storage"),
        ),
        physical_effects=("stage_motion", "image_acquisition"),
        preconditions=(
            "The host validates fresh stage anchor, exact mode context and physical travel limits.",
            "Exactly one supported combined or separate XY stage binding is supplied.",
            "The camera mode is fixed for every calibration capture.",
            "For a grid run, odd shape, spacing and total extent are validated before motion.",
        ),
        completion_condition=(
            "Reference and probe/grid captures complete, then the stage is commanded back to "
            "the declared anchor for closure. Fitting remains offline."
        ),
        failure_condition=(
            "An interrupted move/capture may leave the stage away from the anchor; the host "
            "must reconcile actual position before further action."
        ),
    ),
    PlanBlockDescriptor(
        block_id="planner.replay_dataset",
        contract_version=1,
        plan_stub=replay_dataset_plan,
        summary="Replay stored image frames as Bluesky events without device calls.",
        parameters=(
            PlanParameterContract(
                "source",
                "ReplaySource",
                "Lazy source of descriptor-consistent frames and provenance.",
            ),
            PlanParameterContract(
                "interval_s",
                "float",
                "Wall-clock spacing between emitted frames.",
                units="s",
                required=False,
                has_default=True,
                default=0.0,
            ),
            PlanParameterContract(
                "md",
                "Mapping[str, Any] | None",
                "Additional run metadata; planner-owned keys are reserved.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stream_name",
                "str",
                "Bluesky event stream name.",
                required=False,
                has_default=True,
                default="primary",
            ),
        ),
        device_contracts=(),
        resource_requirements=(),
        physical_effects=(),
        preconditions=(
            "The replay source yields frames matching its declared descriptors and identity.",
        ),
        completion_condition=(
            "All source frames are emitted with source frame/time context and the run closes."
        ),
        failure_condition=(
            "Source iteration or event emission errors propagate; no physical device state exists."
        ),
    ),
    PlanBlockDescriptor(
        block_id="planner.prepare_live_stream",
        contract_version=1,
        plan_stub=prepare_live_stream,
        summary="Prepare live flyers and optionally declare their stream.",
        parameters=(
            PlanParameterContract(
                "detectors", "Sequence[LiveFlyer]", "Flyers to prepare."
            ),
            PlanParameterContract(
                "trigger_info",
                "TriggerInfo",
                "Trigger configuration passed to each flyer.",
            ),
            PlanParameterContract(
                "stream_name", "str", "Optional stream declaration name."
            ),
            PlanParameterContract(
                "collect",
                "bool",
                "Whether a declared stream is collected.",
                required=False,
                has_default=True,
                default=True,
            ),
            PlanParameterContract(
                "declare",
                "bool",
                "Whether to declare an event stream after preparation.",
                required=False,
                has_default=True,
                default=True,
            ),
        ),
        device_contracts=("planner.protocols.LiveFlyer",),
        resource_requirements=(
            ResourceRequirement("camera"),
            ResourceRequirement("storage"),
        ),
        physical_effects=("detector_prepare",),
        preconditions=(
            "The host stages the flyers and retains camera/storage resources.",
            "Every flyer accepts the supplied trigger_info structure.",
        ),
        completion_condition=(
            "Every flyer is prepared and the requested event stream is declared."
        ),
        failure_condition=(
            "A preparation error may leave earlier flyers prepared; the host reconciles "
            "their state and performs required cleanup."
        ),
    ),
    PlanBlockDescriptor(
        block_id="planner.teardown_live_stream",
        contract_version=1,
        plan_stub=teardown_live_stream,
        summary="Complete live flyers, collect their assets, and unstage them.",
        parameters=(
            PlanParameterContract(
                "detectors",
                "Sequence[LiveFlyer]",
                "Active flyers to complete and drain.",
            ),
            PlanParameterContract(
                "stream_name", "str", "Name of the active event stream."
            ),
            PlanParameterContract(
                "flush_period_s",
                "float",
                "Maximum wait between completion checks and asset collection.",
                units="s",
                required=False,
                has_default=True,
                default=0.5,
            ),
            PlanParameterContract(
                "on_complete",
                "Callable[[LiveFlyer, Any], Iterable[Any]] | None",
                "Optional host callback after each flyer completion request.",
                required=False,
                has_default=True,
            ),
        ),
        device_contracts=("planner.protocols.LiveFlyer",),
        resource_requirements=(
            ResourceRequirement("camera"),
            ResourceRequirement("storage"),
        ),
        physical_effects=("acquisition_complete", "stream_asset_collect"),
        preconditions=(
            "Flyers are staged, prepared, and active in the declared stream.",
            "The host retains camera/storage resources through final asset collection.",
        ),
        completion_condition=(
            "Every flyer completes, all available stream assets are collected, and flyers are unstaged."
        ),
        failure_condition=(
            "Completion, collection, or callback errors may leave partial assets or staged devices; "
            "the host reconciles outcome and cleanup."
        ),
    ),
    PlanBlockDescriptor(
        block_id="planner.collect_while_waiting",
        contract_version=1,
        plan_stub=collect_while_waiting_plan,
        summary="Collect streaming assets while a wrapped Bluesky plan sleeps.",
        parameters=(
            PlanParameterContract("plan", "Iterable[BlueskyMsg]", "Plan to wrap."),
            PlanParameterContract(
                "detectors", "Sequence[LiveFlyer]", "Active flyers to collect."
            ),
            PlanParameterContract(
                "stream_name", "str", "Name of the active event stream."
            ),
            PlanParameterContract(
                "flush_period_s",
                "float",
                "Accumulated requested-sleep interval between collection operations.",
                units="s",
                required=False,
                has_default=True,
                default=0.5,
            ),
            PlanParameterContract(
                "on_flush",
                "Callable[[LiveFlyer], Iterable[Any]] | None",
                "Optional host callback after each asset collection.",
                required=False,
                has_default=True,
            ),
        ),
        device_contracts=("planner.protocols.LiveFlyer",),
        resource_requirements=(
            ResourceRequirement("camera"),
            ResourceRequirement("storage"),
        ),
        physical_effects=("stream_asset_collect",),
        preconditions=(
            "Flyers are active and have a declared stream.",
            "The wrapped plan emits sleep messages for periodic collection.",
        ),
        completion_condition=(
            "The wrapped plan completes after assets are collected at configured sleep boundaries."
        ),
        failure_condition=(
            "A wrapped-plan, collection, or host callback error propagates with any prior events retained."
        ),
    ),
    PlanBlockDescriptor(
        block_id="planner.operating_point_schedule",
        contract_version=1,
        plan_stub=operating_point_schedule_plan,
        summary="Lower an accepted operating-point schedule into ordered host callbacks.",
        parameters=(
            PlanParameterContract(
                "request", "OperatingPointRequest", "Validated schedule request."
            ),
            PlanParameterContract(
                "capabilities",
                "OperatingPointCapabilities",
                "Host-supplied capability snapshot used for feasibility checks.",
            ),
            PlanParameterContract(
                "configure_segment",
                "Callable[[OperatingPointSegment], Iterable[Any]]",
                "Host plan to configure the next segment before motion.",
            ),
            PlanParameterContract(
                "acquire_segment",
                "Callable[[OperatingPointSegment], Iterable[Any]]",
                "Host plan that acquires the configured segment.",
            ),
            PlanParameterContract(
                "md",
                "Mapping[str, Any] | None",
                "Additional run metadata; planner-owned keys are reserved.",
                required=False,
                has_default=True,
            ),
        ),
        device_contracts=(),
        resource_requirements=(),
        physical_effects=("host_defined_segment_effects",),
        preconditions=(
            "The schedule passes planner feasibility and measurement assessment.",
            "The host validates callback capabilities, limits, resource ownership, and cancellation policy.",
        ),
        completion_condition=(
            "All segments are configured and acquired in order and the run closes."
        ),
        failure_condition=(
            "A host callback failure can follow partial physical actions; the host reconciles the last "
            "segment outcome before retry or continuation."
        ),
    ),
    PlanBlockDescriptor(
        block_id="planner.survey_scan",
        contract_version=1,
        plan_stub=survey_scan_plan,
        summary="Acquire the overlapping first-round survey tiles.",
        parameters=(
            PlanParameterContract(
                "recipe", "SearchImagingRecipe", "Two-round search-and-image recipe."
            ),
            PlanParameterContract(
                "stage_xy",
                "ReadableMovable | None",
                "Combined XY stage binding, if used.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stage_x",
                "Movable | None",
                "Separate X stage axis, when stage_xy is not used.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stage_y",
                "Movable | None",
                "Separate Y stage axis, when stage_xy is not used.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract("camera", "Any", "Bluesky camera detector."),
            PlanParameterContract(
                "readables",
                "Sequence[Readable]",
                "Additional stage readback signals.",
                required=False,
                has_default=True,
                default=(),
            ),
            PlanParameterContract(
                "prepare_camera",
                "Callable[[Any], Iterable[Any]] | None",
                "Host adapter plan run before each capture.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "prepare_capture",
                "Callable[[Any, int], Iterable[Any]] | None",
                "Host plan that prepares one external capture for the survey.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "capture_one",
                "Callable[[Any, bool], Iterable[Any]] | None",
                "Host plan that triggers or reads each tile image.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "md",
                "Mapping[str, Any] | None",
                "Additional run metadata; planner-owned keys are reserved.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stream_name",
                "str",
                "Bluesky survey event stream name.",
                required=False,
                has_default=True,
                default=DEFAULT_SURVEY_STREAM,
            ),
        ),
        device_contracts=(
            "Bluesky Readable/Movable stage",
            "Bluesky Triggerable/Readable camera",
        ),
        resource_requirements=(
            ResourceRequirement("stage"),
            ResourceRequirement("camera"),
            ResourceRequirement("storage"),
        ),
        physical_effects=("stage_motion", "image_acquisition"),
        preconditions=(
            "The host preflights the complete survey footprint and camera mode.",
            "The host owns detection/review between survey and focused revisit rounds.",
        ),
        completion_condition=(
            "All overlapping survey tiles are captured and the stage remains at the final survey position."
        ),
        failure_condition=(
            "An interrupted capture may leave partial survey data and the stage at an intermediate position."
        ),
    ),
    PlanBlockDescriptor(
        block_id="planner.focused_revisit",
        contract_version=1,
        plan_stub=focused_revisit_plan,
        summary="Acquire host-selected targets in the second-round focused revisit.",
        parameters=(
            PlanParameterContract(
                "recipe", "SearchImagingRecipe", "Two-round search-and-image recipe."
            ),
            PlanParameterContract(
                "targets",
                "Iterable[FocusedTarget]",
                "Targets selected by host detection and operator review.",
            ),
            PlanParameterContract(
                "stage_xy",
                "ReadableMovable | None",
                "Combined XY stage binding, if used.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stage_x",
                "Movable | None",
                "Separate X stage axis, when stage_xy is not used.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stage_y",
                "Movable | None",
                "Separate Y stage axis, when stage_xy is not used.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract("camera", "Any", "Bluesky camera detector."),
            PlanParameterContract(
                "readables",
                "Sequence[Readable]",
                "Additional stage readback signals.",
                required=False,
                has_default=True,
                default=(),
            ),
            PlanParameterContract(
                "prepare_camera",
                "Callable[[Any], Iterable[Any]] | None",
                "Host adapter plan run before each capture.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "prepare_capture",
                "Callable[[Any, int], Iterable[Any]] | None",
                "Host plan that prepares one external capture for the revisit.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "capture_one",
                "Callable[[Any, bool], Iterable[Any]] | None",
                "Host plan that triggers or reads each target image.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "md",
                "Mapping[str, Any] | None",
                "Additional run metadata; planner-owned keys are reserved.",
                required=False,
                has_default=True,
            ),
            PlanParameterContract(
                "stream_name",
                "str",
                "Bluesky focused-revisit event stream name.",
                required=False,
                has_default=True,
                default=DEFAULT_FOCUSED_STREAM,
            ),
        ),
        device_contracts=(
            "Bluesky Readable/Movable stage",
            "Bluesky Triggerable/Readable camera",
        ),
        resource_requirements=(
            ResourceRequirement("stage"),
            ResourceRequirement("camera"),
            ResourceRequirement("storage"),
        ),
        physical_effects=("stage_motion", "image_acquisition"),
        preconditions=(
            "The host validates target identity, calibration freshness, and limits before execution.",
            "Targets have completed detection/review; this plan performs neither operation.",
        ),
        completion_condition=(
            "Every supplied target is visited and captured and the run closes."
        ),
        failure_condition=(
            "An interrupted move/capture may leave the stage at an intermediate target; the host "
            "reconciles position and partial data."
        ),
    ),
)


def _validate_plan_catalog() -> None:
    """Reject duplicate identifiers or duplicate callable registrations."""
    block_ids = [block.block_id for block in PLAN_BLOCKS]
    if len(block_ids) != len(set(block_ids)):
        raise ValueError("plan-block identifiers must be unique")
    plan_ids = [id(block.plan_stub) for block in PLAN_BLOCKS]
    if len(plan_ids) != len(set(plan_ids)):
        raise ValueError("each plan stub must have exactly one catalog descriptor")


_validate_plan_catalog()


def get_plan_block(block_id: str) -> PlanBlockDescriptor:
    """Return a registered plan-block descriptor by its stable identifier.

    Raises
    ------
    KeyError
        If the identifier is not in the current package catalog.
    """
    for block in PLAN_BLOCKS:
        if block.block_id == block_id:
            return block
    raise KeyError(f"unknown planner plan block: {block_id}")


__all__ = [
    "PLAN_BLOCKS",
    "PlanBlockDescriptor",
    "PlanParameterContract",
    "get_plan_block",
]
