"""Backend-independent contracts for serial sample-search workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit
from uuid import uuid4

if TYPE_CHECKING:
    from collections.abc import Mapping
    from datetime import datetime


class Phase(str, Enum):
    """Workflow phase a recipe or observation belongs to."""

    SURVEY = "survey"
    DETECTION = "detection"
    SELECTION = "selection"
    REVISIT = "revisit"


class RunStatus(str, Enum):
    """Terminal or in-progress state of one workflow run."""

    RUNNING = "running"
    COMPLETED = "completed"
    EMPTY = "empty"
    CANCELLED = "cancelled"
    FAILED = "failed"
    OUTCOME_UNKNOWN = "outcome_unknown"
    REVIEW_PENDING = "review_pending"


class ActionStatus(str, Enum):
    """Lifecycle state recorded for one proposed action."""

    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    OUTCOME_UNKNOWN = "outcome_unknown"


@dataclass(frozen=True)
class ResourceRequirement:
    """A device resource a recipe or plan needs, with its sharing mode."""

    name: str
    mode: str = "exclusive"

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not self.name:
            raise ValueError("resource name is required")
        if self.mode != "exclusive":
            raise ValueError(
                "the serial prototype supports exclusive resource requirements only"
            )


@dataclass(frozen=True)
class AcquisitionSettings:
    """Camera and channel settings requested for one acquisition."""

    exposure_ms: float = 50.0
    frames: int = 1
    channel_id: str | None = None
    image_shape_yx: tuple[int, int] | None = None
    roi_origin: tuple[float, float] | None = None
    binning: tuple[float, float] | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if (
            isinstance(self.exposure_ms, bool)
            or not isinstance(self.exposure_ms, (int, float))
            or not isfinite(self.exposure_ms)
            or self.exposure_ms <= 0
        ):
            raise ValueError("exposure_ms must be finite and positive")
        if (
            isinstance(self.frames, bool)
            or not isinstance(self.frames, int)
            or self.frames < 1
        ):
            raise ValueError("exposure_ms and frames must be positive")
        if self.image_shape_yx is not None and (
            len(self.image_shape_yx) != 2
            or any(
                isinstance(v, bool) or not isinstance(v, int) or v <= 0
                for v in self.image_shape_yx
            )
        ):
            raise ValueError("image_shape_yx must be a pair of positive integers")
        if self.roi_origin is not None and (
            len(self.roi_origin) != 2
            or any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not isfinite(x)
                for x in self.roi_origin
            )
        ):
            raise ValueError("ROI origin must be finite")
        if self.binning is not None and (
            len(self.binning) != 2
            or min(self.binning) <= 0
            or any(
                isinstance(x, bool)
                or not isinstance(x, (int, float))
                or not isfinite(x)
                for x in self.binning
            )
        ):
            raise ValueError("binning must be finite and positive")


@dataclass(frozen=True)
class DatasetReference:
    """Framework-neutral reference to stored, integrity-described image data."""

    uri: str
    format: str
    axes: str
    shape: tuple[int, ...]
    dtype: str
    frames_written: int
    terminal_status: str
    manifest_sha256: str

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not self.uri or not urlsplit(self.uri).scheme:
            raise ValueError("dataset uri must be an absolute URI with a scheme")
        if not self.format or not self.dtype:
            raise ValueError("dataset format and dtype are required")
        if (
            not self.axes
            or len(set(self.axes)) != len(self.axes)
            or any(not axis.isupper() or not axis.isalpha() for axis in self.axes)
            or not self.axes.endswith("YX")
        ):
            raise ValueError("dataset axes must be unique uppercase names ending in YX")
        if len(self.axes) != len(self.shape) or any(size < 0 for size in self.shape):
            raise ValueError("dataset axes and nonnegative shape dimensions must agree")
        if self.frames_written < 0 or self.terminal_status not in {
            "completed",
            "cancelled",
            "failed",
            "outcome_unknown",
        }:
            raise ValueError("invalid dataset frame count or terminal status")
        if self.frames_written and any(size < 1 for size in self.shape):
            raise ValueError("nonempty datasets require positive shape dimensions")
        if (
            self.axes[0] in "QT"
            and self.frames_written
            and self.shape[0] != self.frames_written
        ):
            raise ValueError(
                "sequence/time axis length must match the verified frame count"
            )
        if len(self.manifest_sha256) != 64 or any(
            ch not in "0123456789abcdef" for ch in self.manifest_sha256
        ):
            raise ValueError("manifest_sha256 must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class WorkflowRecipe:
    """Versioned, executable sample-search workflow descriptor."""

    recipe_id: str
    version: str
    specimen_id: str
    coordinate_frame: str
    calibration_id: str
    survey_settings: AcquisitionSettings
    full_settings: AcquisitionSettings
    image_shape_yx: tuple[int, int]
    overlap: float = 0.2
    deduplication_um: float = 3.0
    max_targets: int | None = None
    required_capabilities: frozenset[str] = frozenset()
    resources: tuple[ResourceRequirement, ...] = (
        ResourceRequirement("stage"),
        ResourceRequirement("camera"),
        ResourceRequirement("storage"),
    )
    preconditions: tuple[str, ...] = ()
    physical_effects: tuple[str, ...] = ("stage_motion", "image_acquisition")
    completion_condition: str = "all selected revisit acquisitions complete"
    failure_condition: str = "any failed or outcome-unknown device action"
    parameters: Mapping[str, Any] = field(default_factory=dict)
    validation_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not isinstance(self.survey_settings, AcquisitionSettings) or not isinstance(
            self.full_settings, AcquisitionSettings
        ):
            raise TypeError(
                "survey_settings and full_settings must be AcquisitionSettings"
            )
        if len(self.image_shape_yx) != 2 or any(
            isinstance(v, bool) or not isinstance(v, int) or v <= 0
            for v in self.image_shape_yx
        ):
            raise ValueError("image_shape_yx must be a pair of positive integers")
        if not isfinite(self.overlap) or not 0 <= self.overlap < 1:
            raise ValueError("overlap must be finite and in [0, 1)")
        if not isfinite(self.deduplication_um) or self.deduplication_um < 0:
            raise ValueError("deduplication_um must be finite and nonnegative")
        if self.max_targets is not None and (
            isinstance(self.max_targets, bool)
            or not isinstance(self.max_targets, int)
            or self.max_targets < 0
        ):
            raise ValueError("max_targets must be a nonnegative integer")
        if (
            not self.resources
            or any(not isinstance(r, ResourceRequirement) for r in self.resources)
            or len({r.name for r in self.resources}) != len(self.resources)
        ):
            raise ValueError("recipe resources must be unique typed requirements")


@dataclass(frozen=True)
class SearchObservation:
    """One recorded survey or revisit observation with its target context."""

    specimen_id: str
    phase: Phase
    coordinate_frame: str
    calibration_id: str
    acquired_at: datetime
    published_at: datetime
    stage_xy_um: tuple[float, float]
    tile_id: str
    request_id: str
    run_id: str = ""
    value: Any = None
    axes: tuple[str, ...] = ("Y", "X")
    missing: bool = False
    target_id: str | None = None
    data_reference: DatasetReference | str | None = None
    channel_id: str | None = None
    exposure_ms: float | None = None
    image_shape_yx: tuple[int, int] | None = None

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if (
            len(set(self.axes)) != len(self.axes)
            or not self.axes
            or any(not axis.isupper() or not axis.isalpha() for axis in self.axes)
        ):
            raise ValueError("observation axes must be unique uppercase names")
        if self.axes.count("X") != 1 or self.axes.count("Y") != 1:
            raise ValueError("observation axes must include X and Y exactly once")
        if self.image_shape_yx is not None and (
            len(self.image_shape_yx) != 2
            or any(size < 1 for size in self.image_shape_yx)
        ):
            raise ValueError("image_shape_yx must contain two positive dimensions")


@dataclass(frozen=True)
class Target:
    """A selected specimen target at a calibrated stage position."""

    target_id: str
    specimen_id: str
    coordinate_frame: str
    calibration_id: str
    x_um: float
    y_um: float
    observations: tuple[SearchObservation, ...] = ()
    score: float | None = None
    bounds_xyxy_um: tuple[float, float, float, float] | None = None


@dataclass(frozen=True)
class DetectedObject:
    """One detected object inside a survey image, in pixel coordinates."""

    x_pixel: float
    y_pixel: float
    score: float | None = None
    bounds_xyxy_pixel: tuple[float, float, float, float] | None = None
    touches_edge: bool = False


@dataclass(frozen=True)
class ActionRecord:
    """Journal entry for one proposed, accepted, started or finished action."""

    request_id: str
    action: str
    status: ActionStatus
    phase: Phase
    started_at: datetime
    finished_at: datetime | None = None
    parameters: Mapping[str, Any] = field(default_factory=dict)
    readback: Any = None
    error: str | None = None


@dataclass(frozen=True)
class ExecutionBindingSnapshot:
    """Framework-neutral adapter declaration; it does not identify physical hardware."""

    provider: str
    capabilities: frozenset[str]
    resources: frozenset[str]
    settings: frozenset[str]
    reference: str | None = None

    def __post_init__(self) -> None:
        """Check the declared fields for consistency."""
        if not self.provider:
            raise ValueError("execution binding provider is required")
        if not self.capabilities or not self.resources:
            raise ValueError(
                "execution binding must declare capabilities and resources"
            )
        if any(
            not item for item in (*self.capabilities, *self.resources, *self.settings)
        ):
            raise ValueError("execution binding declarations must be nonempty strings")
        if self.reference is not None and not self.reference:
            raise ValueError(
                "execution binding reference must be nonempty when supplied"
            )


@dataclass
class WorkflowRun:
    """Resolved record of one workflow run, its actions and its outcomes."""

    run_id: str = field(default_factory=lambda: str(uuid4()))
    recipe_id: str = ""
    recipe_version: str = ""
    specimen_id: str = ""
    status: RunStatus = RunStatus.FAILED
    phase: Phase = Phase.SURVEY
    started_at: datetime | None = None
    finished_at: datetime | None = None
    actions: list[ActionRecord] = field(default_factory=list)
    observations: list[SearchObservation] = field(default_factory=list)
    targets: list[Target] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    review_binding_sha256: str | None = None
    review_manifest_sha256: str | None = None
    review_fixed_z_um: float | None = None
    review_decision: Mapping[str, Any] | None = None
    manual_provenance: tuple[Mapping[str, Any], ...] = ()
    review_resumed_at: datetime | None = None
    execution_bindings: tuple[ExecutionBindingSnapshot, ...] = ()


def new_request_id() -> str:
    """Return a fresh request identifier for one proposed action."""
    return str(uuid4())
