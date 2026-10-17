"""Pure recipe definitions, calculations and records for acquisition hosts."""

from .braking import (
    CalibratedRoiBrakingAssessment,
    RoiBrakingAssessment,
    assess_calibrated_roi_braking,
    assess_roi_braking,
)
from .contracts import (
    AcquisitionSettings,
    ActionRecord,
    ActionStatus,
    DatasetReference,
    DetectedObject,
    ExecutionBindingSnapshot,
    Phase,
    ResourceRequirement,
    RunStatus,
    SearchObservation,
    Target,
    WorkflowRecipe,
    WorkflowRun,
)
from .geometry import Calibration, Region, Tile, deduplicate_targets, plan_tiles
from .hand_eye import (
    CAPTURE_LABELS,
    HandEyeClosure,
    HandEyeFit,
    HandEyePair,
    HandEyeQualityCriteria,
    HandEyeQualityResult,
    evaluate_hand_eye_quality,
    fit_hand_eye,
    matches_hand_eye_mode,
)
from .intensity_events import (
    IntensityEventTrace,
    IntensityRiseConfig,
    detect_intensity_rise,
)
from .motion_state import (
    StageMotionEstimate,
    StagePositionObservation,
    estimate_stage_motion,
)
from .observed_coverage import (
    ObservedCoverage,
    ObservedSnapGridFit,
    ObservedSnapTile,
    fit_snap_grid_to_coverage,
)
from .plans import (
    hand_eye_calibration_plan,
    push_broom_plan,
    replay_dataset_plan,
    snap_scan_plan,
)
from .protocols import LiveFlyer, XYMotor
from .records import load_replay, load_review_run, replay_targets, save_replay, save_run
from .replay import ReplayFrame, ReplaySource
from .review import (
    ManualTarget,
    ReviewContext,
    ReviewDecision,
    ReviewManifest,
    ReviewResult,
)
from .roi_decision import (
    RoiIntensityEvent,
    RoiMotionDecisionRecord,
    assess_roi_event_transition,
)
from .scanning import ScanPosition, StageScanRecipe, build_grid_recipe, grid_positions
from .speed_policy import SpeedBlendConfig, SpeedCurve, blend_intensity_to_speed

__all__ = [
    "CAPTURE_LABELS",
    "AcquisitionSettings",
    "ActionRecord",
    "ActionStatus",
    "CalibratedRoiBrakingAssessment",
    "Calibration",
    "DatasetReference",
    "DetectedObject",
    "ExecutionBindingSnapshot",
    "HandEyeClosure",
    "HandEyeFit",
    "HandEyePair",
    "HandEyeQualityCriteria",
    "HandEyeQualityResult",
    "IntensityEventTrace",
    "IntensityRiseConfig",
    "LiveFlyer",
    "ManualTarget",
    "ObservedCoverage",
    "ObservedSnapGridFit",
    "ObservedSnapTile",
    "Phase",
    "Region",
    "ReplayFrame",
    "ReplaySource",
    "ResourceRequirement",
    "ReviewContext",
    "ReviewDecision",
    "ReviewManifest",
    "ReviewResult",
    "RoiBrakingAssessment",
    "RoiIntensityEvent",
    "RoiMotionDecisionRecord",
    "RunStatus",
    "ScanPosition",
    "SearchObservation",
    "SpeedBlendConfig",
    "SpeedCurve",
    "StageMotionEstimate",
    "StagePositionObservation",
    "StageScanRecipe",
    "Target",
    "Tile",
    "WorkflowRecipe",
    "WorkflowRun",
    "XYMotor",
    "assess_calibrated_roi_braking",
    "assess_roi_braking",
    "assess_roi_event_transition",
    "blend_intensity_to_speed",
    "build_grid_recipe",
    "deduplicate_targets",
    "detect_intensity_rise",
    "estimate_stage_motion",
    "evaluate_hand_eye_quality",
    "fit_hand_eye",
    "fit_snap_grid_to_coverage",
    "grid_positions",
    "hand_eye_calibration_plan",
    "load_replay",
    "load_review_run",
    "matches_hand_eye_mode",
    "plan_tiles",
    "push_broom_plan",
    "replay_dataset_plan",
    "replay_targets",
    "save_replay",
    "save_run",
    "snap_scan_plan",
]
