"""Hardware-free serialization, checkpoint validation and decision replay."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from collections.abc import Callable

    from .review import ReviewManifest

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
from .geometry import Calibration, Region, deduplicate_targets
from .review import (
    ReviewContext,
    ReviewDecision,
    apply_review,
    create_manifest,
    decision_from_json,
    render_review_html,
)

# Persisted document schema identifiers. These are frozen on-disk tags, not
# importable module names: a reader accepts exactly the value the writer emits.
RUN_SCHEMA = "planner-run-v3"
REPLAY_SCHEMA = "planner-replay-v3"


def save_replay(path: str | Path, observations: Sequence[SearchObservation]) -> None:
    """Save typed observations/detections for decision replay (images stay referenced)."""
    records = [_observation_record(obs) for obs in observations]
    Path(path).write_text(
        json.dumps({"schema": REPLAY_SCHEMA, "observations": records}, indent=2),
        encoding="utf-8",
    )


def _observation_record(obs: SearchObservation) -> dict[str, Any]:
    objects = obs.value if isinstance(obs.value, (tuple, list)) else ()
    record = {
        "specimen_id": obs.specimen_id,
        "phase": obs.phase.value,
        "coordinate_frame": obs.coordinate_frame,
        "calibration_id": obs.calibration_id,
        "acquired_at": obs.acquired_at.isoformat(),
        "published_at": obs.published_at.isoformat(),
        "stage_xy_um": list(obs.stage_xy_um),
        "tile_id": obs.tile_id,
        "request_id": obs.request_id,
        "run_id": obs.run_id,
        "axes": list(obs.axes),
        "missing": obs.missing,
        "target_id": obs.target_id,
        "data_reference": to_record_value(obs.data_reference),
        "channel_id": obs.channel_id,
        "exposure_ms": obs.exposure_ms,
        "image_shape_yx": obs.image_shape_yx,
        "dataset_frame_index": obs.dataset_frame_index,
        "detections": [
            {
                "x_pixel": float(d.x_pixel),
                "y_pixel": float(d.y_pixel),
                "score": d.score,
                "bounds_xyxy_pixel": d.bounds_xyxy_pixel,
                "touches_edge": d.touches_edge,
            }
            for d in objects
        ],
    }
    return record


def to_record_value(value: Any) -> Any:
    """Convert a typed value into a JSON-compatible structure."""
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return to_record_value(asdict(cast("Any", value)))
    if isinstance(value, Mapping):
        return {str(k): to_record_value(v) for k, v in value.items()}
    if isinstance(value, (set, frozenset)):
        return [to_record_value(v) for v in sorted(value, key=repr)]
    if isinstance(value, (tuple, list)):
        return [to_record_value(v) for v in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return repr(value)


def save_run(
    path: str | Path,
    run: WorkflowRun,
    recipe: WorkflowRecipe,
    calibration: Calibration,
    region: Region,
    full_calibration: Calibration | None = None,
) -> None:
    """Write a resolved post-run snapshot with references, decisions, and ledger."""
    payload = {
        "schema": RUN_SCHEMA,
        "run": {
            "run_id": run.run_id,
            "recipe_id": run.recipe_id,
            "recipe_version": run.recipe_version,
            "specimen_id": run.specimen_id,
            "status": run.status.value,
            "phase": run.phase.value,
            "started_at": to_record_value(run.started_at),
            "finished_at": to_record_value(run.finished_at),
            "review_resumed_at": to_record_value(run.review_resumed_at),
            "errors": run.errors,
        },
        "recipe": to_record_value(recipe),
        "calibration": to_record_value(calibration),
        "full_calibration": to_record_value(full_calibration or calibration),
        "region": to_record_value(region),
        "execution_bindings": [
            to_record_value(binding) for binding in run.execution_bindings
        ],
        "actions": [to_record_value(action) for action in run.actions],
        "observations": [_observation_record(obs) for obs in run.observations],
        "targets": [to_record_value(target) for target in run.targets],
        "review": {
            "binding_sha256": run.review_binding_sha256,
            "manifest_sha256": run.review_manifest_sha256,
            "fixed_z_um": run.review_fixed_z_um,
            "decision": to_record_value(run.review_decision),
            "manual_provenance": to_record_value(run.manual_provenance),
        },
    }
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    payload["payload_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    Path(path).write_text(
        json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8"
    )


def load_review_run(
    path: str | Path,
) -> tuple[WorkflowRun, WorkflowRecipe, Calibration, Region, Calibration]:
    """Load and verify an offline review checkpoint; this never opens a backend."""

    def reject_constant(value: str) -> None:
        raise ValueError(f"invalid JSON numeric constant: {value}")

    def reject_duplicate_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    payload = json.loads(
        Path(path).read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicate_pairs,
    )
    schema = payload.get("schema")
    if schema != RUN_SCHEMA:
        raise ValueError("unsupported saved run schema")
    digest = payload.pop("payload_sha256", None)
    canonical = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    if (
        not isinstance(digest, str)
        or hashlib.sha256(canonical.encode("utf-8")).hexdigest() != digest
    ):
        raise ValueError("saved run checksum mismatch")
    if payload.get("run", {}).get("status") != RunStatus.REVIEW_PENDING.value:
        raise ValueError("saved run is not review pending")
    recipe_data = payload["recipe"]
    recipe = _recipe_from_record(recipe_data)
    calibration = _calibration_from_record(payload["calibration"])
    full_calibration = _calibration_from_record(
        payload.get("full_calibration", payload["calibration"])
    )
    region = Region(**payload["region"])
    observations = _observations_from_records(payload["observations"])
    targets = []
    obs_by_request = {o.request_id: o for o in observations}
    for item in payload["targets"]:
        target_obs = tuple(
            obs_by_request[o["request_id"]]
            for o in item.get("observations", [])
            if o.get("request_id") in obs_by_request
        )
        targets.append(
            Target(
                item["target_id"],
                item["specimen_id"],
                item["coordinate_frame"],
                item["calibration_id"],
                item["x_um"],
                item["y_um"],
                target_obs,
                item.get("score"),
                tuple(item["bounds_xyxy_um"]) if item.get("bounds_xyxy_um") else None,
            )
        )
    run_data = payload["run"]
    actions = [
        ActionRecord(
            a["request_id"],
            a["action"],
            ActionStatus(a["status"]),
            Phase(a["phase"]),
            datetime.fromisoformat(a["started_at"]),
            datetime.fromisoformat(a["finished_at"]) if a.get("finished_at") else None,
            a.get("parameters", {}),
            a.get("readback"),
            a.get("error"),
        )
        for a in payload.get("actions", [])
    ]
    review = payload.get("review", {})
    run = WorkflowRun(
        run_id=run_data["run_id"],
        recipe_id=run_data["recipe_id"],
        recipe_version=run_data["recipe_version"],
        specimen_id=run_data["specimen_id"],
        status=RunStatus(run_data["status"]),
        phase=Phase(run_data["phase"]),
        started_at=datetime.fromisoformat(run_data["started_at"])
        if run_data.get("started_at")
        else None,
        finished_at=datetime.fromisoformat(run_data["finished_at"])
        if run_data.get("finished_at")
        else None,
        review_resumed_at=datetime.fromisoformat(run_data["review_resumed_at"])
        if run_data.get("review_resumed_at")
        else None,
        actions=actions,
        observations=list(observations),
        targets=targets,
        errors=list(run_data.get("errors", [])),
        review_binding_sha256=review.get("binding_sha256"),
        review_manifest_sha256=review.get("manifest_sha256"),
        review_fixed_z_um=review.get("fixed_z_um"),
        review_decision=review.get("decision"),
        manual_provenance=tuple(review.get("manual_provenance", ())),
        execution_bindings=tuple(
            _binding_from_record(item) for item in payload.get("execution_bindings", [])
        ),
    )
    if not run.review_binding_sha256 or not run.review_manifest_sha256:
        raise ValueError("saved review checkpoint is missing binding metadata")
    if any(
        o.run_id != run.run_id or o.specimen_id != run.specimen_id for o in observations
    ):
        raise ValueError("saved observation context does not match run")
    context = ReviewContext(
        run.run_id,
        run.specimen_id,
        calibration.coordinate_frame,
        calibration.calibration_id,
    )
    if create_manifest(context, targets).sha256 != run.review_manifest_sha256:
        raise ValueError("saved review manifest does not match checkpoint targets")
    return run, recipe, calibration, region, full_calibration


def render_review_checkpoint(
    checkpoint_path: str | Path,
    output_html_path: str | Path,
    image_loader: Callable[[SearchObservation], Any],
    *,
    include_region: bool = True,
) -> ReviewManifest:
    """Render a verified review-pending run as a self-contained review page.

    The caller provides the storage adapter that resolves each saved survey
    observation to an image array. The planner validates the checkpoint,
    targets and calibration context, then writes the existing accept/reject
    review page. This helper never opens an acquisition backend.
    """
    run, _recipe, calibration, region, _full_calibration = load_review_run(
        checkpoint_path
    )
    context = ReviewContext(
        run.run_id,
        run.specimen_id,
        calibration.coordinate_frame,
        calibration.calibration_id,
    )
    return render_review_html(
        output_html_path,
        context,
        run.targets,
        run.observations,
        image_loader,
        calibration,
        region=region if include_region else None,
    )


def apply_review_checkpoint(
    checkpoint_path: str | Path,
    decision_path: str | Path,
) -> tuple[WorkflowRun, ReviewDecision, tuple[Target, ...]]:
    """Validate an exported decision and return the reviewed target sequence.

    The checkpoint is read-only. A host can use the returned targets to build
    its focused-revisit recipe, then persist a resumed run under its own
    execution lifecycle.
    """
    run, recipe, calibration, region, _full_calibration = load_review_run(
        checkpoint_path
    )
    decision = decision_from_json(decision_path)
    context = ReviewContext(
        run.run_id,
        run.specimen_id,
        calibration.coordinate_frame,
        calibration.calibration_id,
    )
    max_targets = recipe.max_targets
    if max_targets is None:
        max_targets = len(run.targets) + len(decision.additions)
    result = apply_review(
        decision,
        run.targets,
        context=context,
        region=region,
        max_targets=max_targets,
    )
    return run, decision, result.targets


def _calibration_from_record(item: Mapping[str, Any]) -> Calibration:
    matrix = cast(
        "tuple[tuple[float, float], tuple[float, float]]",
        tuple(tuple(v for v in row) for row in item["pixel_to_stage"]),
    )
    return Calibration(
        item["calibration_id"],
        item["coordinate_frame"],
        matrix,
        tuple(item["reference_pixel"]),
        tuple(item["camera_stage_xy_um"]),
        tuple(item.get("roi_origin", (0, 0))),
        tuple(item.get("binning", (1, 1))),
    )


def _binding_from_record(item: Mapping[str, Any]) -> ExecutionBindingSnapshot:
    return ExecutionBindingSnapshot(
        provider=str(item["provider"]),
        capabilities=frozenset(str(value) for value in item["capabilities"]),
        resources=frozenset(str(value) for value in item["resources"]),
        settings=frozenset(str(value) for value in item.get("settings", [])),
        reference=str(item["reference"]) if item.get("reference") is not None else None,
    )


def _recipe_from_record(item: Mapping[str, Any]) -> WorkflowRecipe:
    def settings(x: Mapping[str, Any]) -> AcquisitionSettings:
        return AcquisitionSettings(
            x["exposure_ms"],
            x["frames"],
            x.get("channel_id"),
            tuple(x["image_shape_yx"]) if x.get("image_shape_yx") else None,
            tuple(x["roi_origin"]) if x.get("roi_origin") else None,
            tuple(x["binning"]) if x.get("binning") else None,
            x.get("metadata", {}),
        )

    return WorkflowRecipe(
        item["recipe_id"],
        item["version"],
        item["specimen_id"],
        item["coordinate_frame"],
        item["calibration_id"],
        settings(item["survey_settings"]),
        settings(item["full_settings"]),
        tuple(item["image_shape_yx"]),
        item.get("overlap", 0.2),
        item.get("deduplication_um", 3),
        item.get("max_targets"),
        frozenset(item.get("required_capabilities", [])),
        tuple(
            ResourceRequirement(**r)
            for r in item.get(
                "resources",
                [{"name": "stage"}, {"name": "camera"}, {"name": "storage"}],
            )
        ),
        tuple(item.get("preconditions", [])),
        tuple(item.get("physical_effects", [])),
        item.get("completion_condition", ""),
        item.get("failure_condition", ""),
        item.get("parameters", {}),
        tuple(item.get("validation_evidence", [])),
    )


def _observations_from_records(
    items: Sequence[Mapping[str, Any]],
) -> tuple[SearchObservation, ...]:
    records = []
    for item in items:
        detections = tuple(
            DetectedObject(
                d["x_pixel"],
                d["y_pixel"],
                d.get("score"),
                tuple(d["bounds_xyxy_pixel"]) if d.get("bounds_xyxy_pixel") else None,
                bool(d.get("touches_edge", False)),
            )
            for d in item.get("detections", [])
        )
        records.append(
            SearchObservation(
                item["specimen_id"],
                Phase(item["phase"]),
                item["coordinate_frame"],
                item["calibration_id"],
                datetime.fromisoformat(item["acquired_at"]),
                datetime.fromisoformat(item["published_at"]),
                tuple(item["stage_xy_um"]),
                item["tile_id"],
                item["request_id"],
                run_id=item["run_id"],
                value=detections if detections else None,
                axes=tuple(axis.upper() for axis in item["axes"]),
                missing=item["missing"],
                target_id=item.get("target_id"),
                data_reference=_dataset_reference_from_record(
                    item.get("data_reference")
                ),
                channel_id=item.get("channel_id"),
                exposure_ms=item.get("exposure_ms"),
                image_shape_yx=tuple(item["image_shape_yx"])
                if item.get("image_shape_yx")
                else None,
                dataset_frame_index=item.get("dataset_frame_index"),
            )
        )
    return tuple(records)


def load_replay(path: str | Path) -> tuple[SearchObservation, ...]:
    """Load replay-safe detection records without opening hardware or raw data."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != REPLAY_SCHEMA:
        raise ValueError("unsupported replay schema")
    records = []
    for item in payload["observations"]:
        detections = tuple(
            DetectedObject(
                d["x_pixel"],
                d["y_pixel"],
                d.get("score"),
                tuple(d["bounds_xyxy_pixel"]) if d.get("bounds_xyxy_pixel") else None,
                bool(d.get("touches_edge", False)),
            )
            for d in item["detections"]
        )
        records.append(
            SearchObservation(
                item["specimen_id"],
                Phase(item["phase"]),
                item["coordinate_frame"],
                item["calibration_id"],
                datetime.fromisoformat(item["acquired_at"]),
                datetime.fromisoformat(item["published_at"]),
                tuple(item["stage_xy_um"]),
                item["tile_id"],
                item["request_id"],
                run_id=item["run_id"],
                value=detections,
                axes=tuple(axis.upper() for axis in item["axes"]),
                missing=item["missing"],
                target_id=item.get("target_id"),
                data_reference=_dataset_reference_from_record(
                    item.get("data_reference")
                ),
                channel_id=item.get("channel_id"),
                exposure_ms=item.get("exposure_ms"),
                image_shape_yx=tuple(item["image_shape_yx"])
                if item.get("image_shape_yx")
                else None,
                dataset_frame_index=item.get("dataset_frame_index"),
            )
        )
    return tuple(records)


def _dataset_reference_from_record(value: Any) -> DatasetReference | str | None:
    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, Mapping):
        raise TypeError("data_reference must be a typed object, URI string or null")
    return DatasetReference(
        uri=str(value["uri"]),
        format=str(value["format"]),
        axes=str(value["axes"]),
        shape=tuple(int(size) for size in value["shape"]),
        dtype=str(value["dtype"]),
        frames_written=int(value["frames_written"]),
        terminal_status=str(value["terminal_status"]),
        manifest_sha256=str(value["manifest_sha256"]),
    )


def replay_targets(
    observations: Sequence[SearchObservation],
    calibration: Calibration,
    region: Region,
    *,
    distance_um: float,
    max_targets: int | None = None,
) -> tuple[Target, ...]:
    """Repeat deterministic deduplication and target limiting from saved detections."""
    detections = [obs for obs in observations if obs.phase is Phase.DETECTION]
    targets = list(
        deduplicate_targets(
            detections, calibration, distance_um=distance_um, region=region
        )
    )
    targets.sort(
        key=lambda t: (
            -(t.score if t.score is not None else float("-inf")),
            t.y_um,
            t.x_um,
        )
    )
    return tuple(targets if max_targets is None else targets[:max_targets])
