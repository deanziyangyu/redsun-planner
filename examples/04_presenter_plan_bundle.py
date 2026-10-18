"""Offer planner plans to a Redsun presenter plan bundle, without importing Redsun.

A Redsun component exposes a ``plan_map``: a mapping from a plan name to a
``PlanEntry`` that holds the plan callable and the document callbacks it needs
(``redsun.registry.PlanEntry``). The planner never imports that type. It only
supplies callables that return Bluesky messages, so the host wraps them, names
them and owns the execution policy. This example builds the same mapping with
plain dictionaries.

The replay entry is complete and runs here: it needs no camera and no stage.
The snap entry shows the shape a device-bound host uses. ``snap_scan_plan`` is
a generator function, so constructing it contacts nothing; the host binds real
devices and submits it to its own RunEngine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from bluesky import RunEngine

from planner import (
    ReplayFrame,
    ReplaySource,
    build_grid_recipe,
    replay_dataset_plan,
    snap_scan_plan,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping


class SyntheticReplaySource:
    """Two-frame source standing in for a saved-dataset reader."""

    source_id = "synthetic-presenter-bundle"

    source_metadata: Mapping[str, Any] = {
        "source_reference": "in-memory example",
        "source_time_origin": "frame index",
    }

    def iter_frames(self) -> Iterator[ReplayFrame]:
        """Yield two frames, each with scalar source provenance."""
        for index in (280, 281):
            yield ReplayFrame(
                source_frame_id=str(index),
                source_time_s=float(index),
                images={"image": np.full((4, 5), index, dtype=np.uint16)},
                source_context={
                    "source_stage_x_um": float(index),
                    "source_frame_index": index,
                },
            )


def count_documents(name: str, doc: Any) -> None:
    """Record one emitted document, the way a presenter callback would."""
    counts[name] = counts.get(name, 0) + 1
    if name == "start":
        print("run start metadata:", sorted(doc))


# The bundle a presenter would expose. Each value mirrors a PlanEntry: `plan` is
# required, `callbacks` are the document callbacks the entry needs, and a host
# runs them in order before any callback an operator attaches.
counts: dict[str, int] = {}
source: ReplaySource = SyntheticReplaySource()
recipe = build_grid_recipe(
    (0.0, 0.0),
    specimen_id="specimen",
    rows=2,
    columns=3,
    step_xy_um=(10.0, 10.0),
)
plan_bundle: dict[str, dict[str, Any]] = {
    "replay_dataset": {
        "plan": lambda: replay_dataset_plan(source),
        "callbacks": [count_documents],
    },
    "snap_scan": {
        # A host replaces these names with its own movable and camera objects.
        "plan": lambda: snap_scan_plan(
            recipe, stage_xy="xy_stage", camera="camera", readables=()
        ),
        "callbacks": [count_documents],
    },
}

print("plan bundle entries:", sorted(plan_bundle))
print("snap scan would visit", len(recipe.positions), "positions")

engine = RunEngine({})
engine.subscribe(count_documents)
engine(plan_bundle["replay_dataset"]["plan"]())
print("documents emitted:", dict(sorted(counts.items())))
