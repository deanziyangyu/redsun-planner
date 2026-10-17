# redsun-planner

Backend-independent, modular acquisition orchestration built on `bluesky` plans.
It provides sample-search recipes, geometry and scan calculations, review
metadata, and hardware-free run records for any acquisition host.

The import name is `planner`; the distribution name is `redsun-planner`.

This package generates plans and the calculations behind them. It creates no
execution context, owns no device bindings, and performs no stitching or other
image processing: a host composes these plans with its own devices, execution
policy and storage. That separation is deliberate, and `planner` imports no
acquisition framework or vendor adapter at runtime or under `TYPE_CHECKING`.

Requires Python 3.10 or newer. The only runtime dependencies are `numpy` and
`bluesky`; there are no optional extras.

## Install

```bash
pip install redsun-planner
```

From a checkout, with the development tooling:

```bash
uv sync --group dev
```

## Quick start

```python
from planner import AcquisitionSettings, build_grid_recipe

recipe = build_grid_recipe(
    origin_xy_um=(0.0, 0.0),
    specimen_id="specimen-1",
    rows=2,
    columns=3,
    step_xy_um=(25.0, 25.0),
    settings=AcquisitionSettings(exposure_ms=50, frames=3),
)
print(len(recipe.positions))  # 6
```

A recipe descriptor is inert data. The plan generators turn a geometry plus a
host-supplied camera or flyer into Bluesky plan messages; the host supplies the
devices and executes the result.

## What the package provides

| Area | Modules |
|---|---|
| Plan generators | `planner.plans` |
| Recipes and grid geometry | `planner.scanning`, `planner.geometry`, `planner.bounded_scanning` |
| Shared contracts and enums | `planner.contracts` |
| Records, replay and checkpoints | `planner.records`, `planner.replay` |
| Operator review metadata | `planner.review` |
| Hand-eye fit and quality gates | `planner.hand_eye` |
| Calculations | `planner.detection`, `planner.observed_coverage`, `planner.motion_state`, `planner.speed_policy`, `planner.intensity_events`, `planner.roi_decision`, `planner.braking` |
| Structural device contracts | `planner.protocols` |

The public surface is re-exported from `planner/__init__.py` with an explicit
`__all__`; import from `planner` where the name is available there, and from the
module otherwise.

## Using it from another host

The recipes, geometry, detection, hand-eye fit and record helpers are plain
`numpy` calculations over typed data, and never touch a device, an execution
engine or the filesystem. A host that drives something else, for example
Micro-Manager through Pycro-Manager, can consume those directly: build the recipe
or the plan geometry, drive your own stage and camera, place the detections with
a `Calibration`, then persist the run with `planner.records`.

The plan generators are Bluesky objects, so they are the part that expects a
Bluesky execution context. To use them from another host, implement the small
structural protocols in `planner.protocols` on your own devices and let a Bluesky
`RunEngine` run the plan; you do not have to adopt Bluesky elsewhere in the host.

`bluesky` is a hard dependency of the distribution and `import planner` loads it,
because this package re-exports the plan generators. A host that only uses the
calculations still has it installed, though it never calls it.

Runnable, device-free snippets: [examples/](examples/README.md).

## Plans

[`planner.plans`](docs/bluesky_plans.md) composes Bluesky plan stubs for four
workflows: snake-grid snap scanning, flyer-based push-broom scanning, bounded
stage-to-camera hand-eye calibration, and source-only dataset replay. Replay
preserves source frame/time and scalar source context while making no device
calls.

## Persisted formats

Serialized documents carry a schema identifier, and those identifiers are frozen
on-disk contracts:

| Document | Schema identifier |
|---|---|
| Workflow run / review checkpoint | `planner-run-v3` |
| Detection replay | `planner-replay-v3` |

A reader accepts exactly the value the writer emits, so a document written by a
different schema version is rejected rather than reinterpreted. The
[ROI motion-decision schema](docs/schemas/roi_motion_decision.schema.json)
defines that v1 record contract.

## Documentation

- [Examples](examples/README.md)
- [Bluesky plan surface](docs/bluesky_plans.md)
- [Observed coverage handoff](docs/observed_coverage.md)
- [Presenter plan-widget example](docs/examples/redsun_presenter_plan_widget.md)
- [Offline review page demo](docs/review_demo.html)

## Development

```bash
uv sync --group dev
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest -q
uv run prek run --all-files     # the commit hooks, ruff included
```

Every test is hardware-free and deterministic. See [AGENTS.md](AGENTS.md) for the
conventions this package follows.

## License

Apache-2.0. See [LICENSE](LICENSE).
