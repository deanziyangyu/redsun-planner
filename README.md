# redsun-planner

Backend-independent, modular acquisition orchestration built on `bluesky` plans.
It provides sample-search recipes, geometry and scan calculations, review
metadata, and hardware-free run records for any acquisition host.

The import name is `planner`; the distribution name is `redsun-planner`. It 
implements the acquisition optimization framework [**SPARTA**](./docs/sparta.md): Spatiotemporal
 Pareto Acquisition-Regime Trajectory Allocation. The name is recorded in
`planner.sparta`; the public API keeps the plain `AcquisitionPoint`,
`AcquisitionEnvelope` and `OperatingTrace` names.

This package generates plans and the calculations behind them. It creates no
execution context, owns no device bindings, and performs no stitching or other
image processing: a host composes these plans with its own devices, execution
policy and storage. That separation is deliberate, and `planner` imports no
acquisition framework or vendor adapter at runtime or under `TYPE_CHECKING`.

Requires Python 3.10 or newer. The only runtime dependencies are `numpy` and
`bluesky`; there are no optional extras.

## Install

The bundle builds this distribution from its registered `sample_search`
submodule through each consumer's lock. The source pin observed on 2026-10-18
is `dce582bf8347c92b560d10d9c545d0ecbdbe8aa0`; local modifications are separate
from that commit. A package-index release is not established by that source
pin. To install a wheel from an explicit checkout, use the intended Python
environment's installer with that checkout path.

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

## Command line

The distribution installs a `planner` console script; `python -m planner` is
equivalent. The commands are hardware-free and read the same JSON specification:

```bash
planner simulate spec.json              # compact human report
planner simulate spec.json --json       # the report JSON instead
planner simulate spec.json --output report.json
planner simulate spec.json --html throughput.html
planner validate spec.json              # ok/issues, exit 1 when invalid
planner diff before.json after.json     # changed fields and metric deltas
planner compare                        # adaptive broom vs dual-scan figures
planner compare --html comparison.html  # graphical throughput page
planner compare --envelope envelope.json  # shade declared operating bands
```

Exit codes are stable: `0` success or valid, `1` a semantically invalid
specification, `2` unreadable, malformed or unknown-key input. Every command
accepts `--json` for its machine-readable form.

The specification requires `field_size_um`, `survey`, `focused`, `stage` and
`targets`; `origin_xy_um`, `review_s`, `bleach_time_constant_s`, `envelope` and `limits` are optional.
Unknown keys are rejected at every level.

```json
{
  "field_size_um": [4000.0, 3000.0],
  "survey": {
    "tile_size_um": [900.0, 700.0], "overlap": 0.1, "exposure_s": 0.01,
    "frames": 1, "readout_s": 0.005, "overhead_s": 0.05,
    "pixel_size_um": 0.65, "resolution_um": 1.0
  },
  "focused": {
    "exposure_s": 0.2, "frames": 1, "channels": 2, "z_planes": 1,
    "readout_s": 0.02, "overhead_s": 0.2, "pixel_size_um": 0.108,
    "resolution_um": 0.25, "roi_size_um": [64.0, 64.0]
  },
  "stage": {
    "max_velocity_um_s": 5000.0, "acceleration_um_s2": 20000.0, "settle_s": 0.05
  },
  "targets": {"count": 50, "seed": 7, "margin_um": 0.0},
  "review_s": 30.0,
  "envelope": {
    "velocity_um_s": [1.0, 50.0], "exposure_s": [0.005, 0.05],
    "width_um": [50.0, 150.0]
  },
  "limits": {"max_blur_ratio": 2.0, "max_gamma": 1.0}
}
```

`targets` may instead be `{"positions_xy_um": [[x, y], ...]}`, which yields ids
`target-000`, `target-001`, ... . See the
[two-round workflow](docs/search_imaging.md) for the model and the CLI contract.

## What the package provides

| Area | Modules |
|---|---|
| Plan generators | `planner.plans`, `planner.search_imaging` |
| Recipes and grid geometry | `planner.scanning`, `planner.geometry`, `planner.bounded_scanning` |
| Two-round search-and-image workflow | `planner.search_imaging` |
| Command line | `planner.cli`, `python -m planner` |
| Throughput HTML reports | `planner.report_html` |
| Shared contracts and enums | `planner.contracts` |
| Records, replay and checkpoints | `planner.records`, `planner.replay` |
| Operator review metadata | `planner.review` |
| Hand-eye fit and quality gates | `planner.hand_eye` |
| Calculations | `planner.acquisition_space`, `planner.simulation`, `planner.broom`, `planner.detection`, `planner.observed_coverage`, `planner.motion_state`, `planner.speed_policy`, `planner.intensity_events`, `planner.roi_decision`, `planner.braking` |
| SPARTA framework | `planner.sparta`, `planner.operating_map` |
| Structural device contracts | `planner.protocols` |

The public surface is re-exported from `planner/__init__.py` with an explicit
`__all__`; import from `planner` where the name is available there, and from the
module otherwise.

## Using it from another host

The recipes, geometry, detection and hand-eye fit are plain `numpy`
calculations over typed data and never touch a device or execution engine.
The record/review helpers do read and write files when explicitly called;
they do not acquire data. A host that drives something else, for example
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

[`planner.plans`](src/planner/plans.py) composes Bluesky plan stubs for four
workflows: snake-grid snap scanning, flyer-based push-broom scanning, bounded
stage-to-camera hand-eye calibration, and source-only dataset replay. Replay
preserves source frame/time and scalar source context while making no device
calls.

[`planner.search_imaging`](src/planner/search_imaging.py) adds the two-round
workflow beside those primitives: it builds the survey and focused revisit
recipes and delegates their messages to `snap_scan_plan`. Detection and review
between the rounds remain host-owned; the broom path remains the continuous-scan
alternative.

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
- [Two-round search-and-image workflow](docs/search_imaging.md)
- [SPARTA framework](docs/sparta.md)
- [Acquisition-space model](docs/acquisition_space.md)
- [Search-and-image simulation](docs/simulation.md)
- [Adaptive broom timing](docs/broom.md)
- [Observed coverage handoff](docs/observed_coverage.md)

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

The committed `uv.lock` owns development-tool resolution; consumer locks own
their installed planner wheels. Next, verify each consumer against the intended
planner API and record its source/lock identity, including local modifications.
Keep portable dependency/release selection separate from the existing bundle
source pin; a finalized host contract still needs consumer acceptance.

## License

Apache-2.0. See [LICENSE](LICENSE).
