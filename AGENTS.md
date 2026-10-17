# redsun-planner agent & contributor conventions

Single source of conventions for agents and contributors of this package:
cross-link, don't duplicate.

`redsun-planner` is a standalone, Apache-2.0 distribution. The import name is
`planner`; the distribution name is `redsun-planner`; the repository is
`https://github.com/deanziyangyu/redsun-planner`. `planner` is the only
importable name.

## Repository layout

```text
redsun-planner/
|-- src/planner/
|   |-- __init__.py            public surface: re-exports with an explicit __all__
|   |-- contracts.py           dataclasses and enums shared by every workflow
|   |-- geometry.py            calibration, regions, tiles, target deduplication
|   |-- detection.py           bright-object detection configuration and helpers
|   |-- scanning.py            grid recipes, ScanPosition, StageScanRecipe
|   |-- bounded_scanning.py    operator-confirmed boundary snap and broom plans
|   |-- plans.py               Bluesky plan generators (snap, broom, hand-eye, replay)
|   |-- protocols.py           LiveFlyer and XYMotor device contracts
|   |-- replay.py              ReplayFrame and the replay source adapter
|   |-- records.py             run/replay/checkpoint save and load
|   |-- review.py              operator review metadata and offline HTML page
|   |-- hand_eye.py            hand-eye fit and quality gates
|   |-- motion_state.py        stage motion estimation
|   |-- speed_policy.py        intensity-to-speed calculation
|   |-- intensity_events.py    intensity-rise detection
|   |-- roi_decision.py        ROI event to motion decision record
|   |-- braking.py             ROI braking assessment
|   `-- observed_coverage.py   observed-support coverage fit
|-- tests/                     hardware-free tests, pytest
|-- examples/                  runnable, device-free snippets
|-- scripts/                   validate_review.py
|-- docs/                      usage notes, Bluesky plan reference, JSON schemas
|-- changelog.d/               towncrier entry files, one per change
`-- pyproject.toml             dependencies and all tool config: pytest, ruff, mypy, coverage
```

## Scope

- This package is backend-independent. It computes recipes, geometry, records and
  plan messages; it creates no RunEngine and owns no device bindings.
- **It must import no acquisition host, framework or vendor adapter.** No
  `ix83_acquisition`, `pycro-manager`, `rtaspt`, `napari` or `redsun` import, at
  runtime or under `TYPE_CHECKING`. `tests/test_library_boundary.py` enforces
  this by importing the package under a meta-path hook that rejects those names.
- Do not add a device call, a hardware connection or a RunEngine here. Hosts
  compose these plans; they own execution policy and physical commands.

## Build & validate

```bash
uv sync --group dev                      # base dependencies and all tooling
uv run ruff format --check .             # formatting
uv run ruff check .                      # lint (D, I, TC, FA102, E402, PLC0415)
uv run mypy                              # strict, files = "."
uv run pytest -q                         # hardware-free tests
uv run prek run --all-files              # the commit hooks, ruff included
```

There are no optional extras: the package depends on `numpy` and `bluesky` only.

The package targets Python `>=3.10`, because host applications still run older
interpreters. Do not use `datetime.UTC`, `StrEnum`, `typing.Self`, `tomllib`,
`except*` or other 3.11+ constructs, even though the tooling runs on 3.12.

## Code conventions

- Python `>=3.10`, `from __future__ import annotations` everywhere (ruff `FA102`).
- Ruff lint has `D` (numpy docstring convention), `I`, `TC`, `FA102`, `E402` and
  `PLC0415` enabled. Public classes, methods and functions need docstrings;
  `D100`/`D104` are ignored.
- **All imports at the top of the module**, with no function-level import
  (`PLC0415`). This package has no optional extras, so there is no
  optional-dependency exception: image processing is not here.
- Runtime-unneeded imports go under `if TYPE_CHECKING:` (`TC001`/`TC002`/`TC003`).
- Keep the public surface in `src/planner/__init__.py`: re-export explicitly and
  add every public symbol to `__all__`.
- Dataclasses validate in `__post_init__` and reject bad input with
  `ValueError`; do not silently coerce.
- Docstrings are numpydoc: a one-line summary, a blank line, then only the
  sections that add something. Do not restate the signature or the types.
- Units: write the short engineering-unit spelling a host reads, `"um"`, `"px"`,
  `"s"`, `"ms"`, never a symbol or a spelled-out word, and expose it through a
  descriptor `units` entry where the value is numeric or an array. `egu` is the
  name of a device declaration or plan parameter in RedSun, not a descriptor
  key; this package declares neither devices nor plan parameters, so `egu`
  appears nowhere here. Where a unit genuinely cannot be known, an empty string
  is correct and preferable to guessing.

## Record and identifier stability

Persisted documents carry a schema identifier, and these are **frozen on-disk
contracts**:

| Document | Schema identifier |
|---|---|
| Workflow run / review checkpoint | `planner-run-v3` |
| Detection replay | `planner-replay-v3` |

- A reader accepts exactly the identifier the writer emits, so a change to one is
  a format change: bump the version rather than widening the accepted set.
  `load_review_run` and `load_replay` verify a payload checksum over the document
  as stored, so the identifier is part of the checksum.
- `deduplicate_targets` derives `target_id` from a `uuid5` over the `planner:`
  namespace string plus specimen, calibration and coordinates. Changing that
  prefix changes every generated identifier, so treat it as a data contract too.

## Testing conventions

- Every test is hardware-free and deterministic. No device, no Micro-Manager, no
  network.
- Keep `tests/test_library_boundary.py` green; it is the guard that this package
  stays extractable.

## Changelog entries

Every user-visible change adds a towncrier entry file to `changelog.d/`, named
`<pr>.<type>.md` with `<type>` one of `breaking`, `added`, `changed`,
`deprecated`, `removed`, `fixed`, `security`. A release collects them into
`CHANGELOG.md`. A change that breaks existing code also gets a `breaking` entry.

## Gate status

`ruff format --check`, `ruff check` and strict `mypy` all pass, and the test
suite passes. Keep them green. The type check was worth running: it found real
defects, not only missing annotations, in `replay.py` (an unreachable
`isinstance` chain and a wrongly annotated local), `motion_state.py` (unchecked
`int()` on parsed JSON and `tuple[float, ...]` where a 2-tuple was declared) and
`hand_eye.py` (optional RANSAC metrics used without narrowing, and matrices
rebuilt as ragged tuples).

`uv.lock` is not committed yet, so `prek.toml` and the CI workflows resolve the
groups on demand instead of passing `--locked`. Generate the lock and add
`--locked` to both once network access allows a full resolve.

## Updating this guide

Say **"Update AGENTS.md with..."** to persist a convention here. Durable,
shareable rules belong in this file, not in per-session memory.
