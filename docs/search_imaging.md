# Two-round search-and-image workflow

`planner.search_imaging` is the recipe and plan layer for a staged
search-and-image run:

1. **Survey** — round one covers a declared specimen field with overlapping
   serpentine snap tiles. The tiles are the "preview": overlapping frames a
   host can stitch into a mosaic.
2. **Host-owned detection and review** — a host decides which candidates in the
   survey are worth imaging. This package does not detect, score, deduplicate,
   register, stitch or select.
3. **Focused revisit** — round two visits the host-supplied `FocusedTarget`
   records and acquires the focused settings, one position per target.

The workflow is a composition, not a new scan primitive. It builds ordinary
`StageScanRecipe` records and delegates Bluesky messages to
`planner.plans.snap_scan_plan`; a host still owns the devices, the RunEngine,
preflight, resource arbitration and execution policy. Round one may run at the
same magnification as round two. The two rounds carry independent pixel size and
resolution, so a lower-magnification overview is one supported choice, never an
assumption.

## Why it sits beside the broom plans

`planner.plans` already offers two acquisition primitives: `snap_scan_plan` for
step-and-snap grids and `push_broom_plan` for continuous strips.
`planner.search_imaging` is the two-round workflow on top of the snap primitive:
it reuses the simulator's tile grid, produces the survey and focused recipes,
adds the stitching context and estimates throughput. It does not replace or
wrap the broom path, which remains the right choice for continuous strip
acquisition. Keeping the workflow in its own module leaves detection, review and
stitching outside the package, where the hardware boundary already puts them.

## The host-owned boundary

The package boundary is deliberate and mirrors the rest of `redsun-planner`:

| Step | Owner |
|---|---|
| Field, tile size, overlap, exposure, stage profile | recipe declaration |
| Survey tile grid, overscan, stitching preview | `planner.search_imaging` |
| Survey and focused Bluesky plan messages | `planner.search_imaging` / `planner.plans` |
| Image capture and stage motion | acquisition host |
| Detection, scoring, candidate selection, review | acquisition host |
| Registration, mosaicking, stitching | acquisition host |
| Focused target records | acquisition host, passed in as `FocusedTarget` |

No detection code, image processing, device import or RunEngine call exists in
this package. `search_imaging_records` and the two plan generators accept the
targets the host already selected; `estimate_search_imaging` treats them as
declared mock positions, not detections.

## AutoScanJ provenance

The staged structure follows [AutoScanJ](https://doi.org/10.3389/fbinf.2021.627626)
(Tosi _et al._, *Frontiers in Bioinformatics*, 2021), a suite of ImageJ and
Micro-Manager scripts that scans a specimen, finds targets of interest, and then
drives the microscope to those locations for a second, higher-quality
acquisition. The reusable shape is: primary survey, review, secondary focused
acquisition.

AutoScanJ commonly pairs a lower-magnification overview with a higher-resolution
focused pass. This implementation does not require that: the survey may use the
same pixel size and resolution as the focused round. Only the field of view, the
tile overlap and the stitching context decide whether round one is a useful
preview.

## Public records

| Record | Purpose |
|---|---|
| `SearchImagingRecipe` | Field, survey and focused settings, stage profile, review time, channel ids, optional tile image shape |
| `FocusedTarget` | Host-selected target id, center xy um, optional ROI size |
| `StitchingPreview` | Tile rows/columns, overlap, physical step, overscan, optional expected mosaic shape in pixels |
| `SearchImagingRecords` | The survey recipe, the focused recipe and the stitching preview |

`SearchImagingRecipe.survey_tiles()`, `survey_positions()` and `overscan_um()`
expose the same overlapping serpentine grid as `planner.simulation`, including
the centered single tile when a tile is larger than the field. `survey_tiles`,
`survey_positions`, `stitching_preview` and `survey_overscan_um` are equivalent
module-level functions.

The survey `AcquisitionSettings` carries exposure in milliseconds, frames, the
first declared channel id and the tile image shape. Its metadata records the
overlap, step, grid shape, overscan and the configured snap operating point.
The focused settings multiply frames, channels and z planes into one capture
count per target so a single trigger per position still accounts for every
focused frame. Detection and candidate ordering remain host-owned; targets are
visited in the order supplied. The simulator applies the recipe's focused ROI
size to every target for timing; a per-target ROI is carried into the recipe
metadata but does not change the estimate.

## Plan generators

| Generator | Phase | Run stream |
|---|---|---|
| `survey_scan_plan(recipe, ...)` | `Phase.SURVEY` | `survey` |
| `focused_revisit_plan(recipe, targets, ...)` | `Phase.REVISIT` | `focused` |

Both accept the same device and callback arguments as
`planner.plans.snap_scan_plan` (`stage_xy` or `stage_x`/`stage_y`, `camera`,
`readables`, `prepare_camera`, `prepare_capture`, `capture_one`, `md`,
`stream_name`) and emit one Bluesky run per round. Round metadata includes the
workflow id, recipe id, round name and index, the stream name and, for the
focused round, the target ids. The survey run also records the grid and corner
context.

```python
from planner import (
    FocusedTarget,
    SearchImagingRecipe,
    SurveySpec,
    FocusedSpec,
    StageProfile,
    survey_scan_plan,
    focused_revisit_plan,
)

recipe = SearchImagingRecipe(
    recipe_id="plate-1",
    field_size_um=(4000.0, 3000.0),
    survey=SurveySpec((900.0, 700.0), 0.1, 0.01, 1, 0.005, 0.05, 0.108, 0.25),
    focused=FocusedSpec(0.2, 1, 2, 1, 0.02, 0.2, 0.108, 0.25, (64.0, 64.0)),
    stage=StageProfile(5000.0, 20000.0, 0.05),
    review_s=30.0,
)
targets = (FocusedTarget("target-000", (1000.0, 800.0)),)

# The host runs these with its own stage, camera and RunEngine.
survey_plan = survey_scan_plan(recipe, camera=camera, stage_xy=stage)
focused_plan = focused_revisit_plan(recipe, targets, camera=camera, stage_xy=stage)
```

## Model corners

`estimate_search_imaging` converts the recipe and targets into
`planner.simulation.SearchImagingSpec` and calls `simulate_search_imaging`. It
reports three operating points and, when an envelope is supplied, their
acquisition-space corners:

| Point | Meaning | Typical corner |
|---|---|---|
| `survey_acquisition` | The configured snap tile, stationary | `001` |
| `survey_equivalent` | The continuous point whose ideal area throughput equals the simulated survey throughput | `101` |
| `focused_point` | The stationary focused acquisition | `010` or `011` |

The corners are relative to the caller's declared envelope, not universal
thresholds; without an envelope the corners are `None`. With declared
`MeasurementLimits`, the two acquisition points receive feasibility verdicts
with typed reasons (blur, gap, too few observations, duty). The equivalent
survey point intentionally has no verdict: its motion-blur terms are a
modelling convenience, not a claim about a snap exposure.

The survey and focused corner values are also stored in the recipe metadata
when an envelope is passed to `build_survey_recipe` or `build_focused_recipe`.
The simulation report remains authoritative for timing and verdicts because it
knows the simulated per-position period.

The report also carries per-sub-scan metrics, the averaged view and the overview
photobleaching record. Each round reports its area, duration, throughput,
seconds per position and travel/exposure fractions (`ScanMetrics`), plus the
arithmetic means, combined acquired area and combined throughput across the two
rounds (`AveragedMetrics`). A focused round with no targets is excluded from the
means. `photobleaching` reports the prior overview exposure, its cost relative
to the focused exposure, and the retained brightness when a time constant is
declared. The CLI prints the sub-scan, averaged and photobleaching blocks, and
the JSON report exposes them as `sub_scans`, `averaged` and `photobleaching`.

## Command line

`planner simulate`, `planner validate` and `planner diff` read the JSON
specification documented in the [README](../README.md#command-line). They are
hardware-free and expose stable exit codes: `0` for success or valid specs, `1`
for a semantically invalid spec and `2` for unreadable, malformed or unknown-key
input. `validate` checks the field, tile and target geometry, frame counts,
overscan, envelope classification and any declared limits verdicts. An operating
point that falls between envelope bands has no corner; that is reported as a
warning, not a validation failure.

## Offline review checkpoint

Hosts that save a `planner-run-v3` review-pending checkpoint can render its
validated candidates into the package's self-contained accept/reject page with
`render_review_checkpoint`. The host supplies an image loader for its storage
format; the planner verifies the checkpoint checksum, candidate manifest,
calibration context and survey observations before writing the page. The page
exports a `ReviewDecision` JSON file and cannot access or command hardware.

```python
from planner import render_review_checkpoint

manifest = render_review_checkpoint(
    "review-pending.json",
    "target-review.html",
    image_loader=load_saved_survey_image,
)
print(manifest.sha256)
```

The image loader must resolve each observation to the matching saved survey
image. `SearchObservation.dataset_frame_index` carries an explicit zero-based
index into the dataset named by `data_reference`; hosts must populate and
validate this association rather than infer it from Zarr, TIFF, RTASPT or
application-specific URI conventions. To validate an exported decision and
obtain the ordered focused targets, hosts can call:

```python
from planner import apply_review_checkpoint

run, decision, targets = apply_review_checkpoint(
    "review-pending.json",
    "review-decision.json",
)
focused_targets = targets
```

The checkpoint remains unchanged. The host owns its run-state transition and
execution record when it starts focused acquisition.

## Limits of the workflow

- No detection, tracking, focus search, registration or stitching is modelled
  or implemented.
- Target positions and review time are declared; review editing and false
  negatives are outside the estimate.
- No stage envelope is asserted. A host must preflight field, overscan, tile
  count and target positions against its own limits.
- The nearest-neighbour focused order in the simulation is an estimate; the plan
  generators visit targets in the supplied order.
