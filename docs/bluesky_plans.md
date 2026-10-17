# Bluesky scan plans

`planner.plans` is the Bluesky execution surface over pure scan recipes and
saved-data replay. It provides `snap_scan_plan`, `push_broom_plan`,
`hand_eye_calibration_plan`, and `replay_dataset_plan`. Bluesky is a
required planner dependency because these plans compose its plan stubs and
preprocessors. Constructing a plan does not start or require a RunEngine; the
selected execution host owns plan execution. Pure geometry and recipe helpers
remain separable from device orchestration.

## Saved dataset replay

`replay_dataset_plan` streams a `ReplaySource` as ordinary Bluesky events,
keeping one `ReplayFrame` in memory at a time. Each frame carries its source
frame ID, source timestamp, image arrays and scalar source context. The Run
Engine event timestamp is the replay time; it does not replace the source time.
Run metadata marks the run as simulation-only and records that no hardware
commands were issued. Planner-owned metadata cannot be overridden by caller
metadata.

The source protocol is file-format agnostic. A TIFF/Zarr reader belongs in the
consuming application (or a format-specific example), where axis selection,
frame selection, sidecars and checksums can be interpreted with that format's
contract. Keep source XY/readbacks separate from virtual positions, simulated
motion and virtual integration parameters. This plan issues no camera or stage
commands and does not itself provide image correction, detection, registration,
Presenter callbacks or persistence. A RedSun integration should bind the
selected source adapter through its Presenter plan map and send replayed image
events through the existing display/storage callbacks without binding live
devices.

The public `planner.LiveFlyer` and `planner.XYMotor` runtime-checkable
protocols describe device parameters used by the RedSun `live_stream` and
`live_median_scan` presenter plans. They keep those plan signatures resolvable
by RedSun's plan widget while leaving action management, document callbacks,
and RedSun writers in the host. They are structural Bluesky-facing contracts;
the planner package does not depend on RedSun.

## Snap scan

`snap_scan_plan` accepts a `StageScanRecipe` or confirmed-boundary
`SnapScanPlan`, one combined XY movable (with `.set((x_um, y_um))` and scalar
XY readbacks) or separate X/Y motor interfaces, a triggerable/readable camera
and optional additional readables. The combined form supports one atomic Core
XY command. For each declared position the plan emits the move followed by a
trigger/read event. Start-document metadata records the
ordered position IDs, recipe/specimen identity, coordinate frame, calibration
identity and declared exposure/settings. Event sequence numbers associate
primary-stream observations with that ordered position list.

The instrument host must validate all commanded points against the currently
loaded device profile and travel limits before submitting the plan. It owns
camera configuration, illumination/shutter policy, resource locks, cancellation
and action-level outcomes. The plan does not infer limits from geometry.

## Stage-to-camera startup calibration

`hand_eye_calibration_plan` accepts generic Bluesky stage axes, camera and
readback objects plus an operator-verified anchor. It performs one blocking
point move to the anchor, records two stationary reference images, and probes
positive and negative X/Y offsets, returning to the anchor and capturing a
closure image after each probe. Event order corresponds to the ordered
`planner_calibration_capture_labels` in run metadata. Every event includes
camera and XY readback values. The plan does not configure the camera, infer
the safe anchor, inspect instrument limits, or perform registration.

`planner.hand_eye.fit_hand_eye` is the pure fit API. It requires bidirectional
readback deltas on both axes, adequate registration scores, and a full-rank
probe matrix. Inputs use direct registered specimen-feature motion. The
default maximum fit residual is 2.5 px. A mode context (camera, objective,
ROI, binning, profile revision, and other caller-selected identity fields) is
included in the deterministic calibration ID and can be checked with
`matches_hand_eye_mode`. It returns the fitted 2×2 content-motion matrix, its
sign-reversed mosaic-placement matrix, residuals, and covariance. Image
registration is not part of this package: the host that owns the camera images
supplies the measured shifts. `evaluate_hand_eye_quality` applies the RANSAC
inlier/residual, ECC agreement, and four-axis return-closure gates to those
registered pairs and closure observations. The host still needs to bind these devices,
preflight the selected profile/envelope, persist raw and derived records, and
enforce current-calibration matching at scan startup. RedSun presenter plan
widget registration is also pending.

The ECC and SIFT/RANSAC estimators and the run-document analyzer live in the
host that has the images and the OpenCV dependency. The analyzer
consumes a successful Bluesky start/events/stop sequence and ordered
image planes, with each event's measured XY position normalized to
`data.stage_position_xy` in micrometres. It requires the host's `hand-eye` extra.
The host subscriber remains responsible for writing NPY frames, JSONL/doc
files, checksums and the terminal manifest, and adapts its device readback field
names to the analyzer input contract before delegating analysis. Other Bluesky
hosts can use the same analyzer with their own persistence adapters.

## Push-broom scan

`push_broom_plan` consumes a bounded `BroomScanPlan` and a Bluesky flyer. The
flyer must support `prepare(strip=...)`, `kickoff`, `complete`, `collect` and
`describe_collect`. Preparing a strip configures its trajectory but must not
start motion. Its collected frame events must include camera data, stable frame
IDs, stage positions, exposure intervals, clock-domain/provenance and
uncertainty. The plan declares its collect stream from the flyer contract,
prepares, kicks off, and uses
[`collect_while_completing`](https://blueskyproject.io/bluesky/v1.14.6/generated/bluesky.plan_stubs.collect_while_completing.html)
to emit flyer events while each strip completes. The adapter must reject the
recipe before motion if any capability or evidence field is unsupported.

The pure [`blend_intensity_to_speed`](../src/planner/speed_policy.py) policy
now converts causal intensity and motion-confidence samples into requested and
sample-to-sample slew-limited speeds. It is not yet wired into a running
push-broom plan. Closed-loop use still requires filtering/hysteresis choices,
measured command latency, acceleration/braking characterization, and an
adapter capability for safe in-motion velocity updates. This API also does not
convert the DemoCamera broom emulation into continuous capture: commanded
step-and-snap remains explicitly an emulation.

## Calibrated ROI-to-motion feasibility

`planner.motion_state` converts bracketed host XY position observations into a
recent local velocity/acceleration estimate with timing, age, fit-residual and
position-uncertainty fields. `StagePositionObservation.from_record` accepts the
`stage_position_sample` JSONL schema emitted by the temporary Ludl diagnostic
host. `planner.assess_calibrated_roi_braking` uses the reusable affine
`Calibration` (pixel scale, rotation, ROI origin and binning), detector/detail
ROI geometry, valid measured motion estimate, assumed physical deceleration,
event latency and an explicit distance reserve to estimate braking margin.
These pure helpers issue no device calls and do not run a feedback controller.
The first synthetic experiment was an offline replay; it does not calibrate the model.
The estimator and braking model remain uncalibrated until new sampled stage
traces, camera-stage affine calibration and ROI-size-specific event latency are
measured. Do not translate Ludl controller-property values into physical
speed/acceleration without a separate calibration.

## Integration and evidence status

A host installs the planner as a non-editable wheel and locks its
Bluesky dependency. With Bluesky
1.15.1 and Ophyd 1.11.2, the planner suite and a host suite pass together.
Actual RunEngine checks execute `snap_scan_plan` and `push_broom_plan` with
Ophyd simulated devices. The snap run emits ordered events with stage X/Y
readbacks, setpoints and timestamps; its event record and plot
show the position fields needed to associate stationary snap observations
with the path. The acquisition also includes a hardware-disconnected
snap binding. Its fake-Core RunEngine test verifies one
bounded atomic XY Core command per target and records image, frame ID, host
capture interval and pre-snap stage coordinates in event data. The push-broom
fake flyer emits all three expected frame events through event pages and a
successful stop. This caught and fixed two RunEngine requirements: flyer data
must be included in the collected-device list, and the collection stream must
be declared before the first collect. These remain software integration
checks only: no RedSun composition, operator-loaded Core, real camera, real
flyer, or physical stage was exercised. The supported API references are Bluesky 1.14.6's
[`prepare`](https://blueskyproject.io/bluesky/v1.14.6/generated/bluesky.plan_stubs.prepare.html),
[`collect_while_completing`](https://blueskyproject.io/bluesky/v1.14.6/generated/bluesky.plan_stubs.collect_while_completing.html)
and [`trigger_and_read`](https://blueskyproject.io/bluesky/v1.14.6/generated/bluesky.plan_stubs.trigger_and_read.html).
