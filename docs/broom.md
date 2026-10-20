# Adaptive broom timing model

`planner.broom` estimates a broom raster whose speed changes around declared
mock samples. It reports the adaptive run beside two regular-broom baselines, so
the cost of the speed modulation is explicit rather than buried in one total.

## Inputs

`BroomSpec` combines a rectangular field, a square field of view, the row pitch,
the requested dwell, a `StageProfile` and the mock samples:

| Field | Meaning |
| --- | --- |
| `field_size_um` | raster extent |
| `fov_um` | square swath; a point stays inside it for `dwell_s` at `fov_um / dwell_s` |
| `row_pitch_um` | raster row spacing |
| `dwell_s` | requested per-sample observation time |
| `stage` | travel speed, acceleration and settle |
| `exposure_s`, `readout_s`, `overhead_s` | camera context, not stage time |
| `samples` | `MockTarget` positions inside the field |
| `profile_step_um` | speed-profile resolution (default 2 um) |
| `command_delay_s` | fixed delay per speed transition, e.g. one camera frame |

The row profile extends half a field of view beyond each field edge so a sample
near the edge can still receive the full dwell. Row ends are rest-to-rest turns.

## Speed profile and baselines

Each row gets a target speed: `fov_um / dwell_s` inside every sample window and
the travel speed elsewhere. A forward/backward pass enforces the acceleration
limit, so the stage decelerates before a sample and accelerates after it. The
sampled speed profile is then integrated to travel, slow, transition and fast
time.

`BroomReport` reports:

| Field | Meaning |
| --- | --- |
| `regular_cruise_s` | the same raster at the travel speed with no slow zones |
| `regular_dwell_s` | the same raster at the dwell speed everywhere (two seconds at every point) |
| `adaptive_s` | the adaptive broom run, including command delay |
| `added_vs_cruise_s` | `adaptive_s - regular_cruise_s`: cost of adding the sample dwell to a regular fast broom |
| `saved_vs_dwell_s` | `regular_dwell_s - adaptive_s`: saving from speeding up between samples |
| `slow_s`, `transition_s`, `fast_s` | row-profile time spent at dwell speed, in ramp transitions and at travel speed |
| `delay_s` | total command delay |
| `dwell_nominal_s` | `samples * dwell_s` |
| `min_dwell_s`, `mean_dwell_s` | integrated time inside each sample's swath |
| `throughput_um2_s`, `samples_per_hour` | field area and samples over `adaptive_s` |

`slow_s` is unique low-speed time; overlapping sample windows can make it
smaller than the sum of the per-sample dwells.

The two regular baselines answer different questions. `regular_dwell_s` is the
equivalent regular broom: it scans at the dwell speed everywhere and meets the
two-second requirement at every point, so the adaptive broom is faster than it
whenever the stage can travel faster than the dwell speed. `regular_cruise_s`
is a fast broom that does not dwell at the samples, so it can be faster than the adaptive broom simply because it under-observes. The `added_vs_cruise_s` figure is the cost of
adding the dwell to that non-equivalent fast broom, not a penalty of
adaptivity; `saved_vs_dwell_s` and the adaptive-versus-broom-2s ratio are the
like-for-like comparison.

## Built-in comparison inputs

`planner compare` provides three preconfigured model scenarios: `measured`,
`blend-2500`, and `blend-5000`. The `measured` preset uses 103.7 um/s travel
speed and 92.7 um/s^2 acceleration with no command delay. The blend presets use
200 um/s travel speed, assumed acceleration caps of 2,500 or 5,000 um/s^2, and
20 ms command delay. These values are declared model inputs, not instrument
configuration or physical acceptance. Replace them with host-appropriate values
before interpreting a comparison for a particular system.
## Command line

```text
planner compare [--samples 10 25 50 100] [--scenario all|measured|blend-2500|blend-5000] [--json] [--output PATH] [--html PATH] [--envelope PATH]
```

For each scenario and sample count it prints the regular-broom cruise and dwell
times, the adaptive broom, added/ramp/delay, the dual snap scan total,
dual-minus-adaptive, speedup, minimum dwell and both area throughputs. The JSON
form carries the same rows; `--html` writes the graphical comparison page.
`--envelope` takes a JSON object with `velocity_um_s`, `exposure_s` and
`width_um` low/high pairs and shades the operating-space bands; it does not
change any timing estimate.

## Limits

- The model is an estimate on declared values; it does not assert a stage
  envelope, controller calibration or camera trigger timing.
- Row turns are conservative rest-to-rest moves; a coordinated turn would be
  faster.
- The blend scenario caps are assumed software slew limits, not measured
  stage acceleration.
- `exposure_s`, `readout_s` and `overhead_s` do not change the stage time; they
  are retained for provenance.
- The model assumes a full-field raster, not a sample-only route.
