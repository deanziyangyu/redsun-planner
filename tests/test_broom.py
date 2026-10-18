from __future__ import annotations

import json

import pytest

from planner.broom import BroomSpec, simulate_broom_scan
from planner.simulation import MockTarget, StageProfile


def _spec(
    samples: tuple[MockTarget, ...],
    *,
    v_max: float = 200.0,
    accel: float = 2500.0,
    delay: float = 0.02,
) -> BroomSpec:
    return BroomSpec(
        field_size_um=(500.0, 500.0),
        fov_um=100.0,
        row_pitch_um=100.0,
        dwell_s=2.0,
        stage=StageProfile(v_max, accel),
        exposure_s=0.02,
        readout_s=0.06,
        samples=samples,
        profile_step_um=2.0,
        command_delay_s=delay,
    )


def test_dynamic_broom_brackets_the_regular_baselines() -> None:
    samples = (
        MockTarget("a", (150.0, 150.0)),
        MockTarget("b", (350.0, 250.0)),
        MockTarget("c", (250.0, 350.0)),
    )
    report = simulate_broom_scan(_spec(samples))

    assert report.regular_cruise_s < report.adaptive_s < report.regular_dwell_s
    assert report.added_vs_cruise_s == pytest.approx(
        report.adaptive_s - report.regular_cruise_s
    )
    assert report.saved_vs_dwell_s == pytest.approx(
        report.regular_dwell_s - report.adaptive_s
    )
    assert report.throughput_um2_s == pytest.approx(250000.0 / report.adaptive_s)
    assert report.samples_per_hour == pytest.approx(3 * 3600.0 / report.adaptive_s)
    assert report.min_dwell_s > 1.9
    assert report.mean_dwell_s >= report.min_dwell_s
    assert report.transition_s > 0
    assert report.delay_s == pytest.approx(3 * 2 * 0.02)


def test_adaptive_speed_modulation_is_reported_separately() -> None:
    samples = (MockTarget("a", (250.0, 250.0)),)
    report = simulate_broom_scan(_spec(samples))

    assert report.slow_s > 0
    assert report.dwell_nominal_s == pytest.approx(2.0)
    # The added time over a regular fast broom is the slow-zone and ramp cost.
    assert report.added_vs_cruise_s > 0
    assert report.added_vs_cruise_s < report.regular_dwell_s - report.regular_cruise_s


def test_no_samples_makes_the_dynamic_run_equal_the_cruise_run() -> None:
    report = simulate_broom_scan(_spec(()))

    assert report.samples == 0
    assert report.adaptive_s == pytest.approx(report.regular_cruise_s)
    assert report.added_vs_cruise_s == pytest.approx(0.0)
    assert report.saved_vs_dwell_s == pytest.approx(
        report.regular_dwell_s - report.regular_cruise_s
    )
    assert report.min_dwell_s == 0.0
    assert report.mean_dwell_s == 0.0
    assert report.samples_per_hour == 0.0


def test_equal_speeds_make_every_baseline_the_same() -> None:
    samples = (MockTarget("a", (250.0, 250.0)),)
    report = simulate_broom_scan(_spec(samples, v_max=50.0, accel=2500.0, delay=0.0))

    assert report.adaptive_s == pytest.approx(report.regular_cruise_s)
    assert report.adaptive_s == pytest.approx(report.regular_dwell_s)
    assert report.added_vs_cruise_s == pytest.approx(0.0)
    assert report.saved_vs_dwell_s == pytest.approx(0.0)


def test_report_serializes_to_json_without_nan_or_infinity() -> None:
    report = simulate_broom_scan(_spec((MockTarget("a", (250.0, 250.0)),)))

    encoded = json.dumps(report.to_dict(), allow_nan=False)
    assert "NaN" not in encoded
    assert "Infinity" not in encoded
    assert report.to_dict()["samples"] == 1


def test_spec_rejects_bad_values_and_duplicate_or_outside_samples() -> None:
    with pytest.raises(ValueError, match="positive"):
        BroomSpec(
            field_size_um=(500.0, 500.0),
            fov_um=0.0,
            row_pitch_um=100.0,
            dwell_s=2.0,
            stage=StageProfile(200.0, 2500.0),
            exposure_s=0.02,
        )
    with pytest.raises(ValueError, match="duplicate"):
        _spec((MockTarget("a", (10.0, 10.0)), MockTarget("a", (20.0, 20.0))))
    with pytest.raises(ValueError, match="outside"):
        _spec((MockTarget("a", (600.0, 10.0)),))


def test_spec_rejects_a_non_tuple_sample_collection() -> None:
    with pytest.raises(TypeError, match="tuple"):
        BroomSpec(
            field_size_um=(500.0, 500.0),
            fov_um=100.0,
            row_pitch_um=100.0,
            dwell_s=2.0,
            stage=StageProfile(200.0, 2500.0),
            exposure_s=0.02,
            samples=[MockTarget("a", (10.0, 10.0))],  # type: ignore[arg-type]
        )
