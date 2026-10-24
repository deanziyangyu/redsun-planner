from __future__ import annotations

import pytest

from planner.intensity_events import (
    IntensityRiseConfig,
    IntensityRiseDetector,
    detect_intensity_rise,
)


def test_incremental_detector_matches_batch_trace_and_uses_only_prefix() -> None:
    values = [100.0] * 8 + [130.0] * 5 + [100.0] * 4
    config = IntensityRiseConfig(
        baseline_window_frames=5,
        baseline_min_samples=5,
        persistence_frames=2,
        release_persistence_frames=2,
        enter_relative_rise=0.2,
        exit_relative_rise=0.08,
    )
    detector = IntensityRiseDetector(config)
    incremental = [detector.update(value) for value in values]
    batch = detect_intensity_rise(values, config)

    assert tuple(sample.trailing_baseline for sample in incremental) == (
        batch.trailing_baseline
    )
    assert tuple(sample.relative_rise for sample in incremental) == batch.relative_rise
    assert tuple(sample.active for sample in incremental) == batch.active
    assert tuple(sample.onset for sample in incremental) == batch.onset
    assert tuple(sample.release for sample in incremental) == batch.release
    assert sum(batch.onset) == 1
    assert sum(batch.release) == 1

    prefix = detect_intensity_rise(values[:10], config)
    assert prefix.onset == batch.onset[:10]
    assert prefix.active == batch.active[:10]


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, True])
def test_incremental_detector_rejects_invalid_signal(value: float) -> None:
    expected_error = TypeError if isinstance(value, bool) else ValueError
    with pytest.raises(expected_error):
        IntensityRiseDetector().update(value)
    if isinstance(value, bool):
        with pytest.raises(TypeError, match="not bool"):
            detect_intensity_rise([value])
