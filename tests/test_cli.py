from __future__ import annotations

import copy
import json
from typing import TYPE_CHECKING, Any

import pytest

from planner.cli import main

if TYPE_CHECKING:
    from pathlib import Path


def _base_spec() -> dict[str, Any]:
    return {
        "field_size_um": [400.0, 200.0],
        "survey": {
            "tile_size_um": [200.0, 100.0],
            "overlap": 0.1,
            "exposure_s": 0.01,
            "frames": 1,
            "readout_s": 0.005,
            "overhead_s": 0.02,
            "pixel_size_um": 0.1,
            "resolution_um": 0.25,
        },
        "focused": {
            "exposure_s": 0.2,
            "frames": 1,
            "channels": 1,
            "z_planes": 1,
            "readout_s": 0.02,
            "overhead_s": 0.1,
            "pixel_size_um": 0.1,
            "resolution_um": 0.25,
            "roi_size_um": [64.0, 64.0],
        },
        "stage": {
            "max_velocity_um_s": 1000.0,
            "acceleration_um_s2": 10000.0,
            "settle_s": 0.01,
        },
        "targets": {"count": 3, "seed": 7, "margin_um": 0.0},
        "review_s": 2.0,
    }


def _envelope_spec(spec: dict[str, Any]) -> dict[str, Any]:
    spec["envelope"] = {
        "velocity_um_s": [1.0, 100.0],
        "exposure_s": [0.05, 0.1],
        "width_um": [50.0, 60.0],
    }
    spec["limits"] = {"max_blur_ratio": 2.0, "max_gamma": 1.0}
    return spec


def _write(tmp_path: Path, name: str, payload: Any) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_simulate_prints_the_compact_human_report(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, "spec.json", _base_spec())

    assert main(["simulate", str(path)]) == 0

    output = capsys.readouterr().out
    assert "field" in output
    assert "survey" in output
    assert "survey time" in output
    assert "focused time" in output
    assert "total" in output
    assert "throughput" in output
    assert "targets/hour" in output
    assert "operating points" in output
    assert "corner=" in output
    assert "sub-scan metrics" in output
    assert "photobleaching" in output
    assert "averaged metrics" in output
    assert "combined acquired" in output


def test_simulate_json_includes_sub_scans_and_averages(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, "spec.json", _base_spec())

    assert main(["simulate", str(path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    survey = payload["sub_scans"]["survey"]
    focused = payload["sub_scans"]["focused"]
    averaged = payload["averaged"]
    assert survey["scan_id"] == "survey"
    assert survey["area_um2"] == payload["field_area_um2"]
    assert focused["scan_id"] == "focused"
    assert focused["positions"] == payload["targets"]
    assert focused["area_um2"] == 3 * 64.0 * 64.0
    assert averaged["mean_throughput_um2_s"] == pytest.approx(
        (survey["throughput_um2_s"] + focused["throughput_um2_s"]) / 2
    )
    assert averaged["combined_acquired_area_um2"] == pytest.approx(
        payload["field_area_um2"] + focused["area_um2"]
    )
    assert averaged["total_duration_s"] == payload["total_s"]
    assert payload["photobleaching"]["cost_mean"] >= 0.0


def test_simulate_json_prints_the_report_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, "spec.json", _envelope_spec(_base_spec()))

    assert main(["simulate", str(path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["survey"]["positions"] > 0
    assert payload["total_s"] > 0
    assert payload["survey_acquisition_corner"] == "001"
    assert payload["survey_equivalent_corner"] == "101"
    assert payload["focused_corner"] == "011"
    assert payload["survey_acquisition_verdict"]["feasible"] is True


def test_simulate_output_writes_report_json_and_keeps_the_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, "spec.json", _base_spec())
    output_path = tmp_path / "report.json"

    assert main(["simulate", str(path), "--output", str(output_path)]) == 0

    stdout = capsys.readouterr().out
    written = json.loads(output_path.read_text(encoding="utf-8"))
    assert "field" in stdout
    assert written["survey"]["positions"] > 0
    assert written["total_s"] > 0


def test_validate_prints_ok_for_a_valid_spec(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, "spec.json", _envelope_spec(_base_spec()))

    assert main(["validate", str(path)]) == 0

    output = capsys.readouterr().out
    assert output.splitlines()[0] == "ok"
    assert "tiles" in output
    assert "targets 3" in output
    assert "overscan" in output


def test_validate_json_reports_checks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, "spec.json", _envelope_spec(_base_spec()))

    assert main(["validate", str(path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["issues"] == []
    assert payload["targets"] == 3
    assert payload["corners"]["focused"] == "011"
    assert payload["metrics"]["tile_count"] > 0


def test_validate_reports_a_negative_exposure_as_invalid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _base_spec()
    spec["survey"]["exposure_s"] = -1.0
    path = _write(tmp_path, "spec.json", spec)

    assert main(["validate", str(path)]) == 1

    output = capsys.readouterr().out
    assert output.splitlines()[0] == "invalid"
    assert "exposure_s" in output


def test_validate_reports_infeasible_limits(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _base_spec()
    spec["limits"] = {"max_blur_ratio": 0.5, "max_gamma": 1.0}
    path = _write(tmp_path, "spec.json", spec)

    assert main(["validate", str(path)]) == 1

    output = capsys.readouterr().out
    assert "issue:" in output
    assert "fails limits" in output
    assert "blur" in output


def test_validate_warns_about_an_unclassified_envelope_point(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _base_spec()
    spec["envelope"] = {
        "velocity_um_s": [1.0, 100.0],
        "exposure_s": [0.05, 0.1],
        "width_um": [50.0, 150.0],
    }
    path = _write(tmp_path, "spec.json", spec)

    assert main(["validate", str(path)]) == 0

    output = capsys.readouterr().out
    assert output.splitlines()[0] == "ok"
    assert "warning:" in output
    assert "falls between envelope bands" in output


def test_validate_warns_about_survey_overscan(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _base_spec()
    spec["field_size_um"] = [390.0, 195.0]
    path = _write(tmp_path, "spec.json", spec)

    assert main(["validate", str(path)]) == 0

    output = capsys.readouterr().out
    assert output.splitlines()[0] == "ok"
    assert "warning:" in output
    assert "overscan" in output


def test_malformed_json_exits_two_with_a_clear_message(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")

    assert main(["validate", str(path)]) == 2

    error = capsys.readouterr().err
    assert "invalid JSON" in error


def test_unknown_keys_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _base_spec()
    spec["unexpected"] = 1
    path = _write(tmp_path, "spec.json", spec)

    assert main(["simulate", str(path)]) == 2

    assert "unknown key" in capsys.readouterr().err


def test_unknown_nested_keys_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _base_spec()
    spec["survey"]["mystery"] = 1
    path = _write(tmp_path, "spec.json", spec)

    assert main(["validate", str(path)]) == 2

    assert "survey" in capsys.readouterr().err


def test_missing_required_sections_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _base_spec()
    del spec["focused"]
    path = _write(tmp_path, "spec.json", spec)

    assert main(["simulate", str(path)]) == 2

    assert "focused" in capsys.readouterr().err


def test_wrong_json_value_types_exit_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _base_spec()
    spec["survey"]["frames"] = "one"
    path = _write(tmp_path, "spec.json", spec)

    assert main(["validate", str(path)]) == 2

    assert "must be an integer" in capsys.readouterr().err


def test_json_errors_are_machine_readable(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _base_spec()
    spec["unexpected"] = 1
    path = _write(tmp_path, "spec.json", spec)

    assert main(["validate", str(path), "--json"]) == 2

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "unknown key" in payload["error"]


def test_targets_may_be_given_as_positions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _base_spec()
    spec["targets"] = {"positions_xy_um": [[50.0, 50.0], [150.0, 50.0]]}
    path = _write(tmp_path, "spec.json", spec)

    assert main(["simulate", str(path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["targets"] == 2
    assert payload["focused_order_ids"] == ["target-001", "target-000"]


def test_diff_reports_changed_fields_and_metric_deltas(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = _base_spec()
    right = copy.deepcopy(left)
    right["focused"]["exposure_s"] = 0.4
    left_path = _write(tmp_path, "left.json", left)
    right_path = _write(tmp_path, "right.json", right)

    assert main(["diff", str(left_path), str(right_path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["equal"] is False
    paths = [change["path"] for change in payload["changed_fields"]]
    assert "focused.exposure_s" in paths
    metrics = payload["metrics"]
    assert metrics["survey_time_s"]["delta"] == pytest.approx(0.0)
    assert metrics["focused_time_s"]["delta"] > 0.0
    assert metrics["total_time_s"]["delta"] == pytest.approx(
        metrics["focused_time_s"]["delta"]
    )
    assert metrics["targets_per_hour"]["delta"] < 0.0


def test_diff_human_output_is_compact(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    left = _base_spec()
    right = copy.deepcopy(left)
    right["survey"]["tile_size_um"] = [100.0, 50.0]
    left_path = _write(tmp_path, "left.json", left)
    right_path = _write(tmp_path, "right.json", right)

    assert main(["diff", str(left_path), str(right_path)]) == 0

    output = capsys.readouterr().out
    assert "changed fields" in output
    assert "survey.tile_size_um" in output
    assert "metrics" in output
    assert "tile count:" in output
    assert "survey time s:" in output
    assert "corners" in output


def test_diff_of_identical_specs_is_equal(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = _write(tmp_path, "spec.json", _base_spec())

    assert main(["diff", str(path), str(path), "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["equal"] is True
    assert payload["changed_fields"] == []
    assert payload["metrics"]["total_time_s"]["delta"] == pytest.approx(0.0)


def test_diff_reports_invalid_specs(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    valid = _write(tmp_path, "valid.json", _base_spec())
    invalid = _base_spec()
    invalid["focused"]["channels"] = 0
    invalid_path = _write(tmp_path, "invalid.json", invalid)

    assert main(["diff", str(valid), str(invalid_path)]) == 1

    output = capsys.readouterr().out
    assert "invalid" in output
    assert "focused" in output


def test_diff_parse_error_exits_two(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    broken = tmp_path / "broken.json"
    broken.write_text("[]", encoding="utf-8")
    valid = _write(tmp_path, "valid.json", _base_spec())

    assert main(["diff", str(valid), str(broken)]) == 2

    assert "JSON object" in capsys.readouterr().err


def test_compare_prints_regular_and_adaptive_broom_figures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["compare", "--samples", "10", "--scenario", "measured"]) == 0

    output = capsys.readouterr().out
    assert "dual snap scan vs adaptive broom" in output
    assert "broom-fast" in output
    assert "broom-2s" in output
    assert "adaptive" in output
    assert "added" in output
    assert "ramp" in output
    assert "dual-minus-adaptive" in output
    assert "bleach-cost" in output
    assert "speedup" in output
    assert "vs-broom" in output


def test_compare_json_reports_the_modulation_figures(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main(["compare", "--samples", "10", "--scenario", "measured"]) == 0
    capsys.readouterr()
    assert main(["compare", "--samples", "10", "--scenario", "measured", "--json"]) == 0

    payload = json.loads(capsys.readouterr().out)
    row = payload["rows_data"][0]
    assert row["samples"] == 10
    assert row["broom_cruise_s"] < row["adaptive_broom_s"] < row["broom_dwell_s"]
    assert row["added_vs_cruise_s"] == pytest.approx(
        row["adaptive_broom_s"] - row["broom_cruise_s"]
    )
    assert row["saved_vs_dwell_s"] == pytest.approx(
        row["broom_dwell_s"] - row["adaptive_broom_s"]
    )
    assert row["dual_snap_s"] == pytest.approx(
        row["dual_minus_adaptive_s"] + row["adaptive_broom_s"]
    )
    assert row["bleach_cost_mean"] >= 0.0
    assert row["bleach_retained_mean"] is not None


def test_compare_output_writes_the_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "compare.json"

    assert (
        main(
            [
                "compare",
                "--samples",
                "10",
                "--scenario",
                "measured",
                "--output",
                str(output),
            ]
        )
        == 0
    )

    capsys.readouterr()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["rows_data"][0]["samples"] == 10
    assert payload["field_um"] == [2000.0, 2000.0]


def test_simulate_html_writes_a_self_contained_page(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    spec = _write(tmp_path, "spec.json", _base_spec())
    page = tmp_path / "simulation.html"

    assert main(["simulate", str(spec), "--html", str(page)]) == 0

    capsys.readouterr()
    text = page.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "<svg" in text
    assert "Sub-scan metrics" in text


def test_compare_html_writes_a_self_contained_page(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    page = tmp_path / "comparison.html"

    assert (
        main(
            [
                "compare",
                "--samples",
                "10",
                "--scenario",
                "measured",
                "--html",
                str(page),
            ]
        )
        == 0
    )

    capsys.readouterr()
    text = page.read_text(encoding="utf-8")
    assert text.startswith("<!doctype html>")
    assert "<svg" in text
    assert "broom-fast" in text


def test_compare_envelope_adds_operating_corners(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    envelope = _write(
        tmp_path,
        "envelope.json",
        {
            "velocity_um_s": [66.56, 103.7],
            "exposure_s": [0.02, 2.0],
            "width_um": [64.0, 133.12],
        },
    )

    assert (
        main(
            [
                "compare",
                "--samples",
                "10",
                "--scenario",
                "measured",
                "--envelope",
                str(envelope),
                "--json",
            ]
        )
        == 0
    )

    payload = json.loads(capsys.readouterr().out)
    operating = payload["operating"]
    assert operating["has_envelope"] is True
    assert operating["thresholds"]["velocity"] == [66.56, 103.7]
    assert any(
        point["corner"] is not None
        for trace in operating["traces"]
        for point in trace["points"]
    )
