"""Self-contained HTML throughput reports for planner estimates.

The pages use inline CSS and SVG only: no network request, no JavaScript and no
external asset. They are reporting artifacts over declared and measured values,
not device evidence.
"""

from __future__ import annotations

import html
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .operating_map import simulation_operating_payload

if TYPE_CHECKING:
    from .acquisition_space import AcquisitionEnvelope
    from .simulation import RoundReport, SearchImagingSpec, SimulationReport

_BAR_COLORS = {
    "survey": "#3572a5",
    "focused": "#2f9e6f",
    "review": "#9aa7b4",
    "broom_fast": "#b8c4d0",
    "broom_2s": "#d99a2b",
    "adaptive": "#2f9e6f",
    "dual": "#3572a5",
}
_SERIES = (
    ("broom_cruise_s", "broom-fast", _BAR_COLORS["broom_fast"]),
    ("broom_dwell_s", "broom-2s", _BAR_COLORS["broom_2s"]),
    ("adaptive_broom_s", "adaptive broom", _BAR_COLORS["adaptive"]),
    ("dual_snap_s", "dual snap scan", _BAR_COLORS["dual"]),
)
_CSS = """
:root { color-scheme: light; }
body { margin: 0; padding: 2rem 1rem; background: #f4f7fa; color: #1d2733;
       font: 14px/1.45 system-ui, -apple-system, Segoe UI, sans-serif; }
main { max-width: 1000px; margin: 0 auto; }
h1 { margin: 0 0 0.25rem; font-size: 1.55rem; }
h2 { margin: 0 0 0.75rem; font-size: 1.1rem; }
p.lede { margin: 0 0 1.25rem; color: #50606f; }
.card { background: #fff; border: 1px solid #d8e0e8; border-radius: 8px;
        padding: 1rem 1.25rem; margin: 1rem 0; }
.cards { display: grid; grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
         gap: 0.75rem; }
.metric { border-left: 4px solid #3572a5; padding: 0.35rem 0.75rem; }
.metric .value { font-size: 1.25rem; font-weight: 650; }
.metric .label { color: #50606f; font-size: 0.85rem; }
figure { margin: 0; }
figcaption { font-weight: 650; margin-bottom: 0.35rem; }
svg { display: block; width: 100%; height: auto; }
.legend { display: flex; flex-wrap: wrap; gap: 1rem; margin-top: 0.5rem;
          color: #50606f; font-size: 0.85rem; }
.legend span { display: inline-flex; align-items: center; gap: 0.35rem; }
.swatch { display: inline-block; width: 0.8rem; height: 0.8rem; border-radius: 2px; }
.stack { display: flex; height: 1.1rem; border-radius: 4px; overflow: hidden;
         background: #e9eef3; }
.stack span { display: block; height: 100%; }
table { width: 100%; border-collapse: collapse; font-variant-numeric: tabular-nums; }
th, td { text-align: right; padding: 0.35rem 0.5rem; border-bottom: 1px solid #e4eaf0; }
th:first-child, td:first-child { text-align: left; }
th { color: #50606f; font-weight: 600; }
tr:last-child td { border-bottom: 0; }
.note { color: #50606f; font-size: 0.85rem; margin-top: 0.75rem; }
.maps { display: grid; grid-template-columns: repeat(auto-fit, minmax(340px, 1fr));
        gap: 1rem; }
details { margin-top: 0.75rem; }
summary { cursor: pointer; color: #3572a5; }
"""


def _escape(value: object) -> str:
    """Return an HTML-escaped string for text nodes and attributes."""
    return html.escape(str(value), quote=True)


def _fmt_seconds(value: float) -> str:
    return f"{value:,.1f} s"


def _bar_chart(title: str, entries: Sequence[tuple[str, float, str]]) -> str:
    """Return a horizontal SVG bar chart for labelled values."""
    if not entries:
        return ""
    label_width = 190
    plot_width = 500
    row_height = 30
    top = 10
    width = label_width + plot_width + 130
    height = top * 2 + row_height * len(entries)
    maximum = max(max(value for _label, value, _colour in entries), 1.0)
    bars = []
    for index, (label, value, colour) in enumerate(entries):
        y = top + index * row_height
        bar_width = max(1.0, plot_width * value / maximum)
        bars.append(
            f'<text x="{label_width - 8}" y="{y + 15}" text-anchor="end" '
            f'font-size="12">{_escape(label)}</text>'
            f'<rect x="{label_width}" y="{y + 3}" width="{bar_width:.2f}" '
            f'height="18" rx="3" fill="{colour}"><title>{_escape(label)}: '
            f"{value:,.3f}</title></rect>"
            f'<text x="{label_width + bar_width + 8}" y="{y + 16}" '
            f'font-size="12">{value:,.2f}</text>'
        )
    return (
        f"<figure><figcaption>{_escape(title)}</figcaption>"
        f'<svg role="img" aria-label="{_escape(title)}" viewBox="0 0 {width} {height}">'
        f"{''.join(bars)}</svg></figure>"
    )


def _stacked(round_report: RoundReport) -> str:
    """Return a stacked time-breakdown bar for one round."""
    total = round_report.total_s or 1.0
    parts = (
        ("travel", round_report.travel_s, "#3572a5"),
        ("settle", round_report.settle_s, "#7fa8cc"),
        ("exposure", round_report.exposure_s, "#2f9e6f"),
        ("readout", round_report.readout_s, "#e0b05a"),
        ("overhead", round_report.overhead_s, "#9aa7b4"),
    )
    segments = "".join(
        f'<span style="width:{value / total * 100:.4f}%;background:{colour}" '
        f'title="{_escape(name)}: {_fmt_seconds(value)}"></span>'
        for name, value, colour in parts
    )
    legend = "".join(
        f'<span><i class="swatch" style="background:{colour}"></i>'
        f"{_escape(name)} {_fmt_seconds(value)}</span>"
        for name, value, colour in parts
    )
    return f'<div class="stack" aria-label="time breakdown">{segments}</div><div class="legend">{legend}</div>'


def _grouped_bar_chart(
    title: str,
    samples: Sequence[int],
    rows_by_sample: Mapping[int, Mapping[str, float]],
) -> str:
    """Return a grouped SVG bar chart over sample counts and series."""
    if not samples:
        return ""
    label_width = 40
    group_width = 150
    bar_width = 22
    len(_SERIES)
    plot_height = 190
    width = label_width + group_width * len(samples) + 20
    height = plot_height + 80
    maximum = max(
        max(row[key] for key, _label, _colour in _SERIES)
        for row in rows_by_sample.values()
    )
    maximum = max(maximum, 1.0)
    bars = []
    for group_index, count in enumerate(samples):
        base_x = label_width + group_index * group_width
        row = rows_by_sample[count]
        for series_index, (key, label, colour) in enumerate(_SERIES):
            value = row[key]
            bar_height = max(1.0, plot_height * value / maximum)
            x = base_x + series_index * (bar_width + 4)
            y = 10 + plot_height - bar_height
            bars.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_width}" '
                f'height="{bar_height:.2f}" rx="3" fill="{colour}">'
                f"<title>{_escape(label)} at {count} samples: {_fmt_seconds(value)}</title>"
                "</rect>"
            )
        label = f"{count} samples"
        bars.append(
            f'<text x="{base_x + group_width / 2 - 10:.2f}" y="{height - 44}" '
            f'text-anchor="middle" font-size="12">{_escape(label)}</text>'
        )
    legend = "".join(
        f'<span><i class="swatch" style="background:{colour}"></i>{_escape(label)}</span>'
        for _key, label, colour in _SERIES
    )
    return (
        f"<figure><figcaption>{_escape(title)}</figcaption>"
        f'<svg role="img" aria-label="{_escape(title)}" viewBox="0 0 {width} {height}">'
        f'{"".join(bars)}</svg><div class="legend">{legend}</div></figure>'
    )


def _layout_svg(
    title: str,
    field_size_um: tuple[float, float],
    *,
    survey_tiles: Sequence[tuple[float, float]] = (),
    tile_size_um: tuple[float, float] | None = None,
    samples: Sequence[tuple[float, float]] = (),
    focused_roi_um: tuple[float, float] | None = None,
    broom_rows: Sequence[float] = (),
    broom_fov_um: float | None = None,
    slow_zones: Sequence[tuple[float, float, float]] = (),
) -> str:
    """Return a spatial SVG map of the survey, broom and focused boxes."""
    width, height = field_size_um
    margin = max(width, height) * 0.08
    font = max(width, height) * 0.016
    parts = [
        (
            f'<rect x="0" y="0" width="{width:.3f}" height="{height:.3f}" '
            'fill="#f8fafc" stroke="#334155" stroke-width="2">'
            f"<title>Field {width:.1f} x {height:.1f} um</title></rect>"
        )
    ]
    for centre_x, centre_y in survey_tiles:
        tile_width, tile_height = tile_size_um or (0.0, 0.0)
        parts.append(
            f'<rect x="{centre_x - tile_width / 2:.3f}" '
            f'y="{height - centre_y - tile_height / 2:.3f}" '
            f'width="{tile_width:.3f}" height="{tile_height:.3f}" '
            'fill="#3572a5" fill-opacity="0.10" stroke="#3572a5" '
            'stroke-opacity="0.45" stroke-width="1">'
            f"<title>Survey tile at {centre_x:.1f}, {centre_y:.1f} um</title>"
            "</rect>"
        )
    if broom_fov_um is not None:
        for row_y in broom_rows:
            parts.append(
                f'<rect x="0" y="{height - row_y - broom_fov_um / 2:.3f}" '
                f'width="{width:.3f}" height="{broom_fov_um:.3f}" '
                'fill="#2f9e6f" fill-opacity="0.08" stroke="#2f9e6f" '
                'stroke-opacity="0.55" stroke-dasharray="6 4" stroke-width="1.5">'
                f"<title>Broom strip at {row_y:.1f} um</title></rect>"
            )
    for start_x, end_x, zone_y, zone_fov in (
        (*zone, broom_fov_um or 0.0) for zone in slow_zones
    ):
        parts.append(
            f'<rect x="{start_x:.3f}" '
            f'y="{height - zone_y - zone_fov / 2:.3f}" '
            f'width="{max(end_x - start_x, 0.1):.3f}" height="{zone_fov:.3f}" '
            'fill="#e0b05a" fill-opacity="0.30" stroke="#b7791f" stroke-width="1">'
            f"<title>Two-second slow zone {start_x:.1f}-{end_x:.1f} um</title>"
            "</rect>"
        )
    if focused_roi_um is not None:
        roi_width, roi_height = focused_roi_um
        for sample_x, sample_y in samples:
            parts.append(
                f'<rect x="{sample_x - roi_width / 2:.3f}" '
                f'y="{height - sample_y - roi_height / 2:.3f}" '
                f'width="{roi_width:.3f}" height="{roi_height:.3f}" '
                'fill="none" stroke="#3572a5" stroke-dasharray="5 3" stroke-width="1.5">'
                f"<title>Second-round ROI at {sample_x:.1f}, {sample_y:.1f} um</title>"
                "</rect>"
            )
    for sample_x, sample_y in samples:
        parts.append(
            f'<circle cx="{sample_x:.3f}" cy="{height - sample_y:.3f}" '
            f'r="{max(max(width, height) * 0.004, 2.0):.3f}" fill="#c2410c" '
            'stroke="#7c2d12" stroke-width="1">'
            f"<title>Sample at {sample_x:.1f}, {sample_y:.1f} um</title></circle>"
        )
    parts.append(
        f'<text x="0" y="{height + font * 1.8:.3f}" font-size="{font:.3f}" '
        'fill="#475569">0</text>'
        f'<text x="{width:.3f}" y="{height + font * 1.8:.3f}" '
        f'text-anchor="end" font-size="{font:.3f}" fill="#475569">'
        f"{width:.0f} um</text>"
    )
    legend = []
    if survey_tiles:
        tile_width, tile_height = tile_size_um or (0.0, 0.0)
        legend.append(
            (f"Survey tile {tile_width:.1f} x {tile_height:.1f} um", "#3572a5", "0.25")
        )
    if broom_rows:
        legend.append((f"Broom strip {broom_fov_um or 0.0:.1f} um", "#2f9e6f", "0.25"))
    if slow_zones:
        legend.append(("2 s slow zone", "#e0b05a", "0.6"))
    if focused_roi_um is not None:
        legend.append(
            (
                f"Focused ROI {focused_roi_um[0]:.1f} x {focused_roi_um[1]:.1f} um",
                "#3572a5",
                "0",
            )
        )
    if samples:
        legend.append(("Mock sample", "#c2410c", "1"))
    legend_html = "".join(
        f'<span><i class="swatch" style="background:{colour};opacity:{opacity}"></i>'
        f"{_escape(label)}</span>"
        for label, colour, opacity in legend
    )
    view_width = width + 2 * margin
    view_height = height + 2.8 * margin
    return (
        f"<figure><figcaption>{_escape(title)}</figcaption>"
        f'<svg role="img" aria-label="{_escape(title)}" '
        f'viewBox="{-margin:.3f} {-margin:.3f} {view_width:.3f} {view_height:.3f}">'
        f'{"".join(parts)}</svg><div class="legend">{legend_html}</div></figure>'
    )


def _operating_map_svg(payload: Mapping[str, Any]) -> str:
    """Return a three-projection operating-space map with plan trajectories."""
    ranges = payload.get("ranges")
    traces = payload.get("traces")
    if not isinstance(ranges, Mapping) or not isinstance(traces, list):
        return ""
    axis_ranges = {
        name: (float(bounds[0]), float(bounds[1])) for name, bounds in ranges.items()
    }
    thresholds = payload.get("thresholds")
    panels = (
        ("velocity", "exposure", "v (um/s)", "T_e (s)"),
        ("velocity", "width", "v (um/s)", "W (um)"),
        ("exposure", "width", "T_e (s)", "W (um)"),
    )
    panel_width = 300
    panel_height = 230
    top = 30
    bottom = 52
    gap = 34
    width = 3 * panel_width + 2 * gap + 20
    height = panel_height + top + bottom
    defs = []
    body = []
    for index, (x_name, y_name, x_label, y_label) in enumerate(panels):
        x0 = 10 + index * (panel_width + gap)
        y0 = top
        x_min, x_max = axis_ranges[x_name]
        y_min, y_max = axis_ranges[y_name]
        x_span = max(x_max - x_min, 1e-9)
        y_span = max(y_max - y_min, 1e-9)

        inset = 8.0
        plot_left = x0 + inset
        plot_width = panel_width - 2 * inset
        plot_bottom = y0 + panel_height - inset
        plot_height = panel_height - 2 * inset

        def xpix(
            value: float,
            plot_left: float = plot_left,
            x_min: float = x_min,
            x_span: float = x_span,
            plot_width: float = plot_width,
        ) -> float:
            return plot_left + (value - x_min) / x_span * plot_width

        def ypix(
            value: float,
            plot_bottom: float = plot_bottom,
            y_min: float = y_min,
            y_span: float = y_span,
            plot_height: float = plot_height,
        ) -> float:
            return plot_bottom - (value - y_min) / y_span * plot_height

        body.append(
            f'<rect x="{x0}" y="{y0}" width="{panel_width}" height="{panel_height}" '
            'fill="#f8fafc" stroke="#334155" stroke-width="1"/>'
        )
        if isinstance(thresholds, Mapping):
            for axis, orientation in ((x_name, "x"), (y_name, "y")):
                bounds = thresholds.get(axis)
                if not isinstance(bounds, list) or len(bounds) != 2:
                    continue
                low, high = float(bounds[0]), float(bounds[1])
                if orientation == "x":
                    body.append(
                        f'<rect x="{xpix(x_min):.2f}" y="{y0}" '
                        f'width="{max(xpix(low) - xpix(x_min), 0):.2f}" '
                        f'height="{panel_height}" fill="#e0f2fe"/>'
                    )
                    body.append(
                        f'<rect x="{xpix(high):.2f}" y="{y0}" '
                        f'width="{max(xpix(x_max) - xpix(high), 0):.2f}" '
                        f'height="{panel_height}" fill="#ffedd5"/>'
                    )
                else:
                    body.append(
                        f'<rect x="{x0}" y="{ypix(high):.2f}" '
                        f'width="{panel_width}" '
                        f'height="{max(ypix(y_min) - ypix(high), 0):.2f}" '
                        'fill="#ffedd5"/>'
                    )
                    body.append(
                        f'<rect x="{x0}" y="{ypix(low):.2f}" '
                        f'width="{panel_width}" '
                        f'height="{max(ypix(y_max) - ypix(low), 0):.2f}" '
                        'fill="#e0f2fe"/>'
                    )
        for tick in (0.0, 0.5, 1.0):
            x_value = x_min + x_span * tick
            y_value = y_min + y_span * tick
            body.append(
                f'<line x1="{xpix(x_value):.2f}" y1="{y0}" '
                f'x2="{xpix(x_value):.2f}" y2="{y0 + panel_height}" '
                'stroke="#e2e8f0" stroke-width="1"/>'
                f'<line x1="{x0}" y1="{ypix(y_value):.2f}" '
                f'x2="{x0 + panel_width}" y2="{ypix(y_value):.2f}" '
                'stroke="#e2e8f0" stroke-width="1"/>'
                f'<text x="{xpix(x_value):.2f}" y="{y0 + panel_height + 16}" '
                'text-anchor="middle" font-size="10" fill="#475569">'
                f"{x_value:.4g}</text>"
                f'<text x="{x0 - 6}" y="{ypix(y_value) + 3:.2f}" '
                'text-anchor="end" font-size="10" fill="#475569">'
                f"{y_value:.4g}</text>"
            )
        body.append(
            f'<text x="{x0 + panel_width / 2}" y="{y0 + panel_height + 32}" '
            'text-anchor="middle" font-size="11" fill="#334155">'
            f"{_escape(x_label)}</text>"
            f'<text x="{x0 - 40}" y="{y0 + panel_height / 2}" '
            'text-anchor="middle" font-size="11" fill="#334155" '
            f'transform="rotate(-90 {x0 - 40} {y0 + panel_height / 2})">'
            f"{_escape(y_label)}</text>"
        )
        x_key = {"velocity": "v", "exposure": "te", "width": "w"}[x_name]
        y_key = {"velocity": "v", "exposure": "te", "width": "w"}[y_name]
        for trace_index, trace in enumerate(traces):
            if not isinstance(trace, Mapping):
                continue
            points = trace.get("points", [])
            if not isinstance(points, list) or not points:
                continue
            colour = str(trace.get("color", "#3572a5"))
            marker_id = f"op-arrow-{index}-{trace_index}"
            defs.append(
                f'<marker id="{marker_id}" viewBox="0 0 10 10" refX="8" refY="5" '
                'markerWidth="5" markerHeight="5" orient="auto-start-reverse">'
                f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{colour}"/></marker>'
            )
            coordinates = [
                (
                    xpix(float(point[x_key])),
                    ypix(float(point[y_key])),
                )
                for point in points
                if isinstance(point, Mapping) and x_key in point and y_key in point
            ]
            if not coordinates:
                continue
            dash = ' stroke-dasharray="5 3"' if trace.get("dashed") else ""
            path = " ".join(
                f"{'M' if point_index == 0 else 'L'} {x:.2f} {y:.2f}"
                for point_index, (x, y) in enumerate(coordinates)
            )
            body.append(
                f'<path d="{path}" fill="none" stroke="{colour}" stroke-width="2"'
                f'{dash} marker-end="url(#{marker_id})"/>'
            )
            if trace.get("markers"):
                for point, (x, y) in zip(
                    [point for point in points if isinstance(point, Mapping)],
                    coordinates,
                    strict=False,
                ):
                    label = str(point.get("label", ""))
                    body.append(
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" '
                        f'fill="{colour}" stroke="#fff" stroke-width="1">'
                        f"<title>{_escape(label)}</title></circle>"
                    )
    key_points = []
    key_labels = {
        "survey tile",
        "survey equivalent",
        "focused revisit",
        "dwell at sample",
        "cruise between samples",
    }
    for trace in traces:
        if not isinstance(trace, Mapping):
            continue
        trace_name = str(trace.get("name", "trace"))
        trace_colour = str(trace.get("color", "#3572a5"))
        points = trace.get("points", [])
        if not isinstance(points, list):
            continue
        for point in points:
            if not isinstance(point, Mapping) or point.get("label") not in key_labels:
                continue
            corner = point.get("corner")
            key_points.append(
                "<tr>"
                f'<td><i class="swatch" style="background:{_escape(trace_colour)}"></i> '
                f"{_escape(trace_name)}</td>"
                f"<td>{_escape(str(point.get('label', '')))}</td>"
                f"<td>{float(point['v']):.4g}</td>"
                f"<td>{float(point['te']):.4g}</td>"
                f"<td>{float(point['w']):.4g}</td>"
                f"<td>{_escape(str(corner)) if corner else '&mdash;'}</td>"
                "</tr>"
            )
    key_table = ""
    if key_points:
        key_table = (
            "<details><summary>Key operating points</summary>"
            "<table><thead><tr><th>trajectory</th><th>point</th><th>v (um/s)</th>"
            "<th>T_e (s)</th><th>W (um)</th><th>corner</th></tr></thead>"
            f"<tbody>{''.join(key_points)}</tbody></table></details>"
        )
    legend = "".join(
        f'<span><i class="swatch" style="background:{_escape(str(trace.get("color", "#3572a5")))}"></i>'
        f"{_escape(str(trace.get('name', 'trace')))}</span>"
        for trace in traces
        if isinstance(trace, Mapping)
    )
    return (
        "<figure><figcaption>Operating-space trajectories "
        "(arrows follow acquisition order)</figcaption>"
        f'<svg role="img" aria-label="Operating-space trajectories" '
        f'viewBox="0 0 {width} {height}">'
        f"<defs>{''.join(defs)}</defs>{''.join(body)}</svg>"
        f'<div class="legend">{legend}</div>{key_table}</figure>'
    )


def render_simulation_html(
    report: SimulationReport,
    *,
    spec: SearchImagingSpec | None = None,
    envelope: AcquisitionEnvelope | None = None,
    title: str = "Planner throughput report",
) -> str:
    """Return a self-contained HTML report for one simulated dual scan."""
    field = (
        f"{spec.field_size_um[0]:,.1f} x {spec.field_size_um[1]:,.1f} um"
        if spec is not None
        else f"{report.field_area_um2 / 1e6:.3f} mm2 field"
    )
    photobleaching = report.photobleaching
    retained = (
        f"{photobleaching.retained_mean:.4f}"
        if photobleaching.retained_mean is not None
        else "not computed"
    )
    time_chart = _bar_chart(
        "Wall time by component",
        (
            ("survey", report.survey.total_s, _BAR_COLORS["survey"]),
            ("focused", report.focused.total_s, _BAR_COLORS["focused"]),
            ("review", report.review_s, _BAR_COLORS["review"]),
        ),
    )
    throughput_chart = _bar_chart(
        "Throughput (mm2/min)",
        (
            (
                "survey",
                report.survey_scan.throughput_um2_s * 60.0 / 1e6,
                _BAR_COLORS["survey"],
            ),
            (
                "focused",
                report.focused_scan.throughput_um2_s * 60.0 / 1e6,
                _BAR_COLORS["focused"],
            ),
            (
                "mean sub-scan",
                report.averaged.mean_throughput_um2_s * 60.0 / 1e6,
                "#e0b05a",
            ),
            (
                "end-to-end",
                report.end_to_end_throughput_um2_s * 60.0 / 1e6,
                "#9aa7b4",
            ),
        ),
    )
    layout = ""
    if spec is not None:
        layout = _layout_svg(
            "Survey tiles, samples and second-round boxes",
            spec.field_size_um,
            survey_tiles=tuple(tile.center_xy_um for tile in spec.survey_tiles()),
            tile_size_um=spec.survey.tile_size_um,
            samples=tuple(target.center_xy_um for target in spec.targets),
            focused_roi_um=spec.focused.roi_size_um,
        )
    geometry_card = ""
    if spec is not None:
        survey_tiles = spec.survey_tiles()
        survey_rows = max(tile.row for tile in survey_tiles) + 1
        survey_columns = max(tile.column for tile in survey_tiles) + 1
        geometry_rows = (
            (
                "field",
                f"{spec.field_size_um[0]:.1f} x {spec.field_size_um[1]:.1f} um",
            ),
            (
                "survey tile",
                (
                    f"{spec.survey.tile_size_um[0]:.1f} x "
                    f"{spec.survey.tile_size_um[1]:.1f} um"
                ),
            ),
            (
                "survey grid",
                (
                    f"{survey_rows} x {survey_columns} = {len(survey_tiles)} tiles, "
                    f"overlap {spec.survey.overlap:.3f}"
                ),
            ),
            (
                "focused ROI",
                (
                    f"{spec.focused.roi_size_um[0]:.1f} x "
                    f"{spec.focused.roi_size_um[1]:.1f} um"
                ),
            ),
            ("samples", f"{len(spec.targets)}"),
        )
        geometry_table = "".join(
            f"<tr><td>{_escape(label)}</td><td>{_escape(value)}</td></tr>"
            for label, value in geometry_rows
        )
        geometry_card = (
            "<div class='card'><h2>Geometry</h2>"
            f"<table><tbody>{geometry_table}</tbody></table></div>"
        )
    operating_card = ""
    if spec is not None:
        operating_card = (
            "<div class='card'><h2>Operating-space trajectories</h2>"
            f"{_operating_map_svg(simulation_operating_payload(report, envelope))}"
            "<div class='note'>Each point is a declared operating point; arrows "
            "follow acquisition order. Low/high shading appears when the "
            "specification declares an acquisition envelope.</div></div>"
        )
    rows = (
        ("survey", report.survey_scan),
        ("focused", report.focused_scan),
    )
    table = "".join(
        "<tr>"
        f"<td>{_escape(scan.scan_id)}</td>"
        f"<td>{scan.positions}</td>"
        f"<td>{scan.area_um2 / 1e6:.3f}</td>"
        f"<td>{scan.duration_s:.3f}</td>"
        f"<td>{scan.throughput_um2_s * 60.0 / 1e6:.3f}</td>"
        f"<td>{scan.seconds_per_position:.3f}</td>"
        f"<td>{scan.travel_fraction:.0%}</td>"
        f"<td>{scan.exposure_fraction:.0%}</td>"
        "</tr>"
        for _name, scan in rows
    )
    corners = (
        ("survey configured", report.survey_acquisition_corner),
        ("survey equivalent", report.survey_equivalent_corner),
        ("focused", report.focused_corner),
    )
    corner_rows = "".join(
        f"<tr><td>{_escape(label)}</td><td>{_escape('n/a' if corner is None else corner.value)}</td></tr>"
        for label, corner in corners
    )
    photobleaching_rows = (
        (
            "preview exposure",
            f"{photobleaching.preview_exposure_s * 1000.0:.1f} ms/tile",
        ),
        ("mean covering tiles", f"{photobleaching.mean_cover_count:.2f}"),
        (
            "prior exposure",
            (
                f"{photobleaching.prior_exposure_s_min * 1000.0:.1f} - "
                f"{photobleaching.prior_exposure_s_max * 1000.0:.1f} ms "
                f"(mean {photobleaching.prior_exposure_s_mean * 1000.0:.1f})"
            ),
        ),
        ("focused exposure", f"{photobleaching.focused_exposure_s:.3f} s"),
        (
            "photobleaching cost",
            (
                f"{photobleaching.cost_min:.4f} - {photobleaching.cost_max:.4f} "
                f"(mean {photobleaching.cost_mean:.4f})"
            ),
        ),
        ("retained brightness", retained),
    )
    photobleaching_table = "".join(
        f"<tr><td>{_escape(label)}</td><td>{value}</td></tr>"
        for label, value in photobleaching_rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>{_escape(title)}</h1>
<p class="lede">{_escape(field)} &middot; {report.targets} mock samples &middot;
hardware-free planning estimate</p>
<div class="cards">
  <div class="metric"><div class="value">{report.total_s:,.1f} s</div><div class="label">total time</div></div>
  <div class="metric"><div class="value">{report.end_to_end_throughput_um2_s * 60.0 / 1e6:,.3f}</div><div class="label">mm2/min end-to-end</div></div>
  <div class="metric"><div class="value">{report.targets_per_hour:,.1f}</div><div class="label">targets/hour</div></div>
  <div class="metric"><div class="value">{report.survey.positions}</div><div class="label">overview tiles</div></div>
</div>
{geometry_card}
{operating_card}
<div class="card"><h2>Wall time</h2>{time_chart}
<div class="note">Survey time breakdown</div>{_stacked(report.survey)}
<div class="note">Focused time breakdown</div>{_stacked(report.focused)}</div>
<div class="card"><h2>Throughput</h2>{throughput_chart}</div>
<div class="card"><h2>Survey and focused layout</h2>{layout}</div>
<div class="card"><h2>Sub-scan metrics</h2>
<table><thead><tr><th>sub-scan</th><th>positions</th><th>area mm2</th><th>duration s</th>
<th>mm2/min</th><th>s/position</th><th>travel</th><th>exposure</th></tr></thead>
<tbody>{table}</tbody></table>
<div class="note">Means: {report.averaged.mean_throughput_um2_s * 60.0 / 1e6:,.3f} mm2/min,
{report.averaged.mean_seconds_per_position:,.3f} s/position; combined acquired
{report.averaged.combined_acquired_area_um2 / 1e6:,.3f} mm2 in
{report.survey.total_s + report.focused.total_s:,.1f} s.</div></div>
<div class="card"><h2>Photobleaching from the overview</h2>
<table><tbody>{photobleaching_table}</tbody></table>
<div class="note">Cost is prior overview exposure divided by the focused exposure
(relative to 1). Retained brightness is exp(-prior / tau) when a time constant is
declared.</div></div>
<div class="card"><h2>Operating points</h2>
<table><thead><tr><th>point</th><th>corner</th></tr></thead><tbody>{corner_rows}</tbody></table></div>
<p class="note">Planning estimate only. No device, RunEngine or camera was used.</p>
</main>
</body>
</html>
"""


def render_comparison_html(
    payload: Mapping[str, Any],
    *,
    title: str = "Adaptive broom versus dual snap scan",
) -> str:
    """Return a self-contained HTML report for the compare payload."""
    rows = payload.get("rows_data", [])
    if not isinstance(rows, list):
        raise TypeError("rows_data must be a list")
    numeric_rows: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            raise TypeError("each comparison row must be a mapping")
        numeric_rows.append(row)
    scenarios = sorted({str(row["scenario"]) for row in numeric_rows})
    sections = []
    for scenario in scenarios:
        selected = [row for row in numeric_rows if str(row["scenario"]) == scenario]
        sample_counts = sorted({int(row["samples"]) for row in selected})
        by_sample = {
            count: {
                key: float(
                    next(row for row in selected if int(row["samples"]) == count)[key]
                )
                for key, _label, _colour in _SERIES
            }
            for count in sample_counts
        }
        chart = _grouped_bar_chart(
            f"Wall time, {scenario}",
            sample_counts,
            by_sample,
        )
        sections.append(f"<div class='card'><h2>{_escape(scenario)}</h2>{chart}</div>")
    layout_sections = ""
    geometry = payload.get("geometry")
    if isinstance(geometry, Mapping):
        field = tuple(float(value) for value in geometry["field_um"])
        samples = tuple(
            (float(point[0]), float(point[1])) for point in geometry.get("samples", ())
        )
        survey_tiles = tuple(
            (float(point[0]), float(point[1]))
            for point in geometry.get("survey_tiles", ())
        )
        tile_size = tuple(float(value) for value in geometry["survey_tile_size_um"])
        focused_roi = tuple(float(value) for value in geometry["focused_roi_um"])
        broom_rows = tuple(float(value) for value in geometry.get("broom_rows_um", ()))
        broom_fov = float(geometry.get("broom_fov_um", 0.0))
        slow_zones = tuple(
            (float(zone[0]), float(zone[1]), float(zone[2]))
            for zone in geometry.get("slow_zones_um", ())
        )
        snap_map = _layout_svg(
            f"Dual snap scan coverage ({int(geometry.get('sample_count', 0))} samples): "
            "overview tiles, samples and second-round boxes",
            (field[0], field[1]),
            survey_tiles=survey_tiles,
            tile_size_um=(tile_size[0], tile_size[1]),
            samples=samples,
            focused_roi_um=(focused_roi[0], focused_roi[1]),
        )
        broom_map = _layout_svg(
            "Adaptive broom raster: strips, samples and two-second slow zones",
            (field[0], field[1]),
            samples=samples,
            broom_rows=broom_rows,
            broom_fov_um=broom_fov,
            slow_zones=slow_zones,
        )
        layout_sections = (
            "<div class='card'><h2>Plan geometry</h2>"
            f"<div class='maps'>{snap_map}{broom_map}</div></div>"
        )
    operating_section = ""
    operating = payload.get("operating")
    if isinstance(operating, Mapping):
        operating_section = (
            "<div class='card'><h2>Operating-space trajectories</h2>"
            f"{_operating_map_svg(operating)}"
            "<div class='note'>Arrows follow acquisition order; the broom trace "
            "ramps from its two-second dwell to travel speed and back. Low/high "
            "shading appears when the specification declares an acquisition "
            "envelope.</div></div>"
        )
    table_rows = "".join(
        "<tr>"
        f"<td>{_escape(row['scenario'])}</td>"
        f"<td>{int(row['samples'])}</td>"
        f"<td>{float(row['broom_cruise_s']):.1f}</td>"
        f"<td>{float(row['broom_dwell_s']):.1f}</td>"
        f"<td>{float(row['adaptive_broom_s']):.1f}</td>"
        f"<td>{float(row['added_vs_cruise_s']):+.1f}</td>"
        f"<td>{float(row['transition_s']):.1f}</td>"
        f"<td>{float(row['dual_snap_s']):.1f}</td>"
        f"<td>{float(row['speedup_vs_broom']):.2f}x</td>"
        f"<td>{float(row['speedup']):.2f}x</td>"
        f"<td>{float(row['bleach_cost_mean']):.4f}</td>"
        "</tr>"
        for row in numeric_rows
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{_escape(title)}</title>
<style>{_CSS}</style>
</head>
<body>
<main>
<h1>{_escape(title)}</h1>
<p class="lede">{float(payload.get("field_um", [0, 0])[0]):,.0f} x
{float(payload.get("field_um", [0, 0])[1]):,.0f} um &middot;
FOV {float(payload.get("fov_um", 0)):,.2f} um &middot;
dwell {float(payload.get("dwell_s", 0)):,.1f} s &middot; hardware-free estimate</p>
<div class="cards">
  <div class="metric"><div class="value">{float(payload.get("rows", 0))}</div><div class="label">broom rows</div></div>
  <div class="metric"><div class="value">{len(numeric_rows)}</div><div class="label">comparison rows</div></div>
  <div class="metric"><div class="value">{_escape(", ".join(scenarios))}</div><div class="label">stage scenarios</div></div>
</div>
{layout_sections}
{operating_section}
{"".join(sections)}
<div class="card"><h2>Comparison table</h2>
<table><thead><tr><th>scenario</th><th>samples</th><th>broom-fast</th><th>broom-2s</th>
<th>adaptive</th><th>added</th><th>ramp</th><th>dual snap</th><th>adaptive/broom-2s</th>
<th>dual/adaptive</th><th>bleach cost</th></tr></thead><tbody>{table_rows}</tbody></table>
<div class="note">broom-fast does not meet the two-second dwell; broom-2s is the
equivalent regular broom. Times in seconds. bleach cost is prior overview exposure
divided by the focused exposure.</div></div>
<p class="note">Planning estimate only. No device, RunEngine or camera was used.</p>
</main>
</body>
</html>
"""


def write_simulation_html(
    path: str | Path,
    report: SimulationReport,
    *,
    spec: SearchImagingSpec | None = None,
    envelope: AcquisitionEnvelope | None = None,
    title: str = "Planner throughput report",
) -> Path:
    """Write the simulation HTML report and return its path."""
    target = Path(path)
    target.write_text(
        render_simulation_html(report, spec=spec, envelope=envelope, title=title),
        encoding="utf-8",
    )
    return target


def write_comparison_html(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    title: str = "Adaptive broom versus dual snap scan",
) -> Path:
    """Write the comparison HTML report and return its path."""
    target = Path(path)
    target.write_text(render_comparison_html(payload, title=title), encoding="utf-8")
    return target


__all__ = [
    "render_comparison_html",
    "render_simulation_html",
    "write_comparison_html",
    "write_simulation_html",
]
