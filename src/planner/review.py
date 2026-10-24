"""Offline, auditable review of sample-search targets; this module never acquires."""

from __future__ import annotations

import base64
import hashlib
import html
import json
import math
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from .contracts import DatasetReference, SearchObservation, Target

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from .geometry import Calibration, Region


@dataclass(frozen=True)
class ReviewContext:
    """Identity of the saved run a review applies to."""

    run_id: str
    specimen_id: str
    coordinate_frame: str
    calibration_id: str


@dataclass(frozen=True)
class ManualTarget:
    """An operator-supplied target added during review."""

    target_id: str
    x_um: float
    y_um: float
    reason: str


@dataclass(frozen=True)
class ReviewManifest:
    """Checksummed identity of the candidate set an operator reviewed."""

    context: ReviewContext
    targets: tuple[dict[str, Any], ...]
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        """Return the manifest as a JSON-compatible mapping."""
        return {
            "context": vars(self.context),
            "targets": list(self.targets),
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class ReviewDecision:
    """Recorded operator decision: accepted, rejected and added targets."""

    manifest_sha256: str
    run_id: str
    specimen_id: str
    coordinate_frame: str
    calibration_id: str
    accepted_target_ids: tuple[str, ...] = ()
    rejected_target_ids: tuple[str, ...] = ()
    additions: tuple[ManualTarget, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Return the decision as a JSON-compatible mapping."""
        return {
            "manifest_sha256": self.manifest_sha256,
            "run_id": self.run_id,
            "specimen_id": self.specimen_id,
            "coordinate_frame": self.coordinate_frame,
            "calibration_id": self.calibration_id,
            "accepted_target_ids": list(self.accepted_target_ids),
            "rejected_target_ids": list(self.rejected_target_ids),
            "additions": [vars(a) for a in self.additions],
        }


@dataclass(frozen=True)
class ManualProvenance:
    """Why and by whom one manual target was added."""

    target_id: str
    reason: str
    x_um: float
    y_um: float


@dataclass(frozen=True)
class ReviewResult:
    """Target set produced by applying a decision to the candidates."""

    targets: tuple[Target, ...]
    manual_provenance: tuple[ManualProvenance, ...]
    manifest_sha256: str


def _target_record(t: Target) -> dict[str, Any]:
    return {
        "target_id": t.target_id,
        "specimen_id": t.specimen_id,
        "coordinate_frame": t.coordinate_frame,
        "calibration_id": t.calibration_id,
        "x_um": t.x_um,
        "y_um": t.y_um,
        "score": t.score,
        "bounds_xyxy_um": t.bounds_xyxy_um,
        "observations": [
            {
                "tile_id": o.tile_id,
                "request_id": o.request_id,
                "stage_xy_um": o.stage_xy_um,
                "data_reference": (
                    vars(o.data_reference)
                    if isinstance(o.data_reference, DatasetReference)
                    else o.data_reference
                ),
                "image_shape_yx": o.image_shape_yx,
            }
            for o in t.observations
        ],
    }


def create_manifest(
    context: ReviewContext, targets: Iterable[Target]
) -> ReviewManifest:
    """Build the checksummed manifest for a set of candidate targets."""
    records = tuple(_target_record(t) for t in targets)
    if len({r["target_id"] for r in records}) != len(records):
        raise ValueError("target IDs must be unique")
    for r in records:
        if (r["specimen_id"], r["coordinate_frame"], r["calibration_id"]) != (
            context.specimen_id,
            context.coordinate_frame,
            context.calibration_id,
        ):
            raise ValueError("target context does not match review context")
    payload = json.dumps(
        {"context": vars(context), "targets": records},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode()
    return ReviewManifest(context, records, hashlib.sha256(payload).hexdigest())


def make_decision(
    manifest: ReviewManifest,
    *,
    accepted_target_ids: Iterable[str] = (),
    rejected_target_ids: Iterable[str] = (),
    additions: Iterable[ManualTarget] = (),
) -> ReviewDecision:
    """Build a review decision from the operator's selections."""
    c = manifest.context
    return ReviewDecision(
        manifest.sha256,
        c.run_id,
        c.specimen_id,
        c.coordinate_frame,
        c.calibration_id,
        tuple(accepted_target_ids),
        tuple(rejected_target_ids),
        tuple(additions),
    )


def decision_from_json(source: str | Path) -> ReviewDecision:
    """Load a review decision from a saved JSON document."""
    if isinstance(source, Path):
        raw = source.read_text(encoding="utf-8")
    else:
        candidate = source.lstrip()
        raw = (
            source
            if candidate.startswith("{")
            else Path(source).read_text(encoding="utf-8")
        )
    data = json.loads(raw)
    return ReviewDecision(
        data["manifest_sha256"],
        data["run_id"],
        data["specimen_id"],
        data["coordinate_frame"],
        data["calibration_id"],
        tuple(data.get("accepted_target_ids", ())),
        tuple(data.get("rejected_target_ids", ())),
        tuple(ManualTarget(**item) for item in data.get("additions", ())),
    )


def apply_review(
    decision: ReviewDecision,
    candidates: Iterable[Target],
    *,
    context: ReviewContext,
    region: Region,
    max_targets: int,
) -> ReviewResult:
    """Apply a review decision to the reviewed candidates."""
    original = tuple(candidates)
    manifest = create_manifest(context, original)
    if decision.manifest_sha256 != manifest.sha256:
        raise ValueError("decision manifest hash is stale or targets were modified")
    if (
        decision.run_id,
        decision.specimen_id,
        decision.coordinate_frame,
        decision.calibration_id,
    ) != (
        context.run_id,
        context.specimen_id,
        context.coordinate_frame,
        context.calibration_id,
    ):
        raise ValueError("decision context does not match this run")
    ids = {t.target_id for t in original}
    accepted, rejected = decision.accepted_target_ids, decision.rejected_target_ids
    if (
        len(set(accepted)) != len(accepted)
        or len(set(rejected)) != len(rejected)
        or set(accepted) & set(rejected)
    ):
        raise ValueError("duplicate or conflicting review IDs")
    if not set(accepted + rejected) <= ids:
        raise ValueError("decision contains unknown target IDs")
    selected_ids = set(accepted) if accepted else ids - set(rejected)
    selected = [t for t in original if t.target_id in selected_ids]
    manual = []
    added_ids = set()
    for a in decision.additions:
        if not a.target_id or a.target_id in ids or a.target_id in added_ids:
            raise ValueError("manual target IDs must be nonempty and unique")
        if not a.reason.strip():
            raise ValueError("manual target requires a reason")
        if not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in (a.x_um, a.y_um)
        ):
            raise ValueError("manual target coordinates must be finite")
        if not region.contains(a.x_um, a.y_um):
            raise ValueError("manual target is outside permitted region")
        added_ids.add(a.target_id)
        selected.append(
            Target(
                a.target_id,
                context.specimen_id,
                context.coordinate_frame,
                context.calibration_id,
                float(a.x_um),
                float(a.y_um),
            )
        )
        manual.append(
            ManualProvenance(a.target_id, a.reason, float(a.x_um), float(a.y_um))
        )
    if (
        isinstance(max_targets, bool)
        or not isinstance(max_targets, int)
        or max_targets < 0
    ):
        raise ValueError("max_targets must be a nonnegative integer")
    if len(selected) > max_targets:
        raise ValueError("review result exceeds max_targets")
    return ReviewResult(tuple(selected), tuple(manual), manifest.sha256)


def _png_data_uri(image: Any) -> str:
    """Contrast-scale and encode a bounded ndarray preview as a PNG data URI."""
    array = np.asarray(image)
    if array.ndim not in (2, 3):
        raise ValueError("images must be HxW or HxWxC arrays")
    rows, cols = int(array.shape[0]), int(array.shape[1])
    channels = 1 if array.ndim == 2 else int(array.shape[2])
    if channels not in (1, 3, 4) or rows < 1 or cols < 1:
        raise ValueError("images must be nonempty grayscale, RGB or RGBA arrays")
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        scaled = np.zeros(array.shape, dtype=np.uint8)
    elif array.dtype == np.uint8:
        scaled = array
    else:
        low, high = np.percentile(finite.astype(np.float64), (1, 99))
        if high <= low:
            low, high = float(finite.min()), float(finite.max())
        if high <= low:
            scaled = np.zeros(array.shape, dtype=np.uint8)
        else:
            scaled = np.rint(
                np.clip(
                    (np.nan_to_num(array, nan=low) - low) * (255.0 / (high - low)),
                    0,
                    255,
                )
            ).astype(np.uint8)
    stride = max(1, math.ceil(max(rows, cols) / 512))
    scaled = scaled[::stride, ::stride]
    # ceil division: the slice keeps every stride-th row and column
    rows = (rows + stride - 1) // stride
    cols = (cols + stride - 1) // stride
    raw = bytearray()
    for y in range(rows):
        raw.append(0)
        for x in range(cols):
            pixel = scaled[y, x]
            if channels == 1:
                values = (pixel,)
            else:
                values = pixel
            raw.extend(max(0, min(255, int(v))) for v in values)
    color = {1: 0, 3: 2, 4: 6}[channels]

    def chunk(kind: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
        )

    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", cols, rows, 8, color, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes(raw)))
        + chunk(b"IEND", b"")
    )
    return "data:image/png;base64," + base64.b64encode(png).decode("ascii")


def render_review_html(
    path: str | Path,
    context: ReviewContext,
    targets: Iterable[Target],
    observations: Iterable[SearchObservation],
    image_loader: Callable[[SearchObservation], Any],
    calibration: Calibration,
    region: Region | None = None,
) -> ReviewManifest:
    """Write a self-contained offline review page. Loader only reads saved pixels."""
    targets, observations = tuple(targets), tuple(observations)
    manifest = create_manifest(context, targets)
    if (
        calibration.calibration_id != context.calibration_id
        or calibration.coordinate_frame != context.coordinate_frame
    ):
        raise ValueError("overlay calibration does not match review context")
    records = []
    tiles: list[tuple[SearchObservation, tuple[tuple[float, float], ...]]] = []
    for obs in observations:
        if obs.phase.value != "survey":
            continue
        if (obs.run_id, obs.specimen_id, obs.coordinate_frame, obs.calibration_id) != (
            context.run_id,
            context.specimen_id,
            context.coordinate_frame,
            context.calibration_id,
        ):
            raise ValueError("survey observation context does not match review context")
        if obs.missing:
            continue
        if obs.image_shape_yx:
            h, w = obs.image_shape_yx
            corners = tuple(
                calibration.image_to_stage(px, py, obs.stage_xy_um)
                for px, py in ((0, 0), (w - 1, 0), (w - 1, h - 1), (0, h - 1))
            )
            tiles.append((obs, corners))
        uri = _png_data_uri(image_loader(obs))
        linked = [
            t
            for t in targets
            if any(e.request_id == obs.request_id for e in t.observations)
        ]
        points = []
        for t in linked:
            h, w = obs.image_shape_yx or (0, 0)
            if h and w:
                a = calibration.pixel_to_stage
                det = a[0][0] * a[1][1] - a[0][1] * a[1][0]
                dx, dy = t.x_um - obs.stage_xy_um[0], t.y_um - obs.stage_xy_um[1]
                sx = (a[1][1] * dx - a[0][1] * dy) / det
                sy = (-a[1][0] * dx + a[0][0] * dy) / det
                px = (
                    sx
                    + calibration.reference_pixel[0]
                    - calibration.roi_origin[0]
                    - (calibration.binning[0] - 1) / 2
                ) / calibration.binning[0]
                py = (
                    sy
                    + calibration.reference_pixel[1]
                    - calibration.roi_origin[1]
                    - (calibration.binning[1] - 1) / 2
                ) / calibration.binning[1]
                if 0 <= px < w and 0 <= py < h:
                    points.append({"id": t.target_id, "x": px, "y": py})
        records.append(
            {
                "tile": obs.tile_id,
                "uri": uri,
                "width": (obs.image_shape_yx or (0, 0))[1],
                "height": (obs.image_shape_yx or (0, 0))[0],
                "points": points,
            }
        )
    extent_points = [p for _, corners in tiles for p in corners]
    extent_points.extend((t.x_um, t.y_um) for t in targets)
    if region is not None:
        extent_points.extend(
            ((region.x_min_um, region.y_min_um), (region.x_max_um, region.y_max_um))
        )
    overview = "<p>No survey tile dimensions were recorded; overview unavailable.</p>"
    if extent_points:
        xmin = min(p[0] for p in extent_points)
        xmax = max(p[0] for p in extent_points)
        ymin = min(p[1] for p in extent_points)
        ymax = max(p[1] for p in extent_points)
        padx, pady = max((xmax - xmin) * 0.05, 1.0), max((ymax - ymin) * 0.05, 1.0)
        xmin, xmax, ymin, ymax = xmin - padx, xmax + padx, ymin - pady, ymax + pady
        shapes = []
        for obs, corners in tiles:
            point_text = " ".join(f"{x:.6f},{-y:.6f}" for x, y in corners)
            shapes.append(
                f'<polygon points="{point_text}" fill="#aaccee" fill-opacity="0.20" stroke="#3572a5"><title>{html.escape(obs.tile_id)}</title></polygon>'
            )
        if region is not None:
            point_text = " ".join(
                f"{x:.6f},{-y:.6f}"
                for x, y in (
                    (region.x_min_um, region.y_min_um),
                    (region.x_max_um, region.y_min_um),
                    (region.x_max_um, region.y_max_um),
                    (region.x_min_um, region.y_max_um),
                )
            )
            shapes.append(
                f'<polygon points="{point_text}" fill="none" stroke="#d22" stroke-dasharray="4 2"><title>Permitted region</title></polygon>'
            )
        for t in targets:
            shapes.append(
                f'<circle cx="{t.x_um:.6f}" cy="{-t.y_um:.6f}" r="2" fill="#c22"><title>{html.escape(t.target_id)}: {t.x_um:.3f}, {t.y_um:.3f} um</title></circle>'
            )
        overview = f'<p>Survey overview, stage X/Y in micrometres (Y increases upward).</p><svg role="img" aria-label="Survey coverage and candidate target overview" viewBox="{xmin:.6f} {-ymax:.6f} {xmax - xmin:.6f} {ymax - ymin:.6f}">{"".join(shapes)}</svg>'
    safe_manifest = json.dumps(manifest.to_dict(), ensure_ascii=True).replace(
        "<", "\\u003c"
    )
    payload = json.dumps(records, ensure_ascii=True).replace("<", "\\u003c")
    doc = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sample target review</title>
  <style>
    body {{ font: 14px sans-serif; margin: 1.5em; color: #222; }}
    .tile {{ display: inline-block; vertical-align: top; margin: 8px; max-width: 320px; }}
    .tile img {{ display: block; max-width: 320px; height: auto; }}
    .frame {{ position: relative; }}
    .frame svg {{ position: absolute; inset: 0; width: 100%; height: 100%; }}
    #overview svg {{ display: block; position: static; width: min(80vw, 700px); height: auto; max-height: 500px; }}
    label {{ margin-right: 1em; }}
  </style>
</head>
<body>
<h1>Offline target review</h1>
<p>This saved-data review cannot move or control hardware.</p>
<p>Run {html.escape(context.run_id)} &middot; specimen {html.escape(context.specimen_id)} &middot; manifest {manifest.sha256}</p>
<div id="overview">
{overview}
</div>
<h2>Survey images</h2>
<div id="tiles"></div>
<h2>Manual target (physical coordinates, &micro;m)</h2>
<label>X <input id="x" type="number" step="any"></label>
<label>Y <input id="y" type="number" step="any"></label>
<label>Reason <input id="reason"></label>
<button onclick="addManual()">Add</button>
<p id="message" role="status"></p>
<ul id="manual"></ul>
<button onclick="download()">Export decision JSON</button>
<script type="application/json" id="manifest">{safe_manifest}</script>
<script>
const tiles={payload}, manifest=JSON.parse(document.querySelector('#manifest').textContent), selected=new Set(manifest.targets.map(t=>t.target_id)), rejected=new Set(), additions=[];
const host=document.querySelector('#tiles'); for(const tile of tiles){{const d=document.createElement('div'); d.className='tile'; d.innerHTML='<b></b><br><div class="frame"><img><svg></svg></div>'; d.querySelector('b').textContent=tile.tile; const im=d.querySelector('img'); im.src=tile.uri; im.alt='Saved survey '+tile.tile; const svg=d.querySelector('svg'); for(const p of tile.points){{const c=document.createElementNS('http://www.w3.org/2000/svg','circle');c.setAttribute('cx',p.x/tile.width*100+'%');c.setAttribute('cy',p.y/tile.height*100+'%');c.setAttribute('r','5');c.setAttribute('fill','none');c.setAttribute('stroke','lime');svg.append(c)}} host.append(d)}}
for(const t of manifest.targets){{const l=document.createElement('label');l.innerHTML='<input type="checkbox" checked> ';l.querySelector('input').onchange=e=>{{if(e.target.checked){{selected.add(t.target_id);rejected.delete(t.target_id)}}else{{selected.delete(t.target_id);rejected.add(t.target_id)}}}};l.append(document.createTextNode(t.target_id+' ('+t.x_um+', '+t.y_um+' \u00b5m)'));host.append(l,document.createElement('br'))}}
function addManual(){{const xs=document.querySelector('#x').value.trim(),ys=document.querySelector('#y').value.trim(),reason=document.querySelector('#reason').value.trim(),x=Number(xs),y=Number(ys),message=document.querySelector('#message');if(!xs||!ys||!Number.isFinite(x)||!Number.isFinite(y)||!reason){{message.textContent='Enter finite X and Y coordinates and a reason.';return}}const a={{target_id:'manual-'+(additions.length+1),x_um:x,y_um:y,reason}}; additions.push(a);const li=document.createElement('li');li.textContent=a.target_id+': '+a.x_um+', '+a.y_um+' \u00b5m — '+a.reason;document.querySelector('#manual').append(li);message.textContent='';document.querySelector('#x').value='';document.querySelector('#y').value='';document.querySelector('#reason').value=''}}
function download(){{const d=Object.assign({{manifest_sha256:manifest.sha256}},manifest.context,{{accepted_target_ids:Array.from(selected),rejected_target_ids:Array.from(rejected),additions}});const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([JSON.stringify(d,null,2)],{{type:'application/json'}}));a.download='review-decision.json';a.click();URL.revokeObjectURL(a.href)}}
</script>
</body>
</html>"""
    Path(path).write_text(doc, encoding="utf-8")
    return manifest
