import base64
import io
import json
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pytest

from planner.contracts import Phase, SearchObservation, Target
from planner.geometry import Calibration, Region
from planner.review import (
    ManualTarget,
    ReviewContext,
    _png_data_uri,
    apply_review,
    create_manifest,
    decision_from_json,
    make_decision,
    render_review_html,
)

Image = pytest.importorskip("PIL.Image", reason="test images are written with Pillow")

CTX = ReviewContext("run-1", "specimen-1", "stage-xy", "cal-1")
REGION = Region(0, 0, 100, 100)
NOW = datetime.now(timezone.utc)


def candidate(target_id: str = "t1", x: float = 20, y: float = 30) -> Target:
    return Target(
        target_id, CTX.specimen_id, CTX.coordinate_frame, CTX.calibration_id, x, y
    )


def test_review_accepts_rejects_adds_and_roundtrips(tmp_path: Path) -> None:
    items = (candidate(), candidate("t2", 50, 50))
    manifest = create_manifest(CTX, items)
    decision = make_decision(
        manifest,
        accepted_target_ids=("t1",),
        rejected_target_ids=("t2",),
        additions=(ManualTarget("m1", 10, 12, "visible feature"),),
    )
    restored = decision_from_json(json.dumps(decision.to_dict()))
    result = apply_review(restored, items, context=CTX, region=REGION, max_targets=2)
    assert [t.target_id for t in result.targets] == ["t1", "m1"]
    assert result.manual_provenance[0].reason == "visible feature"
    decision_path = tmp_path / "decision.json"
    decision_path.write_text(json.dumps(decision.to_dict()), encoding="utf-8")
    assert decision_from_json(str(decision_path)) == decision


@pytest.mark.parametrize(
    "addition",
    [
        ManualTarget("m", 101, 2, "outside"),
        ManualTarget("m", float("nan"), 2, "bad"),
        ManualTarget("t1", 2, 2, "duplicate"),
        ManualTarget("m", 2, 2, " "),
    ],
)
def test_invalid_manual_targets_rejected(addition: ManualTarget) -> None:
    items = (candidate(),)
    d = make_decision(create_manifest(CTX, items), additions=(addition,))
    with pytest.raises(ValueError):
        apply_review(d, items, context=CTX, region=REGION, max_targets=3)


def test_stale_manifest_context_tampering_and_limit_rejected() -> None:
    items = (candidate(),)
    d = make_decision(create_manifest(CTX, items), accepted_target_ids=("t1",))
    with pytest.raises(ValueError):
        apply_review(d, (candidate(x=21),), context=CTX, region=REGION, max_targets=4)
    with pytest.raises(ValueError):
        apply_review(
            d,
            items,
            context=ReviewContext(
                "other", CTX.specimen_id, CTX.coordinate_frame, CTX.calibration_id
            ),
            region=REGION,
            max_targets=4,
        )
    d2 = make_decision(
        create_manifest(CTX, items), additions=(ManualTarget("m", 2, 2, "manual"),)
    )
    with pytest.raises(ValueError):
        apply_review(d2, items, context=CTX, region=REGION, max_targets=1)


def test_duplicate_decision_ids_rejected() -> None:
    items = (candidate(),)
    d = make_decision(create_manifest(CTX, items), accepted_target_ids=("t1", "t1"))
    with pytest.raises(ValueError):
        apply_review(d, items, context=CTX, region=REGION, max_targets=4)


def test_html_escapes_metadata_and_renders_saved_thumbnail(tmp_path: Path) -> None:
    obs = SearchObservation(
        CTX.specimen_id,
        Phase.SURVEY,
        CTX.coordinate_frame,
        CTX.calibration_id,
        NOW,
        NOW,
        (50, 50),
        "tile-1",
        "req-1",
        CTX.run_id,
        data_reference="</script><img src=x>",
        image_shape_yx=(4, 4),
    )
    t = Target(
        "t1", CTX.specimen_id, CTX.coordinate_frame, CTX.calibration_id, 50, 50, (obs,)
    )
    cal = Calibration("cal-1", "stage-xy", ((1, 0), (0, 1)), (0, 0), (0, 0))
    path = tmp_path / "review.html"
    render_review_html(
        path, CTX, (t,), (obs,), lambda _: np.ones((4, 4), dtype=np.uint8), cal
    )
    page = path.read_text()
    assert "data:image/png;base64," in page
    assert "\\u003c/script>" in page
    assert '"x":50.0' not in page  # point coordinates are image-space zero-based
    assert "l.querySelector('input').onchange" in page
    assert "Number.isFinite(x)" in page
    assert "cannot move or control hardware" in page
    assert ".frame svg { position: absolute;" in page
    assert "#overview svg { display: block; position: static;" in page
    assert "svg{position:absolute" not in page
    assert '<div id="overview">' in page


def test_png_preserves_uint16_contrast_and_bounds_preview_size() -> None:
    image = np.zeros((700, 900), dtype=np.uint16)
    image[100, 100] = 300
    image[600, 800] = 5000
    uri = _png_data_uri(image)
    decoded = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert decoded.size == (450, 350)
    pixels = np.asarray(decoded)
    assert pixels.max() == 255
    assert pixels.min() < pixels.max()


def test_png_handles_nonfinite_and_constant_images() -> None:
    uri = _png_data_uri(np.full((2, 3), np.nan, dtype=np.float32))
    decoded = Image.open(io.BytesIO(base64.b64decode(uri.split(",", 1)[1])))
    assert np.asarray(decoded).max() == 0
    constant = Image.open(
        io.BytesIO(
            base64.b64decode(
                _png_data_uri(np.ones((2, 3), dtype=np.uint16)).split(",", 1)[1]
            )
        )
    )
    assert np.asarray(constant).max() == 0


def test_renderer_rejects_observation_context_before_loading_image(
    tmp_path: Path,
) -> None:
    obs = SearchObservation(
        CTX.specimen_id,
        Phase.SURVEY,
        CTX.coordinate_frame,
        "wrong-cal",
        NOW,
        NOW,
        (50, 50),
        "tile-1",
        "req-1",
        CTX.run_id,
        image_shape_yx=(4, 4),
    )
    cal = Calibration("cal-1", "stage-xy", ((1, 0), (0, 1)), (0, 0), (0, 0))
    loaded = []
    with pytest.raises(ValueError, match="observation context"):
        render_review_html(
            tmp_path / "review.html", CTX, (), (obs,), lambda o: loaded.append(o), cal
        )
    assert loaded == []


def test_overview_uses_calibrated_footprints_and_physical_targets(
    tmp_path: Path,
) -> None:
    calibration = Calibration(
        "cal-1",
        "stage-xy",
        ((0, -2), (3, 0)),
        (10, 20),
        (0, 0),
        roi_origin=(4, 5),
        binning=(2, 1),
    )
    obs1 = SearchObservation(
        CTX.specimen_id,
        Phase.SURVEY,
        CTX.coordinate_frame,
        CTX.calibration_id,
        NOW,
        NOW,
        (100, 200),
        "tile-a",
        "req-a",
        CTX.run_id,
        image_shape_yx=(4, 6),
    )
    obs2 = SearchObservation(
        CTX.specimen_id,
        Phase.SURVEY,
        CTX.coordinate_frame,
        CTX.calibration_id,
        NOW,
        NOW,
        (120, 220),
        "tile-b",
        "req-b",
        CTX.run_id,
        image_shape_yx=(4, 6),
    )
    target = Target(
        "mapped-target",
        CTX.specimen_id,
        CTX.coordinate_frame,
        CTX.calibration_id,
        100,
        200,
        (obs1,),
    )
    path = tmp_path / "overview.html"
    render_review_html(
        path,
        CTX,
        (target,),
        (obs1, obs2),
        lambda _: np.ones((4, 6), dtype=np.uint8),
        calibration,
        Region(90, 190, 130, 230),
    )
    page = path.read_text(encoding="utf-8")
    svg_match = re.search(r"(<svg[\s\S]*?</svg>)", page)
    assert svg_match is not None
    svg = ET.fromstring(svg_match.group(1))
    polygons = [child for child in svg if child.tag.endswith("polygon")]
    first = tuple(
        tuple(map(float, pair.split(",")))
        for pair in polygons[0].attrib["points"].split()
    )
    expected = tuple(
        (x, -y)
        for x, y in (
            calibration.image_to_stage(px, py, obs1.stage_xy_um)
            for px, py in ((0, 0), (5, 0), (5, 3), (0, 3))
        )
    )
    assert np.allclose(first, expected)
    assert len([child for child in svg if child.tag.endswith("circle")]) == 1
    assert "Permitted region" in page


def test_generated_javascript_parses_in_qt_engine(tmp_path: Path) -> None:
    qtcore = pytest.importorskip("PyQt6.QtCore")
    qtqml = pytest.importorskip("PyQt6.QtQml")
    app = qtcore.QCoreApplication.instance() or qtcore.QCoreApplication([])
    assert app is not None
    cal = Calibration("cal-1", "stage-xy", ((1, 0), (0, 1)), (0, 0), (0, 0))
    path = tmp_path / "review.html"
    render_review_html(path, CTX, (), (), lambda _: None, cal)
    page = path.read_text(encoding="utf-8")
    script_match = re.search(r"<script>([\s\S]*?)</script>", page)
    assert script_match is not None
    script = script_match.group(1)
    engine = qtqml.QJSEngine()
    result = engine.evaluate("new Function(" + json.dumps(script) + ")")
    assert not result.isError(), result.toString()


def test_generated_javascript_review_controls_work_with_dom_stub(
    tmp_path: Path,
) -> None:
    qtcore = pytest.importorskip("PyQt6.QtCore")
    qtqml = pytest.importorskip("PyQt6.QtQml")
    app = qtcore.QCoreApplication.instance() or qtcore.QCoreApplication([])
    assert app is not None
    cal = Calibration("cal-1", "stage-xy", ((1, 0), (0, 1)), (0, 0), (0, 0))
    target = candidate()
    path = tmp_path / "review.html"
    manifest = render_review_html(path, CTX, (target,), (), lambda _: None, cal)
    page = path.read_text(encoding="utf-8")
    script_match = re.search(r"<script>([\s\S]*?)</script>", page)
    assert script_match is not None
    script = script_match.group(1)
    manifest_json = json.dumps(manifest.to_dict()).replace("<", "\\u003c")
    stub = f"""this.handlers=[]; this.fields={{'#x':{{value:''}},'#y':{{value:''}},'#reason':{{value:''}},'#message':{{textContent:''}}}}; this.downloaded='';
this.document={{querySelector:s=>s==='#manifest'?{{textContent:{json.dumps(manifest_json)}}}:s==='#tiles'?{{append:()=>{{}}}}:s==='#manual'?{{append:()=>{{}}}}:fields[s],createElement:n=>({{set innerHTML(v){{}},querySelector:()=>({{set onchange(fn){{handlers.push(fn)}}}}),append:()=>{{}},click:()=>{{}}}}),createTextNode:()=>({{}})}};
this.Blob=class{{constructor(parts){{downloaded=parts[0]}}}};this.URL={{createObjectURL:()=>'',revokeObjectURL:()=>{{}}}};"""
    engine = qtqml.QJSEngine()
    result = engine.evaluate(
        stub
        + "\n"
        + script
        + "\n"
        + "handlers[0]({target:{checked:false}});addManual();"
        + "const blankMessage=fields['#message'].textContent;"
        + "fields['#x'].value='10';fields['#y'].value='20';fields['#reason'].value='annotated';addManual();download();"
        + "JSON.stringify({blankMessage,downloaded})"
    )
    assert not result.isError(), result.toString()
    data = json.loads(result.toString())
    assert "Enter finite" in data["blankMessage"]
    decision = json.loads(data["downloaded"])
    assert decision["accepted_target_ids"] == []
    assert decision["rejected_target_ids"] == [target.target_id]
    assert decision["additions"] == [
        {"target_id": "manual-1", "x_um": 10, "y_um": 20, "reason": "annotated"}
    ]
