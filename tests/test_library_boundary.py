import subprocess
import sys
from pathlib import Path

import pytest

from planner import (
    AcquisitionSettings,
    ResourceRequirement,
    ScanPosition,
    StageScanRecipe,
    build_grid_recipe,
)


def test_library_import_and_recipe_compilation_need_no_acquisition_host(
    tmp_path: Path,
) -> None:
    library = str(Path(__file__).resolve().parents[1] / "src")
    code = f"""
import importlib.abc
import sys
class RejectHost(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {{"ix83_search_host", "ix83_acquisition", "pycromanager", "rtaspt", "napari", "redsun"}}:
            raise RuntimeError("library imported host/framework: " + fullname)
sys.meta_path.insert(0, RejectHost())
sys.path.insert(0, {library!r})
import planner
recipe = planner.build_grid_recipe((0, 0), specimen_id="seeded", rows=2, columns=3, step_xy_um=(25, 25))
assert len(recipe.positions) == 6
assert not hasattr(planner, "SampleSearchRunner")
assert not hasattr(planner, "StageScanRunner")
print("recipe compiled without host")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", code],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "recipe compiled without host" in result.stdout


def test_recipe_declares_capabilities_and_resolved_positions() -> None:
    recipe = build_grid_recipe(
        (10, 20),
        specimen_id="s",
        rows=2,
        columns=2,
        step_xy_um=(5, 7),
        settings=AcquisitionSettings(frames=3),
    )
    assert recipe.positions[-1].x_um == 15 and recipe.positions[-1].y_um == 27
    assert "camera.sequence" in recipe.required_capabilities
    assert recipe.settings.exposure_ms == 50


def test_invalid_published_positions_are_rejected() -> None:
    with pytest.raises(ValueError, match="unique"):
        StageScanRecipe(
            "scan", "s", (ScanPosition("same", 0, 0), ScanPosition("same", 10, 10))
        )
    with pytest.raises(ValueError, match="finite"):
        StageScanRecipe("scan", "s", (ScanPosition("invalid", float("nan"), 0),))


def test_serial_recipes_reject_unsupported_shared_resources() -> None:
    with pytest.raises(ValueError, match="exclusive"):
        ResourceRequirement("camera", mode="shared")
