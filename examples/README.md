# Examples

Runnable snippets that need only `numpy` and this package. None of them imports a
Bluesky RunEngine, opens a device, or writes a file, so you can run them straight
from a checkout:

```bash
PYTHONPATH=src python examples/01_recipe_and_tiles.py
```

| Script | Shows |
|---|---|
| [`01_recipe_and_tiles.py`](01_recipe_and_tiles.py) | build a grid recipe, tile a region in serpentine order, map image pixels to stage micrometres |
| [`02_detect_and_place.py`](02_detect_and_place.py) | detect bright objects in a synthetic frame and place them on the stage through a calibration |
| [`03_hand_eye_fit.py`](03_hand_eye_fit.py) | fit the stage-to-camera transform from measured pairs and run the quality gates |

## Where the framework boundary is

The package holds three kinds of thing, and only one of them is Bluesky-specific:

- **Calculations and typed data.** Recipes, tiles, detections, targets, hand-eye
  fits, record documents. Plain `numpy` over dataclasses. No framework.
- **Persisted documents.** Written and read by `planner.records` and
  `planner.replay`. No framework, apart from the `Reading` type the replay
  adapter reuses.
- **Plan generators.** `planner.plans` composes Bluesky plan stubs, and
  `planner.protocols` refines Bluesky device protocols. These are the only parts
  tied to Bluesky.

So a host that drives something else, for example Micro-Manager through
Pycro-Manager, can use the first two directly:

1. Build a recipe or a bounded plan geometry.
2. Take its positions and hand them to your own stage and camera API.
3. Turn each captured frame into detections and place them with a `Calibration`.
4. Persist the run and its review metadata with `planner.records`.

If you also want the plan generators, implement the small structural protocols in
`planner.protocols` (`LiveFlyer`, `XYMotor`) on your own objects and let a
Bluesky `RunEngine` execute the plan; that is how a device adapter is normally
built. You do not have to adopt Bluesky elsewhere in the host.

## Note on the dependency

`bluesky` is a hard dependency of the distribution, and `import planner` imports
it because `planner/__init__.py` re-exports the plan generators. A host that only
wants the calculations still has Bluesky installed, though it never calls it.
