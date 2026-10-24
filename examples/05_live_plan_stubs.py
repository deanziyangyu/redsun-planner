"""Construct live plan stubs without creating a RunEngine or opening devices."""

from __future__ import annotations

from inspect import isgenerator
from typing import Any

from planner import live_median_scan, live_stream


class BufferStub:
    """Name-only stand-in for the live image buffer used in plan messages."""

    name = "demo-camera-buffer"


class FlyerStub:
    """Structural stand-in for a host-bound live flyer."""

    name = "demo-camera"
    buffer = BufferStub()

    def get_index(self) -> Any:
        """Return a Bluesky-readable placeholder."""
        return None


class AxisStub:
    """Name-only stand-in for a host-bound movable axis."""

    def __init__(self, name: str) -> None:
        self.name = name


class MotorStub:
    """Readable XY motor shape used to construct the median scan messages."""

    axis = {"x": AxisStub("demo-x"), "y": AxisStub("demo-y")}


detector = FlyerStub()
motor = MotorStub()

# The median scan's messages can be inspected without running them because no
# message return value controls the plan's Python flow.
median_messages = list(live_median_scan([detector], motor, 5.0, 2))
print("median scan message commands:", [message.command for message in median_messages])

# A live capture yields a status-dependent wait loop, so it must be submitted
# to the host's RunEngine to execute. Constructing the generator is still a
# pure plan-building step: it sends no messages and requires no engine.
capture_plan = live_stream([detector], "live_stream", parent="demo-parent-run")
assert isgenerator(capture_plan)
print("live capture stub constructed:", isgenerator(capture_plan))
