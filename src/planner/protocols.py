"""Structural contracts shared by planner-level scan plans."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from bluesky.protocols import (
    Collectable,
    Flyable,
    Movable,
    Preparable,
    Readable,
    WritesStreamAssets,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


@runtime_checkable
class LiveFlyer(Preparable, Flyable, Collectable, WritesStreamAssets, Protocol):
    """A Bluesky flyer that also exposes its live image buffer to a viewer."""

    @property
    def name(self) -> str:
        """Return the flyer's name."""
        ...

    @property
    def buffer(self) -> Any:
        """Return the live image buffer a viewer reads."""
        ...

    def get_index(self) -> Any:
        """Return the index of the newest frame in the buffer."""
        ...


@runtime_checkable
class XYMotor(Readable[Any], Protocol):
    """Readable Bluesky device exposing separately movable X and Y axes."""

    @property
    def axis(self) -> Mapping[str, Movable[Any]]:
        """Return the separately movable X and Y axes, keyed by name."""
        ...


__all__ = ["LiveFlyer", "XYMotor"]
