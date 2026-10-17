"""Typed source-frame records and inline readables for saved-data replay plans."""

from __future__ import annotations

import time
from dataclasses import dataclass
from math import isfinite
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from bluesky.protocols import Reading
    from numpy.typing import NDArray

SourceScalar = str | int | float | bool


@dataclass(frozen=True)
class ReplayFrame:
    """One source observation plus image arrays and source-coordinate fields.

    ``source_time_s`` is expressed in the time origin declared by the source
    metadata. It is not the RunEngine replay event time. Image data are emitted
    inline one frame at a time, so sources need not materialize a full stack.
    """

    source_frame_id: str
    source_time_s: float
    images: Mapping[str, NDArray[Any]]
    source_context: Mapping[str, SourceScalar]

    def __post_init__(self) -> None:
        """Reject ambiguous IDs, times, image keys and coordinate provenance."""
        if not isinstance(self.source_frame_id, str) or not self.source_frame_id:
            raise ValueError("source_frame_id must be a nonempty string")
        if (
            isinstance(self.source_time_s, bool)
            or not isinstance(self.source_time_s, (int, float))
            or not isfinite(self.source_time_s)
        ):
            raise ValueError("source_time_s must be finite")
        if not self.images:
            raise ValueError("each replay frame must contain at least one image")
        reserved = {"source_frame_id", "source_time_s"}
        for key, value in self.images.items():
            if not isinstance(key, str) or not key or key.startswith("source_"):
                raise ValueError(
                    "image keys must be nonempty and not start with source_"
                )
            if key in reserved or not isinstance(value, np.ndarray) or value.ndim < 2:
                raise ValueError(f"image {key!r} must be a 2D-or-higher NumPy array")
        for context_key, context_value in self.source_context.items():
            # Widen to object: the annotation promises a scalar, but this guard
            # exists precisely for records that arrived untyped from JSON.
            raw: object = context_value
            if (
                not isinstance(context_key, str)
                or not context_key.startswith("source_")
                or context_key in reserved
                or isinstance(raw, (dict, list, tuple))
                or not isinstance(raw, (str, int, float, bool))
                or (isinstance(raw, float) and not isfinite(raw))
            ):
                raise ValueError(
                    "source_context keys must start with source_ and map to "
                    "finite scalar values"
                )


class ReplaySource(Protocol):
    """Structural base for streaming source readers used by replay plans."""

    @property
    def source_id(self) -> str:
        """Stable identifier for the saved dataset or selected data region."""
        ...

    @property
    def source_metadata(self) -> Mapping[str, Any]:
        """JSON-compatible source identity, timing origin and selection data."""
        ...

    def iter_frames(self) -> Iterator[ReplayFrame]:
        """Yield source frames in the order the replay plan should emit them."""
        ...


class _ReplayFrameReadable:
    """Mutable one-frame reading whose Bluesky descriptor remains stable."""

    def __init__(self, first: ReplayFrame) -> None:
        self.name = "dataset-replay"
        self.parent = None
        self._image_keys = tuple(first.images)
        self._context_keys = tuple(first.source_context)
        self._shapes = {key: first.images[key].shape for key in self._image_keys}
        self._dtypes = {key: first.images[key].dtype.str for key in self._image_keys}
        self._context_types = {
            key: type(first.source_context[key]) for key in self._context_keys
        }
        self._frame = first

    def update(self, frame: ReplayFrame) -> None:
        """Replace payload after checking the fixed event descriptor contract."""
        if tuple(frame.images) != self._image_keys:
            raise ValueError("image keys changed within a replay source")
        if tuple(frame.source_context) != self._context_keys:
            raise ValueError("source context keys changed within a replay source")
        for key in self._image_keys:
            image = frame.images[key]
            if image.shape != self._shapes[key] or image.dtype.str != self._dtypes[key]:
                raise ValueError(f"image {key!r} shape or dtype changed during replay")
        for key in self._context_keys:
            if type(frame.source_context[key]) is not self._context_types[key]:
                raise ValueError(f"source context field {key!r} changed type")
        self._frame = frame

    def describe(self) -> dict[str, dict[str, Any]]:
        """Describe the stable frame arrays and source provenance fields.

        Every field except the frame identifier carries a ``units`` entry, so a
        host can render the engineering unit the way a RedSun view does
        (``descriptor.get("units", "")``). Only the time field has a unit this
        package can know; an arbitrary ``source_*`` scalar or an image array
        declares an empty unit rather than guessing one. ``source_frame_id`` is
        an identifier, so it declares none.
        """
        described: dict[str, dict[str, Any]] = {
            "source_frame_id": {
                "source": self.name,
                "dtype": "string",
                "shape": [],
            },
            "source_time_s": {
                "source": self.name,
                "dtype": "number",
                "units": "s",
                "shape": [],
            },
        }
        for key in self._image_keys:
            described[key] = {
                "source": self.name,
                "dtype": "array",
                "dtype_numpy": self._dtypes[key],
                "units": "",
                "shape": list(self._shapes[key]),
            }
        for key in self._context_keys:
            described[key] = {
                "source": self.name,
                "dtype": _source_dtype(self._context_types[key]),
                "units": "",
                "shape": [],
            }
        return described

    def read(self) -> dict[str, Reading[Any]]:
        """Return one frame with replay read time and original source time."""
        timestamp = time.time()
        data: dict[str, Any] = {
            "source_frame_id": self._frame.source_frame_id,
            "source_time_s": float(self._frame.source_time_s),
            **self._frame.images,
            **self._frame.source_context,
        }
        return {
            key: {"value": value, "timestamp": timestamp} for key, value in data.items()
        }


def _source_dtype(value_type: type[Any]) -> str:
    """Map supported scalar provenance values to Bluesky schema names."""
    if value_type is bool:
        return "boolean"
    if value_type is int:
        return "integer"
    if value_type is float:
        return "number"
    if value_type is str:
        return "string"
    raise TypeError(f"unsupported source context type: {value_type!r}")


__all__ = ["ReplayFrame", "ReplaySource"]
