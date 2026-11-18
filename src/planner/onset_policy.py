"""Causal leading/trailing-band onset policy for stepped search-to-detail moves.

This module extracts the policy that the 2026-10-26 closed-loop and whole-strip
sessions executed from a runner script into a pure, versioned calculation. It
consumes one scalar density per declared band, decides whether the traverse is
approaching, sampling densely because a target was detected, or finished, and
returns one typed request per frame.

The policy emits requests only. It opens no device, reads no image and makes no
claim about what a stage or camera actually does with a request. A host slices
the rows :func:`band_rows` names, supplies their densities, and executes the
returned stepped increment or speed request under its own execution policy.

The two trigger modes are the two the sessions measured:

``leading_band``
    A direction-aware edge band carries the oncoming target. Its relative rise
    starts the dense stretch, and a rise on the trailing band - the target
    reaching the far edge - ends it. The bands are a matched pair, so this mode
    needs both densities.
``whole_frame``
    One whole-frame density carries the same events. A rise starts the dense
    stretch and a fall below the release threshold ends it. This is the
    direction-agnostic signal the whole-strip run used.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from math import isfinite
from typing import Literal

from .intensity_events import IntensityRiseConfig, IntensityRiseDetector

#: Identifier of the persisted policy/proposal contract.
ONSET_POLICY_SCHEMA = "planner-onset-policy-v1"

#: Increment this when a field changes meaning, not when a default is retuned.
ONSET_POLICY_VERSION = 1

OnsetDirection = Literal["increasing_x", "decreasing_x"]
"""Stage-x travel direction the bands are laid out for."""

OnsetTrigger = Literal["leading_band", "whole_frame"]
"""Which scalar signal starts the dense stretch."""

OnsetRelease = Literal["trailing_band", "signal_fall"]
"""Which event ends the dense stretch."""

OnsetState = Literal["approach", "dense", "exit"]
"""Current policy state for one traverse."""

OnsetMode = Literal["stepped", "continuous"]
"""Request form: one point increment, or one speed for a declared period."""

OnsetReason = Literal[
    "advance_coarse",
    "advance_dense",
    "hold_missing_observations",
    "blocked_missing_observations",
    "released_by_trailing_band",
    "released_by_signal_fall",
    "dense_budget_exhausted",
    "after_dense_budget_exhausted",
]
"""Machine-readable cause of the request in the same record."""


def _finite(value: object, name: str, *, positive: bool) -> float:
    """Return a finite real number, rejecting booleans and out-of-range values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a real number")
    number = float(value)
    if not isfinite(number):
        raise ValueError(f"{name} must be finite")
    if positive and number <= 0:
        raise ValueError(f"{name} must be positive")
    if not positive and number < 0:
        raise ValueError(f"{name} must be nonnegative")
    return number


def _positive_int(value: object, name: str) -> int:
    """Return a positive integer, rejecting booleans and floats."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class BandRows:
    """Half-open row ranges of the leading and trailing detector bands."""

    leading: tuple[int, int]
    trailing: tuple[int, int]

    def __post_init__(self) -> None:
        """Check both ranges are nonempty half-open spans."""
        for name in ("leading", "trailing"):
            rows = getattr(self, name)
            if (
                not isinstance(rows, tuple)
                or len(rows) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in rows
                )
            ):
                raise ValueError(f"{name} must be an integer (start, stop) pair")
            start, stop = rows
            if start < 0 or stop <= start:
                raise ValueError(f"{name} must be a nonempty nonnegative row span")


def band_rows(
    sensor_height_px: int,
    band_fraction: float,
    direction: OnsetDirection,
) -> BandRows:
    """Return the sensor rows that carry the leading and trailing bands.

    Content moves toward increasing image rows for increasing stage x, which the
    2026-10-26 strip measured as 5.341 +- 0.295 px per stage um along rows. The
    band that a target reaches first therefore sits at the low-row edge for
    ``increasing_x`` and at the high-row edge for ``decreasing_x``; the trailing
    band is the opposite edge. ``band_fraction`` is the band depth as a fraction
    of the sensor height, rounded to the nearest row and clamped to at least two
    rows. At a fraction of exactly ``0.5`` an odd-height sensor leaves one
    unclaimed centre row between the bands.

    Parameters
    ----------
    sensor_height_px
        Delivered frame height in pixels.
    band_fraction
        Band depth as a fraction of the sensor height, in ``(0, 0.5]``.
    direction
        Stage-x travel direction of the traverse.

    Returns
    -------
    BandRows
        Half-open ``(start, stop)`` row spans for both bands.
    """
    height = _positive_int(sensor_height_px, "sensor_height_px")
    fraction = _finite(band_fraction, "band_fraction", positive=True)
    if fraction > 0.5:
        raise ValueError("band_fraction must not exceed 0.5")
    if direction not in ("increasing_x", "decreasing_x"):
        raise ValueError("direction must be increasing_x or decreasing_x")
    depth = max(2, min(height, int(round(height * fraction))))
    if direction == "increasing_x":
        return BandRows(leading=(0, depth), trailing=(height - depth, height))
    return BandRows(leading=(height - depth, height), trailing=(0, depth))


@dataclass(frozen=True)
class OnsetPolicyConfig:
    """Declared geometry, event thresholds and step budget for one traverse.

    The policy identity is the serialized content of this object, so every
    threshold that can change a decision belongs here rather than in a host.
    ``sensor_height_px`` and ``band_fraction`` are required for
    ``trigger="leading_band"`` and unused for ``trigger="whole_frame"``.
    ``frame_period_s`` is required for ``mode="continuous"`` and unused for
    ``mode="stepped"``.
    """

    direction: OnsetDirection = "increasing_x"
    trigger: OnsetTrigger = "leading_band"
    release: OnsetRelease = "trailing_band"
    sensor_height_px: int | None = 1200
    band_fraction: float | None = 0.17
    rise: IntensityRiseConfig = IntensityRiseConfig()
    coarse_step_um: float = 10.0
    dense_step_um: float = 2.0
    dense_budget_um: float = 120.0
    max_missing_observations: int = 3
    mode: OnsetMode = "stepped"
    frame_period_s: float | None = None

    def __post_init__(self) -> None:
        """Validate the declared geometry, thresholds and request form."""
        if self.direction not in ("increasing_x", "decreasing_x"):
            raise ValueError("direction must be increasing_x or decreasing_x")
        if self.trigger not in ("leading_band", "whole_frame"):
            raise ValueError("trigger must be leading_band or whole_frame")
        if self.release not in ("trailing_band", "signal_fall"):
            raise ValueError("release must be trailing_band or signal_fall")
        expected_release = (
            "trailing_band" if self.trigger == "leading_band" else "signal_fall"
        )
        if self.release != expected_release:
            raise ValueError(
                f"trigger {self.trigger!r} requires release {expected_release!r}: a "
                "leading-edge band cannot report that a target has crossed the field, "
                "and a whole-frame rise cannot report the target leaving one edge"
            )
        if self.trigger == "leading_band":
            if self.sensor_height_px is None or self.band_fraction is None:
                raise ValueError(
                    "sensor_height_px and band_fraction are required for "
                    "trigger='leading_band'"
                )
            band_rows(self.sensor_height_px, self.band_fraction, self.direction)
        if not isinstance(self.rise, IntensityRiseConfig):
            raise TypeError("rise must be an IntensityRiseConfig")
        object.__setattr__(
            self,
            "coarse_step_um",
            _finite(self.coarse_step_um, "coarse_step_um", positive=True),
        )
        object.__setattr__(
            self,
            "dense_step_um",
            _finite(self.dense_step_um, "dense_step_um", positive=True),
        )
        if self.dense_step_um > self.coarse_step_um:
            raise ValueError("dense_step_um must not exceed coarse_step_um")
        object.__setattr__(
            self,
            "dense_budget_um",
            _finite(self.dense_budget_um, "dense_budget_um", positive=False),
        )
        object.__setattr__(
            self,
            "max_missing_observations",
            _positive_int(self.max_missing_observations, "max_missing_observations"),
        )
        if self.mode not in ("stepped", "continuous"):
            raise ValueError("mode must be stepped or continuous")
        if self.mode == "continuous":
            if self.frame_period_s is None:
                raise ValueError("frame_period_s is required for mode='continuous'")
            object.__setattr__(
                self,
                "frame_period_s",
                _finite(self.frame_period_s, "frame_period_s", positive=True),
            )
        elif self.frame_period_s is not None:
            raise ValueError("frame_period_s belongs to mode='continuous'")

    @property
    def band_rows(self) -> BandRows:
        """Return the band rows this policy uses, refusing a whole-frame config."""
        if (
            self.trigger != "leading_band"
            or self.sensor_height_px is None
            or self.band_fraction is None
        ):
            raise ValueError("this config uses no band geometry")
        return band_rows(self.sensor_height_px, self.band_fraction, self.direction)

    def as_dict(self) -> dict[str, object]:
        """Return the JSON-safe policy snapshot a host records with a run."""
        return {
            "schema": ONSET_POLICY_SCHEMA,
            "version": ONSET_POLICY_VERSION,
            "direction": self.direction,
            "trigger": self.trigger,
            "release": self.release,
            "sensor_height_px": self.sensor_height_px,
            "band_fraction": self.band_fraction,
            "band_rows": None
            if self.trigger != "leading_band"
            else {
                "leading": list(self.band_rows.leading),
                "trailing": list(self.band_rows.trailing),
            },
            "rise": {
                "baseline_window_frames": self.rise.baseline_window_frames,
                "baseline_min_samples": self.rise.baseline_min_samples,
                "persistence_frames": self.rise.persistence_frames,
                "release_persistence_frames": self.rise.release_persistence_frames,
                "enter_relative_rise": self.rise.enter_relative_rise,
                "exit_relative_rise": self.rise.exit_relative_rise,
                "minimum_absolute_rise": self.rise.minimum_absolute_rise,
                "baseline_floor": self.rise.baseline_floor,
            },
            "coarse_step_um": self.coarse_step_um,
            "dense_step_um": self.dense_step_um,
            "dense_budget_um": self.dense_budget_um,
            "max_missing_observations": self.max_missing_observations,
            "mode": self.mode,
            "frame_period_s": self.frame_period_s,
        }

    @property
    def policy_id(self) -> str:
        """Return a deterministic identifier derived from the full config."""
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
        return f"onset-v{ONSET_POLICY_VERSION}-{digest}"


@dataclass(frozen=True)
class OnsetObservation:
    """One delivered frame's scalar band densities.

    ``frame_density`` is the leading-band density for
    ``trigger="leading_band"`` and the whole-frame density for
    ``trigger="whole_frame"``. ``trailing_density`` is required for the
    dual-band release. ``None`` means the host could not measure that signal;
    the policy then holds instead of advancing on missing evidence.
    """

    frame_index: int
    frame_density: float | None
    trailing_density: float | None = None

    def __post_init__(self) -> None:
        """Check the frame index and every supplied density."""
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a nonnegative integer")
        for name in ("frame_density", "trailing_density"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _finite(value, name, positive=False))


@dataclass(frozen=True)
class OnsetProposal:
    """One typed policy request with the evidence that selected it.

    ``mode`` decides which request field is populated. A ``stepped`` proposal
    carries ``step_um`` for one point increment and no speed; a ``continuous``
    proposal carries ``speed_um_s`` for the declared frame period and no step.
    A zero step (or speed) is a hold for a missing observation, not a command to
    move zero distance at full speed.
    """

    policy_id: str
    frame_index: int
    state: OnsetState
    reason: OnsetReason
    mode: OnsetMode
    direction: OnsetDirection
    step_um: float | None
    speed_um_s: float | None
    onset: bool
    release: bool
    budget_exhausted: bool
    dense_used_um: float
    dense_budget_um: float
    dense_remaining_um: float
    relative_rise: float | None
    trailing_relative_rise: float | None

    def __post_init__(self) -> None:
        """Check that exactly the request field this mode names is populated."""
        if not isinstance(self.policy_id, str) or not self.policy_id:
            raise ValueError("policy_id is required")
        if (
            isinstance(self.frame_index, bool)
            or not isinstance(self.frame_index, int)
            or self.frame_index < 0
        ):
            raise ValueError("frame_index must be a nonnegative integer")
        if self.state not in ("approach", "dense", "exit"):
            raise ValueError("state must be approach, dense or exit")
        if self.mode not in ("stepped", "continuous"):
            raise ValueError("mode must be stepped or continuous")
        if self.mode == "stepped":
            if self.step_um is None or self.speed_um_s is not None:
                raise ValueError("a stepped proposal carries step_um and no speed_um_s")
            object.__setattr__(
                self, "step_um", _finite(self.step_um, "step_um", positive=False)
            )
        else:
            if self.speed_um_s is None or self.step_um is not None:
                raise ValueError(
                    "a continuous proposal carries speed_um_s and no step_um"
                )
            object.__setattr__(
                self,
                "speed_um_s",
                _finite(self.speed_um_s, "speed_um_s", positive=False),
            )
        for name in (
            "dense_used_um",
            "dense_budget_um",
            "dense_remaining_um",
        ):
            object.__setattr__(
                self, name, _finite(getattr(self, name), name, positive=False)
            )
        if self.dense_used_um > self.dense_budget_um + 1e-9:
            raise ValueError("dense_used_um cannot exceed the declared dense budget")
        if (
            abs(self.dense_remaining_um - (self.dense_budget_um - self.dense_used_um))
            > 1e-9
        ):
            raise ValueError("dense_remaining_um must equal budget minus used distance")
        for name in ("relative_rise", "trailing_relative_rise"):
            value = getattr(self, name)
            if value is not None and not isfinite(value):
                raise ValueError(f"{name} must be finite when present")

    def to_dict(self) -> dict[str, object]:
        """Return the JSON-safe proposal a host writes into its run record."""
        return {
            "schema": ONSET_POLICY_SCHEMA,
            "version": ONSET_POLICY_VERSION,
            "policy_id": self.policy_id,
            "frame_index": self.frame_index,
            "state": self.state,
            "reason": self.reason,
            "mode": self.mode,
            "direction": self.direction,
            "step_um": self.step_um,
            "speed_um_s": self.speed_um_s,
            "onset": self.onset,
            "release": self.release,
            "budget_exhausted": self.budget_exhausted,
            "dense_used_um": self.dense_used_um,
            "dense_budget_um": self.dense_budget_um,
            "dense_remaining_um": self.dense_remaining_um,
            "relative_rise": self.relative_rise,
            "trailing_relative_rise": self.trailing_relative_rise,
        }


class OnsetPolicy:
    """Turn per-frame band densities into causal attack/dense/finish requests.

    The instance is scoped to one traverse and one camera/band epoch. Both
    detectors are updated on every measured frame, so a band that has already
    carried one target clears itself and can report the next one. The dense
    budget is cumulative for the traverse: once it is spent, the policy stops
    requesting dense frames and stays in ``exit``, which is what bounds a large
    bright region instead of braking across it indefinitely.

    Missing observations never advance the traverse. The first misses hold with
    ``hold_missing_observations``; after ``max_missing_observations`` consecutive
    misses the reason becomes ``blocked_missing_observations``. Either way the
    request is a zero step (or zero speed), and the host decides whether to stop
    the run.
    """

    def __init__(self, config: OnsetPolicyConfig | None = None) -> None:
        """Create a policy for one traverse and one declared band geometry."""
        self._config = OnsetPolicyConfig() if config is None else config
        if not isinstance(self._config, OnsetPolicyConfig):
            raise TypeError("config must be an OnsetPolicyConfig")
        self._onset_detector = IntensityRiseDetector(self._config.rise)
        self._trailing_detector: IntensityRiseDetector | None = (
            IntensityRiseDetector(self._config.rise)
            if self._config.trigger == "leading_band"
            else None
        )
        self._state: OnsetState = "approach"
        self._dense_used_um = 0.0
        self._missing = 0
        self._last_frame_index = -1

    @property
    def config(self) -> OnsetPolicyConfig:
        """Return the immutable policy configuration."""
        return self._config

    @property
    def policy_id(self) -> str:
        """Return the deterministic identifier of this policy configuration."""
        return self._config.policy_id

    @property
    def state(self) -> OnsetState:
        """Return the current policy state."""
        return self._state

    @property
    def dense_used_um(self) -> float:
        """Return the dense distance already requested on this traverse."""
        return self._dense_used_um

    @property
    def dense_remaining_um(self) -> float:
        """Return the dense distance still budgeted for this traverse."""
        return max(0.0, self._config.dense_budget_um - self._dense_used_um)

    @property
    def band_rows(self) -> BandRows:
        """Return the declared band rows for the host to slice."""
        return self._config.band_rows

    def update(self, observation: OnsetObservation) -> OnsetProposal:
        """Consume one frame's densities and return the request for the next move.

        Observations must arrive with strictly increasing ``frame_index`` so a
        replayed or duplicated frame cannot silently change the state machine.
        """
        if not isinstance(observation, OnsetObservation):
            raise TypeError("observation must be an OnsetObservation")
        if observation.frame_index <= self._last_frame_index:
            raise ValueError(
                "observations must arrive in strictly increasing frame order"
            )
        self._last_frame_index = observation.frame_index
        reason: OnsetReason
        needs_trailing = self._config.trigger == "leading_band"
        if observation.frame_density is None or (
            needs_trailing and observation.trailing_density is None
        ):
            self._missing += 1
            reason = (
                "hold_missing_observations"
                if self._missing <= self._config.max_missing_observations
                else "blocked_missing_observations"
            )
            return self._proposal(
                observation.frame_index,
                step_um=0.0,
                state=self._state,
                reason=reason,
                onset=False,
                release=False,
                budget_exhausted=False,
                relative_rise=None,
                trailing_relative_rise=None,
            )
        self._missing = 0
        onset_sample = self._onset_detector.update(observation.frame_density)
        trailing_sample = (
            self._trailing_detector.update(observation.trailing_density)
            if self._trailing_detector is not None
            and observation.trailing_density is not None
            else None
        )
        trailing_rise = (
            None if trailing_sample is None else trailing_sample.relative_rise
        )
        release = False
        budget_exhausted = False
        if self._state == "exit":
            step_um = self._config.coarse_step_um
            reason = "after_dense_budget_exhausted"
        elif self._state == "approach":
            if onset_sample.onset and (
                self._config.dense_step_um <= self._config.dense_budget_um + 1e-9
            ):
                self._state = "dense"
                self._dense_used_um += self._config.dense_step_um
                step_um = self._config.dense_step_um
                reason = "advance_dense"
            elif onset_sample.onset:
                self._state = "exit"
                step_um = self._config.coarse_step_um
                reason = "dense_budget_exhausted"
                budget_exhausted = True
            else:
                step_um = self._config.coarse_step_um
                reason = "advance_coarse"
        elif self._config.release == "trailing_band":
            release = bool(trailing_sample is not None and trailing_sample.onset)
            if release:
                self._state = "approach"
                step_um = self._config.coarse_step_um
                reason = "released_by_trailing_band"
            elif (
                self._dense_used_um + self._config.dense_step_um
                > self._config.dense_budget_um + 1e-9
            ):
                self._state = "exit"
                step_um = self._config.coarse_step_um
                reason = "dense_budget_exhausted"
                budget_exhausted = True
            else:
                self._dense_used_um += self._config.dense_step_um
                step_um = self._config.dense_step_um
                reason = "advance_dense"
        else:
            release = onset_sample.release
            if release:
                self._state = "approach"
                step_um = self._config.coarse_step_um
                reason = "released_by_signal_fall"
            elif (
                self._dense_used_um + self._config.dense_step_um
                > self._config.dense_budget_um + 1e-9
            ):
                self._state = "exit"
                step_um = self._config.coarse_step_um
                reason = "dense_budget_exhausted"
                budget_exhausted = True
            else:
                self._dense_used_um += self._config.dense_step_um
                step_um = self._config.dense_step_um
                reason = "advance_dense"
        return self._proposal(
            observation.frame_index,
            step_um=step_um,
            state=self._state,
            reason=reason,
            onset=onset_sample.onset,
            release=release,
            budget_exhausted=budget_exhausted,
            relative_rise=onset_sample.relative_rise,
            trailing_relative_rise=trailing_rise,
        )

    def _proposal(
        self,
        frame_index: int,
        *,
        step_um: float,
        state: OnsetState,
        reason: OnsetReason,
        onset: bool,
        release: bool,
        budget_exhausted: bool,
        relative_rise: float | None,
        trailing_relative_rise: float | None,
    ) -> OnsetProposal:
        """Assemble a proposal in the configured request form."""
        if self._config.mode == "stepped":
            return OnsetProposal(
                policy_id=self.policy_id,
                frame_index=frame_index,
                state=state,
                reason=reason,
                mode="stepped",
                direction=self._config.direction,
                step_um=step_um,
                speed_um_s=None,
                onset=onset,
                release=release,
                budget_exhausted=budget_exhausted,
                dense_used_um=self._dense_used_um,
                dense_budget_um=self._config.dense_budget_um,
                dense_remaining_um=self.dense_remaining_um,
                relative_rise=relative_rise,
                trailing_relative_rise=trailing_relative_rise,
            )
        assert self._config.frame_period_s is not None  # validated in the config
        return OnsetProposal(
            policy_id=self.policy_id,
            frame_index=frame_index,
            state=state,
            reason=reason,
            mode="continuous",
            direction=self._config.direction,
            step_um=None,
            speed_um_s=step_um / self._config.frame_period_s,
            onset=onset,
            release=release,
            budget_exhausted=budget_exhausted,
            dense_used_um=self._dense_used_um,
            dense_budget_um=self._config.dense_budget_um,
            dense_remaining_um=self.dense_remaining_um,
            relative_rise=relative_rise,
            trailing_relative_rise=trailing_relative_rise,
        )


__all__ = [
    "ONSET_POLICY_SCHEMA",
    "ONSET_POLICY_VERSION",
    "BandRows",
    "OnsetDirection",
    "OnsetMode",
    "OnsetObservation",
    "OnsetPolicy",
    "OnsetPolicyConfig",
    "OnsetProposal",
    "OnsetReason",
    "OnsetRelease",
    "OnsetState",
    "OnsetTrigger",
    "band_rows",
]
