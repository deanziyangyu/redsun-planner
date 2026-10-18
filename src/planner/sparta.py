"""SPARTA: Spatiotemporal Pareto Acquisition-Regime Trajectory Allocation.

SPARTA is the conceptual name for the optimization framework over the
acquisition space. It is not a vocabulary users must learn before calling the
planner: the public API keeps the plain names ``AcquisitionPoint``,
``AcquisitionEnvelope``, ``OperatingTrace`` and ``SearchUtility``. This module
exists so the framework name has one stable code location without prefixing
ordinary types or workflows.
"""

from __future__ import annotations

FRAMEWORK_NAME = "SPARTA"
FRAMEWORK_TITLE = "Spatiotemporal Pareto Acquisition-Regime Trajectory Allocation"

__all__ = ["FRAMEWORK_NAME", "FRAMEWORK_TITLE"]
