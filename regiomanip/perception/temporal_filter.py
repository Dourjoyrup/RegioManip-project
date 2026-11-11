"""Temporal smoothing filter for 3-D grasp pose estimates across video frames.

Usage (video loop)
------------------
    from regiomanip.perception.temporal_filter import TemporalGraspFilter

    filt = TemporalGraspFilter(alpha=0.30)

    for frame in camera_stream():
        result = run_region_level_affordance_from_sam(...)
        for obj_result in result["results"]:
            filt.update_from_decision(obj_result)

        if filt.is_warm():
            send_to_robot(filt.grip_point_3d, filt.contact_normal_3d)

Design
------
* grip_point_3d and contact_point_3d live in R³ — standard exponential
  moving averages.
* contact_normal_3d and axis_direction_3d live on S² (unit sphere) — naive
  linear blending then re-normalising introduces bias when vectors are far
  apart.  SLERP (spherical linear interpolation) is used instead, which
  moves along the great-circle arc at constant angular velocity and exactly
  preserves the unit-norm constraint.
* alpha is the weight of the *new* observation: 0 → ignore new data,
  1 → always use latest.  Values in [0.2, 0.4] work well for typical
  30 Hz manipulation cameras.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# SLERP helper
# ---------------------------------------------------------------------------

def _slerp_unit(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    """Spherical linear interpolation between two unit vectors on S².

    Handles the antipodal case (dot < 0) by flipping b to take the shorter
    arc.  Falls back to normalised linear interpolation when the vectors are
    nearly parallel to avoid division by sin(0).

    Parameters
    ----------
    a, b : unit vectors in R³ (callers must normalise before calling)
    t    : interpolation parameter in [0, 1] (0 → a, 1 → b)

    Returns
    -------
    Unit vector in R³ along the great-circle arc from a to b at parameter t.
    """
    dot = float(np.clip(np.dot(a, b), -1.0, 1.0))
    if dot < 0.0:       # take the short arc across the equator
        b = -b
        dot = -dot
    if dot > 1.0 - 1e-7:   # nearly parallel — linear fallback
        result = (1.0 - t) * a + t * b
        n = float(np.linalg.norm(result))
        return result / n if n > 1e-8 else a.copy()
    omega = math.acos(dot)
    sin_omega = math.sin(omega)
    result = (math.sin((1.0 - t) * omega) / sin_omega) * a + \
             (math.sin(t * omega) / sin_omega) * b
    n = float(np.linalg.norm(result))
    return result / n if n > 1e-8 else a.copy()


# ---------------------------------------------------------------------------
# Filter
# ---------------------------------------------------------------------------

class TemporalGraspFilter:
    """Exponential smoothing filter for 3-D grasp pose estimates across frames.

    Parameters
    ----------
    alpha : float, default 0.30
        Weight of the new observation per update call.
        Effective time constant (in frames) ≈ 1 / alpha.
        * 0.10 — heavy smoothing, ~10-frame lag
        * 0.30 — moderate (recommended for 30 Hz)
        * 0.50 — light smoothing, ~2-frame lag

    Attributes updated per call
    ---------------------------
    grip_point_3d      : R³ position, exponential moving average
    contact_normal_3d  : S² unit vector, SLERP
    axis_direction_3d  : S² unit vector, SLERP
    """

    def __init__(self, alpha: float = 0.30) -> None:
        if not (0.0 < alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1]; got {alpha}")
        self.alpha: float = float(alpha)
        self._grip: Optional[np.ndarray] = None
        self._contact: Optional[np.ndarray] = None
        self._normal: Optional[np.ndarray] = None
        self._axis: Optional[np.ndarray] = None
        self._n_updates: int = 0
        self._track_id: Optional[str] = None

    # ------------------------------------------------------------------
    # Public update interface
    # ------------------------------------------------------------------

    def update(
        self,
        grip_point_3d: Optional[List[float]],
        contact_point_3d: Optional[List[float]] = None,
        contact_normal_3d: Optional[List[float]] = None,
        axis_direction_3d: Optional[List[float]] = None,
    ) -> None:
        """Incorporate one new frame observation.

        Any of the three vectors may be None (e.g. depth unavailable for
        this frame); only non-None values update the running estimate.
        """
        if grip_point_3d is not None:
            g = np.array(grip_point_3d, dtype=np.float64)
            if self._grip is None:
                self._grip = g.copy()
            else:
                self._grip = (1.0 - self.alpha) * self._grip + self.alpha * g

        if contact_point_3d is not None:
            c = np.array(contact_point_3d, dtype=np.float64)
            if self._contact is None:
                self._contact = c.copy()
            else:
                self._contact = (1.0 - self.alpha) * self._contact + self.alpha * c

        if contact_normal_3d is not None:
            n = np.array(contact_normal_3d, dtype=np.float64)
            nn = float(np.linalg.norm(n))
            if nn > 1e-8:
                n /= nn
                if self._normal is None:
                    self._normal = n.copy()
                else:
                    self._normal = _slerp_unit(self._normal, n, self.alpha)

        if axis_direction_3d is not None:
            a = np.array(axis_direction_3d, dtype=np.float64)
            an = float(np.linalg.norm(a))
            if an > 1e-8:
                a /= an
                if self._axis is None:
                    self._axis = a.copy()
                else:
                    self._axis = _slerp_unit(self._axis, a, self.alpha)

        self._n_updates += 1

    def update_from_decision(self, decision: Any) -> None:
        """Update from a HandleDecision dataclass instance or its dict equivalent.

        Accepts either a HandleDecision object (has attributes) or the dict
        returned by ``asdict(decision)`` / ``result["results"][i]``.
        """
        if hasattr(decision, "__dataclass_fields__"):
            d: Dict[str, Any] = {
                "grip_point_3d":     getattr(decision, "grip_point_3d", None),
                "contact_point_3d":  getattr(decision, "work_end_contact_3d", None),
                "contact_normal_3d": getattr(decision, "contact_normal_3d", None),
                "axis_direction_3d": getattr(decision, "axis_direction_3d", None),
            }
        elif isinstance(decision, dict):
            d = decision
        else:
            return
        self.update(
            grip_point_3d=d.get("grip_point_3d"),
            contact_point_3d=d.get("contact_point_3d", d.get("work_end_contact_3d")),
            contact_normal_3d=d.get("contact_normal_3d"),
            axis_direction_3d=d.get("axis_direction_3d"),
        )

    # ------------------------------------------------------------------
    # Read-out properties
    # ------------------------------------------------------------------

    @property
    def grip_point_3d(self) -> Optional[List[float]]:
        return self._grip.tolist() if self._grip is not None else None

    @property
    def contact_normal_3d(self) -> Optional[List[float]]:
        return self._normal.tolist() if self._normal is not None else None

    @property
    def contact_point_3d(self) -> Optional[List[float]]:
        return self._contact.tolist() if self._contact is not None else None

    @property
    def axis_direction_3d(self) -> Optional[List[float]]:
        return self._axis.tolist() if self._axis is not None else None

    @property
    def n_updates(self) -> int:
        return self._n_updates

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def is_warm(self, min_updates: int = 3) -> bool:
        """True when enough frames have been processed to trust the estimate.

        The first observation is always the raw value (alpha=1 for init);
        subsequent frames blend it in.  Three updates is the minimum for the
        smoother to have had any meaningful effect.
        """
        return self._n_updates >= min_updates

    def reset(self) -> None:
        """Clear the running estimate (e.g. when tracking a new object)."""
        self._grip = None
        self._contact = None
        self._normal = None
        self._axis = None
        self._n_updates = 0

    def prepare_track(self, track_id: str) -> bool:
        """Select a tool-target track, resetting state when identity changes.

        Returns True when a reset was required.
        """
        normalized = str(track_id).strip()
        changed = self._track_id is not None and self._track_id != normalized
        if changed:
            self.reset()
        self._track_id = normalized
        return changed

    def as_dict(self) -> Dict[str, Any]:
        """Return the current smoothed estimate as a plain dict."""
        return {
            "grip_point_3d":     self.grip_point_3d,
            "contact_point_3d":  self.contact_point_3d,
            "contact_normal_3d": self.contact_normal_3d,
            "axis_direction_3d": self.axis_direction_3d,
            "n_updates":         self._n_updates,
            "alpha":             self.alpha,
            "track_id":          self._track_id,
        }

    def __repr__(self) -> str:
        return (
            f"TemporalGraspFilter(alpha={self.alpha}, n_updates={self._n_updates}, "
            f"warm={self.is_warm()})"
        )
