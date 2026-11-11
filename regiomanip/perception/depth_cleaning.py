"""Conservative, training-free cleaning for masked RGB-D point clouds.

The cleaner is deliberately a preprocessing safeguard, not a new geometric
model.  A cleaned fit is used only when it remains consistent with the raw fit
and does not make its robust residual worse.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.spatial import cKDTree


FitFn = Callable[[np.ndarray], Optional[Dict[str, Any]]]


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 180.0
    # Line axes and unoriented plane normals are sign-equivalent here.
    cosine = float(np.clip(abs(np.dot(a, b)) / denom, 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def statistical_inlier_mask(
    points_xyz: np.ndarray,
    *,
    neighbors: int = 12,
    mad_scale: float = 3.5,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Return a robust k-nearest-neighbour inlier mask.

    Median/MAD thresholding adapts to the local point spacing and requires no
    CAD model, learned prior, or fixed sensor-specific distance threshold.
    """
    points = np.asarray(points_xyz, dtype=np.float32)
    finite = np.all(np.isfinite(points), axis=1) if points.ndim == 2 else np.zeros(0, dtype=bool)
    mask = finite.copy()
    valid = points[finite]
    report: Dict[str, Any] = {
        "method": "knn_median_mad",
        "neighbors": 0,
        "median_neighbor_distance_m": None,
        "robust_sigma_m": None,
        "threshold_m": None,
    }
    if valid.shape[0] < 4:
        return mask, report

    k = min(max(2, int(neighbors)), valid.shape[0] - 1)
    distances, _ = cKDTree(valid).query(valid, k=k + 1)
    local_distance = np.mean(distances[:, 1:], axis=1)
    median = float(np.median(local_distance))
    mad = float(np.median(np.abs(local_distance - median)))
    robust_sigma = 1.4826 * mad
    # Avoid a zero-MAD threshold on repeated/quantized depth samples.
    threshold = median + float(mad_scale) * max(robust_sigma, 1e-6)
    valid_keep = local_distance <= threshold
    mask[np.flatnonzero(finite)] = valid_keep
    report.update(
        {
            "neighbors": int(k),
            "median_neighbor_distance_m": median,
            "robust_sigma_m": robust_sigma,
            "threshold_m": float(threshold),
        }
    )
    return mask, report


def clean_and_fit(
    points_xyz: np.ndarray,
    fit_fn: FitFn,
    *,
    primitive: str,
    role: str,
    min_points: int,
    min_retained_ratio: float = 0.55,
    max_orientation_change_deg: float = 20.0,
    max_residual_ratio: float = 1.05,
    neighbors: int = 12,
    mad_scale: float = 3.5,
) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]], np.ndarray, Dict[str, Any]]:
    """Fit raw and cleaned clouds, conservatively selecting the accepted fit."""
    points = np.asarray(points_xyz, dtype=np.float32)
    raw_fit = fit_fn(points)
    inliers, stats = statistical_inlier_mask(points, neighbors=neighbors, mad_scale=mad_scale)
    cleaned = points[inliers] if points.ndim == 2 else np.empty((0, 3), dtype=np.float32)
    clean_fit = fit_fn(cleaned)
    raw_count = int(points.shape[0]) if points.ndim == 2 else 0
    retained_ratio = float(cleaned.shape[0] / raw_count) if raw_count else 0.0

    orientation_key = "axis" if primitive == "line" else "normal"
    orientation_change = None
    if raw_fit is not None and clean_fit is not None:
        orientation_change = _angle_deg(raw_fit[orientation_key], clean_fit[orientation_key])

    reasons = []
    if raw_fit is None:
        reasons.append("raw_fit_unavailable")
    if cleaned.shape[0] < int(min_points) or clean_fit is None:
        reasons.append("too_few_clean_points")
    if retained_ratio < float(min_retained_ratio):
        reasons.append("too_many_points_removed")
    if raw_fit is not None and clean_fit is not None:
        raw_p90 = float(raw_fit["residual_p90"])
        clean_p90 = float(clean_fit["residual_p90"])
        if clean_p90 > max(raw_p90 * float(max_residual_ratio), raw_p90 + 1e-6):
            reasons.append("residual_worsened")
        strong_consensus_recovery = bool(
            retained_ratio >= 0.90
            and raw_p90 > 1e-6
            and clean_p90 <= 0.25 * raw_p90
        )
        if (
            orientation_change is not None
            and orientation_change > float(max_orientation_change_deg)
            and not strong_consensus_recovery
        ):
            reasons.append("orientation_changed")
    else:
        strong_consensus_recovery = False

    accepted_cleaning = raw_fit is not None and clean_fit is not None and not reasons
    accepted_fit = clean_fit if accepted_cleaning else raw_fit
    report: Dict[str, Any] = {
        "enabled": True,
        "role": role,
        "primitive": primitive,
        "accepted_cleaning": bool(accepted_cleaning),
        "fallback_to_raw": bool(not accepted_cleaning),
        "fallback_reasons": reasons,
        "raw_num_points": raw_count,
        "clean_num_points": int(cleaned.shape[0]),
        "removed_num_points": int(max(0, raw_count - cleaned.shape[0])),
        "retained_ratio": retained_ratio,
        "orientation_change_deg": orientation_change,
        "orientation_safeguard_overridden_by_strong_consensus": bool(strong_consensus_recovery),
        "raw_residual_p90_m": None if raw_fit is None else float(raw_fit["residual_p90"]),
        "clean_residual_p90_m": None if clean_fit is None else float(clean_fit["residual_p90"]),
        "accepted_residual_p90_m": None if accepted_fit is None else float(accepted_fit["residual_p90"]),
        "safeguards": {
            "min_points": int(min_points),
            "min_retained_ratio": float(min_retained_ratio),
            "max_orientation_change_deg": float(max_orientation_change_deg),
            "max_residual_ratio": float(max_residual_ratio),
        },
        "outlier_statistics": stats,
    }
    return accepted_fit, raw_fit, cleaned, report
