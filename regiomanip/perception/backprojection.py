"""Backprojection helpers for RegioManip VLM3 lifting."""

from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np


def nearest_valid_depth(depth: np.ndarray, xy: Tuple[int, int], radius: int = 8) -> Optional[float]:
    x, y = int(xy[0]), int(xy[1])
    h, w = depth.shape[:2]
    best: Optional[float] = None
    best_d2 = 1e18
    for yy in range(max(0, y - radius), min(h, y + radius + 1)):
        for xx in range(max(0, x - radius), min(w, x + radius + 1)):
            dv = float(depth[yy, xx])
            if not np.isfinite(dv) or dv <= 0:
                continue
            d2 = float((xx - x) ** 2 + (yy - y) ** 2)
            if d2 < best_d2:
                best_d2 = d2
                best = dv
    return best


def backproject_xy(
    xy: Tuple[int, int],
    depth: np.ndarray,
    intr: Dict[str, float],
    depth_scale: float,
) -> Optional[np.ndarray]:
    d_raw = nearest_valid_depth(depth, xy)
    if d_raw is None:
        return None
    z = float(d_raw) * float(depth_scale)
    if z <= 0:
        return None
    x = (float(xy[0]) - intr["cx"]) * z / intr["fx"]
    y = (float(xy[1]) - intr["cy"]) * z / intr["fy"]
    return np.array([x, y, z], dtype=np.float32)


def backproject_mask(
    mask01: np.ndarray,
    depth: np.ndarray,
    intr: Dict[str, float],
    depth_scale: float,
    max_points: int = 20000,
) -> np.ndarray:
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return np.zeros((0, 3), dtype=np.float32)
    step = max(1, int(math.ceil(xs.size / float(max_points))))
    xs = xs[::step]
    ys = ys[::step]
    d = depth[ys, xs].astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if int(valid.sum()) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    xs = xs[valid].astype(np.float32)
    ys = ys[valid].astype(np.float32)
    z = d[valid] * float(depth_scale)
    x = (xs - float(intr["cx"])) * z / float(intr["fx"])
    y = (ys - float(intr["cy"])) * z / float(intr["fy"])
    return np.stack([x, y, z], axis=1).astype(np.float32)


def contact_edge_band_mask(
    mask01: np.ndarray,
    endpoint_a_xy: Tuple[float, float],
    endpoint_b_xy: Tuple[float, float],
    half_width_px: float = 6.0,
) -> np.ndarray:
    """Return mask pixels in a narrow band around a selected contact edge.

    Fitting a line to an entire blade or scraper face measures the width of
    the part rather than the quality of its terminal contact edge.  This band
    intersects the grounded work-end mask with the finite 2-D edge segment
    selected by the region policy.
    """
    mask = np.asarray(mask01) > 0
    if mask.ndim != 2:
        raise ValueError(f"mask01 must be 2-D; got shape {mask.shape}")
    p0 = np.asarray(endpoint_a_xy, dtype=np.float32).reshape(2)
    p1 = np.asarray(endpoint_b_xy, dtype=np.float32).reshape(2)
    segment = p1 - p0
    length2 = float(np.dot(segment, segment))
    if length2 < 1e-8:
        return np.zeros_like(mask, dtype=np.uint8)
    yy, xx = np.indices(mask.shape, dtype=np.float32)
    pixels = np.stack([xx, yy], axis=-1)
    position = np.clip(np.sum((pixels - p0) * segment, axis=-1) / length2, 0.0, 1.0)
    projection = p0 + position[..., None] * segment
    distance = np.linalg.norm(pixels - projection, axis=-1)
    return (mask & (distance <= float(half_width_px))).astype(np.uint8)
