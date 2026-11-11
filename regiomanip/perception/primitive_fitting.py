"""Primitive fitting helpers for RegioManip VLM3 lifting."""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


def unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(arr))
    if norm < 1e-8:
        return np.zeros((arr.shape[0],), dtype=np.float32)
    return (arr / norm).astype(np.float32)


def fit_line(points_xyz: np.ndarray) -> Optional[Dict[str, Any]]:
    if points_xyz.ndim != 2 or points_xyz.shape[0] < 8 or points_xyz.shape[1] != 3:
        return None
    center = np.mean(points_xyz, axis=0)
    q = points_xyz - center[None, :]
    cov = (q.T @ q) / max(1.0, float(len(q)))
    vals, vecs = np.linalg.eigh(cov)
    axis = unit(vecs[:, int(np.argmax(vals))])
    proj = q @ axis
    p1 = center + axis * float(np.min(proj))
    p2 = center + axis * float(np.max(proj))
    residual = np.linalg.norm(q - np.outer(proj, axis), axis=1)
    return {
        "point": center.astype(np.float32),
        "axis": axis.astype(np.float32),
        "endpoint_a": p1.astype(np.float32),
        "endpoint_b": p2.astype(np.float32),
        "residual_mean": float(np.mean(residual)),
        "residual_p90": float(np.percentile(residual, 90)),
        "num_points": int(points_xyz.shape[0]),
    }


def fit_plane(points_xyz: np.ndarray) -> Optional[Dict[str, Any]]:
    if points_xyz.ndim != 2 or points_xyz.shape[0] < 12 or points_xyz.shape[1] != 3:
        return None
    center = np.mean(points_xyz, axis=0)
    q = points_xyz - center[None, :]
    cov = (q.T @ q) / max(1.0, float(len(q)))
    vals, vecs = np.linalg.eigh(cov)
    normal = unit(vecs[:, int(np.argmin(vals))])
    residual = np.abs(q @ normal)
    tangent = unit(vecs[:, int(np.argmax(vals))])
    bitangent = unit(np.cross(normal, tangent))
    return {
        "point": center.astype(np.float32),
        "normal": normal.astype(np.float32),
        "tangent": tangent.astype(np.float32),
        "bitangent": bitangent.astype(np.float32),
        "residual_mean": float(np.mean(residual)),
        "residual_p90": float(np.percentile(residual, 90)),
        "num_points": int(points_xyz.shape[0]),
    }
