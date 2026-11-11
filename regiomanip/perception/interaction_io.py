"""I/O and parsing helpers for RegioManip interaction lift."""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np

from regiomanip.region_grounding.utils import load_json
from .primitive_fitting import unit


def parse_vec3(text: Optional[str]) -> Optional[np.ndarray]:
    s = str(text or "").strip()
    if not s:
        return None
    parts = [p.strip() for p in s.replace(";", ",").split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated values, got: {s}")
    return unit(np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=np.float32))


def load_transform_json(path: str) -> np.ndarray:
    obj = load_json(path)
    candidate = None
    if isinstance(obj, dict):
        for key in ("T_world_cam", "T_cam_world", "transform", "matrix", "extrinsics"):
            value = obj.get(key)
            arr = np.array(value, dtype=np.float32) if value is not None else None
            if arr is not None and arr.shape == (4, 4):
                candidate = arr
                break
    if candidate is None:
        arr = np.array(obj, dtype=np.float32)
        if arr.shape == (4, 4):
            candidate = arr
    if candidate is None:
        raise ValueError(f"Could not parse 4x4 transform from {path}")
    return candidate.astype(np.float32)


def resolve_world_up_cam(
    world_up_cam_text: Optional[str],
    world_up_world_text: Optional[str],
    t_world_cam_json: Optional[str],
) -> Tuple[Optional[np.ndarray], str]:
    direct = parse_vec3(world_up_cam_text)
    if direct is not None:
        return direct.astype(np.float32), "world_up_cam"

    world_up_world = parse_vec3(world_up_world_text)
    if world_up_world is None:
        return None, "image_up_fallback"

    if not t_world_cam_json:
        return None, "image_up_fallback"

    T_world_cam = load_transform_json(t_world_cam_json)
    R_world_cam = np.array(T_world_cam[:3, :3], dtype=np.float32)
    world_up_cam = unit(R_world_cam.T @ world_up_world.astype(np.float32))
    return world_up_cam.astype(np.float32), "world_up_world_via_T_world_cam"


def find_results_json(out_dir: str) -> str:
    candidates = [
        os.path.join(out_dir, "region_grounding", "region_grounding_results.json"),
        os.path.join(out_dir, "vlm2", "vlm2_results.json"),
    ]
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Cannot locate region grounding results in {out_dir}")


def load_depth(depth_path: str) -> np.ndarray:
    depth = np.load(depth_path).astype(np.float32)
    depth[~np.isfinite(depth)] = 0.0
    return depth


def resolve_intrinsics(
    image_shape: Tuple[int, int],
    fx: Optional[float],
    fy: Optional[float],
    cx: Optional[float],
    cy: Optional[float],
) -> Dict[str, float]:
    h, w = int(image_shape[0]), int(image_shape[1])
    return {
        "fx": float(fx if fx is not None else 615.0),
        "fy": float(fy if fy is not None else 615.0),
        "cx": float(cx if cx is not None else (w * 0.5)),
        "cy": float(cy if cy is not None else (h * 0.5)),
    }


def load_vlm1_objects(results_json_from_vlm1: str) -> Dict[str, Dict[str, Any]]:
    obj = load_json(results_json_from_vlm1)
    return {str(o.get("name", "")).strip(): o for o in obj.get("objects", []) if str(o.get("name", "")).strip()}


def part_mask_map(vlm1_obj: Dict[str, Any]) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for part in vlm1_obj.get("parts", []):
        name = str(part.get("part", "")).strip()
        path = str(part.get("mask_path", "")).strip()
        if name and path:
            out[name] = path
    return out


def part_readiness(vlm1_obj: Dict[str, Any], part_name: str) -> float:
    quality = vlm1_obj.get("quality_control", {})
    readiness = quality.get("lift_readiness", {}).get(part_name, {})
    return float(readiness.get("score", 0.0) or 0.0)
