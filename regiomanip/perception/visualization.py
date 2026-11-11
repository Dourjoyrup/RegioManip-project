"""Visualization helpers for RegioManip VLM3 lifting."""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from regiomanip.region_grounding.utils import ensure_dir


def project_xyz(xyz: np.ndarray, intr: Dict[str, float]) -> Optional[Tuple[int, int]]:
    p = np.asarray(xyz, dtype=np.float32).reshape(3)
    z = float(p[2])
    if not np.isfinite(z) or z <= 1e-8:
        return None
    x = int(round(float((p[0] * intr["fx"] / z) + intr["cx"])))
    y = int(round(float((p[1] * intr["fy"] / z) + intr["cy"])))
    return x, y


def draw_projected_segment(
    img: np.ndarray,
    p1_xyz: np.ndarray,
    p2_xyz: np.ndarray,
    intr: Dict[str, float],
    color: Tuple[int, int, int],
    thickness: int,
) -> None:
    a = project_xyz(p1_xyz, intr)
    b = project_xyz(p2_xyz, intr)
    if a is None or b is None:
        return
    cv2.line(img, a, b, color, thickness, cv2.LINE_AA)


def write_visual_overlay(
    out_dir: str,
    full_bgr: np.ndarray,
    intr: Dict[str, float],
    best_hypothesis: Optional[Dict[str, Any]],
    tool_primitives: List[Dict[str, Any]],
    target_primitives: List[Dict[str, Any]],
    frame_bundle: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    if best_hypothesis is None or full_bgr is None:
        return None
    vis = full_bgr.copy()
    tool = next((t for t in tool_primitives if str(t.get("name")) == str(best_hypothesis.get("tool_name"))), None)
    target = next((t for t in target_primitives if str(t.get("name")) == str(best_hypothesis.get("target_name"))), None)
    if tool is None or target is None:
        return None

    # work_edge and edge_line segments intentionally not drawn

    grip_xy = project_xyz(np.array(tool.get("grip_xyz", []), dtype=np.float32), intr)
    tool_touch_xy = project_xyz(np.array(tool.get("touch_xyz", []), dtype=np.float32), intr)
    target_touch_xy = project_xyz(np.array(target.get("touch_xyz", []), dtype=np.float32), intr)
    if grip_xy is not None:
        cv2.circle(vis, grip_xy, 6, (0, 255, 255), -1, lineType=cv2.LINE_AA)
        cv2.putText(vis, "grip", (grip_xy[0] + 6, max(16, grip_xy[1] - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2, cv2.LINE_AA)
    if tool_touch_xy is not None:
        cv2.circle(vis, tool_touch_xy, 5, (255, 255, 0), -1, lineType=cv2.LINE_AA)
    if target_touch_xy is not None:
        cv2.circle(vis, target_touch_xy, 5, (0, 165, 255), -1, lineType=cv2.LINE_AA)
    if tool_touch_xy is not None and target_touch_xy is not None:
        cv2.line(vis, tool_touch_xy, target_touch_xy, (255, 255, 0), 1, cv2.LINE_AA)

    def draw_frame(T: np.ndarray, prefix: str, axis_len: float, origin_color: Tuple[int, int, int]) -> None:
        if T.shape != (4, 4):
            return
        t_xyz = np.array(T[:3, 3], dtype=np.float32)
        R = np.array(T[:3, :3], dtype=np.float32)
        origin_xy = project_xyz(t_xyz, intr)
        if origin_xy is None:
            return
        cv2.circle(vis, origin_xy, 4, origin_color, -1, lineType=cv2.LINE_AA)
        axes = [
            (R[:, 0] * axis_len, (0, 0, 255), f"x{prefix}"),
            (R[:, 1] * axis_len, (0, 255, 0), f"y{prefix}"),
            (R[:, 2] * axis_len, (255, 0, 0), f"z{prefix}"),
        ]
        for axis_vec, color, label in axes:
            end_xy = project_xyz(t_xyz + axis_vec.astype(np.float32), intr)
            if end_xy is None:
                continue
            cv2.line(vis, origin_xy, end_xy, color, 2, cv2.LINE_AA)
            cv2.putText(vis, label, (end_xy[0] + 4, max(14, end_xy[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    if isinstance(frame_bundle, dict):
        T_cam_T = np.array(frame_bundle.get("T_cam_T", []), dtype=np.float32)
        T_cam_P = np.array(frame_bundle.get("T_cam_P", []), dtype=np.float32)
        draw_frame(T_cam_T, prefix="T", axis_len=0.035, origin_color=(255, 255, 0))
        draw_frame(T_cam_P, prefix="P", axis_len=0.040, origin_color=(255, 255, 255))
        # +zP (blue, solid) is the outward target normal.  The contact
        # approach is a separate vector -zP, shown dashed in magenta.
        if T_cam_P.shape == (4, 4):
            origin_xyz = np.asarray(T_cam_P[:3, 3], dtype=np.float32)
            approach = np.asarray(frame_bundle.get("approach_direction_cam", []), dtype=np.float32)
            if approach.shape == (3,) and float(np.linalg.norm(approach)) > 1e-8:
                approach = approach / float(np.linalg.norm(approach))
                start_xy = project_xyz(origin_xyz, intr)
                end_xy = project_xyz(origin_xyz + 0.035 * approach, intr)
                if start_xy is not None and end_xy is not None:
                    for index in range(0, 10, 2):
                        alpha0 = index / 10.0
                        alpha1 = (index + 1) / 10.0
                        p0 = (
                            int(round(start_xy[0] + alpha0 * (end_xy[0] - start_xy[0]))),
                            int(round(start_xy[1] + alpha0 * (end_xy[1] - start_xy[1]))),
                        )
                        p1 = (
                            int(round(start_xy[0] + alpha1 * (end_xy[0] - start_xy[0]))),
                            int(round(start_xy[1] + alpha1 * (end_xy[1] - start_xy[1]))),
                        )
                        cv2.line(vis, p0, p1, (255, 0, 255), 2, cv2.LINE_AA)
                    cv2.putText(vis, "approach=-zP", (end_xy[0] + 4, max(14, end_xy[1] - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 0, 255), 1, cv2.LINE_AA)

    cv2.putText(
        vis,
        f"frame: {best_hypothesis.get('tool_name')} -> {best_hypothesis.get('target_name')} score={float(best_hypothesis.get('score', 0.0)):.3f}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    overlay_dir = os.path.join(out_dir, "interaction_lift")
    ensure_dir(overlay_dir)
    overlay_path = os.path.join(overlay_dir, "best_frame_overlay.png")
    cv2.imwrite(overlay_path, vis)
    return overlay_path
