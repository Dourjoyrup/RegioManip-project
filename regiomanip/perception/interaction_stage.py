#!/usr/bin/env python3
"""Interaction lift stage runner for RegioManip."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from regiomanip.region_grounding.utils import ensure_dir, load_json, load_mask01, save_json, to_builtin
from .backprojection import backproject_mask as _backproject_mask
from .backprojection import backproject_xy as _backproject_xy
from .backprojection import contact_edge_band_mask as _contact_edge_band_mask
from .frame_hypotheses import build_tool_target_frames as _build_tool_target_frames
from .frame_hypotheses import inv_se3 as _inv_se3
from .frame_hypotheses import manuscript_contact_frame_hypotheses as _manuscript_contact_frame_hypotheses
from .frame_hypotheses import section33_confidence_components as _section33_confidence_components
from .frame_hypotheses import tool_axis_and_line_scores as _tool_axis_and_line_scores
from .depth_cleaning import clean_and_fit as _clean_and_fit
from .interaction_io import (
    find_results_json as _find_results_json,
    load_depth as _load_depth,
    load_vlm1_objects as _load_vlm1_objects,
    part_mask_map as _part_mask_map,
    part_readiness as _part_readiness,
    resolve_intrinsics as _resolve_intrinsics,
    resolve_world_up_cam as _resolve_world_up_cam,
)
from .primitive_fitting import fit_line as _fit_line
from .primitive_fitting import fit_plane as _fit_plane
from .primitive_fitting import unit as _unit
from .temporal_filter import TemporalGraspFilter
from .visualization import project_xyz as _project_xyz
from .visualization import write_visual_overlay as _write_visual_overlay


DEFAULT_DATA_PATH = os.environ.get("VLM3_DATA_PATH", "./data")
DEFAULT_FRAME_NO = int(os.environ.get("VLM3_FRAME_NO", "10"))
DEFAULT_OUT_DIR = os.environ.get("VLM3_OUT_DIR", "./data/vlm3")


def run_interaction_lift(
    out_dir: str,
    data_path: str,
    frame_no: int,
    depth_path: Optional[str] = None,
    vlm2_results_json: Optional[str] = None,
    fx: Optional[float] = None,
    fy: Optional[float] = None,
    cx: Optional[float] = None,
    cy: Optional[float] = None,
    depth_scale: float = 0.001,
    readiness_threshold: float = 0.60,
    world_up_cam: Optional[str] = None,
    world_up_world: Optional[str] = None,
    t_world_cam_json: Optional[str] = None,
    min_valid_depth_m: float = 0.05,
    max_valid_depth_m: float = 1.50,
    enforce_depth_range: bool = True,
    rgb_path_override: Optional[str] = None,
    temporal_filter: Optional[TemporalGraspFilter] = None,
    temporal_alpha: float = 0.30,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    primitives_dir = os.path.join(out_dir, "primitives")
    ensure_dir(primitives_dir)
    robot_dir = os.path.join(out_dir, "robot")
    ensure_dir(robot_dir)

    vlm2_json = vlm2_results_json or _find_results_json(out_dir)
    vlm2_payload = load_json(vlm2_json)
    vlm1_json = str(vlm2_payload.get("results_json_from_vlm1", "") or "")
    if not vlm1_json or not os.path.exists(vlm1_json):
        raise FileNotFoundError(f"Missing VLM1 results referenced by VLM2: {vlm1_json}")

    vlm1_payload = load_json(vlm1_json)
    vlm1_objects = _load_vlm1_objects(vlm1_json)
    rgb_path = str(rgb_path_override or vlm1_payload.get("rgb_path", "") or "")

    depth_file = depth_path or os.path.join(data_path, f"depth_{int(frame_no):06d}.npy")
    if not os.path.exists(depth_file):
        raise FileNotFoundError(f"Depth file not found: {depth_file}")
    depth = _load_depth(depth_file)
    intr = _resolve_intrinsics(depth.shape[:2], fx=fx, fy=fy, cx=cx, cy=cy)
    world_up_cam_vec, world_up_source = _resolve_world_up_cam(
        world_up_cam_text=world_up_cam,
        world_up_world_text=world_up_world,
        t_world_cam_json=t_world_cam_json,
    )
    full_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR) if rgb_path and os.path.exists(rgb_path) else None

    tool_entries = [o for o in vlm2_payload.get("objects", []) if bool(o.get("stage2_applicable", False)) and bool(o.get("ok", False))]
    target_entries = [o for o in vlm2_payload.get("objects", []) if not bool(o.get("stage2_applicable", True)) and bool(o.get("ok", False))]

    tool_primitives: List[Dict[str, Any]] = []
    target_primitives: List[Dict[str, Any]] = []
    hypotheses: List[Dict[str, Any]] = []
    all_sign_hypotheses: List[Dict[str, Any]] = []
    ranked_contact_frame_hypotheses: List[Dict[str, Any]] = []
    temporal_fusion_records: List[Dict[str, Any]] = []
    active_temporal_filter = temporal_filter or TemporalGraspFilter(alpha=temporal_alpha)
    virtual_target_fallback_used = False
    depth_filter_stats = {
        "enabled": bool(enforce_depth_range),
        "min_valid_depth_m": float(min_valid_depth_m),
        "max_valid_depth_m": float(max_valid_depth_m),
        "tool_rejected_count": 0,
        "target_rejected_count": 0,
    }

    def _depth_ok(xyz: Optional[np.ndarray]) -> bool:
        if xyz is None:
            return False
        z = float(xyz[2])
        if not np.isfinite(z):
            return False
        if not enforce_depth_range:
            return True
        return (float(min_valid_depth_m) <= z <= float(max_valid_depth_m))

    for tool in tool_entries:
        name = str(tool.get("name", "")).strip()
        if not name or name not in vlm1_objects:
            continue
        vlm1_obj = vlm1_objects[name]
        part_masks = _part_mask_map(vlm1_obj)
        work_path = part_masks.get("work_end")
        handle_path = part_masks.get("handle")
        if not work_path or not os.path.exists(work_path) or not handle_path or not os.path.exists(handle_path):
            continue
        work_mask = load_mask01(work_path)
        handle_mask = load_mask01(handle_path)
        work_pc = _backproject_mask(work_mask, depth, intr, depth_scale)
        handle_pc = _backproject_mask(handle_mask, depth, intr, depth_scale)
        work_line, work_line_raw, work_pc_clean, work_cleaning = _clean_and_fit(
            work_pc, _fit_line, primitive="line", role="work_end", min_points=8
        )
        handle_line, handle_line_raw, handle_pc_clean, handle_cleaning = _clean_and_fit(
            handle_pc, _fit_line, primitive="line", role="handle", min_points=8
        )

        grip_map = tool.get("grip_point_global", {})
        grip_xy = tuple(next(iter(grip_map.values()))) if grip_map else None
        touch_xy = tuple(tool.get("work_end_touch_mid_global", [])) if tool.get("work_end_touch_mid_global") else None
        edge_map = tool.get("work_end_points_global", {})
        edge_vals = list(edge_map.values())
        grip_xyz = _backproject_xy(grip_xy, depth, intr, depth_scale) if grip_xy is not None else None
        touch_xyz = _backproject_xy(touch_xy, depth, intr, depth_scale) if touch_xy is not None else None
        edge_p1_xyz = _backproject_xy(tuple(edge_vals[0]), depth, intr, depth_scale) if len(edge_vals) == 2 else None
        edge_p2_xyz = _backproject_xy(tuple(edge_vals[1]), depth, intr, depth_scale) if len(edge_vals) == 2 else None
        if grip_xyz is None or touch_xyz is None or edge_p1_xyz is None or edge_p2_xyz is None:
            continue
        if not (_depth_ok(grip_xyz) and _depth_ok(touch_xyz) and _depth_ok(edge_p1_xyz) and _depth_ok(edge_p2_xyz)):
            depth_filter_stats["tool_rejected_count"] += 1
            continue

        pattern = str(tool.get("pattern", "")).upper()
        tool_axis = _unit(touch_xyz - grip_xyz)
        work_edge_axis = _unit(edge_p2_xyz - edge_p1_xyz)
        contact_edge_band = _contact_edge_band_mask(
            work_mask,
            tuple(edge_vals[0]),
            tuple(edge_vals[1]),
            half_width_px=6.0,
        )
        contact_edge_pc = _backproject_mask(contact_edge_band, depth, intr, depth_scale)
        # Preserve the narrow contact band with a less aggressive threshold.
        contact_edge_line, contact_edge_line_raw, contact_edge_pc_clean, contact_edge_cleaning = _clean_and_fit(
            contact_edge_pc,
            _fit_line,
            primitive="line",
            role="contact_edge_band",
            min_points=8,
            min_retained_ratio=0.65,
            mad_scale=4.5,
        )
        tool_primitive = {
            "name": name,
            "pattern": pattern,
            "grip_xyz": grip_xyz,
            "touch_xyz": touch_xyz,
            "tool_axis": tool_axis,
            "grip_point_3d": tool.get("grip_point_3d"),
            "work_end_contact_3d": tool.get("work_end_contact_3d"),
            "axis_direction_3d": tool.get("axis_direction_3d"),
            "contact_normal_3d": tool.get("contact_normal_3d"),
            "contact_geometry": tool.get("contact_geometry"),
            "force_mode": tool.get("force_mode"),
            "wrench_basis": tool.get("wrench_basis"),
            "grasp_frame_cam": tool.get("grasp_frame_cam"),
            "grasp_quaternion_xyzw": tool.get("grasp_quaternion_xyzw"),
            "functional_area_px": tool.get("functional_area_px"),
            "grasp_width_px": tool.get("grasp_width_px"),
            "grasp_width_m": tool.get("grasp_width_m"),
            "grasp_width_mm": tool.get("grasp_width_mm"),
            "geometry_vlm_agreement": tool.get("geometry_vlm_agreement"),
            "affordance_uncertainty": tool.get("affordance_uncertainty"),
            "coherence_report": tool.get("coherence_report"),
            "affordance_hypotheses": tool.get("affordance_hypotheses"),
            "work_edge": {
                "endpoint_a": edge_p1_xyz,
                "endpoint_b": edge_p2_xyz,
                "axis": work_edge_axis,
            },
            "work_line": work_line,
            "work_line_raw": work_line_raw,
            "work_depth_cleaning": work_cleaning,
            "contact_edge_line": contact_edge_line,
            "contact_edge_line_raw": contact_edge_line_raw,
            "contact_edge_depth_cleaning": contact_edge_cleaning,
            "contact_edge_band_half_width_px": 6.0,
            "contact_edge_band_num_pixels": int(np.count_nonzero(contact_edge_band)),
            "handle_line": handle_line,
            "handle_line_raw": handle_line_raw,
            "handle_depth_cleaning": handle_cleaning,
            "readiness": {
                "handle": _part_readiness(vlm1_obj, "handle"),
                "work_end": _part_readiness(vlm1_obj, "work_end"),
                "object": float(vlm1_obj.get("object_lift_readiness", 0.0) or 0.0),
            },
        }
        tool_primitives.append(to_builtin(tool_primitive))

    for target in target_entries:
        name = str(target.get("name", "")).strip()
        if not name or name not in vlm1_objects:
            continue
        vlm1_obj = vlm1_objects[name]
        part_masks = _part_mask_map(vlm1_obj)
        contact_path = part_masks.get("contact_region") or part_masks.get("whole")
        whole_path = part_masks.get("whole")
        if not contact_path or not os.path.exists(contact_path):
            continue
        contact_mask = load_mask01(contact_path)
        whole_mask = load_mask01(whole_path) if whole_path and os.path.exists(whole_path) else contact_mask

        contact_pc = _backproject_mask(contact_mask, depth, intr, depth_scale)
        whole_pc = _backproject_mask(whole_mask, depth, intr, depth_scale)
        plane_pc = contact_pc if contact_pc.shape[0] >= 12 else whole_pc
        plane, plane_raw, plane_pc_clean, plane_cleaning = _clean_and_fit(
            plane_pc, _fit_plane, primitive="plane", role="target_contact_region", min_points=12
        )
        edge_map = target.get("target_edge_points_global", {})
        edge_vals = list(edge_map.values())
        if len(edge_vals) != 2:
            continue
        p1_xyz = _backproject_xy(tuple(edge_vals[0]), depth, intr, depth_scale)
        p2_xyz = _backproject_xy(tuple(edge_vals[1]), depth, intr, depth_scale)
        touch_xy = tuple(target.get("target_edge_touch_mid_global", [])) if target.get("target_edge_touch_mid_global") else None
        touch_xyz = _backproject_xy(touch_xy, depth, intr, depth_scale) if touch_xy is not None else None
        if p1_xyz is None or p2_xyz is None or touch_xyz is None:
            continue
        if not (_depth_ok(p1_xyz) and _depth_ok(p2_xyz) and _depth_ok(touch_xyz)):
            depth_filter_stats["target_rejected_count"] += 1
            continue
        line_axis = _unit(p2_xyz - p1_xyz)
        target_primitive = {
            "name": name,
            "touch_xyz": touch_xyz,
            "edge_line": {
                "endpoint_a": p1_xyz,
                "endpoint_b": p2_xyz,
                "axis": line_axis,
            },
            "plane": plane,
            "plane_raw": plane_raw,
            "plane_depth_cleaning": plane_cleaning,
            "readiness": {
                "contact_region": _part_readiness(vlm1_obj, "contact_region"),
                "whole": _part_readiness(vlm1_obj, "whole"),
                "object": float(vlm1_obj.get("object_lift_readiness", 0.0) or 0.0),
            },
        }
        target_primitives.append(to_builtin(target_primitive))

    # Single-object fallback:
    # If no target is grounded (e.g., "grasp tool and lift"), synthesize a virtual target
    # from tool geometry so downstream frame lifting can still run.
    if not target_primitives and tool_primitives:
        for tool in tool_primitives:
            touch_xyz = np.array(tool.get("touch_xyz", []), dtype=np.float32)
            work_edge = tool.get("work_edge") or {}
            e1 = np.array(work_edge.get("endpoint_a", []), dtype=np.float32)
            e2 = np.array(work_edge.get("endpoint_b", []), dtype=np.float32)
            axis = _unit(np.array(work_edge.get("axis", []), dtype=np.float32))
            tool_axis = _unit(np.array(tool.get("tool_axis", []), dtype=np.float32))
            if touch_xyz.shape != (3,) or e1.shape != (3,) or e2.shape != (3,):
                continue
            if float(np.linalg.norm(axis)) < 1e-6:
                axis = _unit(e2 - e1)
            if float(np.linalg.norm(axis)) < 1e-6:
                axis = _unit(tool_axis)
            if float(np.linalg.norm(axis)) < 1e-6:
                axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            # Plane normal must not be parallel to edge axis.
            plane_n = _unit(np.cross(axis, tool_axis))
            if float(np.linalg.norm(plane_n)) < 1e-6:
                up = world_up_cam_vec if world_up_cam_vec is not None else np.array([0.0, -1.0, 0.0], dtype=np.float32)
                plane_n = _unit(np.cross(axis, up))
            if float(np.linalg.norm(plane_n)) < 1e-6:
                plane_n = np.array([0.0, 0.0, 1.0], dtype=np.float32)

            t_ready = float(min(tool["readiness"]["handle"], tool["readiness"]["work_end"], tool["readiness"]["object"]))
            virtual_target = {
                "name": f"{tool.get('name', 'tool')}_virtual_target",
                "touch_xyz": touch_xyz.astype(np.float32),
                "edge_line": {
                    "endpoint_a": e1.astype(np.float32),
                    "endpoint_b": e2.astype(np.float32),
                    "axis": axis.astype(np.float32),
                },
                "plane": {
                    "normal": plane_n.astype(np.float32),
                    "point": touch_xyz.astype(np.float32),
                    "residual_mean": 0.0,
                    "residual_p90": 0.0,
                    "num_points": 0,
                },
                "readiness": {
                    "contact_region": t_ready,
                    "whole": t_ready,
                    "object": t_ready,
                },
                "virtual": True,
            }
            target_primitives.append(to_builtin(virtual_target))
            virtual_target_fallback_used = True

    for tool in tool_primitives:
        tool_touch = np.array(tool["touch_xyz"], dtype=np.float32)
        if not target_primitives:
            continue
        nearest_target = min(
            target_primitives,
            key=lambda t: float(np.sum((np.array(t["touch_xyz"], dtype=np.float32) - tool_touch) ** 2)),
        )

        t_ready = float(min(tool["readiness"]["handle"], tool["readiness"]["work_end"], tool["readiness"]["object"]))
        g_ready = float(max(nearest_target["readiness"]["contact_region"], nearest_target["readiness"]["whole"], nearest_target["readiness"]["object"]))
        if t_ready < readiness_threshold or g_ready < readiness_threshold:
            hypotheses.append(
                {
                    "tool_name": tool["name"],
                    "target_name": nearest_target["name"],
                    "ok": False,
                    "reason": "lift_readiness_below_threshold",
                    "tool_readiness": t_ready,
                    "target_readiness": g_ready,
                }
            )
            continue

        plane = nearest_target.get("plane")
        if not isinstance(plane, dict):
            hypotheses.append(
                {
                    "tool_name": tool["name"],
                    "target_name": nearest_target["name"],
                    "ok": False,
                    "reason": "target_plane_fit_failed",
                }
            )
            continue

        track_id = f"{tool['name']}->{nearest_target['name']}"
        track_reset = active_temporal_filter.prepare_track(track_id)
        raw_temporal = {
            "grip_point_3d": list(tool["grip_xyz"]),
            "contact_point_3d": list(tool["touch_xyz"]),
            "axis_direction_3d": list(tool["tool_axis"]),
            "contact_normal_3d": list(plane["normal"]),
        }
        active_temporal_filter.update(
            grip_point_3d=raw_temporal["grip_point_3d"],
            contact_point_3d=raw_temporal["contact_point_3d"],
            axis_direction_3d=raw_temporal["axis_direction_3d"],
            contact_normal_3d=raw_temporal["contact_normal_3d"],
        )
        fused_temporal = active_temporal_filter.as_dict()
        tool["grip_xyz"] = fused_temporal["grip_point_3d"]
        tool["touch_xyz"] = fused_temporal["contact_point_3d"]
        tool["tool_axis"] = fused_temporal["axis_direction_3d"]
        plane["normal"] = fused_temporal["contact_normal_3d"]
        nearest_target["plane"] = plane
        temporal_fusion_records.append(
            {
                "track_id": track_id,
                "track_reset": bool(track_reset),
                "alpha": float(active_temporal_filter.alpha),
                "n_updates": int(active_temporal_filter.n_updates),
                "warm": bool(active_temporal_filter.is_warm()),
                "raw": raw_temporal,
                "fused": fused_temporal,
            }
        )

        tool_axis = np.array(tool["tool_axis"], dtype=np.float32)
        work_edge_axis = np.array(tool["work_edge"]["axis"], dtype=np.float32)
        target_line_axis = np.array(nearest_target["edge_line"]["axis"], dtype=np.float32)
        align = _tool_axis_and_line_scores(
            pattern=str(tool.get("pattern", "")),
            tool_axis=tool_axis,
            work_edge_axis=work_edge_axis,
            target_line_axis=target_line_axis,
        )
        plane_normal = np.array(plane["normal"], dtype=np.float32)
        touch_xyz = np.array(nearest_target["touch_xyz"], dtype=np.float32)

        residual_limit_m = 0.015
        contact_edge_line = tool.get("contact_edge_line")
        residual_sources = {
            "handle_line": None if not isinstance(tool.get("handle_line"), dict) else tool["handle_line"].get("residual_p90"),
            "contact_edge_line": None if not isinstance(contact_edge_line, dict) else contact_edge_line.get("residual_p90"),
            "target_plane": plane.get("residual_p90"),
        }
        primitive_validation = {
            name: {
                "residual_p90_m": None if value is None else float(value),
                "limit_m": residual_limit_m,
                "valid": bool(value is not None and np.isfinite(float(value)) and float(value) <= residual_limit_m),
            }
            for name, value in residual_sources.items()
        }
        strict_geometry_valid = all(item["valid"] for item in primitive_validation.values())
        finite_residuals = [float(value) for value in residual_sources.values() if value is not None and np.isfinite(float(value))]
        worst_residual = max(finite_residuals) if finite_residuals else float("inf")
        confidence = _section33_confidence_components(tool, worst_residual, residual_limit_m)
        sign_candidates = _manuscript_contact_frame_hypotheses(
            axis=tool_axis,
            normal=plane_normal,
            translation=np.asarray(tool["grip_xyz"], dtype=np.float32),
            approach_along_negative_normal=True,
        )
        candidates: List[Dict[str, Any]] = []
        for candidate in sign_candidates:
            enriched = dict(candidate)
            enriched.update(
                {
                    "tool_name": tool["name"],
                    "target_name": nearest_target["name"],
                    "pattern": tool["pattern"],
                    "ok": bool(candidate["proper_rotation"] and strict_geometry_valid),
                    "score": float(confidence["score"]),
                    "confidence_equation_7": confidence,
                    "primitive_validation": primitive_validation,
                    "strict_geometry_valid": strict_geometry_valid,
                    "tool_readiness": t_ready,
                    "target_readiness": g_ready,
                    "alignment": align,
                }
            )
            if candidate["proper_rotation"] and not strict_geometry_valid:
                enriched["rejection_reason"] = "primitive_residual_validation_failed"
            all_sign_hypotheses.append(to_builtin(enriched))
            if candidate["proper_rotation"]:
                candidates.append(enriched)
        if not candidates:
            hypotheses.append(
                {
                    "tool_name": tool["name"],
                    "target_name": nearest_target["name"],
                    "ok": False,
                    "reason": "frame_generation_failed",
                }
            )
            continue
        candidates.sort(
            key=lambda c: (bool(c.get("surface_approach_preferred")), float(c["score"])),
            reverse=True,
        )
        for rank, candidate in enumerate(candidates, start=1):
            candidate["rank"] = rank
            ranked_contact_frame_hypotheses.append(to_builtin(candidate))
        if not strict_geometry_valid:
            hypotheses.append(
                {
                    "tool_name": tool["name"],
                    "target_name": nearest_target["name"],
                    "ok": False,
                    "reason": "primitive_residual_validation_failed",
                    "primitive_validation": primitive_validation,
                    "ranked_hypotheses": to_builtin(candidates),
                }
            )
            continue
        best = candidates[0].copy()
        best["alternatives"] = to_builtin(candidates[1:])
        hypotheses.append(to_builtin(best))

    best_hypothesis = next((h for h in hypotheses if bool(h.get("ok"))), None)
    frame_bundle: Optional[Dict[str, Any]] = None
    robot_report_path: Optional[str] = None
    if best_hypothesis is not None:
        best_tool = next((t for t in tool_primitives if str(t.get("name")) == str(best_hypothesis.get("tool_name"))), None)
        best_target = next((t for t in target_primitives if str(t.get("name")) == str(best_hypothesis.get("target_name"))), None)
        if best_tool is not None and best_target is not None:
            frame_bundle = _build_tool_target_frames(
                best_tool,
                best_target,
                intr=intr,
                tangent_sign=float(best_hypothesis.get("tangent_sign", 1.0) or 1.0),
                # Target +zP is always outward.  The selected grasp sign is
                # already encoded in best_hypothesis["rotation"].
                normal_sign=1.0,
                world_up_cam=world_up_cam_vec,
                project_xyz_fn=_project_xyz,
            )
            if frame_bundle is not None:
                # The ranked Section 3.3 hypothesis is the grasp frame that
                # downstream compilation must consume.  Keep the separately
                # fitted target frame/contact geometry from the bundle.
                selected_grasp_frame = np.eye(4, dtype=np.float32)
                selected_grasp_frame[:3, :3] = np.asarray(best_hypothesis["rotation"], dtype=np.float32)
                selected_grasp_frame[:3, 3] = np.asarray(best_hypothesis["translation_xyz"], dtype=np.float32)
                frame_bundle["T_cam_T"] = selected_grasp_frame
                frame_bundle["T_tool_to_target"] = frame_bundle["T_cam_P"] @ _inv_se3(selected_grasp_frame)
                robot_report = {
                    "frame": int(frame_no),
                    "tool_type": best_hypothesis.get("tool_name"),
                    "target_type": best_hypothesis.get("target_name"),
                    "pattern": best_hypothesis.get("pattern"),
                    "frames_cam": {
                        "tool": frame_bundle["T_cam_T"],
                        "target": frame_bundle["T_cam_P"],
                        "T_tool_to_target": frame_bundle["T_tool_to_target"],
                    },
                    "target_action_edge_cam": frame_bundle["target_action_edge_cam"],
                    "contact_points_P": frame_bundle["contact_points_P"],
                    "target_axes_P": frame_bundle["target_axes_P"],
                    "target_normal_outward_cam": frame_bundle["target_normal_outward_cam"],
                    "approach_direction_cam": frame_bundle["approach_direction_cam"],
                    "target_normal_orientation_source": frame_bundle["target_normal_orientation_source"],
                    "target_normal_flipped_from_raw": frame_bundle["target_normal_flipped_from_raw"],
                    "world_up_cam": frame_bundle.get("world_up_cam"),
                    "world_up_source": world_up_source,
                    "best_hypothesis_score": float(best_hypothesis.get("score", 0.0)),
                }
                robot_report_path = os.path.join(robot_dir, f"eyes_report_{int(frame_no):06d}.json")
                save_json(robot_report_path, robot_report)

    overlay_path = _write_visual_overlay(
        out_dir=out_dir,
        full_bgr=full_bgr,
        intr=intr,
        best_hypothesis=best_hypothesis,
        tool_primitives=tool_primitives,
        target_primitives=target_primitives,
        frame_bundle=frame_bundle,
    )

    result = {
        "ok": any(bool(h.get("ok")) for h in hypotheses),
        "frame": int(frame_no),
        "rgb_path": rgb_path or None,
        "depth_path": depth_file,
        "depth_scale": float(depth_scale),
        "intrinsics": intr,
        "world_up_cam": None if world_up_cam_vec is None else world_up_cam_vec,
        "world_up_source": world_up_source,
        "t_world_cam_json": t_world_cam_json,
        "readiness_threshold": float(readiness_threshold),
        "depth_sanity": depth_filter_stats,
        "vlm2_results_json": vlm2_json,
        "vlm1_results_json": vlm1_json,
        "best_frame_overlay_path": overlay_path,
        "robot_report_path": robot_report_path,
        "frames_cam": None if frame_bundle is None else {
            "tool": frame_bundle["T_cam_T"],
            "target": frame_bundle["T_cam_P"],
            "T_tool_to_target": frame_bundle["T_tool_to_target"],
        },
        "target_action_edge_cam": None if frame_bundle is None else frame_bundle["target_action_edge_cam"],
        "contact_points_P": None if frame_bundle is None else frame_bundle["contact_points_P"],
        "target_axes_P": None if frame_bundle is None else frame_bundle["target_axes_P"],
        "target_normal_outward_cam": None if frame_bundle is None else frame_bundle["target_normal_outward_cam"],
        "approach_direction_cam": None if frame_bundle is None else frame_bundle["approach_direction_cam"],
        "target_normal_orientation_source": None if frame_bundle is None else frame_bundle["target_normal_orientation_source"],
        "target_normal_flipped_from_raw": None if frame_bundle is None else frame_bundle["target_normal_flipped_from_raw"],
        "tool_primitives": to_builtin(tool_primitives),
        "target_primitives": to_builtin(target_primitives),
        "virtual_target_fallback_used": bool(virtual_target_fallback_used),
        "hypotheses": to_builtin(hypotheses),
        "all_sign_hypotheses": to_builtin(all_sign_hypotheses),
        "ranked_contact_frame_hypotheses": to_builtin(ranked_contact_frame_hypotheses),
        "temporal_fusion": to_builtin(temporal_fusion_records),
        "best_hypothesis": best_hypothesis,
    }
    interaction_dir = os.path.join(out_dir, "interaction_lift")
    ensure_dir(interaction_dir)
    save_json(os.path.join(interaction_dir, "interaction_lift_results.json"), result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Interaction lift: 3D lifting + contact-frame hypotheses")
    ap.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    ap.add_argument("--frame_no", type=int, default=DEFAULT_FRAME_NO)
    ap.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--depth_path", default=None)
    ap.add_argument("--rgb_path", default=None, help="Optional RGB override for archived runs.")
    ap.add_argument("--vlm2_results_json", default=None)
    ap.add_argument("--fx", type=float, default=None)
    ap.add_argument("--fy", type=float, default=None)
    ap.add_argument("--cx", type=float, default=None)
    ap.add_argument("--cy", type=float, default=None)
    ap.add_argument("--depth_scale", type=float, default=0.001)
    ap.add_argument("--readiness_threshold", type=float, default=0.60)
    ap.add_argument("--temporal_alpha", type=float, default=0.30)
    ap.add_argument("--min_valid_depth_m", type=float, default=0.05)
    ap.add_argument("--max_valid_depth_m", type=float, default=1.50)
    ap.add_argument("--disable_depth_range_check", action="store_true")
    ap.add_argument("--world_up_cam", default=os.environ.get("VLM3_WORLD_UP_CAM"))
    ap.add_argument("--world_up_world", default=os.environ.get("VLM3_WORLD_UP_WORLD", "0,0,1"))
    ap.add_argument("--t_world_cam_json", default=os.environ.get("VLM3_T_WORLD_CAM_JSON"))
    args = ap.parse_args()

    out = run_interaction_lift(
        out_dir=args.out_dir,
        data_path=args.data_path,
        frame_no=args.frame_no,
        depth_path=args.depth_path,
        vlm2_results_json=args.vlm2_results_json,
        fx=args.fx,
        fy=args.fy,
        cx=args.cx,
        cy=args.cy,
        depth_scale=args.depth_scale,
        readiness_threshold=args.readiness_threshold,
        min_valid_depth_m=args.min_valid_depth_m,
        max_valid_depth_m=args.max_valid_depth_m,
        enforce_depth_range=not bool(args.disable_depth_range_check),
        rgb_path_override=args.rgb_path,
        temporal_alpha=args.temporal_alpha,
        world_up_cam=args.world_up_cam,
        world_up_world=args.world_up_world,
        t_world_cam_json=args.t_world_cam_json,
    )
    print(json.dumps(to_builtin(out), indent=2))


if __name__ == "__main__":
    main()
