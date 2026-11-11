#!/usr/bin/env python3
"""Compile RegioManip action intents into grounded constraint programs.

This module sits between:
- high-level action intent outputs (motion / force primitives)
- VLM3 grounded tool-target geometry
- confidence-conditioned scheduling

and produces a structured ``compiled_action_program.json`` that can later be
consumed by the ReKep solvers.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

GWS_FRICTION_MU = float(os.environ.get("REGIOMANIP_GWS_FRICTION_MU", "0.40"))
GWS_MIN_NORMAL_FORCE_N = float(os.environ.get("REGIOMANIP_GWS_MIN_NORMAL_FORCE_N", "0.25"))
GWS_APPROACH_FORCE_N = float(os.environ.get("REGIOMANIP_GWS_APPROACH_FORCE_N", "0.75"))
GWS_MAINTAIN_FORCE_N = float(os.environ.get("REGIOMANIP_GWS_MAINTAIN_FORCE_N", "4.0"))
GWS_POINT_CONTACT_RADIUS_M = float(os.environ.get("REGIOMANIP_GWS_POINT_CONTACT_RADIUS_M", "0.004"))
GWS_SURFACE_HALF_EXTENT_M = float(os.environ.get("REGIOMANIP_GWS_SURFACE_HALF_EXTENT_M", "0.012"))


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_json(path: str, payload: Dict[str, Any]) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _clip(x: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, x)))


def _mean(values: List[float], fallback: float) -> float:
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return float(fallback)
    return float(sum(vals) / len(vals))


def _unit(vec: List[float], fallback: Optional[List[float]] = None) -> List[float]:
    if not isinstance(vec, list) or len(vec) != 3:
        return [0.0, 0.0, 1.0] if fallback is None else list(fallback)
    norm = math.sqrt(sum(float(v) * float(v) for v in vec))
    if norm < 1e-8:
        return [0.0, 0.0, 1.0] if fallback is None else list(fallback)
    return [float(v) / norm for v in vec]


def _vector_sub(a: Optional[List[float]], b: Optional[List[float]]) -> Optional[List[float]]:
    if not isinstance(a, list) or not isinstance(b, list) or len(a) != 3 or len(b) != 3:
        return None
    return [float(a[i]) - float(b[i]) for i in range(3)]


def _vector_add(a: List[float], b: List[float]) -> List[float]:
    return [float(a[i]) + float(b[i]) for i in range(3)]


def _vector_scale(vec: List[float], scale: float) -> List[float]:
    return [float(scale) * float(v) for v in vec]


def _dot(a: List[float], b: List[float]) -> float:
    return float(sum(float(a[i]) * float(b[i]) for i in range(3)))


def _norm(vec: List[float]) -> float:
    return float(math.sqrt(sum(float(v) * float(v) for v in vec)))


def _cross(a: List[float], b: List[float]) -> List[float]:
    return [
        float(a[1]) * float(b[2]) - float(a[2]) * float(b[1]),
        float(a[2]) * float(b[0]) - float(a[0]) * float(b[2]),
        float(a[0]) * float(b[1]) - float(a[1]) * float(b[0]),
    ]


def _project_to_plane(vec: List[float], normal: List[float]) -> List[float]:
    coeff = _dot(vec, normal)
    return [float(vec[i]) - coeff * float(normal[i]) for i in range(3)]


def _safe_scalar(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _parse_key(token: str) -> Optional[str]:
    s = str(token).strip()
    return s or None


def _parse_motion_primitive(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    m = re.match(r"^([A-Za-z]+)\((.*)\)$", raw)
    if not m:
        return {
            "raw": raw,
            "name": raw or "Unknown",
            "args": [],
            "ok": False,
        }
    args = [part.strip() for part in m.group(2).split(",")] if m.group(2).strip() else []
    return {
        "raw": raw,
        "name": m.group(1),
        "args": args,
        "ok": True,
    }


def _parse_force_primitive(text: str) -> Dict[str, Any]:
    raw = str(text or "").strip()
    if raw.lower() == "off":
        return {
            "raw": raw,
            "name": "Off",
            "args": [],
            "ok": True,
        }
    m = re.match(r"^([A-Za-z]+)\((.*)\)$", raw)
    if not m:
        return {
            "raw": raw,
            "name": raw or "Unknown",
            "args": [],
            "ok": False,
        }
    args: List[Any] = []
    for part in m.group(2).split(","):
        token = part.strip()
        try:
            args.append(float(token))
        except ValueError:
            args.append(token)
    return {
        "raw": raw,
        "name": m.group(1),
        "args": args,
        "ok": True,
    }


def _infer_force_class(force_spec: Dict[str, Any], contact_intent: str) -> str:
    if force_spec.get("name") == "Off" or contact_intent == "break_contact":
        return "gentle"
    if force_spec.get("name") not in {"Const", "Thresh"}:
        return "nominal"
    force_val = float(force_spec.get("args", [0.0])[0] or 0.0) if force_spec.get("args") else 0.0
    if force_val <= 2.0:
        return "gentle"
    if force_val <= 6.0:
        return "nominal"
    return "firm"


def _extract_confidence(confidence_payload: Optional[Dict[str, Any]], vlm3_payload: Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(confidence_payload, dict):
        rho = confidence_payload.get("rho_k")
        subscores = confidence_payload.get("subscores", {})
        return {
            "interaction": float(rho if rho is not None else 0.5),
            "perception": float(subscores.get("geometry_reliability", rho if rho is not None else 0.5)),
            "source": "confidence_json",
            "raw": confidence_payload,
        }

    best = vlm3_payload.get("best_hypothesis") or {}
    target_primitives = vlm3_payload.get("target_primitives") or []
    readiness_values: List[float] = []
    for target in target_primitives:
        readiness = target.get("readiness", {})
        for key in ("contact_region", "whole", "object"):
            if readiness.get(key) is not None:
                readiness_values.append(float(readiness[key]))
    best_score = best.get("score")
    interaction = _mean(
        [float(best_score)] if best_score is not None else [],
        fallback=0.5,
    )
    perception = _mean(readiness_values, fallback=interaction)
    return {
        "interaction": float(_clip(interaction, 0.0, 1.0)),
        "perception": float(_clip(perception, 0.0, 1.0)),
        "source": "vlm3_fallback",
        "raw": {
            "best_hypothesis_score": best_score,
            "target_readiness_values": readiness_values,
        },
    }


def _collect_frame_candidates(vlm3_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    best = vlm3_payload.get("best_hypothesis") or {}
    if not best:
        return []
    candidates = [
        {
            "name": "best",
            "score": float(best.get("score", 0.0)),
            "normal_sign": int(best.get("normal_sign", 1) or 1),
            "tangent_sign": int(best.get("tangent_sign", 1) or 1),
        }
    ]
    for idx, alt in enumerate(best.get("alternatives") or []):
        candidates.append(
            {
                "name": f"alt_{idx + 1}",
                "score": float(alt.get("score", 0.0)),
                "normal_sign": int(alt.get("normal_sign", 1) or 1),
                "tangent_sign": int(alt.get("tangent_sign", 1) or 1),
            }
        )
    return candidates


def _find_target_primitive(vlm3_payload: Dict[str, Any]) -> Dict[str, Any]:
    best = vlm3_payload.get("best_hypothesis") or {}
    target_name = str(best.get("target_name", "")).strip()
    for target in vlm3_payload.get("target_primitives") or []:
        if str(target.get("name", "")).strip() == target_name:
            return target
    return (vlm3_payload.get("target_primitives") or [{}])[0]


def _find_tool_primitive(vlm3_payload: Dict[str, Any]) -> Dict[str, Any]:
    best = vlm3_payload.get("best_hypothesis") or {}
    tool_name = str(best.get("tool_name", "")).strip()
    for tool in vlm3_payload.get("tool_primitives") or []:
        if str(tool.get("name", "")).strip() == tool_name:
            return tool
    return (vlm3_payload.get("tool_primitives") or [{}])[0]


def _default_geometry(vlm3_payload: Dict[str, Any]) -> Dict[str, Any]:
    target = _find_target_primitive(vlm3_payload)
    tool = _find_tool_primitive(vlm3_payload)
    target_axes = vlm3_payload.get("target_axes_P") or {}
    edge = vlm3_payload.get("target_action_edge_cam") or {}
    contact_points = vlm3_payload.get("contact_points_P") or []
    tool_frame = (((vlm3_payload.get("frames_cam") or {}).get("tool")) or [])

    line_start = edge.get("p1_3d_m")
    line_end = edge.get("p2_3d_m")
    line_dir = edge.get("dir_xP_3d") or (target.get("edge_line") or {}).get("axis")
    surface_normal = target_axes.get("zP") or (target.get("plane") or {}).get("normal")
    tool_axis = (tool.get("work_edge") or {}).get("axis") or tool.get("tool_axis")

    out = {
        "contact_point_P": contact_points[0] if contact_points else [0.0, 0.0, 0.0],
        "line_start_cam": line_start,
        "line_end_cam": line_end,
        "line_dir_cam": _unit(line_dir or [1.0, 0.0, 0.0], fallback=[1.0, 0.0, 0.0]),
        "surface_normal_cam": _unit(surface_normal or [0.0, 0.0, 1.0], fallback=[0.0, 0.0, 1.0]),
        "tool_axis_cam": _unit(tool_axis or [1.0, 0.0, 0.0], fallback=[1.0, 0.0, 0.0]),
        "target_touch_cam": target.get("touch_xyz"),
        "tool_touch_cam": tool.get("touch_xyz"),
        "tool_grip_cam": tool.get("grip_xyz"),
        "tool_frame_cam": tool_frame,
        "affordance_contact_geometry": tool.get("contact_geometry"),
        "affordance_force_mode": tool.get("force_mode"),
        "affordance_contact_normal_cam": tool.get("contact_normal_3d"),
        "affordance_wrench_basis": tool.get("wrench_basis"),
        "affordance_grip_point_3d": tool.get("grip_point_3d"),
        "affordance_work_end_contact_3d": tool.get("work_end_contact_3d"),
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
        "affordance_grasp_type": ((tool.get("coherence_report") or {}).get("selected_triple") or {}).get("grasp_type"),
        "work_edge_axis_cam": (tool.get("work_edge") or {}).get("axis"),
        "work_edge_endpoint_a_cam": (tool.get("work_edge") or {}).get("endpoint_a"),
        "work_edge_endpoint_b_cam": (tool.get("work_edge") or {}).get("endpoint_b"),
    }
    if isinstance(line_start, list) and isinstance(line_end, list) and len(line_start) == 3 and len(line_end) == 3:
        delta = _vector_sub(line_end, line_start)
        if delta is not None:
            out["progress_span_m"] = math.sqrt(sum(float(v) * float(v) for v in delta))
    return out


def _geometry_for_motion(
    motion_spec: Dict[str, Any],
    toolmanip_vlm2_payload: Dict[str, Any],
    vlm3_payload: Dict[str, Any],
) -> Dict[str, Any]:
    geom = _default_geometry(vlm3_payload)
    args = motion_spec.get("args") or []
    symbolic = {
        "start_ref": _parse_key(args[0]) if len(args) > 0 else None,
        "end_ref": _parse_key(args[1]) if len(args) > 1 else None,
        "center_ref": _parse_key(args[1]) if motion_spec.get("name") == "Rotate" and len(args) > 1 else None,
        "angle_ref": args[2] if motion_spec.get("name") == "Rotate" and len(args) > 2 else None,
    }
    geom["symbolic_refs"] = symbolic
    geom["symbolic_grounding_refs"] = {
        "H": toolmanip_vlm2_payload.get("Keypoint H"),
        "F": toolmanip_vlm2_payload.get("Keypoint F"),
        "O": toolmanip_vlm2_payload.get("Keypoint O"),
        "Q": toolmanip_vlm2_payload.get("Keypoint Q"),
        "t": toolmanip_vlm2_payload.get("Vector t"),
        "o": toolmanip_vlm2_payload.get("Vector o"),
    }
    if motion_spec.get("name") == "Rotate":
        geom["rotation_axis_cam"] = geom["surface_normal_cam"]
        geom["rotation_center"] = geom["contact_point_P"]
        geom["rotation_target"] = motion_spec.get("args", [None, None, 0.0])[-1]
    return geom


def _load_template_and_mode(
    motion_spec: Dict[str, Any],
    templates_payload: Dict[str, Any],
) -> Tuple[str, Dict[str, Any]]:
    name = str(motion_spec.get("name", "")).strip()
    args = motion_spec.get("args") or []
    symbolic_map = templates_payload.get("symbolic_action_map") or {}

    if name == "Linear" and len(args) >= 2 and str(args[1]).lower().startswith("z+"):
        mode_key = "RetractMove"
    else:
        mode_key = symbolic_map.get(name, name)
    template = deepcopy((templates_payload.get("templates") or {}).get(mode_key, {}))
    if not template:
        template = {
            "mode": mode_key or "Unknown",
            "contact_intent": "unknown",
            "terminal_constraints": [],
            "path_constraints": [],
            "theta_defaults": {},
        }
    return str(template.get("mode", mode_key)), template


def _confidence_schedule(theta: Dict[str, Any], confidence: Dict[str, Any], mode: str) -> Dict[str, Any]:
    interaction = float(confidence.get("interaction", 0.5))
    perception = float(confidence.get("perception", 0.5))
    out = deepcopy(theta)

    if "contact_band_m" in out and isinstance(out["contact_band_m"], list) and len(out["contact_band_m"]) == 2:
        lo, hi = float(out["contact_band_m"][0]), float(out["contact_band_m"][1])
        scale = 1.0 + 0.6 * (1.0 - perception)
        out["contact_band_m"] = [lo, hi * scale]

    tol_keys = [
        "normal_align_tol_deg",
        "axis_align_tol_deg",
    ]
    for key in tol_keys:
        if key in out:
            out[key] = float(out[key]) * (1.0 + 0.4 * (1.0 - perception))

    weight_keys = [
        "progress_weight",
        "frame_align_weight",
        "rotation_weight",
        "wrench_weight",
    ]
    for key in weight_keys:
        if key in out:
            out[key] = float(out[key]) * (0.8 + 0.4 * interaction)

    out["activation_logic"] = {
        "confidence_source": str(confidence.get("source", "unknown")),
        "interaction_confidence": interaction,
        "perception_confidence": perception,
        "mode": mode,
    }
    return out


def _normalize_contact_geometry(label: Any) -> str:
    text = str(label or "").strip().lower()
    if text in {"point", "line", "surface"}:
        return text
    return "surface"


def _contact_frame_from_geometry(geometry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    basis = geometry.get("affordance_wrench_basis") or {}
    normal_hint = geometry.get("affordance_contact_normal_cam") or basis.get("normal") or geometry.get("surface_normal_cam")
    axis_hint = (
        geometry.get("work_edge_axis_cam")
        or basis.get("axis")
        or geometry.get("tool_axis_cam")
    )
    tangent_hint = basis.get("tangent")

    normal = _unit(normal_hint or [0.0, 0.0, 1.0], fallback=[0.0, 0.0, 1.0])
    tangent_1 = _project_to_plane(
        list(axis_hint) if isinstance(axis_hint, list) else list(tangent_hint or [1.0, 0.0, 0.0]),
        normal,
    )
    if _norm(tangent_1) < 1e-6 and isinstance(tangent_hint, list):
        tangent_1 = _project_to_plane(tangent_hint, normal)
    if _norm(tangent_1) < 1e-6:
        tangent_1 = _cross([1.0, 0.0, 0.0], normal)
    if _norm(tangent_1) < 1e-6:
        tangent_1 = _cross([0.0, 1.0, 0.0], normal)
    tangent_1 = _unit(tangent_1, fallback=[1.0, 0.0, 0.0])

    tangent_2 = _cross(normal, tangent_1)
    if _norm(tangent_2) < 1e-6:
        return None
    tangent_2 = _unit(tangent_2, fallback=[0.0, 1.0, 0.0])
    tangent_1 = _unit(_cross(tangent_2, normal), fallback=tangent_1)

    return {
        "normal_cam": normal,
        "tangent_1_cam": tangent_1,
        "tangent_2_cam": tangent_2,
        "line_axis_cam": tangent_1,
        "source": "affordance_wrench_basis",
    }


def _contact_patch_metrics(geometry: Dict[str, Any]) -> Dict[str, float]:
    endpoint_a = geometry.get("work_edge_endpoint_a_cam")
    endpoint_b = geometry.get("work_edge_endpoint_b_cam")
    line_span = None
    if isinstance(endpoint_a, list) and isinstance(endpoint_b, list) and len(endpoint_a) == 3 and len(endpoint_b) == 3:
        delta = _vector_sub(endpoint_b, endpoint_a)
        if delta is not None:
            line_span = _norm(delta)
    line_half_length = max(float(line_span or 0.0) * 0.5, GWS_POINT_CONTACT_RADIUS_M)
    surface_half_t1 = max(line_half_length, GWS_SURFACE_HALF_EXTENT_M)
    surface_half_t2 = max(min(line_half_length, GWS_SURFACE_HALF_EXTENT_M), GWS_POINT_CONTACT_RADIUS_M * 2.0)
    return {
        "point_radius_m": float(GWS_POINT_CONTACT_RADIUS_M),
        "line_half_length_m": float(line_half_length),
        "surface_half_extent_t1_m": float(surface_half_t1),
        "surface_half_extent_t2_m": float(surface_half_t2),
    }


def _requested_force_and_torque(
    force_spec: Dict[str, Any],
    contact_intent: str,
    confidence: Dict[str, Any],
) -> Tuple[float, Optional[float]]:
    interaction = float(_clip(float(confidence.get("interaction", 0.5)), 0.0, 1.0))
    args = force_spec.get("args") or []
    explicit_force = abs(float(args[0])) if len(args) > 0 and isinstance(args[0], (int, float)) else None
    explicit_torque = abs(float(args[1])) if len(args) > 1 and isinstance(args[1], (int, float)) else None

    if contact_intent == "gripper_only":
        return 0.0, explicit_torque
    if force_spec.get("name") == "Off" or contact_intent == "break_contact":
        return 0.0, explicit_torque
    if explicit_force is not None:
        return explicit_force, explicit_torque
    if contact_intent == "maintain_contact":
        return float(GWS_MAINTAIN_FORCE_N * (0.85 + 0.30 * interaction)), explicit_torque
    if contact_intent == "no_contact":
        return float(GWS_APPROACH_FORCE_N * (0.80 + 0.20 * interaction)), explicit_torque
    return float(max(GWS_MIN_NORMAL_FORCE_N, GWS_APPROACH_FORCE_N)), explicit_torque


def _contact_wrench_ray(
    *,
    name: str,
    frame: Dict[str, Any],
    f_n: float,
    f_t1: float,
    f_t2: float,
    tau_n: float = 0.0,
    tau_t1: float = 0.0,
    tau_t2: float = 0.0,
) -> Dict[str, Any]:
    normal = frame["normal_cam"]
    tangent_1 = frame["tangent_1_cam"]
    tangent_2 = frame["tangent_2_cam"]
    force_cam = _vector_add(
        _vector_add(_vector_scale(normal, f_n), _vector_scale(tangent_1, f_t1)),
        _vector_scale(tangent_2, f_t2),
    )
    torque_cam = _vector_add(
        _vector_add(_vector_scale(normal, tau_n), _vector_scale(tangent_1, tau_t1)),
        _vector_scale(tangent_2, tau_t2),
    )
    return {
        "name": str(name),
        "force_local_contact": {
            "normal": float(f_n),
            "tangent_1": float(f_t1),
            "tangent_2": float(f_t2),
        },
        "torque_local_contact": {
            "normal": float(tau_n),
            "tangent_1": float(tau_t1),
            "tangent_2": float(tau_t2),
        },
        "force_cam": force_cam,
        "torque_cam": torque_cam,
    }


def _geometry_gws_wrench_bounds(
    geometry: Dict[str, Any],
    force_spec: Dict[str, Any],
    confidence: Dict[str, Any],
    contact_intent: str,
) -> Optional[Dict[str, Any]]:
    if not geometry.get("affordance_contact_normal_cam") and not geometry.get("affordance_wrench_basis"):
        return None

    frame = _contact_frame_from_geometry(geometry)
    if frame is None:
        return None

    contact_geometry = _normalize_contact_geometry(geometry.get("affordance_contact_geometry"))
    requested_force_n, requested_torque_nm = _requested_force_and_torque(
        force_spec=force_spec,
        contact_intent=contact_intent,
        confidence=confidence,
    )
    if requested_force_n <= 0.0:
        f_n_min = 0.0
        f_n_max = 0.0 if contact_intent in {"break_contact", "gripper_only"} else GWS_MIN_NORMAL_FORCE_N
    else:
        f_n_max = max(float(requested_force_n), GWS_MIN_NORMAL_FORCE_N)
        f_n_min = 0.45 * f_n_max if contact_intent == "maintain_contact" else 0.0

    f_t_max = float(GWS_FRICTION_MU * f_n_max)
    patch = _contact_patch_metrics(geometry)
    tau_n_max = float(f_t_max * patch["point_radius_m"])

    if contact_geometry == "point":
        tau_t1_max = 0.0
        tau_t2_max = 0.0
    elif contact_geometry == "line":
        tau_t1_max = float(f_n_max * patch["line_half_length_m"])
        tau_t2_max = 0.0
    else:
        tau_t1_max = float(f_n_max * patch["surface_half_extent_t1_m"])
        tau_t2_max = float(f_n_max * patch["surface_half_extent_t2_m"])

    tangential_component = float(f_t_max / math.sqrt(2.0)) if f_t_max > 0.0 else 0.0
    sign_pairs = [(1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)]
    rays: List[Dict[str, Any]] = []
    for idx, (s1, s2) in enumerate(sign_pairs):
        rays.append(
            _contact_wrench_ray(
                name=f"cone_{idx + 1}",
                frame=frame,
                f_n=f_n_max,
                f_t1=s1 * tangential_component,
                f_t2=s2 * tangential_component,
            )
        )
    if contact_geometry == "line" and tau_t1_max > 0.0:
        line_rays: List[Dict[str, Any]] = []
        for idx, (s1, s2) in enumerate(sign_pairs):
            for sign_tau in (1.0, -1.0):
                line_rays.append(
                    _contact_wrench_ray(
                        name=f"line_{idx + 1}_{'pos' if sign_tau > 0 else 'neg'}",
                        frame=frame,
                        f_n=f_n_max,
                        f_t1=s1 * tangential_component,
                        f_t2=s2 * tangential_component,
                        tau_t1=sign_tau * tau_t1_max,
                    )
                )
        rays = line_rays
    elif contact_geometry == "surface" and (tau_t1_max > 0.0 or tau_t2_max > 0.0):
        surface_rays: List[Dict[str, Any]] = []
        for idx, (s1, s2) in enumerate(sign_pairs):
            for sign_tau in (1.0, -1.0):
                surface_rays.append(
                    _contact_wrench_ray(
                        name=f"surface_t1_{idx + 1}_{'pos' if sign_tau > 0 else 'neg'}",
                        frame=frame,
                        f_n=f_n_max,
                        f_t1=s1 * tangential_component,
                        f_t2=s2 * tangential_component,
                        tau_t1=sign_tau * tau_t1_max,
                    )
                )
                if tau_t2_max > 0.0:
                    surface_rays.append(
                        _contact_wrench_ray(
                            name=f"surface_t2_{idx + 1}_{'pos' if sign_tau > 0 else 'neg'}",
                            frame=frame,
                            f_n=f_n_max,
                            f_t1=s1 * tangential_component,
                            f_t2=s2 * tangential_component,
                            tau_t2=sign_tau * tau_t2_max,
                        )
                    )
        rays = surface_rays

    tau_t_max = float(max(tau_t1_max, tau_t2_max))
    return {
        "f_n_min": float(f_n_min),
        "f_n_max": float(f_n_max),
        "f_t_max": float(f_t_max),
        "tau_n_max": float(tau_n_max),
        "tau_t_max": float(tau_t_max),
        "frame": "contact_C",
        "target_force_mode": str(force_spec.get("name", "Unknown")),
        "contact_geometry": contact_geometry,
        "force_mode": geometry.get("affordance_force_mode"),
        "friction_mu": float(GWS_FRICTION_MU),
        "provenance": "affordance_geometry_gws",
        "contact_frame_cam": {
            "normal": frame["normal_cam"],
            "tangent_1": frame["tangent_1_cam"],
            "tangent_2": frame["tangent_2_cam"],
            "line_axis": frame["line_axis_cam"],
        },
        "force_bounds_local": {
            "normal": [float(f_n_min), float(f_n_max)],
            "tangent_1": [-float(f_t_max), float(f_t_max)],
            "tangent_2": [-float(f_t_max), float(f_t_max)],
        },
        "torque_bounds_local": {
            "normal": [-float(tau_n_max), float(tau_n_max)],
            "tangent_1": [-float(tau_t1_max), float(tau_t1_max)],
            "tangent_2": [-float(tau_t2_max), float(tau_t2_max)],
        },
        "requested_force_n": float(requested_force_n),
        "requested_torque_nm": None if requested_torque_nm is None else float(requested_torque_nm),
        "patch_geometry": patch,
        "gws_extreme_rays": rays,
        "source_affordance": {
            "contact_normal_3d": deepcopy(geometry.get("affordance_contact_normal_cam")),
            "wrench_basis": deepcopy(geometry.get("affordance_wrench_basis")),
            "geometry_vlm_agreement": geometry.get("geometry_vlm_agreement"),
            "affordance_uncertainty": deepcopy(geometry.get("affordance_uncertainty")),
        },
    }


def _clone_geometry_with_hypothesis(
    geometry: Dict[str, Any],
    hypothesis: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    geom = deepcopy(geometry)
    triple = {}
    if isinstance(hypothesis, dict):
        triple = hypothesis.get("triple", {}) or {}
    if triple.get("contact_geometry") is not None:
        geom["affordance_contact_geometry"] = triple.get("contact_geometry")
    if triple.get("force_mode") is not None:
        geom["affordance_force_mode"] = triple.get("force_mode")
    if triple.get("grasp_type") is not None:
        geom["affordance_grasp_type"] = triple.get("grasp_type")
    return geom


def _requested_task_wrench_local(
    *,
    mode: str,
    force_mode: str,
    force_spec: Dict[str, Any],
    confidence: Dict[str, Any],
    contact_intent: str,
) -> Dict[str, float]:
    force_mag, torque_mag = _requested_force_and_torque(
        force_spec=force_spec,
        contact_intent=contact_intent,
        confidence=confidence,
    )
    mode_name = str(mode or "")
    requested_mode = str(force_mode or "").strip().lower()
    if mode_name == "RotateMove":
        requested_mode = "torque"
        if torque_mag is None or torque_mag <= 0.0:
            torque_mag = max(0.05, 0.15 * max(force_mag, GWS_MIN_NORMAL_FORCE_N))

    req = {
        "mode": requested_mode,
        "normal_force_n": 0.0,
        "tangential_force_n": 0.0,
        "torque_nm": 0.0,
        "normal_preload_n": 0.0,
    }
    if requested_mode in {"push", "normal"}:
        req["normal_force_n"] = float(force_mag)
    elif requested_mode == "pull":
        req["tangential_force_n"] = float(force_mag)
        req["normal_preload_n"] = float(max(GWS_MIN_NORMAL_FORCE_N, 0.25 * force_mag))
    elif requested_mode == "torque":
        req["torque_nm"] = float(torque_mag or 0.0)
        req["normal_preload_n"] = float(max(GWS_MIN_NORMAL_FORCE_N, 0.20 * max(force_mag, 1.0)))
    return req


def _wrench_request_feasibility(
    *,
    bounds: Dict[str, Any],
    mode: str,
    force_mode: str,
    force_spec: Dict[str, Any],
    confidence: Dict[str, Any],
    contact_intent: str,
) -> Dict[str, Any]:
    req = _requested_task_wrench_local(
        mode=mode,
        force_mode=force_mode,
        force_spec=force_spec,
        confidence=confidence,
        contact_intent=contact_intent,
    )
    normal_lo, normal_hi = (bounds.get("force_bounds_local", {}) or {}).get("normal", [0.0, float(bounds.get("f_n_max", 0.0) or 0.0)])
    tangent_lo, tangent_hi = (bounds.get("force_bounds_local", {}) or {}).get("tangent_1", [-(float(bounds.get("f_t_max", 0.0) or 0.0)), float(bounds.get("f_t_max", 0.0) or 0.0)])
    torque_n_lo, torque_n_hi = (bounds.get("torque_bounds_local", {}) or {}).get("normal", [-(float(bounds.get("tau_n_max", 0.0) or 0.0)), float(bounds.get("tau_n_max", 0.0) or 0.0)])
    torque_t1_lo, torque_t1_hi = (bounds.get("torque_bounds_local", {}) or {}).get("tangent_1", [-(float(bounds.get("tau_t_max", 0.0) or 0.0)), float(bounds.get("tau_t_max", 0.0) or 0.0)])

    checks: List[bool] = []
    semantic_compatible = (
        req["mode"] == str(force_mode or "").strip().lower()
        or (req["mode"] in {"push", "normal"} and str(force_mode or "").strip().lower() in {"push", "normal"})
    )
    if str(mode or "") == "RotateMove":
        checks.append(bool(semantic_compatible))
    if req["normal_force_n"] > 0.0:
        checks.append(float(normal_lo) - 1e-9 <= req["normal_force_n"] <= float(normal_hi) + 1e-9)
    if req["normal_preload_n"] > 0.0:
        checks.append(req["normal_preload_n"] <= float(normal_hi) + 1e-9)
    if req["tangential_force_n"] > 0.0:
        checks.append(abs(req["tangential_force_n"]) <= max(abs(float(tangent_lo)), abs(float(tangent_hi))) + 1e-9)
    if req["torque_nm"] > 0.0:
        torque_cap = max(abs(float(torque_n_lo)), abs(float(torque_n_hi)), abs(float(torque_t1_lo)), abs(float(torque_t1_hi)))
        checks.append(abs(req["torque_nm"]) <= torque_cap + 1e-9)
    feasible = all(checks) if checks else True
    return {
        "feasible": bool(feasible),
        "requested_wrench_local": req,
        "semantic_compatible": bool(semantic_compatible),
        "normal_bounds": [float(normal_lo), float(normal_hi)],
        "tangent_bounds": [float(tangent_lo), float(tangent_hi)],
        "torque_normal_bounds": [float(torque_n_lo), float(torque_n_hi)],
        "torque_tangent_bounds": [float(torque_t1_lo), float(torque_t1_hi)],
    }


def _select_affordance_hypothesis_for_action(
    *,
    geometry: Dict[str, Any],
    mode: str,
    force_spec: Dict[str, Any],
    confidence: Dict[str, Any],
    contact_intent: str,
) -> Dict[str, Any]:
    hypotheses = list(geometry.get("affordance_hypotheses") or [])
    if not hypotheses:
        fallback_hyp = {
            "triple": {
                "grasp_type": geometry.get("affordance_grasp_type"),
                "contact_geometry": geometry.get("affordance_contact_geometry"),
                "force_mode": geometry.get("affordance_force_mode"),
            },
            "adjusted_probability": 1.0,
            "raw_probability": 1.0,
            "coherence": {
                "status": "fallback",
                "weight": 1.0,
            },
        }
        hypotheses = [fallback_hyp]

    evaluations: List[Dict[str, Any]] = []
    for idx, hyp in enumerate(hypotheses):
        triple = dict(hyp.get("triple", {}) or {})
        geom_h = _clone_geometry_with_hypothesis(geometry, hyp)
        bounds = _geometry_gws_wrench_bounds(
            geometry=geom_h,
            force_spec=force_spec,
            confidence=confidence,
            contact_intent=contact_intent,
        )
        if bounds is None:
            evaluations.append({
                "rank": idx + 1,
                "hypothesis": hyp,
                "feasible": False,
                "reason": "wrench_bounds_unavailable",
                "selection_score": -1.0,
            })
            continue
        feasibility = _wrench_request_feasibility(
            bounds=bounds,
            mode=mode,
            force_mode=str(triple.get("force_mode", geometry.get("affordance_force_mode", "push"))),
            force_spec=force_spec,
            confidence=confidence,
            contact_intent=contact_intent,
        )
        selection_score = float(hyp.get("adjusted_probability", hyp.get("adjusted_score", 0.0)) or 0.0)
        evaluations.append({
            "rank": idx + 1,
            "hypothesis": hyp,
            "wrench_bounds": bounds,
            "feasibility": feasibility,
            "feasible": bool(feasibility.get("feasible", False)),
            "selection_score": selection_score,
        })

    feasible = [item for item in evaluations if bool(item.get("feasible"))]
    if feasible:
        feasible.sort(key=lambda item: float(item.get("selection_score", 0.0)), reverse=True)
        selected = feasible[0]
        selection_policy = "highest_probability_feasible"
    else:
        evaluations.sort(key=lambda item: float(item.get("selection_score", 0.0)), reverse=True)
        selected = evaluations[0]
        selection_policy = "fallback_highest_probability"

    return {
        "selected": selected,
        "evaluations": evaluations,
        "selection_policy": selection_policy,
        "num_hypotheses": len(evaluations),
    }


def _scale_wrench_bounds(
    base: Dict[str, Any],
    force_spec: Dict[str, Any],
    confidence: Dict[str, Any],
    contact_intent: str,
) -> Dict[str, Any]:
    out = deepcopy(base)
    interaction = float(confidence.get("interaction", 0.5))

    if force_spec.get("name") == "Off" or contact_intent == "break_contact":
        out.update(
            {
                "f_n_min": 0.0,
                "f_n_max": min(float(out.get("f_n_max", 0.5)), 0.5),
                "f_t_max": min(float(out.get("f_t_max", 0.5)), 0.5),
                "tau_n_max": min(float(out.get("tau_n_max", 0.05)), 0.05),
                "tau_t_max": min(float(out.get("tau_t_max", 0.05)), 0.05),
            }
        )
        out["target_force_mode"] = "off"
        return out

    scale = 0.7 + 0.6 * interaction
    target_force = None
    target_torque = None
    if force_spec.get("name") in {"Const", "Thresh"}:
        args = force_spec.get("args") or []
        if len(args) > 0 and isinstance(args[0], (int, float)):
            target_force = float(args[0])
        if len(args) > 1 and isinstance(args[1], (int, float)):
            target_torque = float(args[1])

    if target_force is not None:
        out["f_n_target"] = float(target_force)
        out["f_n_min"] = max(0.0, 0.7 * target_force * scale)
        out["f_n_max"] = max(out["f_n_min"], 1.3 * target_force * scale)
        out["f_t_max"] = max(float(out.get("f_t_max", 0.0)) * scale, 0.5 * target_force * scale)

    if target_torque is not None:
        tau_budget = max(0.02, abs(target_torque) * scale if target_torque != 0 else float(out.get("tau_n_max", 0.05)) * scale)
        out["tau_n_max"] = tau_budget
        out["tau_t_max"] = max(float(out.get("tau_t_max", 0.0)), 0.8 * tau_budget)

    out["target_force_mode"] = str(force_spec.get("name", "Unknown"))
    return {k: float(v) if isinstance(v, (int, float)) else v for k, v in out.items()}


def _compile_action_entry(
    step_id: int,
    action_id: int,
    motion_primitive: str,
    force_primitive: str,
    toolmanip_vlm2_payload: Dict[str, Any],
    vlm3_payload: Dict[str, Any],
    templates_payload: Dict[str, Any],
    wrench_profiles_payload: Dict[str, Any],
    confidence: Dict[str, Any],
) -> Dict[str, Any]:
    motion_spec = _parse_motion_primitive(motion_primitive)
    force_spec = _parse_force_primitive(force_primitive)
    mode, template = _load_template_and_mode(motion_spec, templates_payload)
    contact_intent = str(template.get("contact_intent", "unknown"))
    force_class = _infer_force_class(force_spec, contact_intent=contact_intent)

    geometry = _geometry_for_motion(motion_spec, toolmanip_vlm2_payload, vlm3_payload)
    hypothesis_selection = _select_affordance_hypothesis_for_action(
        geometry=geometry,
        mode=mode,
        force_spec=force_spec,
        confidence=confidence,
        contact_intent=contact_intent,
    )
    selected_hypothesis_eval = dict(hypothesis_selection.get("selected") or {})
    selected_hypothesis = dict(selected_hypothesis_eval.get("hypothesis") or {})
    selected_wrench_bounds = deepcopy(selected_hypothesis_eval.get("wrench_bounds"))

    profiles = (wrench_profiles_payload.get("profiles") or {}).get(mode, {})
    base_wrench = deepcopy(profiles.get(force_class, {}))
    wrench_bounds = selected_wrench_bounds
    if wrench_bounds is None:
        wrench_bounds = _scale_wrench_bounds(
            base_wrench,
            force_spec=force_spec,
            confidence=confidence,
            contact_intent=contact_intent,
        )
        wrench_bounds["frame"] = "target_P"
        wrench_bounds["provenance"] = "profile_scaled_fallback"
    wrench_bounds["force_class"] = force_class

    theta = _confidence_schedule(
        deepcopy(template.get("theta_defaults", {})),
        confidence=confidence,
        mode=mode,
    )

    return {
        "step_id": int(step_id),
        "action_id": int(action_id),
        "label": f"step{int(step_id)}_action{int(action_id)}",
        "mode": mode,
        "contact_intent": contact_intent,
        "symbolic_action": {
            "motion": motion_primitive,
            "force": force_primitive,
            "motion_spec": motion_spec,
            "force_spec": force_spec,
        },
        "geometry": geometry,
        "constraints": {
            "terminal": list(template.get("terminal_constraints", [])),
            "path": list(template.get("path_constraints", [])),
        },
        "selected_affordance_hypothesis": selected_hypothesis,
        "affordance_hypothesis_selection": {
            "selection_policy": hypothesis_selection.get("selection_policy"),
            "selected_feasible": bool((selected_hypothesis_eval.get("feasible")) if selected_hypothesis_eval else False),
            "selected_feasibility": deepcopy(selected_hypothesis_eval.get("feasibility")),
            "evaluations": deepcopy(hypothesis_selection.get("evaluations", [])),
        },
        "wrench_bounds": wrench_bounds,
        "theta": theta,
    }


def _make_residual_spec(name: str, phase: str, action: Dict[str, Any]) -> Dict[str, Any]:
    theta = action.get("theta", {}) or {}
    wrench_bounds = action.get("wrench_bounds", {}) or {}
    geometry = action.get("geometry", {}) or {}

    weight_key_map = {
        "frame_alignment": "frame_align_weight",
        "axis_alignment": "frame_align_weight",
        "reach_pose": "pose_weight",
        "reach_line_end": "progress_weight",
        "reach_rotation_target": "rotation_weight",
        "contact_band": "frame_align_weight",
        "normal_alignment": "frame_align_weight",
        "line_following": "progress_weight",
        "progress_monotonic": "progress_weight",
        "clearance_increase": "release_weight",
        "clearance_growth": "release_weight",
        "wrench_limits": "wrench_weight",
        "collision_avoidance": "path_weight",
        "smoothness": "path_weight",
        "ik_feasibility": "path_weight",
        "gripper_state": "gripper_weight",
    }
    tolerance_key_map = {
        "contact_band": "contact_band_m",
        "normal_alignment": "normal_align_tol_deg",
        "axis_alignment": "axis_align_tol_deg",
        "clearance_growth": "clearance_margin_m",
    }
    role_map = {
        "frame_alignment": "pose_to_frame_alignment",
        "axis_alignment": "axis_alignment",
        "reach_pose": "terminal_reach",
        "reach_line_end": "terminal_progress_target",
        "reach_rotation_target": "terminal_rotation_target",
        "contact_band": "contact_consistency",
        "normal_alignment": "contact_normal_alignment",
        "line_following": "primitive_following",
        "progress_monotonic": "progress_regularizer",
        "clearance_increase": "terminal_clearance_target",
        "clearance_growth": "clearance_release_regularizer",
        "wrench_limits": "wrench_limit_compliance",
        "collision_avoidance": "shared_feasibility_regularizer",
        "smoothness": "shared_smoothness_regularizer",
        "ik_feasibility": "shared_kinematic_regularizer",
        "gripper_state": "end_effector_state",
    }

    spec = {
        "name": str(name),
        "phase": str(phase),
        "role": role_map.get(name, "residual"),
        "enabled": True,
        "weight_key": weight_key_map.get(name),
        "weight": float(theta.get(weight_key_map[name], 1.0)) if weight_key_map.get(name) in theta else 1.0,
    }

    tol_key = tolerance_key_map.get(name)
    if tol_key in theta:
        spec["tolerance_key"] = tol_key
        spec["tolerance"] = deepcopy(theta[tol_key])

    if name in {"frame_alignment", "line_following", "progress_monotonic", "reach_line_end"}:
        spec["geometry_ref"] = "Li.line"
    elif name in {"contact_band", "normal_alignment", "axis_alignment"}:
        spec["geometry_ref"] = "Li.contact_geometry"
    elif name in {"reach_pose", "clearance_increase", "clearance_growth"}:
        spec["geometry_ref"] = "Li.target_pose"

    if name == "wrench_limits":
        spec["wrench_ref"] = "Wi"
        spec["frame"] = wrench_bounds.get("frame", "target_P")
        spec["target_force_mode"] = wrench_bounds.get("target_force_mode", "Unknown")

    if name == "progress_monotonic":
        spec["target_span_m"] = float(geometry.get("progress_span_m", 0.0) or 0.0)

    return spec


def _build_constraint_program(
    step_id: int,
    actions: List[Dict[str, Any]],
    frame_candidates: List[Dict[str, Any]],
    global_frame_id: str,
) -> Optional[Dict[str, Any]]:
    if not actions:
        return None

    motion_actions = [action for action in actions if str(action.get("mode", "")) != "GripperCommand"]
    primary_action = motion_actions[-1] if motion_actions else actions[-1]
    auxiliary_actions = [
        {
            "action_id": int(action.get("action_id", 0)),
            "label": action.get("label"),
            "mode": action.get("mode"),
            "motion": ((action.get("symbolic_action") or {}).get("motion")),
            "force": ((action.get("symbolic_action") or {}).get("force")),
        }
        for action in actions
        if action is not primary_action
    ]
    geometry = deepcopy(primary_action.get("geometry", {}))
    line_program = {
        "start_cam": deepcopy(geometry.get("line_start_cam")),
        "end_cam": deepcopy(geometry.get("line_end_cam")),
        "direction_cam": deepcopy(geometry.get("line_dir_cam")),
        "progress_span_m": float(geometry.get("progress_span_m", 0.0) or 0.0),
    }
    contact_geometry = {
        "contact_point_P": deepcopy(geometry.get("contact_point_P")),
        "target_touch_cam": deepcopy(geometry.get("target_touch_cam")),
        "tool_touch_cam": deepcopy(geometry.get("tool_touch_cam")),
        "surface_normal_cam": deepcopy(geometry.get("surface_normal_cam")),
        "tool_axis_cam": deepcopy(geometry.get("tool_axis_cam")),
        "affordance_contact_geometry": geometry.get("affordance_contact_geometry"),
        "affordance_force_mode": geometry.get("affordance_force_mode"),
        "selected_affordance_hypothesis": deepcopy(primary_action.get("selected_affordance_hypothesis")),
        "affordance_contact_normal_cam": deepcopy(geometry.get("affordance_contact_normal_cam")),
        "affordance_wrench_basis": deepcopy(geometry.get("affordance_wrench_basis")),
        "geometry_vlm_agreement": geometry.get("geometry_vlm_agreement"),
    }
    target_pose = {
        "symbolic_refs": deepcopy(geometry.get("symbolic_refs", {})),
        "tool_frame_cam": deepcopy(geometry.get("tool_frame_cam")),
        "tool_grip_cam": deepcopy(geometry.get("tool_grip_cam")),
        "grasp_frame_cam": deepcopy(geometry.get("grasp_frame_cam")),
        "grasp_quaternion_xyzw": deepcopy(geometry.get("grasp_quaternion_xyzw")),
        "grasp_width_m": deepcopy(geometry.get("grasp_width_m")),
        "grasp_width_mm": deepcopy(geometry.get("grasp_width_mm")),
        "grasp_width_px": deepcopy(geometry.get("grasp_width_px")),
    }
    if geometry.get("rotation_axis_cam") is not None:
        contact_geometry["rotation_axis_cam"] = deepcopy(geometry.get("rotation_axis_cam"))
    if geometry.get("rotation_center") is not None:
        contact_geometry["rotation_center"] = deepcopy(geometry.get("rotation_center"))
    if geometry.get("rotation_target") is not None:
        target_pose["rotation_target"] = deepcopy(geometry.get("rotation_target"))

    terminal_residuals = [
        _make_residual_spec(name, "terminal", primary_action)
        for name in (primary_action.get("constraints", {}) or {}).get("terminal", [])
    ]
    path_residuals = [
        _make_residual_spec(name, "path", primary_action)
        for name in (primary_action.get("constraints", {}) or {}).get("path", [])
    ]
    shared_regularizers = [
        residual
        for residual in path_residuals
        if residual["name"] in {"collision_avoidance", "smoothness", "ik_feasibility"}
    ]

    symbolic_plan = {
        "primary_motion": ((primary_action.get("symbolic_action") or {}).get("motion")),
        "primary_force": ((primary_action.get("symbolic_action") or {}).get("force")),
        "primary_action_label": primary_action.get("label"),
        "auxiliary_actions": auxiliary_actions,
        "action_trace": [
            {
                "action_id": int(action.get("action_id", 0)),
                "label": action.get("label"),
                "mode": action.get("mode"),
                "motion": ((action.get("symbolic_action") or {}).get("motion")),
                "force": ((action.get("symbolic_action") or {}).get("force")),
            }
            for action in actions
        ],
    }

    return {
        "program_id": f"Pi_{int(step_id)}",
        "mode": primary_action.get("mode"),
        "contact_intent": primary_action.get("contact_intent"),
        "symbolic_plan": symbolic_plan,
        "Fi": {
            "frame_id": global_frame_id,
            "selected_hypothesis": deepcopy(frame_candidates[0]) if frame_candidates else None,
            "alternatives": deepcopy(frame_candidates[1:]) if len(frame_candidates) > 1 else [],
            "execution_frame": "target_P",
            "frame_selection_policy": "best_plus_alternatives",
        },
        "Li": {
            "line": line_program,
            "contact_geometry": contact_geometry,
            "target_pose": target_pose,
        },
        "Ci_term": {
            "residuals": terminal_residuals,
            "objective": "terminal_mode_goal_satisfaction",
        },
        "Ci_path": {
            "residuals": path_residuals,
            "shared_regularizers": shared_regularizers,
            "objective": "receding_horizon_path_feasibility",
        },
        "Wi": deepcopy(primary_action.get("wrench_bounds", {})),
        "Theta_i": deepcopy(primary_action.get("theta", {})),
        "trace": {
            "primary_action_id": int(primary_action.get("action_id", 0)),
            "primary_action_label": primary_action.get("label"),
            "num_actions_in_step": len(actions),
            "selected_affordance_hypothesis": deepcopy(primary_action.get("selected_affordance_hypothesis")),
            "affordance_hypothesis_selection": deepcopy(primary_action.get("affordance_hypothesis_selection")),
        },
    }


def _build_paper_view(full_program: Dict[str, Any]) -> Dict[str, Any]:
    def _paper_source(source_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "vlm3_results_json": source_payload.get("vlm3_results_json"),
            "high_level_plan_json": source_payload.get("toolmanip_plan_json"),
            "symbolic_grounding_json": source_payload.get("toolmanip_vlm2_json"),
            "action_templates_json": source_payload.get("action_templates_json"),
            "wrench_profiles_json": source_payload.get("wrench_profiles_json"),
            "confidence_json": source_payload.get("confidence_json"),
        }

    def _paper_high_level_step(pi_payload: Dict[str, Any]) -> Dict[str, Any]:
        symbolic_plan = pi_payload.get("symbolic_plan", {}) or {}
        return {
            "interaction_intent": symbolic_plan.get("primary_motion"),
            "wrench_intent": symbolic_plan.get("primary_force"),
            "auxiliary_intents": deepcopy(symbolic_plan.get("auxiliary_actions", [])),
        }

    def _paper_subgoal_program(pi_payload: Dict[str, Any]) -> Dict[str, Any]:
        ci_term = pi_payload.get("Ci_term", {}) or {}
        return {
            "type": "terminal_constraint_manifold",
            "objective": ci_term.get("objective"),
            "terminal_residuals": deepcopy(ci_term.get("residuals", [])),
        }

    def _paper_path_program(pi_payload: Dict[str, Any]) -> Dict[str, Any]:
        ci_path = pi_payload.get("Ci_path", {}) or {}
        shared_names = {
            str(item.get("name"))
            for item in (ci_path.get("shared_regularizers", []) or [])
            if isinstance(item, dict) and item.get("name") is not None
        }
        residuals = ci_path.get("residuals", []) or []
        mode_residuals = [
            deepcopy(item)
            for item in residuals
            if not (isinstance(item, dict) and str(item.get("name")) in shared_names)
        ]
        return {
            "type": "path_constraint_bundle",
            "objective": ci_path.get("objective"),
            "mode_residuals": mode_residuals,
            "shared_regularizers": deepcopy(ci_path.get("shared_regularizers", [])),
        }

    def _paper_pi(pi_payload: Dict[str, Any]) -> Dict[str, Any]:
        li_payload = deepcopy(pi_payload.get("Li", {}))
        target_pose = li_payload.get("target_pose")
        if isinstance(target_pose, dict):
            target_pose.pop("symbolic_grounding_refs", None)
        return {
            "program_id": pi_payload.get("program_id"),
            "mode": pi_payload.get("mode"),
            "contact_intent": pi_payload.get("contact_intent"),
            "Fi": deepcopy(pi_payload.get("Fi", {})),
            "Li": li_payload,
            "Wi": deepcopy(pi_payload.get("Wi", {})),
            "Theta_i": deepcopy(pi_payload.get("Theta_i", {})),
        }

    global_payload = full_program.get("global", {}) or {}
    best_hypothesis = global_payload.get("best_hypothesis", {}) or {}
    confidence = global_payload.get("confidence", {}) or {}

    programs: List[Dict[str, Any]] = []
    for step in full_program.get("steps", []):
        constraint_program = step.get("constraint_program")
        if not isinstance(constraint_program, dict):
            continue
        programs.append(
            {
                "step_id": int(step.get("step_id", 0)),
                "step_name": step.get("step_name"),
                "high_level_step": _paper_high_level_step(constraint_program),
                "subgoal_program": _paper_subgoal_program(constraint_program),
                "path_program": _paper_path_program(constraint_program),
                "Pi": _paper_pi(constraint_program),
            }
        )

    return {
        "version": full_program.get("version", "0.2"),
        "view": "paper",
        "notation": "Pi = <Fi, Li, Ci_term, Ci_path, Wi, Theta_i>",
        "method": "RegioManip action constraint synthesis",
        "scene_summary": {
            "tool": best_hypothesis.get("tool_name"),
            "target": best_hypothesis.get("target_name"),
            "pattern": best_hypothesis.get("pattern"),
            "selected_frame": global_payload.get("frame_id"),
            "interaction_confidence": confidence.get("interaction"),
            "perception_confidence": confidence.get("perception"),
        },
        "constraint_programs": programs,
        "source": _paper_source(full_program.get("source", {}) or {}),
        "output_path": full_program.get("output_path"),
    }


def compile_action_program(
    vlm3_results_json: str,
    toolmanip_plan_json: str,
    toolmanip_vlm2_json: str,
    action_templates_json: str,
    wrench_profiles_json: str,
    out_path: Optional[str] = None,
    confidence_json: Optional[str] = None,
) -> Dict[str, Any]:
    vlm3_payload = _load_json(vlm3_results_json)
    toolmanip_plan_payload = _load_json(toolmanip_plan_json)
    toolmanip_vlm2_payload = _load_json(toolmanip_vlm2_json)
    templates_payload = _load_json(action_templates_json)
    wrench_profiles_payload = _load_json(wrench_profiles_json)
    confidence_payload = _load_json(confidence_json) if confidence_json else None
    confidence = _extract_confidence(confidence_payload, vlm3_payload)
    frame_candidates = _collect_frame_candidates(vlm3_payload)
    global_frame_id = "tool_target_best"

    steps: List[Dict[str, Any]] = []
    step_keys = sorted(
        toolmanip_plan_payload.keys(),
        key=lambda key: int(re.search(r"(\d+)", str(key)).group(1)) if re.search(r"(\d+)", str(key)) else 0,
    )
    for step_key in step_keys:
        step_match = re.search(r"(\d+)", str(step_key))
        if step_match is None:
            continue
        step_id = int(step_match.group(1))
        step_payload = toolmanip_plan_payload[step_key]
        action_keys = sorted(
            [key for key in step_payload.keys() if str(key).startswith("motion")],
            key=lambda x: int(re.search(r"(\d+)", x).group(1)) if re.search(r"(\d+)", x) else 0,
        )
        actions: List[Dict[str, Any]] = []
        for motion_key in action_keys:
            idx_match = re.search(r"(\d+)", motion_key)
            if idx_match is None:
                continue
            action_id = int(idx_match.group(1))
            force_key = f"force{action_id}"
            actions.append(
                _compile_action_entry(
                    step_id=step_id,
                    action_id=action_id,
                    motion_primitive=str(step_payload.get(motion_key, "")),
                    force_primitive=str(step_payload.get(force_key, "Off")),
                    toolmanip_vlm2_payload=toolmanip_vlm2_payload,
                    vlm3_payload=vlm3_payload,
                    templates_payload=templates_payload,
                    wrench_profiles_payload=wrench_profiles_payload,
                    confidence=confidence,
                )
            )
        steps.append(
            {
                "step_id": step_id,
                "step_name": step_key,
                "actions": actions,
                "constraint_program": _build_constraint_program(
                    step_id=step_id,
                    actions=actions,
                    frame_candidates=frame_candidates,
                    global_frame_id=global_frame_id,
                ),
            }
        )

    output = {
        "version": "0.2",
        "source": {
            "vlm3_results_json": os.path.abspath(vlm3_results_json),
            "toolmanip_plan_json": os.path.abspath(toolmanip_plan_json),
            "toolmanip_vlm2_json": os.path.abspath(toolmanip_vlm2_json),
            "action_templates_json": os.path.abspath(action_templates_json),
            "wrench_profiles_json": os.path.abspath(wrench_profiles_json),
            "confidence_json": None if not confidence_json else os.path.abspath(confidence_json),
        },
        "global": {
            "frame_id": global_frame_id,
            "frame_candidates": frame_candidates,
            "confidence": confidence,
            "frames_cam": deepcopy(vlm3_payload.get("frames_cam")),
            "target_axes_P": deepcopy(vlm3_payload.get("target_axes_P")),
            "contact_points_P": deepcopy(vlm3_payload.get("contact_points_P")),
            "best_hypothesis": deepcopy(vlm3_payload.get("best_hypothesis")),
        },
        "steps": steps,
        "compiler_notes": {
            "representation": "mode_conditioned_constraint_program",
            "notation": "Pi = <Fi, Li, Ci_term, Ci_path, Wi, Theta_i>",
            "backward_compatible_action_trace": True,
        },
    }

    out_file = out_path
    if not out_file:
        out_file = os.path.join(os.path.dirname(os.path.abspath(vlm3_results_json)), "compiled_action_program.json")
    output["output_path"] = os.path.abspath(out_file)
    _save_json(out_file, output)
    paper_out_file = os.path.join(os.path.dirname(os.path.abspath(out_file)), "compiled_action_program_paper.json")
    output["paper_output_path"] = os.path.abspath(paper_out_file)
    _save_json(paper_out_file, _build_paper_view(output))
    return output


def main() -> None:
    package_dir = Path(__file__).resolve().parents[1]
    default_templates = str(package_dir / "configs" / "action_templates.json")
    default_wrench = str(package_dir / "configs" / "wrench_profiles.json")
    default_high_level_plan = str(package_dir / "data" / "high_level_plan.json")
    default_symbolic_grounding = str(package_dir / "data" / "symbolic_grounding.json")

    ap = argparse.ArgumentParser(description="Compile RegioManip action intents into grounded action-constraint programs")
    ap.add_argument("--vlm3_results_json", required=True)
    ap.add_argument(
        "--toolmanip_plan_json",
        "--high_level_plan_json",
        dest="toolmanip_plan_json",
        default=default_high_level_plan,
        help="Path to the high-level action plan JSON",
    )
    ap.add_argument(
        "--toolmanip_vlm2_json",
        "--symbolic_grounding_json",
        dest="toolmanip_vlm2_json",
        default=default_symbolic_grounding,
        help="Path to the symbolic grounding JSON",
    )
    ap.add_argument("--action_templates_json", default=default_templates)
    ap.add_argument("--wrench_profiles_json", default=default_wrench)
    ap.add_argument("--confidence_json", default=None)
    ap.add_argument("--out_path", default=None)
    ap.add_argument("--view", choices=["full", "paper"], default="full")
    args = ap.parse_args()

    out = compile_action_program(
        vlm3_results_json=args.vlm3_results_json,
        toolmanip_plan_json=args.toolmanip_plan_json,
        toolmanip_vlm2_json=args.toolmanip_vlm2_json,
        action_templates_json=args.action_templates_json,
        wrench_profiles_json=args.wrench_profiles_json,
        out_path=args.out_path,
        confidence_json=args.confidence_json,
    )
    payload = _build_paper_view(out) if args.view == "paper" else out
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
