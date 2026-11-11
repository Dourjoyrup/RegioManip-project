"""Frame-hypothesis helpers for RegioManip VLM3 lifting."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .primitive_fitting import unit


def tool_axis_and_line_scores(
    pattern: str,
    tool_axis: np.ndarray,
    work_edge_axis: np.ndarray,
    target_line_axis: np.ndarray,
) -> Dict[str, float]:
    tool_axis = unit(tool_axis)
    work_edge_axis = unit(work_edge_axis)
    target_line_axis = unit(target_line_axis)
    axis_align = float(abs(np.dot(tool_axis, target_line_axis)))
    line_align = float(abs(np.dot(work_edge_axis, target_line_axis)))
    if pattern == "T":
        score = 0.70 * line_align + 0.30 * axis_align
    else:
        score = 0.75 * axis_align + 0.25 * line_align
    return {
        "tool_axis_target_align": axis_align,
        "work_line_target_align": line_align,
        "alignment_score": float(score),
    }


def se3(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=np.float32)
    T[:3, :3] = np.asarray(R, dtype=np.float32).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float32).reshape(3)
    return T


def inv_se3(T: np.ndarray) -> np.ndarray:
    R = T[:3, :3]
    t = T[:3, 3]
    Ti = np.eye(4, dtype=np.float32)
    Ti[:3, :3] = R.T
    Ti[:3, 3] = -R.T @ t
    return Ti


def project_to_plane_axis(axis: np.ndarray, normal: np.ndarray) -> np.ndarray:
    a = np.asarray(axis, dtype=np.float32).reshape(3)
    n = unit(normal)
    a = a - float(np.dot(a, n)) * n
    return unit(a)


def make_rotation(primary: np.ndarray, normal: np.ndarray) -> Optional[np.ndarray]:
    x = unit(primary)
    z = unit(normal)
    if float(np.linalg.norm(x)) < 1e-6 or float(np.linalg.norm(z)) < 1e-6:
        return None
    y = unit(np.cross(z, x))
    if float(np.linalg.norm(y)) < 1e-6:
        return None
    x = unit(np.cross(y, z))
    return np.stack([x, y, z], axis=1).astype(np.float32)


def manuscript_contact_frame_hypotheses(
    axis: np.ndarray,
    normal: np.ndarray,
    translation: np.ndarray,
    approach_along_negative_normal: bool = True,
) -> List[Dict[str, Any]]:
    """Construct the sign-resolved SE(3) hypotheses from Section 3.3.3.

    The manuscript defines ``t = a x n`` and ``R = [eps_t t | a | eps_n n]``.
    We first Gram--Schmidt the measured normal against the tool axis so noisy
    PCA estimates still yield an orthonormal basis.  All four sign pairs are
    reported, but only positive-determinant rotations are marked valid.
    """
    a = unit(axis)
    n_raw = np.asarray(normal, dtype=np.float32).reshape(3)
    if float(np.linalg.norm(a)) < 1e-6:
        return []
    n = unit(n_raw - float(np.dot(n_raw, a)) * a)
    if float(np.linalg.norm(n)) < 1e-6:
        return []
    t = unit(np.cross(a, n))
    if float(np.linalg.norm(t)) < 1e-6:
        return []

    origin = np.asarray(translation, dtype=np.float32).reshape(3)
    hypotheses: List[Dict[str, Any]] = []
    for eps_t in (1, -1):
        for eps_n in (1, -1):
            R = np.stack([float(eps_t) * t, a, float(eps_n) * n], axis=1).astype(np.float32)
            det = float(np.linalg.det(R))
            orthogonality_error = float(np.linalg.norm(R.T @ R - np.eye(3), ord="fro"))
            proper = bool(det > 0.0 and abs(det - 1.0) <= 1e-4 and orthogonality_error <= 1e-4)
            surface_preferred = bool(
                proper and approach_along_negative_normal and eps_n == -1
            )
            hypotheses.append(
                {
                    "hypothesis_id": f"et{eps_t:+d}_en{eps_n:+d}",
                    "tangent_sign": int(eps_t),
                    "normal_sign": int(eps_n),
                    "rotation": R,
                    "translation_xyz": origin,
                    "determinant": det,
                    "orthogonality_error": orthogonality_error,
                    "proper_rotation": proper,
                    "surface_approach_preferred": surface_preferred,
                    "rejection_reason": None if proper else "reflection_det_negative",
                }
            )
    return hypotheses


def section33_confidence_components(
    tool: Dict[str, Any],
    residual_p90_m: float,
    residual_limit_m: float = 0.015,
) -> Dict[str, Any]:
    """Compute the four auditable terms in manuscript Equation (7)."""
    psi = float(np.clip(float(tool.get("geometry_vlm_agreement", 0.0) or 0.0), 0.0, 1.0))
    uncertainty = dict(tool.get("affordance_uncertainty") or {})
    attr_uncertainties = [
        float(uncertainty[k])
        for k in ("grasp_type", "contact_geometry", "force_mode")
        if uncertainty.get(k) is not None
    ]
    # Eq. (5): u_k = 1 - eta_k * psi.  Recover eta when legacy outputs do
    # not store the per-attribute VLM confidence explicitly.
    if psi > 1e-8 and attr_uncertainties:
        eta_values = [float(np.clip((1.0 - u) / psi, 0.0, 1.0)) for u in attr_uncertainties]
        eta_bar = float(np.mean(eta_values))
        eta_source = "derived_from_eq5_uncertainty"
    else:
        eta_bar = 0.0
        eta_source = "unavailable"

    affordance_hypotheses = list(tool.get("affordance_hypotheses") or [])
    p_h = float(
        np.clip(
            float(affordance_hypotheses[0].get("adjusted_probability", 0.0) or 0.0)
            if affordance_hypotheses else 0.0,
            0.0,
            1.0,
        )
    )
    q_geo = float(max(0.0, 1.0 - float(residual_p90_m) / float(residual_limit_m)))
    weights = {"psi": 0.30, "eta_vlm_mean": 0.30, "p_h": 0.25, "q_geo": 0.15}
    score = float(
        weights["psi"] * psi
        + weights["eta_vlm_mean"] * eta_bar
        + weights["p_h"] * p_h
        + weights["q_geo"] * q_geo
    )
    return {
        "geometry_vlm_agreement": psi,
        "mean_vlm_confidence": eta_bar,
        "mean_vlm_confidence_source": eta_source,
        "affordance_hypothesis_probability": p_h,
        "geometric_quality": q_geo,
        "primitive_residual_p90_m": float(residual_p90_m),
        "primitive_residual_limit_m": float(residual_limit_m),
        "weights": weights,
        "score": score,
    }


def build_tool_target_frames(
    tool: Dict[str, Any],
    target: Dict[str, Any],
    intr: Dict[str, float],
    tangent_sign: float = 1.0,
    normal_sign: float = 1.0,
    world_up_cam: Optional[np.ndarray] = None,
    project_xyz_fn=None,
) -> Optional[Dict[str, Any]]:
    plane = target.get("plane")
    edge_line = target.get("edge_line")
    work_edge = tool.get("work_edge")
    if not isinstance(plane, dict) or not isinstance(edge_line, dict) or not isinstance(work_edge, dict):
        return None

    t_P = np.array(target.get("touch_xyz", []), dtype=np.float32)
    t_T = np.array(tool.get("touch_xyz", []), dtype=np.float32)
    grip_T = np.array(tool.get("grip_xyz", []), dtype=np.float32)
    if t_P.shape != (3,) or t_T.shape != (3,) or grip_T.shape != (3,):
        return None

    # The target frame has an invariant semantic convention: +zP is the
    # outward surface normal.  Grasp/approach sign hypotheses must never flip
    # the target frame; they are represented separately as approach=-zP.
    zP = unit(np.array(plane.get("normal", []), dtype=np.float32))
    normal_flipped = False
    normal_orientation_source = "raw_plane_normal"
    up_cam = None if world_up_cam is None else unit(np.array(world_up_cam, dtype=np.float32))
    if up_cam is not None and float(np.linalg.norm(up_cam)) > 1e-6:
        if float(np.dot(zP, up_cam)) < 0.0:
            zP = -zP
            normal_flipped = True
        normal_orientation_source = "calibrated_world_up_cam"
    elif project_xyz_fn is not None:
        zp_origin_xy = project_xyz_fn(t_P, intr)
        zp_tip_xy = project_xyz_fn(t_P + (0.035 * zP).astype(np.float32), intr)
        if zp_origin_xy is not None and zp_tip_xy is not None and zp_tip_xy[1] > zp_origin_xy[1]:
            zP = -zP
            normal_flipped = True
        normal_orientation_source = "image_up_projection_fallback"

    xP = project_to_plane_axis(float(tangent_sign) * np.array(edge_line.get("axis", []), dtype=np.float32), zP)
    if float(np.linalg.norm(xP)) < 1e-6:
        return None
    yP = unit(np.cross(zP, xP))
    xP = unit(np.cross(yP, zP))
    T_cam_P = se3(np.stack([xP, yP, zP], axis=1).astype(np.float32), t_P)

    zT = zP.copy()
    xT = project_to_plane_axis(float(tangent_sign) * np.array(work_edge.get("axis", []), dtype=np.float32), zT)
    if float(np.linalg.norm(xT)) < 1e-6:
        xT = project_to_plane_axis(float(tangent_sign) * np.array(tool.get("tool_axis", []), dtype=np.float32), zT)
    if float(np.linalg.norm(xT)) < 1e-6:
        return None

    desired_yT = project_to_plane_axis(t_T - grip_T, zT)
    if float(np.linalg.norm(desired_yT)) < 1e-6:
        desired_yT = project_to_plane_axis(np.array(tool.get("tool_axis", []), dtype=np.float32), zT)
    yT = unit(np.cross(zT, xT))
    if float(np.linalg.norm(yT)) < 1e-6:
        return None
    xT = unit(np.cross(yT, zT))
    if float(np.linalg.norm(desired_yT)) > 1e-6 and float(np.dot(yT, desired_yT)) < 0.0:
        xT = -xT
        yT = -yT
    T_cam_T = se3(np.stack([xT, yT, zT], axis=1).astype(np.float32), t_T)

    p1 = np.array(edge_line.get("endpoint_a", []), dtype=np.float32)
    p2 = np.array(edge_line.get("endpoint_b", []), dtype=np.float32)
    if p1.shape != (3,) or p2.shape != (3,):
        return None
    contacts_cam = [0.5 * (p1 + p2)]
    T_P_cam = inv_se3(T_cam_P)
    contacts_P = []
    for c in contacts_cam:
        c_h = np.concatenate([c.astype(np.float32), np.array([1.0], dtype=np.float32)], axis=0)
        contacts_P.append((T_P_cam @ c_h)[:3].astype(np.float32))

    return {
        "T_cam_T": T_cam_T,
        "T_cam_P": T_cam_P,
        "T_tool_to_target": T_cam_P @ inv_se3(T_cam_T),
        "tangent_sign": float(tangent_sign),
        "normal_sign": float(normal_sign),
        "target_normal_outward_cam": zP.astype(np.float32),
        "approach_direction_cam": (-zP).astype(np.float32),
        "target_normal_orientation_source": normal_orientation_source,
        "target_normal_flipped_from_raw": bool(normal_flipped),
        "grasp_normal_sign_not_applied_to_target": True,
        "world_up_cam": None if up_cam is None else up_cam.astype(np.float32),
        "target_action_edge_cam": {"p1_3d_m": p1.astype(np.float32), "p2_3d_m": p2.astype(np.float32), "dir_xP_3d": xP.astype(np.float32)},
        "target_axes_P": {"xP": xP.astype(np.float32), "yP": yP.astype(np.float32), "zP": zP.astype(np.float32)},
        "contact_points_P": contacts_P,
    }
