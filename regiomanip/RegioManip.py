#!/usr/bin/env python3
"""RegioManip — pipeline orchestrator.

Runs the full three-stage pipeline from a single RGB-D frame:

  Stage 1  Region-level affordance grounding  (DINO → SAM → VLM + geometry)
  Stage 2  Contact-frame hypothesis lifting   (back-projection → SE(3) frames)
  Stage 3  Wrench-bounded constraint synthesis (GWS bounds → action program)

Typical usage (preferred: python run.py from the repo root):

  python run.py \\
      --image-path data/Scraper/color_000032.png \\
      --instruction "scrape the putty by the scraper" \\
      --target-object yellow_putty \\
      --output-dir results/demo

Or directly:

  python -m regiomanip \\
      --image-path data/Scraper/color_000032.png \\
      --instruction "pick up the scraper"

DINO/SAM artifacts are written under <output-dir>/dino_sam/.
Use --skip-downstream to stop after Stage 1 (perception only).
Use --face-direction lower/upper to override contact-face orientation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[1]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

from regiomanip.env_utils import load_workspace_env

load_workspace_env()

from regiomanip.region_grounding.region_policy import run_region_level_affordance_from_sam, _draw_point_callout as _region_callout
from regiomanip.region_grounding.sam import KEEP_MAX_DEFAULT
import numpy as np

from regiomanip.region_grounding.utils import save_json, to_builtin


class _NpEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
        return super().default(o)


from regiomanip.perception.interaction_lift import run_interaction_lift
from regiomanip.compiler.constraint_synthesis import compile_action_program

DEFAULT_DATA_PATH = str(Path(__file__).resolve().parents[1] / "data")
DEFAULT_OUTPUT_DIR = "./results/regiomanip2"


def _package_dir() -> Path:
    return Path(__file__).resolve().parent


def _default_asset_paths() -> Dict[str, str]:
    pkg = _package_dir()
    return {
        "high_level_plan_json":     str(pkg / "data"    / "high_level_plan.json"),
        "symbolic_grounding_json":  str(pkg / "data"    / "symbolic_grounding.json"),
        "action_templates_json":    str(pkg / "configs" / "action_templates.json"),
        "wrench_profiles_json":     str(pkg / "configs" / "wrench_profiles.json"),
    }


def _parse_categories(values: Optional[List[str]]) -> Optional[List[str]]:
    if not values:
        return None
    out: List[str] = []
    for value in values:
        for item in str(value).split(","):
            item = item.strip()
            if item:
                out.append(item)
    return out or None


def _run_dino_sam(
    *,
    image_path: Optional[str],
    frame_no: Optional[int],
    data_path: str,
    instruction: str,
    output_dir: str,
    categories: Optional[List[str]],
    keep_max: int,
    max_inner_masks: int,
) -> Dict[str, Any]:
    """Run DINO + SAM and return paths to the produced artifacts."""
    from regiomanip.region_grounding.dino import _resolve_image_path, run_dino
    from regiomanip.region_grounding.sam import run_sam_stage

    dino_dir = str(Path(output_dir) / "dino_sam" / "dino")
    sam_dir = str(Path(output_dir) / "dino_sam" / "sam")

    resolved_image = image_path or _resolve_image_path(
        data_path=data_path,
        frame_no=int(frame_no),
        out_dir=dino_dir,
    )

    print(f"[RegioManip2] Image: {resolved_image}")
    print("[RegioManip2] Stage 1/2: DINO")
    dino_result = run_dino(
        image_path=resolved_image,
        out_dir=dino_dir,
        instruction=instruction,
        categories=categories,
    )

    print("[RegioManip2] Stage 2/2: SAM")
    run_sam_stage(
        image_path=resolved_image,
        dino_result=dino_result,
        out_dir=sam_dir,
        keep_max=keep_max,
        max_inner_masks=max_inner_masks,
        instruction=instruction,
    )

    sam_json = str(Path(sam_dir) / "sam_summary.json")
    if not Path(sam_json).exists():
        raise RuntimeError(f"SAM did not produce expected output: {sam_json}")

    return {
        "image_path": resolved_image,
        "sam_json": sam_json,
        "dino_dir": dino_dir,
        "sam_dir": sam_dir,
    }


# ---------------------------------------------------------------------------
# Bridge: generate VLM1/VLM2-compatible JSONs from handle-region results
# ---------------------------------------------------------------------------

def _build_compat_jsons(
    handle_summary: Dict[str, Any],
    out_dir: str,
    depth_path: Optional[str] = None,
) -> Optional[str]:
    """Write VLM1 + VLM2 compatible JSONs that run_interaction_lift can consume.

    Returns the path to the VLM2 JSON, or None if no valid result exists.
    """
    results = handle_summary.get("results", [])
    rgb_path = str(handle_summary.get("rgb_path", "") or "")

    vlm1_objects = []
    vlm2_objects = []
    _tool_count = 0  # track tools separately; targets alone can't drive a lift

    for res in results:
        if not bool(res.get("ok")):
            continue
        name = str(res.get("object_name", "")).strip()
        if not name:
            continue
        arts = res.get("artifacts", {}) or {}
        h_mask = str(arts.get("handle_fullmask_path", "") or "")
        we_mask = str(arts.get("work_end_fullmask_path", "") or "")
        if not h_mask or not Path(h_mask).exists():
            continue
        if not we_mask or not Path(we_mask).exists():
            continue

        grip_global = res.get("grip_point_xy_global")
        contact_global = res.get("work_end_contact_xy_global")
        edge_pts = res.get("work_end_edge_points_global")

        if grip_global is None or contact_global is None:
            continue
        # Need 2 work-end edge points; fall back to contact ± tiny offset if missing
        if not edge_pts or len(edge_pts) != 2:
            cx, cy = contact_global
            edge_pts = [[cx - 1.0, cy], [cx + 1.0, cy]]

        vlm1_objects.append({
            "name": name,
            "object_lift_readiness": 1.0,
            "parts": [
                {"part": "handle",   "mask_path": h_mask},
                {"part": "work_end", "mask_path": we_mask},
            ],
            "quality_control": {
                "lift_readiness": {
                    "handle":   {"score": 1.0},
                    "work_end": {"score": 1.0},
                }
            },
        })
        _tool_count += 1

        vlm2_objects.append({
            "name": name,
            "ok": True,
            "stage2_applicable": True,
            "pattern": "GRASP",
            "grip_point_global":        {"G": list(grip_global)},
            "work_end_touch_mid_global": list(contact_global),
            "work_end_points_global":   {"p1": list(edge_pts[0]), "p2": list(edge_pts[1])},
            "grip_point_3d": res.get("grip_point_3d"),
            "work_end_contact_3d": res.get("work_end_contact_3d"),
            "axis_direction_3d": res.get("axis_direction_3d"),
            "contact_normal_3d": res.get("contact_normal_3d"),
            "contact_geometry": res.get("contact_geometry"),
            "force_mode": res.get("force_mode"),
            "wrench_basis": res.get("wrench_basis"),
            "grasp_frame_cam": res.get("grasp_frame_cam"),
            "grasp_quaternion_xyzw": res.get("grasp_quaternion_xyzw"),
            "functional_area_px": res.get("functional_area_px"),
            "grasp_width_px": res.get("grasp_width_px"),
            "grasp_width_m": res.get("grasp_width_m"),
            "grasp_width_mm": res.get("grasp_width_mm"),
            "geometry_vlm_agreement": res.get("geometry_vlm_agreement"),
            "affordance_uncertainty": res.get("affordance_uncertainty"),
            "coherence_report": res.get("coherence_report"),
            "affordance_hypotheses": res.get("affordance_hypotheses"),
        })

    # ---- target objects (passive, ≤1 mask) ---------------------------------
    # Add each target to VLM1 (whole mask) and VLM2 (stage2_applicable=False,
    # edge points derived from bbox so interaction_stage can do real 3D lifting
    # instead of falling back to a virtual copy of the tool geometry).
    for tgt in handle_summary.get("targets", []):
        tgt_name = str(tgt.get("object_name", "")).strip()
        if not tgt_name:
            continue
        whole_path = str(tgt.get("mask_path") or "")
        if not whole_path or not Path(whole_path).exists():
            continue
        centroid = tgt.get("centroid_xy_global")
        bbox = tgt.get("bbox_xyxy_global")
        if centroid is None or bbox is None:
            continue

        cx, cy = float(centroid[0]), float(centroid[1])
        x1, y1, x2, y2 = [float(v) for v in bbox]

        # Direction from target centroid toward tool (average of G + C points).
        tool_kp_xs: list = []
        tool_kp_ys: list = []
        for obj in vlm2_objects:
            for g in (obj.get("grip_point_global") or {}).values():
                tool_kp_xs.append(float(g[0])); tool_kp_ys.append(float(g[1]))
            c_pt = obj.get("work_end_touch_mid_global")
            if c_pt:
                tool_kp_xs.append(float(c_pt[0])); tool_kp_ys.append(float(c_pt[1]))

        if tool_kp_xs:
            tx = float(np.mean(tool_kp_xs))
            ty = float(np.mean(tool_kp_ys))
            dn = max(1e-6, float(np.hypot(tx - cx, ty - cy)))
            dx, dy = (tx - cx) / dn, (ty - cy) / dn
        else:
            dx, dy = 1.0, 0.0

        # Use mask contour points facing the tool, filtered to pixels with valid
        # depth.  Contour corners at the object/floor boundary often have depth=0;
        # keeping only depth>0 pixels avoids the silent None-backprojection that
        # makes interaction_stage skip the target and fall back to a virtual clone.
        import cv2 as _cv2
        _depth_arr: Optional[np.ndarray] = None
        if depth_path and Path(depth_path).exists():
            try:
                _da = np.load(depth_path)
                _depth_arr = _da if _da.ndim == 2 else None
            except Exception:
                _depth_arr = None

        touch_mid = [cx, cy]
        _s = min(x2 - x1, y2 - y1) / 4.0
        edge_p1 = [cx + dy * _s, cy - dx * _s]
        edge_p2 = [cx - dy * _s, cy + dx * _s]

        _m_img = _cv2.imread(whole_path, _cv2.IMREAD_GRAYSCALE)
        if _m_img is not None:
            _cnts, _ = _cv2.findContours(_m_img, _cv2.RETR_EXTERNAL, _cv2.CHAIN_APPROX_NONE)
            if _cnts:
                _bpts = _cnts[0].reshape(-1, 2).astype(float)
                # Filter to pixels with valid depth when a depth map is available.
                if _depth_arr is not None:
                    _dH, _dW = _depth_arr.shape
                    _xi = np.clip(np.round(_bpts[:, 0]).astype(int), 0, _dW - 1)
                    _yi = np.clip(np.round(_bpts[:, 1]).astype(int), 0, _dH - 1)
                    _valid = _depth_arr[_yi, _xi] > 0
                    if _valid.sum() >= 2:
                        _bpts = _bpts[_valid]
                # Facing points = top 5% by dot product toward tool.
                _dots = _bpts[:, 0] * dx + _bpts[:, 1] * dy
                _facing = _bpts[_dots >= _dots.max() * 0.95]
                if len(_facing) >= 2:
                    touch_mid = [float(_facing[:, 0].mean()), float(_facing[:, 1].mean())]
                    _perp = _facing[:, 0] * (-dy) + _facing[:, 1] * dx
                    edge_p1 = [float(_facing[int(np.argmin(_perp)), 0]),
                               float(_facing[int(np.argmin(_perp)), 1])]
                    edge_p2 = [float(_facing[int(np.argmax(_perp)), 0]),
                               float(_facing[int(np.argmax(_perp)), 1])]

        vlm1_objects.append({
            "name": tgt_name,
            "object_lift_readiness": 0.8,
            "parts": [{"part": "whole", "mask_path": whole_path}],
            "quality_control": {"lift_readiness": {"whole": {"score": 0.8}}},
        })
        vlm2_objects.append({
            "name": tgt_name,
            "ok": True,
            "stage2_applicable": False,
            "pattern": "TARGET",
            "target_edge_points_global": {
                "p1": [float(edge_p1[0]), float(edge_p1[1])],
                "p2": [float(edge_p2[0]), float(edge_p2[1])],
            },
            "target_edge_touch_mid_global": [float(touch_mid[0]), float(touch_mid[1])],
        })

    if _tool_count == 0:
        return None

    compat_dir = Path(out_dir) / "compat"
    compat_dir.mkdir(parents=True, exist_ok=True)

    vlm1_path = str(compat_dir / "vlm1_compat.json")
    vlm2_path = str(compat_dir / "vlm2_compat.json")

    save_json(vlm1_path, {"rgb_path": rgb_path, "objects": vlm1_objects})
    save_json(vlm2_path, {
        "results_json_from_vlm1": vlm1_path,
        "objects": vlm2_objects,
    })
    return vlm2_path


# ---------------------------------------------------------------------------
# Downstream stages: interaction lift + constraint synthesis
# ---------------------------------------------------------------------------

def _run_downstream_stages(
    handle_summary: Dict[str, Any],
    out_dir: str,
    data_path: str,
    frame_no: Optional[int],
    depth_path: Optional[str],
    depth_scale: float,
    readiness_threshold: float,
    world_up_cam: Optional[str],
    world_up_world: Optional[str],
    t_world_cam_json: Optional[str],
) -> Dict[str, Any]:
    assets = _default_asset_paths()

    print("[RegioManip2] Stage 2/4: building interaction-lift inputs")
    vlm2_path = _build_compat_jsons(handle_summary, out_dir, depth_path=depth_path)
    if vlm2_path is None:
        print("[RegioManip2] No valid handle-region result — skipping downstream stages.")
        return handle_summary

    print("[RegioManip2] Stage 3/4: interaction lift")
    vlm3_result = run_interaction_lift(
        out_dir=out_dir,
        data_path=data_path,
        frame_no=frame_no if frame_no is not None else 0,
        depth_path=depth_path,
        vlm2_results_json=vlm2_path,
        depth_scale=depth_scale,
        readiness_threshold=readiness_threshold,
        world_up_cam=world_up_cam,
        world_up_world=world_up_world,
        t_world_cam_json=t_world_cam_json,
    )
    interaction_results_json = str(Path(out_dir) / "interaction_lift" / "interaction_lift_results.json")
    print(f"[RegioManip2] Saved: {interaction_results_json}")

    if not bool(vlm3_result.get("ok", False)):
        print("[RegioManip2] Interaction lift failed — skipping constraint synthesis.")
        return {**handle_summary, "interaction_lift": vlm3_result}

    print("[RegioManip2] Stage 4/4: action constraint synthesis")
    compiled = compile_action_program(
        vlm3_results_json=interaction_results_json,
        toolmanip_plan_json=assets["high_level_plan_json"],
        toolmanip_vlm2_json=assets["symbolic_grounding_json"],
        action_templates_json=assets["action_templates_json"],
        wrench_profiles_json=assets["wrench_profiles_json"],
        out_path=str(Path(out_dir) / "compiled_action_program.json"),
    )

    summary = {
        **handle_summary,
        "interaction_lift": vlm3_result,
        "compiled_action_program_ok": bool(compiled.get("steps")),
        "selected_interaction": {
            "tool":    ((compiled.get("global", {}) or {}).get("best_hypothesis", {}) or {}).get("tool_name"),
            "target":  ((compiled.get("global", {}) or {}).get("best_hypothesis", {}) or {}).get("target_name"),
            "pattern": ((compiled.get("global", {}) or {}).get("best_hypothesis", {}) or {}).get("pattern"),
        },
    }
    summary = to_builtin(summary)
    save_json(str(Path(out_dir) / "run_summary.json"), summary)
    return summary


# ---------------------------------------------------------------------------
# Full-frame keypoint visualisation
# ---------------------------------------------------------------------------

def _morph_skel(mask: np.ndarray) -> np.ndarray:
    """Morphological thinning skeleton (same algorithm as region_policy._morph_skeleton)."""
    import cv2
    img = (mask > 0).astype(np.uint8).copy()
    if img.sum() == 0:
        return img
    skel = np.zeros_like(img, dtype=np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(512):
        eroded = cv2.erode(img, ker)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, ker)
        skel = cv2.bitwise_or(skel, cv2.subtract(eroded, opened))
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)


def _make_tool_target_skeleton_overlay(
    handle_summary: Dict[str, Any],
    region_afford_dir: str,
) -> Optional[str]:
    """Full-frame skeleton overlay showing tool regions + target region with morph skeletons."""
    import cv2

    rgb_path = str(handle_summary.get("rgb_path", "") or "")
    if not rgb_path or not Path(rgb_path).exists():
        return None
    frame = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if frame is None:
        return None
    H, W = frame.shape[:2]

    def _load_mask(path: str) -> Optional[np.ndarray]:
        if not path or not Path(path).exists():
            return None
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return m

    HANDLE_COLOR  = (50, 200, 50)   # BGR green  — matches _make_combined_pgl
    WORK_END_COLOR = (50, 80, 220)  # BGR orange-red — matches _make_combined_pgl
    TARGET_COLOR = (0, 210, 220)
    label_y = H - 10

    for res in handle_summary.get("results", []):
        if not bool(res.get("ok")):
            continue
        arts = res.get("artifacts", {}) or {}
        for mask_key, color, corner_lbl in [
            ("handle_fullmask_path",   HANDLE_COLOR,   "Handle"),
            ("work_end_fullmask_path", WORK_END_COLOR, "Work-End"),
        ]:
            m = _load_mask(str(arts.get(mask_key) or ""))
            if m is None:
                continue
            tint = np.zeros_like(frame, dtype=np.uint8)
            tint[:, :] = color
            sel = (m > 0).astype(bool)
            if sel.any():
                frame[sel] = cv2.addWeighted(frame[sel], 0.78, tint[sel], 0.22, 0.0)
            cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cnts, -1, color, 2, cv2.LINE_AA)
            skel = _morph_skel((m > 0).astype(np.uint8))
            skel_sel = skel.astype(bool)
            if skel_sel.any():
                frame[skel_sel] = cv2.addWeighted(frame[skel_sel], 0.25, tint[skel_sel], 0.75, 0.0)
            cv2.putText(frame, corner_lbl, (8, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            label_y -= 16

    for tgt in handle_summary.get("targets", []):
        m = _load_mask(str(tgt.get("mask_path") or ""))
        if m is None:
            continue
        tint = np.zeros_like(frame, dtype=np.uint8)
        tint[:, :] = TARGET_COLOR
        sel = (m > 0).astype(bool)
        if sel.any():
            frame[sel] = cv2.addWeighted(frame[sel], 0.78, tint[sel], 0.22, 0.0)
        cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(frame, cnts, -1, TARGET_COLOR, 2, cv2.LINE_AA)
        skel = _morph_skel((m > 0).astype(np.uint8))
        skel_sel = skel.astype(bool)
        if skel_sel.any():
            frame[skel_sel] = cv2.addWeighted(frame[skel_sel], 0.25, tint[skel_sel], 0.75, 0.0)
        cv2.putText(frame, str(tgt.get("object_name", "Target")), (8, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, TARGET_COLOR, 1, cv2.LINE_AA)
        label_y -= 16

    out_path = str(Path(region_afford_dir) / "tool_target_skeleton_overlay.png")
    cv2.imwrite(out_path, frame)
    print(f"[RegioManip2] Skeleton overlay: {out_path}")
    return out_path


def _augment_contact_overlay_with_target(
    handle_summary: Dict[str, Any],
    region_afford_dir: str,
) -> None:
    """Add target 3D point cloud + axis arrows + T callout to the contact overlay."""
    import cv2, json as _json
    from regiomanip.region_grounding.region_policy import _pca_3d, _draw_point_callout

    results = handle_summary.get("results", [])
    if not results:
        return
    res = results[0]
    if not bool(res.get("ok")):
        return
    obj_name = str(res.get("object_name", ""))
    overlay_path = str(Path(region_afford_dir) / f"{obj_name}__contact_overlay.png")
    if not Path(overlay_path).exists():
        return
    vis = cv2.imread(overlay_path, cv2.IMREAD_COLOR)
    if vis is None:
        return
    H, W = vis.shape[:2]

    rgb_path = str(handle_summary.get("rgb_path", "") or "")
    if not rgb_path or not Path(rgb_path).exists():
        return
    p = Path(rgb_path)

    # Load camera intrinsics from frame meta JSON
    stem = p.stem
    meta_path = p.parent / f"frame_{stem.replace('color_', '')}_meta.json"
    fx, fy, ppx, ppy, depth_scale = 609.24, 611.98, 309.79, 240.47, 0.001
    if meta_path.exists():
        try:
            m = _json.loads(meta_path.read_text())
            ci = m.get("color_intrinsics", {})
            fx = float(ci.get("fx", fx))
            fy = float(ci.get("fy", fy))
            ppx = float(ci.get("ppx", ppx))
            ppy = float(ci.get("ppy", ppy))
            ds = float(m.get("depth_scale", 0))
            if ds > 0:
                depth_scale = ds
        except Exception:
            pass

    depth_npy = p.parent / p.name.replace("color_", "depth_").replace(".png", ".npy")
    if not depth_npy.exists():
        return
    depth_raw = np.load(str(depth_npy))

    TARGET_COLOR = (0, 210, 220)
    ARROW_LEN = 0.025

    for tgt in handle_summary.get("targets", []):
        mask_p = str(tgt.get("mask_path") or "")
        if not mask_p or not Path(mask_p).exists():
            continue
        m_img = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if m_img is None:
            continue
        dH, dW = depth_raw.shape[:2]
        if m_img.shape != (dH, dW):
            m_img = cv2.resize(m_img, (dW, dH), interpolation=cv2.INTER_NEAREST)

        ys, xs = np.where(m_img > 0)
        if xs.size < 4:
            continue
        Z_raw = depth_raw[ys, xs].astype(np.float64)
        Z = Z_raw * depth_scale
        valid = (Z_raw > 0) & (Z >= 0.05) & (Z <= 3.0)
        if valid.sum() < 4:
            continue
        xsv = xs[valid].astype(np.float64)
        ysv = ys[valid].astype(np.float64)
        Zv = Z[valid]
        cloud = np.column_stack([(xsv - ppx) * Zv / fx, (ysv - ppy) * Zv / fy, Zv])

        # Draw point cloud — subsample to ≤1500 pts, radius=1, neutral colour so axes stay visible
        MAX_PTS = 1500
        pt_cloud = cloud
        if len(pt_cloud) > MAX_PTS:
            idx = np.random.default_rng(42).choice(len(pt_cloud), MAX_PTS, replace=False)
            pt_cloud = pt_cloud[idx]
        us = np.round(pt_cloud[:, 0] / pt_cloud[:, 2] * fx + ppx).astype(int)
        vs = np.round(pt_cloud[:, 1] / pt_cloud[:, 2] * fy + ppy).astype(int)
        v2 = (us >= 0) & (us < W) & (vs >= 0) & (vs < H)
        DOT_COLOR = (160, 240, 200)  # pale teal — distinct from blue/red/yellow axes
        for u_pt, v_pt in zip(us[v2], vs[v2]):
            cv2.circle(vis, (int(u_pt), int(v_pt)), 1, DOT_COLOR, -1)

        # PCA → contact point + axes
        major, _mid, minor = _pca_3d(cloud)
        contact_3d = cloud.mean(axis=0)
        axis_3d = major.tolist()
        normal_3d = (minor if minor[2] <= 0 else -minor).tolist()
        t_arr = np.cross(np.array(axis_3d), np.array(normal_3d))
        tn = float(np.linalg.norm(t_arr))
        tangent_3d = (t_arr / tn).tolist() if tn > 1e-6 else None

        cp = (
            max(0, min(W - 1, int(round(contact_3d[0] / contact_3d[2] * fx + ppx)))),
            max(0, min(H - 1, int(round(contact_3d[1] / contact_3d[2] * fy + ppy)))),
        )

        # Axis arrows (same colors as tool: blue / red / yellow)
        for dir_3d, color in [
            (axis_3d,    (220, 80,  50)),
            (normal_3d,  (50,  50,  220)),
            (tangent_3d, (50,  220, 220)),
        ]:
            if dir_3d is None:
                continue
            tip = [contact_3d[i] + dir_3d[i] * ARROW_LEN for i in range(3)]
            if tip[2] <= 0:
                continue
            tp = (
                max(0, min(W - 1, int(round(tip[0] / tip[2] * fx + ppx)))),
                max(0, min(H - 1, int(round(tip[1] / tip[2] * fy + ppy)))),
            )
            cv2.arrowedLine(vis, cp, tp, color, 4, cv2.LINE_AA, tipLength=0.3)

        # T callout — same style as _draw_point_callout / pgl_combined
        _draw_point_callout(vis, cp, "T", TARGET_COLOR, direction=(-0.7, -0.7), marker_type=cv2.MARKER_CROSS)

    cv2.imwrite(overlay_path, vis)
    print(f"[RegioManip2] Contact overlay + target axes: {overlay_path}")


def _augment_depth_colormap_with_target(
    handle_summary: Dict[str, Any],
    region_afford_dir: str,
    depth_scale: float = 0.001,
) -> None:
    """Regenerate depth_colormap.png as full-frame with tool regions + target tinted (same style)."""
    import cv2
    from regiomanip.region_grounding.utils import colorize_depth
    rgb_path = str(handle_summary.get("rgb_path", "") or "")
    if not rgb_path or not Path(rgb_path).exists():
        return
    p = Path(rgb_path)
    depth_npy = p.parent / p.name.replace("color_", "depth_").replace(".png", ".npy")
    if not depth_npy.exists():
        return
    depth_raw = np.load(str(depth_npy)).astype(np.float32) * depth_scale
    depth_vis = colorize_depth(depth_raw)
    if depth_vis is None:
        return
    H, W = depth_vis.shape[:2]

    def _overlay_mask(img: np.ndarray, mask_path: str, color: tuple) -> None:
        if not mask_path or not Path(mask_path).exists():
            return
        m = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        tint = np.zeros_like(img, dtype=np.uint8)
        tint[:, :] = color
        sel = (m > 0).astype(bool)
        if sel.any():
            img[sel] = cv2.addWeighted(img[sel], 0.72, tint[sel], 0.28, 0.0)
        cnts, _ = cv2.findContours((m > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(img, cnts, -1, color, 2, cv2.LINE_AA)

    # Tool regions
    for res in handle_summary.get("results", []):
        if not bool(res.get("ok")):
            continue
        arts = res.get("artifacts", {}) or {}
        _overlay_mask(depth_vis, str(arts.get("handle_fullmask_path") or ""),   (50, 200, 50))
        _overlay_mask(depth_vis, str(arts.get("work_end_fullmask_path") or ""), (50, 80, 220))

    # Target regions
    for tgt in handle_summary.get("targets", []):
        _overlay_mask(depth_vis, str(tgt.get("mask_path") or ""), (0, 210, 220))

    out_path = str(Path(region_afford_dir) / "depth_colormap.png")
    cv2.imwrite(out_path, depth_vis)
    print(f"[RegioManip2] Depth colormap (tool + target): {out_path}")


def _draw_target_affordance_outputs(
    handle_summary: Dict[str, Any],
    region_afford_dir: str,
) -> None:
    """Generate per-target visualizations (regions, skeleton, keypoints) matching tool output style."""
    import cv2

    rgb_path = str(handle_summary.get("rgb_path", "") or "")
    if not rgb_path or not Path(rgb_path).exists():
        return
    full_bgr = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if full_bgr is None:
        return
    H, W = full_bgr.shape[:2]
    TARGET_COLOR = (0, 210, 220)

    for tgt in handle_summary.get("targets", []):
        tgt_name = str(tgt.get("object_name", "")).strip()
        mask_p = str(tgt.get("mask_path") or "")
        if not tgt_name or not mask_p or not Path(mask_p).exists():
            continue
        m = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

        bbox = tgt.get("bbox_xyxy_global")
        mask_u8 = (m > 0).astype(np.uint8)

        # ── {name}__skeleton.png: skeleton style on cropped region ──────────
        if bbox:
            bx1, by1, bx2, by2 = [int(v) for v in bbox]
            pad = 20
            x1c, y1c = max(0, bx1 - pad), max(0, by1 - pad)
            x2c, y2c = min(W, bx2 + pad), min(H, by2 + pad)
            crop = full_bgr[y1c:y2c, x1c:x2c].copy()
            m_crop = mask_u8[y1c:y2c, x1c:x2c]
        else:
            crop = full_bgr.copy()
            m_crop = mask_u8
        tint_c = np.zeros_like(crop, dtype=np.uint8)
        tint_c[:, :] = TARGET_COLOR
        sel_c = m_crop.astype(bool)
        if sel_c.any():
            crop[sel_c] = cv2.addWeighted(crop[sel_c], 0.78, tint_c[sel_c], 0.22, 0.0)
        cnts_c, _ = cv2.findContours(m_crop, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(crop, cnts_c, -1, TARGET_COLOR, 2, cv2.LINE_AA)
        skel_c = _morph_skel(m_crop)
        skel_sel_c = skel_c.astype(bool)
        if skel_sel_c.any():
            crop[skel_sel_c] = cv2.addWeighted(
                crop[skel_sel_c], 0.25, tint_c[skel_sel_c], 0.75, 0.0)
        cv2.imwrite(str(Path(region_afford_dir) / f"{tgt_name}__skeleton.png"), crop)
        print(f"[RegioManip2] Target skeleton: {tgt_name}__skeleton.png")


def _blend_mask(frame: np.ndarray, mask: np.ndarray, color_bgr: tuple) -> np.ndarray:
    """In-place tint pixels where mask>0, return modified frame."""
    sel = mask > 0
    if not sel.any():
        return frame
    tint = np.full_like(frame, color_bgr, dtype=np.uint8)
    blended = (frame[sel].astype(np.float32) * 0.72
               + tint[sel].astype(np.float32) * 0.28).clip(0, 255).astype(np.uint8)
    frame[sel] = blended
    return frame



def _draw_full_frame_keypoints(
    handle_summary: Dict[str, Any],
    out_dir: str,
) -> Optional[str]:
    """pgl_combined style on the full frame: tinted regions + G/C/T callouts, no arrow."""
    import cv2

    rgb_path = str(handle_summary.get("rgb_path", "") or "")
    if not rgb_path or not Path(rgb_path).exists():
        return None
    frame = cv2.imread(rgb_path, cv2.IMREAD_COLOR)
    if frame is None:
        return None

    H, W = frame.shape[:2]
    label_y = H - 10  # corner labels accumulate upward

    def _load_mask(path: str) -> Optional[np.ndarray]:
        if not path or not Path(path).exists():
            return None
        m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if m is None:
            return None
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)
        return m

    # ------------------------------------------------------------------ #
    # Tool regions: handle (green) + work-end (red)                       #
    # ------------------------------------------------------------------ #
    HANDLE_COLOR  = (50, 200, 50)
    WORKEND_COLOR = (50,  80, 220)
    TARGET_COLOR  = (0,  210, 220)   # warm yellow (BGR 0,210,220 = RGB 220,210,0)

    # Collect all tool keypoints (grip + contact) while drawing tool regions.
    all_tool_keypoints: List[List[float]] = []
    tool_grip_xy: Optional[Any] = None
    tool_contact_xy: Optional[Any] = None

    for res in handle_summary.get("results", []):
        if not bool(res.get("ok")):
            continue
        arts = res.get("artifacts", {}) or {}

        for mask_key, color, corner_label in [
            ("handle_fullmask_path",   HANDLE_COLOR,  "Handle"),
            ("work_end_fullmask_path", WORKEND_COLOR, "Work-End"),
        ]:
            m = _load_mask(str(arts.get(mask_key) or ""))
            if m is None:
                continue
            frame = _blend_mask(frame, m, color)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame, cnts, -1, color, 2, cv2.LINE_AA)
            cv2.putText(frame, corner_label, (8, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
            label_y -= 16

        grip_xy    = res.get("grip_point_xy_global")
        contact_xy = res.get("work_end_contact_xy_global")
        if grip_xy:
            _region_callout(frame, (int(round(grip_xy[0])), int(round(grip_xy[1]))),
                            "G", (30, 30, 220), direction=(0.7, -0.7), marker_type=cv2.MARKER_CROSS)
            all_tool_keypoints.append([float(grip_xy[0]), float(grip_xy[1])])
            tool_grip_xy = grip_xy
        if contact_xy:
            _region_callout(frame, (int(round(contact_xy[0])), int(round(contact_xy[1]))),
                            "C", (200, 50, 0), direction=(0.7, 0.7), marker_type=cv2.MARKER_CROSS)
            all_tool_keypoints.append([float(contact_xy[0]), float(contact_xy[1])])
            tool_contact_xy = contact_xy

    # ------------------------------------------------------------------ #
    # Target region: distinct tint + T callout at border contact point    #
    # ------------------------------------------------------------------ #
    for tgt in handle_summary.get("targets", []):
        m = _load_mask(str(tgt.get("mask_path") or ""))
        border_contact: Optional[tuple] = None

        if m is not None:
            frame = _blend_mask(frame, m, TARGET_COLOR)
            cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(frame, cnts, -1, TARGET_COLOR, 2, cv2.LINE_AA)
            cv2.putText(frame, str(tgt.get("object_name", "Target")), (8, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, TARGET_COLOR, 1, cv2.LINE_AA)
            label_y -= 16

            if cnts and all_tool_keypoints:
                border_pts = cnts[0].reshape(-1, 2).astype(float)

                # Option B: contour point with smallest perpendicular distance to
                # the G→C scraper axis line.  For a horizontal scraper the line sits
                # at the scraper's y-level; the closest target-contour point is
                # therefore the edge of the putty that faces the blade — which is
                # exactly where the scrape action contacts the material.
                if tool_grip_xy is not None and tool_contact_xy is not None:
                    gx = float(tool_grip_xy[0]);   gy  = float(tool_grip_xy[1])
                    cx_t = float(tool_contact_xy[0]); cy_t = float(tool_contact_xy[1])
                    d = np.array([cx_t - gx, cy_t - gy], dtype=np.float64)
                    d_len = float(np.hypot(d[0], d[1]))
                    if d_len > 1e-6:
                        d /= d_len
                        v = border_pts - np.array([gx, gy])
                        perp = np.abs(v[:, 0] * d[1] - v[:, 1] * d[0])
                        best_idx = int(np.argmin(perp))
                        border_contact = (int(round(border_pts[best_idx, 0])),
                                          int(round(border_pts[best_idx, 1])))

                if border_contact is None:
                    # Fallback: outermost border point toward the G+C centroid.
                    ys_m, xs_m = np.where(m > 0)
                    cx0 = xs_m.mean() if len(xs_m) else border_pts[:, 0].mean()
                    cy0 = ys_m.mean() if len(ys_m) else border_pts[:, 1].mean()
                    kp_arr = np.array(all_tool_keypoints)
                    tx, ty = kp_arr[:, 0].mean(), kp_arr[:, 1].mean()
                    dx, dy = tx - cx0, ty - cy0
                    norm = max(1e-6, np.hypot(dx, dy))
                    dx, dy = dx / norm, dy / norm
                    vx = border_pts[:, 0] - cx0
                    vy = border_pts[:, 1] - cy0
                    dots = vx * dx + vy * dy
                    facing = border_pts[dots >= dots.max() * 0.95]
                    border_contact = (int(round(facing[:, 0].mean())),
                                      int(round(facing[:, 1].mean())))

        explicit_contact = tgt.get("border_contact_xy_global")
        if explicit_contact and len(explicit_contact) >= 2:
            t_pt = (int(round(explicit_contact[0])), int(round(explicit_contact[1])))
        else:
            # T callout at the border edge facing the tool — use centroid only as fallback.
            t_pt = border_contact or (
            (int(round(tgt["centroid_xy_global"][0])),
             int(round(tgt["centroid_xy_global"][1])))
            if tgt.get("centroid_xy_global") else None
            )
        if t_pt:
            _region_callout(frame, (int(round(t_pt[0])), int(round(t_pt[1]))),
                            "T", TARGET_COLOR, direction=(-0.7, -0.7), marker_type=cv2.MARKER_CROSS)

        # Store back into summary so JSON also reflects the border contact.
        if border_contact is not None:
            tgt["border_contact_xy_global"] = list(border_contact)

    out_path = str(Path(out_dir) / "full_frame_combined.png")
    cv2.imwrite(out_path, frame)
    print(f"[RegioManip2] Full-frame combined: {out_path}")

    # Mirror into region_affordance/ so all per-scene outputs live together.
    _ra = Path(out_dir) / "region_affordance"
    if _ra.exists():
        cv2.imwrite(str(_ra / "full_frame_combined.png"), frame)

    return out_path


def _paint_targets_on_overlay(
    handle_summary: Dict[str, Any],
    overlay_path: str,
) -> None:
    """Paint target-object mask tint + T callout onto best_frame_overlay.png.

    Called after interaction lift so the 3D axes are already on the image;
    the target region is drawn on top so it remains clearly visible.
    """
    import cv2

    path = Path(overlay_path)
    if not path.exists():
        return
    frame = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if frame is None:
        return

    H, W = frame.shape[:2]
    TARGET_COLOR = (0, 210, 220)

    # Collect tool keypoints for the T-marker direction calculation.
    all_tool_kp: List[List[float]] = []
    ov_grip_xy: Optional[Any] = None
    ov_contact_xy: Optional[Any] = None
    for res in handle_summary.get("results", []):
        if not bool(res.get("ok")):
            continue
        g = res.get("grip_point_xy_global")
        c = res.get("work_end_contact_xy_global")
        if g:
            all_tool_kp.append([float(g[0]), float(g[1])])
            ov_grip_xy = g
        if c:
            all_tool_kp.append([float(c[0]), float(c[1])])
            ov_contact_xy = c

    for tgt in handle_summary.get("targets", []):
        mask_p = str(tgt.get("mask_path") or "")
        if not mask_p or not Path(mask_p).exists():
            continue
        m = cv2.imread(mask_p, cv2.IMREAD_GRAYSCALE)
        if m is None:
            continue
        if m.shape != (H, W):
            m = cv2.resize(m, (W, H), interpolation=cv2.INTER_NEAREST)

        frame = _blend_mask(frame, m, TARGET_COLOR)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame, cnts, -1, TARGET_COLOR, 2, cv2.LINE_AA)

        border_contact: Optional[tuple] = None
        if cnts and all_tool_kp:
            bpts = cnts[0].reshape(-1, 2).astype(float)

            # Option B: nearest contour point to the G→C axis line.
            if ov_grip_xy is not None and ov_contact_xy is not None:
                gx = float(ov_grip_xy[0]);    gy  = float(ov_grip_xy[1])
                cx_t = float(ov_contact_xy[0]); cy_t = float(ov_contact_xy[1])
                d = np.array([cx_t - gx, cy_t - gy], dtype=np.float64)
                d_len = float(np.hypot(d[0], d[1]))
                if d_len > 1e-6:
                    d /= d_len
                    v = bpts - np.array([gx, gy])
                    perp = np.abs(v[:, 0] * d[1] - v[:, 1] * d[0])
                    best_idx = int(np.argmin(perp))
                    border_contact = (int(round(bpts[best_idx, 0])),
                                      int(round(bpts[best_idx, 1])))

            if border_contact is None:
                # Fallback: outermost border point toward the G+C centroid.
                ys_m, xs_m = np.where(m > 0)
                cx0 = xs_m.mean() if len(xs_m) else bpts[:, 0].mean()
                cy0 = ys_m.mean() if len(ys_m) else bpts[:, 1].mean()
                kp_arr = np.array(all_tool_kp)
                tx, ty = kp_arr[:, 0].mean(), kp_arr[:, 1].mean()
                dx, dy = tx - cx0, ty - cy0
                norm = max(1e-6, np.hypot(dx, dy))
                dx, dy = dx / norm, dy / norm
                vx = bpts[:, 0] - cx0
                vy = bpts[:, 1] - cy0
                dots = vx * dx + vy * dy
                facing = bpts[dots >= dots.max() * 0.95]
                border_contact = (int(round(facing[:, 0].mean())),
                                  int(round(facing[:, 1].mean())))

        t_pt = border_contact or (
            (int(round(tgt["centroid_xy_global"][0])),
             int(round(tgt["centroid_xy_global"][1])))
            if tgt.get("centroid_xy_global") else None
        )
        if t_pt:
            _region_callout(frame, (int(round(t_pt[0])), int(round(t_pt[1]))),
                            "T", TARGET_COLOR, direction=(-0.7, -0.7), marker_type=cv2.MARKER_CROSS)

    cv2.imwrite(str(path), frame)
    print(f"[RegioManip2] Target regions painted on overlay: {overlay_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the handle-region policy (DINO + SAM + VLM) "
                    "then interaction lift + constraint synthesis."
    )

    src = parser.add_argument_group("Input source (one required)")
    src.add_argument("--image-path", default=None, help="Direct path to RGB image.")
    src.add_argument("--frame-no", type=int, default=None,
                     help="Frame number; looks for color_<N>.png under --data-path.")

    dino_sam = parser.add_argument_group("DINO / SAM options")
    dino_sam.add_argument("--data-path", default=DEFAULT_DATA_PATH,
                          help="Root data directory searched when --frame-no is used.")
    dino_sam.add_argument("--category", action="append", default=None, metavar="LABEL",
                          help="DINO query category. Can be repeated or comma-separated.")
    dino_sam.add_argument("--keep-max", type=int, default=KEEP_MAX_DEFAULT,
                          help="Max SAM whole-mask candidates per object.")
    dino_sam.add_argument("--max-inner-masks", type=int, default=2,
                          help="Max inner part-masks SAM exports per tool object.")

    parser.add_argument("--instruction", default="", required=True,
                        help="Task instruction, e.g. 'pick up the spoon'.")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR,
                        help="Directory to save all outputs.")
    parser.add_argument("--object-name", default=None,
                        help="Filter to a specific object name from the SAM output.")
    parser.add_argument("--target-object", default=None,
                        help="Name of the target object (e.g. 'putty') in the SAM output. "
                             "When set, its RGB crop is shown to the VLM alongside the tool "
                             "so that contact_geometry/force_mode are conditioned on the "
                             "actual target surface.")
    parser.add_argument("--rgb-path", default=None,
                        help="Override the RGB image path used for handle-region visualization.")
    parser.add_argument("--depth-path", default=None,
                        help="Optional depth .npy path for depth-aware scoring.")
    parser.add_argument("--face-direction", default="upper", choices=["upper", "lower"],
                        help="Contact face direction: 'upper'=top of image, 'lower'=bottom. Default: upper.")
    parser.add_argument("--require-api", action="store_true",
                        help="Fail if the VLM / API is unavailable.")

    ds = parser.add_argument_group("Interaction lift / constraint synthesis")
    ds.add_argument("--depth-scale", type=float, default=0.001,
                    help="Depth scale (metres per raw unit). Auto-read from meta JSON if available.")
    ds.add_argument("--readiness-threshold", type=float, default=0.60,
                    help="Minimum lift-readiness score to accept tool/target.")
    ds.add_argument("--world-up-cam", default="0,-1,0",
                    help="World-up vector in camera frame (default: '0,-1,0' for OpenCV y-down cameras).")
    ds.add_argument("--world-up-world", default="0,0,1",
                    help="World-up vector in world frame (default: z-up).")
    ds.add_argument("--t-world-cam-json", default=None,
                    help="Path to 4x4 T_world_cam JSON for world-up rotation.")
    ds.add_argument("--skip-downstream", action="store_true",
                    help="Stop after handle-region policy (skip interaction lift + synthesis).")

    args = parser.parse_args()

    if not args.image_path and args.frame_no is None:
        parser.error("Provide --image-path or --frame-no.")

    depth_scale = float(args.depth_scale)
    if args.image_path:
        _stem = Path(args.image_path).stem
        if _stem.startswith("color_") and args.frame_no is None:
            try:
                args.frame_no = int(_stem[len("color_"):])
            except ValueError:
                pass
        # Infer data_path from the image's parent when only --image-path is given
        if str(args.data_path) == DEFAULT_DATA_PATH:
            _img_parent = str(Path(args.image_path).parent.resolve())
            if _img_parent != DEFAULT_DATA_PATH:
                args.data_path = _img_parent
        if _stem.startswith("color_"):
            _meta = Path(args.image_path).parent / f"frame_{_stem[len('color_'):]}_meta.json"
            if _meta.exists():
                try:
                    _m = json.loads(_meta.read_text())
                    _ds = float(_m.get("depth_scale", 0))
                    if _ds > 0:
                        depth_scale = _ds
                        print(f"[RegioManip2] depth_scale={depth_scale} (from {_meta})")
                except Exception:
                    pass

    print(f"[RegioManip2] Running DINO + SAM under {args.output_dir}/dino_sam/")
    artifacts = _run_dino_sam(
        image_path=args.image_path,
        frame_no=args.frame_no,
        data_path=str(args.data_path),
        instruction=str(args.instruction),
        output_dir=str(args.output_dir),
        categories=_parse_categories(args.category),
        keep_max=int(args.keep_max),
        max_inner_masks=int(args.max_inner_masks),
    )
    sam_json = artifacts["sam_json"]
    if not args.rgb_path:
        args.rgb_path = artifacts["image_path"]

    print(f"[RegioManip2] Stage 1/4: handle-region policy on: {sam_json}")
    handle_summary = run_region_level_affordance_from_sam(
        sam_json_path=sam_json,
        instruction=str(args.instruction),
        out_dir=str(args.output_dir),
        object_name=args.object_name,
        rgb_path=args.rgb_path,
        depth_path=args.depth_path,
        require_api=bool(args.require_api),
        face_direction_override=args.face_direction,
        target_object_name=args.target_object,
    )

    _draw_full_frame_keypoints(handle_summary, str(args.output_dir))

    if args.skip_downstream:
        print(json.dumps(handle_summary, indent=2))
        return

    final_summary = _run_downstream_stages(
        handle_summary=handle_summary,
        out_dir=str(args.output_dir),
        data_path=str(args.data_path),
        frame_no=args.frame_no,
        depth_path=args.depth_path,
        depth_scale=depth_scale,
        readiness_threshold=float(args.readiness_threshold),
        world_up_cam=args.world_up_cam,
        world_up_world=args.world_up_world,
        t_world_cam_json=args.t_world_cam_json,
    )

    # Paint target mask + T callout onto the interaction-lift overlay so that
    # best_frame_overlay.png shows both the tool (3-D axes) and the target
    # object (yellow region tint) in the same image.
    overlay_path = (
        (final_summary.get("interaction_lift") or {})
        .get("best_frame_overlay_path")
    )
    if overlay_path:
        _paint_targets_on_overlay(handle_summary, str(overlay_path))

    _draw_full_frame_keypoints(final_summary, str(args.output_dir))

    # Generate per-scene outputs inside region_affordance/
    _ra_dir = str(Path(args.output_dir) / "region_affordance")
    Path(_ra_dir).mkdir(parents=True, exist_ok=True)
    _make_tool_target_skeleton_overlay(handle_summary, _ra_dir)
    _draw_target_affordance_outputs(handle_summary, _ra_dir)
    _augment_contact_overlay_with_target(handle_summary, _ra_dir)
    _augment_depth_colormap_with_target(handle_summary, _ra_dir, depth_scale=depth_scale)

    print("[RegioManip2] Finished")
    print(json.dumps(to_builtin(final_summary), indent=2, cls=_NpEncoder))


if __name__ == "__main__":
    main()
