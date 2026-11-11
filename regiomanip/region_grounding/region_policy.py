"""Region-level affordance grounding for tool manipulation.

Given SAM segmentation output, this module identifies the handle and work-end
of a tool object via VLM-guided scoring, computes grip and contact points in
both cropped and full-image coordinates, and returns a structured
:class:`HandleDecision` result consumed by the interaction-lift stage.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from regiomanip.env_utils import load_workspace_env
from regiomanip.region_grounding.prompt import (
    MODEL_CFG,
    _repair_tool_role_selection,
    build_openai_client_with_status,
    classify_openai_failure,
    query_vlm_content,
)
from regiomanip.region_grounding.utils import (
    colorize_depth,
    crop_from_masks,
    encode_image_to_data_url,
    ensure_dir,
    extract_json_block,
    load_depth,
    load_json,
    load_mask01,
    mask_area,
    mask_bbox_xyxy,
    save_json,
    union_masks,
)

load_workspace_env()

_MIN_VIS_PX = 600
RETRY_GEOMETRY_VLM_AGREEMENT_TH = float(os.environ.get("REGIOMANIP_RETRY_AGREEMENT_TH", "0.40"))
COHERENCE_IMPROBABLE_WEIGHT = float(os.environ.get("REGIOMANIP_COHERENCE_IMPROBABLE_WEIGHT", "0.45"))
COHERENCE_AUTO_REPAIR_MARGIN = float(os.environ.get("REGIOMANIP_COHERENCE_AUTO_REPAIR_MARGIN", "1.15"))


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------

@dataclass
class HandleDecision:
    object_name: str
    ok: bool
    status: str
    selected_handle_id: Optional[int]
    selected_handle_ids: List[int]
    confidence: float
    selected_work_end_id: Optional[int]
    selected_work_end_ids: List[int]
    work_end_ok: bool
    work_end_status: str
    work_end_confidence: float
    axis_direction: Optional[List[float]]
    vlm: Dict[str, Any]
    artifacts: Dict[str, Any]
    grip_point_xy: Optional[List[float]] = field(default=None)
    work_end_edge: Optional[Dict[str, Any]] = field(default=None)  # {left, mid, right}
    work_end_contact_xy: Optional[List[float]] = field(default=None)
    grip_target_line: Optional[List[float]] = field(default=None)
    # Legacy fields kept for compatibility with region_stage.py
    geometry_family: str = field(default="vlm_direct")
    agreement: str = field(default="vlm_direct")
    reasons: List[str] = field(default_factory=list)
    geometry_scores: List[Dict[str, Any]] = field(default_factory=list)
    fused_scores: List[Dict[str, Any]] = field(default_factory=list)
    work_end_agreement: str = field(default="vlm_direct")
    work_end_reasons: List[str] = field(default_factory=list)
    work_end_geometry_scores: List[Dict[str, Any]] = field(default_factory=list)
    work_end_fused_scores: List[Dict[str, Any]] = field(default_factory=list)
    candidate_features: List[Dict[str, Any]] = field(default_factory=list)
    grip_anchor_label: Optional[str] = field(default=None)
    work_end_side_label: Optional[str] = field(default=None)
    contact_face: str = field(default="primary")
    target_region: str = field(default="middle")
    tool_morphology: Optional[str] = field(default=None)
    pgl_handle: Optional[Dict[str, Any]] = field(default=None)
    pgl_work_end: Optional[Dict[str, Any]] = field(default=None)
    geometry_top_id: Optional[int] = field(default=None)
    vlm_top_id: Optional[int] = field(default=None)
    work_end_geometry_top_id: Optional[int] = field(default=None)
    work_end_vlm_top_id: Optional[int] = field(default=None)
    grip_point_3d: Optional[List[float]] = field(default=None)
    work_end_contact_3d: Optional[List[float]] = field(default=None)
    axis_direction_3d: Optional[List[float]] = field(default=None)
    contact_normal_3d: Optional[List[float]] = field(default=None)
    grasp_rotation_cam: Optional[List[List[float]]] = field(default=None)
    grasp_frame_cam: Optional[List[List[float]]] = field(default=None)
    grasp_quaternion_xyzw: Optional[List[float]] = field(default=None)
    grasp_approach_axis_3d: Optional[List[float]] = field(default=None)
    grasp_closing_axis_3d: Optional[List[float]] = field(default=None)
    # Full-image (global) pixel coordinates — needed for interaction lift
    grip_point_xy_global: Optional[List[float]] = field(default=None)
    work_end_contact_xy_global: Optional[List[float]] = field(default=None)
    work_end_edge_points_global: Optional[List[List[float]]] = field(default=None)
    # ---------------------------------------------------------------------------
    # Structured region-level affordance fields (journal contribution)
    # Replaces binary label with a formally-defined tuple grounded jointly by
    # VLM semantic prediction and geometric priors, with calibrated uncertainty.
    # ---------------------------------------------------------------------------
    grasp_type: Optional[str] = field(default=None)          # power|precision|pinch|hook
    contact_geometry: Optional[str] = field(default=None)    # point|line|surface
    force_mode: Optional[str] = field(default=None)          # push|pull|torque|normal
    task_semantic: Optional[str] = field(default=None)       # verb-conditioned label
    affordance_uncertainty: Optional[Dict[str, float]] = field(default=None)
    geometry_vlm_agreement: Optional[float] = field(default=None)
    geometry_vlm_agreement_details: Optional[Dict[str, Any]] = field(default=None)
    coherence_report: Optional[Dict[str, Any]] = field(default=None)
    affordance_hypotheses: Optional[List[Dict[str, Any]]] = field(default=None)
    wrench_basis: Optional[Dict[str, List[float]]] = field(default=None)
    functional_area_px: Optional[int] = field(default=None)
    grasp_width_px: Optional[float] = field(default=None)
    grasp_width_m: Optional[float] = field(default=None)
    grasp_width_mm: Optional[float] = field(default=None)
    handle_affordance: Optional[Dict[str, Any]] = field(default=None)
    work_end_affordance: Optional[Dict[str, Any]] = field(default=None)
    retry_info: Optional[Dict[str, Any]] = field(default=None)


# ---------------------------------------------------------------------------
# Structured affordance representation (journal contribution)
# ---------------------------------------------------------------------------

@dataclass
class RegionAffordanceTuple:
    """Formally structured affordance for one tool region, conditioned on the task.

    The tuple (functional_role, grasp_type, contact_geometry, force_mode)
    replaces the informal binary handle/work-end label.  It is:
      (a) conditioned on the task instruction (action verb)
      (b) grounded jointly by VLM semantic prediction and geometric priors
      (c) equipped with calibrated per-attribute confidence scores

    This structured representation is the key contribution enabling the
    downstream wrench-bounded constraint synthesis stage to select physically
    consistent manipulation primitives without task-specific training.
    """
    functional_role: str                     # "handle" | "work_end"
    grasp_type: str                          # "power" | "precision" | "pinch" | "hook" | "none"
    contact_geometry: str                    # "point" | "line" | "surface"
    force_mode: str                          # "push" | "pull" | "torque" | "normal"
    task_semantic: str                       # verb-conditioned label, e.g. "scrape_blade"
    confidence: float                        # aggregate ∈ [0, 1]
    attribute_confidence: Dict[str, float]   # per-attribute ∈ [0, 1]


# ---------------------------------------------------------------------------
# Low-level mask / skeleton helpers
# ---------------------------------------------------------------------------

def _crop_binary_masks(
    masks: List[np.ndarray], pad: int = 12
) -> Tuple[List[np.ndarray], Optional[Tuple[int, int, int, int]]]:
    if not masks:
        return [], None
    support = union_masks([(m > 0).astype(np.uint8) for m in masks])
    bb = mask_bbox_xyxy(support)
    if bb is None:
        return [(m > 0).astype(np.uint8).copy() for m in masks], None
    h, w = support.shape[:2]
    x1, y1, x2, y2 = bb
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    return [((m > 0).astype(np.uint8))[y1 : y2 + 1, x1 : x2 + 1].copy() for m in masks], (x1, y1, x2, y2)


def _morph_skeleton(mask01: np.ndarray) -> np.ndarray:
    img = (mask01 > 0).astype(np.uint8).copy()
    if mask_area(img) == 0:
        return img
    skel = np.zeros_like(img, dtype=np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(512):
        eroded = cv2.erode(img, ker)
        opened = cv2.morphologyEx(eroded, cv2.MORPH_OPEN, ker)
        temp = cv2.subtract(eroded, opened)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded
        if cv2.countNonZero(img) == 0:
            break
    return (skel > 0).astype(np.uint8)


def _mask_centroid(mask: np.ndarray) -> Tuple[float, float]:
    m = (mask > 0)
    ys, xs = np.where(m)
    if xs.size == 0:
        h, w = mask.shape[:2]
        return float(w / 2), float(h / 2)
    return float(xs.mean()), float(ys.mean())


def _count_components(mask01: np.ndarray) -> int:
    m = (mask01 > 0).astype(np.uint8)
    n, _, _, _ = cv2.connectedComponentsWithStats(m)
    return max(0, int(n - 1))


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def _upscale_for_display(img: np.ndarray, min_px: int = _MIN_VIS_PX) -> np.ndarray:
    h, w = img.shape[:2]
    if max(h, w) >= min_px:
        return img.copy()
    scale = min_px / float(max(h, w))
    nh = max(1, int(round(h * scale)))
    nw = max(1, int(round(w * scale)))
    return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)


def _normalize_direction(
    dx: float, dy: float, fallback: Tuple[float, float] = (1.0, -1.0)
) -> Tuple[float, float]:
    n = float(np.sqrt(dx ** 2 + dy ** 2))
    if n < 1e-6:
        return fallback
    return dx / n, dy / n


def _label_box_topleft(
    *,
    point_xy: Tuple[int, int],
    text_size: Tuple[int, int],
    direction: Tuple[float, float],
    canvas_shape: Tuple[int, ...],
    distance: int = 30,
    pad: int = 4,
) -> Tuple[int, int]:
    px, py = point_xy
    text_w, text_h = text_size
    box_w = text_w + 2 * pad
    box_h = text_h + 2 * pad
    cx = px + int(direction[0] * distance)
    cy = py + int(direction[1] * distance)
    x = int(np.clip(cx - box_w // 2, 0, max(0, canvas_shape[1] - box_w)))
    y = int(np.clip(cy - box_h // 2, 0, max(0, canvas_shape[0] - box_h)))
    return x, y


def _draw_badge(
    canvas: np.ndarray,
    text: str,
    point_xy: Tuple[int, int],
    direction: Tuple[float, float],
    *,
    text_color: Tuple[int, int, int],
    box_color: Tuple[int, int, int] = (255, 255, 255),
    border_color: Tuple[int, int, int] = (24, 24, 24),
    font_scale: float = 0.58,
    text_thickness: int = 2,
    distance: int = 24,
    pad: int = 4,
) -> None:
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
    x, y = _label_box_topleft(
        point_xy=point_xy,
        text_size=(text_w, text_h),
        direction=direction,
        canvas_shape=canvas.shape,
        distance=distance,
        pad=pad,
    )
    box_w = text_w + 2 * pad
    box_h = text_h + 2 * pad
    cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), box_color, -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (x, y), (x + box_w, y + box_h), border_color, 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (x + pad, y + pad + text_h),
                cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, text_thickness, cv2.LINE_AA)


def _draw_point_callout(
    canvas: np.ndarray,
    point_xy: Tuple[int, int],
    label: str,
    color: Tuple[int, int, int],
    *,
    direction: Tuple[float, float],
    marker_type: int,
) -> None:
    dir_x, dir_y = _normalize_direction(direction[0], direction[1])
    cv2.circle(canvas, point_xy, 10, color, -1, cv2.LINE_AA)
    cv2.circle(canvas, point_xy, 10, (255, 255, 255), 2, cv2.LINE_AA)
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    box_x, box_y = _label_box_topleft(
        point_xy=point_xy,
        text_size=(text_w, text_h),
        direction=(dir_x, dir_y),
        canvas_shape=canvas.shape,
        distance=50,
        pad=5,
    )
    box_w = text_w + 10
    box_h = text_h + 10
    box_center = (box_x + box_w // 2, box_y + box_h // 2)
    cv2.line(canvas, point_xy, box_center, color, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), (255, 255, 255), -1, cv2.LINE_AA)
    cv2.rectangle(canvas, (box_x, box_y), (box_x + box_w, box_y + box_h), color, 2, cv2.LINE_AA)
    cv2.putText(canvas, label, (box_x + 5, box_y + text_h + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def _draw_skeleton_overlay(crop_bgr: np.ndarray, crop_masks: List[np.ndarray]) -> np.ndarray:
    canvas = crop_bgr.copy()
    palette = [
        (255, 90, 90), (90, 220, 255), (90, 255, 90),
        (255, 190, 90), (210, 120, 255), (255, 255, 90),
    ]
    for idx, mask in enumerate(crop_masks):
        m = (mask > 0).astype(np.uint8)
        if mask_area(m) == 0:
            continue
        color = palette[idx % len(palette)]
        tint = np.zeros_like(canvas, dtype=np.uint8)
        tint[:, :] = color
        sel = m.astype(bool)
        canvas[sel] = cv2.addWeighted(canvas[sel], 0.78, tint[sel], 0.22, 0.0)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(canvas, cnts, -1, color, 2)
        skel = _morph_skeleton(m)
        skel_sel = skel.astype(bool)
        canvas[skel_sel] = cv2.addWeighted(canvas[skel_sel], 0.25, tint[skel_sel], 0.75, 0.0)
        ys, xs = np.where(m > 0)
        if xs.size > 0:
            cx = int(round(float(xs.mean())))
            cy = int(round(float(ys.mean())))
            lbl = str(idx)
            (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
            lx = max(0, cx + 6)
            ly = max(lh + 6, cy - 6)
            cv2.rectangle(canvas, (lx - 2, ly - lh - 4), (lx + lw + 4, ly + 4), (20, 20, 20), -1, cv2.LINE_AA)
            cv2.putText(canvas, lbl, (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
    return canvas


def _make_mask_grid(crop_bgr: np.ndarray, crop_masks: List[np.ndarray], tile: int = 400, cols: int = 4) -> np.ndarray:
    if crop_bgr is None or crop_bgr.size == 0:
        return np.zeros((tile, tile, 3), dtype=np.uint8)
    if not crop_masks:
        return cv2.resize(crop_bgr, (tile, tile), interpolation=cv2.INTER_LINEAR)
    palette = [
        (255, 90, 90), (90, 220, 255), (90, 255, 90),
        (255, 190, 90), (210, 120, 255), (255, 255, 90),
    ]
    tiles: List[np.ndarray] = []
    for idx, mask in enumerate(crop_masks):
        vis = crop_bgr.copy()
        m = (mask > 0).astype(np.uint8)
        color = palette[idx % len(palette)]
        tint = np.zeros_like(vis, dtype=np.uint8)
        tint[:, :] = color
        sel = m.astype(bool)
        vis[sel] = cv2.addWeighted(vis[sel], 0.65, tint[sel], 0.35, 0.0)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 2)
        h, w = vis.shape[:2]
        scale = tile / float(max(h, w))
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        resized = cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_LINEAR)
        out = np.zeros((tile, tile, 3), dtype=np.uint8)
        oy = (tile - nh) // 2
        ox = (tile - nw) // 2
        out[oy : oy + nh, ox : ox + nw] = resized
        lbl = str(idx)
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.90, 2)
        cv2.rectangle(out, (8, 8), (8 + lw + 10, 8 + lh + 10), color, -1, cv2.LINE_AA)
        cv2.putText(out, lbl, (13, 8 + lh + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.90, (255, 255, 255), 2, cv2.LINE_AA)
        tiles.append(out)
    rows = int(np.ceil(len(tiles) / float(cols)))
    grid = np.zeros((rows * tile, cols * tile, 3), dtype=np.uint8)
    for i, tile_img in enumerate(tiles):
        r = i // cols
        c = i % cols
        grid[r * tile : (r + 1) * tile, c * tile : (c + 1) * tile] = tile_img
    return grid


# ---------------------------------------------------------------------------
# Journal composite figure
# ---------------------------------------------------------------------------

def make_journal_figure(
    rgb_crop_path: Optional[str],
    candidate_grid_path: Optional[str],
    handle_vis_path: Optional[str],
    work_end_vis_path: Optional[str],
    out_path: str,
    *,
    panel_titles: Tuple[str, str, str, str] = ("Input", "Candidates", "Handle region", "Work-end region"),
    dpi: int = 220,
) -> Optional[str]:
    """Compose a 2×2 journal-quality figure with panel labels (A)–(D)."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        paths = [rgb_crop_path, candidate_grid_path, handle_vis_path, work_end_vis_path]
        imgs = []
        for p in paths:
            if p and os.path.exists(p):
                bgr = cv2.imread(p)
                imgs.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None)
            else:
                imgs.append(None)

        if all(im is None for im in imgs):
            return None

        fig, axes = plt.subplots(2, 2, figsize=(10.0, 7.4), dpi=dpi)
        fig.patch.set_facecolor("white")
        panel_letters = ["A", "B", "C", "D"]

        for ax, im, title, letter in zip(axes.flat, imgs, panel_titles, panel_letters):
            ax.set_facecolor("#f4f4f4")
            if im is not None:
                ax.imshow(im)
            else:
                ax.text(0.5, 0.5, "n/a", ha="center", va="center",
                        fontsize=12, color="#aaaaaa", transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_edgecolor("#cccccc")
            ax.text(0.02, 0.04, f"({letter})", transform=ax.transAxes,
                    fontsize=14, fontweight="bold", color="white",
                    bbox=dict(boxstyle="round,pad=0.18", facecolor="#444444", edgecolor="none", alpha=0.82),
                    va="bottom", ha="left")
            ax.set_title(title, fontsize=10, color="#222222", pad=4, fontweight="600")

        fig.tight_layout(pad=0.6, w_pad=0.5, h_pad=0.5)
        ensure_dir(os.path.dirname(out_path))
        fig.savefig(out_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out_path
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Region rendering with axis
# ---------------------------------------------------------------------------

def _render_region(
    crop_bgr: np.ndarray,
    mask: np.ndarray,
    color_bgr: Tuple[int, int, int],
    label: str,
) -> np.ndarray:
    """Upscaled region tint + contour + skeleton — no axis line."""
    vis = _upscale_for_display(crop_bgr)
    h1, w1 = vis.shape[:2]
    m = cv2.resize((mask > 0).astype(np.uint8), (w1, h1), interpolation=cv2.INTER_NEAREST)

    tint = np.full_like(vis, color_bgr, dtype=np.uint8)
    sel = m.astype(bool)
    if sel.any():
        blended = (vis[sel].astype(np.float32) * 0.72 + tint[sel].astype(np.float32) * 0.28).clip(0, 255).astype(np.uint8)
        vis[sel] = blended
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, color_bgr, 2, cv2.LINE_AA)

    skel = _morph_skeleton(m)
    skel_col = tuple(max(0, int(c) - 70) for c in color_bgr)
    vis[skel.astype(bool)] = skel_col  # type: ignore[assignment]

    cv2.putText(vis, label, (8, h1 - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 140, 140), 1, cv2.LINE_AA)
    return vis


def _render_axis_image(
    crop_bgr: np.ndarray,
    h_mask: Optional[np.ndarray],
    we_mask: Optional[np.ndarray],
    handle_center: Tuple[float, float],
    work_end_center: Tuple[float, float],
) -> np.ndarray:
    """Crop with both regions lightly tinted and the tool axis arrow drawn."""
    vis = _upscale_for_display(crop_bgr)
    h0, w0 = crop_bgr.shape[:2]
    h1, w1 = vis.shape[:2]
    sx = w1 / max(w0, 1)
    sy = h1 / max(h0, 1)

    for mask, color in [(h_mask, (50, 200, 50)), (we_mask, (50, 80, 220))]:
        if mask is None or mask_area((mask > 0).astype(np.uint8)) == 0:
            continue
        m = cv2.resize((mask > 0).astype(np.uint8), (w1, h1), interpolation=cv2.INTER_NEAREST)
        tint = np.full_like(vis, color, dtype=np.uint8)
        sel = m.astype(bool)
        vis[sel] = cv2.addWeighted(vis[sel], 0.80, tint[sel], 0.20, 0.0)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 1, cv2.LINE_AA)

    p1 = (int(round(handle_center[0] * sx)), int(round(handle_center[1] * sy)))
    p2 = (int(round(work_end_center[0] * sx)), int(round(work_end_center[1] * sy)))
    cv2.arrowedLine(vis, p1, p2, (0, 140, 255), 2, cv2.LINE_AA, tipLength=0.06)
    cv2.circle(vis, p1, 6, (50, 200, 50), -1, cv2.LINE_AA)
    cv2.circle(vis, p2, 6, (0, 140, 255), -1, cv2.LINE_AA)
    cv2.putText(vis, "H", (p1[0] + 8, p1[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (50, 200, 50), 2, cv2.LINE_AA)
    cv2.putText(vis, "WE", (p2[0] + 8, p2[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 140, 255), 2, cv2.LINE_AA)
    return vis


def _render_selected_skeleton(
    crop_bgr: np.ndarray,
    h_mask: Optional[np.ndarray],
    we_mask: Optional[np.ndarray],
) -> np.ndarray:
    """Skeleton of the selected handle (green) and work-end (blue) on the crop."""
    vis = _upscale_for_display(crop_bgr)
    h1, w1 = vis.shape[:2]

    for mask, color in [(h_mask, (50, 200, 50)), (we_mask, (50, 80, 220))]:
        if mask is None or mask_area((mask > 0).astype(np.uint8)) == 0:
            continue
        m = cv2.resize((mask > 0).astype(np.uint8), (w1, h1), interpolation=cv2.INTER_NEAREST)
        skel = _morph_skeleton(m)
        vis[skel.astype(bool)] = color
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 1, cv2.LINE_AA)

    return vis


def _make_combined_pgl(
    crop_bgr: np.ndarray,
    h_mask: Optional[np.ndarray],
    we_mask: Optional[np.ndarray],
    grip_xy: Optional[List[float]] = None,
    contact_xy: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Both masks overlaid on one image: green=Handle, red=Work-End, with G and C callouts."""
    vis = _upscale_for_display(crop_bgr)
    h1, w1 = vis.shape[:2]
    h0, w0 = crop_bgr.shape[:2]
    sx, sy = w1 / max(w0, 1), h1 / max(h0, 1)

    layers = [
        (h_mask,  (50, 200, 50),  "Handle"),
        (we_mask, (50,  80, 220), "Work-End"),
    ]
    label_y = h1 - 8
    for mask, color, label in layers:
        if mask is None or mask_area((mask > 0).astype(np.uint8)) == 0:
            continue
        m = cv2.resize((mask > 0).astype(np.uint8), (w1, h1), interpolation=cv2.INTER_NEAREST)
        tint = np.full_like(vis, color, dtype=np.uint8)
        sel = m.astype(bool)
        vis[sel] = cv2.addWeighted(vis[sel], 0.72, tint[sel], 0.28, 0.0)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cnts, -1, color, 2, cv2.LINE_AA)
        skel = _morph_skeleton(m)
        skel_col = tuple(max(0, int(c) - 70) for c in color)
        vis[skel.astype(bool)] = skel_col  # type: ignore[assignment]
        cv2.putText(vis, label, (8, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
        label_y -= 14

    # G — grip point callout
    if grip_xy is not None:
        gx = int(round(grip_xy[0] * sx))
        gy = int(round(grip_xy[1] * sy))
        _draw_point_callout(vis, (gx, gy), "G", (30, 30, 220),
                            direction=(0.7, -0.7), marker_type=cv2.MARKER_CROSS)

    # C — contact point callout
    if contact_xy is not None:
        cx = int(round(contact_xy[0] * sx))
        cy = int(round(contact_xy[1] * sy))
        _draw_point_callout(vis, (cx, cy), "C", (200, 50, 0),
                            direction=(0.7, -0.7), marker_type=cv2.MARKER_CROSS)

    return vis


# ---------------------------------------------------------------------------
# VLM-1  –  part selection  (handle_id + work_end_id)
# ---------------------------------------------------------------------------

def _build_part_selection_prompt(
    instruction: str,
    object_name: str,
    n_candidates: int,
    has_depth: bool,
    has_target: bool = False,
) -> str:
    """Build the task-conditioned structured affordance grounding prompt (VLM-1).

    The prompt requests a formally structured affordance tuple per region, enabling
    the downstream wrench-bounded constraint synthesis to operate without task-specific
    training.  Output attributes are conditioned on the action verb extracted from
    the instruction, grounding each attribute to the manipulation context.

    When has_target=True, a crop of the target region is appended as an additional
    image so that contact_geometry and force_mode are conditioned on the actual
    target surface rather than solely on the task verb.
    """
    _skel_img_no = 4 if has_depth else 3
    _tgt_img_no = _skel_img_no + 1
    image_desc = (
        "Image 1: RGB crop of the tool.\n"
        "Image 2: candidate mask grid — each region has a numbered badge (0 … N).\n"
        + ("Image 3: depth colormap (warm = close, cool = far).\n" if has_depth else "")
        + f"Image {_skel_img_no}: skeleton overlay showing the medial axis of each candidate region.\n"
        + (f"Image {_tgt_img_no}: RGB crop of the TARGET region the tool will contact.\n" if has_target else "")
    )
    target_context = ""
    if has_target:
        target_context = (
            f"\n=== TARGET SURFACE CONTEXT (Image {_tgt_img_no}) ===\n"
            "Inspect the target region image before predicting contact_geometry and force_mode.\n"
            "Let the actual surface properties drive your prediction:\n"
            "  · Rough or textured surface  → prefer surface contact_geometry\n"
            "  · Hard flat surface          → line or surface; push or normal force_mode\n"
            "  · Soft or deformable surface → surface; normal force_mode\n"
            "  · Point or cavity target     → point contact_geometry\n"
            "  · Long straight edge/groove  → line contact_geometry; pull force_mode\n"
            "  · Screw head / hex socket    → point; torque force_mode\n"
            "Your contact_geometry and force_mode must be consistent with BOTH the task verb "
            "AND the target surface visible in this image.\n\n"
        )
    return (
        f"Task: {instruction}\n"
        f"Tool: {object_name}\n\n"
        + image_desc
        + "\n=== PART I — Region Identification ===\n"
        "Identify the two functional parts:\n"
        "  HANDLE   — the graspable shaft or grip: elongated, stable, uniform width.\n"
        "  WORK_END — the functional terminal that contacts or acts on the target.\n\n"
        f"Choose one integer ID each from 0 to {n_candidates - 1}. They must be different.\n"
        "If the work-end cannot be identified with confidence, set work_end_id to null.\n\n"
        "contact_face — which face of the work-end makes contact with the target:\n"
        '  "primary"   — main working face: tip, blade edge, bristle end, striking face.\n'
        '  "secondary" — opposite face: used only when task requires back/reverse side.\n'
        '  "lateral"   — side face perpendicular to tool axis: chisel bevel, knife flat.\n\n'
        "tool_morphology:\n"
        '  "inline"  — contact face along tool axis (scraper, brush, screwdriver, spatula).\n'
        '  "L_shape" — contact face PERPENDICULAR to handle axis (hammer, hex key, '
        "toothbrush).\n\n"
        "face_direction (L_shape only) — image-y side of the contact face:\n"
        '  "upper" — contact face toward TOP of image.\n'
        '  "lower" — contact face toward BOTTOM of image (most common).\n\n'
        "target_region — where on the target to apply the tool:\n"
        '  "border"      — along the edge/border.\n'
        '  "middle"      — across the center.\n'
        '  "face_center" — into a hollow cavity.\n\n'
        "=== PART II — Structured Affordance Attributes (task-conditioned) ===\n"
        "Predict per-region affordance attributes conditioned on the action verb in the "
        "task instruction.  These ground each region into a formal manipulation primitive.\n\n"
        + target_context
        + "grasp_type — how the human hand grips the HANDLE:\n"
        '  "power"     — full-hand cylindrical grip (hammer, screwdriver, heavy tool shaft).\n'
        '  "precision" — thumb + 2-finger fingertip grip (thin shaft, pen-like tool).\n'
        '  "pinch"     — thumb + index tip pinch (very narrow rod, pin, small handle).\n'
        '  "hook"      — hook or overhand grip (trowel, L-shape head, hook-handle).\n\n'
        "contact_geometry — surface type of the WORK_END contact region:\n"
        '  "point"   — concentrated point contact (screwdriver tip, punch, needle).\n'
        '  "line"    — linear edge contact (scraper blade, knife edge, chisel edge).\n'
        '  "surface" — broad area contact (spatula face, roller, brush, sponge).\n\n'
        "force_mode — dominant wrench component at the WORK_END during task execution:\n"
        '  "push"   — compressive force along tool axis (hammering, pressing, driving).\n'
        '  "pull"   — tensile/dragging force (scraping, raking, pulling).\n'
        '  "torque" — rotational moment (screwdriver, wrench, drill).\n'
        '  "normal" — force normal to surface (painting, coating, brushing).\n\n'
        "task_semantic — a compact verb-part label encoding the action-affordance pair,\n"
        '  e.g. "scrape_blade", "hammer_head", "turn_bit", "paint_bristle".\n\n'
        "For each attribute also output a per-attribute confidence in [0.0, 1.0].\n\n"
        "Return JSON ONLY — no prose before or after:\n"
        "{\n"
        '  "handle_id": <int>,\n'
        '  "work_end_id": <int_or_null>,\n'
        '  "tool_morphology": "inline"|"L_shape",\n'
        '  "contact_face": "primary"|"secondary"|"lateral",\n'
        '  "face_direction": "upper"|"lower",\n'
        '  "target_region": "border"|"middle"|"face_center",\n'
        '  "confidence": 0.0-1.0,\n'
        '  "grasp_type": "power"|"precision"|"pinch"|"hook",\n'
        '  "contact_geometry": "point"|"line"|"surface",\n'
        '  "force_mode": "push"|"pull"|"torque"|"normal",\n'
        '  "task_semantic": "<verb>_<part>",\n'
        '  "handle_confidence": 0.0-1.0,\n'
        '  "work_end_confidence": 0.0-1.0,\n'
        '  "attribute_confidence": {\n'
        '    "grasp_type": 0.0-1.0,\n'
        '    "contact_geometry": 0.0-1.0,\n'
        '    "force_mode": 0.0-1.0\n'
        "  },\n"
        '  "reason": "<15 words describing key visual evidence>"\n'
        "}\n"
    )


_VALID_CONTACT_FACES = {"primary", "secondary", "lateral"}
_VALID_GRASP_TYPES = {"power", "precision", "pinch", "hook", "none"}
_VALID_CONTACT_GEOMS = {"point", "line", "surface"}
_VALID_FORCE_MODES = {"push", "pull", "torque", "normal"}
_AGREEMENT_GRASP_LABELS = ["power", "precision", "pinch", "hook"]
_AGREEMENT_CONTACT_LABELS = ["point", "line", "surface"]
_AGREEMENT_FORCE_LABELS = ["push", "pull", "torque", "normal"]


def _parse_part_selection(raw_text: str, n_candidates: int) -> Dict[str, Any]:
    """Parse the structured affordance grounding VLM response.

    Returns a dict containing both the classical fields (handle_id, contact_face …)
    and the new structured affordance attributes (grasp_type, contact_geometry,
    force_mode, task_semantic, per-attribute confidences).  All new fields default
    gracefully so callers that only use classical fields remain unaffected.
    """
    parsed = extract_json_block(raw_text)
    result: Dict[str, Any] = {
        # Classical fields
        "handle_id": None, "work_end_id": None,
        "tool_morphology": "inline",
        "contact_face": "primary",
        "face_direction": "lower",
        "target_region": "middle",
        "confidence": 0.0,
        # Structured affordance attributes (new)
        "grasp_type": "power",
        "contact_geometry": "surface",
        "force_mode": "push",
        "task_semantic": "",
        "handle_confidence": 0.0,
        "work_end_confidence": 0.0,
        "attribute_confidence": {},
    }
    if not isinstance(parsed, dict):
        return result

    # Region IDs
    for key in ("handle_id", "work_end_id"):
        v = parsed.get(key)
        if v is None:
            continue
        try:
            vid = int(v)
            if 0 <= vid < n_candidates:
                result[key] = vid
        except Exception:
            pass
    if result["handle_id"] is not None and result["work_end_id"] == result["handle_id"]:
        result["work_end_id"] = None

    # Classical morphology / contact / region fields
    tm = str(parsed.get("tool_morphology", "inline") or "inline").strip().lower()
    result["tool_morphology"] = "L_shape" if tm in ("l_shape", "lshape", "l shape") else "inline"
    cf = str(parsed.get("contact_face", "primary") or "primary").strip().lower()
    result["contact_face"] = cf if cf in _VALID_CONTACT_FACES else "primary"
    fd = str(parsed.get("face_direction", "lower") or "lower").strip().lower()
    result["face_direction"] = fd if fd in ("upper", "lower") else "lower"
    tr = str(parsed.get("target_region", "middle") or "middle").strip().lower()
    result["target_region"] = tr if tr in ("border", "middle", "face_center") else "middle"
    try:
        result["confidence"] = float(np.clip(float(parsed.get("confidence", 0.0) or 0.0), 0.0, 1.0))
    except Exception:
        pass

    # Structured affordance attributes
    gt = str(parsed.get("grasp_type", "power") or "power").strip().lower()
    result["grasp_type"] = gt if gt in _VALID_GRASP_TYPES else "power"
    cg = str(parsed.get("contact_geometry", "surface") or "surface").strip().lower()
    result["contact_geometry"] = cg if cg in _VALID_CONTACT_GEOMS else "surface"
    fm = str(parsed.get("force_mode", "push") or "push").strip().lower()
    result["force_mode"] = fm if fm in _VALID_FORCE_MODES else "push"
    result["task_semantic"] = str(parsed.get("task_semantic", "") or "").strip()[:64]
    try:
        result["handle_confidence"] = float(np.clip(
            float(parsed.get("handle_confidence", result["confidence"]) or result["confidence"]), 0.0, 1.0))
    except Exception:
        result["handle_confidence"] = result["confidence"]
    try:
        result["work_end_confidence"] = float(np.clip(
            float(parsed.get("work_end_confidence", result["confidence"]) or result["confidence"]), 0.0, 1.0))
    except Exception:
        result["work_end_confidence"] = result["confidence"]

    raw_attr = parsed.get("attribute_confidence", {}) or {}
    if isinstance(raw_attr, dict):
        for k, v in raw_attr.items():
            try:
                result["attribute_confidence"][str(k)] = float(np.clip(float(v or 0.0), 0.0, 1.0))
            except Exception:
                pass

    return result


def _build_handle_flip_retry_prompt(
    *,
    instruction: str,
    object_name: str,
    candidate_a: Dict[str, Any],
    candidate_b: Dict[str, Any],
    has_depth: bool,
    geom_handle_id: Optional[int] = None,
    geom_work_end_id: Optional[int] = None,
    mask_stats: Optional[List[Dict[str, Any]]] = None,
) -> str:
    image_desc = (
        "Image 1: RGB crop of the tool.\n"
        "Image 2: candidate mask grid with numeric IDs.\n"
        + ("Image 3: depth colormap.\n" if has_depth else "")
        + "Image 4: skeleton overlay.\n"
    )

    # Determine which candidate the geometry analysis endorses.
    geom_section = ""
    if geom_handle_id is not None and geom_work_end_id is not None:
        geom_agrees_with = None
        if (candidate_a["handle_id"] == geom_handle_id
                and candidate_a["work_end_id"] == geom_work_end_id):
            geom_agrees_with = "A"
        elif (candidate_b["handle_id"] == geom_handle_id
                and candidate_b["work_end_id"] == geom_work_end_id):
            geom_agrees_with = "B"

        # Pull elongation and area stats for the geometry-predicted handle.
        geom_stats_desc = ""
        if mask_stats:
            h_stat = next((s for s in mask_stats
                           if int(s.get("id", -1)) == geom_handle_id), None)
            we_stat = next((s for s in mask_stats
                            if int(s.get("id", -1)) == geom_work_end_id), None)
            parts = []
            if h_stat:
                parts.append(
                    f"region {geom_handle_id}: elongation="
                    f"{float(h_stat.get('elongation', 0.0)):.2f}, "
                    f"area_ratio={float(h_stat.get('area_ratio_vs_support', 0.0)):.2f}"
                )
            if we_stat:
                parts.append(
                    f"region {geom_work_end_id}: elongation="
                    f"{float(we_stat.get('elongation', 0.0)):.2f}, "
                    f"area_ratio={float(we_stat.get('area_ratio_vs_support', 0.0)):.2f}"
                )
            if parts:
                geom_stats_desc = " (" + "; ".join(parts) + ")"

        if geom_agrees_with:
            geom_section = (
                f"\n*** GEOMETRY ANALYSIS RECOMMENDATION ***\n"
                f"Shape-based analysis predicts handle_id={geom_handle_id}, "
                f"work_end_id={geom_work_end_id}{geom_stats_desc}.\n"
                f"This matches Candidate {geom_agrees_with}. "
                f"Your initial choice disagreed with this geometric evidence, "
                f"which triggered this review. Reconsider carefully — the "
                f"geometry signal is based on elongation and area measurements "
                f"of the mask regions.\n"
            )
        else:
            geom_section = (
                f"\n*** GEOMETRY ANALYSIS RECOMMENDATION ***\n"
                f"Shape-based analysis predicts handle_id={geom_handle_id}, "
                f"work_end_id={geom_work_end_id}{geom_stats_desc}.\n"
                f"This does not cleanly match either candidate label, but "
                f"your initial choice had low geometric-semantic agreement. "
                f"Reconsider which candidate is more consistent with the "
                f"tool geometry.\n"
            )

    return (
        f"Task: {instruction}\n"
        f"Tool: {object_name}\n\n"
        + image_desc
        + geom_section
        + "\nYou must resolve a binary ambiguity in handle/work-end assignment.\n"
        + "Your INITIAL selection had low geometry-VLM agreement, meaning the\n"
        + "shape analysis disagrees with your choice. Reconsider carefully.\n\n"
        + f"Candidate A (your initial choice): handle_id={candidate_a['handle_id']}, work_end_id={candidate_a['work_end_id']}.\n"
        + f"Candidate B (swapped assignment): handle_id={candidate_b['handle_id']}, work_end_id={candidate_b['work_end_id']}.\n\n"
        + "Use the following principles:\n"
        + "- HANDLE must be the graspable shaft or stable grip region (where a human holds the tool).\n"
        + "- WORK_END must be the terminal functional contact region (blade, bristles, pad, tip).\n"
        + "- The visually prominent or larger region is often the WORK_END, not the handle.\n"
        + "- Prefer the assignment consistent with the geometry recommendation above.\n\n"
        + "Return JSON ONLY:\n"
        + "{\n"
        + '  "choice": "A"|"B",\n'
        + '  "confidence": 0.0-1.0,\n'
        + '  "tool_morphology": "inline"|"L_shape",\n'
        + '  "contact_face": "primary"|"secondary"|"lateral",\n'
        + '  "face_direction": "upper"|"lower",\n'
        + '  "target_region": "border"|"middle"|"face_center",\n'
        + '  "grasp_type": "power"|"precision"|"pinch"|"hook",\n'
        + '  "contact_geometry": "point"|"line"|"surface",\n'
        + '  "force_mode": "push"|"pull"|"torque"|"normal",\n'
        + '  "task_semantic": "<verb>_<part>",\n'
        + '  "handle_confidence": 0.0-1.0,\n'
        + '  "work_end_confidence": 0.0-1.0,\n'
        + '  "attribute_confidence": {\n'
        + '    "grasp_type": 0.0-1.0,\n'
        + '    "contact_geometry": 0.0-1.0,\n'
        + '    "force_mode": 0.0-1.0\n'
        + "  },\n"
        + '  "reason": "<short contrastive explanation>"\n'
        + "}\n"
    )


def _parse_handle_flip_retry(
    raw_text: str,
    *,
    candidate_a: Dict[str, Any],
    candidate_b: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    parsed = extract_json_block(raw_text)
    if not isinstance(parsed, dict):
        return None
    choice = str(parsed.get("choice", "") or "").strip().upper()
    if choice not in {"A", "B"}:
        return None
    base = dict(candidate_a if choice == "A" else candidate_b)
    base["confidence"] = float(np.clip(float(parsed.get("confidence", base.get("confidence", 0.0)) or 0.0), 0.0, 1.0))
    tm = str(parsed.get("tool_morphology", base.get("tool_morphology", "inline")) or "inline").strip().lower()
    base["tool_morphology"] = "L_shape" if tm in ("l_shape", "lshape", "l shape") else "inline"
    cf = str(parsed.get("contact_face", base.get("contact_face", "primary")) or "primary").strip().lower()
    base["contact_face"] = cf if cf in _VALID_CONTACT_FACES else "primary"
    fd = str(parsed.get("face_direction", base.get("face_direction", "lower")) or "lower").strip().lower()
    base["face_direction"] = fd if fd in ("upper", "lower") else "lower"
    tr = str(parsed.get("target_region", base.get("target_region", "middle")) or "middle").strip().lower()
    base["target_region"] = tr if tr in ("border", "middle", "face_center") else "middle"
    gt = str(parsed.get("grasp_type", base.get("grasp_type", "power")) or "power").strip().lower()
    base["grasp_type"] = gt if gt in _VALID_GRASP_TYPES else base.get("grasp_type", "power")
    cg = str(parsed.get("contact_geometry", base.get("contact_geometry", "surface")) or "surface").strip().lower()
    base["contact_geometry"] = cg if cg in _VALID_CONTACT_GEOMS else base.get("contact_geometry", "surface")
    fm = str(parsed.get("force_mode", base.get("force_mode", "push")) or "push").strip().lower()
    base["force_mode"] = fm if fm in _VALID_FORCE_MODES else base.get("force_mode", "push")
    base["task_semantic"] = str(parsed.get("task_semantic", base.get("task_semantic", "")) or "").strip()[:64]
    try:
        base["handle_confidence"] = float(np.clip(float(parsed.get("handle_confidence", base.get("handle_confidence", base["confidence"])) or 0.0), 0.0, 1.0))
    except Exception:
        pass
    try:
        base["work_end_confidence"] = float(np.clip(float(parsed.get("work_end_confidence", base.get("work_end_confidence", base["confidence"])) or 0.0), 0.0, 1.0))
    except Exception:
        pass
    raw_attr = parsed.get("attribute_confidence", {}) or {}
    if isinstance(raw_attr, dict):
        attr: Dict[str, float] = {}
        for k, v in raw_attr.items():
            try:
                attr[str(k)] = float(np.clip(float(v or 0.0), 0.0, 1.0))
            except Exception:
                pass
        base["attribute_confidence"] = attr
    base["retry_choice"] = choice
    base["retry_reason"] = str(parsed.get("reason", "") or "").strip()[:160]
    return base


def _normalize_label_distribution(
    labels: List[str],
    dist: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    if not labels:
        return {}
    uniform = 1.0 / float(len(labels))
    out = {label: uniform for label in labels}
    if isinstance(dist, dict):
        vals: Dict[str, float] = {}
        for label in labels:
            try:
                vals[label] = max(0.0, float(dist.get(label, 0.0)))
            except Exception:
                vals[label] = 0.0
        total = float(sum(vals.values()))
        if total > 1e-12:
            out = {label: float(vals[label] / total) for label in labels}
    return out


def _fuse_label_distributions(
    labels: List[str],
    geom_dist: Optional[Dict[str, Any]],
    vlm_dist: Optional[Dict[str, Any]],
) -> Dict[str, float]:
    geom = _normalize_label_distribution(labels, geom_dist)
    vlm = _normalize_label_distribution(labels, vlm_dist)
    fused = {label: float(geom[label] * vlm[label]) for label in labels}
    total = float(sum(fused.values()))
    if total <= 1e-12:
        avg = {label: 0.5 * (geom[label] + vlm[label]) for label in labels}
        total = float(sum(avg.values()))
        if total <= 1e-12:
            return geom
        return {label: float(avg[label] / total) for label in labels}
    return {label: float(fused[label] / total) for label in labels}


def _evaluate_affordance_triple_coherence(
    grasp_type: str,
    contact_geometry: str,
    force_mode: str,
) -> Dict[str, Any]:
    grasp = str(grasp_type or "").strip().lower()
    contact = str(contact_geometry or "").strip().lower()
    force = str(force_mode or "").strip().lower()
    reasons: List[str] = []
    suggestions: List[str] = []
    status = "valid"
    weight = 1.0

    def _mark_improbable(reason: str, suggestion: Optional[str] = None) -> None:
        nonlocal status, weight
        if status != "forbidden":
            status = "improbable"
            weight = min(weight, COHERENCE_IMPROBABLE_WEIGHT)
        reasons.append(reason)
        if suggestion:
            suggestions.append(suggestion)

    def _mark_forbidden(reason: str, suggestion: Optional[str] = None) -> None:
        nonlocal status, weight
        status = "forbidden"
        weight = 0.0
        reasons.append(reason)
        if suggestion:
            suggestions.append(suggestion)

    if grasp == "hook" and force == "torque":
        _mark_forbidden(
            "hook grips cannot reliably transmit rotational torque.",
            "Use power or precision grip for torque tasks, or switch force_mode away from torque.",
        )
    if force == "torque" and contact == "surface":
        _mark_forbidden(
            "torque execution usually requires point or line contact, not broad surface contact.",
            "Prefer point or line contact_geometry for torque tasks.",
        )
    if grasp == "pinch" and force == "push":
        _mark_improbable(
            "pinch grips are weak for sustained pushing loads.",
            "Prefer precision or power grip for push tasks.",
        )
    if grasp == "pinch" and force == "normal" and contact == "surface":
        _mark_improbable(
            "pinch grips are biomechanically weak for broad-area normal-force application.",
            "Prefer precision or power grip for surface-normal loading.",
        )

    return {
        "status": status,
        "weight": float(weight),
        "reasons": reasons,
        "suggestions": suggestions,
        "triple": {
            "grasp_type": grasp,
            "contact_geometry": contact,
            "force_mode": force,
        },
    }


def _rank_affordance_triple_hypotheses(
    *,
    geometry_vlm_agreement_details: Dict[str, Any],
) -> Dict[str, Any]:
    geom_dists = dict(geometry_vlm_agreement_details.get("geometry_distributions", {}) or {})
    vlm_dists = dict(geometry_vlm_agreement_details.get("vlm_distributions", {}) or {})

    grasp_dist = _fuse_label_distributions(
        _AGREEMENT_GRASP_LABELS,
        (geom_dists.get("grasp_type") or {}).get("distribution"),
        (vlm_dists.get("grasp_type") or {}).get("distribution"),
    )
    contact_dist = _fuse_label_distributions(
        _AGREEMENT_CONTACT_LABELS,
        (geom_dists.get("contact_geometry") or {}).get("distribution"),
        (vlm_dists.get("contact_geometry") or {}).get("distribution"),
    )
    force_dist = _fuse_label_distributions(
        _AGREEMENT_FORCE_LABELS,
        (geom_dists.get("force_mode") or {}).get("distribution"),
        (vlm_dists.get("force_mode") or {}).get("distribution"),
    )

    hypotheses: List[Dict[str, Any]] = []
    for grasp_type in _AGREEMENT_GRASP_LABELS:
        for contact_geometry in _AGREEMENT_CONTACT_LABELS:
            for force_mode in _AGREEMENT_FORCE_LABELS:
                raw_joint = float(
                    grasp_dist.get(grasp_type, 0.0)
                    * contact_dist.get(contact_geometry, 0.0)
                    * force_dist.get(force_mode, 0.0)
                )
                coherence = _evaluate_affordance_triple_coherence(grasp_type, contact_geometry, force_mode)
                hypotheses.append({
                    "triple": {
                        "grasp_type": grasp_type,
                        "contact_geometry": contact_geometry,
                        "force_mode": force_mode,
                    },
                    "raw_joint": raw_joint,
                    "coherence": coherence,
                    "adjusted_score": float(raw_joint * float(coherence["weight"])),
                })

    hypotheses.sort(key=lambda item: float(item.get("adjusted_score", 0.0)), reverse=True)
    raw_total = float(sum(float(h.get("raw_joint", 0.0)) for h in hypotheses))
    adjusted_total = float(sum(float(h.get("adjusted_score", 0.0)) for h in hypotheses))
    for hyp in hypotheses:
        raw_joint = float(hyp.get("raw_joint", 0.0))
        adjusted_score = float(hyp.get("adjusted_score", 0.0))
        hyp["raw_probability"] = float(raw_joint / raw_total) if raw_total > 1e-12 else 0.0
        hyp["adjusted_probability"] = float(adjusted_score / adjusted_total) if adjusted_total > 1e-12 else 0.0
    valid_hypotheses = [h for h in hypotheses if str((h.get("coherence") or {}).get("status")) == "valid"]
    return {
        "fused_distributions": {
            "grasp_type": grasp_dist,
            "contact_geometry": contact_dist,
            "force_mode": force_dist,
        },
        "hypotheses": hypotheses,
        "valid_hypotheses": valid_hypotheses,
        "probability_mass": {
            "raw_total": raw_total,
            "adjusted_total": adjusted_total,
        },
    }


def _build_affordance_coherence_retry_prompt(
    *,
    instruction: str,
    object_name: str,
    current_triple: Dict[str, str],
    invalid_reasons: List[str],
    candidate_options: List[Dict[str, Any]],
    has_depth: bool,
    has_target: bool,
) -> str:
    image_desc = (
        "Image 1: RGB crop of the tool.\n"
        "Image 2: candidate mask grid with fixed handle/work-end IDs.\n"
        + ("Image 3: depth colormap.\n" if has_depth else "")
        + "Image 4: skeleton overlay.\n"
        + ("Image 5: target region crop.\n" if has_target else "")
    )
    option_lines: List[str] = []
    for opt in candidate_options:
        triple = opt.get("triple", {}) or {}
        option_lines.append(
            f"Candidate {opt['label']}: grasp_type={triple.get('grasp_type')}, "
            f"contact_geometry={triple.get('contact_geometry')}, force_mode={triple.get('force_mode')}."
        )
    reasons_block = "\n".join(f"- {reason}" for reason in invalid_reasons) if invalid_reasons else "- Mechanical coherence is low."
    return (
        f"Task: {instruction}\n"
        f"Tool: {object_name}\n\n"
        + image_desc
        + "\nThe handle/work-end IDs are already fixed. Only resolve the affordance triple.\n"
        + "Current triple appears mechanically inconsistent:\n"
        + f"Current: grasp_type={current_triple['grasp_type']}, contact_geometry={current_triple['contact_geometry']}, force_mode={current_triple['force_mode']}\n"
        + reasons_block + "\n\n"
        + "Choose the most physically consistent candidate for this tool and task.\n"
        + "\n".join(option_lines)
        + "\n\nReturn JSON ONLY:\n"
        + "{\n"
        + '  "choice": "A"|"B"|"C",\n'
        + '  "confidence": 0.0-1.0,\n'
        + '  "task_semantic": "<verb>_<part>",\n'
        + '  "attribute_confidence": {\n'
        + '    "grasp_type": 0.0-1.0,\n'
        + '    "contact_geometry": 0.0-1.0,\n'
        + '    "force_mode": 0.0-1.0\n'
        + "  },\n"
        + '  "reason": "<short mechanical explanation>"\n'
        + "}\n"
    )


def _parse_affordance_coherence_retry(
    raw_text: str,
    *,
    candidate_options: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    parsed = extract_json_block(raw_text)
    if not isinstance(parsed, dict):
        return None
    choice = str(parsed.get("choice", "") or "").strip().upper()
    chosen = next((opt for opt in candidate_options if str(opt.get("label", "")).upper() == choice), None)
    if chosen is None:
        return None
    out = {
        "choice": choice,
        "triple": dict(chosen.get("triple", {}) or {}),
        "confidence": float(np.clip(float(parsed.get("confidence", 0.0) or 0.0), 0.0, 1.0)),
        "task_semantic": str(parsed.get("task_semantic", "") or "").strip()[:64],
        "attribute_confidence": {},
        "reason": str(parsed.get("reason", "") or "").strip()[:200],
    }
    raw_attr = parsed.get("attribute_confidence", {}) or {}
    if isinstance(raw_attr, dict):
        for key in ("grasp_type", "contact_geometry", "force_mode"):
            try:
                out["attribute_confidence"][key] = float(np.clip(float(raw_attr.get(key, 0.0) or 0.0), 0.0, 1.0))
            except Exception:
                pass
    return out


# ---------------------------------------------------------------------------
# Geometric grip and contact-point extraction
# ---------------------------------------------------------------------------

def _handle_skeleton_keypoints(
    mask: np.ndarray,
    axis_dir: Tuple[float, float],
    n: int = 7,
) -> List[Tuple[int, int]]:
    """Sample n skeleton pixels along the handle ordered tip→junction (axis direction)."""
    m = (mask > 0).astype(np.uint8)
    skel = _morph_skeleton(m)
    ys, xs = np.where(skel > 0)
    if xs.size == 0:
        ys, xs = np.where(m > 0)
    if xs.size == 0:
        return []
    proj = xs.astype(float) * axis_dir[0] + ys.astype(float) * axis_dir[1]
    order = np.argsort(proj)
    xs_s, ys_s = xs[order], ys[order]
    idx = np.linspace(0, len(xs_s) - 1, min(n, len(xs_s)), dtype=int)
    return [(int(xs_s[i]), int(ys_s[i])) for i in idx]


def _handle_pca_axis(
    h_mask: np.ndarray,
    fallback: Tuple[float, float],
) -> Tuple[float, float]:
    """Major PCA axis of the handle mask pixels, oriented toward the work-end.

    More accurate than handle_centroid→work_end_centroid because it captures the
    true elongation direction of the handle without the vertical offset error that
    arises when the head sits higher or lower than the grip.
    """
    m = (h_mask > 0)
    ys, xs = np.where(m)
    if xs.size < 4:
        return fallback
    pts = np.column_stack([xs.astype(float), ys.astype(float)])
    pts -= pts.mean(axis=0)
    _, _, vt = np.linalg.svd(pts, full_matrices=False)
    major = vt[0]          # first right singular vector = major axis direction
    dx, dy = float(major[0]), float(major[1])
    # Orient so it points in the same half-space as the fallback (handle→work-end)
    if dx * fallback[0] + dy * fallback[1] < 0:
        dx, dy = -dx, -dy
    n = float(np.sqrt(dx ** 2 + dy ** 2))
    return (dx / n, dy / n) if n > 1e-6 else fallback


def _axis_ray_exit(
    mask: np.ndarray,
    origin: Tuple[float, float],
    direction: Tuple[float, float],
) -> Optional[Tuple[int, int]]:
    """Walk a ray from *origin* along *direction* and return the last mask pixel hit.

    This finds where the tool axis exits the work-end mask on the distal side.
    Unlike extremal-boundary methods it is immune to off-axis appendages (e.g.
    a hammer claw) because the axis ray only passes through the on-axis body.
    Falls back to None if the ray never enters the mask.
    """
    h, w = mask.shape[:2]
    m = (mask > 0)
    ox, oy = float(origin[0]), float(origin[1])
    dx, dy = float(direction[0]), float(direction[1])
    step = 0.5
    t_max = float(np.sqrt(h ** 2 + w ** 2)) * 2.0
    last: Optional[Tuple[int, int]] = None
    t = 0.0
    while t <= t_max:
        px = int(round(ox + dx * t))
        py = int(round(oy + dy * t))
        if 0 <= px < w and 0 <= py < h:
            if m[py, px]:
                last = (px, py)
        elif last is not None:
            break  # left image after having been inside mask
        t += step
    return last


def _workend_fat_point(
    mask: np.ndarray,
    handle_center: Tuple[float, float],
    axis_dir: Tuple[float, float],
    contact_face: str = "primary",
    tool_morphology: str = "inline",
    face_direction: str = "lower",
) -> Optional[Tuple[int, int]]:
    """Functional Axis Terminus (FAT): one contact point on the work-end boundary.

    contact_face controls which face of the work-end is returned:
      "primary"   — axis-ray exit: last mask pixel along the tool axis ray.
      "secondary" — proximal cap centroid (opposite end from primary).
      "lateral"   — side face: centroid of the maximum-perp-projection boundary cap.

    For L_shape tools, face_direction ("upper"/"lower") picks which y-extreme cap
    of the head is the primary contact face.

    perp(a) = axis rotated 90° CCW = (−a_y, a_x).
    """
    m = (mask > 0).astype(np.uint8)
    hx, hy = handle_center
    perp = (-axis_dir[1], axis_dir[0])

    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        ys, xs = np.where(m > 0)
        if xs.size == 0:
            return None
        cxs, cys = xs.astype(float), ys.astype(float)
    else:
        pts = np.vstack([c.reshape(-1, 2) for c in cnts])
        cxs = pts[:, 0].astype(float)
        cys = pts[:, 1].astype(float)

    if cxs.size == 0:
        return None

    proj_axis = (cxs - hx) * axis_dir[0] + (cys - hy) * axis_dir[1]
    proj_perp = cxs * perp[0] + cys * perp[1]
    axis_offset = hx * perp[0] + hy * perp[1]
    offset_perp = np.abs(proj_perp - axis_offset)

    # L_shape tools always have the contact face perpendicular to the handle axis —
    # force lateral, then face_direction picks which of the two lateral faces.
    effective_face = "lateral" if tool_morphology == "L_shape" else contact_face

    cap_frac = 0.15
    if effective_face == "primary":
        pt = _axis_ray_exit(m, (hx, hy), axis_dir)
        if pt is not None:
            return pt
        a_min, a_max = float(proj_axis.min()), float(proj_axis.max())
        sel = proj_axis >= (a_min + (a_max - a_min) * 0.90)
        if not np.any(sel):
            sel = proj_axis >= float(proj_axis.max())
        off = offset_perp[sel]
        eps = max(1.0, float(off.max()) * 0.05)
        w = 1.0 / (off + eps)
        w /= w.sum()
        return (int(round(float((cxs[sel] * w).sum()))), int(round(float((cys[sel] * w).sum()))))
    elif effective_face == "secondary":
        a_min, a_max = float(proj_axis.min()), float(proj_axis.max())
        sel = proj_axis <= (a_min + (a_max - a_min) * 0.10)
        if not np.any(sel):
            sel = proj_axis <= float(proj_axis.min())
        return (int(round(float(cxs[sel].mean()))), int(round(float(cys[sel].mean()))))
    elif effective_face == "lateral":
        p_min, p_max = float(proj_perp.min()), float(proj_perp.max())
        p_range = max(p_max - p_min, 1e-6)
        sel_a = proj_perp <= (p_min + p_range * cap_frac)
        sel_b = proj_perp >= (p_min + p_range * (1.0 - cap_frac))
        if not np.any(sel_a):
            sel_a = proj_perp <= float(proj_perp.min())
        if not np.any(sel_b):
            sel_b = proj_perp >= float(proj_perp.max())
        if face_direction in ("upper", "lower"):
            # L_shape upper/lower: pick the perp cap that sits higher/lower in image
            cy_a = float(cys[sel_a].mean())
            cy_b = float(cys[sel_b].mean())
            if face_direction == "upper":
                sel = sel_a if cy_a <= cy_b else sel_b
            else:
                sel = sel_a if cy_a >= cy_b else sel_b
        else:
            # inline + lateral: furthest perp face (one side only)
            sel = sel_b
        if not np.any(sel):
            sel = np.ones(cxs.size, dtype=bool)
        return (int(round(float(cxs[sel].mean()))), int(round(float(cys[sel].mean()))))
    else:
        idx = int(np.argmax(proj_axis))
        return (int(cxs[idx]), int(cys[idx]))


# ---------------------------------------------------------------------------
# 3D back-projection helpers
# ---------------------------------------------------------------------------

def _derive_meta_json_path(rgb_path: str) -> Optional[str]:
    """Infer frame_XXXXXX_meta.json from a color_XXXXXX.png path."""
    p = Path(rgb_path)
    stem = p.stem
    if stem.startswith("color_"):
        candidate = p.parent / f"frame_{stem[len('color_'):]}_meta.json"
        if candidate.exists():
            return str(candidate)
    return None


def _load_camera_intrinsics(
    meta_json_path: str,
) -> Tuple[Optional[Tuple[float, float, float, float]], Optional[float]]:
    """Return ((fx, fy, ppx, ppy), depth_scale) from a frame meta JSON."""
    try:
        meta = json.loads(Path(meta_json_path).read_text(encoding="utf-8"))
        ci = meta["color_intrinsics"]
        fx, fy = float(ci["fx"]), float(ci["fy"])
        ppx, ppy = float(ci["ppx"]), float(ci["ppy"])
        ds_raw = meta.get("depth_scale")
        depth_scale = float(ds_raw) if ds_raw is not None and float(ds_raw) > 0 else None
        return (fx, fy, ppx, ppy), depth_scale
    except Exception:
        return None, None


def _pixel_to_3d(
    u: float, v: float,
    depth_map: np.ndarray,
    fx: float, fy: float, ppx: float, ppy: float,
    depth_scale: float,
    min_z: float = 0.05,
    max_z: float = 3.0,
) -> Optional[List[float]]:
    """Back-project pixel (u, v) to 3D camera coordinates [X, Y, Z] in metres."""
    ui, vi = int(round(u)), int(round(v))
    h, w = depth_map.shape[:2]
    if not (0 <= ui < w and 0 <= vi < h):
        return None
    d_raw = float(depth_map[vi, ui])
    if d_raw <= 0:
        return None
    Z = d_raw * depth_scale
    if not (min_z <= Z <= max_z):
        return None
    return [float((u - ppx) * Z / fx), float((v - ppy) * Z / fy), float(Z)]


def _mask_to_3d_cloud(
    mask: np.ndarray,
    depth_map: np.ndarray,
    offset_xy: Tuple[int, int],
    fx: float, fy: float, ppx: float, ppy: float,
    depth_scale: float,
    max_pts: int = 2000,
    min_z: float = 0.05,
    max_z: float = 3.0,
) -> Optional[np.ndarray]:
    """Back-project mask pixels (crop space) to a 3D point cloud Nx3 in metres."""
    ys, xs = np.where(mask > 0)
    if xs.size == 0:
        return None
    if xs.size > max_pts:
        idx = np.random.default_rng(0).choice(xs.size, max_pts, replace=False)
        xs, ys = xs[idx], ys[idx]
    x_off, y_off = offset_xy
    full_u = xs.astype(np.float64) + x_off
    full_v = ys.astype(np.float64) + y_off
    ui = np.round(full_u).astype(int)
    vi = np.round(full_v).astype(int)
    dh, dw = depth_map.shape[:2]
    valid = (ui >= 0) & (ui < dw) & (vi >= 0) & (vi < dh)
    ui, vi, full_u, full_v = ui[valid], vi[valid], full_u[valid], full_v[valid]
    d_raw = depth_map[vi, ui].astype(np.float64)
    Z = d_raw * depth_scale
    z_valid = (d_raw > 0) & (Z >= min_z) & (Z <= max_z)
    if not np.any(z_valid):
        return None
    Z = Z[z_valid]
    full_u, full_v = full_u[z_valid], full_v[z_valid]
    cloud = np.column_stack([(full_u - ppx) * Z / fx, (full_v - ppy) * Z / fy, Z])
    return cloud if len(cloud) >= 4 else None


def _pca_3d(cloud: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """SVD PCA on Nx3 cloud → (major, mid, minor) unit right singular vectors."""
    centered = cloud - cloud.mean(axis=0)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return vt[0], vt[1], vt[2]


def _save_3d_geometry_figure(
    h_cloud: Optional[np.ndarray],
    we_cloud: Optional[np.ndarray],
    grip_3d: Optional[List[float]],
    contact_3d: Optional[List[float]],
    axis_3d: Optional[List[float]],
    normal_3d: Optional[List[float]],
    out_path: str,
    *,
    arrow_len_frac: float = 0.35,
    dpi: int = 150,
) -> Optional[str]:
    """3D scatter figure: handle/work-end clouds + axis arrow + contact normal."""
    if h_cloud is None and we_cloud is None:
        return None
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

        fig = plt.figure(figsize=(8, 6), dpi=dpi)
        ax = fig.add_subplot(111, projection="3d")
        fig.patch.set_facecolor("white")

        all_pts: List[np.ndarray] = []
        if h_cloud is not None:
            ax.scatter(h_cloud[:, 0], h_cloud[:, 1], h_cloud[:, 2],
                       s=1, c="#2ecc71", alpha=0.4, label="Handle cloud")
            all_pts.append(h_cloud)
        if we_cloud is not None:
            ax.scatter(we_cloud[:, 0], we_cloud[:, 1], we_cloud[:, 2],
                       s=1, c="#3498db", alpha=0.4, label="Work-end cloud")
            all_pts.append(we_cloud)

        extent = float(np.max(np.ptp(np.vstack(all_pts), axis=0))) if all_pts else 0.05
        arrow_len = max(0.01, extent * arrow_len_frac)

        if grip_3d is not None:
            gx, gy, gz = grip_3d
            ax.scatter([gx], [gy], [gz], s=120, c="#27ae60", marker="o",
                       zorder=6, edgecolors="white", linewidths=0.8, label="G (grip)")
            if axis_3d is not None:
                da = [v * arrow_len for v in axis_3d]
                ax.quiver(gx, gy, gz, da[0], da[1], da[2],
                          color="#27ae60", linewidth=2.0,
                          arrow_length_ratio=0.18, label="Axis (3D)")

        if contact_3d is not None:
            cx, cy, cz = contact_3d
            ax.scatter([cx], [cy], [cz], s=120, c="#e74c3c", marker="^",
                       zorder=6, edgecolors="white", linewidths=0.8, label="C (contact)")
            if normal_3d is not None:
                dn = [v * arrow_len * 0.55 for v in normal_3d]
                ax.quiver(cx, cy, cz, dn[0], dn[1], dn[2],
                          color="#e74c3c", linewidth=2.0,
                          arrow_length_ratio=0.22, label="Normal (3D)")

        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.set_zlabel("Z (m)", fontsize=8)
        ax.set_title("3D contact geometry", fontsize=10, fontweight="600")
        ax.legend(fontsize=7, markerscale=3, loc="upper left")
        ax.tick_params(labelsize=7)
        fig.tight_layout()
        ensure_dir(os.path.dirname(out_path))
        fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return out_path
    except Exception:
        return None


def _project_3d_to_2d(
    pt_3d: List[float],
    fx: float, fy: float, ppx: float, ppy: float,
) -> Tuple[int, int]:
    X, Y, Z = pt_3d
    return (int(round(X / Z * fx + ppx)), int(round(Y / Z * fy + ppy)))


def _render_contact_overlay(
    full_bgr: np.ndarray,
    grip_3d: Optional[List[float]],
    contact_3d: Optional[List[float]],
    axis_3d: Optional[List[float]],
    normal_3d: Optional[List[float]],
    fx: float, fy: float, ppx: float, ppy: float,
    *,
    arrow_len_m: float = 0.025,
    h_cloud: Optional[np.ndarray] = None,
    we_cloud: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Contact frame overlay: point clouds + three arrows (axis/normal/tangent) from C, glow at C, dot at G."""
    vis = full_bgr.copy()
    h, w = vis.shape[:2]

    # --- Draw point clouds projected onto 2D image ---
    _MAX_PTS = 1500
    _RNG = np.random.default_rng(42)

    def _draw_cloud(cloud: np.ndarray, color: tuple) -> None:
        pts = cloud
        if len(pts) > _MAX_PTS:
            pts = pts[_RNG.choice(len(pts), _MAX_PTS, replace=False)]
        X, Y, Z = pts[:, 0], pts[:, 1], pts[:, 2]
        us = np.round(X / Z * fx + ppx).astype(int)
        vs = np.round(Y / Z * fy + ppy).astype(int)
        valid = (us >= 0) & (us < w) & (vs >= 0) & (vs < h)
        for u_pt, v_pt in zip(us[valid], vs[valid]):
            cv2.circle(vis, (int(u_pt), int(v_pt)), 1, color, -1)

    if h_cloud is not None and len(h_cloud) > 0:
        _draw_cloud(h_cloud, (200, 220, 60))   # handle: pale green-cyan

    if we_cloud is not None and len(we_cloud) > 0:
        _draw_cloud(we_cloud, (40, 140, 255))  # work-end: orange

    if contact_3d is None:
        return vis

    cp = _project_3d_to_2d(contact_3d, fx, fy, ppx, ppy)

    # Compute tangent = axis × normal (third axis of the contact frame)
    tangent_3d: Optional[List[float]] = None
    if axis_3d is not None and normal_3d is not None:
        t = np.cross(np.array(axis_3d), np.array(normal_3d))
        n = float(np.linalg.norm(t))
        if n > 1e-6:
            tangent_3d = (t / n).tolist()

    # Three contact-frame arrows from C  (BGR colors matching reference image)
    frame_axes = [
        (axis_3d,    (220, 80,  50),  "axis"),    # blue
        (normal_3d,  (50,  50,  220), "normal"),  # red
        (tangent_3d, (50,  220, 220), "tangent"), # yellow
    ]
    for dir_3d, color, _ in frame_axes:
        if dir_3d is None:
            continue
        tip_3d = [contact_3d[i] + dir_3d[i] * arrow_len_m for i in range(3)]
        tp = _project_3d_to_2d(tip_3d, fx, fy, ppx, ppy)
        cv2.arrowedLine(vis, cp, tp, color, 4, cv2.LINE_AA, tipLength=0.3)

    # G and C — same callout style as _make_combined_pgl
    if grip_3d is not None:
        gp = _project_3d_to_2d(grip_3d, fx, fy, ppx, ppy)
        _draw_point_callout(vis, gp, "G", (30, 30, 220), direction=(0.7, -0.7), marker_type=cv2.MARKER_CROSS)

    if contact_3d is not None:
        _draw_point_callout(vis, cp, "C", (200, 50, 0), direction=(0.7, 0.7), marker_type=cv2.MARKER_CROSS)

    return vis


def _draw_coord_panel(
    img: np.ndarray,
    grip_3d: Optional[List[float]],
    contact_3d: Optional[List[float]],
    axis_3d: Optional[List[float]],
    normal_3d: Optional[List[float]],
) -> None:
    """Overlay an XYZ readout panel in the top-left corner (in-place)."""
    def _fmt(v: Optional[List[float]], unit: str = " m") -> str:
        if v is None:
            return "n/a"
        return f"X:{v[0]:+.3f}  Y:{v[1]:+.3f}  Z:{v[2]:+.3f}{unit}"

    rows = [
        ("G grip   ", _fmt(grip_3d),    (50, 230, 50)),
        ("C contact", _fmt(contact_3d), (80, 100, 240)),
        ("axis     ", _fmt(axis_3d, ""), (200, 180, 60)),
        ("normal   ", _fmt(normal_3d, ""), (60, 60, 220)),
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    fs, th = 0.55, 2
    lh, pad = 24, 10

    widths = [cv2.getTextSize(f"{lbl}: {val}", font, fs, th)[0][0] for lbl, val, _ in rows]
    box_w = max(widths) + 2 * pad + 6
    box_h = len(rows) * lh + 2 * pad + 4
    x0, y0 = 6, 6

    panel = img[y0:y0 + box_h, x0:x0 + box_w].copy()
    dark = np.full_like(panel, (15, 15, 15))
    blended = cv2.addWeighted(dark, 0.80, panel, 0.20, 0)
    img[y0:y0 + box_h, x0:x0 + box_w] = blended
    cv2.rectangle(img, (x0, y0), (x0 + box_w, y0 + box_h), (80, 80, 80), 1)

    for i, (lbl, val, col) in enumerate(rows):
        ty = y0 + pad + (i + 1) * lh - 2
        cv2.putText(img, f"{lbl}:", (x0 + pad, ty), font, fs, (200, 200, 200), th, cv2.LINE_AA)
        lw = cv2.getTextSize(f"{lbl}:", font, fs, th)[0][0]
        cv2.putText(img, val, (x0 + pad + lw + 4, ty), font, fs, col, th, cv2.LINE_AA)


def _draw_keypoints_numbered(
    canvas: np.ndarray,
    keypoints: List[Tuple[int, int]],
    color: Tuple[int, int, int],
    sx: float = 1.0,
    sy: float = 1.0,
) -> np.ndarray:
    """Draw numbered circle badges at each keypoint (scaled to display space)."""
    out = canvas.copy()
    for kid, (kx, ky) in enumerate(keypoints):
        px = int(round(kx * sx))
        py = int(round(ky * sy))
        cv2.circle(out, (px, py), 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(out, (px, py), 9, color, 2, cv2.LINE_AA)
        lbl = str(kid)
        (lw, lh), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.putText(out, lbl, (px - lw // 2, py + lh // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out




# ---------------------------------------------------------------------------
# Structured affordance helper functions (journal contribution)
# ---------------------------------------------------------------------------

def _infer_grasp_type_from_geometry(
    h_mask: np.ndarray,
    mask_stats: List[Dict[str, Any]],
    selected_id: int,
    instruction: str,
    tool_morphology: str = "inline",
) -> Tuple[str, float]:
    """Predict grasp type (power/precision/pinch/hook) from handle shape geometry.

    Decision rules derived from grasp taxonomy (Feix et al. 2016):
      hook      — L_shape morphology requires hook/overhand grip
      pinch     — very small handle area or width < 6 px
      precision — moderate elongation and narrow width
      power     — elongated shaft with sufficient cross-section
    """
    if tool_morphology == "L_shape":
        return "hook", 0.78

    stat = next((s for s in mask_stats if int(s.get("id", -1)) == selected_id), {})
    elong = float(stat.get("elongation", 2.0) or 2.0)
    mean_w = float(stat.get("mean_half_width_px", 8.0) or 8.0) * 2.0
    area_px = int(mask_area((h_mask > 0).astype(np.uint8)))

    if area_px < 180 or mean_w < 6.0:
        return "pinch", 0.72
    if elong >= 4.5 and mean_w >= 14.0:
        return "power", 0.84
    if elong >= 2.5 and mean_w >= 9.0:
        return "precision", 0.76
    return "power", 0.62


def _infer_contact_geometry_from_mask(
    we_mask: np.ndarray,
    tool_morphology: str = "inline",
    contact_face: str = "primary",
) -> Tuple[str, float]:
    """Predict contact geometry (point/line/surface) from work-end mask shape.

    Decision rules based on bounding-box aspect ratio and mask area:
      point   — compact, small area (screwdriver tip, punch, needle)
      line    — elongated aspect ratio (scraper blade, knife edge)
      surface — broad, low-aspect-ratio region (spatula face, roller)
    """
    m = (we_mask > 0).astype(np.uint8)
    area_px = int(mask_area(m))
    if area_px == 0:
        return "point", 0.50

    ys, xs = np.where(m > 0)
    bb_w = float(xs.max() - xs.min() + 1)
    bb_h = float(ys.max() - ys.min() + 1)
    aspect = max(bb_w, bb_h) / max(min(bb_w, bb_h), 1.0)

    if area_px < 160:
        return "point", 0.76
    if aspect >= 4.0 or (contact_face in {"primary", "lateral"} and aspect >= 2.5):
        return "line", 0.80
    if area_px >= 500 and aspect < 3.0:
        return "surface", 0.74
    if aspect >= 2.5:
        return "line", 0.66
    return "surface", 0.61


def _infer_force_mode_from_instruction(
    instruction: str,
    tool_morphology: str = "inline",
    contact_geometry: str = "surface",
) -> Tuple[str, float]:
    """Predict dominant wrench mode from action verb and tool geometry.

    Returns (force_mode, confidence).  Verb matching takes priority over
    geometric fallback so task-conditioned grounding drives the prediction.
    """
    ins_l = (instruction or "").lower()

    _torque = {"screw", "unscrew", "tighten", "loosen", "turn", "twist", "rotate", "drill", "drive"}
    _push   = {"hammer", "strike", "punch", "press", "push", "insert", "hit", "poke", "nail"}
    _pull   = {"scrape", "pull", "drag", "rake", "pry", "lever", "scape"}
    _normal = {"paint", "brush", "roll", "spread", "coat", "apply", "wipe", "dab"}

    for hint in _torque:
        if hint in ins_l:
            return "torque", 0.87
    for hint in _push:
        if hint in ins_l:
            return "push", 0.84
    for hint in _pull:
        if hint in ins_l:
            return "pull", 0.82
    for hint in _normal:
        if hint in ins_l:
            return "normal", 0.80

    if tool_morphology == "L_shape":
        return "push", 0.62
    if contact_geometry == "surface":
        return "normal", 0.57
    return "push", 0.50


def _build_task_semantic_label(instruction: str, functional_role: str) -> str:
    """Compose a verb-object semantic label, e.g. 'scrape_blade', 'hammer_handle'."""
    ins_l = (instruction or "").lower().strip()
    verb = ins_l.split()[0] if ins_l else "use"
    verb = "".join(c for c in verb if c.isalnum() or c == "_")[:16]
    role_suffix = {"handle": "grip", "work_end": "contact"}.get(functional_role, functional_role)
    return f"{verb}_{role_suffix}"


def _compute_wrench_basis(
    axis_3d: Optional[List[float]],
    normal_3d: Optional[List[float]],
) -> Optional[Dict[str, List[float]]]:
    """Compute the full SO(3) contact wrench basis {axis, normal, tangent}.

    The three unit vectors form an orthonormal frame at the contact point C:
      axis    — tool long axis (handle→work-end), from handle PCA
      normal  — contact surface outward normal, from work-end PCA minor axis
      tangent — tangential direction = axis × normal (lateral wrench component)

    This frame is the mechanistic bridge between affordance grounding and the
    downstream wrench-bounded constraint synthesis stage.
    """
    if axis_3d is None or normal_3d is None:
        return None
    a = np.array(axis_3d, dtype=np.float64)
    n = np.array(normal_3d, dtype=np.float64)
    an, nn = float(np.linalg.norm(a)), float(np.linalg.norm(n))
    if an < 1e-8 or nn < 1e-8:
        return None
    a /= an
    n /= nn

    tangent = np.cross(a, n)
    t_norm = float(np.linalg.norm(tangent))
    if t_norm < 1e-6:
        fb = np.array([1.0, 0.0, 0.0]) if abs(float(a[0])) < 0.9 else np.array([0.0, 1.0, 0.0])
        tangent = np.cross(a, fb)
        tangent /= max(float(np.linalg.norm(tangent)), 1e-8)
    else:
        tangent /= t_norm

    # Re-orthogonalize normal to ensure strict SO(3) basis
    n = np.cross(tangent, a)
    n /= max(float(np.linalg.norm(n)), 1e-8)

    return {"axis": a.tolist(), "normal": n.tolist(), "tangent": tangent.tolist()}


def _rotation_matrix_to_quaternion_xyzw(R: np.ndarray) -> List[float]:
    """Convert a 3x3 rotation matrix to a normalized quaternion [x, y, z, w]."""
    m = np.asarray(R, dtype=np.float64)
    trace = float(np.trace(m))
    if trace > 0.0:
        s = 2.0 * np.sqrt(trace + 1.0)
        qw = 0.25 * s
        qx = (m[2, 1] - m[1, 2]) / s
        qy = (m[0, 2] - m[2, 0]) / s
        qz = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2]))
        qw = (m[2, 1] - m[1, 2]) / s
        qx = 0.25 * s
        qy = (m[0, 1] + m[1, 0]) / s
        qz = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2]))
        qw = (m[0, 2] - m[2, 0]) / s
        qx = (m[0, 1] + m[1, 0]) / s
        qy = 0.25 * s
        qz = (m[1, 2] + m[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1]))
        qw = (m[1, 0] - m[0, 1]) / s
        qx = (m[0, 2] + m[2, 0]) / s
        qy = (m[1, 2] + m[2, 1]) / s
        qz = 0.25 * s
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q /= max(float(np.linalg.norm(q)), 1e-12)
    return q.tolist()


def _compute_grasp_pose_from_wrench_basis(
    grip_point_3d: Optional[List[float]],
    wrench_basis: Optional[Dict[str, List[float]]],
) -> Dict[str, Optional[List[Any]]]:
    """Build a robot-consumable grasp SE(3) pose from the grounded affordance frame.

    Convention:
      x = gripper closing axis  (tool tangent)
      y = handle/tool axis      (handle -> work-end)
      z = approach axis         (-contact normal, approaching into the handle)
    """
    if grip_point_3d is None or wrench_basis is None:
        return {
            "grasp_rotation_cam": None,
            "grasp_frame_cam": None,
            "grasp_quaternion_xyzw": None,
            "grasp_approach_axis_3d": None,
            "grasp_closing_axis_3d": None,
        }

    axis = np.array(wrench_basis.get("axis", []), dtype=np.float64)
    normal = np.array(wrench_basis.get("normal", []), dtype=np.float64)
    tangent = np.array(wrench_basis.get("tangent", []), dtype=np.float64)
    if axis.shape != (3,) or normal.shape != (3,) or tangent.shape != (3,):
        return {
            "grasp_rotation_cam": None,
            "grasp_frame_cam": None,
            "grasp_quaternion_xyzw": None,
            "grasp_approach_axis_3d": None,
            "grasp_closing_axis_3d": None,
        }

    y_axis = axis / max(float(np.linalg.norm(axis)), 1e-8)
    x_axis = tangent / max(float(np.linalg.norm(tangent)), 1e-8)
    z_axis = (-normal) / max(float(np.linalg.norm(normal)), 1e-8)

    # Re-orthogonalize to ensure a strict right-handed SO(3) frame.
    x_axis = x_axis - float(np.dot(x_axis, y_axis)) * y_axis
    x_axis /= max(float(np.linalg.norm(x_axis)), 1e-8)
    z_axis = np.cross(x_axis, y_axis)
    z_axis /= max(float(np.linalg.norm(z_axis)), 1e-8)
    if float(np.dot(z_axis, -normal)) < 0.0:
        x_axis = -x_axis
        z_axis = -z_axis

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = np.array(grip_point_3d, dtype=np.float64)

    return {
        "grasp_rotation_cam": R.tolist(),
        "grasp_frame_cam": T.tolist(),
        "grasp_quaternion_xyzw": _rotation_matrix_to_quaternion_xyzw(R),
        "grasp_approach_axis_3d": z_axis.tolist(),
        "grasp_closing_axis_3d": x_axis.tolist(),
    }


def _geometry_vlm_agreement_score(
    vlm_handle_id: Optional[int],
    vlm_work_end_id: Optional[int],
    geom_handle_id: Optional[int],
    geom_work_end_id: Optional[int],
    vlm_confidence: float,
    *,
    instruction: str = "",
    geom_handle_mask: Optional[np.ndarray] = None,
    geom_work_end_mask: Optional[np.ndarray] = None,
    mask_stats: Optional[List[Dict[str, Any]]] = None,
    tool_morphology: str = "inline",
    contact_face: str = "primary",
    vlm_grasp_type: str = "",
    vlm_contact_geometry: str = "",
    vlm_force_mode: str = "",
    vlm_attribute_confidence: Optional[Dict[str, float]] = None,
    return_details: bool = False,
) -> Any:
    """Compute geometry-VLM agreement using categorical JSD over affordance labels.

    The agreement score is the mean of all comparable components:
      - role agreement for handle/work-end IDs
      - 1 - JSD(grasp_type)
      - 1 - JSD(contact_geometry)
      - 1 - JSD(force_mode)

    Scalar confidences are mapped to categorical distributions via a uniform-to-
    onehot interpolation: confidence 0 => uniform, confidence 1 => delta.
    """

    def _safe_conf(value: Any, fallback: float) -> float:
        try:
            return float(np.clip(float(value if value is not None else fallback), 0.0, 1.0))
        except Exception:
            return float(np.clip(float(fallback), 0.0, 1.0))

    def _dist_from_label(
        labels: List[str],
        predicted_label: str,
        confidence: float,
    ) -> Optional[Dict[str, float]]:
        if not labels:
            return None
        pred = str(predicted_label or "").strip().lower()
        if pred not in labels:
            uniform = 1.0 / float(len(labels))
            return {label: uniform for label in labels}
        n = float(len(labels))
        conf = _safe_conf(confidence, 0.0)
        uniform = 1.0 / n
        out: Dict[str, float] = {}
        for label in labels:
            base = uniform
            if label == pred:
                out[label] = float(base + conf * (1.0 - base))
            else:
                out[label] = float(base * (1.0 - conf))
        total = float(sum(out.values()))
        if total <= 1e-12:
            return {label: uniform for label in labels}
        return {label: float(out[label] / total) for label in labels}

    def _entropy_bits(dist: Dict[str, float], labels: List[str]) -> float:
        h = 0.0
        for label in labels:
            p = float(dist.get(label, 0.0))
            if p > 1e-12:
                h -= p * float(np.log2(p))
        return float(h)

    def _jsd_bits(
        p: Dict[str, float],
        q: Dict[str, float],
        labels: List[str],
    ) -> float:
        m = {label: 0.5 * (float(p.get(label, 0.0)) + float(q.get(label, 0.0))) for label in labels}
        jsd = _entropy_bits(m, labels) - 0.5 * _entropy_bits(p, labels) - 0.5 * _entropy_bits(q, labels)
        return float(np.clip(jsd, 0.0, 1.0))

    def _attr_bundle(
        predicted_label: str,
        confidence: float,
        labels: List[str],
    ) -> Optional[Dict[str, Any]]:
        dist = _dist_from_label(labels, predicted_label, confidence)
        if dist is None:
            return None
        return {
            "predicted": str(predicted_label or "").strip().lower(),
            "confidence": float(_safe_conf(confidence, 0.0)),
            "distribution": dist,
        }

    details: Dict[str, Any] = {
        "method": "jsd_categorical_agreement",
        "role_agreement": None,
        "role_components": {
            "handle_match": None if geom_handle_id is None else float(vlm_handle_id == geom_handle_id) if vlm_handle_id is not None else 0.5,
            "work_end_match": None if geom_work_end_id is None else float(vlm_work_end_id == geom_work_end_id) if vlm_work_end_id is not None else 0.5,
        },
        "attribute_jsd_bits": {},
        "attribute_agreement": {},
        "geometry_distributions": {},
        "vlm_distributions": {},
        "comparable_components": [],
        "fallback_used": False,
    }

    component_scores: List[float] = []
    if geom_handle_id is not None and geom_work_end_id is not None:
        h_agree = float(vlm_handle_id == geom_handle_id) if vlm_handle_id is not None else 0.5
        w_agree = float(vlm_work_end_id == geom_work_end_id) if vlm_work_end_id is not None else 0.5
        role_agreement = 0.5 * h_agree + 0.5 * w_agree
        details["role_agreement"] = float(role_agreement)
        component_scores.append(float(role_agreement))
        details["comparable_components"].append("role")

    geom_mask_stats = mask_stats or []
    geom_grasp_label: Optional[str] = None
    geom_grasp_conf: Optional[float] = None
    if geom_handle_mask is not None and geom_handle_id is not None:
        _label, _conf = _infer_grasp_type_from_geometry(
            geom_handle_mask,
            geom_mask_stats,
            geom_handle_id,
            instruction,
            tool_morphology,
        )
        geom_grasp_label, geom_grasp_conf = _label, float(_conf)

    geom_contact_label: Optional[str] = None
    geom_contact_conf: Optional[float] = None
    if geom_work_end_mask is not None:
        _label, _conf = _infer_contact_geometry_from_mask(
            geom_work_end_mask,
            tool_morphology,
            contact_face,
        )
        geom_contact_label, geom_contact_conf = _label, float(_conf)

    geom_force_label, geom_force_conf = _infer_force_mode_from_instruction(
        instruction,
        tool_morphology,
        geom_contact_label or "surface",
    )

    vlm_attr_conf = dict(vlm_attribute_confidence or {})
    attr_specs = [
        (
            "grasp_type",
            _AGREEMENT_GRASP_LABELS,
            geom_grasp_label,
            geom_grasp_conf,
            vlm_grasp_type,
            vlm_attr_conf.get("grasp_type", vlm_confidence),
        ),
        (
            "contact_geometry",
            _AGREEMENT_CONTACT_LABELS,
            geom_contact_label,
            geom_contact_conf,
            vlm_contact_geometry,
            vlm_attr_conf.get("contact_geometry", vlm_confidence),
        ),
        (
            "force_mode",
            _AGREEMENT_FORCE_LABELS,
            geom_force_label,
            geom_force_conf,
            vlm_force_mode,
            vlm_attr_conf.get("force_mode", vlm_confidence),
        ),
    ]

    for attr_name, labels, geom_label, geom_conf, vlm_label, vlm_attr_score in attr_specs:
        if geom_label is None or geom_conf is None:
            continue
        geom_bundle = _attr_bundle(geom_label, geom_conf, labels)
        vlm_bundle = _attr_bundle(vlm_label, vlm_attr_score, labels)
        if geom_bundle is None or vlm_bundle is None:
            continue
        details["geometry_distributions"][attr_name] = geom_bundle
        details["vlm_distributions"][attr_name] = vlm_bundle
        jsd = _jsd_bits(geom_bundle["distribution"], vlm_bundle["distribution"], labels)
        agreement = float(np.clip(1.0 - jsd, 0.0, 1.0))
        details["attribute_jsd_bits"][attr_name] = float(jsd)
        details["attribute_agreement"][attr_name] = agreement
        component_scores.append(agreement)
        details["comparable_components"].append(attr_name)

    if component_scores:
        score = float(np.clip(float(np.mean(component_scores)), 0.0, 1.0))
    else:
        score = float(np.clip(vlm_confidence, 0.0, 1.0))
        details["fallback_used"] = True

    details["score"] = score
    details["component_count"] = len(component_scores)
    if return_details:
        return score, details
    return score


def _compute_calibrated_uncertainty(
    vlm_confidence: float,
    geometry_vlm_agreement: float,
    attribute_confidences: Dict[str, float],
) -> Dict[str, float]:
    """Compute per-attribute calibrated uncertainty ∈ [0, 1].

    0 = fully certain, 1 = maximally uncertain.
    Reliability is the product of VLM confidence and geometry-VLM agreement, so
    high JSD disagreement automatically widens the uncertainty bands.
    """
    reliability = float(np.clip(vlm_confidence, 0.0, 1.0) * np.clip(geometry_vlm_agreement, 0.0, 1.0))
    aggregate_u = float(np.clip(1.0 - reliability, 0.0, 1.0))

    per_attr: Dict[str, float] = {}
    for attr, conf in attribute_confidences.items():
        attr_rel = float(np.clip(float(conf), 0.0, 1.0) * np.clip(geometry_vlm_agreement, 0.0, 1.0))
        per_attr[attr] = float(np.clip(1.0 - attr_rel, 0.0, 1.0))
    per_attr["aggregate"] = aggregate_u
    return per_attr


def _compute_grasp_width_from_skeleton(
    h_mask: np.ndarray,
    axis_dir: Tuple[float, float],
) -> Optional[float]:
    """Estimate the grasp width (pixels) as the median perpendicular extent of the handle.

    Computed by binning mask pixels along the tool axis and measuring the
    perpendicular span in each bin, then taking the median of the middle 60%
    of bins (excluding tip and junction artefacts).  Used as a proxy for
    gripper aperture requirements in downstream grasp planning.
    """
    m = (h_mask > 0).astype(np.uint8)
    ys, xs = np.where(m > 0)
    if xs.size < 12:
        return None
    ax, ay = axis_dir
    perp = (-ay, ax)
    proj_axis = xs.astype(float) * ax + ys.astype(float) * ay
    proj_perp = xs.astype(float) * perp[0] + ys.astype(float) * perp[1]

    n_bins = max(3, min(12, int(xs.size // 8)))
    a_min, a_max = float(proj_axis.min()), float(proj_axis.max())
    if a_max - a_min < 1.0:
        return float(proj_perp.max() - proj_perp.min())

    edges = np.linspace(a_min, a_max, n_bins + 1)
    widths: List[float] = []
    for i in range(n_bins):
        sel = (proj_axis >= edges[i]) & (proj_axis < edges[i + 1])
        if np.any(sel):
            pp = proj_perp[sel]
            widths.append(float(pp.max() - pp.min()))

    if not widths:
        return None
    widths.sort()
    lo = int(len(widths) * 0.20)
    hi = max(lo + 1, int(len(widths) * 0.80))
    return float(np.median(widths[lo:hi]))


def _compute_grasp_width_from_cloud_3d(
    handle_cloud_3d: Optional[np.ndarray],
    axis_direction_3d: Optional[List[float]],
    n_bins: int = 12,
) -> Optional[float]:
    """Estimate physical grasp width in meters from the 3D handle cloud.

    The handle cloud is projected into a local frame whose primary axis is the
    handle direction. We then measure robust cross-sectional span in the plane
    orthogonal to the handle axis by binning along the axis and taking the
    median span of the middle bins, mirroring the 2D pixel heuristic but in
    metric 3D space.
    """
    if handle_cloud_3d is None or axis_direction_3d is None:
        return None
    pts = np.asarray(handle_cloud_3d, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 12:
        return None
    axis = np.asarray(axis_direction_3d, dtype=np.float64)
    if axis.shape != (3,):
        return None
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm < 1e-8:
        return None
    axis /= axis_norm

    centroid = np.mean(pts, axis=0)
    centered = pts - centroid[None, :]
    proj_axis = centered @ axis

    # Build an orthonormal basis for the cross-section plane.
    ref = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(ref, axis))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    u = ref - float(np.dot(ref, axis)) * axis
    u_norm = float(np.linalg.norm(u))
    if u_norm < 1e-8:
        return None
    u /= u_norm
    v = np.cross(axis, u)
    v_norm = float(np.linalg.norm(v))
    if v_norm < 1e-8:
        return None
    v /= v_norm

    proj_u = centered @ u
    proj_v = centered @ v
    a_min, a_max = float(np.min(proj_axis)), float(np.max(proj_axis))
    if a_max - a_min < 1e-6:
        radial = np.sqrt(proj_u ** 2 + proj_v ** 2)
        if radial.size == 0:
            return None
        return float(2.0 * np.percentile(radial, 90.0))

    n_bins = max(3, min(int(n_bins), max(3, int(pts.shape[0] // 20))))
    edges = np.linspace(a_min, a_max, n_bins + 1)
    widths: List[float] = []
    for i in range(n_bins):
        if i == n_bins - 1:
            sel = (proj_axis >= edges[i]) & (proj_axis <= edges[i + 1])
        else:
            sel = (proj_axis >= edges[i]) & (proj_axis < edges[i + 1])
        if int(np.count_nonzero(sel)) < 4:
            continue
        slice_u = proj_u[sel]
        slice_v = proj_v[sel]
        radial = np.sqrt(slice_u ** 2 + slice_v ** 2)
        if radial.size == 0:
            continue
        widths.append(float(2.0 * np.percentile(radial, 90.0)))

    if not widths:
        radial = np.sqrt(proj_u ** 2 + proj_v ** 2)
        if radial.size == 0:
            return None
        return float(2.0 * np.percentile(radial, 90.0))

    widths.sort()
    lo = int(len(widths) * 0.20)
    hi = max(lo + 1, int(len(widths) * 0.80))
    return float(np.median(widths[lo:hi]))


def _build_region_affordance_tuple(
    functional_role: str,
    instruction: str,
    h_mask: Optional[np.ndarray],
    we_mask: Optional[np.ndarray],
    mask_stats: List[Dict[str, Any]],
    selected_id: Optional[int],
    tool_morphology: str,
    contact_face: str,
    vlm_parsed: Dict[str, Any],
    geometry_vlm_agreement: float,
) -> RegionAffordanceTuple:
    """Construct a RegionAffordanceTuple by fusing VLM predictions and geometry priors.

    Fusion strategy:
      - Use VLM attribute if the predicted value is valid (in the permitted set).
      - Fall back to geometry inference when VLM output is absent or invalid.
      - Aggregate confidence = mean(attr_confs) * sqrt(agreement) to down-weight
        predictions under high geometry–VLM disagreement.
    """
    vlm_grasp = str(vlm_parsed.get("grasp_type", "") or "").strip().lower()
    vlm_cg    = str(vlm_parsed.get("contact_geometry", "") or "").strip().lower()
    vlm_fm    = str(vlm_parsed.get("force_mode", "") or "").strip().lower()
    vlm_attr_conf: Dict[str, float] = {}
    raw_attr = vlm_parsed.get("attribute_confidence", {}) or {}
    if isinstance(raw_attr, dict):
        for k, v in raw_attr.items():
            try:
                vlm_attr_conf[str(k)] = float(np.clip(float(v or 0.0), 0.0, 1.0))
            except Exception:
                pass

    geom_grasp, geom_grasp_conf = "power", 0.50
    geom_cg, geom_cg_conf = "surface", 0.50

    if functional_role == "handle" and h_mask is not None and selected_id is not None:
        geom_grasp, geom_grasp_conf = _infer_grasp_type_from_geometry(
            h_mask, mask_stats, selected_id, instruction, tool_morphology)
    if functional_role == "work_end" and we_mask is not None:
        geom_cg, geom_cg_conf = _infer_contact_geometry_from_mask(
            we_mask, tool_morphology, contact_face)

    vlm_conf = float(vlm_parsed.get("confidence", 0.6) or 0.6)

    grasp_type = vlm_grasp if vlm_grasp in _VALID_GRASP_TYPES else geom_grasp
    grasp_conf = vlm_attr_conf.get("grasp_type", geom_grasp_conf if vlm_grasp not in _VALID_GRASP_TYPES else vlm_conf)

    contact_geom = vlm_cg if vlm_cg in _VALID_CONTACT_GEOMS else geom_cg
    contact_conf = vlm_attr_conf.get("contact_geometry", geom_cg_conf if vlm_cg not in _VALID_CONTACT_GEOMS else vlm_conf)

    force_mode, force_conf = _infer_force_mode_from_instruction(instruction, tool_morphology, contact_geom)
    if vlm_fm in _VALID_FORCE_MODES:
        force_mode = vlm_fm
        force_conf = vlm_attr_conf.get("force_mode", vlm_conf)

    task_sem = str(vlm_parsed.get("task_semantic", "") or "").strip() or _build_task_semantic_label(instruction, functional_role)

    attr_conf_dict: Dict[str, float] = {
        "grasp_type":       float(np.clip(float(grasp_conf),   0.0, 1.0)),
        "contact_geometry": float(np.clip(float(contact_conf), 0.0, 1.0)),
        "force_mode":       float(np.clip(force_conf,          0.0, 1.0)),
    }
    agg_conf = float(np.mean(list(attr_conf_dict.values())) * float(geometry_vlm_agreement) ** 0.5)

    return RegionAffordanceTuple(
        functional_role=functional_role,
        grasp_type=grasp_type if functional_role == "handle" else "none",
        contact_geometry=contact_geom,
        force_mode=force_mode,
        task_semantic=task_sem,
        confidence=float(np.clip(agg_conf, 0.0, 1.0)),
        attribute_confidence=attr_conf_dict,
    )


# ---------------------------------------------------------------------------
# Main policy
# ---------------------------------------------------------------------------

def run_handle_region_policy(
    *,
    object_name: str,
    instruction: str,
    masks: List[np.ndarray],
    full_bgr: np.ndarray,
    depth: Optional[np.ndarray],
    out_dir: str,
    require_api: bool = False,
    client: Optional[Any] = None,
    face_direction_override: Optional[str] = None,
    camera_intrinsics: Optional[Tuple[float, float, float, float]] = None,
    depth_scale: float = 0.001,
    target_crop_bgr: Optional[np.ndarray] = None,
) -> HandleDecision:
    ensure_dir(out_dir)

    crop_bgr, depth_crop = crop_from_masks(full_bgr=full_bgr, masks=masks, depth=depth)
    crop_masks, crop_bb = _crop_binary_masks(masks, pad=12)

    # ------------------------------------------------------------------ #
    # Save VLM-1 input images                                             #
    # ------------------------------------------------------------------ #
    grid_img = _make_mask_grid(crop_bgr, crop_masks)
    skeleton_overlay = _draw_skeleton_overlay(crop_bgr, crop_masks)
    rgb_path = str(Path(out_dir) / "rgb_crop.png")
    grid_path = str(Path(out_dir) / "candidate_grid.png")
    skeleton_path = str(Path(out_dir) / "skeleton_overlay.png")
    cv2.imwrite(grid_path, grid_img)

    depth_vis = colorize_depth(depth_crop)
    depth_vis_path = None
    if depth_vis is not None:
        depth_vis_path = str(Path(out_dir) / "depth_colormap.png")
        cv2.imwrite(depth_vis_path, depth_vis)

    from regiomanip.region_grounding.utils import build_mask_geometry_stats  # local import to avoid circular

    mask_stats_aff = build_mask_geometry_stats(mask_candidates=crop_masks, depth=depth_crop)
    empty_ids: Dict[str, List[int]] = {"whole_ids": [], "handle_ids": [], "work_end_ids": [], "contact_region_ids": []}
    geom_ids, geom_debug = _repair_tool_role_selection(
        empty_ids,
        object_name=object_name,
        instruction=instruction,
        tool_mode=True,
        mask_stats=mask_stats_aff,
    )
    geom_handle_id: Optional[int] = int(geom_ids["handle_ids"][0]) if geom_ids.get("handle_ids") else None
    geom_work_end_id: Optional[int] = int(geom_ids["work_end_ids"][0]) if geom_ids.get("work_end_ids") else None
    geom_handle_mask: Optional[np.ndarray] = (
        (crop_masks[geom_handle_id] > 0).astype(np.uint8).copy()
        if geom_handle_id is not None and 0 <= geom_handle_id < len(crop_masks)
        else None
    )
    geom_work_end_mask: Optional[np.ndarray] = (
        (crop_masks[geom_work_end_id] > 0).astype(np.uint8).copy()
        if geom_work_end_id is not None and 0 <= geom_work_end_id < len(crop_masks)
        else None
    )

    # ------------------------------------------------------------------ #
    # VLM client                                                          #
    # ------------------------------------------------------------------ #
    working_client = client
    client_status = None
    if working_client is None:
        working_client, client_status = build_openai_client_with_status()
    if require_api and working_client is None:
        raise RuntimeError(f"handle_region_api_unavailable: {client_status}")

    vlm_info: Dict[str, Any] = {
        "client_status": client_status,
        "vlm1_raw": None,
    }

    # ------------------------------------------------------------------ #
    # VLM-1: semantic part selection — handle_id, work_end_id, contact_face#
    # ------------------------------------------------------------------ #
    selected_id: Optional[int] = None
    we_selected_id: Optional[int] = None
    vlm1_confidence = 0.0
    contact_face: str = "primary"
    face_direction: str = "lower"
    target_region: str = "middle"
    tool_morphology: str = "inline"
    # Structured affordance attribute defaults (populated from VLM response below)
    vlm1_parsed_extended: Dict[str, Any] = {}
    _vlm_grasp_type: str = "power"
    _vlm_contact_geometry: str = "surface"
    _vlm_force_mode: str = "push"
    _vlm_task_semantic: str = ""
    _vlm_handle_confidence: float = 0.0
    _vlm_work_end_confidence: float = 0.0
    _vlm_attribute_confidence: Dict[str, float] = {}
    geometry_vlm_agreement_details: Dict[str, Any] = {}
    retry_info: Optional[Dict[str, Any]] = None

    if working_client is not None:
        prompt1 = _build_part_selection_prompt(
            instruction=instruction,
            object_name=object_name,
            n_candidates=len(crop_masks),
            has_depth=(depth_vis is not None),
            has_target=(target_crop_bgr is not None),
        )
        content1: List[Dict[str, Any]] = [
            {"type": "input_text", "text": prompt1},
            {"type": "input_image", "image_url": encode_image_to_data_url(crop_bgr)},
            {"type": "input_image", "image_url": encode_image_to_data_url(grid_img)},
        ]
        if depth_vis is not None:
            content1.append({"type": "input_image", "image_url": encode_image_to_data_url(depth_vis)})
        content1.append({"type": "input_image", "image_url": encode_image_to_data_url(skeleton_overlay)})
        if target_crop_bgr is not None:
            content1.append({"type": "input_image", "image_url": encode_image_to_data_url(target_crop_bgr)})

        try:
            raw1, _ = query_vlm_content(client=working_client, model=MODEL_CFG["vlm_model"], content=content1)
        except Exception as _vlm_exc:
            print(f"[VLM1] API call failed ({type(_vlm_exc).__name__}): {_vlm_exc} — will use geometry fallback")
            vlm_info["client_status"] = classify_openai_failure(_vlm_exc)
            working_client = None
            raw1 = None
        vlm_info["vlm1_raw"] = raw1

        parsed1 = _parse_part_selection(raw1 or "", n_candidates=len(crop_masks))
        selected_id = parsed1["handle_id"]
        we_selected_id = parsed1["work_end_id"]
        tool_morphology = str(parsed1.get("tool_morphology", "inline"))
        contact_face = str(parsed1["contact_face"])
        face_direction = str(parsed1["face_direction"])
        target_region = str(parsed1["target_region"])
        vlm1_confidence = float(parsed1["confidence"])

        # Structured affordance attributes from VLM-1 response
        _vlm_grasp_type = str(parsed1.get("grasp_type", "power") or "power")
        _vlm_contact_geometry = str(parsed1.get("contact_geometry", "surface") or "surface")
        _vlm_force_mode = str(parsed1.get("force_mode", "push") or "push")
        _vlm_task_semantic = str(parsed1.get("task_semantic", "") or "")
        _vlm_handle_confidence = float(parsed1.get("handle_confidence", vlm1_confidence) or vlm1_confidence)
        _vlm_work_end_confidence = float(parsed1.get("work_end_confidence", vlm1_confidence) or vlm1_confidence)
        _vlm_attribute_confidence = dict(parsed1.get("attribute_confidence", {}) or {})
        vlm1_parsed_extended = dict(parsed1)

        print(f"[VLM1] tool_morphology={tool_morphology!r}  contact_face={contact_face!r}  face_direction={face_direction!r}")

        # CLI override takes priority over VLM face_direction.
        if face_direction_override:
            face_direction = face_direction_override

        if require_api and selected_id is None:
            raise RuntimeError("handle_region_vlm_parse_failed")

        initial_agreement, initial_agreement_details = _geometry_vlm_agreement_score(
            vlm_handle_id=selected_id,
            vlm_work_end_id=we_selected_id,
            geom_handle_id=geom_handle_id,
            geom_work_end_id=geom_work_end_id,
            vlm_confidence=vlm1_confidence,
            instruction=instruction,
            geom_handle_mask=geom_handle_mask,
            geom_work_end_mask=geom_work_end_mask,
            mask_stats=mask_stats_aff,
            tool_morphology=tool_morphology,
            contact_face=contact_face,
            vlm_grasp_type=_vlm_grasp_type,
            vlm_contact_geometry=_vlm_contact_geometry,
            vlm_force_mode=_vlm_force_mode,
            vlm_attribute_confidence=_vlm_attribute_confidence,
            return_details=True,
        )
        geometry_vlm_agreement_details = dict(initial_agreement_details)
        can_retry = (
            selected_id is not None
            and we_selected_id is not None
            and selected_id != we_selected_id
            and geom_handle_id is not None
            and geom_work_end_id is not None
            and initial_agreement < RETRY_GEOMETRY_VLM_AGREEMENT_TH
        )
        if can_retry:
            candidate_a = {
                "handle_id": selected_id,
                "work_end_id": we_selected_id,
                "tool_morphology": tool_morphology,
                "contact_face": contact_face,
                "face_direction": face_direction,
                "target_region": target_region,
                "confidence": vlm1_confidence,
                "grasp_type": _vlm_grasp_type,
                "contact_geometry": _vlm_contact_geometry,
                "force_mode": _vlm_force_mode,
                "task_semantic": _vlm_task_semantic,
                "handle_confidence": _vlm_handle_confidence,
                "work_end_confidence": _vlm_work_end_confidence,
                "attribute_confidence": dict(_vlm_attribute_confidence),
            }
            candidate_b = dict(candidate_a)
            candidate_b["handle_id"] = we_selected_id
            candidate_b["work_end_id"] = selected_id
            retry_prompt = _build_handle_flip_retry_prompt(
                instruction=instruction,
                object_name=object_name,
                candidate_a=candidate_a,
                candidate_b=candidate_b,
                has_depth=(depth_vis is not None),
                geom_handle_id=geom_handle_id,
                geom_work_end_id=geom_work_end_id,
                mask_stats=mask_stats_aff,
            )
            retry_content: List[Dict[str, Any]] = [
                {"type": "input_text", "text": retry_prompt},
                {"type": "input_image", "image_url": encode_image_to_data_url(crop_bgr)},
                {"type": "input_image", "image_url": encode_image_to_data_url(grid_img)},
            ]
            if depth_vis is not None:
                retry_content.append({"type": "input_image", "image_url": encode_image_to_data_url(depth_vis)})
            retry_content.append({"type": "input_image", "image_url": encode_image_to_data_url(skeleton_overlay)})
            raw_retry, _ = query_vlm_content(client=working_client, model=MODEL_CFG["vlm_model"], content=retry_content)
            parsed_retry = _parse_handle_flip_retry(
                raw_retry,
                candidate_a=candidate_a,
                candidate_b=candidate_b,
            )
            retry_info = {
                "used": True,
                "threshold": RETRY_GEOMETRY_VLM_AGREEMENT_TH,
                "initial_agreement": initial_agreement,
                "initial_agreement_details": initial_agreement_details,
                "geometry_prior": {
                    "handle_id": geom_handle_id,
                    "work_end_id": geom_work_end_id,
                    "debug": geom_debug,
                },
                "candidate_a": {"handle_id": candidate_a["handle_id"], "work_end_id": candidate_a["work_end_id"]},
                "candidate_b": {"handle_id": candidate_b["handle_id"], "work_end_id": candidate_b["work_end_id"]},
                "raw_response": raw_retry,
            }
            if parsed_retry is not None:
                retry_choice = str(parsed_retry.get("retry_choice", "A"))
                selected_id = parsed_retry["handle_id"]
                we_selected_id = parsed_retry["work_end_id"]
                tool_morphology = str(parsed_retry.get("tool_morphology", tool_morphology))
                contact_face = str(parsed_retry.get("contact_face", contact_face))
                face_direction = str(parsed_retry.get("face_direction", face_direction))
                target_region = str(parsed_retry.get("target_region", target_region))
                vlm1_confidence = float(parsed_retry.get("confidence", vlm1_confidence) or vlm1_confidence)
                _vlm_grasp_type = str(parsed_retry.get("grasp_type", _vlm_grasp_type) or _vlm_grasp_type)
                _vlm_contact_geometry = str(parsed_retry.get("contact_geometry", _vlm_contact_geometry) or _vlm_contact_geometry)
                _vlm_force_mode = str(parsed_retry.get("force_mode", _vlm_force_mode) or _vlm_force_mode)
                _vlm_task_semantic = str(parsed_retry.get("task_semantic", _vlm_task_semantic) or _vlm_task_semantic)
                _vlm_handle_confidence = float(parsed_retry.get("handle_confidence", _vlm_handle_confidence) or _vlm_handle_confidence)
                _vlm_work_end_confidence = float(parsed_retry.get("work_end_confidence", _vlm_work_end_confidence) or _vlm_work_end_confidence)
                _vlm_attribute_confidence = dict(parsed_retry.get("attribute_confidence", _vlm_attribute_confidence) or _vlm_attribute_confidence)
                vlm1_parsed_extended.update(parsed_retry)
                if face_direction_override:
                    face_direction = face_direction_override
                final_agreement, final_agreement_details = _geometry_vlm_agreement_score(
                    vlm_handle_id=selected_id,
                    vlm_work_end_id=we_selected_id,
                    geom_handle_id=geom_handle_id,
                    geom_work_end_id=geom_work_end_id,
                    vlm_confidence=vlm1_confidence,
                    instruction=instruction,
                    geom_handle_mask=geom_handle_mask,
                    geom_work_end_mask=geom_work_end_mask,
                    mask_stats=mask_stats_aff,
                    tool_morphology=tool_morphology,
                    contact_face=contact_face,
                    vlm_grasp_type=_vlm_grasp_type,
                    vlm_contact_geometry=_vlm_contact_geometry,
                    vlm_force_mode=_vlm_force_mode,
                    vlm_attribute_confidence=_vlm_attribute_confidence,
                    return_details=True,
                )
                geometry_vlm_agreement_details = dict(final_agreement_details)
                retry_info.update({
                    "applied": True,
                    "choice": retry_choice,
                    "final_agreement": final_agreement,
                    "final_agreement_details": final_agreement_details,
                    "selected_ids": {"handle_id": selected_id, "work_end_id": we_selected_id},
                    "reason": parsed_retry.get("retry_reason", ""),
                })
                print(
                    f"[Retry] low agreement={initial_agreement:.2f} -> choice={retry_choice}"
                    f" final_agreement={final_agreement:.2f}"
                )
            else:
                retry_info["applied"] = False
                retry_info["parse_failed"] = True
                print(f"[Retry] low agreement={initial_agreement:.2f} but retry parse failed; keeping original selection")

    # CLI override applies even when VLM was not called.
    if face_direction_override:
        face_direction = face_direction_override

    # Geometry-only fallback: when the VLM was unavailable (no client) and
    # the shape-based analysis found candidates, use them so the pipeline
    # can continue without the API.
    if selected_id is None and working_client is None and geom_handle_id is not None:
        selected_id = geom_handle_id
        if we_selected_id is None:
            we_selected_id = geom_work_end_id
        print(
            f"[Fallback] VLM unavailable — using geometry-only selection: "
            f"handle={selected_id} work_end={we_selected_id}"
        )

    accepted = selected_id is not None

    # Save handle + work-end crop masks side-by-side in one image
    _h_crop_raw = (crop_masks[selected_id] > 0).astype(np.uint8) * 255 if (selected_id is not None and 0 <= selected_id < len(crop_masks)) else None
    _we_crop_raw = (crop_masks[we_selected_id] > 0).astype(np.uint8) * 255 if (we_selected_id is not None and 0 <= we_selected_id < len(crop_masks)) else None
    handle_mask_path = None
    we_mask_path = None
    if _h_crop_raw is not None or _we_crop_raw is not None:
        _lbl_h = np.zeros_like(_h_crop_raw if _h_crop_raw is not None else _we_crop_raw)
        _lbl_we = np.zeros_like(_we_crop_raw if _we_crop_raw is not None else _h_crop_raw)
        _left = cv2.cvtColor(_h_crop_raw if _h_crop_raw is not None else _lbl_h, cv2.COLOR_GRAY2BGR)
        _right = cv2.cvtColor(_we_crop_raw if _we_crop_raw is not None else _lbl_we, cv2.COLOR_GRAY2BGR)
        _lh, _lw = _left.shape[:2]; _rh, _rw = _right.shape[:2]
        _oh = max(_lh, _rh); _gap = 6
        _regions = np.zeros((_oh, _lw + _gap + _rw, 3), dtype=np.uint8)
        _regions[(_oh - _lh) // 2:(_oh - _lh) // 2 + _lh, :_lw] = _left
        _regions[(_oh - _rh) // 2:(_oh - _rh) // 2 + _rh, _lw + _gap:] = _right
        cv2.putText(_regions, "Handle", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 220, 80), 1, cv2.LINE_AA)
        cv2.putText(_regions, "Work-End", (_lw + _gap + 4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (80, 80, 220), 1, cv2.LINE_AA)
        regions_path = str(Path(out_dir) / f"{object_name}__regions.png")
        cv2.imwrite(regions_path, _regions)

    # ------------------------------------------------------------------ #
    # Axis: center-to-center                                              #
    # ------------------------------------------------------------------ #
    h_mask = crop_masks[selected_id] if (selected_id is not None and 0 <= selected_id < len(crop_masks)) else None
    we_mask = crop_masks[we_selected_id] if (we_selected_id is not None and 0 <= we_selected_id < len(crop_masks)) else None

    # Full-image (global) masks — embed crop masks back into full image space
    x1_off = int(crop_bb[0]) if crop_bb is not None else 0
    y1_off = int(crop_bb[1]) if crop_bb is not None else 0
    full_h, full_w = full_bgr.shape[:2]

    # Full-image masks saved to _masks/ (pipeline use only, not visual outputs)
    _masks_dir = str(Path(out_dir) / "_masks")
    ensure_dir(_masks_dir)
    handle_fullmask_path: Optional[str] = None
    if h_mask is not None:
        fm = np.zeros((full_h, full_w), dtype=np.uint8)
        if crop_bb is not None:
            x1c, y1c, x2c, y2c = crop_bb
            fm[y1c:y2c + 1, x1c:x2c + 1] = (h_mask > 0).astype(np.uint8) * 255
        else:
            fm = (h_mask > 0).astype(np.uint8) * 255
        handle_fullmask_path = str(Path(_masks_dir) / f"{object_name}__handle_fullmask.png")
        cv2.imwrite(handle_fullmask_path, fm)

    we_fullmask_path: Optional[str] = None
    if we_mask is not None:
        fm = np.zeros((full_h, full_w), dtype=np.uint8)
        if crop_bb is not None:
            x1c, y1c, x2c, y2c = crop_bb
            fm[y1c:y2c + 1, x1c:x2c + 1] = (we_mask > 0).astype(np.uint8) * 255
        else:
            fm = (we_mask > 0).astype(np.uint8) * 255
        we_fullmask_path = str(Path(_masks_dir) / f"{object_name}__workend_fullmask.png")
        cv2.imwrite(we_fullmask_path, fm)

    hx, hy = _mask_centroid(h_mask) if h_mask is not None else (float(crop_bgr.shape[1] / 2), float(crop_bgr.shape[0] / 2))
    wx, wy = _mask_centroid(we_mask) if we_mask is not None else (hx, hy)

    dx, dy = wx - hx, wy - hy
    norm = float(np.sqrt(dx ** 2 + dy ** 2))
    center_dir: Tuple[float, float] = (dx / norm, dy / norm) if norm > 1e-6 else (1.0, 0.0)
    # Use handle PCA major axis as the tool direction: more accurate than
    # centroid-to-centroid because the handle's elongation captures the true
    # handle orientation without vertical offset errors from head position.
    axis_dir: Tuple[float, float] = (
        _handle_pca_axis(h_mask, center_dir) if h_mask is not None else center_dir
    )
    axis_direction = [axis_dir[0], axis_dir[1]]

    # ------------------------------------------------------------------ #
    # Render region images (no axis line)                                  #
    # ------------------------------------------------------------------ #
    handle_vis = _render_region(
        crop_bgr, h_mask if h_mask is not None else np.zeros(crop_bgr.shape[:2], dtype=np.uint8),
        color_bgr=(50, 200, 50), label="Handle",
    )
    handle_vis_path = str(Path(out_dir) / f"{object_name}__pgl_handle.png")

    we_vis: Optional[np.ndarray] = None
    we_vis_path: Optional[str] = None
    if we_mask is not None:
        we_vis = _render_region(
            crop_bgr, we_mask, color_bgr=(50, 80, 220), label="Work-End",
        )
        we_vis_path = str(Path(out_dir) / f"{object_name}__pgl_work_end.png")

    # Dedicated axis image
    axis_img = _render_axis_image(crop_bgr, h_mask, we_mask, (hx, hy), (wx, wy))
    axis_path = str(Path(out_dir) / f"{object_name}__axis.png")
    cv2.imwrite(axis_path, axis_img)

    sel_skeleton_path = str(Path(out_dir) / f"{object_name}__skeleton.png")

    # ------------------------------------------------------------------ #
    # Skeleton keypoints for handle and work-end                          #
    # ------------------------------------------------------------------ #
    _h0, _w0 = crop_bgr.shape[:2]
    _h1, _w1 = handle_vis.shape[:2]
    _sx = _w1 / max(_w0, 1)
    _sy = _h1 / max(_h0, 1)

    _hmask = h_mask if h_mask is not None else np.zeros((_h0, _w0), dtype=np.uint8)
    h_kpts = _handle_skeleton_keypoints(_hmask, axis_dir, n=7)

    # ------------------------------------------------------------------ #
    # Axis-aligned affordance parameterization (closed-form, no VLM-2)   #
    # G   = 1/3 from the free (butt) end along the handle skeleton        #
    # C*  = Functional Axis Terminus — single contact point on work-end   #
    # ------------------------------------------------------------------ #
    n_hk = len(h_kpts)
    # Keypoints are ordered free-end → junction (ascending axis projection).
    # 1/3 from the free end is the natural power-grip position, clearly on
    # the proximal (handle) side rather than near the head junction.
    grip_xy: List[float] = list(h_kpts[max(0, n_hk // 3)]) if h_kpts else [hx, hy]

    contact_xy: Optional[Tuple[int, int]] = (
        _workend_fat_point(we_mask, (hx, hy), axis_dir,
                           contact_face=contact_face, tool_morphology=tool_morphology,
                           face_direction=face_direction)
        if we_mask is not None else None
    )

    # Global (full-image) pixel coordinates for interaction lift
    grip_point_xy_global: Optional[List[float]] = (
        [float(grip_xy[0] + x1_off), float(grip_xy[1] + y1_off)] if grip_xy is not None else None
    )
    work_end_contact_xy_global: Optional[List[float]] = (
        [float(contact_xy[0] + x1_off), float(contact_xy[1] + y1_off)] if contact_xy is not None else None
    )
    # Two work-end edge points perpendicular to tool axis (for interaction lift)
    work_end_edge_points_global: Optional[List[List[float]]] = None
    if we_mask is not None:
        _eys, _exs = np.where(we_mask > 0)
        if _exs.size >= 2:
            _ax, _ay = axis_dir
            _perp = _exs.astype(float) * (-_ay) + _eys.astype(float) * _ax
            _i1, _i2 = int(np.argmin(_perp)), int(np.argmax(_perp))
            work_end_edge_points_global = [
                [float(_exs[_i1] + x1_off), float(_eys[_i1] + y1_off)],
                [float(_exs[_i2] + x1_off), float(_eys[_i2] + y1_off)],
            ]

    # ------------------------------------------------------------------ #
    # 3D lifting: back-project G and C; PCA axis and contact normal       #
    # ------------------------------------------------------------------ #
    grip_point_3d: Optional[List[float]] = None
    work_end_contact_3d: Optional[List[float]] = None
    axis_direction_3d: Optional[List[float]] = None
    contact_normal_3d: Optional[List[float]] = None
    _h_cloud_3d: Optional[np.ndarray] = None
    _we_cloud_3d: Optional[np.ndarray] = None
    geometry_3d_path: Optional[str] = None
    contact_overlay_path: Optional[str] = None

    if depth is not None and camera_intrinsics is not None:
        fx3, fy3, ppx3, ppy3 = camera_intrinsics
        x1_off = int(crop_bb[0]) if crop_bb is not None else 0
        y1_off = int(crop_bb[1]) if crop_bb is not None else 0

        # Back-project G (grip) and C (contact) to 3D
        if grip_xy is not None:
            grip_point_3d = _pixel_to_3d(
                u=grip_xy[0] + x1_off, v=grip_xy[1] + y1_off,
                depth_map=depth, fx=fx3, fy=fy3, ppx=ppx3, ppy=ppy3,
                depth_scale=depth_scale,
            )
        if contact_xy is not None:
            work_end_contact_3d = _pixel_to_3d(
                u=contact_xy[0] + x1_off, v=contact_xy[1] + y1_off,
                depth_map=depth, fx=fx3, fy=fy3, ppx=ppx3, ppy=ppy3,
                depth_scale=depth_scale,
            )

        # 3D axis from handle mask PCA (major axis)
        if h_mask is not None:
            _h_cloud_3d = _mask_to_3d_cloud(
                mask=h_mask, depth_map=depth,
                offset_xy=(x1_off, y1_off),
                fx=fx3, fy=fy3, ppx=ppx3, ppy=ppy3,
                depth_scale=depth_scale,
            )
            if _h_cloud_3d is not None:
                major3, _mid3, _minor3 = _pca_3d(_h_cloud_3d)
                if grip_point_3d is not None and work_end_contact_3d is not None:
                    ref = np.array(work_end_contact_3d) - np.array(grip_point_3d)
                    if np.dot(major3, ref) < 0:
                        major3 = -major3
                elif major3[2] < 0:
                    major3 = -major3
                axis_direction_3d = major3.tolist()

        # Contact normal from work-end mask PCA (minor axis = plane normal)
        if we_mask is not None:
            _we_cloud_3d = _mask_to_3d_cloud(
                mask=we_mask, depth_map=depth,
                offset_xy=(x1_off, y1_off),
                fx=fx3, fy=fy3, ppx=ppx3, ppy=ppy3,
                depth_scale=depth_scale,
            )
            if _we_cloud_3d is not None:
                _major3, _mid3, normal3 = _pca_3d(_we_cloud_3d)
                if normal3[2] > 0:
                    normal3 = -normal3
                contact_normal_3d = normal3.tolist()

        geometry_3d_path = None

        # 2D RGB overlay: point clouds + 3D geometry back-projected onto full image
        overlay_bgr = _render_contact_overlay(
            full_bgr=full_bgr,
            grip_3d=grip_point_3d,
            contact_3d=work_end_contact_3d,
            axis_3d=axis_direction_3d,
            normal_3d=contact_normal_3d,
            fx=fx3, fy=fy3, ppx=ppx3, ppy=ppy3,
            h_cloud=_h_cloud_3d,
            we_cloud=_we_cloud_3d,
        )
        contact_overlay_path = str(Path(out_dir) / f"{object_name}__contact_overlay.png")
        cv2.imwrite(contact_overlay_path, overlay_bgr)

        # Print 3D coordinates to terminal
        _f = lambda v, u=" m": f"({v[0]:+.4f}, {v[1]:+.4f}, {v[2]:+.4f}){u}" if v else "n/a"
        print(f"[3D] {object_name}")
        print(f"     G  grip    : {_f(grip_point_3d)}")
        print(f"     C  contact : {_f(work_end_contact_3d)}")
        print(f"     axis 3D   : {_f(axis_direction_3d, '')}")
        print(f"     normal 3D : {_f(contact_normal_3d, '')}")

    # ------------------------------------------------------------------ #
    # Structured region-level affordance computation (journal contribution)#
    # ------------------------------------------------------------------ #

    geometry_vlm_agreement, geometry_vlm_agreement_details = _geometry_vlm_agreement_score(
        vlm_handle_id=selected_id,
        vlm_work_end_id=we_selected_id,
        geom_handle_id=geom_handle_id,
        geom_work_end_id=geom_work_end_id,
        vlm_confidence=vlm1_confidence,
        instruction=instruction,
        geom_handle_mask=geom_handle_mask,
        geom_work_end_mask=geom_work_end_mask,
        mask_stats=mask_stats_aff,
        tool_morphology=tool_morphology,
        contact_face=contact_face,
        vlm_grasp_type=_vlm_grasp_type,
        vlm_contact_geometry=_vlm_contact_geometry,
        vlm_force_mode=_vlm_force_mode,
        vlm_attribute_confidence=_vlm_attribute_confidence,
        return_details=True,
    )

    # Full SO(3) contact wrench basis from 3D geometry
    wrench_basis: Optional[Dict[str, List[float]]] = _compute_wrench_basis(
        axis_direction_3d, contact_normal_3d)

    # Functional area (work-end) and grasp width (handle) in pixels
    functional_area_px: Optional[int] = (
        int(mask_area((we_mask > 0).astype(np.uint8))) if we_mask is not None else None)
    grasp_width_px: Optional[float] = (
        _compute_grasp_width_from_skeleton(h_mask, axis_dir) if h_mask is not None else None)
    grasp_width_m: Optional[float] = _compute_grasp_width_from_cloud_3d(
        handle_cloud_3d=_h_cloud_3d,
        axis_direction_3d=axis_direction_3d,
    )
    grasp_width_mm: Optional[float] = (
        None if grasp_width_m is None else float(grasp_width_m * 1000.0)
    )
    grasp_pose = _compute_grasp_pose_from_wrench_basis(
        grip_point_3d=grip_point_3d,
        wrench_basis=wrench_basis,
    )

    # Build formal RegionAffordanceTuple for handle and work-end
    handle_affordance: Optional[Dict[str, Any]] = None
    work_end_affordance: Optional[Dict[str, Any]] = None

    if h_mask is not None and selected_id is not None:
        h_aff = _build_region_affordance_tuple(
            functional_role="handle", instruction=instruction,
            h_mask=h_mask, we_mask=we_mask, mask_stats=mask_stats_aff,
            selected_id=selected_id, tool_morphology=tool_morphology,
            contact_face=contact_face, vlm_parsed=vlm1_parsed_extended,
            geometry_vlm_agreement=geometry_vlm_agreement,
        )
        handle_affordance = {
            "functional_role": h_aff.functional_role,
            "grasp_type": h_aff.grasp_type,
            "contact_geometry": h_aff.contact_geometry,
            "force_mode": h_aff.force_mode,
            "task_semantic": h_aff.task_semantic,
            "confidence": h_aff.confidence,
            "attribute_confidence": h_aff.attribute_confidence,
        }
    if we_mask is not None:
        we_aff = _build_region_affordance_tuple(
            functional_role="work_end", instruction=instruction,
            h_mask=h_mask, we_mask=we_mask, mask_stats=mask_stats_aff,
            selected_id=selected_id, tool_morphology=tool_morphology,
            contact_face=contact_face, vlm_parsed=vlm1_parsed_extended,
            geometry_vlm_agreement=geometry_vlm_agreement,
        )
        work_end_affordance = {
            "functional_role": we_aff.functional_role,
            "grasp_type": we_aff.grasp_type,
            "contact_geometry": we_aff.contact_geometry,
            "force_mode": we_aff.force_mode,
            "task_semantic": we_aff.task_semantic,
            "confidence": we_aff.confidence,
            "attribute_confidence": we_aff.attribute_confidence,
        }

    # Final merged affordance attributes (VLM primary, geometry fallback)
    _grasp_type_final = _vlm_grasp_type if _vlm_grasp_type in _VALID_GRASP_TYPES else (
        handle_affordance.get("grasp_type", "power") if handle_affordance else "power")
    _contact_geom_final = _vlm_contact_geometry if _vlm_contact_geometry in _VALID_CONTACT_GEOMS else (
        work_end_affordance.get("contact_geometry", "surface") if work_end_affordance else "surface")
    _force_mode_final = _vlm_force_mode if _vlm_force_mode in _VALID_FORCE_MODES else (
        work_end_affordance.get("force_mode", "push") if work_end_affordance else "push")
    _task_sem_final = _vlm_task_semantic or _build_task_semantic_label(instruction, "handle")
    coherence_rank = _rank_affordance_triple_hypotheses(
        geometry_vlm_agreement_details=geometry_vlm_agreement_details,
    )
    fused_dists = dict(coherence_rank.get("fused_distributions", {}) or {})
    current_triple = {
        "grasp_type": _grasp_type_final,
        "contact_geometry": _contact_geom_final,
        "force_mode": _force_mode_final,
    }
    current_coherence = _evaluate_affordance_triple_coherence(
        current_triple["grasp_type"],
        current_triple["contact_geometry"],
        current_triple["force_mode"],
    )

    def _joint_prob(triple: Dict[str, str]) -> float:
        return float(
            (fused_dists.get("grasp_type", {}) or {}).get(str(triple.get("grasp_type", "")).strip().lower(), 0.0)
            * (fused_dists.get("contact_geometry", {}) or {}).get(str(triple.get("contact_geometry", "")).strip().lower(), 0.0)
            * (fused_dists.get("force_mode", {}) or {}).get(str(triple.get("force_mode", "")).strip().lower(), 0.0)
        )

    current_raw_joint = _joint_prob(current_triple)
    current_adjusted = float(current_raw_joint * float(current_coherence.get("weight", 0.0)))
    valid_hypotheses = list(coherence_rank.get("valid_hypotheses", []) or [])
    best_valid = valid_hypotheses[0] if valid_hypotheses else None
    coherence_report: Dict[str, Any] = {
        "current_triple": dict(current_triple),
        "current_coherence": current_coherence,
        "current_raw_joint": current_raw_joint,
        "current_adjusted_score": current_adjusted,
        "fused_distributions": fused_dists,
        "top_hypotheses": (coherence_rank.get("hypotheses", []) or [])[:5],
        "selected_source": "initial",
    }

    coherence_retry_used = False
    if str(current_coherence.get("status", "valid")) != "valid":
        if str(current_coherence.get("status")) == "forbidden" and working_client is not None and valid_hypotheses:
            candidate_options: List[Dict[str, Any]] = []
            for idx, hyp in enumerate(valid_hypotheses[:3]):
                candidate_options.append({
                    "label": chr(ord("A") + idx),
                    "triple": dict(hyp.get("triple", {}) or {}),
                    "score": float(hyp.get("adjusted_score", 0.0)),
                })
            retry_prompt = _build_affordance_coherence_retry_prompt(
                instruction=instruction,
                object_name=object_name,
                current_triple=current_triple,
                invalid_reasons=list(current_coherence.get("reasons", []) or []),
                candidate_options=candidate_options,
                has_depth=(depth_vis is not None),
                has_target=(target_crop_bgr is not None),
            )
            retry_content: List[Dict[str, Any]] = [
                {"type": "input_text", "text": retry_prompt},
                {"type": "input_image", "image_url": encode_image_to_data_url(crop_bgr)},
                {"type": "input_image", "image_url": encode_image_to_data_url(grid_img)},
            ]
            if depth_vis is not None:
                retry_content.append({"type": "input_image", "image_url": encode_image_to_data_url(depth_vis)})
            retry_content.append({"type": "input_image", "image_url": encode_image_to_data_url(skeleton_overlay)})
            if target_crop_bgr is not None:
                retry_content.append({"type": "input_image", "image_url": encode_image_to_data_url(target_crop_bgr)})
            raw_coh_retry, _ = query_vlm_content(client=working_client, model=MODEL_CFG["vlm_model"], content=retry_content)
            parsed_coh_retry = _parse_affordance_coherence_retry(
                raw_coh_retry,
                candidate_options=candidate_options,
            )
            coherence_report["retry"] = {
                "used": True,
                "raw_response": raw_coh_retry,
                "candidate_options": candidate_options,
            }
            if parsed_coh_retry is not None:
                repaired_triple = dict(parsed_coh_retry.get("triple", {}) or {})
                _grasp_type_final = str(repaired_triple.get("grasp_type", _grasp_type_final))
                _contact_geom_final = str(repaired_triple.get("contact_geometry", _contact_geom_final))
                _force_mode_final = str(repaired_triple.get("force_mode", _force_mode_final))
                _task_sem_final = str(parsed_coh_retry.get("task_semantic", _task_sem_final) or _task_sem_final)
                _vlm_grasp_type = _grasp_type_final
                _vlm_contact_geometry = _contact_geom_final
                _vlm_force_mode = _force_mode_final
                if parsed_coh_retry.get("attribute_confidence"):
                    _vlm_attribute_confidence.update(dict(parsed_coh_retry["attribute_confidence"]))
                vlm1_parsed_extended.update({
                    "grasp_type": _vlm_grasp_type,
                    "contact_geometry": _vlm_contact_geometry,
                    "force_mode": _vlm_force_mode,
                    "task_semantic": _task_sem_final,
                    "attribute_confidence": dict(_vlm_attribute_confidence),
                })
                coherence_report["retry"].update({
                    "applied": True,
                    "choice": parsed_coh_retry.get("choice"),
                    "reason": parsed_coh_retry.get("reason", ""),
                    "selected_triple": {
                        "grasp_type": _grasp_type_final,
                        "contact_geometry": _contact_geom_final,
                        "force_mode": _force_mode_final,
                    },
                })
                coherence_report["selected_source"] = "coherence_retry"
                coherence_retry_used = True
            else:
                coherence_report["retry"]["applied"] = False

        if not coherence_retry_used and best_valid is not None:
            best_valid_score = float(best_valid.get("adjusted_score", 0.0))
            should_autorepair = (
                str(current_coherence.get("status")) == "forbidden"
                or best_valid_score > max(1e-9, current_adjusted * COHERENCE_AUTO_REPAIR_MARGIN)
            )
            if should_autorepair:
                repaired_triple = dict(best_valid.get("triple", {}) or {})
                _grasp_type_final = str(repaired_triple.get("grasp_type", _grasp_type_final))
                _contact_geom_final = str(repaired_triple.get("contact_geometry", _contact_geom_final))
                _force_mode_final = str(repaired_triple.get("force_mode", _force_mode_final))
                _vlm_grasp_type = _grasp_type_final
                _vlm_contact_geometry = _contact_geom_final
                _vlm_force_mode = _force_mode_final
                _vlm_attribute_confidence.update({
                    "grasp_type": float((fused_dists.get("grasp_type", {}) or {}).get(_grasp_type_final, _vlm_attribute_confidence.get("grasp_type", vlm1_confidence))),
                    "contact_geometry": float((fused_dists.get("contact_geometry", {}) or {}).get(_contact_geom_final, _vlm_attribute_confidence.get("contact_geometry", vlm1_confidence))),
                    "force_mode": float((fused_dists.get("force_mode", {}) or {}).get(_force_mode_final, _vlm_attribute_confidence.get("force_mode", vlm1_confidence))),
                })
                vlm1_parsed_extended.update({
                    "grasp_type": _vlm_grasp_type,
                    "contact_geometry": _vlm_contact_geometry,
                    "force_mode": _vlm_force_mode,
                    "attribute_confidence": dict(_vlm_attribute_confidence),
                })
                coherence_report["auto_repair"] = {
                    "applied": True,
                    "selected_triple": repaired_triple,
                    "source_hypothesis": best_valid,
                }
                coherence_report["selected_source"] = "coherence_auto_repair"

    if coherence_report.get("selected_source") != "initial":
        if h_mask is not None and selected_id is not None:
            h_aff = _build_region_affordance_tuple(
                functional_role="handle", instruction=instruction,
                h_mask=h_mask, we_mask=we_mask, mask_stats=mask_stats_aff,
                selected_id=selected_id, tool_morphology=tool_morphology,
                contact_face=contact_face, vlm_parsed=vlm1_parsed_extended,
                geometry_vlm_agreement=geometry_vlm_agreement,
            )
            handle_affordance = {
                "functional_role": h_aff.functional_role,
                "grasp_type": h_aff.grasp_type,
                "contact_geometry": h_aff.contact_geometry,
                "force_mode": h_aff.force_mode,
                "task_semantic": h_aff.task_semantic,
                "confidence": h_aff.confidence,
                "attribute_confidence": h_aff.attribute_confidence,
            }
        if we_mask is not None:
            we_aff = _build_region_affordance_tuple(
                functional_role="work_end", instruction=instruction,
                h_mask=h_mask, we_mask=we_mask, mask_stats=mask_stats_aff,
                selected_id=selected_id, tool_morphology=tool_morphology,
                contact_face=contact_face, vlm_parsed=vlm1_parsed_extended,
                geometry_vlm_agreement=geometry_vlm_agreement,
            )
            work_end_affordance = {
                "functional_role": we_aff.functional_role,
                "grasp_type": we_aff.grasp_type,
                "contact_geometry": we_aff.contact_geometry,
                "force_mode": we_aff.force_mode,
                "task_semantic": we_aff.task_semantic,
                "confidence": we_aff.confidence,
                "attribute_confidence": we_aff.attribute_confidence,
            }

    selected_triple = {
        "grasp_type": _grasp_type_final,
        "contact_geometry": _contact_geom_final,
        "force_mode": _force_mode_final,
    }
    coherence_report["selected_triple"] = dict(selected_triple)
    coherence_report["selected_coherence"] = _evaluate_affordance_triple_coherence(
        _grasp_type_final,
        _contact_geom_final,
        _force_mode_final,
    )
    affordance_hypotheses: List[Dict[str, Any]] = []
    for hyp in (coherence_rank.get("hypotheses", []) or [])[:8]:
        affordance_hypotheses.append({
            "triple": dict(hyp.get("triple", {}) or {}),
            "coherence": dict(hyp.get("coherence", {}) or {}),
            "raw_probability": float(hyp.get("raw_probability", 0.0) or 0.0),
            "adjusted_probability": float(hyp.get("adjusted_probability", 0.0) or 0.0),
            "adjusted_score": float(hyp.get("adjusted_score", 0.0) or 0.0),
        })
    coherence_report["affordance_hypotheses"] = affordance_hypotheses

    # Calibrated per-attribute uncertainty after coherence repair
    affordance_uncertainty: Optional[Dict[str, float]] = _compute_calibrated_uncertainty(
        vlm_confidence=vlm1_confidence,
        geometry_vlm_agreement=geometry_vlm_agreement,
        attribute_confidences=_vlm_attribute_confidence,
    )

    print(
        f"[Affordance] grasp={_grasp_type_final!r}  contact_geom={_contact_geom_final!r}"
        f"  force_mode={_force_mode_final!r}  agreement={geometry_vlm_agreement:.2f}"
    )
    if grasp_width_mm is not None:
        print(f"[GraspWidth] width={grasp_width_mm:.1f} mm")
    if grasp_pose["grasp_quaternion_xyzw"] is not None:
        qx, qy, qz, qw = grasp_pose["grasp_quaternion_xyzw"]
        print(
            f"[GraspPose] quat_xyzw=({qx:+.4f}, {qy:+.4f}, {qz:+.4f}, {qw:+.4f})"
        )

    # Save handle skeleton overlay (debug visualisation)
    handle_kpt_vis = _draw_keypoints_numbered(handle_vis, h_kpts, (30, 30, 220), _sx, _sy)
    handle_kpt_path = str(Path(out_dir) / f"{object_name}__handle_keypoints.png")
    cv2.imwrite(handle_kpt_path, handle_kpt_vis)

    # Combined handle + work-end overlaid on one image with G and C callouts
    combined_pgl = _make_combined_pgl(crop_bgr, h_mask, we_mask,
                                      grip_xy=grip_xy, contact_xy=contact_xy)
    combined_pgl_path = str(Path(out_dir) / f"{object_name}__pgl_combined.png")

    # ------------------------------------------------------------------ #
    # Result                                                               #
    # ------------------------------------------------------------------ #
    decision = HandleDecision(
        object_name=object_name,
        ok=accepted,
        status="accepted" if accepted else "rejected",
        selected_handle_id=selected_id,
        selected_handle_ids=[] if selected_id is None else [selected_id],
        confidence=vlm1_confidence,
        selected_work_end_id=we_selected_id,
        selected_work_end_ids=[] if we_selected_id is None else [we_selected_id],
        work_end_ok=(we_selected_id is not None),
        work_end_status="accepted" if we_selected_id is not None else "rejected",
        work_end_confidence=vlm1_confidence,
        axis_direction=axis_direction,
        vlm=vlm_info,
        artifacts={
            "rgb_crop_path": rgb_path,
            "candidate_grid_path": grid_path,
            "skeleton_overlay_path": skeleton_path,
            "depth_colormap_path": depth_vis_path,
            "handle_mask_path": handle_mask_path,
            "work_end_mask_path": we_mask_path,
            "handle_fullmask_path": handle_fullmask_path,
            "work_end_fullmask_path": we_fullmask_path,
            "handle_vis_path": handle_vis_path,
            "work_end_vis_path": we_vis_path,
            "axis_image_path": axis_path,
            "skeleton_image_path": sel_skeleton_path,
            "combined_pgl_path": combined_pgl_path,
            "geometry_3d_path": geometry_3d_path,
            "contact_overlay_path": contact_overlay_path,
            "crop_bb": list(crop_bb) if crop_bb is not None else None,
        },
        grip_point_xy=grip_xy,
        grip_anchor_label="G",
        work_end_edge=None,
        work_end_contact_xy=list(contact_xy) if contact_xy is not None else None,
        work_end_side_label="C",
        contact_face=contact_face,
        target_region=target_region,
        grip_target_line=axis_direction,
        grip_point_3d=grip_point_3d,
        work_end_contact_3d=work_end_contact_3d,
        axis_direction_3d=axis_direction_3d,
        contact_normal_3d=contact_normal_3d,
        grasp_rotation_cam=grasp_pose["grasp_rotation_cam"],
        grasp_frame_cam=grasp_pose["grasp_frame_cam"],
        grasp_quaternion_xyzw=grasp_pose["grasp_quaternion_xyzw"],
        grasp_approach_axis_3d=grasp_pose["grasp_approach_axis_3d"],
        grasp_closing_axis_3d=grasp_pose["grasp_closing_axis_3d"],
        grip_point_xy_global=grip_point_xy_global,
        work_end_contact_xy_global=work_end_contact_xy_global,
        work_end_edge_points_global=work_end_edge_points_global,
        # Structured affordance fields (journal contribution)
        grasp_type=_grasp_type_final,
        contact_geometry=_contact_geom_final,
        force_mode=_force_mode_final,
        task_semantic=_task_sem_final,
        affordance_uncertainty=affordance_uncertainty,
        geometry_vlm_agreement=geometry_vlm_agreement,
        geometry_vlm_agreement_details=geometry_vlm_agreement_details,
        coherence_report=coherence_report,
        affordance_hypotheses=affordance_hypotheses,
        wrench_basis=wrench_basis,
        functional_area_px=functional_area_px,
        grasp_width_px=grasp_width_px,
        grasp_width_m=grasp_width_m,
        grasp_width_mm=grasp_width_mm,
        handle_affordance=handle_affordance,
        work_end_affordance=work_end_affordance,
        retry_info=retry_info,
    )
    save_json(str(Path(out_dir) / "handle_region_result.json"), asdict(decision))
    return decision


# ---------------------------------------------------------------------------
# Entry point from SAM JSON
# ---------------------------------------------------------------------------

def _resolve_depth_from_rgb(rgb_path: str, explicit_depth_path: Optional[str] = None) -> Optional[str]:
    if explicit_depth_path:
        return explicit_depth_path
    p = Path(rgb_path)
    guess = p.parent / p.name.replace("color_", "depth_").replace(".png", ".npy")
    return str(guess) if guess.exists() else None


def _with_whole_first_object(obj: Dict[str, Any]) -> Dict[str, Any]:
    row = dict(obj)
    whole_path = str(row.get("whole_mask_path", "") or "")
    mask_paths = [str(p) for p in row.get("mask_paths", []) if str(p)]
    if whole_path and whole_path not in mask_paths:
        row["mask_paths"] = [whole_path, *mask_paths]
        row["num_candidates"] = len(row["mask_paths"])
    return row


def run_handle_region_policy_from_sam(
    *,
    sam_json_path: str,
    instruction: str,
    out_dir: str,
    object_name: Optional[str] = None,
    rgb_path: Optional[str] = None,
    depth_path: Optional[str] = None,
    require_api: bool = False,
    face_direction_override: Optional[str] = None,
    target_object_name: Optional[str] = None,
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    sam_summary = load_json(sam_json_path)
    resolved_rgb = rgb_path or str(sam_summary.get("image_path") or "")
    if not resolved_rgb:
        raise FileNotFoundError("RGB path is missing from sam summary and CLI arguments.")
    full_bgr = cv2.imread(resolved_rgb, cv2.IMREAD_COLOR)
    if full_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB image: {resolved_rgb}")
    resolved_depth = _resolve_depth_from_rgb(resolved_rgb, explicit_depth_path=depth_path)
    depth = load_depth(resolved_depth)

    # Load camera intrinsics and depth scale from per-frame meta JSON.
    camera_intrinsics: Optional[Tuple[float, float, float, float]] = None
    depth_scale: float = 0.001
    meta_path = _derive_meta_json_path(resolved_rgb)
    if meta_path:
        cam_intr, ds = _load_camera_intrinsics(meta_path)
        if cam_intr is not None:
            camera_intrinsics = cam_intr
        if ds is not None:
            depth_scale = ds

    objects = sam_summary.get("objects", [])
    selected_objects: List[Dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict) or not bool(obj.get("ok")):
            continue
        name = str(obj.get("name", "")).strip()
        if not name:
            continue
        if object_name and name != object_name:
            continue
        selected_objects.append(_with_whole_first_object(obj))

    if not selected_objects:
        raise ValueError("No matching SAM objects found for handle-region policy.")

    # Resolve target crop once for the whole batch.
    # When target_object_name is given, find its SAM entry, read the whole-mask,
    # and crop the RGB image to a padded bounding box of that mask.  The crop is
    # passed to every tool's VLM call so contact_geometry / force_mode are
    # conditioned on the actual target surface rather than only on the task verb.
    resolved_target_crop: Optional[np.ndarray] = None
    if target_object_name:
        for t_obj in objects:
            if str(t_obj.get("name", "")).strip() != target_object_name:
                continue
            t_whole = str(t_obj.get("whole_mask_path", "") or "")
            if t_whole and Path(t_whole).exists():
                try:
                    t_mask = load_mask01(t_whole)
                    bb = mask_bbox_xyxy(t_mask)
                    if bb is not None:
                        x1t, y1t, x2t, y2t = bb
                        pad = 12
                        h_img, w_img = full_bgr.shape[:2]
                        x1t = max(0, x1t - pad)
                        y1t = max(0, y1t - pad)
                        x2t = min(w_img - 1, x2t + pad)
                        y2t = min(h_img - 1, y2t + pad)
                        resolved_target_crop = full_bgr[y1t:y2t + 1, x1t:x2t + 1].copy()
                        print(
                            f"[RegioManip] target_object={target_object_name!r}"
                            f" crop={resolved_target_crop.shape[:2]} from mask {t_whole}"
                        )
                except Exception as _te:
                    print(f"[RegioManip] warning: could not resolve target crop: {_te}")
            break

    client, client_status = build_openai_client_with_status()
    if require_api and client is None:
        raise RuntimeError(f"handle_region_api_unavailable: {client_status}")

    results: List[Dict[str, Any]] = []
    for obj in selected_objects:
        name = str(obj["name"]).strip()

        # Skip the target/passive object — it is not a tool to grasp.
        if target_object_name and name == target_object_name:
            print(f"[RegioManip] Skipping {name!r} — it is the target object, not a tool.")
            continue

        mask_paths = [str(p) for p in obj.get("mask_paths", []) if str(p)]

        # Objects with only one mask have no part decomposition (handle vs work-end).
        # These are passive/target objects detected by DINO, not graspable tools.
        if len(mask_paths) < 2:
            print(f"[RegioManip] Skipping {name!r} — only {len(mask_paths)} mask(s), no part decomposition.")
            continue

        masks = [load_mask01(p) for p in mask_paths]
        obj_out_dir = str(Path(out_dir) / "region_affordance")
        decision = run_handle_region_policy(
            object_name=name,
            instruction=instruction,
            masks=masks,
            full_bgr=full_bgr,
            depth=depth,
            out_dir=obj_out_dir,
            require_api=require_api,
            client=client,
            face_direction_override=face_direction_override,
            camera_intrinsics=camera_intrinsics,
            depth_scale=depth_scale,
            target_crop_bgr=resolved_target_crop,
        )
        results.append(asdict(decision))

    # Append a lightweight "target" entry for each object that was skipped as a
    # passive/target object so downstream code and visualisations can use it.
    target_results: List[Dict[str, Any]] = []
    for obj in objects:
        if not isinstance(obj, dict) or not bool(obj.get("ok")):
            continue
        name = str(obj.get("name", "")).strip()
        if not name:
            continue
        # Only include objects we deliberately skipped (they have ≤1 mask or are the named target).
        is_named_target = bool(target_object_name and name == target_object_name)
        mask_paths_obj = [str(p) for p in obj.get("mask_paths", []) if str(p)]
        is_whole_only = len(mask_paths_obj) < 2
        if not (is_named_target or is_whole_only):
            continue
        # Compute centroid and bbox from whole mask.
        whole_path = str(obj.get("whole_mask_path", "") or "")
        centroid: Optional[List[float]] = None
        bbox: Optional[List[int]] = None
        if whole_path and Path(whole_path).exists():
            try:
                t_mask = load_mask01(whole_path)
                bb = mask_bbox_xyxy(t_mask)
                if bb is not None:
                    x1t, y1t, x2t, y2t = bb
                    bbox = [int(x1t), int(y1t), int(x2t), int(y2t)]
                    ys, xs = np.where(t_mask > 0)
                    if len(xs):
                        centroid = [float(xs.mean()), float(ys.mean())]
            except Exception:
                pass
        target_results.append({
            "object_name": name,
            "role": "target",
            "ok": True,
            "mask_path": whole_path or None,
            "centroid_xy_global": centroid,
            "bbox_xyxy_global": bbox,
        })

    summary = {
        "ok": any(bool(item.get("ok")) for item in results),
        "instruction": instruction,
        "sam_json_path": sam_json_path,
        "rgb_path": resolved_rgb,
        "depth_path": resolved_depth,
        "results": results,
        "targets": target_results,
    }
    save_json(str(Path(out_dir) / "handle_region_results.json"), summary)
    return summary


run_region_level_affordance = run_handle_region_policy
run_region_level_affordance_from_sam = run_handle_region_policy_from_sam

__all__ = [
    "HandleDecision",
    "RegionAffordanceTuple",
    "run_handle_region_policy",
    "run_handle_region_policy_from_sam",
    "run_region_level_affordance",
    "run_region_level_affordance_from_sam",
    "make_journal_figure",
]
