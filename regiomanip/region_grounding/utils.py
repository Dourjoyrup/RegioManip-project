from __future__ import annotations

import base64
import json
import os
import re
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

TOOL_NAMES = {"scraper", "putty_knife", "screwdriver", "roller", "brush", "paint_brush", "hammer", "mallet", "tool"}

TOOL_HINT_WORDS = {
    "tool",
    "scraper",
    "knife",
    "spatula",
    "screwdriver",
    "roller",
    "hammer",
    "mallet",
    "brush",
    "wrench",
    "plier",
    "spanner",
    "trowel",
    "stick",
    "rod",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def save_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_builtin(payload), f, indent=2)


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        out = json.load(f)
    if not isinstance(out, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return out


def is_tool(name: str) -> bool:
    raw = (name or "").lower().strip()
    if raw in TOOL_NAMES:
        return True

    norm = re.sub(r"[_\-]+", " ", raw)
    if norm.replace(" ", "_") in TOOL_NAMES:
        return True

    for w in TOOL_HINT_WORDS:
        if re.search(rf"(?<!\w){re.escape(w)}(?!\w)", norm):
            return True
    return False


def load_mask01(path: str) -> np.ndarray:
    m = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if m is None:
        raise FileNotFoundError(f"Cannot read mask: {path}")
    return (m > 0).astype(np.uint8)


def keep_largest_cc(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 2:
        return m
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (lab == largest).astype(np.uint8)


def mask_area(mask01: np.ndarray) -> int:
    return int((mask01 > 0).sum())


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    aa = a.astype(bool)
    bb = b.astype(bool)
    inter = np.logical_and(aa, bb).sum()
    uni = np.logical_or(aa, bb).sum()
    return float(inter) / float(uni + 1e-6)


def union_masks(masks: List[np.ndarray]) -> np.ndarray:
    if not masks:
        raise ValueError("union_masks requires at least one mask")
    out = np.zeros_like(masks[0], dtype=np.uint8)
    for m in masks:
        out |= (m > 0).astype(np.uint8)
    return out


def _num_components(mask01: np.ndarray) -> int:
    m = (mask01 > 0).astype(np.uint8)
    n, _, _, _ = cv2.connectedComponentsWithStats(m)
    return max(0, int(n - 1))


def _keep_significant_components(mask01: np.ndarray, rel_min: float = 0.06, pix_min: int = 120) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m)
    if n <= 2:
        return m
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    largest = int(areas.max())
    th = max(int(pix_min), int(round(float(largest) * float(rel_min))))
    keep_labels = [i + 1 for i, a in enumerate(areas.tolist()) if int(a) >= th]
    if not keep_labels:
        keep_labels = [1 + int(np.argmax(areas))]
    return np.isin(lab, keep_labels).astype(np.uint8)


def _bridge_two_main_components(mask01: np.ndarray) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)
    n, _, stats, cents = cv2.connectedComponentsWithStats(m)
    if n <= 2:
        return m

    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    order = (np.argsort(areas)[::-1] + 1).tolist()
    if len(order) < 2:
        return m

    l1, l2 = int(order[0]), int(order[1])
    c1 = cents[l1]
    c2 = cents[l2]
    h, w = m.shape[:2]
    diag = float(np.hypot(h, w))
    thickness = max(2, int(round(diag * 0.010)))
    k = max(3, int(round(diag * 0.010)))
    if k % 2 == 0:
        k += 1

    bridge = np.zeros_like(m, dtype=np.uint8)
    p1 = (int(round(float(c1[0]))), int(round(float(c1[1]))))
    p2 = (int(round(float(c2[0]))), int(round(float(c2[1]))))
    cv2.line(bridge, p1, p2, 1, thickness=thickness)

    out = np.logical_or(m > 0, bridge > 0).astype(np.uint8)
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker, iterations=1)
    out = np.logical_or(out > 0, m > 0).astype(np.uint8)
    return out


def compose_tool_whole(handle: np.ndarray, work_end: np.ndarray, whole_hint: Optional[np.ndarray] = None) -> np.ndarray:
    parts = [(handle > 0).astype(np.uint8), (work_end > 0).astype(np.uint8)]
    if whole_hint is not None:
        parts.append((whole_hint > 0).astype(np.uint8))

    whole = union_masks(parts)
    whole = _keep_significant_components(whole, rel_min=0.04, pix_min=80)
    if _num_components(whole) >= 2:
        whole = _bridge_two_main_components(whole)
    whole = _keep_significant_components(whole, rel_min=0.03, pix_min=60)
    return whole


def centroid(mask01: np.ndarray) -> np.ndarray:
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return np.array([0.0, 0.0], dtype=np.float32)
    return np.array([float(xs.mean()), float(ys.mean())], dtype=np.float32)


def pca_axis_from_mask(mask01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.where(mask01 > 0)
    if xs.size < 10:
        return np.array([1.0, 0.0], dtype=np.float32), np.array([0.0, 0.0], dtype=np.float32)
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mu = pts.mean(axis=0)
    centered = pts - mu
    cov = (centered.T @ centered) / max(1.0, float(len(pts)))
    eigvals, eigvecs = np.linalg.eigh(cov)
    axis = eigvecs[:, int(np.argmax(eigvals))]
    axis = axis / (np.linalg.norm(axis) + 1e-6)
    return axis.astype(np.float32), mu.astype(np.float32)


def mask_elongation(mask01: np.ndarray) -> float:
    ys, xs = np.where(mask01 > 0)
    if xs.size < 10:
        return 1.0
    pts = np.stack([xs, ys], axis=1).astype(np.float32)
    mu = pts.mean(axis=0)
    centered = pts - mu
    cov = (centered.T @ centered) / max(1.0, float(len(pts)))
    eigvals, _ = np.linalg.eigh(cov)
    lmin = float(max(1e-6, np.min(eigvals)))
    lmax = float(max(1e-6, np.max(eigvals)))
    return float(np.sqrt(lmax / lmin))


def mask_bbox_xyxy(mask01: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def crop_from_masks(
    full_bgr: np.ndarray,
    masks: List[np.ndarray],
    depth: Optional[np.ndarray],
    pad: int = 12,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if not masks:
        return full_bgr.copy(), None if depth is None else depth.copy()
    support = union_masks([(m > 0).astype(np.uint8) for m in masks])
    bb = mask_bbox_xyxy(support)
    if bb is None:
        return full_bgr.copy(), None if depth is None else depth.copy()
    h, w = full_bgr.shape[:2]
    x1, y1, x2, y2 = bb
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(w - 1, x2 + pad)
    y2 = min(h - 1, y2 + pad)
    crop_bgr = full_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
    crop_depth = None if depth is None else depth[y1 : y2 + 1, x1 : x2 + 1].copy()
    return crop_bgr, crop_depth


def _depth_summary_for_mask(mask01: np.ndarray, depth: Optional[np.ndarray]) -> Dict[str, Any]:
    if depth is None or depth.shape[:2] != mask01.shape[:2]:
        return {
            "depth_valid_ratio": None,
            "depth_median": None,
            "depth_mad": None,
            "depth_p10": None,
            "depth_p90": None,
        }
    m = mask01.astype(bool)
    total = int(m.sum())
    if total <= 0:
        return {
            "depth_valid_ratio": 0.0,
            "depth_median": None,
            "depth_mad": None,
            "depth_p10": None,
            "depth_p90": None,
        }
    vals = depth[m]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    if vals.size == 0:
        return {
            "depth_valid_ratio": 0.0,
            "depth_median": None,
            "depth_mad": None,
            "depth_p10": None,
            "depth_p90": None,
        }
    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    p10, p90 = np.percentile(vals, [10, 90])
    return {
        "depth_valid_ratio": float(vals.size / float(total + 1e-6)),
        "depth_median": med,
        "depth_mad": mad,
        "depth_p10": float(p10),
        "depth_p90": float(p90),
    }


def _skel_morph(mask01: np.ndarray) -> np.ndarray:
    """Morphological thinning skeleton of a binary mask."""
    img = (mask01 > 0).astype(np.uint8).copy()
    if int(img.sum()) == 0:
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


def _skel_endpoint_count(skel01: np.ndarray) -> int:
    """Count skeleton endpoints (pixels with <= 1 neighbour in 3×3)."""
    s = (skel01 > 0).astype(np.uint8)
    if int(s.sum()) == 0:
        return 0
    neigh = cv2.filter2D(s, cv2.CV_16S, np.ones((3, 3), dtype=np.uint8))
    return int(((s > 0) & (neigh <= 2)).sum())


def _width_stats_from_mask(mask01: np.ndarray) -> Tuple[float, float]:
    """Return (mean_half_width_px, width_cv) via distance-transform along skeleton."""
    m = (mask01 > 0).astype(np.uint8)
    if int(m.sum()) == 0:
        return 0.0, 1.0
    dist = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    skel = _skel_morph(m)
    vals = dist[skel > 0]
    if vals.size == 0:
        vals = dist[m > 0]
    if vals.size == 0:
        return 0.0, 1.0
    mean_hw = float(np.mean(vals))
    return mean_hw, float(np.std(vals) / (mean_hw + 1e-6))


def build_mask_geometry_stats(mask_candidates: List[np.ndarray], depth: Optional[np.ndarray]) -> List[Dict[str, Any]]:
    masks = [keep_largest_cc((m > 0).astype(np.uint8)) for m in mask_candidates if mask_area(m) > 0]
    if not masks:
        return []
    support = union_masks(masks)
    axis, mu = pca_axis_from_mask(support)
    support_area = float(mask_area(support) + 1e-6)

    stats: List[Dict[str, Any]] = []
    for idx, m in enumerate(masks):
        c = centroid(m)
        bb = mask_bbox_xyxy(m)
        a = float(mask_area(m))
        if bb is None:
            bw = 0
            bh = 0
            bdiag = 0.0
        else:
            bw = int(bb[2] - bb[0] + 1)
            bh = int(bb[3] - bb[1] + 1)
            bdiag = float(np.hypot(float(bw), float(bh)))
        elong = float(mask_elongation(m))
        skel = _skel_morph(m)
        skel_len = float(int(skel.sum()))
        ep_count = _skel_endpoint_count(skel)
        mean_hw, width_cv = _width_stats_from_mask(m)
        stat: Dict[str, Any] = {
            "id": idx,
            "area_px": int(a),
            "area_ratio_vs_support": float(a / support_area),
            "centroid_xy": [float(c[0]), float(c[1])],
            "axis_projection": float((c - mu) @ axis),
            "bbox_xyxy": None if bb is None else [int(v) for v in bb],
            "bbox_w": int(bw),
            "bbox_h": int(bh),
            "bbox_diag": bdiag,
            "elongation": elong,
            "num_components": _num_components(m),
            "skeleton_length_px": round(skel_len, 1),
            "endpoint_count": ep_count,
            "mean_half_width_px": round(mean_hw, 2),
            "width_cv": round(width_cv, 3),
        }
        stat.update(_depth_summary_for_mask(m, depth))
        stats.append(stat)

    # Second pass: terminality = normalised |axis_projection| across all candidates
    max_abs_proj = max(abs(float(s["axis_projection"])) for s in stats) if stats else 1.0
    max_abs_proj = max(max_abs_proj, 1.0)
    for s in stats:
        s["terminality"] = round(float(abs(float(s["axis_projection"])) / max_abs_proj), 3)

    return stats


def _depth_values(mask01: np.ndarray, depth: Optional[np.ndarray]) -> np.ndarray:
    if depth is None or depth.shape[:2] != mask01.shape[:2]:
        return np.zeros((0,), dtype=np.float32)
    vals = depth[mask01.astype(bool)]
    vals = vals[np.isfinite(vals) & (vals > 0)]
    return vals.astype(np.float32)


def _canonical_cc(mask01: np.ndarray, preserve_components: bool, rel_min: float = 0.06, pix_min: int = 120) -> np.ndarray:
    m = (mask01 > 0).astype(np.uint8)
    if preserve_components:
        return _keep_significant_components(m, rel_min=rel_min, pix_min=pix_min)
    return keep_largest_cc(m)


def _stabilize_mask_by_depth(
    mask01: np.ndarray,
    depth: Optional[np.ndarray],
    preserve_components: bool = False,
    rel_min: float = 0.06,
    pix_min: int = 120,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    m = _canonical_cc((mask01 > 0).astype(np.uint8), preserve_components=preserve_components, rel_min=rel_min, pix_min=pix_min)
    vals = _depth_values(m, depth)
    if vals.size < 20:
        return m, {"applied": False, "kept_ratio": 1.0}

    med = float(np.median(vals))
    mad = float(np.median(np.abs(vals - med)))
    tol = max(0.01, 3.5 * mad)
    keep = np.logical_and(m > 0, np.isfinite(depth), np.abs(depth.astype(np.float32) - med) <= tol).astype(np.uint8)
    keep = _canonical_cc(keep, preserve_components=preserve_components, rel_min=rel_min, pix_min=pix_min)
    kept_ratio = float(mask_area(keep)) / float(mask_area(m) + 1e-6)
    if mask_area(keep) > 0 and kept_ratio >= 0.60:
        return keep, {"applied": True, "kept_ratio": kept_ratio, "depth_median": med, "depth_mad": mad, "depth_tol": tol}
    return m, {"applied": False, "kept_ratio": 1.0, "depth_median": med, "depth_mad": mad, "depth_tol": tol}


def _repair_pair_overlap(
    mask_a: np.ndarray,
    mask_b: np.ndarray,
    depth: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    a = keep_largest_cc((mask_a > 0).astype(np.uint8))
    b = keep_largest_cc((mask_b > 0).astype(np.uint8))
    overlap = np.logical_and(a > 0, b > 0)
    overlap_px = int(overlap.sum())
    if overlap_px == 0:
        return a, b, {"overlap_px_before": 0, "overlap_px_after": 0, "reassigned_px": 0}

    a_only = np.logical_and(a > 0, np.logical_not(overlap)).astype(np.uint8)
    b_only = np.logical_and(b > 0, np.logical_not(overlap)).astype(np.uint8)
    ca = centroid(a_only if mask_area(a_only) > 0 else a)
    cb = centroid(b_only if mask_area(b_only) > 0 else b)
    da = None
    db = None
    va = _depth_values(a_only if mask_area(a_only) > 0 else a, depth)
    vb = _depth_values(b_only if mask_area(b_only) > 0 else b, depth)
    if va.size > 0:
        da = float(np.median(va))
    if vb.size > 0:
        db = float(np.median(vb))

    ys, xs = np.where(overlap)
    a_new = a.copy()
    b_new = b.copy()
    for x, y in zip(xs.tolist(), ys.tolist()):
        p = np.array([float(x), float(y)], dtype=np.float32)
        ca_cost = float(np.sum((p - ca) ** 2))
        cb_cost = float(np.sum((p - cb) ** 2))
        if depth is not None and 0 <= y < depth.shape[0] and 0 <= x < depth.shape[1]:
            dv = float(depth[y, x])
            if np.isfinite(dv) and dv > 0:
                if da is not None:
                    ca_cost += 2500.0 * abs(dv - da)
                if db is not None:
                    cb_cost += 2500.0 * abs(dv - db)
        if ca_cost <= cb_cost:
            b_new[y, x] = 0
        else:
            a_new[y, x] = 0

    a_new = keep_largest_cc(a_new)
    b_new = keep_largest_cc(b_new)
    overlap_after = int(np.logical_and(a_new > 0, b_new > 0).sum())
    return a_new, b_new, {
        "overlap_px_before": overlap_px,
        "overlap_px_after": overlap_after,
        "reassigned_px": int(overlap_px - overlap_after),
    }


def _lift_readiness_for_part(
    mask01: np.ndarray,
    depth: Optional[np.ndarray],
    sibling_mask: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    m = (mask01 > 0).astype(np.uint8)
    area = float(mask_area(m))
    n_comp = _num_components(m)
    depth_stats = _depth_summary_for_mask(m, depth)
    overlap_iou = float(mask_iou(m, sibling_mask)) if sibling_mask is not None and mask_area(sibling_mask) > 0 else 0.0

    depth_valid = float(depth_stats["depth_valid_ratio"]) if depth_stats["depth_valid_ratio"] is not None else 0.0
    depth_mad = float(depth_stats["depth_mad"]) if depth_stats["depth_mad"] is not None else 0.05
    depth_p10 = float(depth_stats["depth_p10"]) if depth_stats["depth_p10"] is not None else None
    depth_p90 = float(depth_stats["depth_p90"]) if depth_stats["depth_p90"] is not None else None
    if depth_p10 is not None and depth_p90 is not None and depth_p90 > depth_p10:
        depth_span = float(depth_p90 - depth_p10)
        depth_stability = float(np.clip(1.0 - (depth_mad / (depth_span + 1e-6)), 0.0, 1.0))
    else:
        depth_stability = float(np.clip(1.0 - (depth_mad / 0.02), 0.0, 1.0))

    area_score = float(np.clip(area / 250.0, 0.0, 1.0))
    component_score = 1.0 if n_comp <= 1 else float(np.clip(1.0 / float(n_comp), 0.0, 1.0))
    overlap_clean = float(np.clip(1.0 - overlap_iou, 0.0, 1.0))

    score = float(
        0.30 * depth_valid
        + 0.25 * depth_stability
        + 0.20 * overlap_clean
        + 0.15 * component_score
        + 0.10 * area_score
    )
    return {
        "score": score,
        "ready": bool(score >= 0.60),
        "area_px": int(area),
        "num_components": int(n_comp),
        "overlap_iou": overlap_iou,
        "depth_valid_ratio": depth_stats["depth_valid_ratio"],
        "depth_median": depth_stats["depth_median"],
        "depth_mad": depth_stats["depth_mad"],
        "depth_p10": depth_stats["depth_p10"],
        "depth_p90": depth_stats["depth_p90"],
        "components_clean": component_score,
        "overlap_clean": overlap_clean,
        "depth_stability": depth_stability,
        "area_score": area_score,
    }


def consolidate_canonical_parts(
    parts: Dict[str, np.ndarray],
    depth: Optional[np.ndarray],
    tool_mode: bool,
) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    refined: Dict[str, np.ndarray] = {}
    for k, v in parts.items():
        if mask_area(v) <= 0:
            continue
        pname = str(k)
        preserve = (not tool_mode) and pname in {"whole", "contact_region"}
        pix_min = 80 if pname == "whole" else 40
        refined[pname] = _canonical_cc((v > 0).astype(np.uint8), preserve_components=preserve, rel_min=0.04, pix_min=pix_min)
    qc: Dict[str, Any] = {"tool_mode": bool(tool_mode)}

    if tool_mode:
        handle = refined.get("handle")
        work_end = refined.get("work_end")
        whole_hint = refined.get("whole")
        if handle is None or work_end is None:
            return refined, qc

        handle, dbg_h = _stabilize_mask_by_depth(handle, depth)
        work_end, dbg_w = _stabilize_mask_by_depth(work_end, depth)
        handle, work_end, overlap_dbg = _repair_pair_overlap(handle, work_end, depth)
        whole = compose_tool_whole(handle=handle, work_end=work_end, whole_hint=whole_hint)
        refined = {"whole": whole, "handle": handle, "work_end": work_end}
        qc.update(
            {
                "depth_stabilization": {"handle": dbg_h, "work_end": dbg_w},
                "overlap_repair": overlap_dbg,
            }
        )
        qc["lift_readiness"] = {
            "handle": _lift_readiness_for_part(handle, depth=depth, sibling_mask=work_end),
            "work_end": _lift_readiness_for_part(work_end, depth=depth, sibling_mask=handle),
            "whole": _lift_readiness_for_part(whole, depth=depth, sibling_mask=None),
        }
        qc["object_lift_readiness"] = float(
            min(
                qc["lift_readiness"]["handle"]["score"],
                qc["lift_readiness"]["work_end"]["score"],
            )
        )
        return refined, qc

    whole = refined.get("whole")
    contact = refined.get("contact_region")
    if whole is None and contact is None:
        return refined, qc
    if whole is None:
        whole = contact.copy()
    if contact is None:
        contact = whole.copy()

    contact, dbg_c = _stabilize_mask_by_depth(contact, depth, preserve_components=True, rel_min=0.04, pix_min=40)
    whole = union_masks([_canonical_cc(whole, preserve_components=True, rel_min=0.04, pix_min=80), contact])
    whole = _canonical_cc(whole, preserve_components=True, rel_min=0.04, pix_min=80)
    refined = {"whole": whole, "contact_region": contact}
    qc.update({"depth_stabilization": {"contact_region": dbg_c}})
    qc["lift_readiness"] = {
        "contact_region": _lift_readiness_for_part(contact, depth=depth, sibling_mask=None),
        "whole": _lift_readiness_for_part(whole, depth=depth, sibling_mask=None),
    }
    qc["object_lift_readiness"] = float(qc["lift_readiness"]["contact_region"]["score"])
    return refined, qc


def colorize_depth(depth: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if depth is None:
        return None
    d = depth.astype(np.float32)
    valid = np.isfinite(d) & (d > 0)
    if int(valid.sum()) < 10:
        return None
    dv = d[valid]
    lo, hi = np.percentile(dv, [5, 95])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(dv))
        hi = float(np.max(dv))
        if hi <= lo:
            return None
    norm = np.zeros_like(d, dtype=np.float32)
    norm[valid] = np.clip((d[valid] - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    vis = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    vis[~valid] = 0
    return vis


def encode_image_to_data_url(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("Failed to encode image for VLM prompt.")
    b64 = base64.b64encode(buf.tobytes()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def extract_json_block(text: str) -> Optional[Any]:
    t = (text or "").strip()
    if not t:
        return None
    if t.startswith("```"):
        t = re.sub(r"^```[a-zA-Z0-9_-]*\n?", "", t)
        t = re.sub(r"\n?```$", "", t).strip()
    for candidate in (t,):
        try:
            return json.loads(candidate)
        except Exception:
            pass
    m_obj = re.search(r"\{[\s\S]*\}", t)
    if m_obj:
        try:
            return json.loads(m_obj.group(0))
        except Exception:
            pass
    m_arr = re.search(r"\[[\s\S]*\]", t)
    if m_arr:
        try:
            return json.loads(m_arr.group(0))
        except Exception:
            pass
    return None


def clean_mask_ids(raw: Any, n_masks: int) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, (int, np.integer, float, np.floating, str)):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    out: List[int] = []
    for v in raw:
        try:
            i = int(v)
        except Exception:
            continue
        if 0 <= i < n_masks:
            out.append(i)
    return sorted(set(out))


def canonical_part_name(name: str) -> str:
    t = (name or "").lower().strip().replace("-", "_").replace(" ", "_")
    aliases = {
        "grip": "handle",
        "grip_part": "handle",
        "hold": "handle",
        "tip": "work_end",
        "end": "work_end",
        "head": "work_end",
        "blade": "work_end",
        "contact": "contact_region",
        "contact_part": "contact_region",
        "contact_area": "contact_region",
        "contact_region": "contact_region",
        "all": "whole",
        "object": "whole",
    }
    return aliases.get(t, t)


def load_depth(depth_path: Optional[str]) -> Optional[np.ndarray]:
    if not depth_path or not os.path.exists(depth_path):
        return None
    try:
        d = np.load(depth_path).astype(np.float32)
        d[~np.isfinite(d)] = 0.0
        return d
    except Exception:
        return None
