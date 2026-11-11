#!/usr/bin/env python3
"""Compact SAM stage: DINO crop -> whole mask -> optional two-part split."""

from __future__ import annotations

import argparse
from contextlib import ExitStack
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

try:
    from segment_anything import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
    SAM1_AVAILABLE = True
except ImportError:
    SamAutomaticMaskGenerator = Any  # type: ignore[misc,assignment]
    SamPredictor = Any  # type: ignore[misc,assignment]
    sam_model_registry = {}
    SAM1_AVAILABLE = False

try:
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2
    from sam2.sam2_image_predictor import SAM2ImagePredictor
    SAM2_AVAILABLE = True
except ImportError:
    SAM2AutomaticMaskGenerator = Any  # type: ignore[misc,assignment]
    SAM2ImagePredictor = Any  # type: ignore[misc,assignment]
    build_sam2 = None  # type: ignore[assignment]
    SAM2_AVAILABLE = False

if __package__ in {None, ""}:
    _ROOT_DIR = Path(__file__).resolve().parents[2]
    if str(_ROOT_DIR) not in sys.path:
        sys.path.insert(0, str(_ROOT_DIR))

from regiomanip.env_utils import load_workspace_env
from regiomanip.region_grounding.dino import _resolve_image_path, run_dino

load_workspace_env()

REPO_ROOT = Path(__file__).resolve().parents[2]

SAM_BACKEND = os.environ.get("SAM_BACKEND", "auto")
SAM_MODEL_TYPE = os.environ.get("SAM_MODEL_TYPE", "vit_h")
SAM_WEIGHTS = os.environ.get("SAM_WEIGHTS", "sam_vit_h_4b8939.pth")
SAM2_MODEL_CFG = os.environ.get("SAM2_MODEL_CFG", "")
SAM2_APPLY_POSTPROCESSING = os.environ.get("SAM2_APPLY_POSTPROCESSING", "0").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
}

KEEP_MAX_DEFAULT = int(os.environ.get("KEEP_MAX", "1"))
MASK_PIX_MIN = int(os.environ.get("MASK_PIX_MIN", "120"))
TINY_RATIO_MIN = float(os.environ.get("TINY_RATIO_MIN", "0.003"))
NEARFULL_RATIO_MAX = float(os.environ.get("NEARFULL_RATIO_MAX", "0.98"))
BORDER_MARGIN = int(os.environ.get("BORDER_MARGIN", "2"))
GRID_TILE = int(os.environ.get("GRID_TILE", "256"))
GRID_COLS = int(os.environ.get("GRID_COLS", "4"))
OVERLAY_ALPHA = float(os.environ.get("OVERLAY_ALPHA", "0.55"))

DEFAULT_DATA_PATH = "./data"
DEFAULT_FRAME_NO = 13
DEFAULT_INSTRUCTION = "scrape the white putty"
DEFAULT_OUT_DIR = "./data/dino_out"
DEFAULT_DINO_JSON = os.path.join(DEFAULT_OUT_DIR, "dino_summary.json")
DEFAULT_SAM_OUT_DIR = "./data/sam_out"

TOOL_LIKE_KEYWORDS = {
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
    "gripper",
}
WHITE_FLAT_TARGET_KEYWORDS = {
    "putty",
    "clay",
    "paper",
    "tape",
    "label",
    "sticker",
    "sheet",
}


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def choose_sam_device() -> str:
    raw = str(os.environ.get("SAM_DEVICE", "cpu")).strip().lower()
    if raw == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _resolve_local_or_repo_path(raw: str) -> Optional[str]:
    candidate_text = str(raw or "").strip()
    if not candidate_text:
        return None
    direct = Path(candidate_text).expanduser()
    candidates: List[Path] = [direct]
    if not direct.is_absolute():
        candidates.extend([Path.cwd() / direct, REPO_ROOT / direct])
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return None


def _looks_like_sam2_artifact(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return "sam2" in lowered or "hiera" in lowered


def _require_backend_available(backend: str) -> None:
    if backend == "sam1" and not SAM1_AVAILABLE:
        raise ImportError(
            "SAM backend 'sam1' requested, but segment_anything is not installed."
        )
    if backend == "sam2" and not SAM2_AVAILABLE:
        raise ImportError(
            "SAM backend 'sam2' requested, but sam2 is not installed. "
            "Install facebookresearch/sam2 first, e.g. `pip install -e \".[notebooks]\"` from the SAM2 repo."
        )


def _resolve_sam_backend(
    *,
    requested_backend: str,
    weights: str,
    model_type: str,
    sam2_model_cfg: str,
) -> str:
    requested = str(requested_backend or "auto").strip().lower()
    if requested not in {"auto", "sam1", "sam2"}:
        raise ValueError(f"Unsupported SAM_BACKEND={requested_backend!r}; expected one of: auto, sam1, sam2")
    if requested in {"sam1", "sam2"}:
        _require_backend_available(requested)
        return requested

    sam2_cues = any(
        _looks_like_sam2_artifact(value)
        for value in (weights, model_type, sam2_model_cfg)
        if str(value or "").strip()
    )
    if sam2_cues:
        _require_backend_available("sam2")
        return "sam2"

    _require_backend_available("sam1")
    return "sam1"


def _infer_sam2_cfg_from_checkpoint(weights: str) -> Optional[str]:
    lower_name = Path(str(weights or "")).name.lower()
    mapping = {
        "sam2.1_hiera_large": "configs/sam2.1/sam2.1_hiera_l.yaml",
        "sam2.1_hiera_base_plus": "configs/sam2.1/sam2.1_hiera_b+.yaml",
        "sam2.1_hiera_small": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "sam2.1_hiera_tiny": "configs/sam2.1/sam2.1_hiera_t.yaml",
        "sam2_hiera_large": "sam2_hiera_l.yaml",
        "sam2_hiera_base_plus": "sam2_hiera_b+.yaml",
        "sam2_hiera_small": "sam2_hiera_s.yaml",
        "sam2_hiera_tiny": "sam2_hiera_t.yaml",
    }
    for key, value in mapping.items():
        if key in lower_name:
            return value
    return None


def _resolve_sam2_config(model_cfg: str, *, weights: str) -> str:
    raw = str(model_cfg or "").strip()
    if raw:
        return raw
    inferred = _infer_sam2_cfg_from_checkpoint(weights)
    if inferred:
        return inferred
    raise FileNotFoundError(
        "SAM2 config path is empty. Set SAM2_MODEL_CFG to the official config name "
        "(for example `configs/sam2.1/sam2.1_hiera_l.yaml`) or to a readable local YAML path."
    )


def _build_sam2_model(*, model_cfg: str, checkpoint: str, device: str) -> Any:
    if build_sam2 is None:
        raise ImportError("SAM2 backend requested but build_sam2 is unavailable.")

    local_cfg = _resolve_local_or_repo_path(model_cfg)
    if local_cfg and Path(local_cfg).is_file():
        try:
            from hydra import initialize_config_dir
            from hydra.core.global_hydra import GlobalHydra
        except ImportError:
            pass
        else:
            cfg_path = Path(local_cfg)
            GlobalHydra.instance().clear()
            with initialize_config_dir(version_base=None, config_dir=str(cfg_path.parent)):
                return build_sam2(
                    cfg_path.name,
                    checkpoint,
                    device=device,
                    apply_postprocessing=SAM2_APPLY_POSTPROCESSING,
                )

    return build_sam2(
        model_cfg,
        checkpoint,
        device=device,
        apply_postprocessing=SAM2_APPLY_POSTPROCESSING,
    )


def _sam_backend_for_model(
    sam_model: Optional[Any],
    *,
    requested_backend: str,
    weights: str,
    model_type: str,
    sam2_model_cfg: str,
) -> str:
    if sam_model is not None:
        return str(getattr(sam_model, "_sam_backend", "") or _resolve_sam_backend(
            requested_backend=requested_backend,
            weights=weights,
            model_type=model_type,
            sam2_model_cfg=sam2_model_cfg,
        ))
    return _resolve_sam_backend(
        requested_backend=requested_backend,
        weights=weights,
        model_type=model_type,
        sam2_model_cfg=sam2_model_cfg,
    )


def _sam_inference_context(*, backend: str, device: str) -> ExitStack:
    stack = ExitStack()
    stack.enter_context(torch.inference_mode())
    if backend == "sam2" and device == "cuda":
        stack.enter_context(torch.autocast("cuda", dtype=torch.bfloat16))
    return stack


class _MaskGeneratorAdapter:
    def __init__(self, generator: Any, *, backend: str, device: str) -> None:
        self._generator = generator
        self._sam_backend = backend
        self._inference_device = device

    def generate(self, image: np.ndarray) -> Any:
        with _sam_inference_context(backend=self._sam_backend, device=self._inference_device):
            return self._generator.generate(image)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._generator, name)


class _PredictorAdapter:
    def __init__(self, predictor: Any, *, backend: str, device: str) -> None:
        self._predictor = predictor
        self._sam_backend = backend
        self._inference_device = device

    def set_image(self, image: np.ndarray) -> Any:
        with _sam_inference_context(backend=self._sam_backend, device=self._inference_device):
            return self._predictor.set_image(image)

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        with _sam_inference_context(backend=self._sam_backend, device=self._inference_device):
            return self._predictor.predict(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._predictor, name)


def _resolve_sam_weights(weights: str) -> str:
    raw = str(weights or "").strip()
    if not raw:
        raise FileNotFoundError("SAM checkpoint path is empty. Set SAM_WEIGHTS or place sam_vit_h_4b8939.pth in the repo root.")

    direct = Path(raw).expanduser()
    basename = direct.name
    candidates: List[Path] = [direct]
    if not direct.is_absolute():
        candidates.extend([Path.cwd() / direct, REPO_ROOT / direct])
    elif basename:
        # Recover from stale placeholder env vars like /path/to/foo.pt by
        # also checking repo-local checkpoint locations for the same filename.
        candidates.extend([
            Path.cwd() / basename,
            Path.cwd() / "checkpoints" / basename,
            REPO_ROOT / basename,
            REPO_ROOT / "checkpoints" / basename,
        ])

    seen = set()
    unique_candidates: List[Path] = []
    for candidate in candidates:
        key = str(candidate.resolve(strict=False))
        if key in seen:
            continue
        seen.add(key)
        unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.is_file():
            return str(candidate.resolve())

    looked_in = ", ".join(str(path.resolve(strict=False)) for path in unique_candidates)
    raise FileNotFoundError(
        "SAM checkpoint not found. "
        f"Looked for: {looked_in}. "
        "Set SAM_WEIGHTS to the checkpoint path or place sam_vit_h_4b8939.pth in the repo root."
    )


def keep_significant_cc(mask01: np.ndarray, rel_min: float = 0.08, pix_min: int = 120) -> np.ndarray:
    mask = (mask01 > 0).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    if n <= 2:
        return mask
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    largest = int(areas.max())
    threshold = max(int(pix_min), int(round(float(largest) * float(rel_min))))
    keep = [i + 1 for i, area in enumerate(areas.tolist()) if int(area) >= threshold]
    if not keep:
        keep = [1 + int(np.argmax(areas))]
    return np.isin(labels, keep).astype(np.uint8)


def touches_border(mask01: np.ndarray, margin: int) -> bool:
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return True
    h, w = mask01.shape[:2]
    return bool(
        xs.min() <= margin
        or ys.min() <= margin
        or xs.max() >= w - 1 - margin
        or ys.max() >= h - 1 - margin
    )


def mask_bbox_xyxy(mask01: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask01 > 0)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    if ix2 < ix1 or iy2 < iy1:
        return 0.0
    inter = (ix2 - ix1 + 1) * (iy2 - iy1 + 1)
    area_a = (ax2 - ax1 + 1) * (ay2 - ay1 + 1)
    area_b = (bx2 - bx1 + 1) * (by2 - by1 + 1)
    return float(inter) / float(area_a + area_b - inter + 1e-6)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    a = mask_a.astype(bool)
    b = mask_b.astype(bool)
    inter = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    return float(inter) / float(union + 1e-6)


def overlay_mask(bgr: np.ndarray, mask01: np.ndarray, alpha: float = OVERLAY_ALPHA) -> np.ndarray:
    out = bgr.copy()
    red = np.zeros_like(out)
    red[..., 2] = 255
    keep = mask01.astype(bool)
    out[keep] = (out[keep] * (1.0 - alpha) + red[keep] * alpha).astype(np.uint8)
    return out


def make_grid(bgr_crop: np.ndarray, masks_crop01: List[np.ndarray]) -> np.ndarray:
    if bgr_crop is None or bgr_crop.size == 0:
        return np.zeros((GRID_TILE, GRID_TILE, 3), dtype=np.uint8)
    if not masks_crop01:
        return cv2.resize(bgr_crop, (GRID_TILE, GRID_TILE), interpolation=cv2.INTER_LINEAR)

    tiles: List[np.ndarray] = []
    for i, mask in enumerate(masks_crop01):
        vis = overlay_mask(bgr_crop, mask)
        h, w = vis.shape[:2]
        scale = GRID_TILE / float(max(h, w))
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        resized = cv2.resize(vis, (nw, nh), interpolation=cv2.INTER_LINEAR)
        tile = np.zeros((GRID_TILE, GRID_TILE, 3), dtype=np.uint8)
        oy = (GRID_TILE - nh) // 2
        ox = (GRID_TILE - nw) // 2
        tile[oy : oy + nh, ox : ox + nw] = resized
        cv2.putText(tile, f"id {i}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
        tiles.append(tile)

    rows = int(np.ceil(len(tiles) / float(GRID_COLS)))
    grid = np.zeros((rows * GRID_TILE, GRID_COLS * GRID_TILE, 3), dtype=np.uint8)
    for i, tile in enumerate(tiles):
        row, col = divmod(i, GRID_COLS)
        grid[row * GRID_TILE : (row + 1) * GRID_TILE, col * GRID_TILE : (col + 1) * GRID_TILE] = tile
    return grid


def nms_by_bbox(masks: List[np.ndarray], iou_th: float) -> List[np.ndarray]:
    kept: List[np.ndarray] = []
    kept_boxes: List[Tuple[int, int, int, int]] = []
    for mask in masks:
        bb = mask_bbox_xyxy(mask)
        if bb is None:
            continue
        if any(bbox_iou(bb, old) > float(iou_th) for old in kept_boxes):
            continue
        kept.append(mask)
        kept_boxes.append(bb)
    return kept


def dedup_by_mask_iou(masks: List[np.ndarray], th: float) -> List[np.ndarray]:
    kept: List[np.ndarray] = []
    for mask in masks:
        if all(mask_iou(mask, old) <= float(th) for old in kept):
            kept.append(mask)
    return kept


def _name_has_any(name: str, keywords: Sequence[str]) -> bool:
    lower = str(name or "").lower()
    return any(word in lower for word in keywords)


def _is_tool_like_name(name: str) -> bool:
    return _name_has_any(name, tuple(TOOL_LIKE_KEYWORDS))


def _mask_elongation(mask01: np.ndarray) -> float:
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


def _looks_tool_like_shape(mask01: np.ndarray) -> bool:
    m = keep_significant_cc((mask01 > 0).astype(np.uint8), rel_min=0.03, pix_min=80)
    area = int(m.sum())
    if area < 500:
        return False
    bb = mask_bbox_xyxy(m)
    if bb is None:
        return False
    bw = int(bb[2] - bb[0] + 1)
    bh = int(bb[3] - bb[1] + 1)
    bbox_aspect = float(max(bw / max(bh, 1), bh / max(bw, 1)))
    elong = float(_mask_elongation(m))
    fill_ratio = float(area) / float(max(bw * bh, 1))
    # Broad-but-elongated utensils/tools (e.g. spoon, ladle, spatula) often have
    # moderate bbox aspect ratio but high PCA elongation and a sparse fill ratio.
    return bool(
        elong >= 3.2
        or (bbox_aspect >= 2.1 and fill_ratio <= 0.72)
        or (elong >= 2.6 and fill_ratio <= 0.45)
    )


# Manipulation verbs that signal "this object will be grasped and used as a tool"
# even when the object name is generic / novel.
_MANIP_VERBS: frozenset = frozenset({
    "grip", "grasp", "grab", "hold", "pick up", "pickup", "lift",
    "use", "scrape", "roll", "hammer", "tighten", "loosen", "screw",
    "unscrew", "drive", "paint", "cut", "slice", "pry", "turn", "twist",
    "push", "pull",
})


def _looks_tool_like_shape_relaxed(mask01: np.ndarray) -> bool:
    """More permissive than _looks_tool_like_shape.

    Used together with manipulation-verb intent to catch head-dominant tools
    (hammers, wrenches) whose whole mask is not highly elongated but whose
    multi-component structure or moderate aspect ratio still signals a tool.
    """
    m = keep_significant_cc((mask01 > 0).astype(np.uint8), rel_min=0.03, pix_min=80)
    area = int(m.sum())
    if area < 300:
        return False
    bb = mask_bbox_xyxy(m)
    if bb is None:
        return False
    bw = int(bb[2] - bb[0] + 1)
    bh = int(bb[3] - bb[1] + 1)
    bbox_aspect = float(max(bw / max(bh, 1), bh / max(bw, 1)))
    elong = float(_mask_elongation(m))
    fill_ratio = float(area) / float(max(bw * bh, 1))
    n_comp, _, _, _ = cv2.connectedComponentsWithStats(m)
    n_comp = max(0, n_comp - 1)
    return bool(
        elong >= 2.2
        or (bbox_aspect >= 1.7 and fill_ratio <= 0.80)
        or (elong >= 1.8 and fill_ratio <= 0.40)
        or n_comp >= 2  # disconnected components → likely head + shaft
    )


def _mask_box_overlap_stats(mask01: np.ndarray, ref_box_xyxy: Optional[Tuple[int, int, int, int]]) -> Tuple[float, float]:
    if ref_box_xyxy is None:
        return 0.0, 0.0
    x1, y1, x2, y2 = [int(v) for v in ref_box_xyxy]
    box_area = float(max((x2 - x1 + 1) * (y2 - y1 + 1), 1))
    mask = (mask01 > 0).astype(np.uint8)
    mask_area = float(max(int(mask.sum()), 1))
    clipped = np.zeros_like(mask, dtype=np.uint8)
    clipped[y1 : y2 + 1, x1 : x2 + 1] = mask[y1 : y2 + 1, x1 : x2 + 1]
    inter = float(int(clipped.sum()))
    return inter / mask_area, inter / box_area


def _union_masks(masks: List[np.ndarray]) -> Optional[np.ndarray]:
    if not masks:
        return None
    out = np.zeros_like(masks[0], dtype=np.uint8)
    for mask in masks:
        out = np.maximum(out, (mask > 0).astype(np.uint8))
    out = keep_significant_cc(out, rel_min=0.04, pix_min=80)
    return out if int(out.sum()) > 0 else None


def _tool_part_candidates(
    whole: np.ndarray,
    candidates: List[np.ndarray],
) -> List[np.ndarray]:
    whole_area = float(max(int(whole.sum()), 1))
    out: List[np.ndarray] = []
    for cand in candidates:
        inter = keep_significant_cc(np.logical_and(cand > 0, whole > 0).astype(np.uint8), rel_min=0.02, pix_min=40)
        inter_area = int(inter.sum())
        cand_area = int(np.count_nonzero(cand))
        if inter_area < 40 or cand_area <= 0:
            continue
        inside_ratio = float(inter_area) / float(cand_area)
        cover_ratio = float(inter_area) / whole_area
        if inside_ratio < 0.50 or cover_ratio < 0.03 or cover_ratio > 0.82:
            continue
        out.append(inter)
    return out


def _clip_box_local(box: Tuple[int, int, int, int], width: int, height: int) -> Optional[Tuple[int, int, int, int]]:
    x1, y1, x2, y2 = [int(v) for v in box]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 < x1 or y2 < y1:
        return None
    return x1, y1, x2, y2


def dino_box_in_crop(obj: Dict[str, Any]) -> Optional[Tuple[int, int, int, int]]:
    raw_box = obj.get("box_xyxy")
    crop_box = obj.get("crop_box_xyxy")
    if not (isinstance(raw_box, list) and len(raw_box) == 4 and isinstance(crop_box, list) and len(crop_box) == 4):
        return None
    rx1, ry1, rx2, ry2 = [int(v) for v in raw_box]
    cx1, cy1, cx2, cy2 = [int(v) for v in crop_box]
    width = cx2 - cx1 + 1
    height = cy2 - cy1 + 1
    return _clip_box_local((rx1 - cx1, ry1 - cy1, rx2 - cx1, ry2 - cy1), width, height)


def place_mask_into_full(
    mask_crop01: np.ndarray,
    full_hw: Tuple[int, int],
    crop_box_xyxy: Tuple[int, int, int, int],
) -> np.ndarray:
    h, w = full_hw
    x1, y1, x2, y2 = [int(v) for v in crop_box_xyxy]
    out = np.zeros((h, w), dtype=np.uint8)
    out[y1 : y2 + 1, x1 : x2 + 1] = (mask_crop01 > 0).astype(np.uint8)
    return out


def _extract_white_flat_component(
    crop_bgr: np.ndarray,
    object_name: str,
    ref_box_xyxy: Optional[Tuple[int, int, int, int]] = None,
) -> Optional[np.ndarray]:
    if crop_bgr is None or crop_bgr.size == 0 or not _name_has_any(object_name, tuple(WHITE_FLAT_TARGET_KEYWORDS)):
        return None
    b = crop_bgr[..., 0].astype(np.int16)
    g = crop_bgr[..., 1].astype(np.int16)
    r = crop_bgr[..., 2].astype(np.int16)
    white = (
        (r > 150)
        & (g > 150)
        & (b > 150)
        & (np.abs(r - g) < 24)
        & (np.abs(r - b) < 24)
        & (np.abs(g - b) < 24)
    ).astype(np.uint8)
    white = keep_significant_cc(white, rel_min=0.03, pix_min=80)
    if int(white.sum()) == 0:
        return None
    if ref_box_xyxy is not None:
        in_box_ratio, box_cover_ratio = _mask_box_overlap_stats(white, ref_box_xyxy)
        if in_box_ratio < 0.20 and box_cover_ratio < 0.15:
            return None
    return white


def _refine_white_flat_target_mask(mask01: np.ndarray, crop_bgr: np.ndarray, object_name: str) -> np.ndarray:
    refined = keep_significant_cc(mask01, rel_min=0.03, pix_min=80)
    white_comp = _extract_white_flat_component(crop_bgr, object_name)
    if white_comp is not None and int(white_comp.sum()) > int(refined.sum()) * 0.35:
        refined = keep_significant_cc(np.maximum(refined, white_comp), rel_min=0.03, pix_min=80)
    return refined


def build_sam_model(
    weights: str = SAM_WEIGHTS,
    model_type: str = SAM_MODEL_TYPE,
    backend: str = SAM_BACKEND,
    sam2_model_cfg: str = SAM2_MODEL_CFG,
):
    backend_name = _resolve_sam_backend(
        requested_backend=backend,
        weights=weights,
        model_type=model_type,
        sam2_model_cfg=sam2_model_cfg,
    )
    device = choose_sam_device()
    resolved_weights = _resolve_sam_weights(weights)
    if backend_name == "sam2":
        resolved_cfg = _resolve_sam2_config(sam2_model_cfg, weights=resolved_weights)
        print(
            f"[sam] loading backend=sam2 cfg={resolved_cfg} device={device} weights={resolved_weights}",
            flush=True,
        )
        sam = _build_sam2_model(model_cfg=resolved_cfg, checkpoint=resolved_weights, device=device)
    else:
        print(
            f"[sam] loading backend=sam1 model_type={model_type} device={device} weights={resolved_weights}",
            flush=True,
        )
        sam = sam_model_registry[model_type](checkpoint=resolved_weights)
        sam.to(device)
        sam.eval()
    setattr(sam, "_inference_device", device)
    setattr(sam, "_sam_backend", backend_name)
    return sam


def build_sam_generator(
    weights: str = SAM_WEIGHTS,
    model_type: str = SAM_MODEL_TYPE,
    backend: str = SAM_BACKEND,
    sam2_model_cfg: str = SAM2_MODEL_CFG,
    sam_model=None,
) -> Any:
    sam = sam_model if sam_model is not None else build_sam_model(
        weights=weights,
        model_type=model_type,
        backend=backend,
        sam2_model_cfg=sam2_model_cfg,
    )
    backend_name = _sam_backend_for_model(
        sam,
        requested_backend=backend,
        weights=weights,
        model_type=model_type,
        sam2_model_cfg=sam2_model_cfg,
    )
    generator_cls = SAM2AutomaticMaskGenerator if backend_name == "sam2" else SamAutomaticMaskGenerator
    generator = generator_cls(
        model=sam,
        points_per_side=int(os.environ.get("SAM_POINTS_PER_SIDE", "16")),
        pred_iou_thresh=float(os.environ.get("SAM_PRED_IOU_TH", "0.90")),
        stability_score_thresh=float(os.environ.get("SAM_STABILITY_TH", "0.92")),
        crop_n_layers=int(os.environ.get("SAM_CROP_N_LAYERS", "0")),
        min_mask_region_area=int(os.environ.get("SAM_MIN_REGION_AREA", "80")),
    )
    device = str(getattr(sam, "_inference_device", choose_sam_device()))
    setattr(generator, "_inference_device", device)
    setattr(generator, "_sam_backend", backend_name)
    return _MaskGeneratorAdapter(generator, backend=backend_name, device=device)


def build_inner_sam_generator(
    weights: str = SAM_WEIGHTS,
    model_type: str = SAM_MODEL_TYPE,
    backend: str = SAM_BACKEND,
    sam2_model_cfg: str = SAM2_MODEL_CFG,
    sam_model=None,
) -> Any:
    sam = sam_model if sam_model is not None else build_sam_model(
        weights=weights,
        model_type=model_type,
        backend=backend,
        sam2_model_cfg=sam2_model_cfg,
    )
    backend_name = _sam_backend_for_model(
        sam,
        requested_backend=backend,
        weights=weights,
        model_type=model_type,
        sam2_model_cfg=sam2_model_cfg,
    )
    generator_cls = SAM2AutomaticMaskGenerator if backend_name == "sam2" else SamAutomaticMaskGenerator
    generator = generator_cls(
        model=sam,
        points_per_side=int(os.environ.get("SAM_INNER_POINTS_PER_SIDE", "16")),
        pred_iou_thresh=float(os.environ.get("SAM_INNER_PRED_IOU_TH", "0.80")),
        stability_score_thresh=float(os.environ.get("SAM_INNER_STABILITY_TH", "0.85")),
        crop_n_layers=int(os.environ.get("SAM_INNER_CROP_N_LAYERS", "0")),
        min_mask_region_area=int(os.environ.get("SAM_INNER_MIN_REGION_AREA", "40")),
    )
    device = str(getattr(sam, "_inference_device", choose_sam_device()))
    setattr(generator, "_inference_device", device)
    setattr(generator, "_sam_backend", backend_name)
    return _MaskGeneratorAdapter(generator, backend=backend_name, device=device)


def _build_sam_predictor(
    weights: str = SAM_WEIGHTS,
    model_type: str = SAM_MODEL_TYPE,
    backend: str = SAM_BACKEND,
    sam2_model_cfg: str = SAM2_MODEL_CFG,
    sam_model=None,
) -> Any:
    sam = sam_model if sam_model is not None else build_sam_model(
        weights=weights,
        model_type=model_type,
        backend=backend,
        sam2_model_cfg=sam2_model_cfg,
    )
    backend_name = _sam_backend_for_model(
        sam,
        requested_backend=backend,
        weights=weights,
        model_type=model_type,
        sam2_model_cfg=sam2_model_cfg,
    )
    predictor_cls = SAM2ImagePredictor if backend_name == "sam2" else SamPredictor
    predictor = predictor_cls(sam)
    device = str(getattr(sam, "_inference_device", choose_sam_device()))
    setattr(predictor, "_inference_device", device)
    setattr(predictor, "_sam_backend", backend_name)
    return _PredictorAdapter(predictor, backend=backend_name, device=device)


def sam_mask_pool_in_crop(
    rgb_crop: np.ndarray,
    generator: Any,
    object_name: str = "",
    ref_box_xyxy: Optional[Tuple[int, int, int, int]] = None,
) -> List[np.ndarray]:
    if rgb_crop is None or rgb_crop.size == 0:
        return []
    raw = generator.generate(rgb_crop)
    if not raw:
        return []

    h, w = rgb_crop.shape[:2]
    crop_area = float(max(h * w, 1))
    kept: List[Tuple[float, np.ndarray]] = []
    for item in raw:
        seg = item.get("segmentation")
        if seg is None:
            continue
        mask = keep_significant_cc(np.asarray(seg, dtype=np.uint8), rel_min=0.03, pix_min=MASK_PIX_MIN)
        pix = int(mask.sum())
        if pix < MASK_PIX_MIN:
            continue
        ratio = pix / crop_area
        if ratio < TINY_RATIO_MIN or ratio > NEARFULL_RATIO_MAX:
            continue
        in_box_ratio, box_cover_ratio = _mask_box_overlap_stats(mask, ref_box_xyxy)
        if ref_box_xyxy is not None and in_box_ratio < 0.20 and box_cover_ratio < 0.10:
            continue
        if touches_border(mask, BORDER_MARGIN) and not _is_tool_like_name(object_name):
            continue
        score = float(item.get("predicted_iou", 0.0)) + 0.35 * box_cover_ratio + 0.20 * in_box_ratio
        kept.append((score, mask))

    kept.sort(key=lambda pair: pair[0], reverse=True)
    return [mask for _, mask in kept]


def keep_main_part_masks(
    masks: List[np.ndarray],
    ref_box_xyxy: Optional[Tuple[int, int, int, int]],
    keep_max: int,
    object_name: str = "",
) -> List[np.ndarray]:
    ranked: List[Tuple[float, np.ndarray]] = []
    for mask in masks:
        area = float(int(mask.sum()))
        in_box_ratio, box_cover_ratio = _mask_box_overlap_stats(mask, ref_box_xyxy)
        score = 0.50 * box_cover_ratio + 0.30 * in_box_ratio + 0.20 * min(area / 5000.0, 1.0)
        ranked.append((score, keep_significant_cc(mask, rel_min=0.03, pix_min=MASK_PIX_MIN)))
    ranked.sort(key=lambda pair: pair[0], reverse=True)
    return [mask for _, mask in ranked[: max(1, int(keep_max))]]


def _inner_masks_from_whole(
    rgb_crop: np.ndarray,
    whole_mask: np.ndarray,
    generator: Any,
    max_raw_masks: Optional[int] = None,
) -> List[np.ndarray]:
    if rgb_crop is None or rgb_crop.size == 0 or whole_mask is None or int(whole_mask.sum()) <= 0:
        return []
    raw = generator.generate(rgb_crop)
    if not raw:
        return []

    whole = keep_significant_cc((whole_mask > 0).astype(np.uint8), rel_min=0.03, pix_min=80)
    whole_area = float(max(int(whole.sum()), 1))
    whole_bb = mask_bbox_xyxy(whole)
    kept: List[np.ndarray] = []
    for item in raw:
        seg = item.get("segmentation")
        if seg is None:
            continue
        cand = (np.asarray(seg, dtype=np.uint8) > 0).astype(np.uint8)
        cand_area = int(cand.sum())
        if cand_area <= 0:
            continue
        inter = keep_significant_cc(np.logical_and(cand > 0, whole > 0).astype(np.uint8), rel_min=0.02, pix_min=40)
        inter_area = int(inter.sum())
        if inter_area < 40:
            continue
        inside_ratio = float(inter_area) / float(cand_area)
        area_ratio = float(inter_area) / whole_area
        if inside_ratio < 0.75:
            continue
        if area_ratio < 0.02 or area_ratio > 0.70:
            continue
        if mask_iou(inter, whole) > 0.85:
            continue
        inter_bb = mask_bbox_xyxy(inter)
        if inter_bb is not None and whole_bb is not None and bbox_iou(inter_bb, whole_bb) > 0.85 and area_ratio > 0.50:
            continue
        kept.append(inter)

    kept = dedup_by_mask_iou(nms_by_bbox(kept, iou_th=0.60), th=0.75)
    kept.sort(key=lambda m: int(m.sum()), reverse=True)
    if max_raw_masks is not None:
        return kept[: max(0, int(max_raw_masks))]
    return kept


def _debug_inner_masks_from_whole(
    rgb_crop: np.ndarray,
    whole_mask: np.ndarray,
    generator: Any,
    max_masks: int,
) -> List[np.ndarray]:
    if rgb_crop is None or rgb_crop.size == 0 or whole_mask is None or int(whole_mask.sum()) <= 0:
        return []
    raw = generator.generate(rgb_crop)
    if not raw:
        return []

    whole = keep_significant_cc((whole_mask > 0).astype(np.uint8), rel_min=0.03, pix_min=80)
    whole_area = float(max(int(whole.sum()), 1))
    kept: List[np.ndarray] = []
    for item in raw:
        seg = item.get("segmentation")
        if seg is None:
            continue
        cand = (np.asarray(seg, dtype=np.uint8) > 0).astype(np.uint8)
        cand_area = int(cand.sum())
        if cand_area <= 0:
            continue
        inter = keep_significant_cc(np.logical_and(cand > 0, whole > 0).astype(np.uint8), rel_min=0.01, pix_min=30)
        inter_area = int(inter.sum())
        if inter_area < 30:
            continue
        area_ratio = float(inter_area) / whole_area
        if area_ratio < 0.01 or area_ratio > 0.98:
            continue
        kept.append(inter)

    kept.sort(key=lambda m: int(m.sum()), reverse=True)
    return kept[: max(0, int(max_masks))]


def _select_best_inner_masks(masks: List[np.ndarray], max_masks: int = 2) -> List[np.ndarray]:
    max_masks = max(2, int(max_masks))
    if len(masks) <= max_masks:
        return masks

    if max_masks != 2:
        return masks[:max_masks]

    best_pair: Optional[Tuple[int, int]] = None
    best_score = -1e9
    for i in range(len(masks)):
        for j in range(i + 1, len(masks)):
            a = masks[i]
            b = masks[j]
            overlap = mask_iou(a, b)
            if overlap > 0.35:
                continue
            union = np.logical_or(a > 0, b > 0).astype(np.uint8)
            score = float(int(union.sum())) - 0.25 * float(min(int(a.sum()), int(b.sum())))
            if score > best_score:
                best_score = score
                best_pair = (i, j)
    if best_pair is not None:
        return [masks[best_pair[0]], masks[best_pair[1]]]
    return masks[:2]


def _best_box_prompt_mask(
    predictor: Any,
    crop_rgb: np.ndarray,
    ref_box_xyxy: Optional[Tuple[int, int, int, int]],
) -> Optional[np.ndarray]:
    if crop_rgb is None or crop_rgb.size == 0 or ref_box_xyxy is None:
        return None
    predictor.set_image(crop_rgb)
    input_box = np.array(ref_box_xyxy, dtype=np.float32)
    masks, scores, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box,
        multimask_output=True,
    )
    if masks is None or len(masks) == 0:
        return None

    best_mask: Optional[np.ndarray] = None
    best_score = -1e9
    for idx in range(len(masks)):
        mask = keep_significant_cc(np.asarray(masks[idx], dtype=np.uint8), rel_min=0.03, pix_min=MASK_PIX_MIN)
        if int(mask.sum()) < MASK_PIX_MIN:
            continue
        in_box_ratio, box_cover_ratio = _mask_box_overlap_stats(mask, ref_box_xyxy)
        bb = mask_bbox_xyxy(mask)
        bb_iou = bbox_iou(bb, ref_box_xyxy) if bb is not None else 0.0
        score = float(scores[idx]) + 0.45 * box_cover_ratio + 0.30 * in_box_ratio + 0.20 * bb_iou
        if score > best_score:
            best_score = score
            best_mask = mask
    return best_mask


def _build_object_masks_crop(
    crop_bgr: np.ndarray,
    object_name: str,
    ref_box_local: Optional[Tuple[int, int, int, int]],
    predictor: Any,
    generator: Any,
    inner_generator: Any,
    keep_max: int,
    force_whole_only: bool = False,
    max_inner_masks: int = 2,
    debug_inner_masks: int = 0,
    instruction: str = "",
) -> Dict[str, np.ndarray]:
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    print(f"[sam] segmenting object={object_name} crop_shape={crop_rgb.shape[:2]}", flush=True)
    tool_split_mode = _is_tool_like_name(object_name)
    name_declared_tool = tool_split_mode  # remember whether name triggered it
    whole = _best_box_prompt_mask(predictor, crop_rgb, ref_box_local)

    candidates: List[np.ndarray] = []
    if whole is None or not tool_split_mode:
        print(f"[sam] object={object_name} generating base masks", flush=True)
        candidates = sam_mask_pool_in_crop(
            rgb_crop=crop_rgb,
            generator=generator,
            object_name=object_name,
            ref_box_xyxy=ref_box_local,
        )
        candidates = nms_by_bbox(candidates, iou_th=0.50)
        candidates = dedup_by_mask_iou(candidates, th=0.85)
        candidates = keep_main_part_masks(
            masks=candidates,
            ref_box_xyxy=ref_box_local,
            keep_max=max(4, int(keep_max)),
            object_name=object_name,
        )

    if whole is None and not tool_split_mode:
        whole = _extract_white_flat_component(crop_bgr, object_name, ref_box_xyxy=ref_box_local)

    if whole is None:
        whole = _union_masks(candidates)

    if whole is None or int(whole.sum()) == 0:
        print(f"[sam] object={object_name} no whole mask", flush=True)
        return {}

    if bool(force_whole_only):
        print(f"[sam] object={object_name} forced whole-only mask", flush=True)
        return {"whole": whole}

    if not tool_split_mode and _looks_tool_like_shape(whole):
        tool_split_mode = True
        print(f"[sam] object={object_name} inferred tool-like shape from whole mask; trying inner masks", flush=True)

    # W1: instruction-intent + relaxed shape — catches novel tools with generic names
    # whose whole mask is not strongly elongated (e.g. hammer, wrench).
    if not tool_split_mode and instruction:
        ins_l = instruction.lower()
        if any(v in ins_l for v in _MANIP_VERBS) and _looks_tool_like_shape_relaxed(whole):
            tool_split_mode = True
            print(f"[sam] object={object_name} instruction-intent + shape => tool mode inferred", flush=True)

    # W4: shape-inferred tools get 3 inner-mask candidates instead of 2 so that
    # head-dominant and asymmetric tools expose a better handle/work-end boundary.
    effective_max_inner = (
        max(3, max_inner_masks) if (tool_split_mode and not name_declared_tool) else max_inner_masks
    )

    if not tool_split_mode:
        whole = _refine_white_flat_target_mask(whole, crop_bgr=crop_bgr, object_name=object_name)
        print(f"[sam] object={object_name} saved whole-only mask", flush=True)
        return {"whole": whole}

    print(f"[sam] object={object_name} generating inner masks", flush=True)
    debug_masks = []
    if int(debug_inner_masks) > 0:
        debug_masks = _debug_inner_masks_from_whole(
            rgb_crop=crop_rgb,
            whole_mask=whole,
            generator=inner_generator,
            max_masks=int(debug_inner_masks),
        )
    inner_masks = debug_masks if debug_masks else _inner_masks_from_whole(rgb_crop=crop_rgb, whole_mask=whole, generator=inner_generator)
    if not inner_masks:
        inner_masks = _tool_part_candidates(whole, candidates)
    inner_masks = _select_best_inner_masks(inner_masks, max_masks=effective_max_inner)
    if len(inner_masks) < 2:
        print(f"[sam] object={object_name} only found {len(inner_masks)} inner masks", flush=True)
        return {"whole": whole}
    print(f"[sam] object={object_name} found {len(inner_masks)} inner masks", flush=True)
    out = {"whole": whole}
    for idx, mask in enumerate(inner_masks, start=1):
        out[f"inner_{idx:02d}"] = mask
    for idx, mask in enumerate(debug_masks, start=1):
        out[f"raw_inner_{idx:02d}"] = mask
    return out


def build_sam_generator_and_predictor() -> Tuple[Any, Any, Any]:
    sam_model = build_sam_model()
    return build_sam_generator(sam_model=sam_model), build_inner_sam_generator(sam_model=sam_model), _build_sam_predictor(sam_model=sam_model)


def run_dino_from_sam(
    data_path: str,
    frame_no: int,
    instruction: str,
    dino_out_dir: str,
    image_path: Optional[str] = None,
) -> Dict[str, Any]:
    resolved_image = image_path or _resolve_image_path(data_path=data_path, frame_no=int(frame_no), out_dir=dino_out_dir)
    return run_dino(
        image_path=resolved_image,
        out_dir=dino_out_dir,
        instruction=instruction,
    )


def run_sam_stage(
    image_path: str,
    dino_result: Dict[str, Any],
    out_dir: str,
    keep_max: int = KEEP_MAX_DEFAULT,
    whole_only_for_tools: bool = False,
    save_grid: bool = True,
    max_inner_masks: int = 2,
    debug_inner_masks: int = 0,
    instruction: str = "",
) -> Dict[str, Any]:
    ensure_dir(out_dir)
    masks_dir = os.path.join(out_dir, "masks")
    ensure_dir(masks_dir)
    print(f"[sam] start image={image_path} out_dir={out_dir}", flush=True)

    full_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if full_bgr is None:
        raise FileNotFoundError(f"Cannot read RGB image: {image_path}")
    h, w = full_bgr.shape[:2]

    sam_gen, inner_sam_gen, predictor = build_sam_generator_and_predictor()
    outputs: List[Dict[str, Any]] = []

    for obj in dino_result.get("objects", []):
        name = str(obj.get("name", "object"))
        print(f"[sam] processing object={name}", flush=True)
        if not bool(obj.get("ok")):
            outputs.append({"name": name, "ok": False, "reason": obj.get("reason", "dino_failed")})
            continue

        crop_box = obj.get("crop_box_xyxy")
        if not isinstance(crop_box, list) or len(crop_box) != 4:
            outputs.append({"name": name, "ok": False, "reason": "missing_crop_box"})
            continue
        x1, y1, x2, y2 = [int(v) for v in crop_box]
        crop_bgr = full_bgr[y1 : y2 + 1, x1 : x2 + 1].copy()
        ref_box_local = dino_box_in_crop(obj)
        parts_crop = _build_object_masks_crop(
            crop_bgr=crop_bgr,
            object_name=name,
            ref_box_local=ref_box_local,
            predictor=predictor,
            generator=sam_gen,
            inner_generator=inner_sam_gen,
            keep_max=keep_max,
            force_whole_only=bool(whole_only_for_tools),
            max_inner_masks=max_inner_masks,
            debug_inner_masks=debug_inner_masks,
            instruction=instruction,
        )

        grid_path: Optional[str] = None
        raw_grid_path: Optional[str] = None
        raw_mask_paths: List[str] = []
        if save_grid:
            grid_path = os.path.join(out_dir, f"grid_{name}.png")
            stale_raw_grid_path = os.path.join(out_dir, f"raw_grid_{name}.png")
            if os.path.exists(stale_raw_grid_path):
                os.remove(stale_raw_grid_path)
            inner_keys = sorted(key for key in parts_crop if key.startswith("inner_"))
            grid_masks = [parts_crop[key] for key in ["whole", *inner_keys] if key in parts_crop]
            cv2.imwrite(grid_path, make_grid(crop_bgr, grid_masks))

        if not parts_crop:
            row = {"name": name, "ok": False, "reason": "no_valid_mask"}
            if grid_path:
                row["grid_path"] = grid_path
            outputs.append(row)
            continue

        saved_paths: Dict[str, str] = {}
        for role, mask_crop in parts_crop.items():
            mask_full = place_mask_into_full(mask_crop, (h, w), (x1, y1, x2, y2))
            mask_path = os.path.join(masks_dir, f"{name}_{role}.png")
            cv2.imwrite(mask_path, (mask_full * 255).astype(np.uint8))
            saved_paths[role] = mask_path
            if role.startswith("raw_inner_"):
                raw_mask_paths.append(mask_path)

        inner_keys = sorted(key for key in saved_paths if key.startswith("inner_"))
        if not bool(whole_only_for_tools) and len(inner_keys) >= 2:
            mask_roles = inner_keys
            mask_paths = [saved_paths[key] for key in inner_keys]
        else:
            mask_roles = ["whole"]
            mask_paths = [saved_paths["whole"]]

        outputs.append(
            {
                "name": name,
                "ok": True,
                "num_candidates": len(mask_paths),
                "mask_paths": mask_paths,
                "mask_roles": mask_roles,
                "whole_mask_path": saved_paths.get("whole"),
            }
        )
        if grid_path:
            outputs[-1]["grid_path"] = grid_path
        if raw_grid_path:
            outputs[-1]["raw_grid_path"] = raw_grid_path
        if raw_mask_paths:
            outputs[-1]["raw_mask_paths"] = raw_mask_paths

    summary = {
        "image_path": image_path,
        "num_objects": len(outputs),
        "objects": outputs,
        "mode": "whole_only" if bool(whole_only_for_tools) else f"whole_then_{max(2, int(max_inner_masks))}_inner_masks",
    }
    with open(os.path.join(out_dir, "sam_summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact SAM stage from dino.py result")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--frame_no", type=int, default=DEFAULT_FRAME_NO)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--image_path", default=None, help="Optional direct image path override")
    parser.add_argument("--dino_out_dir", default=DEFAULT_OUT_DIR, help="Output folder used by dino.py")
    parser.add_argument("--dino_json", default=DEFAULT_DINO_JSON, help="Path to existing dino_summary.json")
    parser.add_argument("--no_refresh_dino", action="store_true", help="Do not run dino.py; use existing --dino_json only")
    parser.add_argument("--out_dir", default=DEFAULT_SAM_OUT_DIR, help="Output folder for SAM artifacts")
    parser.add_argument("--keep_max", type=int, default=KEEP_MAX_DEFAULT)
    parser.add_argument("--whole_only_for_tools", action="store_true", help="For tool objects, export only whole mask (no inner masks).")
    parser.add_argument("--max_inner_masks", type=int, default=2, help="Maximum inner tool-part masks to export. Default keeps original two-part behavior.")
    parser.add_argument("--debug_inner_masks", type=int, default=0, help="Also save this many raw inner SAM proposals for diagnosis.")
    parser.add_argument("--no_grid", action="store_true", help="Do not save grid visualization image.")
    args = parser.parse_args()

    if args.no_refresh_dino:
        if not os.path.exists(args.dino_json):
            raise FileNotFoundError(f"dino_json not found: {args.dino_json}")
        with open(args.dino_json, "r", encoding="utf-8") as handle:
            dino_result = json.load(handle)
    else:
        dino_result = run_dino_from_sam(
            data_path=args.data_path,
            frame_no=args.frame_no,
            instruction=args.instruction,
            dino_out_dir=args.dino_out_dir,
            image_path=args.image_path,
        )

    image_path = args.image_path or dino_result.get("image_path")
    if not image_path:
        raise ValueError("image_path missing and not found inside dino_json")

    output = run_sam_stage(
        image_path=image_path,
        dino_result=dino_result,
        out_dir=args.out_dir,
        keep_max=args.keep_max,
        whole_only_for_tools=bool(args.whole_only_for_tools),
        save_grid=not bool(args.no_grid),
        max_inner_masks=int(args.max_inner_masks),
        debug_inner_masks=int(args.debug_inner_masks),
    )
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
