#!/usr/bin/env python3
"""Compact standalone GroundingDINO runner."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np
import torch

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
GROUNDING_DINO_ROOT = os.path.join(REPO_ROOT, "GroundingDINO")
if GROUNDING_DINO_ROOT not in sys.path:
    sys.path.insert(0, GROUNDING_DINO_ROOT)

from groundingdino.util.inference import (  # type: ignore
    load_image as dino_load_image,
    load_model as dino_load_model,
    predict as dino_predict,
)
from torchvision.ops import box_convert


DEFAULT_DATA_PATH = "./data"
DEFAULT_FRAME_NO = 13
DEFAULT_INSTRUCTION = "scrape the white putty"
DEFAULT_OUT_DIR = "./data/dino_out"
DEFAULT_TOOL_IDENTITY_MODE = os.environ.get("DINO_TOOL_IDENTITY_MODE", "force_requested_label")

DINO_CFG_PATH = os.path.join(GROUNDING_DINO_ROOT, "groundingdino", "config", "GroundingDINO_SwinT_OGC.py")
DINO_WEIGHTS = os.path.join(GROUNDING_DINO_ROOT, "ckpts", "groundingdino_swint_ogc.pth")

PER_CLASS_CFG: Dict[str, Dict[str, float]] = {
    "scraper": {"box_th": 0.30, "text_th": 0.22, "area_cap": 0.70, "pad_px": 22},
    "roller": {"box_th": 0.30, "text_th": 0.22, "area_cap": 0.75, "pad_px": 22},
    "hammer": {"box_th": 0.30, "text_th": 0.22, "area_cap": 0.70, "pad_px": 22},
    "screwdriver": {"box_th": 0.30, "text_th": 0.22, "area_cap": 0.70, "pad_px": 20},
    "brush": {"box_th": 0.30, "text_th": 0.22, "area_cap": 0.70, "pad_px": 20},
    "tool": {"box_th": 0.30, "text_th": 0.22, "area_cap": 0.80, "pad_px": 22},
    "white_putty": {"box_th": 0.30, "text_th": 0.22, "area_cap": 0.22, "pad_px": 10},
    "yellow_putty": {"box_th": 0.30, "text_th": 0.22, "area_cap": 0.22, "pad_px": 10},
    "screw": {"box_th": 0.24, "text_th": 0.18, "area_cap": 0.05, "pad_px": 8},
    "tape": {"box_th": 0.24, "text_th": 0.18, "area_cap": 0.30, "pad_px": 10},
    "generic_object": {"box_th": 0.30, "text_th": 0.22, "area_cap": 0.65, "pad_px": 14},
}

QUERY_VARIANTS: Dict[str, List[str]] = {
    "scraper": ["scraper", "scraper tool", "putty knife", "scraping tool"],
    "roller": ["roller", "roller tool", "paint roller", "brayer"],
    "hammer": ["hammer", "hammer tool"],
    "screwdriver": ["screwdriver", "screwdriver tool"],
    "brush": ["brush", "paint brush", "paintbrush"],
    "tape": ["tape", "white tape", "paper", "white paper", "tape paper"],
    "white_putty": ["white putty", "putty", "white clay", "clay"],
    "yellow_putty": ["yellow putty", "putty", "yellow clay", "clay"],
    "screw": ["screw", "bolt", "nail"],
}

TOOL_COMPETITOR_LABELS = {
    "scraper": ["scraper", "roller"],
    "roller": ["roller", "scraper"],
}

TOOL_NAMES = {"scraper", "roller", "hammer", "screwdriver", "brush", "tool"}
TARGET_NAMES = {"white_putty", "yellow_putty", "screw", "tape"}
TOOL_ONLY_VERBS = ("grip", "grasp", "hold", "grab", "pick up", "pickup", "take")
ACTION_TOOL_HINTS: Sequence[Tuple[str, Sequence[str]]] = (
    ("scraper", ("scrape", "scraping", "spread", "scoop", "spatula")),
    ("roller", ("roll", "rolling", "brayer")),
    ("hammer", ("hammer", "hammering", "mallet", "pound", "strike", "hit")),
    ("screwdriver", ("screw", "unscrew", "tighten", "loosen")),
)
ALIASES: Sequence[Tuple[str, Sequence[str]]] = (
    ("white_putty", ("white putty", "white clay")),
    ("yellow_putty", ("yellow putty", "yellow clay", "putty", "clay")),
    ("scraper", ("scraper", "scraper tool", "putty knife", "scraping tool", "spatula")),
    ("roller", ("roller", "roller tool", "paint roller", "brayer")),
    ("hammer", ("hammer", "hammer tool", "mallet", "hummer")),
    ("screwdriver", ("screwdriver", "screwdriver tool")),
    ("brush", ("brush", "paint brush", "paintbrush")),
    ("screw", ("screw", "bolt", "nail")),
    ("tape", ("tape", "white tape", "paper", "white paper", "tape paper", "label", "sticker")),
)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _safe_name(name: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", str(name or "").strip()).strip("_")
    return slug or "object"


def _normalize_text(text: str) -> str:
    text = str(text or "").lower().replace("_", " ").replace("-", " ")
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _contains_phrase(text: str, phrase: str) -> bool:
    text_norm = _normalize_text(text)
    phrase_norm = _normalize_text(phrase)
    if not text_norm or not phrase_norm:
        return False
    pattern = r"(?<!\w)" + re.escape(phrase_norm).replace(r"\ ", r"\s+") + r"(?!\w)"
    return re.search(pattern, text_norm) is not None


def _match_known_category(text: str, *, tools: bool = True, targets: bool = True) -> Optional[str]:
    text_norm = _normalize_text(text)
    if not text_norm:
        return None
    for category, names in ALIASES:
        if category in TOOL_NAMES and not tools:
            continue
        if category in TARGET_NAMES and not targets:
            continue
        if any(_contains_phrase(text_norm, alias) for alias in names):
            return category
    return None


def canonical_category(name: str) -> str:
    matched = _match_known_category(name, tools=True, targets=True)
    return matched or "tool"


def _caption_for(category: str) -> str:
    key = _match_known_category(category, tools=True, targets=True)
    if key is None:
        return _normalize_text(category) or "object"
    return QUERY_VARIANTS.get(key, [key.replace("_", " ")])[0]


def _extract_phrase_after(text_norm: str, tokens: Sequence[str], max_words: int = 5) -> str:
    for token in tokens:
        pattern = (
            r"\b" + re.escape(token) + r"\s+(?:the\s+|a\s+|an\s+)?"
            r"([a-z0-9]+(?:\s+[a-z0-9]+){0," + str(max_words - 1) + r"})"
        )
        match = re.search(pattern, text_norm)
        if match:
            return _normalize_text(match.group(1))
    return ""


def infer_queries_from_instruction(instruction: str) -> List[str]:
    text = _normalize_text(instruction)
    if not text:
        return ["tool"]

    instrument = _extract_phrase_after(text, ("by", "with", "using"), max_words=4)
    tool = _match_known_category(instrument, tools=True, targets=False) if instrument else None
    if tool is None:
        tool = _match_known_category(text, tools=True, targets=False)
    if tool is None:
        for tool_name, verbs in ACTION_TOOL_HINTS:
            if any(_contains_phrase(text, verb) for verb in verbs):
                tool = tool_name
                break

    direct_object = _extract_phrase_after(
        text,
        ("grip", "grasp", "hold", "grab", "pick up", "pickup", "take", "move", "place", "put", "push", "pull", "drag", "lift", "drop", "scrape", "roll", "hammer", "tighten", "loosen"),
        max_words=5,
    )
    tool_only = any(_contains_phrase(text, verb) for verb in TOOL_ONLY_VERBS) and not instrument

    target_text = text
    for phrase in ("putty knife", "scraper tool", "roller tool", "screwdriver tool", "hammer tool"):
        target_text = re.sub(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)", " ", target_text)
    if instrument:
        target_text = target_text.replace(instrument, " ")
    target_text = _normalize_text(target_text)
    target = _match_known_category(target_text, tools=False, targets=True)

    if tool is None and direct_object:
        maybe_tool = _match_known_category(direct_object, tools=True, targets=False)
        if maybe_tool is not None:
            tool = maybe_tool
        elif tool_only:
            tool = direct_object.replace(" ", "_")
    if tool is None:
        tool = "tool"

    if target is None and direct_object:
        maybe_target = _match_known_category(direct_object, tools=False, targets=True)
        if maybe_target is not None and maybe_target != tool:
            target = maybe_target

    if target is None:
        return [tool]
    if _normalize_text(target) == _normalize_text(tool):
        return [tool]
    return [tool, target]


def to_builtin(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_builtin(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return to_builtin(obj.detach().cpu().tolist())
    if isinstance(obj, np.ndarray):
        return to_builtin(obj.tolist())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def save_json(path: str, payload: Dict[str, Any]) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(to_builtin(payload), handle, indent=2)


def choose_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_dino(cfg_path: str = DINO_CFG_PATH, weights_path: str = DINO_WEIGHTS):
    model = dino_load_model(cfg_path, weights_path)
    device = choose_device()
    model = model.to(device)
    model.eval()
    setattr(model, "_inference_device", device)
    return model


def run_predict_compat(model, image_tensor, caption: str, box_th: float, text_th: float, device: str):
    try:
        return dino_predict(
            model=model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_th,
            text_threshold=text_th,
            device=device,
        )
    except TypeError:
        return dino_predict(
            model=model,
            image=image_tensor,
            caption=caption,
            box_threshold=box_th,
            text_threshold=text_th,
        )


def clamp_box(x1: int, y1: int, x2: int, y2: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width - 1))
    y2 = max(0, min(y2, height - 1))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return x1, y1, x2, y2


def expand_box(box_xyxy: Tuple[int, int, int, int], pad_px: int, width: int, height: int) -> Tuple[int, int, int, int]:
    x1, y1, x2, y2 = box_xyxy
    return clamp_box(x1 - pad_px, y1 - pad_px, x2 + pad_px, y2 + pad_px, width, height)


def draw_box(img: np.ndarray, box: Tuple[int, int, int, int], label: str) -> None:
    x1, y1, x2, y2 = box
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(img, label, (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)


def _score_detection(text: str, box: Optional[torch.Tensor], logit: Any) -> float:
    score = float(logit)
    key = str(text).strip().lower()
    if box is None:
        return score
    cx, cy, width, height = [float(v) for v in box.tolist()]
    x0 = cx - width / 2.0
    y0 = cy - height / 2.0
    x1 = cx + width / 2.0
    area = max(width, 0.0) * max(height, 0.0)
    if "putty" in key or "clay" in key:
        if y0 < 0.08:
            score -= 0.8
        if x0 < 0.03 or x1 > 0.97:
            score -= 0.35
        if area > 0.22:
            score -= 0.35
        if cy > 0.30:
            score += 0.15
    if key in {"screw", "bolt", "nail"}:
        # Screws are usually tall/narrow and occupy a small but still visible
        # vertical extent; this suppresses tiny circular blobs and holes.
        aspect_h_over_w = height / max(width, 1e-6)
        if aspect_h_over_w > 1.8:
            score += min(0.28, 0.10 * (aspect_h_over_w - 1.8) + 0.12)
        elif aspect_h_over_w < 1.1:
            score -= 0.22

        if height > 0.08:
            score += min(0.18, 0.9 * (height - 0.08))
        elif height < 0.035:
            score -= 0.30

        if area < 0.0009:
            score -= 0.18
    return score


def _appearance_score(text: str, box: Optional[torch.Tensor], image_np: Optional[np.ndarray]) -> float:
    key = str(text).strip().lower()
    if box is None or image_np is None:
        return 0.0
    height, width = image_np.shape[:2]
    cx, cy, bw, bh = [float(v) for v in box.tolist()]
    x0 = max(0, int((cx - bw / 2.0) * width))
    x1 = min(width, int((cx + bw / 2.0) * width))
    y0 = max(0, int((cy - bh / 2.0) * height))
    y1 = min(height, int((cy + bh / 2.0) * height))
    if x1 <= x0 or y1 <= y0:
        return 0.0

    crop = image_np[y0:y1, x0:x1]
    r = crop[..., 0].astype(np.float32)
    g = crop[..., 1].astype(np.float32)
    b = crop[..., 2].astype(np.float32)

    wood_ratio = (
        (r > 90) & (r < 210) &
        (g > 60) & (g < 170) &
        (b > 20) & (b < 140) &
        (r > g) & (g > b)
    ).mean()
    green_ratio = ((g > r + 15) & (g > b + 10) & (g > 80)).mean()

    score = 0.0
    if key == "scraper":
        score += min(0.14, 2.2 * float(wood_ratio))
        score -= min(0.18, 1.2 * float(green_ratio))
    elif key == "roller":
        score += min(0.12, 0.9 * float(green_ratio))
    return score


def _box_iou(box_a: Optional[torch.Tensor], box_b: Optional[torch.Tensor]) -> float:
    if box_a is None or box_b is None:
        return 0.0
    ax, ay, aw, ah = [float(v) for v in box_a.tolist()]
    bx, by, bw, bh = [float(v) for v in box_b.tolist()]
    ax0, ay0, ax1, ay1 = ax - aw / 2.0, ay - ah / 2.0, ax + aw / 2.0, ay + ah / 2.0
    bx0, by0, bx1, by1 = bx - bw / 2.0, by - bh / 2.0, bx + bw / 2.0, by + bh / 2.0
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0.0, ix1 - ix0), max(0.0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0.0, aw) * max(0.0, ah)
    area_b = max(0.0, bw) * max(0.0, bh)
    union = area_a + area_b - inter
    return 0.0 if union <= 0.0 else inter / union


def _resolve_detection_query(category: str) -> Tuple[str, bool, str, str]:
    raw = _normalize_text(category)
    known = _match_known_category(raw, tools=True, targets=True)
    if known is None:
        return raw.replace(" ", "_") or "object", False, "generic_object", raw or "object"
    cfg_key = known if known in PER_CLASS_CFG else "generic_object"
    return known, True, cfg_key, _caption_for(known)


def _query_variants_for(category: str, caption_override: Optional[str] = None) -> List[str]:
    if caption_override is not None:
        return [str(caption_override)]
    key = _match_known_category(category, tools=True, targets=True)
    if key is None:
        return [str(category)]
    return list(QUERY_VARIANTS.get(key, [key.replace("_", " ")]))


def _pixel_box_from_norm(box: torch.Tensor, width: int, height: int) -> Tuple[int, int, int, int]:
    box = box.detach().cpu().reshape(1, 4)
    scale = torch.tensor([[width, height, width, height]], dtype=box.dtype)
    xyxy = box_convert(box * scale, in_fmt="cxcywh", out_fmt="xyxy")[0]
    return clamp_box(
        int(round(float(xyxy[0]))),
        int(round(float(xyxy[1]))),
        int(round(float(xyxy[2]))),
        int(round(float(xyxy[3]))),
        width,
        height,
    )


def _best_detection_for_text(
    model,
    image_torch,
    image_np: np.ndarray,
    device: str,
    text: str,
    box_threshold: float,
    text_threshold: float,
    area_cap: float,
    caption_override: Optional[str] = None,
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[str], Optional[float], Optional[str]]:
    chosen_box: Optional[torch.Tensor] = None
    chosen_logit: Optional[torch.Tensor] = None
    chosen_phrase: Optional[str] = None
    chosen_score: Optional[float] = None
    chosen_caption: Optional[str] = None
    img_h, img_w = image_np.shape[:2]

    for query in _query_variants_for(text, caption_override=caption_override):
        boxes, logits, phrases = run_predict_compat(model, image_torch, query, box_threshold, text_threshold, device)
        if logits is None or len(logits) == 0:
            continue
        for idx in range(len(logits)):
            box = boxes[idx]
            score = _score_detection(text, box, logits[idx])
            score += _appearance_score(text, box, image_np)
            pixel_box = _pixel_box_from_norm(box, img_w, img_h)
            box_area = float((pixel_box[2] - pixel_box[0] + 1) * (pixel_box[3] - pixel_box[1] + 1))
            area_ratio = box_area / float(max(img_w * img_h, 1))
            if area_ratio > area_cap:
                score -= 0.35 + max(0.0, area_ratio - area_cap)
            if chosen_score is None or score > chosen_score:
                chosen_box = box.detach().clone()
                chosen_logit = logits[idx].detach().clone() if torch.is_tensor(logits[idx]) else torch.tensor(float(logits[idx]))
                chosen_phrase = phrases[idx] if isinstance(phrases, list) and idx < len(phrases) else None
                chosen_score = float(score)
                chosen_caption = query
    return chosen_box, chosen_logit, chosen_phrase, chosen_score, chosen_caption


def detect_one(
    image_path: str,
    model,
    category: str,
    caption: Optional[str] = None,
    tool_identity_mode: str = DEFAULT_TOOL_IDENTITY_MODE,
) -> Optional[Dict[str, Any]]:
    name, is_known, cfg_key, fallback_caption = _resolve_detection_query(category)
    cfg = PER_CLASS_CFG.get(cfg_key, PER_CLASS_CFG["generic_object"])
    image_np, image_torch = dino_load_image(image_path)
    img_h, img_w = image_np.shape[:2]
    device = getattr(model, "_inference_device", "cpu")

    box, logit, phrase, score, chosen_caption = _best_detection_for_text(
        model=model,
        image_torch=image_torch,
        image_np=image_np,
        device=device,
        text=name,
        box_threshold=float(cfg["box_th"]),
        text_threshold=float(cfg["text_th"]),
        area_cap=float(cfg["area_cap"]),
        caption_override=caption if caption is not None else fallback_caption if not is_known else None,
    )
    if box is None or logit is None:
        return None

    final_name = name
    identity_debug: Optional[Dict[str, Any]] = None
    best_score = float(score if score is not None else float(logit))
    competitors = TOOL_COMPETITOR_LABELS.get(name, [])
    if name in TOOL_NAMES and competitors:
        scores = {name: best_score}
        for competitor in competitors:
            if competitor == name:
                continue
            alt_cfg = PER_CLASS_CFG.get(competitor, cfg)
            alt_box, _, _, alt_score, _ = _best_detection_for_text(
                model=model,
                image_torch=image_torch,
                image_np=image_np,
                device=device,
                text=competitor,
                box_threshold=float(alt_cfg["box_th"]),
                text_threshold=float(alt_cfg["text_th"]),
                area_cap=float(alt_cfg["area_cap"]),
            )
            if alt_box is None or alt_score is None:
                continue
            scores[competitor] = float(alt_score)
            same_instance = _box_iou(box, alt_box) > 0.45
            if same_instance and float(alt_score) > best_score + 0.03:
                final_name = competitor
                best_score = float(alt_score)
        if len(scores) > 1:
            identity_debug = {"requested_name": name, "scores": scores}

    if final_name != name:
        mode = str(tool_identity_mode or DEFAULT_TOOL_IDENTITY_MODE).strip().lower()
        if mode == "strict":
            return None
        if mode != "relabel":
            final_name = name

    raw_box = _pixel_box_from_norm(box, img_w, img_h)
    crop_box = expand_box(raw_box, int(cfg["pad_px"]), img_w, img_h)
    result: Dict[str, Any] = {
        "name": final_name,
        "query": _normalize_text(category),
        "is_known_category": bool(is_known),
        "category_cfg_key": cfg_key,
        "ok": True,
        "caption": chosen_caption or fallback_caption,
        "box_xyxy": list(raw_box),
        "crop_box_xyxy": list(crop_box),
        "phrase": phrase,
        "logit": float(logit.item()),
        "params": {
            "box_th": float(cfg["box_th"]),
            "text_th": float(cfg["text_th"]),
            "area_cap": float(cfg["area_cap"]),
            "pad_px": int(cfg["pad_px"]),
            "device": device,
        },
    }
    if final_name != name:
        result["requested_name"] = name
        result["identity_resolution"] = "relabelled_by_tool_competitor_check"
    elif identity_debug is not None and str(tool_identity_mode or "").strip().lower() != "force_requested_label":
        result["tool_identity_debug"] = identity_debug
    return result


def run_dino(
    image_path: str,
    out_dir: str,
    instruction: str = DEFAULT_INSTRUCTION,
    categories: Optional[List[str]] = None,
    cfg_path: str = DINO_CFG_PATH,
    weights_path: str = DINO_WEIGHTS,
    save_overlay: str = "debug_dino_rois.png",
    tool_identity_mode: str = DEFAULT_TOOL_IDENTITY_MODE,
) -> Dict[str, Any]:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")

    ensure_dir(out_dir)
    ensure_dir(os.path.join(out_dir, "crops"))

    image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    model = build_dino(cfg_path=cfg_path, weights_path=weights_path)
    queries = categories if categories else infer_queries_from_instruction(instruction)
    objects: List[Dict[str, Any]] = []

    for query in queries:
        query_name = str(query).strip()
        caption = None
        if ":" in query_name:
            query_name, caption = [part.strip() for part in query_name.split(":", 1)]
        det = detect_one(
            image_path=image_path,
            model=model,
            category=query_name,
            caption=caption,
            tool_identity_mode=tool_identity_mode,
        )
        if det is None:
            preview_name, _, _, _ = _resolve_detection_query(query_name)
            objects.append({"name": preview_name, "query": _normalize_text(query_name), "ok": False, "reason": "no_roi"})
        else:
            objects.append(det)

    overlay = image_bgr.copy()
    for index, obj in enumerate(objects):
        if not obj.get("ok"):
            continue
        box_xyxy = obj.get("box_xyxy")
        crop_xyxy = obj.get("crop_box_xyxy")
        if not isinstance(box_xyxy, list) or len(box_xyxy) != 4:
            continue
        if not isinstance(crop_xyxy, list) or len(crop_xyxy) != 4:
            continue
        raw_box = tuple(int(v) for v in box_xyxy)
        crop_box = tuple(int(v) for v in crop_xyxy)
        x1, y1, x2, y2 = crop_box
        crop_path = os.path.join(out_dir, "crops", f"{index:02d}_{_safe_name(str(obj.get('name', 'object')))}.png")
        cv2.imwrite(crop_path, image_bgr[y1 : y2 + 1, x1 : x2 + 1])
        obj["crop_path"] = crop_path
        draw_box(overlay, raw_box, str(obj.get("name", "object")))

    overlay_path = save_overlay if os.path.isabs(save_overlay) else os.path.join(out_dir, save_overlay)
    cv2.imwrite(overlay_path, overlay)

    result = {
        "image_path": image_path,
        "instruction": instruction,
        "tool_identity_mode": tool_identity_mode,
        "queries": queries,
        "overlay_path": overlay_path,
        "objects": objects,
    }
    save_json(os.path.join(out_dir, "dino_summary.json"), result)
    return result


def run_dino_stage(image_path: str, instruction: str, out_dir: str = DEFAULT_OUT_DIR, **kwargs: Any) -> Dict[str, Any]:
    return run_dino(image_path=image_path, out_dir=out_dir, instruction=instruction, **kwargs)


def _list_color_pngs(data_path: str) -> List[str]:
    if not os.path.isdir(data_path):
        return []
    return sorted(
        os.path.join(data_path, filename)
        for filename in os.listdir(data_path)
        if filename.startswith("color_") and filename.endswith(".png")
    )


def _latest_frame_png(data_path: str) -> Optional[str]:
    files = _list_color_pngs(data_path)
    return files[-1] if files else None


def _resolve_image_path(data_path: str, frame_no: int, out_dir: str) -> str:
    png_path = os.path.join(data_path, f"color_{int(frame_no):06d}.png")
    if os.path.exists(png_path):
        return png_path

    npy_path = os.path.join(data_path, f"color_{int(frame_no):06d}.npy")
    if os.path.exists(npy_path):
        array = np.load(npy_path)
        if array.ndim != 3:
            raise ValueError(f"Unexpected npy image shape: {array.shape}")
        if array.dtype != np.uint8:
            array = (np.clip(array, 0.0, 1.0) * 255).astype(np.uint8)
        ensure_dir(out_dir)
        tmp_png = os.path.join(out_dir, f"_tmp_color_{int(frame_no):06d}.png")
        cv2.imwrite(tmp_png, array[..., ::-1])
        return tmp_png

    latest = _latest_frame_png(data_path)
    if latest is not None:
        print(f"[WARN] Requested frame {frame_no} not found. Using latest available image: {latest}")
        return latest

    raise FileNotFoundError(f"No RGB input found for frame {frame_no} in {data_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Compact GroundingDINO ROI localization demo")
    parser.add_argument("--data_path", default=DEFAULT_DATA_PATH)
    parser.add_argument("--frame_no", type=int, default=DEFAULT_FRAME_NO)
    parser.add_argument("--image_path", default=None)
    parser.add_argument("--instruction", default=DEFAULT_INSTRUCTION)
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR)
    parser.add_argument("--categories", nargs="+", default=None)
    parser.add_argument("--cfg_path", default=DINO_CFG_PATH)
    parser.add_argument("--weights_path", default=DINO_WEIGHTS)
    parser.add_argument("--save_overlay", default="debug_dino_rois.png")
    parser.add_argument("--tool_identity_mode", default=DEFAULT_TOOL_IDENTITY_MODE, choices=["strict", "relabel", "force_requested_label"])
    args = parser.parse_args()

    image_path = args.image_path or _resolve_image_path(args.data_path, args.frame_no, args.out_dir)
    result = run_dino(
        image_path=image_path,
        out_dir=args.out_dir,
        instruction=args.instruction,
        categories=args.categories,
        cfg_path=args.cfg_path,
        weights_path=args.weights_path,
        save_overlay=args.save_overlay,
        tool_identity_mode=args.tool_identity_mode,
    )
    print(json.dumps(to_builtin(result), indent=2))


if __name__ == "__main__":
    main()
