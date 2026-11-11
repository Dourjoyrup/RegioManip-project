#!/usr/bin/env python3
"""Render Chapter 1 of the RegioManip contribution video.

This first chapter explains region-level affordance grounding on a clean white
canvas.  It uses only archived run outputs and creates an MP4 plus lossless
stage keyframes.  A later chapter can continue from the final depth frame into
the rotating SE(3) scene.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


INK = (28, 38, 51)
MUTED = (91, 105, 120)
BLUE = (19, 108, 181)
CYAN = (0, 158, 205)
GREEN = (30, 143, 86)
ORANGE = (235, 126, 46)
LIGHT = (237, 242, 247)


def _dejavu(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    paths = [Path("/usr/share/fonts/truetype/dejavu") / name, Path("/usr/share/fonts") / name]
    for path in paths:
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


_LATO_DIR = Path("/usr/share/fonts/truetype/lato")
_FONT_CACHE: Dict[Tuple[int, bool], ImageFont.FreeTypeFont] = {}
_LATO_CMAP = None


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    """Lato (publication sans) with automatic DejaVu fallback for glyphs Lato
    lacks (e.g. ∈, ✓).  Lato is optically smaller, so sizes are scaled 8%."""
    key = (size, bold)
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    path = _LATO_DIR / ("Lato-Bold.ttf" if bold else "Lato-Medium.ttf")
    if not path.exists():
        font = _dejavu(size, bold)
    else:
        font = ImageFont.truetype(str(path), size=int(round(size * 1.08)))
        font._fallback = _dejavu(size, bold)  # type: ignore[attr-defined]
    _FONT_CACHE[key] = font
    return font


def _lato_missing(text: str) -> bool:
    global _LATO_CMAP
    if _LATO_CMAP is None:
        try:
            from fontTools.ttLib import TTFont
            _LATO_CMAP = set(TTFont(str(_LATO_DIR / "Lato-Bold.ttf")).getBestCmap())
        except Exception:
            _LATO_CMAP = set(range(32, 0x2000))
    return any(ord(c) not in _LATO_CMAP for c in str(text) if c not in "\n")


def _pick(font, text):
    fb = getattr(font, "_fallback", None)
    return fb if fb is not None and _lato_missing(text) else font


_orig_text = ImageDraw.ImageDraw.text
_orig_multiline = ImageDraw.ImageDraw.multiline_text
_orig_textlength = ImageDraw.ImageDraw.textlength


def _runs(font, text):
    """Split text into (substring, font) runs: Lato, DejaVu only where needed."""
    fb = getattr(font, "_fallback", None)
    if fb is None or not _lato_missing(text):
        return [(text, font)]
    runs = []
    for ch in text:
        f = fb if _lato_missing(ch) else font
        if runs and runs[-1][1] is f:
            runs[-1] = (runs[-1][0] + ch, f)
        else:
            runs.append((ch, f))
    return runs


def _patched_text(self, xy, text, fill=None, font=None, *args, **kwargs):
    if font is None or "\n" in str(text) or kwargs.get("anchor") not in (None, "la", "lm", "ls"):
        if font is not None and kwargs.get("anchor") not in (None, "la") and "\n" not in str(text):
            runs = _runs(font, text)
            if len(runs) > 1:
                # Resolve the anchor once on the full width, then draw runs left-aligned.
                anchor = kwargs.pop("anchor")
                total = sum(_orig_textlength(self, r, f) for r, f in runs)
                x, y = xy
                if anchor[0] == "m": x -= total / 2
                elif anchor[0] == "r": x -= total
                kwargs["anchor"] = "l" + anchor[1]
                for r, f in runs:
                    _orig_text(self, (x, y), r, fill, f, *args, **kwargs); x += _orig_textlength(self, r, f)
                return None
        return _orig_text(self, xy, text, fill, _pick(font, text) if font is not None else font, *args, **kwargs)
    runs = _runs(font, text)
    if len(runs) == 1:
        return _orig_text(self, xy, text, fill, font, *args, **kwargs)
    x, y = xy
    for r, f in runs:
        _orig_text(self, (x, y), r, fill, f, *args, **kwargs)
        x += _orig_textlength(self, r, f)
    return None


def _patched_multiline(self, xy, text, fill=None, font=None, *args, **kwargs):
    return _orig_multiline(self, xy, text, fill, _pick(font, text) if font is not None else font, *args, **kwargs)


def _patched_textlength(self, text, font=None, *args, **kwargs):
    if font is None:
        return _orig_textlength(self, text, font, *args, **kwargs)
    return sum(_orig_textlength(self, r, f, *args, **kwargs) for r, f in _runs(font, text))


ImageDraw.ImageDraw.text = _patched_text
ImageDraw.ImageDraw.multiline_text = _patched_multiline
ImageDraw.ImageDraw.textlength = _patched_textlength


def _smoothstep(value: float) -> float:
    x = float(np.clip(value, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def _fade(t: float, start: float, duration: float = 0.55) -> float:
    return _smoothstep((t - start) / duration)


def _fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 248, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _fit_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    source_h, source_w = mask.shape[:2]
    scale = min(width / source_w, height / source_h)
    resized = cv2.resize(mask, (int(round(source_w * scale)), int(round(source_h * scale))), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((height, width), dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _region_depth_visual(depth_m: np.ndarray, masks: Dict[str, np.ndarray], rgb: np.ndarray) -> np.ndarray:
    union = np.zeros(depth_m.shape[:2], dtype=np.uint8)
    for mask in masks.values():
        union = np.maximum(union, (mask > 0).astype(np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    union = cv2.erode(union, kernel, iterations=1)
    valid = (union > 0) & np.isfinite(depth_m) & (depth_m > 0)
    values = depth_m[valid]
    if values.size == 0:
        raise ValueError("No valid masked depth samples")
    low, high = np.percentile(values, [2, 98])
    normalized = np.clip((depth_m - low) / max(float(high - low), 1e-6), 0, 1)
    depth_u8 = np.round((1.0 - normalized) * 255).astype(np.uint8)
    colored = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
    background = cv2.addWeighted(rgb, 0.10, np.full_like(rgb, 248), 0.90, 0)
    background[valid] = colored[valid]
    return background


def _overlay_mask(image: np.ndarray, mask: np.ndarray, color: Sequence[int], alpha: float) -> np.ndarray:
    if alpha <= 0.0:
        return image
    resized = _fit_mask(mask.astype(np.uint8), image.shape[1], image.shape[0]) > 0
    out = image.astype(np.float32)
    out[resized] = (1.0 - alpha) * out[resized] + alpha * np.asarray(color, dtype=np.float32)
    contours, _ = cv2.findContours(resized.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, tuple(int(c) for c in color), 3, cv2.LINE_AA)
    return np.clip(out, 0, 255).astype(np.uint8)


def _point(mapping: Dict, key: str) -> Tuple[int, int] | None:
    value = mapping.get(key)
    if isinstance(value, dict) and value:
        value = next(iter(value.values()))
    if isinstance(value, list) and len(value) >= 2:
        return int(round(value[0])), int(round(value[1]))
    return None


def _scaled_xy(point: Tuple[int, int], source_shape: Tuple[int, int], panel_shape: Tuple[int, int]) -> Tuple[int, int]:
    source_h, source_w = source_shape
    panel_h, panel_w = panel_shape
    scale = min(panel_w / source_w, panel_h / source_h)
    xoff = (panel_w - source_w * scale) / 2.0
    yoff = (panel_h - source_h * scale) / 2.0
    return int(round(xoff + point[0] * scale)), int(round(yoff + point[1] * scale))


def _draw_keypoints(panel: np.ndarray, tool: Dict, target: Dict, source_shape: Tuple[int, int], alpha: float) -> np.ndarray:
    if alpha <= 0:
        return panel
    layer = panel.copy()
    grip = _point(tool, "grip_point_global")
    contact = _point(tool, "work_end_touch_mid_global")
    target_touch = _point(target, "target_edge_touch_mid_global")
    points = [(grip, "G", (35, 215, 255)), (contact, "C", (235, 158, 0)), (target_touch, "T", (46, 126, 235))]
    for point, label, bgr in points:
        if point is None:
            continue
        xy = _scaled_xy(point, source_shape, panel.shape[:2])
        cv2.circle(layer, xy, 11, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(layer, xy, 8, bgr, -1, cv2.LINE_AA)
        cv2.putText(layer, label, (xy[0] + 13, xy[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, bgr, 2, cv2.LINE_AA)
    edge_map = tool.get("work_end_points_global") or {}
    edge = list(edge_map.values())
    if len(edge) == 2:
        a = _scaled_xy(tuple(edge[0]), source_shape, panel.shape[:2])
        b = _scaled_xy(tuple(edge[1]), source_shape, panel.shape[:2])
        cv2.line(layer, a, b, (235, 158, 0), 5, cv2.LINE_AA)
    return cv2.addWeighted(layer, float(alpha), panel, 1.0 - float(alpha), 0)


def _rounded(draw: ImageDraw.ImageDraw, box: Tuple[int, int, int, int], fill, outline=None, radius: int = 18, width: int = 2) -> None:
    draw.rounded_rectangle(box, radius=radius, fill=fill, outline=outline, width=width)


# Shared page header so the brand, title, and subtitle never jump between
# Stages 1, 2, and 3.  Everything stays above y=170 (the header band).
HEADER_BRAND = ((65, 40), 30)
HEADER_TITLE = ((65, 80), 42)
HEADER_SUBTITLE = ((67, 135), 21)


def draw_header(draw: ImageDraw.ImageDraw, title: str, subtitle: str = "") -> None:
    draw.text(HEADER_BRAND[0], "RegioManip", font=_font(HEADER_BRAND[1], True), fill=BLUE)
    draw.text(HEADER_TITLE[0], title, font=_font(HEADER_TITLE[1], True), fill=INK)
    if subtitle:
        draw.text(HEADER_SUBTITLE[0], subtitle, font=_font(HEADER_SUBTITLE[1]), fill=MUTED)


def wrap_text(draw: ImageDraw.ImageDraw, text: str, font, max_width: int) -> str:
    """Greedy word wrap so captions always stay inside their cards."""
    lines, current = [], ""
    for word in text.split():
        trial = word if not current else current + " " + word
        if draw.textlength(trial, font=font) <= max_width:
            current = trial
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    return "\n".join(lines)


def ffmpeg_quality_args() -> List[str]:
    """Encoder settings for per-stage clips.

    REGIOMANIP_LOSSLESS=1 (set by reproduce_final_video.sh) keeps stage clips
    lossless so the complete video is compressed exactly once.
    """
    import os
    if os.environ.get("REGIOMANIP_LOSSLESS") == "1":
        return ["-c:v", "libx264", "-preset", "veryfast", "-qp", "0", "-pix_fmt", "yuv420p"]
    return ["-c:v", "libx264", "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart"]


def _draw_text_block(draw: ImageDraw.ImageDraw, t: float, stage: int) -> None:
    draw.text((86, 56), "RegioManip", font=_font(34, True), fill=BLUE)
    draw.text((86, 104), "Pixels to interaction geometry", font=_font(46, True), fill=INK)
    draw.text((88, 171), "Chapter 1  ·  Region-level affordance grounding", font=_font(25), fill=MUTED)

    steps = [
        ("01", "Open-world observation", "A single RGB-D frame; no CAD model."),
        ("02", "Semantic region grounding", "Separate handle, work end, and target."),
        ("03", "Affordance anchors", "Grip, contact edge, and target contact."),
        ("04", "Depth-aware lifting", "Clean local depth before 3D fitting."),
    ]
    top = 285
    for index, (number, title, subtitle) in enumerate(steps):
        y = top + index * 145
        active = index == stage
        passed = index < stage
        fill = (232, 244, 253) if active else (248, 250, 252)
        border = BLUE if active else (220, 227, 234)
        _rounded(draw, (78, y, 760, y + 112), fill=fill, outline=border, radius=18, width=3 if active else 1)
        circle = GREEN if passed else BLUE if active else (178, 190, 201)
        draw.ellipse((100, y + 28, 154, y + 82), fill=circle)
        draw.text((111, y + 38), number, font=_font(19, True), fill=(255, 255, 255))
        draw.text((176, y + 23), title, font=_font(27, True), fill=INK if active or passed else MUTED)
        draw.text((176, y + 65), subtitle, font=_font(20), fill=MUTED)

    _rounded(draw, (78, 887, 760, 986), fill=(246, 249, 252), outline=(220, 227, 234), radius=16)
    draw.text((104, 908), "Evidence path", font=_font(19, True), fill=BLUE)
    draw.text((104, 944), "RGB regions  →  depth cleaning  →  3D primitives  →  SE(3)", font=_font(20), fill=INK)


def render_frame(
    t: float,
    duration: float,
    rgb: np.ndarray,
    depth_color: np.ndarray,
    masks: Dict[str, np.ndarray],
    tool: Dict,
    target: Dict,
) -> np.ndarray:
    width, height = 1920, 1080
    frame = np.full((height, width, 3), 255, dtype=np.uint8)
    panel_x, panel_y, panel_w, panel_h = 835, 154, 1005, 754
    rgb_panel = _fit_image(rgb, panel_w, panel_h)
    depth_panel = _fit_image(depth_color, panel_w, panel_h)

    # Four explanatory stages, with an uncluttered transition between each.
    stage = min(3, max(0, int(t / (duration / 4.0))))
    region_alpha = _fade(t, duration * 0.25)
    anchor_alpha = _fade(t, duration * 0.50)
    depth_alpha = _fade(t, duration * 0.76, 0.75)
    panel = rgb_panel.copy()
    panel = _overlay_mask(panel, masks["handle"], (46, 166, 92), 0.34 * region_alpha)
    panel = _overlay_mask(panel, masks["work_end"], (22, 157, 205), 0.34 * region_alpha)
    panel = _overlay_mask(panel, masks["target"], (235, 126, 46), 0.27 * region_alpha)
    panel = _draw_keypoints(panel, tool, target, rgb.shape[:2], anchor_alpha)
    panel = cv2.addWeighted(depth_panel, depth_alpha, panel, 1.0 - depth_alpha, 0)

    frame[panel_y : panel_y + panel_h, panel_x : panel_x + panel_w] = panel
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    _draw_text_block(draw, t, stage)
    _rounded(draw, (panel_x - 5, panel_y - 5, panel_x + panel_w + 5, panel_y + panel_h + 5), fill=None, outline=(210, 219, 228), radius=22, width=3)

    labels = ["RGB observation", "Region masks", "Affordance geometry", "Region-filtered depth"]
    _rounded(draw, (panel_x + 28, panel_y + 26, panel_x + 355, panel_y + 76), fill=(255, 255, 255), outline=(210, 219, 228), radius=13)
    draw.text((panel_x + 48, panel_y + 38), labels[stage], font=_font(20, True), fill=INK)

    footer = "Region-specific geometry reduces irrelevant depth before contact-frame lifting."
    draw.text((panel_x, 950), footer, font=_font(23), fill=INK)
    draw.text((panel_x, 992), "No CAD model  ·  No task-specific 3D training", font=_font(21, True), fill=BLUE)
    progress = int((width - 160) * np.clip(t / duration, 0, 1))
    draw.rounded_rectangle((80, 1041, width - 80, 1051), radius=5, fill=(226, 232, 238))
    draw.rounded_rectangle((80, 1041, 80 + progress, 1051), radius=5, fill=BLUE)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render RegioManip contribution video, Chapter 1.")
    parser.add_argument("--run-dir", default="results/regiomanip2/color_000012_run3")
    parser.add_argument("--output-dir", default="results/video/regiomanip_step1")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--duration", type=float, default=12.0)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    output_dir = Path(args.output_dir)
    keyframe_dir = output_dir / "keyframes"
    keyframe_dir.mkdir(parents=True, exist_ok=True)
    vlm1 = json.loads((run_dir / "compat/vlm1_compat.json").read_text(encoding="utf-8"))
    vlm2 = json.loads((run_dir / "compat/vlm2_compat.json").read_text(encoding="utf-8"))
    rgb_path = Path(vlm1["rgb_path"])
    if not rgb_path.exists():
        archived = sorted(Path("data").glob(f"*/{rgb_path.name}"))
        if not archived:
            raise FileNotFoundError(f"RGB frame is unavailable: {rgb_path}")
        rgb_path = archived[0]
    rgb = cv2.imread(str(rgb_path), cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError("RGB frame is unavailable")

    objects1 = {o["name"]: o for o in vlm1["objects"]}
    objects2 = {o["name"]: o for o in vlm2["objects"]}
    tool_name = next(o["name"] for o in vlm2["objects"] if o.get("stage2_applicable"))
    target_name = next(o["name"] for o in vlm2["objects"] if not o.get("stage2_applicable", True))
    part_paths = {p["part"]: p["mask_path"] for p in objects1[tool_name]["parts"]}
    target_parts = {p["part"]: p["mask_path"] for p in objects1[target_name]["parts"]}
    masks = {
        "handle": cv2.imread(part_paths["handle"], cv2.IMREAD_GRAYSCALE),
        "work_end": cv2.imread(part_paths["work_end"], cv2.IMREAD_GRAYSCALE),
        "target": cv2.imread(target_parts.get("whole") or target_parts["contact_region"], cv2.IMREAD_GRAYSCALE),
    }
    if any(mask is None for mask in masks.values()):
        raise FileNotFoundError("One or more semantic masks are unavailable")
    depth_path = rgb_path.with_name(rgb_path.name.replace("color_", "depth_")).with_suffix(".npy")
    if not depth_path.exists():
        raise FileNotFoundError(f"Depth array is unavailable: {depth_path}")
    depth_raw = np.load(depth_path).astype(np.float32)
    depth_m = depth_raw * (0.001 if float(np.nanmedian(depth_raw)) > 10.0 else 1.0)
    depth_color = _region_depth_visual(depth_m, masks, rgb)

    fps = max(1, int(args.fps))
    duration = max(4.0, float(args.duration))
    output_path = output_dir / "regiomanip_contribution_step1.mp4"
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (1920, 1080))
    if not writer.isOpened():
        raise RuntimeError("OpenCV could not initialize the MP4 writer")
    total_frames = int(round(duration * fps))
    key_times = [0.8, duration * 0.30, duration * 0.56, duration * 0.84]
    saved: List[str] = []
    for index in range(total_frames):
        t = index / fps
        frame = render_frame(t, duration, rgb, depth_color, masks, objects2[tool_name], objects2[target_name])
        writer.write(frame)
        for key_index, key_time in enumerate(key_times):
            if abs(t - key_time) <= 0.5 / fps and key_index >= len(saved):
                path = keyframe_dir / f"stage_{key_index + 1}.png"
                cv2.imwrite(str(path), frame)
                saved.append(str(path))
    writer.release()
    manifest = {
        "video": str(output_path),
        "resolution": [1920, 1080],
        "fps": fps,
        "duration_seconds": duration,
        "frames": total_frames,
        "codec": "mp4v",
        "keyframes": saved,
        "source_run": str(run_dir),
        "next_chapter": "cleaned depth to rotating SE(3) scene",
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
