#!/usr/bin/env python3
"""Shared visual system for the RegioManip Stage 1–3 video.

Every frame of every stage uses the same chrome, in fixed screen regions:

* header band (y < 165): brand, stage title, subtitle (stage renderers)
* global stage tracker (top right): Stage 1 → Stage 2 → Stage 3
* step indicator (y 168–222): the current stage's sub-steps with arrows
* narration bar (y 962–1046): one plain-language sentence per step

Content lives between y = 225 and y = 950.  Chrome is drawn on top of the
finished content frame (``apply_chrome``), so it never takes part in content
fades and stays perfectly stable.
"""

from __future__ import annotations

import bisect
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw

from render_contribution_video_step1 import BLUE, INK, MUTED, _font, _rounded, _smoothstep, draw_header

W, H = 1920, 1080
WHITE = (255, 255, 255)
GREEN = (20, 140, 78)
FUTURE = (176, 190, 203)
FUTURE_TEXT = (128, 142, 156)
BORDER = (198, 210, 221)
PANEL = (247, 249, 252)
PALE_BLUE = (234, 243, 252)

STAGES = (
    ("Grounding", "Region-level affordance grounding",
     "RGB-D + instruction  →  RegionAffordanceTuple"),
    ("Checking", "Geometry checking and lattice pruning",
     "VLM tuple + geometry  →  ranked hypothesis Δ⁺"),
    ("Lifting", "Contact-frame lifting into SE(3)",
     "Δ⁺ + metric 3D  →  contact frame T*C ∈ SE(3)"),
)

STAGE1_STEPS = ("Input", "3D scene", "Detect", "Segment", "Skeletons", "VLM prompt", "VLM output")
STAGE2_STEPS = ("Geometry predictors", "Depth gate", "Agreement Ψ", "Physics lattice", "Ranked Δ⁺")
STAGE3_STEPS = ("Back-project regions", "Fit contact primitives", "Rank SE(3) hypotheses", "Construct contact frame")

TRACKER_BOX = (1150, 26, 1880, 92)
STEPPER_BAND = (0, 166, W, 218)
NARRATION_BOX = (65, 964, 1855, 1042)


def ease(x: float) -> float:
    return _smoothstep(float(np.clip(x, 0.0, 1.0)))


def mix(c1, c2, p):
    p = float(np.clip(p, 0, 1))
    return tuple(int(round(a + (b - a) * p)) for a, b in zip(c1, c2))


# --------------------------------------------------------------------------
# Pacing
# --------------------------------------------------------------------------
class TimeMap:
    """Map real (video) time to the renderer's source time.

    ``slow`` stretches every animation; ``holds`` freezes the source clock at
    given source times for the given number of seconds (reading pauses).
    """

    def __init__(self, src_end: float, slow: float = 1.0, holds: Iterable[Tuple[float, float]] = ()):
        self.slow = slow
        self.holds = sorted((float(a), float(b)) for a, b in holds)
        self.src_end = src_end
        self.real_end = src_end * slow + sum(h for _, h in self.holds)

    def __call__(self, real_t: float) -> float:
        src = 0.0; real = 0.0
        for at, hold in self.holds:
            seg = (at - src) * self.slow
            if real_t < real + seg:
                return src + (real_t - real) / self.slow
            real += seg; src = at
            if real_t < real + hold:
                return src
            real += hold
        return min(self.src_end, src + (real_t - real) / self.slow)


def fade(frame, k, to=255):
    """k=1 → frame, k=0 → white."""
    k = float(np.clip(k, 0, 1))
    if k >= 1: return frame
    return cv2.addWeighted(frame, k, np.full_like(frame, to), 1 - k, 0)


# --------------------------------------------------------------------------
# Glyphs
# --------------------------------------------------------------------------
def arrow(d: ImageDraw.ImageDraw, a, b, color, width=4, head=None, progress=1.0):
    """Straight arrow with a proportionate, filled arrowhead."""
    p = float(np.clip(progress, 0, 1))
    if p <= 0: return
    ax, ay = a; bx, by = b
    ex, ey = ax + (bx - ax) * p, ay + (by - ay) * p
    L = max(1e-6, np.hypot(ex - ax, ey - ay))
    ux, uy = (ex - ax) / L, (ey - ay) / L
    head = head or max(12, width * 3.2)
    sx, sy = ex - ux * head * .85, ey - uy * head * .85
    d.line((ax, ay, sx, sy), fill=color, width=width)
    nx, ny = -uy, ux
    d.polygon([(ex, ey), (ex - ux * head + nx * head * .55, ey - uy * head + ny * head * .55),
               (ex - ux * head - nx * head * .55, ey - uy * head - ny * head * .55)], fill=color)


def check_badge(d, cx, cy, r=15, color=GREEN):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    d.line((cx - r * .45, cy + r * .02, cx - r * .1, cy + r * .38), fill=WHITE, width=max(2, int(r * .22)))
    d.line((cx - r * .1, cy + r * .38, cx + r * .5, cy - r * .35), fill=WHITE, width=max(2, int(r * .22)))


def cross_badge(d, cx, cy, r=15, color=(205, 50, 50)):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=color)
    k = r * .38; w = max(2, int(r * .22))
    d.line((cx - k, cy - k, cx + k, cy + k), fill=WHITE, width=w)
    d.line((cx + k, cy - k, cx - k, cy + k), fill=WHITE, width=w)


def number_badge(d, cx, cy, text, r=15, fill=BLUE, fg=WHITE, outline=None):
    d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=fill, outline=outline, width=2 if outline else 0)
    f = _font(int(r * 1.0), True)
    d.text((cx, cy), text, font=f, fill=fg, anchor="mm")


def text_center(d, cx, y, text, font, fill):
    d.text((cx - d.textlength(text, font=font) / 2, y), text, font=font, fill=fill)


# --------------------------------------------------------------------------
# Chrome
# --------------------------------------------------------------------------
def draw_tracker(d: ImageDraw.ImageDraw, stage: int):
    """Global Stage 1 → 2 → 3 tracker, right-aligned in the header."""
    f = _font(17, True)
    items = []
    for i, (short, _, _) in enumerate(STAGES, start=1):
        items.append((i, f"Stage {i} · {short}"))
    widths = [d.textlength(t, font=f) + 58 for _, t in items]
    gap = 34
    x = TRACKER_BOX[2] - (sum(widths) + gap * (len(items) - 1)); y0, y1 = 40, 78
    for (i, label), w in zip(items, widths):
        done, active = i < stage, i == stage
        if active:
            _rounded(d, (x, y0, x + w, y1), fill=BLUE, outline=BLUE, radius=19, width=2)
            fg = WHITE
        elif done:
            _rounded(d, (x, y0, x + w, y1), fill=(236, 247, 240), outline=GREEN, radius=19, width=2)
            fg = GREEN
        else:
            _rounded(d, (x, y0, x + w, y1), fill=WHITE, outline=BORDER, radius=19, width=2)
            fg = FUTURE_TEXT
        if done:
            check_badge(d, x + 22, (y0 + y1) / 2, 10)
        else:
            number_badge(d, x + 22, (y0 + y1) / 2, str(i), 10, fill=WHITE if active else FUTURE,
                         fg=BLUE if active else WHITE)
        d.text((x + 40, (y0 + y1) / 2), label, font=f, fill=fg, anchor="lm")
        if i < len(items):
            arrow(d, (x + w + 7, (y0 + y1) / 2), (x + w + gap - 7, (y0 + y1) / 2), FUTURE, 3, 10)
        x += w + gap


def draw_stepper(d: ImageDraw.ImageDraw, steps: Sequence[str], active: int, stage: int):
    """Sub-step indicator with arrows.  ``active`` is 0-based; -1 = none."""
    f_on, f_off = _font(18, True), _font(18, True)
    r = 14; cy = 195
    widths = [2 * r + 12 + d.textlength(s, font=f_on) for s in steps]
    left, right = 65, 1855
    gap = (right - left - sum(widths)) / max(1, len(steps) - 1)
    gap = min(gap, 120)
    x = left
    for i, (label, w) in enumerate(zip(steps, widths)):
        done, act = i < active, i == active
        cx = x + r
        if done:
            check_badge(d, cx, cy, r)
            col = INK
        elif act:
            d.ellipse((cx - r - 5, cy - r - 5, cx + r + 5, cy + r + 5), fill=PALE_BLUE)
            number_badge(d, cx, cy, str(i + 1), r, fill=BLUE)
            col = BLUE
        else:
            number_badge(d, cx, cy, str(i + 1), r, fill=WHITE, fg=FUTURE_TEXT, outline=FUTURE)
            col = FUTURE_TEXT
        d.text((x + 2 * r + 12, cy), label, font=f_on if (act or done) else f_off, fill=col, anchor="lm")
        if act:
            tl = d.textlength(label, font=f_on)
            d.line((x + 2 * r + 12, cy + 17, x + 2 * r + 12 + tl, cy + 17), fill=BLUE, width=3)
        if i < len(steps) - 1:
            a0 = x + w + 16; a1 = x + w + gap - 16
            if a1 - a0 > 18:
                arrow(d, (a0, cy), (a1, cy), GREEN if done else FUTURE, 3, 11)
        x += w + gap


def draw_narration(d: ImageDraw.ImageDraw, text: str, stage: int, step: int, alpha: float = 1.0):
    x0, y0, x1, y1 = NARRATION_BOX
    _rounded(d, (x0, y0, x1, y1), fill=(244, 248, 252), outline=BORDER, radius=16, width=1)
    d.rounded_rectangle((x0, y0, x0 + 8, y1), radius=4, fill=BLUE)
    tag = f"{stage}.{step}" if step > 0 else f"{stage}"
    ft = _font(17, True)
    tw = d.textlength(tag, font=ft) + 28
    _rounded(d, (x0 + 30, y0 + 22, x0 + 30 + tw, y0 + 56), fill=BLUE, outline=BLUE, radius=10, width=1)
    d.text((x0 + 30 + tw / 2, y0 + 39), tag, font=ft, fill=WHITE, anchor="mm")
    col = mix((244, 248, 252), INK, alpha)
    d.text((x0 + 60 + tw, (y0 + y1) / 2), text, font=_font(24), fill=col, anchor="lm")


def apply_chrome(frame_bgr: np.ndarray, stage: int, steps: Sequence[str], active: int,
                 narration: str | None = None, narr_alpha: float = 1.0, opacity: float = 1.0) -> np.ndarray:
    """Draw tracker, stepper, and narration on top of a finished frame."""
    out = frame_bgr.copy()
    x0, y0, x1, y1 = TRACKER_BOX; out[y0:y1, x0:x1] = 255
    out[STEPPER_BAND[1]:STEPPER_BAND[3], :] = 255
    out[952:1056, :] = 255
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB)); d = ImageDraw.Draw(pil)
    draw_tracker(d, stage)
    draw_stepper(d, steps, active, stage)
    if narration:
        draw_narration(d, narration, stage, active + 1, narr_alpha)
    chromed = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    if opacity >= 1: return chromed
    return cv2.addWeighted(chromed, opacity, frame_bgr, 1 - opacity, 0)


class Narrator:
    """Narration text with a short fade when the sentence changes."""

    def __init__(self, cues: Sequence[Tuple[float, str]], fade_s: float = .35):
        self.times = [c[0] for c in cues]; self.texts = [c[1] for c in cues]; self.fade = fade_s

    def __call__(self, t: float):
        i = max(0, bisect.bisect_right(self.times, t) - 1)
        since = t - self.times[i]
        return self.texts[i], (1.0 if i == 0 and since < 0 else ease(since / self.fade))


# --------------------------------------------------------------------------
# Title cards
# --------------------------------------------------------------------------
def title_card(stage: int) -> np.ndarray:
    """Stage title card.  stage=0 is the opening overview card."""
    pil = Image.new("RGB", (W, H), WHITE); d = ImageDraw.Draw(pil)
    if stage == 0:
        text_center(d, 960, 170, "RegioManip", _font(34, True), BLUE)
        text_center(d, 960, 228, "Region-Level Affordances for Zero-Shot", _font(50, True), INK)
        text_center(d, 960, 294, "Wrench-Bounded Tool Manipulation", _font(50, True), INK)
        text_center(d, 960, 380, "From one RGB-D image and an instruction to an executable SE(3) contact frame",
                    _font(25), MUTED)
    else:
        name = STAGES[stage - 1][1]
        text_center(d, 960, 190, f"STAGE {stage} OF 3", _font(24, True), BLUE)
        text_center(d, 960, 240, name, _font(56, True), INK)
        text_center(d, 960, 330, STAGES[stage - 1][2], _font(27), MUTED)
    # Pipeline overview: three boxes with arrows; the current stage is highlighted.
    bw, bh, gap = 470, 230, 95
    x = 960 - (3 * bw + 2 * gap) / 2; y = 520
    examples = ("DINO · SAM 2 · skeletons · VLM", "geometry–VLM agreement · physics lattice",
                "primitives · SE(3) ranking · contact frame")
    for i, (short, name, io) in enumerate(STAGES, start=1):
        act, done = i == stage, (0 < i < stage)
        fill = PALE_BLUE if act else (240, 248, 243) if done else PANEL
        out = BLUE if act else GREEN if done else BORDER
        _rounded(d, (x, y, x + bw, y + bh), fill=fill, outline=out, radius=22, width=4 if act else 2)
        if done: check_badge(d, x + 44, y + 48, 18)
        else: number_badge(d, x + 44, y + 48, str(i), 18, fill=BLUE if (act or stage == 0) else FUTURE)
        d.text((x + 76, y + 48), f"Stage {i}", font=_font(19, True), fill=out if act or done else MUTED, anchor="lm")
        d.text((x + 30, y + 92), short, font=_font(32, True), fill=INK if (act or stage == 0) else MUTED)
        io_txt = io.replace("  ", " ")
        d.text((x + 30, y + 145), io_txt, font=_font(17, True), fill=INK if (act or stage == 0) else FUTURE_TEXT)
        d.text((x + 30, y + 182), examples[i - 1], font=_font(16), fill=MUTED if (act or stage == 0) else FUTURE_TEXT)
        if i < 3:
            arrow(d, (x + bw + 14, y + bh / 2), (x + bw + gap - 14, y + bh / 2), BLUE if stage == 0 else GREEN if i < stage else FUTURE, 5, 20)
        x += bw + gap
    # Semantic colour key, used identically in every stage.
    if stage in (0, 1):
        ky = 830; f = _font(19, True)
        keys = (("work-end", (245, 75, 30)), ("handle", (20, 190, 55)), ("target", (220, 175, 0)))
        total = sum(d.textlength(k, font=f) + 70 for k, _ in keys) + 30 * 2
        kx = 960 - total / 2
        for label, col in keys:
            d.ellipse((kx, ky, kx + 22, ky + 22), fill=col); d.text((kx + 32, ky - 2), label, font=f, fill=INK)
            kx += d.textlength(label, font=f) + 100
        text_center(d, 960, 872, "semantic colours used throughout the video", _font(16), MUTED)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def io_footer(d, box, src, dst):
    """Input → output summary at the bottom of an explanation panel."""
    x0,y0,x1,y1=box; fy=y1-112
    _rounded(d,(x0+26,fy,x1-26,y1-26),fill=WHITE,outline=BORDER,radius=14,width=1)
    f=_font(14,True); fv=_font(18,True)
    d.text((x0+48,fy+14),"IN",font=f,fill=FUTURE_TEXT); d.text((x0+48,fy+40),src,font=fv,fill=INK)
    mid=(x0+x1)/2+20
    arrow(d,(mid-34,fy+50),(mid+6,fy+50),BLUE,3,11)
    d.text((mid+24,fy+14),"OUT",font=f,fill=FUTURE_TEXT); d.text((mid+24,fy+40),dst,font=fv,fill=BLUE)
