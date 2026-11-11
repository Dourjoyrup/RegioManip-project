#!/usr/bin/env python3
"""Opening overview card (fades in/out through white)."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np

from render_contribution_video_step1 import ffmpeg_quality_args
from video_style import ease, fade, title_card


def write_clip(path: Path, cards, fps: int):
    """cards: list of (image, hold_seconds); each fades in/out over 0.5 s."""
    frames = path.parent / (path.stem + "_frames"); frames.mkdir(parents=True, exist_ok=True)
    i = 0
    for image, hold in cards:
        n = int(round(hold * fps)); f_in = int(.5 * fps)
        for k in range(n):
            a = min(1.0, (k + 1) / f_in, (n - k) / f_in)
            cv2.imwrite(str(frames / f"{i:06d}.png"), fade(image, ease(a))); i += 1
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-v", "error", "-framerate", str(fps), "-i",
                    str(frames / "%06d.png"), *ffmpeg_quality_args(), str(path)], check=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--output-dir", required=True)
    ap.add_argument("--fps", type=int, default=15); args = ap.parse_args()
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    # Only the opening overview page; stages are introduced by the persistent
    # top-right stage tracker instead of full-page stage cards.
    write_clip(out / "card_opening.mp4", [(title_card(0), 5.5)], args.fps)


if __name__ == "__main__":
    main()
