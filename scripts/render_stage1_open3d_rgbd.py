#!/usr/bin/env python3
"""Journal-quality Stage 1 RGB-D fusion using Open3D's real renderer."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Tuple

os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "true")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import cv2
import imageio_ffmpeg
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from render_contribution_video_step1 import BLUE, INK, MUTED, _fit_image, _font, _rounded, _smoothstep
from regiomanip.perception.depth_cleaning import statistical_inlier_mask


GREEN = (30, 143, 86)
ORANGE = (235, 126, 46)
VIEW_BG = np.array([0.055, 0.075, 0.11, 1.0], dtype=np.float32)


def _ease(value: float) -> float:
    return _smoothstep(float(np.clip(value, 0, 1)))


def _rgbd_points(rgb_bgr: np.ndarray, depth_m: np.ndarray, stride: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    ys, xs = np.mgrid[0 : depth_m.shape[0] : stride, 0 : depth_m.shape[1] : stride]
    z = depth_m[::stride, ::stride]
    valid = np.isfinite(z) & (z >= 0.05) & (z <= 1.5)
    x = (xs.astype(np.float32) - 320.0) * z / 615.0
    y = (ys.astype(np.float32) - 240.0) * z / 615.0
    points = np.column_stack((x[valid], y[valid], z[valid])).astype(np.float32)
    colors = rgb_bgr[::stride, ::stride][valid][:, ::-1].astype(np.float32) / 255.0
    keep, _ = statistical_inlier_mask(points, neighbors=12, mad_scale=4.5)
    return points[keep], colors[keep]


def _line_set(points: np.ndarray, lines: np.ndarray, colors: np.ndarray) -> o3d.geometry.LineSet:
    obj = o3d.geometry.LineSet()
    obj.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    obj.lines = o3d.utility.Vector2iVector(lines.astype(np.int32))
    obj.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
    return obj


def _camera_frustum() -> o3d.geometry.LineSet:
    z = 0.075
    half_w, half_h = 0.052, 0.039
    points = np.array([[0, 0, 0], [-half_w, -half_h, z], [half_w, -half_h, z], [half_w, half_h, z], [-half_w, half_h, z]])
    lines = np.array([[0, 1], [0, 2], [0, 3], [0, 4], [1, 2], [2, 3], [3, 4], [4, 1]])
    colors = np.tile(np.array([[0.78, 0.85, 0.94]]), (len(lines), 1))
    return _line_set(points, lines, colors)


class SceneRenderer:
    def __init__(self, width: int, height: int, points: np.ndarray, colors: np.ndarray):
        self.width, self.height = width, height
        self.points, self.colors = points, colors
        self.center = np.median(points, axis=0)
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        self.renderer.scene.set_background(VIEW_BG)
        self.point_mat = o3d.visualization.rendering.MaterialRecord()
        self.point_mat.shader = "defaultUnlit"
        self.point_mat.point_size = 4.0
        self.line_mat = o3d.visualization.rendering.MaterialRecord()
        self.line_mat.shader = "unlitLine"
        self.line_mat.line_width = 2.0
        self.renderer.scene.add_geometry("frustum", _camera_frustum(), self.line_mat)
        axis_points = np.array([[0, 0, 0], [0.045, 0, 0], [0, 0.045, 0], [0, 0, 0.045]], dtype=np.float64)
        axis_lines = np.array([[0, 1], [0, 2], [0, 3]], dtype=np.int32)
        axis_colors = np.array([[0.92, 0.18, 0.15], [0.15, 0.78, 0.30], [0.18, 0.45, 0.96]], dtype=np.float64)
        self.renderer.scene.add_geometry("axes", _line_set(axis_points, axis_lines, axis_colors), self.line_mat)

    def render(self, fusion: float, orbit: float) -> np.ndarray:
        if self.renderer.scene.has_geometry("cloud"):
            self.renderer.scene.remove_geometry("cloud")
        # The starting image plane is at a fixed shallow depth along each ray.
        ray_scale = np.clip(0.09 / np.maximum(self.points[:, 2], 1e-5), 0.0, 1.0)
        image_plane = self.points * ray_scale[:, None]
        current = (1.0 - fusion) * image_plane + fusion * self.points
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(current.astype(np.float64))
        cloud.colors = o3d.utility.Vector3dVector(self.colors.astype(np.float64))
        self.renderer.scene.add_geometry("cloud", cloud, self.point_mat)

        yaw = np.deg2rad(-20.0 + 42.0 * orbit)
        pitch = np.deg2rad(-7.0 + 11.0 * orbit)
        distance = 0.52
        eye = self.center + np.array(
            [distance * np.sin(yaw) * np.cos(pitch), -distance * np.sin(pitch), -distance * np.cos(yaw) * np.cos(pitch)]
        )
        self.renderer.setup_camera(47.0, self.center, eye, [0.0, -1.0, 0.0])
        image = np.asarray(self.renderer.render_to_image())
        return cv2.cvtColor(image[:, :, :3], cv2.COLOR_RGB2BGR)


def _depth_visual(depth_m: np.ndarray) -> np.ndarray:
    valid = np.isfinite(depth_m) & (depth_m >= 0.05) & (depth_m <= 1.5)
    lo, hi = np.percentile(depth_m[valid], [2, 98])
    norm = np.clip((depth_m - lo) / max(float(hi - lo), 1e-6), 0, 1)
    out = cv2.applyColorMap(np.uint8(np.round((1.0 - norm) * 255)), cv2.COLORMAP_TURBO)
    out[~valid] = 245
    return out


def compose(
    t: float,
    duration: float,
    rgb: np.ndarray,
    depth_vis: np.ndarray,
    rendered: np.ndarray,
    instruction: str,
) -> np.ndarray:
    frame = np.full((1080, 1920, 3), 255, np.uint8)
    fusion = _ease((t - 2.1) / 2.2)
    alpha = 1.0 - fusion
    viewport = (680, 210, 1180, 720)
    frame[viewport[1] : viewport[1] + viewport[3], viewport[0] : viewport[0] + viewport[2]] = rendered
    if alpha > 0.01:
        a = _fit_image(rgb, 530, 600)
        b = _fit_image(depth_vis, 530, 600)
        frame[250:850, 710:1240] = cv2.addWeighted(a, alpha, frame[250:850, 710:1240], 1 - alpha, 0)
        frame[250:850, 1300:1830] = cv2.addWeighted(b, alpha, frame[250:850, 1300:1830], 1 - alpha, 0)
    pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    draw.text((65, 48), "RegioManip", font=_font(31, True), fill=BLUE)
    draw.text((65, 95), "Stage 1 · RGB-D observation", font=_font(45, True), fill=INK)
    draw.text((67, 155), "Appearance and metric depth are back-projected through the calibrated camera.", font=_font(23), fill=MUTED)
    if fusion < 0.70:
        draw.text((870, 220), "RGB image", font=_font(23, True), fill=BLUE)
        draw.text((1460, 220), "Metric depth", font=_font(23, True), fill=ORANGE)
        _rounded(draw, (755, 872, 1785, 950), fill=(238, 246, 253), outline=BLUE, radius=17, width=2)
        draw.text((785, 895), "Instruction", font=_font(20, True), fill=BLUE)
        draw.text((970, 891), f'“{instruction}”', font=_font(26, True), fill=INK)
    else:
        _rounded(draw, (720, 850, 1065, 910), fill=(255, 255, 255), outline=(196, 210, 223), radius=15, width=2)
        draw.text((755, 867), "Calibrated RGB-D point cloud", font=_font(20, True), fill=INK)
        if t > duration - 1.1:
            _rounded(draw, (1450, 850, 1810, 910), fill=(229, 245, 236), outline=GREEN, radius=15, width=2)
            draw.text((1500, 867), "Ready for grounding  →", font=_font(20, True), fill=GREEN)
    draw.text((68, 950), "Input evidence", font=_font(21, True), fill=BLUE)
    draw.text((68, 984), "RGB-D frame  ·  natural-language instruction  ·  no CAD model", font=_font(20), fill=INK)
    progress = int(1760 * np.clip(t / duration, 0, 1))
    draw.rounded_rectangle((80, 1041, 1840, 1051), radius=5, fill=(226, 232, 238))
    draw.rounded_rectangle((80, 1041, 80 + progress, 1051), radius=5, fill=BLUE)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rgb", default="data/000/color_000012.png")
    parser.add_argument("--depth", default="data/000/depth_000012.npy")
    parser.add_argument("--summary", default="results/regiomanip2/color_000012_run3/run_summary.json")
    parser.add_argument("--output-dir", default="results/video/regiomanip_stage1_open3d")
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--duration", type=float, default=8.0)
    args = parser.parse_args()
    rgb = cv2.imread(args.rgb, cv2.IMREAD_COLOR)
    if rgb is None:
        raise FileNotFoundError(args.rgb)
    raw = np.load(args.depth).astype(np.float32)
    depth_m = raw * (0.001 if np.nanmedian(raw) > 10 else 1.0)
    points, colors = _rgbd_points(rgb, depth_m)
    instruction = json.loads(Path(args.summary).read_text())["instruction"]
    out = Path(args.output_dir)
    frame_dir = out / "frames"
    key_dir = out / "keyframes"
    frame_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    fps, duration = max(1, args.fps), max(6.0, args.duration)
    total = int(round(fps * duration))
    renderer = SceneRenderer(1180, 720, points, colors)
    depth_vis = _depth_visual(depth_m)
    key_indices = [int(v * fps) for v in (1.2, 3.0, 5.3, 7.5)]
    keyframes: List[str] = []
    for index in range(total):
        t = index / fps
        fusion = _ease((t - 2.1) / 2.2)
        orbit = _ease((t - 4.0) / 3.8)
        view = renderer.render(fusion=fusion, orbit=orbit)
        frame = compose(t, duration, rgb, depth_vis, view, instruction)
        frame_path = frame_dir / f"{index:06d}.png"
        cv2.imwrite(str(frame_path), frame)
        if index in key_indices:
            key = key_dir / f"stage1_{len(keyframes) + 1}.png"
            cv2.imwrite(str(key), frame)
            keyframes.append(str(key))

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    video = out / "regiomanip_stage1_open3d.mp4"
    subprocess.run(
        [ffmpeg, "-y", "-framerate", str(fps), "-i", str(frame_dir / "%06d.png"), "-c:v", "libx264",
         "-preset", "slow", "-crf", "18", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
        check=True,
    )
    manifest = {"video": str(video), "frames_dir": str(frame_dir), "fps": fps, "duration_seconds": duration,
                "resolution": [1920, 1080], "pointcloud_points": int(points.shape[0]), "renderer": "Open3D 0.20 / Vulkan llvmpipe",
                "codec": "H.264", "keyframes": keyframes}
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
