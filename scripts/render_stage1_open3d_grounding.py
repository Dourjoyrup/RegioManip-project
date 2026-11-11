#!/usr/bin/env python3
"""Continue the RGB-D video with DINO, SAM 2, and skeleton overlays in 3D."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("OPEN3D_CPU_RENDERING", "true")
os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "true")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import cv2
import imageio_ffmpeg
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "scripts"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from render_contribution_video_step1 import draw_header, ffmpeg_quality_args, wrap_text, BLUE, INK, MUTED, _font, _rounded, _smoothstep
from render_stage1_open3d_rgbd import VIEW_BG, _camera_frustum, _line_set
from video_style import TimeMap, Narrator, apply_chrome, fade, STAGE1_STEPS, io_footer
from regiomanip.perception.depth_cleaning import statistical_inlier_mask


GREEN = (30, 143, 86)
ORANGE = (220, 175, 0)
CYAN_RGB = np.array([0.05, 0.82, 0.95], np.float32)
HANDLE_RGB = np.array([0.10, 0.72, 1.00], np.float32)
WORKEND_RGB = np.array([0.98, 0.25, 0.55], np.float32)
ORANGE_RGB = np.array([1.00, 0.86, 0.02], np.float32)
SKELETON_RGB = np.array([0.94, 1.0, 0.96], np.float32)
CANDIDATE1_RGB = np.array([1.00, 0.20, 0.06], np.float32)
CANDIDATE2_RGB = np.array([0.04, 0.84, 0.10], np.float32)
# Borders/topology need more luminance contrast than translucent mask fills.
LINE_CANDIDATE1_RGB = np.array([0.82, 0.055, 0.015], np.float32)
LINE_CANDIDATE2_RGB = np.array([0.00, 0.48, 0.055], np.float32)
LINE_TARGET_RGB = np.array([0.82, 0.46, 0.00], np.float32)
# Display-only correction for the lifted skeleton; masks and borders remain at
# their calibrated RGB-D positions.


# Display-only camera distance (was 0.52); the scene now fills more of the viewport.
STAGE1_DIST = .46


def _ease(value: float) -> float:
    return _smoothstep(float(np.clip(value, 0, 1)))


def _mask(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if image is None:
        raise FileNotFoundError(path)
    if image.ndim == 3:
        return np.any(image > 20, axis=2).astype(np.uint8)
    return (image > 20).astype(np.uint8)


def _cloud_with_pixels(rgb: np.ndarray, depth: np.ndarray, stride: int = 5):
    ys, xs = np.mgrid[0 : depth.shape[0] : stride, 0 : depth.shape[1] : stride]
    z = depth[::stride, ::stride]
    valid = np.isfinite(z) & (z >= 0.05) & (z <= 1.5)
    points = np.column_stack((((xs - 320.0) * z / 615.0)[valid], ((ys - 240.0) * z / 615.0)[valid], z[valid])).astype(np.float32)
    colors = rgb[::stride, ::stride][valid][:, ::-1].astype(np.float32) / 255.0
    pixels = np.column_stack((xs[valid], ys[valid])).astype(np.int32)
    keep, _ = statistical_inlier_mask(points, neighbors=12, mad_scale=4.5)
    return points[keep], colors[keep], pixels[keep]


def _backproject(u: float, v: float, z: float) -> np.ndarray:
    return np.array([(u - 320.0) * z / 615.0, (v - 240.0) * z / 615.0, z], dtype=np.float64)


def _bbox_lines(box: List[int], depth: np.ndarray, mask: np.ndarray, color: np.ndarray) -> o3d.geometry.LineSet:
    x1, y1, x2, y2 = box
    region = depth[y1:y2, x1:x2]
    m = mask[y1:y2, x1:x2] > 0
    values = region[m & np.isfinite(region) & (region > 0.05)]
    z = float(np.median(values)) - 0.004 if values.size else 0.3
    # Short corner brackets read as a detector box without covering the object.
    dx, dy = (x2-x1)*0.19, (y2-y1)*0.19
    uv = [(x1,y1),(x1+dx,y1),(x1,y1+dy), (x2,y1),(x2-dx,y1),(x2,y1+dy),
          (x2,y2),(x2-dx,y2),(x2,y2-dy), (x1,y2),(x1+dx,y2),(x1,y2-dy)]
    points = np.asarray([_backproject(u, v, z) for u,v in uv])
    lines = np.asarray([[0,1],[0,2],[3,4],[3,5],[6,7],[6,8],[9,10],[9,11]])
    return _line_set(points, lines, np.tile(color[None, :], (len(lines), 1)))


def _morphological_skeleton(mask: np.ndarray) -> np.ndarray:
    image = (mask > 0).astype(np.uint8) * 255
    skeleton = np.zeros_like(image)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image) > 0:
        opened = cv2.morphologyEx(image, cv2.MORPH_OPEN, element)
        skeleton = cv2.bitwise_or(skeleton, cv2.subtract(image, opened))
        image = cv2.erode(image, element)
    return skeleton > 0


def _skeleton_lines(mask: np.ndarray, depth: np.ndarray, color: np.ndarray) -> o3d.geometry.LineSet:
    # A stable principal centerline is clearer in motion than a noisy pixel skeleton.
    ys, xs = np.where(mask > 0)
    uv = np.column_stack((xs, ys)).astype(np.float64)
    center = uv.mean(axis=0)
    _, _, vh = np.linalg.svd(uv - center, full_matrices=False)
    axis = vh[0]
    projection = (uv - center) @ axis
    lo, hi = np.percentile(projection, (8, 92))
    samples = center[None, :] + np.linspace(lo, hi, 18)[:, None] * axis[None, :]
    points=[]
    for u,v in samples:
        u0,v0=int(round(u)),int(round(v))
        patch=depth[max(0,v0-3):v0+4,max(0,u0-3):u0+4]
        valid=patch[np.isfinite(patch) & (patch>.05) & (patch<1.5)]
        z=float(np.median(valid)) if valid.size else float(np.nanmedian(depth[mask>0]))
        points.append(_backproject(u,v,z))
    points=np.asarray(points)
    lines=np.column_stack((np.arange(len(points)-1),np.arange(1,len(points))))
    return _line_set(points, lines, np.tile(color[None,:],(len(lines),1)))


def _border_lines(mask: np.ndarray, depth: np.ndarray, color: np.ndarray) -> o3d.geometry.LineSet:
    contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contour = max(contours, key=cv2.contourArea)[:, 0, :]
    contour = contour[::2]
    points=[]
    fallback=float(np.nanmedian(depth[mask>0]))
    for u,v in contour:
        y1,y2=max(0,v-3),min(depth.shape[0],v+4); x1,x2=max(0,u-3),min(depth.shape[1],u+4)
        patch=depth[y1:y2,x1:x2]; inside=mask[y1:y2,x1:x2]>0
        valid=patch[inside & np.isfinite(patch) & (patch>.05) & (patch<1.5)]
        z=float(np.median(valid)) if valid.size else fallback
        points.append(_backproject(float(u),float(v),z))
    points=np.asarray(points)
    lines=np.column_stack((np.arange(len(points)),np.roll(np.arange(len(points)),-1)))
    return _line_set(points,lines,np.tile(color[None,:],(len(lines),1)))


def _medial_skeleton_lines(mask: np.ndarray, depth: np.ndarray, color: np.ndarray) -> o3d.geometry.LineSet:
    skel = _morphological_skeleton(mask)
    ys,xs=np.where(skel)
    fallback=float(np.nanmedian(depth[mask>0]))-.016
    z=[]
    for x,y in zip(xs,ys):
        y1,y2=max(0,y-2),min(depth.shape[0],y+3); x1,x2=max(0,x-2),min(depth.shape[1],x+3)
        patch=depth[y1:y2,x1:x2]; inside=mask[y1:y2,x1:x2]>0
        valid=patch[inside & np.isfinite(patch) & (patch>.05) & (patch<1.5)]
        z.append(float(np.median(valid))-.016 if valid.size else fallback)
    z=np.asarray(z)
    points=np.column_stack(((xs-320.0)*z/615.0,(ys-240.0)*z/615.0,z))
    lookup={(int(x),int(y)):i for i,(x,y) in enumerate(zip(xs,ys))}
    lines=[]
    for i,(x,y) in enumerate(zip(xs,ys)):
        for dx,dy in ((1,0),(0,1),(1,1),(-1,1)):
            j=lookup.get((int(x+dx),int(y+dy)))
            if j is not None: lines.append((i,j))
    return _line_set(points,np.asarray(lines,dtype=np.int32),np.tile(color[None,:],(len(lines),1)))


def _shape_skeleton_lines(mask: np.ndarray, depth: np.ndarray, color: np.ndarray, kind: str) -> o3d.geometry.LineSet:
    ys,xs=np.where(mask>0); uv=np.column_stack((xs,ys)).astype(np.float64)
    distance=cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5)
    cy,cx=np.unravel_index(int(np.argmax(distance)),distance.shape)
    center=np.array([float(cx),float(cy)])
    if kind == "blade":
        contour=max(cv2.findContours((mask>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],key=cv2.contourArea)
        polygon=cv2.approxPolyDP(contour,.025*cv2.arcLength(contour,True),True)[:,0,:].astype(np.float64)
        angle=np.arctan2(polygon[:,1]-center[1],polygon[:,0]-center[0])
        anchors=[]
        for target_angle in (-3*np.pi/4,-np.pi/4,np.pi/4,3*np.pi/4):
            delta=np.abs(np.angle(np.exp(1j*(angle-target_angle))))
            anchors.append(polygon[int(np.argmin(delta))])
        uv_points=np.vstack((center,np.asarray(anchors))); edges=np.asarray([[0,1],[0,2],[0,3],[0,4]])
    else:
        mean=uv.mean(axis=0); _,_,vh=np.linalg.svd(uv-mean,full_matrices=False); axis=vh[0]
        if axis[1] < 0: axis=-axis
        normal=np.array([-axis[1],axis[0]])
        p=(uv-mean)@axis; lo,hi=np.percentile(p,(7,93)); ends=[mean+lo*axis,mean+hi*axis]
        midpoint=(ends[0]+ends[1])*.5
        uv_points=[ends[0],midpoint,ends[1]]; edges=[[0,1],[1,2]]
        for end,band in ((ends[0],p<np.percentile(p,14)),(ends[1],p>np.percentile(p,86))):
            q=(uv[band]-end)@normal
            for value in (np.percentile(q,12),np.percentile(q,88)):
                uv_points.append(end+value*normal); edges.append([0 if np.allclose(end,ends[0]) else 2,len(uv_points)-1])
        uv_points=np.asarray(uv_points); edges=np.asarray(edges)
    points=[]; fallback=float(np.nanmedian(depth[mask>0]))
    for u,v in uv_points:
        u0,v0=int(round(u)),int(round(v)); y1,y2=max(0,v0-3),min(depth.shape[0],v0+4); x1,x2=max(0,u0-3),min(depth.shape[1],u0+4)
        patch=depth[y1:y2,x1:x2]; inside=mask[y1:y2,x1:x2]>0
        valid=patch[inside&np.isfinite(patch)&(patch>.05)&(patch<1.5)]; z=float(np.median(valid)) if valid.size else fallback
        points.append(_backproject(u,v,z))
    return _line_set(np.asarray(points),edges,np.tile(color[None,:],(len(edges),1)))


def _lift_toward_eye(geometry: o3d.geometry.LineSet, eye: np.ndarray, distance: float=.012) -> None:
    points=np.asarray(geometry.points); rays=eye[None,:]-points
    rays/=np.maximum(np.linalg.norm(rays,axis=1,keepdims=True),1e-9)
    geometry.points=o3d.utility.Vector3dVector(points+distance*rays)


class GroundingRenderer:
    def __init__(self, width, height, points, colors, pixels, depth, masks, boxes):
        self.width, self.height = width, height
        self.points, self.colors, self.pixels = points, colors, pixels
        self.center = np.median(points, axis=0)
        self.depth, self.masks, self.boxes = depth, masks, boxes
        self.renderer = o3d.visualization.rendering.OffscreenRenderer(width, height)
        # Publication-style white viewport; the point cloud remains the central 3D visual.
        self.renderer.scene.set_background(np.array([1.0, 1.0, 1.0, 1.0],dtype=np.float32))
        self.point_mat = o3d.visualization.rendering.MaterialRecord(); self.point_mat.shader = "defaultUnlit"; self.point_mat.point_size = 4.0
        self.highlight_mat = o3d.visualization.rendering.MaterialRecord(); self.highlight_mat.shader = "defaultUnlit"; self.highlight_mat.point_size = 6.0
        # Keep the topology legible after the 1180x720 viewport is scaled into
        # the 1080p composition. Borders lead; medial axes remain subordinate.
        self.skeleton_mat = o3d.visualization.rendering.MaterialRecord(); self.skeleton_mat.shader = "unlitLine"; self.skeleton_mat.line_width = 3.0
        self.border_mat = o3d.visualization.rendering.MaterialRecord(); self.border_mat.shader = "unlitLine"; self.border_mat.line_width = 4.0
        self.line_mat = o3d.visualization.rendering.MaterialRecord(); self.line_mat.shader = "unlitLine"; self.line_mat.line_width = 1.5
        # DINO corner brackets were 1.5 px and nearly invisible at 1080p.
        self.bracket_mat = o3d.visualization.rendering.MaterialRecord(); self.bracket_mat.shader = "unlitLine"; self.bracket_mat.line_width = 4.0
        self.frustum = _camera_frustum()
        self.renderer.scene.add_geometry("frustum", self.frustum, self.line_mat)
        axes = _line_set(np.array([[0,0,0],[.045,0,0],[0,.045,0],[0,0,.045]]), np.array([[0,1],[0,2],[0,3]]),
                         np.array([[.92,.18,.15],[.15,.78,.30],[.18,.45,.96]]))
        self.renderer.scene.add_geometry("axes", axes, self.line_mat)
        self.tool_sel = masks["tool"][pixels[:, 1], pixels[:, 0]] > 0
        self.target_sel = masks["target"][pixels[:, 1], pixels[:, 0]] > 0
        self.candidate1_sel = masks["workend"][pixels[:, 1], pixels[:, 0]] > 0
        self.candidate2_sel = masks["handle"][pixels[:, 1], pixels[:, 0]] > 0
        self.candidate1_skeleton = _shape_skeleton_lines(masks["workend"],depth,LINE_CANDIDATE1_RGB,"blade")
        self.candidate2_skeleton = _shape_skeleton_lines(masks["handle"],depth,LINE_CANDIDATE2_RGB,"handle")
        self.target_skeleton = _skeleton_lines(masks["target"], depth, LINE_TARGET_RGB)
        self.candidate1_border = _border_lines(masks["workend"],depth,LINE_CANDIDATE1_RGB)
        self.candidate2_border = _border_lines(masks["handle"],depth,LINE_CANDIDATE2_RGB)
        self.target_border = _border_lines(masks["target"], depth, LINE_TARGET_RGB)
        yaw,pitch,dist=np.deg2rad(15.0),np.deg2rad(2.0),STAGE1_DIST
        final_eye=self.center+np.array([dist*np.sin(yaw)*np.cos(pitch),-dist*np.sin(pitch),-dist*np.cos(yaw)*np.cos(pitch)])
        for geometry in (self.candidate1_skeleton,self.candidate2_skeleton,self.target_skeleton):
            _lift_toward_eye(geometry,final_eye)
        for geometry in (self.candidate1_border,self.candidate2_border,self.target_border):
            _lift_toward_eye(geometry,final_eye,.010)

    def _add_cloud(self, name, points, colors, material):
        cloud = o3d.geometry.PointCloud(); cloud.points = o3d.utility.Vector3dVector(points.astype(np.float64)); cloud.colors = o3d.utility.Vector3dVector(colors.astype(np.float64))
        self.renderer.scene.add_geometry(name, cloud, material)

    def _render_checked(self) -> np.ndarray:
        """Render, rejecting the occasional incomplete offscreen frame.

        Headless Open3D sometimes returns a mostly black or completely empty
        image for a single frame, which showed up as a black flash / blink.
        Valid frames have a white background (few dark pixels) and visible
        geometry; anything else is re-rendered.
        """
        best=None
        for attempt in range(8):
            image=np.asarray(self.renderer.render_to_image())[:,:,:3]
            gray=image.mean(axis=2)
            dark=float((gray<60).mean()); content=float((gray<235).mean())
            if dark<.05 and content>.005:
                return image
            if best is None or dark<best[0]:
                best=(dark,image)
        return best[1]

    def _capture_white(self) -> np.ndarray:
        image=self._render_checked().astype(np.float32)
        samples=np.concatenate((image[:24,:24].reshape(-1,3),image[:24,-24:].reshape(-1,3),
                                image[-24:,:24].reshape(-1,3),image[-24:,-24:].reshape(-1,3)),axis=0)
        clear=np.median(samples,axis=0); distance=np.linalg.norm(image-clear[None,None,:],axis=2)
        foreground=np.clip((distance-3.0)/16.0,0.0,1.0)[...,None]
        image=255.0*(1.0-foreground)+image*foreground
        return cv2.cvtColor(np.uint8(np.clip(image,0,255)),cv2.COLOR_RGB2BGR)

    def render_decomposition(self, mode: str) -> np.ndarray:
        """Render one isolated evidence layer from the fixed final camera pose."""
        for name in ("cloud","candidate1","candidate2","target","border1","border2","border_target","sk1","sk2","sk_target","frustum","axes"):
            if self.renderer.scene.has_geometry(name): self.renderer.scene.remove_geometry(name)
        if mode == "depth":
            z=self.points[:,2]; lo,hi=np.percentile(z,(2,98)); scalar=np.uint8(np.clip((z-lo)/max(hi-lo,1e-6),0,1)*255)
            depth_bgr=cv2.applyColorMap(255-scalar[:,None],cv2.COLORMAP_TURBO)[:,0,:]
            colors=depth_bgr[:,::-1].astype(np.float32)/255.0
        else:
            colors=self.colors.copy()
            if mode in ("masks","skeleton","target"): colors*=.62 if mode!="target" else .45
        self._add_cloud("cloud",self.points,colors,self.point_mat)
        if mode in ("masks","skeleton"):
            opacity=.38
            c1=(1-opacity)*self.colors[self.candidate1_sel]+opacity*CANDIDATE1_RGB
            c2=(1-opacity)*self.colors[self.candidate2_sel]+opacity*CANDIDATE2_RGB
            tc=(1-opacity)*self.colors[self.target_sel]+opacity*ORANGE_RGB
            self._add_cloud("candidate1",self.points[self.candidate1_sel]-[0,0,.004],c1,self.highlight_mat)
            self._add_cloud("candidate2",self.points[self.candidate2_sel]-[0,0,.004],c2,self.highlight_mat)
            self._add_cloud("target",self.points[self.target_sel]-[0,0,.003],tc,self.highlight_mat)
        if mode == "target":
            tc=.65*self.colors[self.target_sel]+.35*ORANGE_RGB
            self._add_cloud("target",self.points[self.target_sel]-[0,0,.003],tc,self.highlight_mat)
            self.renderer.scene.add_geometry("border_target",self.target_border,self.border_mat)
        if mode == "skeleton":
            for name,geometry,material in (("border1",self.candidate1_border,self.border_mat),("border2",self.candidate2_border,self.border_mat),
                                           ("border_target",self.target_border,self.border_mat),("sk1",self.candidate1_skeleton,self.skeleton_mat),
                                           ("sk2",self.candidate2_skeleton,self.skeleton_mat),("sk_target",self.target_skeleton,self.skeleton_mat)):
                self.renderer.scene.add_geometry(name,geometry,material)
        yaw,pitch,dist=np.deg2rad(15.0),np.deg2rad(2.0),STAGE1_DIST
        eye=self.center+[dist*np.sin(yaw)*np.cos(pitch),-dist*np.sin(pitch),-dist*np.cos(yaw)*np.cos(pitch)]
        self.renderer.setup_camera(47.0,self.center,eye,[0,-1,0])
        return self._capture_white()

    def render(self, settle: float, dino: float, sam: float, skeleton: float) -> np.ndarray:
        for name in ("cloud", "candidate1", "candidate2", "target", "box_tool", "box_target", "border1", "border2", "border_target", "sk1", "sk2", "sk_target"):
            if self.renderer.scene.has_geometry(name): self.renderer.scene.remove_geometry(name)
        # Calibration aids disappear before detection so they cannot be read as boxes.
        if dino > 0.01:
            for name in ("frustum", "axes"):
                if self.renderer.scene.has_geometry(name): self.renderer.scene.remove_geometry(name)
        # Preserve the natural RGB surface beneath the publication-style overlay.
        base_colors = self.colors * (1.0 - 0.28 * sam)
        self._add_cloud("cloud", self.points, base_colors, self.point_mat)
        if sam > 0.01:
            # Candidate regions are numbered only; semantic roles are inferred later.
            # Use the same semantic palette as the 2D mask grid and decoded
            # VLM output, while retaining enough RGB texture to read as 3D.
            opacity = 0.55 * sam
            c1 = (1-opacity)*self.colors[self.candidate1_sel] + opacity*CANDIDATE1_RGB
            c2 = (1-opacity)*self.colors[self.candidate2_sel] + opacity*CANDIDATE2_RGB
            pc = (1-opacity)*self.colors[self.target_sel] + opacity*ORANGE_RGB
            self._add_cloud("candidate1",self.points[self.candidate1_sel]-np.array([0,0,.004]),c1,self.highlight_mat)
            self._add_cloud("candidate2",self.points[self.candidate2_sel]-np.array([0,0,.004]),c2,self.highlight_mat)
            self._add_cloud("target", self.points[self.target_sel]-np.array([0,0,.003]), pc, self.highlight_mat)
        if dino > 0.01 and sam < 0.01:
            self.renderer.scene.add_geometry("box_tool", _bbox_lines(self.boxes["tool"], self.depth, self.masks["tool"], 1-(1-CYAN_RGB)*dino), self.bracket_mat)
            self.renderer.scene.add_geometry("box_target", _bbox_lines(self.boxes["target"], self.depth, self.masks["target"], 1-(1-ORANGE_RGB)*dino), self.bracket_mat)
        if skeleton > 0.01:
            self.renderer.scene.add_geometry("border1",self.candidate1_border,self.border_mat)
            self.renderer.scene.add_geometry("border2",self.candidate2_border,self.border_mat)
            self.renderer.scene.add_geometry("border_target",self.target_border,self.border_mat)
        # Delay medial axes until the borders are established. This makes the
        # topology construction readable as border -> skeleton, not one flash.
        if skeleton > 0.55:
            self.renderer.scene.add_geometry("sk1",self.candidate1_skeleton,self.skeleton_mat)
            self.renderer.scene.add_geometry("sk2",self.candidate2_skeleton,self.skeleton_mat)
            self.renderer.scene.add_geometry("sk_target", self.target_skeleton, self.skeleton_mat)
        # drift_deg: slow display-only orbit so held 3D views never look frozen.
        yaw = np.deg2rad(22.0 - 7.0 * settle + getattr(self, "drift_deg", 0.0))
        pitch = np.deg2rad(4.0 - 2.0 * settle)
        dist = STAGE1_DIST
        eye = self.center + [dist*np.sin(yaw)*np.cos(pitch), -dist*np.sin(pitch), -dist*np.cos(yaw)*np.cos(pitch)]
        self.renderer.setup_camera(47.0, self.center, eye, [0,-1,0])
        return self._capture_white()


# Shared layout with Stage 3: explanation panel left, 3D viewport right.
VIEW_X, VIEW_Y = 650, 228
PANEL_BOX = (65, 240, 600, 940)
CYAN_UI = (0, 125, 170); AMBER_UI = (215, 125, 0)
CAND1_UI, CAND2_UI, TARGET_UI = (245, 75, 30), (20, 190, 55), (220, 175, 0)


def _panel_header(draw, kicker, title, body):
    x0,y0,x1,y1=PANEL_BOX
    _rounded(draw,PANEL_BOX,fill=(247,249,252),outline=(198,210,221),radius=22,width=2)
    draw.text((x0+32,y0+30),kicker,font=_font(16,True),fill=BLUE)
    draw.text((x0+32,y0+60),title,font=_font(30,True),fill=INK)
    draw.multiline_text((x0+32,y0+112),wrap_text(draw,body,_font(19),x1-x0-64),font=_font(19),fill=MUTED,spacing=8)


def _bar_row(draw, y, label, value, color, alpha=1.0):
    x0=PANEL_BOX[0]+32; x1=PANEL_BOX[2]-32
    draw.ellipse((x0,y+6,x0+18,y+24),fill=color)
    draw.text((x0+30,y+2),label,font=_font(20,True),fill=INK)
    sc=f"{value:.2f}"; f=_font(20,True)
    draw.text((x1-draw.textlength(sc,font=f),y+2),sc,font=f,fill=color)
    draw.rounded_rectangle((x0,y+40,x1,y+52),radius=6,fill=(222,230,238))
    w=int((x1-x0)*value*alpha)
    if w>0: draw.rounded_rectangle((x0,y+40,x0+w,y+52),radius=6,fill=color)


def compose(t, duration, view, dino_data, dino, sam, skeleton, intr):
    frame = np.full((1080,1920,3),255,np.uint8)
    frame[VIEW_Y:VIEW_Y+720,VIEW_X:VIEW_X+1180]=view
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
    draw_header(draw,"Stage 1. Region-level affordance grounding","2D predictions are lifted onto the metric RGB-D observation, so every region can be checked in 3D.")
    x0=PANEL_BOX[0]+32
    if dino<.05:
        _panel_header(draw,"STEP 1.2 · 3D SCENE","Calibrated point cloud",
                      "Each depth pixel is back-projected with the camera intrinsics, giving metric 3D points.")
        rows=(("focal length",f"fx = fy = {intr['fx']:.0f} px"),("principal point",f"({intr['cx']:.0f}, {intr['cy']:.0f}) px"),("valid depth","0.05 – 1.5 m"))
        for k,(a,b) in enumerate(rows):
            y=440+k*70; _rounded(draw,(x0-6,y,PANEL_BOX[2]-26,y+54),fill=(255,255,255),outline=(214,223,231),radius=12,width=1)
            draw.text((x0+14,y+15),a,font=_font(18),fill=MUTED); f=_font(19,True)
            draw.text((PANEL_BOX[2]-46-draw.textlength(b,font=f),y+14),b,font=f,fill=INK)
    elif sam<.45:
        _panel_header(draw,"STEP 1.3 · DETECT","Grounding DINO",
                      "Open-vocabulary detection of the tool and target named in the instruction. Boxes are shown as 3D corner brackets.")
        _bar_row(draw,450,"scraper  (tool)",dino_data['tool_score'],CYAN_UI,dino)
        _bar_row(draw,540,"putty  (target)",dino_data['target_score'],AMBER_UI,dino)
        draw.text((x0,640),"detection confidence",font=_font(16),fill=MUTED)
    else:
        if skeleton<.05:
            _panel_header(draw,"STEP 1.4 · SEGMENT","SAM 2 regions",
                          "Each detection becomes a pixel mask, lifted to 3D points. The tool splits into two numbered candidate regions.")
        else:
            _panel_header(draw,"STEP 1.5 · SKELETONS","Borders + medial skeletons",
                          "Borders and medial axes summarise region shape. No semantic roles are assigned yet; that is the VLM's job.")
        rows=(("candidate 1","tool region",CAND1_UI),("candidate 2","tool region",CAND2_UI),("target","putty region",TARGET_UI))
        for k,(name,desc,col) in enumerate(rows):
            y=450+k*82; _rounded(draw,(x0-6,y,PANEL_BOX[2]-26,y+64),fill=(255,255,255),outline=col,radius=14,width=2)
            draw.rounded_rectangle((x0+12,y+18,x0+40,y+46),radius=6,fill=col)
            draw.text((x0+58,y+12),name,font=_font(20,True),fill=INK); draw.text((x0+58,y+37),desc,font=_font(16),fill=MUTED)
            if skeleton>.05:
                lx=PANEL_BOX[2]-170
                draw.line((lx,y+24,lx+48,y+24),fill=col,width=6); draw.text((lx+58,y+14),"border",font=_font(15),fill=MUTED)
                if skeleton>.55: draw.line((lx,y+46,lx+48,y+46),fill=col,width=3); draw.text((lx+58,y+37),"skeleton",font=_font(15),fill=MUTED)
        if skeleton<.05:
            draw.text((x0,710),"Numbers, not names: roles stay open until the VLM.",font=_font(17),fill=MUTED)
    if dino<.05: io_footer(draw,PANEL_BOX,"depth image","metric point cloud")
    elif sam<.45: io_footer(draw,PANEL_BOX,"RGB + instruction","2 detections")
    elif skeleton<.05: io_footer(draw,PANEL_BOX,"detections","3 regions in 3D")
    else: io_footer(draw,PANEL_BOX,"region masks","shape descriptors")
    # Colour key above the viewport, matching the left panel.
    if dino>.2 and sam<.45:
        pills=((f"scraper {dino_data['tool_score']:.2f}",CYAN_UI),(f"putty {dino_data['target_score']:.2f}",AMBER_UI))
    elif sam>=.45:
        pills=(("candidate 1",CAND1_UI),("candidate 2",CAND2_UI),("target",TARGET_UI))
    else: pills=()
    px=VIEW_X+40
    for label,color in pills:
        f=_font(18,True); w=int(draw.textlength(label,font=f))+56
        _rounded(draw,(px,246,px+w,288),fill=(255,255,255),outline=color,radius=12,width=2)
        draw.ellipse((px+15,261,px+28,274),fill=color); draw.text((px+38,255),label,font=f,fill=color)
        px+=w+16
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def _fit_panel(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """Fit an image inside a white panel without cropping it."""
    scale=min(width/image.shape[1],height/image.shape[0])
    resized=cv2.resize(image,(max(1,int(image.shape[1]*scale)),max(1,int(image.shape[0]*scale))),interpolation=cv2.INTER_AREA)
    panel=np.full((height,width,3),255,np.uint8)
    y=(height-resized.shape[0])//2; x=(width-resized.shape[1])//2
    panel[y:y+resized.shape[0],x:x+resized.shape[1]]=resized
    return panel


def compose_input(rgb: np.ndarray, depth_vis: np.ndarray, instruction: str) -> np.ndarray:
    """Opening multimodal-input screen shown before the 3D reconstruction."""
    frame=np.full((1080,1920,3),255,np.uint8)
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
    draw_header(draw,"Stage 1. Region-level affordance grounding","Input RGB-D observation and task language")
    # Cards are sized to their content (the previous 695 px cards left a
    # ~200 px empty band beneath the images).
    _rounded(draw,(65,250,1060,790),fill=(248,250,252),outline=(180,199,215),radius=22,width=2)
    draw.text((95,275),"RGB-D image",font=_font(27,True),fill=INK)
    draw.text((95,315),"Synchronized appearance and metric depth",font=_font(18),fill=MUTED)
    _rounded(draw,(1100,250,1855,790),fill=(248,250,252),outline=(180,199,215),radius=22,width=2)
    draw.text((1135,275),"Language instruction",font=_font(27,True),fill=INK)
    draw.text((1135,315),"Task-conditioned grounding query",font=_font(18),fill=MUTED)
    frame=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
    # Both inputs are 4:3. Render them directly at 4:3 so no artificial
    # white letterbox rectangles appear above or below the observations.
    # Stronger display exposure to match the clean, bright evidence-card
    # treatment while preserving the full RGB scene and its tabletop context.
    rgb_display=cv2.convertScaleAbs(rgb,alpha=1.24,beta=22)
    frame[375:713,95:545]=cv2.resize(rgb_display,(450,338),interpolation=cv2.INTER_AREA)
    frame[375:713,580:1030]=cv2.resize(depth_vis,(450,338),interpolation=cv2.INTER_AREA)
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
    for cx,label,col in ((320,"RGB",INK),(805,"Depth  (red = near, blue = far)",BLUE)):
        f=_font(19,True); draw.text((cx-draw.textlength(label,font=f)/2,735),label,font=f,fill=col)
    _rounded(draw,(1135,375,1820,713),fill=(255,255,255),outline=(190,205,218),radius=18,width=2)
    draw.text((1175,415),"Instruction",font=_font(18,True),fill=BLUE)
    # The verified instruction stored with this run is intentionally preserved verbatim.
    draw.text((1175,505),f'“{instruction}”',font=_font(30,True),fill=INK)
    draw.text((1175,595),"tool  ·  target  ·  intended interaction",font=_font(18),fill=MUTED)
    banner="RGB  +  depth  +  language  →  grounded 3D candidates"; f=_font(21,True)
    bw=draw.textlength(banner,font=f)+90; bx=960-bw/2
    _rounded(draw,(bx,845,bx+bw,910),fill=(238,246,253),outline=BLUE,radius=14,width=2)
    draw.text((960-draw.textlength(banner,font=f)/2,864),banner,font=f,fill=BLUE)
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",default="results/video/regiomanip_stage1_grounding3d"); ap.add_argument("--fps",type=int,default=15); ap.add_argument("--duration",type=float,default=10.0); ap.add_argument("--intro-duration",type=float,default=3.0); args=ap.parse_args()
    rgb=cv2.imread("data/000/color_000012.png"); raw=np.load("data/000/depth_000012.npy").astype(np.float32); depth=raw*(.001 if np.nanmedian(raw)>10 else 1)
    run=Path("results/regiomanip2/color_000012_run3")
    tool=_mask(run/"dino_sam/sam/masks/scraper_whole.png"); target=_mask(run/"dino_sam/sam/masks/yellow_putty_whole.png")
    handle=_mask(run/"region_affordance/_masks/scraper__handle_fullmask.png"); work=_mask(run/"region_affordance/_masks/scraper__workend_fullmask.png")
    dino_json=json.loads((run/"dino_sam/dino/dino_summary.json").read_text()); objs={o["name"]:o for o in dino_json["objects"]}
    points,colors,pixels=_cloud_with_pixels(rgb,depth)
    tool_box=list(objs["scraper"]["box_xyxy"])
    # Small display-only correction: move the projected tool bracket right/up.
    tool_box=[tool_box[0]+9,tool_box[1]-9,tool_box[2]+9,tool_box[3]-9]
    masks={"tool":tool,"target":target,"handle":handle,"workend":work}; boxes={"tool":tool_box,"target":objs["yellow_putty"]["box_xyxy"]}
    renderer=GroundingRenderer(1180,720,points,colors,pixels,depth,masks,boxes)
    out=Path(args.output_dir); frames=out/"frames"; keys=out/"keyframes"; frames.mkdir(parents=True,exist_ok=True); keys.mkdir(parents=True,exist_ok=True)
    instruction=str(dino_json.get("instruction","scrape the putty by the scraper"))
    valid=np.isfinite(raw)&(raw>0); lo,hi=np.percentile(raw[valid],(2,98)); u8=np.uint8(np.clip((raw-lo)/max(hi-lo,1e-6),0,1)*255); depth_vis=cv2.applyColorMap(255-u8,cv2.COLORMAP_TURBO)
    intro=compose_input(rgb,depth_vis,instruction)
    lift=json.loads((run/"interaction_lift/interaction_lift_results.json").read_text()); intr=lift["intrinsics"]
    fps=args.fps; I=args.intro_duration; src_end=I+args.duration
    # Reading pauses (source time, seconds): input, 3D cloud, detections, masks, skeletons.
    # Reading pauses (source time): input page, 3D cloud, detections, masks.
    tmap=TimeMap(src_end,slow=1.4,holds=((I-.7,.6),(I+1.3,1.2),(I+2.7,.6),(I+5.5,.6)))
    total=int(round(tmap.real_end*fps)); keyframes=[]
    meta={"tool_score":objs["scraper"]["logit"],"target_score":objs["yellow_putty"]["logit"]}
    narr=Narrator(((0,f"Input: one RGB-D frame and the instruction “{instruction}”."),
                   (I,"The depth image is back-projected into a calibrated, metric 3D point cloud."),
                   (I+1.5,"Grounding DINO detects the tool and the target named in the instruction."),
                   (I+4.1,"SAM 2 segments them into two numbered tool regions and one target region."),
                   (I+7.0,"Borders and medial skeletons describe each region's shape; roles are still unassigned.")))
    def params(t):
        return _ease(t/1.2),_ease((t-1.5)/1.0),_ease((t-4.1)/1.2),_ease((t-7.0)/1.2)
    # Discrete 3D changes (frustum removal, brackets → masks, borders, medial
    # axes) are cross-dissolved over BLEND seconds instead of popping in.
    EVENTS=(1.56,4.16,7.06,7.64); BLEND=.45
    real_3d=None; real_end=tmap.real_end
    for i in range(total):
        real=i/fps; global_t=tmap(real); t=max(0.0,global_t-I)
        settle,dino,sam,skeleton=params(t)
        if global_t >= I-.3:
            if real_3d is None: real_3d=real
            u=(real-real_3d)/max(1e-6,real_end-real_3d)
            renderer.drift_deg=3.0*np.sin(np.pi*np.clip(u,0,1))   # returns to 0 for the final frame
            view=renderer.render(settle,dino,sam,skeleton)
            for ev in EVENTS:
                if ev<=t<ev+BLEND:
                    prev=renderer.render(*params(ev-.02)); k=_ease((t-ev)/BLEND)
                    view=cv2.addWeighted(view,k,prev,1-k,0)
            scene=compose(t,args.duration,view,meta,dino,sam,skeleton,intr)
        if global_t < I-.6:
            frame=intro.copy()
            if i < int(.5*fps): frame=fade(frame,_ease(i/(.5*fps)))
        elif global_t < I-.3:
            frame=fade(intro,1-_ease((global_t-(I-.6))/.3))
        elif global_t < I:
            frame=fade(scene,_ease((global_t-(I-.3))/.3))
        else:
            frame=scene
        step=0 if global_t<I else 1 if t<1.5 else 2 if t<4.1 else 3 if t<7.0 else 4
        text,na=narr(global_t)
        frame=apply_chrome(frame,1,STAGE1_STEPS,step,text,na)
        cv2.imwrite(str(frames/f"{i:06d}.png"),frame)
    p=keys/"grounding_4.png"; cv2.imwrite(str(p),frame); keyframes.append(str(p))
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe(); video=out/"regiomanip_grounding_sam_skeleton_3d.mp4"
    subprocess.run([ffmpeg,"-y","-framerate",str(fps),"-i",str(frames/"%06d.png"),*ffmpeg_quality_args(),str(video)],check=True)
    manifest={"video":str(video),"fps":fps,"duration_seconds":total/fps,"frames":total,"resolution":[1920,1080],"stages":["rgbd_language_input","camera_settle","grounding_dino_3d","sam2_3d","skeleton_3d"],"keyframes":keyframes,"scientific_note":"2D predictions projected into the calibrated RGB-D scene for visualization."}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
