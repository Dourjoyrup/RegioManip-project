#!/usr/bin/env python3
"""Publication-style Stage 3: lift contact geometry into ranked SE(3) frames."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw

from render_contribution_video_step1 import BLUE, INK, MUTED, _font, _rounded, draw_header, wrap_text, ffmpeg_quality_args
from video_style import Narrator, apply_chrome, fade, ease, mix, check_badge, number_badge, STAGE3_STEPS, PALE_BLUE, FUTURE, FUTURE_TEXT, io_footer
from render_stage1_open3d_grounding import (
    GroundingRenderer, _cloud_with_pixels, _line_set, _mask, _ease,
    CANDIDATE1_RGB, CANDIDATE2_RGB, ORANGE_RGB,
)

W, H = 1920, 1080
VIEW_X, VIEW_Y, VIEW_W, VIEW_H = 650, 220, 1180, 680
WHITE = (255, 255, 255)
PANEL = (248, 250, 252)
BORDER = (188, 204, 217)
GREEN = (18, 145, 79)
AMBER = (220, 130, 0)
RED = (210, 55, 48)
CYAN = np.array([0.02, 0.66, 0.88])
ORANGE = np.array([0.96, 0.42, 0.08])
RED3 = np.array([0.82, 0.025, 0.015])
GREEN3 = np.array([0.015, 0.52, 0.08])
BLUE3 = np.array([0.015, 0.20, 0.82])


def _raw_cloud(rgb, depth, stride=5):
    """Unfiltered RGB-D cloud used only to visualize the cleaning step."""
    ys,xs=np.mgrid[0:depth.shape[0]:stride,0:depth.shape[1]:stride]
    z=depth[::stride,::stride]
    valid=np.isfinite(z)&(z>=.05)&(z<=1.5)
    points=np.column_stack((((xs-320.0)*z/615.0)[valid],((ys-240.0)*z/615.0)[valid],z[valid])).astype(np.float32)
    colors=rgb[::stride,::stride][valid][:,::-1].astype(np.float32)/255.0
    return points,colors


def _largest_component(mask):
    """Remove disconnected mask islands before metric back-projection."""
    binary=(mask>0).astype(np.uint8)
    count,labels,stats,_=cv2.connectedComponentsWithStats(binary,8)
    if count<=1: return binary
    keep=1+int(np.argmax(stats[1:,cv2.CC_STAT_AREA]))
    return (labels==keep).astype(np.uint8)


def _segments(origin, vectors, colors):
    points = [origin]
    lines = []
    line_colors = []
    for i, (vec, color) in enumerate(zip(vectors, colors), start=1):
        points.append(origin + vec)
        lines.append((0, i))
        line_colors.append(color)
    return _line_set(np.asarray(points), np.asarray(lines), np.asarray(line_colors))


def _plane_grid(center, tangent, bitangent, color, size=.046, steps=4):
    pts, lines, colors = [], [], []
    for axis in (-1, 1):
        pass
    values = np.linspace(-size, size, steps + 1)
    for value in values:
        a = center + value * tangent - size * bitangent
        b = center + value * tangent + size * bitangent
        c = center - size * tangent + value * bitangent
        d = center + size * tangent + value * bitangent
        idx = len(pts); pts += [a, b, c, d]
        lines += [(idx, idx + 1), (idx + 2, idx + 3)]
        colors += [color, color]
    return _line_set(np.asarray(pts), np.asarray(lines), np.asarray(colors))


class Stage3Renderer(GroundingRenderer):
    def __init__(self, *args, lift_data, raw_points, raw_colors, **kwargs):
        super().__init__(*args, **kwargs)
        self.raw_points,self.raw_colors=raw_points,raw_colors
        self.data = lift_data
        tool = lift_data["tool_primitives"][0]
        target = lift_data["target_primitives"][0]
        best = lift_data["best_hypothesis"]
        self.grip = np.asarray(tool["grip_point_3d"], float)
        self.contact = np.asarray(tool["work_end_contact_3d"], float)
        self.tool_axis = np.asarray(tool["axis_direction_3d"], float)
        self.edge_a = np.asarray(tool["work_edge"]["endpoint_a"], float)
        self.edge_b = np.asarray(tool["work_edge"]["endpoint_b"], float)
        self.target_center = np.asarray(target["plane"]["point"], float)
        self.target_touch = np.asarray(target["touch_xyz"], float)
        self.target_edge = np.asarray(target["edge_line"]["axis"], float)
        self.normal = np.asarray(target["plane"]["normal"], float)
        self.tangent = np.asarray(target["plane"]["tangent"], float)
        self.bitangent = np.asarray(target["plane"]["bitangent"], float)
        self.frame_origin = np.asarray(best["translation_xyz"], float)
        self.rotation = np.asarray(best["rotation"], float)
        # Shared upward axis is the outward ground/target-plane normal.
        ground_z=-self.normal/np.linalg.norm(self.normal)
        # Tool red x follows G -> C and is projected into the ground plane.
        tool_x=self.contact-self.grip
        tool_x=tool_x-np.dot(tool_x,ground_z)*ground_z; tool_x/=np.linalg.norm(tool_x)
        tool_y=np.cross(ground_z,tool_x); tool_y/=np.linalg.norm(tool_y)
        self.tool_rotation=np.column_stack((tool_x,tool_y,ground_z))
        # Canonical display convention: +z points out of the target surface
        # (toward the camera/up in this view).  This removes the visually
        # inverted raw camera-coordinate signs without changing stored SE(3).
        z=ground_z
        # Target green y is locked to the measured boundary edge. Red x is
        # its in-plane outward normal, signed from the interior toward T.
        y=self.target_edge-np.dot(self.target_edge,z)*z; y/=np.linalg.norm(y)
        x=np.cross(y,z); x/=np.linalg.norm(x)
        outward=self.target_touch-self.target_center
        outward=outward-np.dot(outward,z)*z
        if np.dot(x,outward)<0:
            x=-x; y=-y
        self.display_rotation=np.column_stack((x,y,z))
        self.score = float(best["score"])
        self.region_sel=self.tool_sel|self.target_sel
        self.geom_mat = o3d.visualization.rendering.MaterialRecord(); self.geom_mat.shader="unlitLine"; self.geom_mat.line_width=5.0
        self.thin_mat = o3d.visualization.rendering.MaterialRecord(); self.thin_mat.shader="unlitLine"; self.thin_mat.line_width=2.5
        self.frame_mat = o3d.visualization.rendering.MaterialRecord(); self.frame_mat.shader="unlitLine"; self.frame_mat.line_width=6.0

    def camera(self, zoom=1.0):
        """Display camera.  After background suppression the view dollies in
        on the metric regions (visualization only; geometry is unchanged)."""
        z=float(np.clip(zoom,0,1))
        region_center=np.median(self.points[self.region_sel],axis=0)
        look=self.center+(region_center-self.center)*z
        yaw,pitch=np.deg2rad(14+getattr(self,"drift_deg",0.0)),np.deg2rad(4); dist=.43+(.305-.43)*z
        eye=look+np.array([dist*np.sin(yaw)*np.cos(pitch),-dist*np.sin(pitch),-dist*np.cos(yaw)*np.cos(pitch)])
        return look,eye

    def _clear(self):
        names=("cloud","candidate1","candidate2","target","tool_axis","work_edge","target_plane","normal","backproject","tool_frame",
               "frame_best","frame_alt1","frame_alt2","frame_alt3","frustum","axes")
        for name in names:
            if self.renderer.scene.has_geometry(name): self.renderer.scene.remove_geometry(name)

    def render_stage3(self, lift, clean, isolate, rays_alpha, primitive, frame, hypotheses, final, zoom=1.0):
        self._clear()
        if clean < .5:
            # Raw sensor cloud: background and unfiltered depth returns.
            self._add_cloud("cloud",self.raw_points,self.raw_colors*.82,self.point_mat)
        elif isolate < .65:
            # Statistical inlier cloud after depth cleaning.
            base=self.colors*(0.88-0.20*min(1.0,primitive))
            self._add_cloud("cloud",self.points,base,self.point_mat)
        else:
            # Background-suppressed metric support used by the lifting stage.
            pts=self.points[self.region_sel]; cols=self.colors[self.region_sel]*(0.92-0.18*min(1.0,primitive))
            self._add_cloud("cloud",pts,cols,self.highlight_mat)
        # Keep the selected regions grounded in their actual RGB-D points.
        opacity = .48 * lift
        if lift > .01:
            c1=(1-opacity)*self.colors[self.candidate1_sel]+opacity*CANDIDATE1_RGB
            c2=(1-opacity)*self.colors[self.candidate2_sel]+opacity*CANDIDATE2_RGB
            tc=(1-opacity)*self.colors[self.target_sel]+opacity*ORANGE_RGB
            self._add_cloud("candidate1",self.points[self.candidate1_sel]-[0,0,.004],c1,self.highlight_mat)
            self._add_cloud("candidate2",self.points[self.candidate2_sel]-[0,0,.004],c2,self.highlight_mat)
            self._add_cloud("target",self.points[self.target_sel]-[0,0,.003],tc,self.highlight_mat)
        if rays_alpha > .08:
            cam=np.array([0.0,0.0,0.0])
            rays=_line_set(np.asarray([cam,self.grip,cam,self.contact,cam,self.frame_origin]),np.asarray([[0,1],[2,3],[4,5]]),
                           np.tile(np.array([[.20,.55,.92]]),(3,1)))
            self.renderer.scene.add_geometry("backproject",rays,self.thin_mat)
        if primitive > .05:
            # Primitives appear one at a time (primitive = number shown, 0..4).
            axis=_segments(self.grip,[self.tool_axis*.105],[CYAN])
            edge=_line_set(np.asarray([self.edge_a,self.edge_b]),np.asarray([[0,1]]),np.asarray([ORANGE]))
            plane=_plane_grid(self.target_center,self.tangent,self.bitangent,np.array([.95,.67,.05]))
            normal=_segments(self.target_center,[self.normal*.075],[GREEN3])
            items=(("tool_axis",axis,self.geom_mat),("work_edge",edge,self.geom_mat),("target_plane",plane,self.thin_mat),("normal",normal,self.geom_mat))
            for k,(name,geom,mat) in enumerate(items):
                if primitive>k+.05: self.renderer.scene.add_geometry(name,geom,mat)
        if frame > .05:
            # Perspective-balanced display lengths: in this camera view the
            # forward/edge axes project long while the ground normal is
            # foreshortened. Directions remain unchanged.
            sx,sy,sz=.026,.030,.060
            # Equal camera-facing offsets and equal axis scales make the tool
            # and target frames visually comparable while retaining true origins.
            _,eye=self.camera(zoom)
            toward_t=eye-self.frame_origin; toward_t/=np.linalg.norm(toward_t)
            target_origin=self.frame_origin+toward_t*.010
            best=_segments(target_origin,[self.display_rotation[:,0]*sx,self.display_rotation[:,1]*sy,self.display_rotation[:,2]*sz],[RED3,GREEN3,BLUE3])
            self.renderer.scene.add_geometry("frame_best",best,self.frame_mat)
            toward_c=eye-self.contact; toward_c/=np.linalg.norm(toward_c)
            tool_origin=self.contact+toward_c*.010
            tool_frame=_segments(tool_origin,[self.tool_rotation[:,0]*sx,self.tool_rotation[:,1]*sy,self.tool_rotation[:,2]*sz],[RED3,GREEN3,BLUE3])
            self.renderer.scene.add_geometry("tool_frame",tool_frame,self.frame_mat)
        if hypotheses is not None and hypotheses >= 0 and frame <= .05:
            # All candidates share the measured T / C / G origins; only their
            # orientation changes during SE(3) ranking.
            rot=self.hypothesis_rotation(int(hypotheses))
            lengths=(.030,.030,.046)
            for name,origin in (("frame_alt1",self.frame_origin),("frame_alt2",self.contact),("frame_alt3",self.grip)):
                alt=_segments(origin,[rot[:,0]*lengths[0],rot[:,1]*lengths[1],rot[:,2]*lengths[2]],[RED3,GREEN3,BLUE3])
                self.renderer.scene.add_geometry(name,alt,self.frame_mat)
        look,eye=self.camera(zoom)
        self.renderer.setup_camera(40.0,look,eye,[0,-1,0])
        return self._capture_white()

    def project(self, point, zoom=1.0):
        """Project a metric point through the exact display camera (viewport px)."""
        look,eye=self.camera(zoom); up=np.array([0.0,-1.0,0.0])
        forward=(look-eye); forward/=np.linalg.norm(forward)
        right=np.cross(forward,up); right/=np.linalg.norm(right)
        true_up=np.cross(right,forward); true_up/=np.linalg.norm(true_up)
        focal=self.height/(2*np.tan(np.deg2rad(40.0)/2))
        rel=np.asarray(point,float)-eye; z=float(np.dot(rel,forward))
        return (self.width/2+focal*float(np.dot(rel,right))/z, self.height/2-focal*float(np.dot(rel,true_up))/z)

    def project_keypoints(self, zoom=1.0):
        return {label:self.project(point,zoom) for label,point in (("G",self.grip),("C",self.contact),("T",self.target_touch))}

    def hypothesis_rotation(self, idx):
        signs=((1,1,1),(1,-1,1),(-1,1,1),(-1,-1,1))
        rot=self.display_rotation.copy()
        for k in range(3): rot[:,k]*=signs[idx][k]
        return rot


PANEL_BOX=(65,240,600,940)
WORK_UI,HANDLE_UI,TARGET_UI=(245,75,30),(20,190,55),(220,175,0)
AXIS_UI=((215,45,40),(20,150,60),(30,95,210))
PRIM_UI=(("tool axis","line fit to the handle",(0,145,190)),("work edge","line fit to the blade edge",(225,100,15)),
         ("target plane","plane fit to the putty",(205,145,0)),("surface normal","outward plane normal",(20,140,78)))

# Source timeline (seconds).  Generous pauses so every result can be read.
T_CLEAN,T_ISOLATE,T_LIFT,T_RAYS,T_ZOOM=2.2,4.2,4.8,6.4,8.4
T_PRIM=(10.0,10.9,11.8,12.7); T_RANK=14.6; HYP_DT=2.4; T_SELECT=T_RANK+4*HYP_DT; T_FRAME=T_SELECT+1.8; T_END=T_FRAME+5.5
DRIFT_DEG=4.0   # slow display-only orbit; zero at the start and the final frame
BLEND=.45       # cross-dissolve for every discrete 3D change


def _panel(d, kicker, title, body):
    x0,y0,x1,y1=PANEL_BOX
    _rounded(d,PANEL_BOX,fill=(247,249,252),outline=(198,210,221),radius=22,width=2)
    d.text((x0+32,y0+30),kicker,font=_font(16,True),fill=BLUE)
    d.text((x0+32,y0+60),title,font=_font(30,True),fill=INK)
    d.multiline_text((x0+32,y0+112),wrap_text(d,body,_font(19),x1-x0-64),font=_font(19),fill=MUTED,spacing=8)


def _check_row(d, y, k, state, title, sub=None, swatch=None):
    """Checklist row; state: 'done' | 'now' | 'todo'."""
    x0=PANEL_BOX[0]+26; x1=PANEL_BOX[2]-26; h=62 if sub else 50
    if state=="now": _rounded(d,(x0,y,x1,y+h),fill=PALE_BLUE,outline=BLUE,radius=12,width=2)
    else: _rounded(d,(x0,y,x1,y+h),fill=WHITE,outline=(214,223,231),radius=12,width=1)
    cy=y+h/2
    if state=="done": check_badge(d,x0+28,cy,14)
    elif state=="now": number_badge(d,x0+28,cy,str(k),14,fill=BLUE)
    else: number_badge(d,x0+28,cy,str(k),14,fill=WHITE,fg=FUTURE_TEXT,outline=FUTURE)
    tx=x0+56
    if swatch is not None:
        d.line((tx,cy,tx+34,cy),fill=swatch if state!="todo" else FUTURE,width=6); tx+=48
    col=INK if state!="todo" else FUTURE_TEXT
    if sub:
        d.text((tx,y+9),title,font=_font(19,True),fill=col); d.text((tx,y+35),sub,font=_font(15),fill=MUTED if state!="todo" else FUTURE_TEXT)
    else:
        d.text((tx,cy),title,font=_font(19,True),fill=col,anchor="lm")


def _tag(d, xy, text, color, anchor_pt=None):
    """Label in the 3D view with an optional leader line to its geometry."""
    f=_font(17,True); x,y=xy; w=d.textlength(text,font=f)+24
    if anchor_pt is not None:
        d.line((anchor_pt[0],anchor_pt[1],x+w/2,y+16),fill=color,width=2)
        d.ellipse((anchor_pt[0]-4,anchor_pt[1]-4,anchor_pt[0]+4,anchor_pt[1]+4),fill=color)
    _rounded(d,(x,y,x+w,y+32),fill=WHITE,outline=color,radius=9,width=2)
    d.text((x+12,y+16),text,font=f,fill=color,anchor="lm")


def _axis_glyph(d, cx, cy, dirs, r=24):
    d.ellipse((cx-r-6,cy-r-6,cx+r+6,cy+r+6),fill=WHITE,outline=(210,219,228),width=1)
    for (dx,dy),col in zip(dirs,AXIS_UI):
        n=max(1e-6,np.hypot(dx,dy)); d.line((cx,cy,cx+dx/n*r,cy+dy/n*r),fill=col,width=4)
    d.ellipse((cx-3,cy-3,cx+3,cy+3),fill=INK)


def compose(view, t, st, renderer, score, translation, hyp_scores, readiness):
    frame=np.full((H,W,3),255,np.uint8)
    frame[VIEW_Y:VIEW_Y+VIEW_H,VIEW_X:VIEW_X+VIEW_W]=cv2.resize(view,(VIEW_W,VIEW_H),interpolation=cv2.INTER_AREA)
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); d=ImageDraw.Draw(pil)
    draw_header(d,"Stage 3. Contact-frame lifting into SE(3)","Grounded regions become metric primitives, ranked SE(3) hypotheses, and one executable contact frame.")
    zoom=st["zoom"]; P=lambda pt:(renderer.project(pt,zoom)[0]+VIEW_X,renderer.project(pt,zoom)[1]+VIEW_Y)
    y0=PANEL_BOX[1]
    # ---------------- left panel -----------------
    if t<T_PRIM[0]-.4:
        _panel(d,"STEP 3.1 · BACK-PROJECT","Metric regions",
               "The grounded regions are separated from the raw sensor data and lifted into metric 3D.")
        ev=(0.0,T_CLEAN,T_ISOLATE,T_RAYS,1e9)
        rows=("Raw RGB-D point cloud","Remove depth outliers","Keep grounded regions","Lift G · C · T to metric 3D")
        for k,r in enumerate(rows):
            state="done" if (t>=ev[k+1] if k<3 else t>=T_ZOOM) else "now" if t>=ev[k] else "todo"
            _check_row(d,y0+250+k*66,k+1,state,r)
        if t>=T_RAYS+.3:
            d.text((PANEL_BOX[0]+32,y0+530),"G  grip      C  contact      T  target touch",font=_font(17,True),fill=MUTED)
        io_footer(d,PANEL_BOX,"raw RGB-D cloud","metric G · C · T")
    elif t<T_RANK-.3:
        _panel(d,"STEP 3.2 · FIT PRIMITIVES","Contact geometry",
               "Simple geometric primitives are fitted to the metric points of each region.")
        for k,(name,sub,col) in enumerate(PRIM_UI):
            state="done" if t>=T_PRIM[k]+.8 else "now" if t>=T_PRIM[k] else "todo"
            _check_row(d,y0+240+k*80,k+1,state,name,sub,col)
        io_footer(d,PANEL_BOX,"region points","4 primitives")
    elif t<T_FRAME:
        cur=min(3,int((t-T_RANK)/HYP_DT)); selecting=t>=T_SELECT
        _panel(d,"STEP 3.3 · RANK SE(3)","Orientation hypotheses",
               "Four sign hypotheses for the frame axes are scored at the same fixed origins T, C, G.")
        for k in range(4):
            y=y0+228+k*84; seen=selecting or k<cur; now=(k==cur and not selecting) or (selecting and k==0)
            x0=PANEL_BOX[0]+26; x1=PANEL_BOX[2]-26
            if selecting and k==0: _rounded(d,(x0,y,x1,y+72),fill=(234,247,239),outline=GREEN,radius=12,width=3)
            elif now: _rounded(d,(x0,y,x1,y+72),fill=PALE_BLUE,outline=BLUE,radius=12,width=2)
            else: _rounded(d,(x0,y,x1,y+72),fill=WHITE,outline=(214,223,231),radius=12,width=1)
            rot=renderer.hypothesis_rotation(k); o=P(renderer.target_touch)
            dirs=[(P(renderer.target_touch+rot[:,j]*.03)[0]-o[0],P(renderer.target_touch+rot[:,j]*.03)[1]-o[1]) for j in range(3)]
            _axis_glyph(d,x0+46,y+36,dirs)
            col=INK if (seen or now) else FUTURE_TEXT
            d.text((x0+92,y+12),f"H{k+1}",font=_font(22,True),fill=GREEN if (selecting and k==0) else BLUE if now else col)
            d.text((x0+92,y+42),("+x +y","+x −y","−x +y","−x −y")[k]+"   z outward",font=_font(15),fill=MUTED)
            if seen or now:
                sc=f"{hyp_scores[k]:.3f}"; f=_font(22,True)
                d.text((x1-20-d.textlength(sc,font=f),y+12),sc,font=f,fill=INK); d.text((x1-20-d.textlength("score",font=_font(14)),y+44),"score",font=_font(14),fill=MUTED)
            if selecting and k==0: check_badge(d,x1-118,y+36,14)
        if selecting:
            _rounded(d,(PANEL_BOX[0]+26,y0+580,PANEL_BOX[2]-26,y0+672),fill=(234,247,239),outline=GREEN,radius=12,width=2)
            d.multiline_text((PANEL_BOX[0]+48,y0+596),wrap_text(d,"All four stored scores are equal; H1, the stored best, is selected.",_font(18,True),PANEL_BOX[2]-PANEL_BOX[0]-96),font=_font(18,True),fill=GREEN,spacing=6)
    else:
        _panel(d,"STEP 3.4 · CONTACT FRAME","Executable SE(3) pose",
               "The selected hypothesis becomes the contact frame passed to the robot controller.")
        x0=PANEL_BOX[0]+26; x1=PANEL_BOX[2]-26; yy=y0+220
        for j,(ax,name) in enumerate((("x","contact direction"),("y","boundary tangent"),("z","outward normal"))):
            d.line((x0+10,yy+16+j*40,x0+44,yy+16+j*40),fill=AXIS_UI[j],width=6)
            d.text((x0+56,yy+4+j*40),ax,font=_font(20,True),fill=AXIS_UI[j]); d.text((x0+84,yy+6+j*40),name,font=_font(18),fill=INK)
        a=ease((t-T_FRAME-.5)/.8)
        if a>0:
            yb=y0+360
            _rounded(d,(x0,yb,x1,yb+300),fill=mix(WHITE,(236,248,241),a),outline=mix(WHITE,GREEN,a),radius=18,width=3)
            c=lambda col:mix(WHITE,col,a)
            d.text((x0+26,yb+22),"FINAL OUTPUT",font=_font(16,True),fill=c(GREEN))
            d.text((x0+26,yb+52),"T*C  ∈  SE(3)",font=_font(36,True),fill=c(INK))
            d.text((x0+26,yb+112),"translation",font=_font(16),fill=c(MUTED))
            d.text((x0+26,yb+136),f"[{translation[0]:+.3f}, {translation[1]:+.3f}, {translation[2]:+.3f}] m",font=_font(22,True),fill=c(INK))
            d.text((x0+26,yb+184),f"confidence  {score:.3f}",font=_font(24,True),fill=c(GREEN))
            d.text((x0+26,yb+218),f"≥ readiness threshold {readiness:.2f}",font=_font(17),fill=c(MUTED))
            _rounded(d,(x0+26,yb+252,x0+250,yb+284),fill=c(GREEN),outline=c(GREEN),radius=9,width=1)
            d.text((x0+138,yb+268),"READY FOR EXECUTION",font=_font(15,True),fill=WHITE if a>.5 else c(GREEN),anchor="mm")
    # ---------------- 3D annotations -----------------
    if st["lift"]>.3:
        px=VIEW_X+30
        for label,col in (("work-end",WORK_UI),("handle",HANDLE_UI),("target",TARGET_UI)):
            f=_font(18,True); w=int(d.textlength(label,font=f))+56
            _rounded(d,(px,246,px+w,288),fill=WHITE,outline=col,radius=12,width=2)
            d.ellipse((px+15,261,px+28,274),fill=col); d.text((px+38,267),label,font=f,fill=col,anchor="lm")
            px+=w+16
    if t>=T_RAYS+.15:
        a=ease((t-T_RAYS-.15)/.4); mf=_font(18,True)
        for label,pt,color in (("G",renderer.grip,GREEN),("C",renderer.contact,RED),("T",renderer.target_touch,(205,150,0))):
            x,y=P(pt); r=17; col=mix(WHITE,color,a)
            d.ellipse((x-r,y-r,x+r,y+r),fill=WHITE,outline=col,width=3)
            d.text((x,y),label,font=mf,fill=col,anchor="mm")
    if T_PRIM[0]<=t<T_RANK:
        anchors=(renderer.grip+renderer.tool_axis*.07,(renderer.edge_a+renderer.edge_b)/2,
                 renderer.target_center-renderer.tangent*.046-renderer.bitangent*.046,renderer.target_center+renderer.normal*.075)
        offsets=((70,-40),(70,30),(-230,40),(-250,-60))
        for k,((name,_,col),pt,off) in enumerate(zip(PRIM_UI,anchors,offsets)):
            if t>=T_PRIM[k]+.25:
                ax,ay=P(pt); _tag(d,(ax+off[0],ay+off[1]),name,col,(ax,ay))
    if T_RANK<=t<T_FRAME:
        cur=0 if t>=T_SELECT else min(3,int((t-T_RANK)/HYP_DT))
        _rounded(d,(VIEW_X+30,880,VIEW_X+520,930),fill=WHITE,outline=GREEN if t>=T_SELECT else BLUE,radius=12,width=2)
        d.text((VIEW_X+50,905),f"showing H{cur+1}  ·  origins fixed at T, C, G" if t<T_SELECT else "selected: H1",font=_font(19,True),fill=GREEN if t>=T_SELECT else BLUE,anchor="lm")
    if t>=T_FRAME+.2:
        tx,ty=P(renderer.target_touch); cx,cy=P(renderer.contact)
        _tag(d,(tx-330,ty-150),"target frame at T",TARGET_UI,(tx-8,ty-8))
        _tag(d,(cx+70,cy+40),"tool frame at C",WORK_UI,(cx+8,cy+8))
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--output-dir",default="results/video/regiomanip_stage3_contact_frame"); ap.add_argument("--fps",type=int,default=15); ap.add_argument("--duration",type=float,default=0.0,help="ignored; the timeline is fixed (T_END)"); args=ap.parse_args()
    run=Path("results/regiomanip2/color_000012_run3")
    rgb=cv2.imread("data/000/color_000012.png"); raw=np.load("data/000/depth_000012.npy").astype(np.float32); depth=raw*(.001 if np.nanmedian(raw)>10 else 1)
    masks={"tool":_largest_component(_mask(run/"dino_sam/sam/masks/scraper_whole.png")),"target":_largest_component(_mask(run/"dino_sam/sam/masks/yellow_putty_whole.png")),
           "handle":_largest_component(_mask(run/"region_affordance/_masks/scraper__handle_fullmask.png")),"workend":_largest_component(_mask(run/"region_affordance/_masks/scraper__workend_fullmask.png"))}
    raw_points,raw_colors=_raw_cloud(rgb,depth)
    points,colors,pixels=_cloud_with_pixels(rgb,depth)
    lift=json.loads((run/"interaction_lift/interaction_lift_results.json").read_text())
    renderer=Stage3Renderer(1180,680,points,colors,pixels,depth,masks,{"tool":[0,0,1,1],"target":[0,0,1,1]},lift_data=lift,raw_points=raw_points,raw_colors=raw_colors)
    out=Path(args.output_dir); frames=out/"frames"; keys=out/"keyframes"; frames.mkdir(parents=True,exist_ok=True); keys.mkdir(parents=True,exist_ok=True)
    total=int(args.duration*args.fps); score=float(lift["best_hypothesis"]["score"]); translation=np.asarray(lift["best_hypothesis"]["translation_xyz"],float); keyframes=[]
    # Stored scores of the selected hypothesis and its three sign alternatives.
    hyp_scores=[score]+[float(a["score"]) for a in lift["best_hypothesis"]["alternatives"]][:3]
    readiness=float(lift.get("readiness_threshold",0.6))
    total=int(round(T_END*args.fps))
    narr=Narrator(((0,"Stage 3 starts from the raw RGB-D point cloud, including background and sensor noise."),
                   (T_CLEAN,"Statistical filtering removes isolated depth outliers."),
                   (T_ISOLATE,"Only the grounded regions are kept: work-end, handle, and target."),
                   (T_RAYS,"Grip G, contact C, and target touch T are back-projected to fixed metric 3D points."),
                   (T_PRIM[0]-.4,"Primitives are fitted one by one: tool axis, work edge, target plane, surface normal."),
                   (T_RANK,"Four orientation hypotheses are scored at the same fixed origins T, C, and G."),
                   (T_SELECT,f"All four stored scores are {score:.3f}; the stored best hypothesis H1 is selected."),
                   (T_FRAME,"The selected contact frame T*C ∈ SE(3) is constructed and ready for execution.")))
    def state(t):
        return dict(clean=_ease((t-T_CLEAN)/.7),isolate=_ease((t-T_ISOLATE)/.7),lift=_ease((t-T_LIFT)/.6),
                rays=_ease((t-T_RAYS)/.4)*(1-_ease((t-(T_ZOOM-.4))/.4)),zoom=_ease((t-T_ZOOM)/1.2),
                prim=(sum(1.0 for k in T_PRIM if t>=k) if T_PRIM[0]<=t<T_RANK else 0.0),
                hyp=(0 if t>=T_SELECT else min(3,int((t-T_RANK)/HYP_DT))) if T_RANK<=t<T_FRAME else -1,
                frame=1.0 if t>=T_FRAME else 0.0)
    def draw3d(st):
        return renderer.render_stage3(st["lift"],st["clean"],st["isolate"],st["rays"],st["prim"],st["frame"],st["hyp"],st["frame"],st["zoom"])
    # Instants where the 3D content changes discretely (raw → clean cloud,
    # background removal, rays on/off, each primitive, each hypothesis, the
    # final frame).  Each is cross-dissolved instead of switching in one frame.
    events=[T_CLEAN+.36,T_ISOLATE+.42,T_RAYS+.08,T_ZOOM-.06,*T_PRIM,T_RANK,
            *[T_RANK+k*HYP_DT for k in (1,2,3)],T_SELECT,T_FRAME]
    for i in range(total):
        t=i/args.fps
        renderer.drift_deg=DRIFT_DEG*np.sin(np.pi*min(1.0,t/T_END))
        st=state(t)
        view=draw3d(st)
        for ev in events:
            if ev<=t<ev+BLEND:
                prev=draw3d(state(ev-.02)); k=_ease((t-ev)/BLEND)
                view=cv2.addWeighted(view,k,prev,1-k,0)
        scene=compose(view,t,st,renderer,score,translation,hyp_scores,readiness)
        step=0 if t<T_PRIM[0]-.4 else 1 if t<T_RANK else 2 if t<T_FRAME else 3
        text,na=narr(t)
        scene=apply_chrome(scene,3,STAGE3_STEPS,step,text,na)
        if t<.5: scene=fade(scene,_ease(t/.5))
        cv2.imwrite(str(frames/f"{i:06d}.png"),scene)
        if i in (int(1*args.fps),int(6*args.fps),int(11.8*args.fps),int((T_RANK+3)*args.fps),int((T_SELECT+.8)*args.fps),total-1):
            p=keys/f"stage3_{len(keyframes)+1}.png"; cv2.imwrite(str(p),scene); keyframes.append(str(p))
    video=out/"regiomanip_stage3_contact_frame_3d.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-framerate",str(args.fps),"-i",str(frames/"%06d.png"),*ffmpeg_quality_args(),str(video)],check=True)
    manifest={"video":str(video),"fps":args.fps,"duration_seconds":T_END,"resolution":[W,H],"score":score,"stages":["backproject_3d","fit_primitives","rank_se3","contact_frame"],"keyframes":keyframes,"display_convention":"Contact-frame signs are canonicalized for visualization so +z points outward from the target surface; stored SE(3) values are unchanged."}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
