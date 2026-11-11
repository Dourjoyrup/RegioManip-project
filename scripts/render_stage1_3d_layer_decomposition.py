#!/usr/bin/env python3
"""Decompose one fixed RGB-D scene into five isolated VLM evidence layers."""

from __future__ import annotations

import argparse, json, subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from render_contribution_video_step1 import BLUE, INK, MUTED, _font, _rounded, _smoothstep
from render_stage1_open3d_grounding import GroundingRenderer, _cloud_with_pixels, _mask


def ease(x): return _smoothstep(float(np.clip(x,0,1)))


def fit(image,w,h,bg=255):
    out=np.full((h,w,3),bg,np.uint8); ih,iw=image.shape[:2]; s=min(w/iw,h/ih)
    nw,nh=max(1,int(iw*s)),max(1,int(ih*s)); r=cv2.resize(image,(nw,nh),interpolation=cv2.INTER_AREA if s<1 else cv2.INTER_CUBIC)
    x,y=(w-nw)//2,(h-nh)//2; out[y:y+nh,x:x+nw]=r; return out


def crop_union(images, masks, pad=14):
    union=np.logical_or.reduce([m>0 for m in masks]); ys,xs=np.where(union)
    x1,x2=max(0,xs.min()-pad),min(union.shape[1],xs.max()+pad+1); y1,y2=max(0,ys.min()-pad),min(union.shape[0],ys.max()+pad+1)
    return [im[y1:y2,x1:x2] for im in images]


def isolated_mask_and_skeleton(handle,work):
    h,w=handle.shape; mask_vis=np.full((h,w,3),255,np.uint8); skel_vis=np.full((h,w,3),255,np.uint8)
    work_fill=(25,70,255); green_fill=(45,210,35)
    work_line=(15,40,205); green_line=(15,135,20)
    mask_vis[work>0]=work_fill; mask_vis[handle>0]=green_fill
    for mask,fill_color,line_color,kind in ((work,work_fill,work_line,"blade"),(handle,green_fill,green_line,"handle")):
        contours,_=cv2.findContours((mask>0).astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(mask_vis,contours,-1,fill_color,3,cv2.LINE_AA)
        cv2.drawContours(skel_vis,contours,-1,line_color,4,cv2.LINE_AA)
        ys,xs=np.where(mask>0); uv=np.column_stack((xs,ys)).astype(np.float64)
        if kind=="blade":
            dist=cv2.distanceTransform((mask>0).astype(np.uint8),cv2.DIST_L2,5); cy,cx=np.unravel_index(int(np.argmax(dist)),dist.shape); center=np.array([cx,cy],float)
            polygon=cv2.approxPolyDP(max(contours,key=cv2.contourArea),.025*cv2.arcLength(max(contours,key=cv2.contourArea),True),True)[:,0,:].astype(float)
            angles=np.arctan2(polygon[:,1]-center[1],polygon[:,0]-center[0]); anchors=[]
            for a in (-3*np.pi/4,-np.pi/4,np.pi/4,3*np.pi/4): anchors.append(polygon[np.argmin(np.abs(np.angle(np.exp(1j*(angles-a)))) )])
            for anchor in anchors: cv2.line(skel_vis,tuple(np.int32(center)),tuple(np.int32(anchor)),line_color,2,cv2.LINE_AA)
        else:
            mean=uv.mean(0); _,_,vh=np.linalg.svd(uv-mean,full_matrices=False); axis=vh[0]; normal=np.array([-axis[1],axis[0]]); p=(uv-mean)@axis
            ends=[mean+np.percentile(p,7)*axis,mean+np.percentile(p,93)*axis]
            cv2.line(skel_vis,tuple(np.int32(ends[0])),tuple(np.int32(ends[1])),line_color,2,cv2.LINE_AA)
            for end,band in ((ends[0],p<np.percentile(p,14)),(ends[1],p>np.percentile(p,86))):
                q=(uv[band]-end)@normal
                for value in (np.percentile(q,12),np.percentile(q,88)): cv2.line(skel_vis,tuple(np.int32(end)),tuple(np.int32(end+value*normal)),line_color,2,cv2.LINE_AA)
    return crop_union([mask_vis,skel_vis],[handle,work])


def compose(t,duration,views,orbit_views,extracts,fps):
    names=["RGB appearance","Metric depth","Candidate masks","Boundary + skeleton","Target context"]
    subtitles=["Natural color carried by the reconstructed 3D points.","Range becomes a metric color field at the identical camera pose.",
               "Anonymous candidate regions are isolated without semantic roles.","Aligned contours and topology expose candidate geometry.",
               "The task target is isolated as the fifth contextual view."]
    colors=[(34,122,183),(232,111,32),(124,88,198),(29,158,96),(235,139,35)]
    intro=2.0; layer_duration=max(5.0,duration-intro); stage_len=layer_duration/5
    in_intro=t<intro
    if in_intro:
        orbit_index=min(len(orbit_views)-1,int((t/intro)*len(orbit_views)))
        view=orbit_views[orbit_index]; stage=-1; local=t
    else:
        layer_t=t-intro; stage=min(4,int(layer_t/stage_len)); local=layer_t-stage*stage_len
        blend=ease(local/.38) if stage>0 else ease(local/.45)
        previous=orbit_views[-1] if stage==0 else views[stage-1]
        view=cv2.addWeighted(views[stage],blend,previous,1-blend,0)
    frame=np.full((1080,1920,3),255,np.uint8)
    has_extract=(not in_intro and stage>=1)
    view_x=150 if has_extract else 450
    frame[215:875,view_x:view_x+1180]=view
    if has_extract:
        alpha=ease((local-.28)/.52); artifact=fit(extracts[stage],360,430)
        layer=frame.copy(); layer[350:780,1460:1820]=artifact
        cv2.addWeighted(layer,alpha,frame,1-alpha,0,frame)
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
    draw.text((65,46),"RegioManip",font=_font(30,True),fill=BLUE)
    draw.text((65,91),"3D evidence decomposition",font=_font(43,True),fill=INK)
    draw.text((67,149),"The fixed reconstructed observation is separated into the five synchronized VLM inputs.",font=_font(22),fill=MUTED)
    _rounded(draw,(485,235,1040,304),fill=(255,255,255),outline=(190,205,218),radius=15,width=2)
    if in_intro:
        draw.ellipse((510,252,544,286),fill=(30,143,86)); draw.text((519,257),"•",font=_font(18,True),fill=(255,255,255))
        draw.text((560,250),"Combined 3D observation",font=_font(24,True),fill=INK)
    else:
        draw.ellipse((510,252,544,286),fill=colors[stage]); draw.text((521,257),str(stage+1),font=_font(16,True),fill=(255,255,255))
        draw.text((560,250),names[stage],font=_font(24,True),fill=INK)
    _rounded(draw,(485,792,1595,852),fill=(250,252,254),outline=(194,208,220),radius=14,width=2)
    draw.text((515,808),"Camera settles before the evidence layers separate." if in_intro else subtitles[stage],font=_font(19),fill=INK)
    # Compact five-step state indicator.
    for i,name in enumerate(("RGB","Depth","Masks","Skeleton","Target")):
        x=410+i*225; active=(not in_intro and i==stage); done=(not in_intro and i<stage)
        draw.ellipse((x,925,x+24,949),fill=colors[i] if active or done else (190,200,210))
        draw.text((x+34,922),name,font=_font(18,True),fill=INK if active else MUTED)
    progress=int(1760*np.clip(t/duration,0,1));
    frame=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
    cv2.line(frame,(80,1048),(1840,1048),(231,236,241),10,cv2.LINE_AA); cv2.line(frame,(80,1048),(80+progress,1048),(181,108,19),10,cv2.LINE_AA)
    return frame


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",default="results/regiomanip2/color_000012_run3"); ap.add_argument("--output-dir",default="results/video/regiomanip_stage1_3d_layers_v2"); ap.add_argument("--fps",type=int,default=15); ap.add_argument("--duration",type=float,default=12.0); args=ap.parse_args()
    run=Path(args.run_dir); rgb=cv2.imread("data/000/color_000012.png"); raw=np.load("data/000/depth_000012.npy").astype(np.float32); depth=raw*(.001 if np.nanmedian(raw)>10 else 1)
    tool=_mask(run/"dino_sam/sam/masks/scraper_whole.png"); target=_mask(run/"dino_sam/sam/masks/yellow_putty_whole.png")
    handle=_mask(run/"region_affordance/_masks/scraper__handle_fullmask.png"); work=_mask(run/"region_affordance/_masks/scraper__workend_fullmask.png")
    points,colors,pixels=_cloud_with_pixels(rgb,depth); masks={"tool":tool,"target":target,"handle":handle,"workend":work}
    renderer=GroundingRenderer(1180,660,points,colors,pixels,depth,masks,{"tool":[0,0,1,1],"target":[0,0,1,1]})
    views=[renderer.render_decomposition(mode) for mode in ("rgb","depth","masks","skeleton","target")]
    orbit_views=[renderer.render(ease(i/14),1.0,1.0,1.0) for i in range(15)]
    target_crop=cv2.imread(str(run/"dino_sam/dino/crops/01_yellow_putty.png"))
    mask_only,skeleton_only=isolated_mask_and_skeleton(handle,work)
    depth_only=cv2.imread(str(run/"region_affordance/depth_colormap.png"))
    # stage indices are RGB=0, depth=1, masks=2, skeleton=3, target=4.
    extracts=[None,depth_only,mask_only,skeleton_only,target_crop]
    out=Path(args.output_dir); frames=out/"frames"; keys=out/"keyframes"; frames.mkdir(parents=True,exist_ok=True); keys.mkdir(parents=True,exist_ok=True)
    total=int(args.fps*args.duration); keyframes=[]; key_at=[int(.9*args.fps)]+[int((2+(i+.72)*(args.duration-2)/5)*args.fps) for i in range(5)]
    for i in range(total):
        frame=compose(i/args.fps,args.duration,views,orbit_views,extracts,args.fps); cv2.imwrite(str(frames/f"{i:06d}.png"),frame)
        if i in key_at:
            p=keys/f"stage_{len(keyframes)}.png"; cv2.imwrite(str(p),frame); keyframes.append(str(p))
    video=out/"regiomanip_3d_evidence_decomposition.mp4"; subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-framerate",str(args.fps),"-i",str(frames/"%06d.png"),"-c:v","libx264","-preset","slow","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(video)],check=True)
    manifest={"video":str(video),"fps":args.fps,"duration_seconds":args.duration,"resolution":[1920,1080],"stages":["moving_combined_3d","rgb_3d","depth_3d","masks_3d","skeleton_3d","target_context"],"keyframes":keyframes}; (out/"manifest.json").write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
