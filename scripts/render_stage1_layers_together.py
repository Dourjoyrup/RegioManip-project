#!/usr/bin/env python3
"""Hold the final combined 3D frame, then reveal all five outputs together."""

from __future__ import annotations

import argparse, json, subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from render_contribution_video_step1 import BLUE, INK, MUTED, _font, _rounded, _smoothstep, draw_header, ffmpeg_quality_args
from render_stage1_3d_layer_decomposition import fit, isolated_mask_and_skeleton
from render_stage1_open3d_grounding import _mask
from video_style import TimeMap, Narrator, apply_chrome, fade, STAGE1_STEPS

HB=165  # header band height (brand, title, subtitle)
STAGE1_TITLE="Stage 1. Region-level affordance grounding"


# Crop of the Stage 1 keyframe containing only the 3D candidate geometry
# (no title capsule, legend pills, or caption).  Embedding the whole UI made a
# nested, illegible screenshot inside the "3D SOURCE" card.
SRC_X1,SRC_Y1,SRC_X2,SRC_Y2=913,325,1687,830
SLOT_Y,SLOT_W,SLOT_H=330,650,424
CANDIDATE_PILLS=(("candidate 1",(245,75,30)),("candidate 2",(20,190,55)),("target",(220,175,0)))


def masked_crop(image, mask, pad=12):
    ys,xs=np.where(mask>0); x1,x2=max(0,xs.min()-pad),min(mask.shape[1],xs.max()+pad+1); y1,y2=max(0,ys.min()-pad),min(mask.shape[0],ys.max()+pad+1)
    crop=image[y1:y2,x1:x2].copy(); m=mask[y1:y2,x1:x2]>0; crop[~m]=255
    return crop


def synchronized_scene_crop(image, masks, pad=12):
    """Cut out multiple regions together while preserving their image geometry."""
    union=np.logical_or.reduce([m>0 for m in masks]); ys,xs=np.where(union)
    x1,x2=max(0,xs.min()-pad),min(union.shape[1],xs.max()+pad+1); y1,y2=max(0,ys.min()-pad),min(union.shape[0],ys.max()+pad+1)
    crop=image[y1:y2,x1:x2].copy(); local=union[y1:y2,x1:x2]; crop[~local]=255
    return crop


def synchronized_mask_skeleton(handle, work, target):
    """Show tool candidates and target in one registered mask/topology view."""
    h,w=handle.shape; mask_vis=np.full((h,w,3),255,np.uint8); skel_vis=np.full((h,w,3),255,np.uint8)
    tool_mask,tool_skel=isolated_mask_and_skeleton(handle,work)
    union=(handle>0)|(work>0); ys,xs=np.where(union); pad=14
    x1,x2=max(0,xs.min()-pad),min(w,xs.max()+pad+1); y1,y2=max(0,ys.min()-pad),min(h,ys.max()+pad+1)
    mask_vis[y1:y2,x1:x2]=tool_mask; skel_vis[y1:y2,x1:x2]=tool_skel
    target_yellow=(0,220,255); target_line=(0,145,210); tm=(target>0).astype(np.uint8)
    mask_vis[tm>0]=target_yellow
    contours,_=cv2.findContours(tm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(mask_vis,contours,-1,target_yellow,3,cv2.LINE_AA); cv2.drawContours(skel_vis,contours,-1,target_line,4,cv2.LINE_AA)
    yy,xx=np.where(tm>0); uv=np.column_stack((xx,yy)).astype(float); mean=uv.mean(0); _,_,vh=np.linalg.svd(uv-mean,full_matrices=False); axis=vh[0]; p=(uv-mean)@axis
    a=mean+np.percentile(p,10)*axis; b=mean+np.percentile(p,90)*axis; cv2.line(skel_vis,tuple(np.int32(a)),tuple(np.int32(b)),target_line,2,cv2.LINE_AA)
    return synchronized_scene_crop(mask_vis,[handle,work,target]),synchronized_scene_crop(skel_vis,[handle,work,target])


def load_transparent_on_white(path):
    image=cv2.imread(str(path),cv2.IMREAD_UNCHANGED)
    if image is None: raise FileNotFoundError(path)
    if image.shape[2]==4:
        alpha=image[:,:,3:4].astype(np.float32)/255.0
        return np.uint8(image[:,:,:3].astype(np.float32)*alpha+255.0*(1.0-alpha))
    return image[:,:,:3]


def target_bundle(rgb, depth_vis, target):
    rgb_cut=masked_crop(rgb,target,8); depth_cut=masked_crop(depth_vis,target,8)
    ys,xs=np.where(target>0); x1,x2=max(0,xs.min()-8),min(target.shape[1],xs.max()+9); y1,y2=max(0,ys.min()-8),min(target.shape[0],ys.max()+9)
    m=target[y1:y2,x1:x2]>0
    # Fuse RGB appearance, depth color, mask tint, border, and skeleton in one image.
    combined=np.full_like(rgb_cut,255); yellow=np.zeros_like(rgb_cut); yellow[:]=(0,220,255)
    fused=cv2.addWeighted(rgb_cut,.52,depth_cut,.33,0); fused=cv2.addWeighted(fused,.85,yellow,.15,0); combined[m]=fused[m]
    contours,_=cv2.findContours(m.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE); cv2.drawContours(combined,contours,-1,(0,220,255),3,cv2.LINE_AA)
    yy,xx=np.where(m); uv=np.column_stack((xx,yy)).astype(float); mean=uv.mean(0); _,_,vh=np.linalg.svd(uv-mean,full_matrices=False); axis=vh[0]; p=(uv-mean)@axis
    a=mean+np.percentile(p,10)*axis; b=mean+np.percentile(p,90)*axis; cv2.line(combined,tuple(np.int32(a)),tuple(np.int32(b)),(25,105,255),2,cv2.LINE_AA)
    return combined


def evidence_layout(assets, prompt_x, heading, show_tuple=False):
    frame=np.full((1080,1920,3),255,np.uint8)
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
    subtitle=("The VLM returns a RegionAffordanceTuple: role, grasp, contact, and force for each region."
              if show_tuple else "Five synchronized views are passed together to the VLM for region-level reasoning.")
    draw_header(draw,STAGE1_TITLE,subtitle)
    frame=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
    labels=["RGB","Depth","Mask grid","Skeleton","Target crop"]
    accents=[(184,119,33),(32,108,232),(198,84,124),(96,157,29),(34,139,235)]
    # The complete prompt first appears centered, then moves left as one unit.
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
    _rounded(draw,(prompt_x,220,prompt_x+700,920),fill=(248,250,252),outline=(176,197,214),radius=22,width=2)
    draw.text((prompt_x+27,242),"Structured 5-image prompt",font=_font(25,True),fill=INK)
    _rounded(draw,(prompt_x+550,240,prompt_x+670,276),fill=(232,243,253),outline=BLUE,radius=9,width=1)
    draw.text((prompt_x+610-draw.textlength("VLM INPUT",font=_font(14,True))/2,249),"VLM INPUT",font=_font(14,True),fill=BLUE)
    draw.text((prompt_x+27,278),"appearance · range · regions · topology · target",font=_font(16),fill=MUTED)
    frame=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
    boxes=[(prompt_x+25,320,300,170),(prompt_x+350,320,300,170),(prompt_x+25,510,300,170),(prompt_x+350,510,300,170),(prompt_x+187,700,300,190)]
    for i,(asset,label,box,accent) in enumerate(zip(assets,labels,boxes,accents)):
        x,y,w,h=box
        pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
        _rounded(draw,(x,y,x+w,y+h),fill=(255,255,255),outline=(195,208,219),radius=13,width=1)
        draw.rounded_rectangle((x,y,x+w,y+6),radius=3,fill=accent)
        draw.ellipse((x+10,y+13,x+36,y+39),fill=accent); draw.text((x+19,y+16),str(i+1),font=_font(13,True),fill=(255,255,255))
        draw.text((x+46,y+14),label,font=_font(16,True),fill=INK)
        frame=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
        frame[y+45:y+h-8,x+8:x+w-8]=fit(asset,w-16,h-53)
    if show_tuple:
        output_overlay=assets[5]
        pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
        _rounded(draw,(820,220,1870,920),fill=(248,250,252),outline=(176,197,214),radius=22,width=2)
        draw.text((850,242),"RegionAffordanceTuple",font=_font(28,True),fill=INK)
        draw.text((850,282),"Role / grasp / contact / force",font=_font(18),fill=MUTED)
        _rounded(draw,(1695,242,1835,280),fill=(232,243,253),outline=BLUE,radius=9,width=1)
        draw.text((1765-draw.textlength("VLM OUTPUT",font=_font(14,True))/2,251),"VLM OUTPUT",font=_font(14,True),fill=BLUE)
        frame=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
        frame[350:688,845:1295]=fit(output_overlay,450,338)
        pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
        draw.text((850,706),"Decoded region overlay: handle / work-end / target · G / C / T",font=_font(15),fill=MUTED)
        # Compact textual tuple beside the decoded visual output.
        draw.text((1330,342),"Resolved assignments",font=_font(18,True),fill=BLUE)
        rows=[
            ("ID 1 · work-end","contact: line  ·  force: pull",(245,75,30)),
            ("ID 2 · handle","grasp: power",(20,190,55)),
            ("Target","target region: border",(220,175,0)),
        ]
        for r,row in enumerate(rows):
            y=382+r*105; _rounded(draw,(1325,y,1840,y+78),fill=(255,255,255),outline=(202,214,224),radius=12,width=1)
            draw.text((1350,y+15),row[0],font=_font(17,True),fill=row[2]); draw.text((1350,y+44),row[1],font=_font(15),fill=INK)
        _rounded(draw,(1325,710,1840,865),fill=(238,246,253),outline=BLUE,radius=14,width=2)
        draw.text((1350,730),"RegionAffordanceTuple =",font=_font(17,True),fill=BLUE)
        draw.text((1350,766),"{ region, role, grasp,",font=_font(18,True),fill=INK)
        draw.text((1350,795),"  contact, force }",font=_font(18,True),fill=INK)
        draw.text((1350,830),"face: primary · semantic: scrape_blade",font=_font(14),fill=MUTED)
        frame=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
    return frame


def candidate_prompt_layout(assets, source, candidate_x, prompt_x=None, vlm_heading=False):
    """Bridge view: grounded 3D source card -> structured five-image card."""
    if prompt_x is None:
        frame=np.full((1080,1920,3),255,np.uint8)
        pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
        subtitle="Five synchronized views are passed together to the VLM for region-level reasoning." if vlm_heading else "The grounded 3D candidates become the source of a structured VLM prompt."
        draw_header(draw,STAGE1_TITLE,subtitle)
        frame=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
    else:
        frame=evidence_layout(assets,prompt_x,"Structured 5-image VLM prompt",False)
    # Draw the exact final Candidate Geometry composition as a card. Keeping
    # this source crop intact allows a continuous scale/translate morph.
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
    _rounded(draw,(candidate_x,220,candidate_x+700,920),fill=(248,250,252),outline=(176,197,214),radius=22,width=2)
    _rounded(draw,(candidate_x+550,240,candidate_x+670,276),fill=(255,245,232),outline=(245,75,30),radius=9,width=1)
    draw.text((candidate_x+610-draw.textlength("3D SOURCE",font=_font(14,True))/2,249),"3D SOURCE",font=_font(14,True),fill=(220,70,25))
    draw.text((candidate_x+27,242),"Candidate geometry",font=_font(25,True),fill=INK)
    draw.text((candidate_x+27,278),"grounded regions · borders · medial skeletons",font=_font(16),fill=MUTED)
    # Crisp, card-scale colour key replaces the shrunken copy of the Stage 1 pills.
    px=candidate_x+40
    for label,color in CANDIDATE_PILLS:
        f=_font(16,True); w=int(draw.textlength(label,font=f))+52
        _rounded(draw,(px,800,px+w,840),fill=(255,255,255),outline=color,radius=11,width=2)
        draw.ellipse((px+14,814,px+26,826),fill=color); draw.text((px+34,810),label,font=f,fill=color)
        px+=w+16
    draw.text((candidate_x+40,862),"numbered candidates · roles not yet assigned",font=_font(16),fill=MUTED)
    frame=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
    group=source[SRC_Y1:SRC_Y2,SRC_X1:SRC_X2]
    frame[SLOT_Y:SLOT_Y+SLOT_H,candidate_x+25:candidate_x+25+SLOT_W]=cv2.resize(group,(SLOT_W,SLOT_H),interpolation=cv2.INTER_AREA)
    if prompt_x is not None:
        pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); draw=ImageDraw.Draw(pil)
        start=candidate_x+720; end=prompt_x-25; mid=(start+end)//2
        # A processing node communicates transformation more clearly than a
        # second arrow embedded inside a long text label.
        node_w=94; node_l=mid-node_w//2; node_r=mid+node_w//2
        draw.line((start,560,node_l-10,560),fill=BLUE,width=4)
        draw.polygon([(node_l-10,560),(node_l-24,551),(node_l-24,569)],fill=BLUE)
        draw.line((node_r+10,560,end,560),fill=BLUE,width=4)
        draw.polygon([(end,560),(end-16,550),(end-16,570)],fill=BLUE)
        _rounded(draw,(node_l,516,node_r,604),fill=(238,246,253),outline=BLUE,radius=16,width=2)
        # Five evidence tiles entering a compact VLM processor.
        for row,col in ((0,0),(0,1),(1,0),(1,1)):
            x=node_l+18+col*18; y=532+row*18
            draw.rounded_rectangle((x,y,x+12,y+12),radius=3,fill=(255,255,255),outline=BLUE,width=1)
        draw.ellipse((node_l+58,539,node_l+76,557),fill=BLUE)
        draw.text((mid-draw.textlength("VLM",font=_font(14,True))/2,572),"VLM",font=_font(14,True),fill=BLUE)
        draw.text((start,526),"3D",font=_font(14,True),fill=(220,70,25))
        input_label="5 views"; input_w=draw.textlength(input_label,font=_font(14,True))
        draw.text((end-input_w,526),input_label,font=_font(14,True),fill=BLUE)
        frame=cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)
    return frame


def live_candidate_morph(source, target, progress):
    """Continuously box, shrink, and reposition the live Candidate Geometry."""
    p=_smoothstep(progress); white=np.full_like(source,255)
    # Old chrome is fully gone before the new card chrome appears, so the two
    # "Candidate geometry" labels are never superimposed.
    out_p=_smoothstep(min(1.0,progress*2.0)); in_p=_smoothstep(max(0.0,progress*2.0-1.0))
    frame=cv2.addWeighted(white,out_p,source,1-out_p,0)
    group=source[SRC_Y1:SRC_Y2,SRC_X1:SRC_X2]
    # Remove the old in-place geometry before drawing its transformed copy;
    # the surrounding Stage 1 chrome simply fades to white.
    frame[SRC_Y1:SRC_Y2,SRC_X1:SRC_X2]=255
    dst_x=610+25
    x=int(round(SRC_X1+(dst_x-SRC_X1)*p)); y=int(round(SRC_Y1+(SLOT_Y-SRC_Y1)*p))
    w=int(round((SRC_X2-SRC_X1)+(SLOT_W-(SRC_X2-SRC_X1))*p)); h=int(round((SRC_Y2-SRC_Y1)+(SLOT_H-(SRC_Y2-SRC_Y1))*p))
    frame[y:y+h,x:x+w]=cv2.resize(group,(w,h),interpolation=cv2.INTER_AREA)
    # The new page heading and card outline materialize progressively around
    # the very same pixels; there is no replacement frame or geometry jump.
    frame[:HB]=source[:HB]
    overlay=target.copy(); keep=np.any(overlay<248,axis=2)
    # Restrict this blend to card chrome outside the transformed content.
    keep[:HB]=False; keep[SLOT_Y:SLOT_Y+SLOT_H,635:635+SLOT_W]=False; alpha=(in_p*keep.astype(np.float32))[:,:,None]
    frame=np.uint8(overlay.astype(np.float32)*alpha+frame.astype(np.float32)*(1-alpha))
    return frame


def switch_candidate_heading(source, target, progress):
    """Change page heading through white after the live card is complete."""
    p=float(np.clip(progress,0,1)); frame=target.copy(); white=np.full_like(source[:HB],255)
    if np.array_equal(source[:HB],target[:HB]): return frame
    if p<.5: frame[:HB]=cv2.addWeighted(white,_smoothstep(p/.5),source[:HB],1-_smoothstep(p/.5),0)
    else: frame[:HB]=cv2.addWeighted(target[:HB],_smoothstep((p-.5)/.5),white,1-_smoothstep((p-.5)/.5),0)
    return frame


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--run-dir",default="results/regiomanip2/color_000012_run3"); ap.add_argument("--source-frame",default="results/video/regiomanip_stage1_grounding3d_v16/keyframes/grounding_4.png"); ap.add_argument("--output-dir",default="results/video/regiomanip_stage1_final"); ap.add_argument("--fps",type=int,default=15); ap.add_argument("--duration",type=float,default=9.2); ap.add_argument("--reveal-at",type=float,default=0.25); args=ap.parse_args()
    run=Path(args.run_dir); out=Path(args.output_dir); frames=out/"frames"; frames.mkdir(parents=True,exist_ok=True)
    handle=_mask(run/"region_affordance/_masks/scraper__handle_fullmask.png"); work=_mask(run/"region_affordance/_masks/scraper__workend_fullmask.png"); target=_mask(run/"dino_sam/sam/masks/yellow_putty_whole.png")
    rgb_full=cv2.imread("data/000/color_000012.png")
    # Gentle display-only exposure correction for the RGB evidence views.
    # Geometry, masks, and all numerical processing continue to use originals.
    rgb_display=cv2.convertScaleAbs(rgb_full,alpha=1.24,beta=22)
    raw=np.load("data/000/depth_000012.npy").astype(np.float32); valid=np.isfinite(raw)&(raw>0); lo,hi=np.percentile(raw[valid],(2,98)); u8=np.uint8(np.clip((raw-lo)/max(hi-lo,1e-6),0,1)*255); depth_full=cv2.applyColorMap(255-u8,cv2.COLORMAP_TURBO)
    tool=np.maximum(handle,work)
    rgb_cut=synchronized_scene_crop(rgb_display,[tool,target]); depth_cut=synchronized_scene_crop(depth_full,[tool,target])
    mask_only,skeleton_only=synchronized_mask_skeleton(handle,work,target)
    target_panel=load_transparent_on_white(
        "results/video/assets/target_crop_reconstructed.png"
    )
    output_overlay=load_transparent_on_white("results/video/assets/vlm_output_affordance_cutout.png")
    assets=[rgb_cut,depth_cut,mask_only,skeleton_only,target_panel,output_overlay]
    source=cv2.imread(args.source_frame)
    candidate_center=candidate_prompt_layout(assets,source,610,None)
    tmap=TimeMap(args.duration,slow=1.4,holds=((1.65,.6),(5.0,1.2),(9.2,2.2)))
    total=int(round(tmap.real_end*args.fps))
    narr=Narrator(((0,"The grounded 3D candidates become the source of the VLM prompt."),
                   (3.25,"Five synchronized views are rendered from the same candidates: RGB, depth, masks, skeletons, target."),
                   (6.9,"The VLM assigns roles: work-end (contact line, force pull), handle (power grasp), target border.")))
    for i in range(total):
        t=tmap(i/args.fps)
        if t<args.reveal_at:
            frame=source.copy()
        elif t<1.25:
            frame=live_candidate_morph(source,candidate_center,(t-args.reveal_at)/(1.25-args.reveal_at))
        elif t<1.65:
            frame=switch_candidate_heading(source,candidate_center,(t-1.25)/.4)
        elif t<2.85:
            move=_smoothstep((t-1.65)/1.2); cx=int(round(610+(110-610)*move))
            candidate_only=candidate_prompt_layout(assets,source,cx,None)
            frame=candidate_only
        elif t<3.25:
            old=candidate_prompt_layout(assets,source,110,None,False)
            new=candidate_prompt_layout(assets,source,110,None,True)
            frame=switch_candidate_heading(old,new,(t-2.85)/.4)
        elif t<4.25:
            source_only=candidate_prompt_layout(assets,source,110,None,True)
            both=candidate_prompt_layout(assets,source,110,1110,True)
            alpha=_smoothstep((t-3.25)/1.0); frame=cv2.addWeighted(both,alpha,source_only,1-alpha,0)
            frame[:HB]=switch_candidate_heading(source_only,both,(t-3.25)/1.0)[:HB]
        elif t<5.0:
            frame=candidate_prompt_layout(assets,source,110,1110,True)
        elif t<5.7:
            both=candidate_prompt_layout(assets,source,110,1110,True)
            prompt_only=evidence_layout(assets,1110,"Structured 5-image VLM prompt",False)
            close=_smoothstep((t-5.0)/.7); frame=cv2.addWeighted(prompt_only,close,both,1-close,0)
            # Different subtitles must not be superimposed: switch via white.
            frame[:HB]=switch_candidate_heading(both,prompt_only,(t-5.0)/.7)[:HB]
        elif t<6.9:
            move=_smoothstep((t-5.7)/1.2); px=int(round(1110+(45-1110)*move))
            frame=evidence_layout(assets,px,"Structured 5-image VLM prompt",False)
        elif t<8.1:
            base=evidence_layout(assets,45,"Structured 5-image VLM prompt",False)
            result=evidence_layout(assets,45,"Structured 5-image VLM prompt",True)
            alpha=_smoothstep((t-6.9)/1.2); frame=cv2.addWeighted(result,alpha,base,1-alpha,0)
            frame[:HB]=switch_candidate_heading(base,result,(t-6.9)/1.2)[:HB]
        else:
            frame=evidence_layout(assets,45,"Structured 5-image VLM prompt",True)
        text,na=narr(t)
        frame=apply_chrome(frame,1,STAGE1_STEPS,4 if t<.25 else 5 if t<6.9 else 6,text,na)
        end=(total-i)/args.fps
        if end<.6: frame=fade(frame,_smoothstep(end/.6))
        cv2.imwrite(str(frames/f"{i:06d}.png"),frame)
    video=out/"regiomanip_hold_then_all_five.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-framerate",str(args.fps),"-i",str(frames/"%06d.png"),*ffmpeg_quality_args(),str(video)],check=True)
    manifest={"video":str(video),"combined_3d_hold_seconds":args.reveal_at,"sequential_reveal":False,"simultaneous_outputs":["rgb","depth","candidate_masks","skeleton_only","target_crop"]}; (out/"manifest.json").write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
