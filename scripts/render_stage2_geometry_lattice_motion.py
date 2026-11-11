#!/usr/bin/env python3
"""Compact motion-first Stage 2 visualization for RegioManip."""

from __future__ import annotations

import argparse, json, subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from render_contribution_video_step1 import BLUE, INK, MUTED, _font, _rounded, _smoothstep

W,H=1920,1080
WHITE=(255,255,255); PANEL=(248,250,252); BORDER=(190,204,216)
GREEN=(20,145,75); PALE_GREEN=(235,248,241); RED=(210,52,55); PALE_RED=(255,241,241)
AMBER=(215,125,0); PALE_AMBER=(255,247,232); PALE_BLUE=(238,246,253)


def ease(x): return _smoothstep(float(np.clip(x,0,1)))


def mix(a,b,p): return tuple(int(round((1-p)*x+p*y)) for x,y in zip(a,b))


def lerp(a,b,p): return int(round(a+(b-a)*ease(p)))


def base(active,subtitle):
    frame=np.full((H,W,3),255,np.uint8)
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); d=ImageDraw.Draw(pil)
    d.text((65,38),"RegioManip",font=_font(29,True),fill=BLUE)
    d.text((65,80),"Stage 2. Geometry checking and lattice pruning",font=_font(41,True),fill=INK)
    d.text((67,135),subtitle,font=_font(20),fill=MUTED)
    steps=("Predict","Agree","Prune","Rank")
    for i,s in enumerate(steps):
        x=70+i*235; col=GREEN if i<active else BLUE if i==active else (190,202,212)
        d.ellipse((x,177,x+22,199),fill=col)
        d.text((x+33,174),s,font=_font(16,True),fill=INK if i<=active else MUTED)
        if i<3: d.line((x+130,188,x+208,188),fill=(209,219,227),width=3)
    return pil,d


def pill(d,xy,text,color,fill=WHITE,font=17,pad=18):
    x,y=xy; f=_font(font,True); tw=int(d.textlength(text,font=f)); w=tw+2*pad; h=46
    _rounded(d,(x,y,x+w,y+h),fill=fill,outline=color,radius=12,width=2)
    d.text((x+pad,y+10),text,font=f,fill=color)
    return (x,y,x+w,y+h)


def arrow(d,a,b,color,width=5,progress=1.0):
    p=ease(progress); x=int(a[0]+(b[0]-a[0])*p); y=int(a[1]+(b[1]-a[1])*p)
    d.line((a[0],a[1],x,y),fill=color,width=width)
    if p>.96:
        ang=np.arctan2(b[1]-a[1],b[0]-a[0]); q1=(int(x-16*np.cos(ang)+9*np.sin(ang)),int(y-16*np.sin(ang)-9*np.cos(ang))); q2=(int(x-16*np.cos(ang)-9*np.sin(ang)),int(y-16*np.sin(ang)+9*np.cos(ang)))
        d.polygon([(x,y),q1,q2],fill=color)


def predict_scene(t):
    pil,d=base(0,"Geometry-only attributes emerge from independent mask and language evidence.")
    _rounded(d,(65,235,1855,930),fill=PANEL,outline=BORDER,radius=22,width=2)
    d.text((95,260),"Geometry-only prediction",font=_font(25,True),fill=INK)
    d.text((95,298),"independent evidence",font=_font(15),fill=MUTED)
    d.text((790,298),"predictors",font=_font(15),fill=MUTED)
    d.text((1445,298),"geometry tuple",font=_font(15),fill=MUTED)
    rows=[("handle elongation + area","grasp","power",BLUE),
          ("work-end aspect ratio","contact","line",GREEN),
          ("instruction: “scrape”","force","pull",AMBER)]
    for i,(evidence,kind,result,color) in enumerate(rows):
        y=365+i*155; delay=i*.18
        p=ease((t-.15-delay)/.65)
        _rounded(d,(105,y,555,y+78),fill=WHITE,outline=(195,208,219),radius=14,width=2)
        d.text((130,y+17),evidence,font=_font(18,True),fill=INK)
        arrow(d,(575,y+39),(755,y+39),color,5,p)
        if p>.28:
            alpha=ease((p-.28)/.72); fc=mix(WHITE,PALE_GREEN,alpha)
            _rounded(d,(780,y,1120,y+78),fill=fc,outline=color,radius=14,width=2)
            d.text((805,y+12),kind+" predictor",font=_font(14),fill=MUTED)
            d.text((805,y+40),result,font=_font(22,True),fill=color)
        q=ease((t-.78-delay)/.6)
        if q>0:
            sx=1120; ex=1430; cx=lerp(sx,ex,q)
            d.line((sx,y+39,cx,y+39),fill=color,width=5)
            if q>.95: pill(d,(1450,y+16),result,color,PALE_GREEN,18,20)
    gate=ease((t-1.05)/.45)
    if gate>0:
        col=mix((210,220,228),GREEN,gate)
        _rounded(d,(435,835,1485,890),fill=mix(WHITE,PALE_GREEN,gate),outline=col,radius=13,width=2)
        d.ellipse((470,851,493,874),fill=col); d.text((477,851),"✓",font=_font(14,True),fill=WHITE)
        d.text((510,850),"depth-consistency gate",font=_font(17,True),fill=INK)
        d.text((760,850),"passed",font=_font(17,True),fill=GREEN)
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def agreement_scene(t,psi):
    pil,d=base(1,"Matching attributes converge into a bounded geometry–VLM agreement score.")
    # Source cards.
    _rounded(d,(80,265,620,880),fill=PANEL,outline=BLUE,radius=20,width=2)
    _rounded(d,(1300,265,1840,880),fill=PANEL,outline=GREEN,radius=20,width=2)
    d.text((110,292),"VLM tuple",font=_font(25,True),fill=BLUE)
    d.text((1330,292),"Geometry tuple",font=_font(25,True),fill=GREEN)
    attrs=("power","line","pull")
    ys=(405,555,705)
    meet=ease((t-.25)/1.0)
    for i,(a,y) in enumerate(zip(attrs,ys)):
        pill(d,(145,y-23),a,BLUE,WHITE,18,28); pill(d,(1485,y-23),a,GREEN,WHITE,18,28)
        lx=lerp(600,885,meet); rx=lerp(1320,1035,meet)
        d.line((600,y,lx,y),fill=BLUE,width=5); d.line((1320,y,rx,y),fill=GREEN,width=5)
        if meet>.72:
            p=ease((meet-.72)/.28); r=int(13+8*p)
            d.ellipse((960-r,y-r,960+r,y+r),fill=GREEN)
            d.text((953,y-11),"✓",font=_font(15,True),fill=WHITE)
    # Agreement gauge.
    gp=ease((t-1.05)/.65)
    d.ellipse((825,330,1095,600),fill=PALE_BLUE,outline=BLUE,width=4)
    d.text((914,365),"Ψ",font=_font(48,True),fill=INK)
    d.text((880,438),f"{psi*gp:.3f}",font=_font(30,True),fill=GREEN)
    d.arc((855,510,1065,720),180,180+int(180*psi*gp),fill=GREEN,width=14)
    d.text((867,620),"JSD agreement",font=_font(15),fill=MUTED)
    if gp>.95:
        _rounded(d,(560,835,1360,892),fill=PALE_GREEN,outline=GREEN,radius=14,width=2)
        d.text((690,851),"power · line · pull agree",font=_font(18,True),fill=GREEN)
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def lattice_columns(d):
    cols=[(130,"GRASP",("power","precision","pinch","hook"),BLUE),
          (730,"CONTACT",("point","line","surface"),GREEN),
          (1330,"FORCE",("push","pull","torque","normal"),AMBER)]
    centers={}
    for x,title,items,color in cols:
        d.text((x,285),title,font=_font(19,True),fill=color)
        for i,item in enumerate(items):
            y=340+i*105; sel=item in ("power","line","pull")
            _rounded(d,(x,y,x+360,y+64),fill=PALE_GREEN if sel else WHITE,outline=GREEN if sel else BORDER,radius=12,width=2 if sel else 1)
            d.text((x+25,y+18),item,font=_font(18,True),fill=GREEN if sel else INK)
            centers[item]=(x+180,y+32)
    return centers


def prune_scene(t):
    pil,d=base(2,"Only representative paths are animated; the full lattice contains 48 triples.")
    _rounded(d,(65,235,1855,930),fill=PANEL,outline=BORDER,radius=22,width=2)
    centers=lattice_columns(d)
    # Progressive paths: valid, improbable, forbidden.
    p1=ease(t/.65); p2=ease((t-.35)/.65); p3=ease((t-.7)/.65)
    arrow(d,centers["power"],centers["line"],GREEN,7,p1); arrow(d,centers["line"],centers["pull"],GREEN,7,p1)
    arrow(d,centers["power"],centers["surface"],AMBER,4,p2); arrow(d,centers["surface"],centers["normal"],AMBER,4,p2)
    arrow(d,centers["hook"],centers["surface"],RED,4,p3); arrow(d,centers["surface"],centers["torque"],RED,4,p3)
    gate=ease((t-1.15)/.5)
    if gate>0:
        # Compact physics gate at bottom; status chips animate in.
        _rounded(d,(230,815,1690,890),fill=WHITE,outline=(183,199,212),radius=15,width=2)
        d.text((270,837),"Λ  PHYSICS GATE",font=_font(18,True),fill=INK)
        pill(d,(600,829),"valid  ×1",GREEN,PALE_GREEN,15,18)
        pill(d,(910,829),"improbable  ×0.45",AMBER,PALE_AMBER,15,18)
        pill(d,(1300,829),"forbidden  ×0",RED,PALE_RED,15,18)
        # Break marks make pruning immediately visible.
        if gate>.45:
            for item in ("normal","torque"):
                x,y=centers[item]; d.line((x-16,y-16,x+16,y+16),fill=RED,width=5); d.line((x+16,y-16,x-16,y+16),fill=RED,width=5)
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def rank_scene(t,psi):
    pil,d=base(3,"Surviving evidence collapses into the ranked affordance simplex Δ⁺.")
    p=ease(t/.65)
    # Incoming colored tokens converge.
    for sx,label,color,dy in ((220,"power",BLUE,-90),(220,"line",GREEN,0),(220,"pull",AMBER,90)):
        x=lerp(sx,620,p); y=lerp(540+dy,540,p); pill(d,(x,y-23),label,color,WHITE,16,18)
    # Ranked card grows once tokens meet.
    q=ease((t-.45)/.55)
    if q>0:
        x1=lerp(850,300,q); x2=lerp(1070,1620,q); y1=lerp(490,355,q); y2=lerp(610,565,q)
        _rounded(d,(x1,y1,x2,y2),fill=PALE_GREEN,outline=GREEN,radius=20,width=3)
        if q>.75:
            d.text((350,390),"1",font=_font(35,True),fill=GREEN)
            d.text((435,386),"power  ·  line  ·  pull",font=_font(28,True),fill=INK)
            d.text((435,438),f"Ψ = {psi:.3f}    Λ = 1",font=_font(17),fill=MUTED)
            d.rounded_rectangle((1120,418,1515,443),radius=12,fill=(215,228,220))
            d.rounded_rectangle((1120,418,1120+int(395*psi),443),radius=12,fill=GREEN)
            d.text((1535,412),f"{psi:.3f}",font=_font(18,True),fill=GREEN)
    end=ease((t-1.05)/.5)
    if end>0:
        _rounded(d,(485,680,1435,770),fill=PALE_BLUE,outline=BLUE,radius=17,width=2)
        d.text((650,706),"Δ⁺  →  forward valid hypotheses to Stage 3",font=_font(21,True),fill=BLUE)
        d.text((686,804),"No contact frame is constructed in Stage 2.",font=_font(16),fill=MUTED)
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def transition(a,b,p):
    """Fade through white so two complete layouts are never superimposed."""
    p=float(np.clip(p,0,1)); white=np.full_like(a,255)
    if p<.5: return cv2.addWeighted(white,ease(p/.5),a,1-ease(p/.5),0)
    return cv2.addWeighted(b,ease((p-.5)/.5),white,1-ease((p-.5)/.5),0)


def load_psi(path):
    d=json.loads(Path(path).read_text()); return float(d["tool_primitives"][0]["geometry_vlm_agreement"])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results-json",default="results/regiomanip2/color_000012_run3/interaction_lift/interaction_lift_results.json")
    ap.add_argument("--stage1-video",default="results/video/regiomanip_stage1_final/regiomanip_stage1_FINAL.mp4")
    ap.add_argument("--output-dir",default="results/video/regiomanip_stage2_motion")
    ap.add_argument("--fps",type=int,default=15); ap.add_argument("--duration",type=float,default=8.0)
    args=ap.parse_args(); psi=load_psi(args.results_json)
    out=Path(args.output_dir); frames=out/"frames"; frames.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(args.stage1_video); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,n-1)); ok,last=cap.read(); cap.release()
    if not ok: last=np.full((H,W,3),255,np.uint8)
    last=cv2.resize(last,(W,H))
    total=int(args.duration*args.fps)
    for i in range(total):
        t=i/args.fps
        if t<2.0: scene=predict_scene(t)
        elif t<3.8: scene=agreement_scene(t-2.0,psi)
        elif t<6.1: scene=prune_scene(t-3.8)
        else: scene=rank_scene(t-6.1,psi)
        if t<.45: scene=transition(last,scene,t/.45)
        elif 2.0<=t<2.25: scene=transition(predict_scene(2.0),scene,(t-2.0)/.25)
        elif 3.8<=t<4.05: scene=transition(agreement_scene(1.8,psi),scene,(t-3.8)/.25)
        elif 6.1<=t<6.35: scene=transition(prune_scene(2.3),scene,(t-6.1)/.25)
        cv2.imwrite(str(frames/f"{i:06d}.png"),scene)
    video=out/"regiomanip_stage2_motion.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-framerate",str(args.fps),"-i",str(frames/"%06d.png"),"-c:v","libx264","-preset","slow","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(video)],check=True)
    manifest={"video":str(video),"fps":args.fps,"duration_seconds":args.duration,"resolution":[W,H],"geometry_vlm_agreement":psi,"selected_triple":["power","line","pull"],"style":"continuous_motion"}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
