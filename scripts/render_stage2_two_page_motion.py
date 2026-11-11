#!/usr/bin/env python3
"""Two-page, slower Stage 2 animation for RegioManip."""

from __future__ import annotations

import argparse, json, subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from render_contribution_video_step1 import BLUE, INK, MUTED, _font, _rounded, draw_header, ffmpeg_quality_args
from render_stage2_geometry_lattice_motion import (
    W,H,WHITE,PANEL,BORDER,GREEN,PALE_GREEN,RED,PALE_RED,AMBER,PALE_AMBER,PALE_BLUE,
    ease,lerp,pill,transition,
)
from video_style import TimeMap, Narrator, apply_chrome, fade, check_badge, cross_badge, STAGE2_STEPS
from video_style import arrow as _vs_arrow


def arrow(d,a,b,color,width=5,progress=1.0):
    """Shared arrow style (proportionate filled heads) used by every stage."""
    _vs_arrow(d,a,b,color,width,None,progress)


def page_base(page,subtitle):
    frame=np.full((H,W,3),255,np.uint8)
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)); d=ImageDraw.Draw(pil)
    draw_header(d,"Stage 2. Geometry checking and lattice pruning",subtitle)
    return pil,d


def page1(t,psi):
    pil,d=page_base(0,"Geometry evidence and VLM semantics are compared on one stable canvas.")
    _rounded(d,(65,240,1165,930),fill=PANEL,outline=BORDER,radius=22,width=2)
    _rounded(d,(1195,240,1855,930),fill=PANEL,outline=BORDER,radius=22,width=2)
    d.text((95,265),"Geometry-only predictors",font=_font(24,True),fill=INK)
    d.text((1225,265),"Geometry–VLM agreement",font=_font(24,True),fill=INK)
    rows=[("handle elongation + area","power",BLUE),
          ("work-end aspect ratio","line",GREEN),
          ("instruction: “scrape”","pull",AMBER)]
    for i,(evidence,result,color) in enumerate(rows):
        y=350+i*145; p=ease((t-.25-i*.28)/1.05)
        _rounded(d,(105,y,510,y+78),fill=WHITE,outline=BORDER,radius=13,width=2)
        d.text((130,y+22),evidence,font=_font(17,True),fill=INK)
        arrow(d,(535,y+39),(700,y+39),color,5,p)
        if p>.30:
            _rounded(d,(725,y,1090,y+78),fill=PALE_GREEN,outline=color,radius=13,width=2)
            d.text((750,y+10),"geometry predictor",font=_font(15),fill=MUTED)
            d.text((750,y+42),result,font=_font(21,True),fill=color)
    gate=ease((t-1.65)/.75)
    _rounded(d,(105,820,1090,878),fill=PALE_BLUE if gate<.8 else PALE_GREEN,outline=BLUE if gate<.8 else GREEN,radius=13,width=2)
    d.text((145,837),"depth-consistency gate",font=_font(16,True),fill=INK)
    d.text((405,837),"checking…" if gate<.8 else "passed ✓",font=_font(16,True),fill=MUTED if gate<.8 else GREEN)

    # Agreement area remains fixed while values arrive gradually.
    d.text((1240,325),"VLM",font=_font(18,True),fill=BLUE)
    d.text((1655,325),"GEOMETRY",font=_font(18,True),fill=GREEN)
    attrs=("power","line","pull")
    for i,(a,color) in enumerate(zip(attrs,(BLUE,GREEN,AMBER))):
        y=390+i*115; appear=ease((t-1.75-i*.22)/.7)
        if appear>0:
            pill(d,(1235,y),a,BLUE,WHITE,15,16); pill(d,(1650,y),a,GREEN,WHITE,15,16)
            arrow(d,(1355,y+23),(1450,y+23),BLUE,4,appear)
            arrow(d,(1640,y+23),(1545,y+23),GREEN,4,appear)
            if appear>.82:
                check_badge(d,1498,y+23,16)
    gp=ease((t-2.65)/1.1)
    # A rounded card provides uniform padding; an ellipse caused the bar to
    # visually touch the curved boundary near its left and right ends.
    _rounded(d,(1365,715,1695,895),fill=PALE_BLUE,outline=BLUE,radius=24,width=3)
    d.text((1530-d.textlength("agreement  Ψ",font=_font(19,True))/2,740),"agreement  Ψ",font=_font(19,True),fill=BLUE)
    # Only the stored value is ever printed; it fades in while the bar grows.
    if gp>0:
        col=tuple(int(round(PALE_BLUE[k]+(GREEN[k]-PALE_BLUE[k])*min(1.0,gp*1.6))) for k in range(3))
        d.text((1530-d.textlength(f"{psi:.3f}",font=_font(32,True))/2,785),f"{psi:.3f}",font=_font(32,True),fill=col)
    d.rounded_rectangle((1415,850,1645,871),radius=10,fill=(215,226,220))
    fill_w=max(0,int(230*psi*gp))
    if fill_w>0:
        d.rounded_rectangle((1415,850,1415+fill_w,871),radius=min(10,max(1,fill_w//2)),fill=GREEN)
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


# Column x positions, centred in the 65–1855 panel (status marks included).
LATTICE_X=(185,755,1325)


def lattice_layout(d):
    cols=[(LATTICE_X[0],"GRASP",("power","precision","pinch","hook"),BLUE),
          (LATTICE_X[1],"CONTACT",("point","line","surface"),GREEN),
          (LATTICE_X[2],"FORCE",("push","pull","torque","normal"),AMBER)]
    anchors={}
    for x,title,items,color in cols:
        d.text((x,290),title,font=_font(18,True),fill=color)
        for i,item in enumerate(items):
            y=330+i*72; sel=item in ("power","line","pull")
            _rounded(d,(x,y,x+330,y+48),fill=PALE_GREEN if sel else WHITE,outline=GREEN if sel else BORDER,radius=10,width=2 if sel else 1)
            d.text((x+23,y+10),item,font=_font(18,True),fill=GREEN if sel else INK)
            # Keep connectors outside the option cards.  Earlier versions
            # connected card centres, so diagonals visibly crossed text and
            # card borders.
            anchors[item]={"left":(x,y+24),"right":(x+330,y+24),"center":(x+165,y+24)}
    return anchors


def page2(t,psi):
    pil,d=page_base(1,"The physics gate prunes representative triples before Δ⁺ is ranked.")
    # The canvas grows only when the ranked footer arrives.  Keeping the
    # final-height panel from frame one produced a large temporary void.
    rank=ease((t-3.55)/1.05)
    panel_bottom=lerp(770,930,rank)
    _rounded(d,(65,240,1855,panel_bottom),fill=PANEL,outline=BORDER,radius=22,width=2)
    anchors=lattice_layout(d)
    # Slow path construction over the first half of this page.
    pv=ease((t-.25)/1.35); pa=ease((t-.85)/1.35); pr=ease((t-1.45)/1.35)
    arrow(d,anchors["power"]["right"],anchors["line"]["left"],GREEN,7,pv)
    arrow(d,anchors["line"]["right"],anchors["pull"]["left"],GREEN,7,pv)
    arrow(d,anchors["power"]["right"],anchors["surface"]["left"],AMBER,4,pa)
    arrow(d,anchors["surface"]["right"],anchors["normal"]["left"],AMBER,4,pa)
    arrow(d,anchors["hook"]["right"],anchors["surface"]["left"],RED,4,pr)
    arrow(d,anchors["surface"]["right"],anchors["torque"]["left"],RED,4,pr)
    gate=ease((t-2.35)/.85)
    d.text((LATTICE_X[0],630),"Λ  PHYSICS GATE",font=_font(18,True),fill=INK)
    d.text((LATTICE_X[0]+200,632),"constraint weight assigned to each representative path",font=_font(16),fill=MUTED)
    if gate>0:
        gate_cards=[
            (LATTICE_X[0],"VALID","×1","power · line · pull",GREEN,PALE_GREEN),
            (LATTICE_X[1],"IMPROBABLE","×0.45","power · surface · normal",AMBER,PALE_AMBER),
            (LATTICE_X[2],"FORBIDDEN","×0","hook · surface · torque",RED,PALE_RED),
        ]
        for x,label,weight,path,color,fill in gate_cards:
            _rounded(d,(x,670,x+330,740),fill=fill,outline=color,radius=13,width=2)
            d.text((x+20,680),label,font=_font(15,True),fill=color)
            wf=_font(17,True); d.text((x+310-d.textlength(weight,font=wf),678),weight,font=wf,fill=color)
            d.text((x+20,709),path,font=_font(15,True),fill=INK)
        if gate>.65:
            # Terminal decisions sit in a dedicated status column outside
            # the FORCE cards, rather than obscuring their labels.
            px,py=anchors["pull"]["right"]
            check_badge(d,px+38,py,17)
            for item,col in (("normal",AMBER),("torque",RED)):
                x,y=anchors[item]["right"]
                cross_badge(d,x+38,y,17,col)
    # Ranked result grows in the same page instead of changing screen.
    if rank>.06:
        # Keep the lattice visible and introduce its ranked output in the
        # unused lower band instead of covering the centre of the page.
        y1=790; y2=878
        x1=lerp(960,115,rank); x2=lerp(960,1805,rank)
        if x2-x1>60: _rounded(d,(x1,y1,x2,y2),fill=WHITE,outline=GREEN,radius=14,width=3)
        if rank>.72:
            d.ellipse((135,y1+21,181,y1+67),fill=GREEN)
            d.text((151,y1+26),"1",font=_font(21,True),fill=WHITE)
            d.text((205,y1+16),"Δ⁺ ranked hypothesis",font=_font(15,True),fill=MUTED)
            d.text((205,y1+44),"power  ·  line  ·  pull",font=_font(20,True),fill=INK)
            pill(d,(545,y1+22),f"Ψ  {psi:.3f}",BLUE,PALE_BLUE,12,14)
            pill(d,(705,y1+22),"Λ  1",GREEN,PALE_GREEN,12,14)
            d.rounded_rectangle((875,y1+33,1245,y1+55),radius=11,fill=(213,226,218))
            d.rounded_rectangle((875,y1+33,875+int(370*psi),y1+55),radius=11,fill=GREEN)
            d.text((1265,y1+28),f"{psi:.3f}",font=_font(16,True),fill=GREEN)
            d.text((1485,y1+31),"forward to Stage 3",font=_font(16,True),fill=BLUE)
            arrow(d,(1695,y1+44),(1770,y1+44),BLUE,4,1.0)
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def load_psi(path):
    data=json.loads(Path(path).read_text()); return float(data["tool_primitives"][0]["geometry_vlm_agreement"])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results-json",default="results/regiomanip2/color_000012_run3/interaction_lift/interaction_lift_results.json")
    ap.add_argument("--stage1-video",default="results/video/regiomanip_stage1_final/regiomanip_stage1_FINAL.mp4")
    ap.add_argument("--output-dir",default="results/video/regiomanip_stage2_two_page")
    ap.add_argument("--fps",type=int,default=15); ap.add_argument("--duration",type=float,default=12.0)
    args=ap.parse_args(); psi=load_psi(args.results_json)
    out=Path(args.output_dir); frames=out/"frames"; frames.mkdir(parents=True,exist_ok=True)
    cap=cv2.VideoCapture(args.stage1_video); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,n-1)); ok,last=cap.read(); cap.release()
    if not ok: last=np.full((H,W,3),255,np.uint8)
    page1_len=4.9; P=page1_len
    tmap=TimeMap(args.duration,slow=1.4,holds=((1.5,1.6),(2.45,1.0),(P+2.9,.5),(P+3.3,1.0),(args.duration,1.0)))
    total=int(round(tmap.real_end*args.fps))
    narr=Narrator(((0,"From region shape and the instruction alone, geometry predicts grasp, contact, and force."),
                   (1.65,"A depth-consistency gate checks the grounded regions in 3D."),
                   (2.5,f"Geometry and the VLM agree on power, line, and pull:  agreement Ψ = {psi:.3f}."),
                   (P,"The physics lattice connects grasp → contact → force for the candidate triples."),
                   (P+2.35,"Each path receives a physics weight Λ:  valid ×1,  improbable ×0.45,  forbidden ×0."),
                   (P+3.55,f"The valid triple power · line · pull is ranked first (Ψ = {psi:.3f}) and passed to Stage 3.")))
    for i in range(total):
        real=i/args.fps; t=tmap(real)
        scene=page1(t,psi) if t<page1_len else page2(t-page1_len,psi)
        if real<.5: scene=fade(scene,ease(real/.5))
        elif page1_len<=t<page1_len+.6: scene=transition(page1(page1_len,psi),scene,(t-page1_len)/.6)
        step=0 if t<1.65 else 1 if t<2.5 else 2 if t<P else 3 if t<P+3.55 else 4
        text,na=narr(t)
        scene=apply_chrome(scene,2,STAGE2_STEPS,step,text,na)
        end=(total-i)/args.fps
        if end<.6: scene=fade(scene,ease(end/.6))
        cv2.imwrite(str(frames/f"{i:06d}.png"),scene)
    video=out/"regiomanip_stage2_two_page.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-framerate",str(args.fps),"-i",str(frames/"%06d.png"),*ffmpeg_quality_args(),str(video)],check=True)
    manifest={"video":str(video),"fps":args.fps,"duration_seconds":total/args.fps,"resolution":[W,H],"pages":["geometry_check_and_agreement","physics_lattice_and_ranking"],"geometry_vlm_agreement":psi}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2)); print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
