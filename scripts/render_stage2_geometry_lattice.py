#!/usr/bin/env python3
"""Render Stage 2: geometry agreement and physics-lattice pruning."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import cv2
import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw

from render_contribution_video_step1 import BLUE, INK, MUTED, _font, _rounded, _smoothstep
from render_stage1_3d_layer_decomposition import fit

WHITE=(255,255,255); PANEL=(248,250,252); BORDER=(184,201,215)
GREEN=(22,145,78); RED=(210,52,55); AMBER=(215,125,0); PALE_GREEN=(235,248,241)


def _pil(frame):
    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    return pil,ImageDraw.Draw(pil)


def _back(pil):
    return cv2.cvtColor(np.asarray(pil),cv2.COLOR_RGB2BGR)


def header(frame,active,title,subtitle):
    pil,draw=_pil(frame)
    draw.text((65,42),"RegioManip",font=_font(30,True),fill=BLUE)
    draw.text((65,86),title,font=_font(42,True),fill=INK)
    draw.text((67,142),subtitle,font=_font(21),fill=MUTED)
    labels=("Geometry check","VLM agreement","Physics lattice","Ranked A⁺")
    x0=65
    for i,label in enumerate(labels):
        x=x0+i*300; color=GREEN if i<active else BLUE if i==active else (190,202,212)
        draw.ellipse((x,184,x+25,209),fill=color)
        draw.text((x+37,182),label,font=_font(17,True),fill=INK if i<=active else MUTED)
        if i<3:
            draw.line((x+210,197,x+278,197),fill=(207,217,226),width=3)
    return _back(pil)


def card(draw,box,title,badge=None,badge_color=BLUE):
    x1,y1,x2,y2=box
    _rounded(draw,box,fill=PANEL,outline=BORDER,radius=20,width=2)
    draw.text((x1+25,y1+22),title,font=_font(25,True),fill=INK)
    if badge:
        bw=max(105,int(draw.textlength(badge,font=_font(12,True)))+32)
        _rounded(draw,(x2-bw-20,y1+20,x2-20,y1+58),fill=WHITE,outline=badge_color,radius=9,width=1)
        draw.text((x2-bw-4,y1+30),badge,font=_font(12,True),fill=badge_color)


def check_row(draw,x,y,label,value,color=GREEN):
    draw.ellipse((x,y,x+25,y+25),fill=color)
    draw.text((x+7,y+2),"✓",font=_font(15,True),fill=WHITE)
    draw.text((x+40,y-1),label,font=_font(17,True),fill=INK)
    draw.text((x+40,y+26),value,font=_font(15),fill=MUTED)


def geometry_screen(values):
    frame=np.full((1080,1920,3),255,np.uint8)
    frame=header(frame,0,"Stage 2. Geometry checking and lattice pruning",
                 "Independent mask statistics predict grasp, contact, and force attributes before fusion.")
    pil,draw=_pil(frame)
    card(draw,(65,245,1215,930),"1. Geometry-only predictors","MASK STATISTICS",GREEN)
    card(draw,(1245,245,1855,930),"Geometry tuple","STAGE 2 EVIDENCE",GREEN)

    # Three manuscript-defined predictors. No image-space frame or XYZ axes:
    # those belong to Stage 3 contact-frame lifting.
    predictors=[
        ("Handle elongation + area","grasp predictor","power",BLUE),
        ("Work-end aspect ratio","contact predictor","line",GREEN),
        ("Instruction verb lexicon","force predictor","pull",AMBER),
    ]
    for i,(evidence,predictor,result,color) in enumerate(predictors):
        y=340+i*150
        _rounded(draw,(110,y,515,y+92),fill=WHITE,outline=(194,207,218),radius=14,width=2)
        draw.text((135,y+18),evidence,font=_font(18,True),fill=INK)
        draw.text((135,y+52),"independent evidence",font=_font(14),fill=MUTED)
        draw.line((535,y+46,650,y+46),fill=color,width=5)
        draw.polygon([(650,y+46),(632,y+35),(632,y+57)],fill=color)
        _rounded(draw,(675,y,1135,y+92),fill=PALE_GREEN,outline=color,radius=14,width=2)
        draw.text((700,y+14),predictor,font=_font(15),fill=MUTED)
        draw.text((700,y+47),result,font=_font(23,True),fill=color)

    _rounded(draw,(110,805,1135,875),fill=(238,246,253),outline=BLUE,radius=13,width=2)
    draw.ellipse((140,827,164,851),fill=GREEN)
    draw.text((147,827),"✓",font=_font(14,True),fill=WHITE)
    draw.text((180,824),"Depth-consistency gate passed before geometry prediction",font=_font(17,True),fill=INK)

    # Compact geometry-only tuple; this is compared with the VLM on the next screen.
    tuple_rows=(("grasp type","power",BLUE),("contact type","line",GREEN),("force mode","pull",AMBER))
    for i,(name,value,color) in enumerate(tuple_rows):
        y=350+i*135
        _rounded(draw,(1290,y,1810,y+94),fill=WHITE,outline=color,radius=14,width=2)
        draw.text((1320,y+16),name,font=_font(16),fill=MUTED)
        draw.text((1320,y+48),value,font=_font(23,True),fill=color)
        draw.ellipse((1748,y+32,1778,y+62),fill=color)
        draw.text((1756,y+34),"✓",font=_font(17,True),fill=WHITE)
    _rounded(draw,(1290,800,1810,875),fill=PALE_GREEN,outline=GREEN,radius=13,width=2)
    draw.text((1370,824),"ready for VLM agreement",font=_font(18,True),fill=GREEN)
    return _back(pil)


def agreement_screen(values):
    frame=np.full((1080,1920,3),255,np.uint8)
    frame=header(frame,1,"Stage 2. Geometry checking and lattice pruning",
                 "Independent geometry predictions are compared with the structured VLM tuple.")
    pil,draw=_pil(frame)
    card(draw,(65,260,690,900),"VLM prediction","SEMANTIC",BLUE)
    card(draw,(1230,260,1855,900),"Geometry-only prediction","METRIC",GREEN)
    for x,title,color in ((100,"RegionAffordanceTuple",BLUE),(1265,"Mask statistics + PCA",GREEN)):
        draw.text((x,350),title,font=_font(20,True),fill=color)
    attrs=(('Grasp','power'),('Contact','line'),('Force','pull'))
    for i,(name,value) in enumerate(attrs):
        y=420+i*120
        for x,color in ((105,BLUE),(1270,GREEN)):
            _rounded(draw,(x,y,x+540,y+78),fill=WHITE,outline=color,radius=12,width=2)
            draw.text((x+22,y+15),name,font=_font(16),fill=MUTED)
            draw.text((x+250,y+13),value,font=_font(21,True),fill=color)
    # Central agreement operator.
    draw.line((690,580,790,580),fill=(176,193,207),width=4)
    draw.polygon([(790,580),(772,569),(772,591)],fill=(176,193,207))
    draw.line((1130,580,1230,580),fill=(176,193,207),width=4)
    draw.polygon([(1130,580),(1148,569),(1148,591)],fill=(176,193,207))
    draw.ellipse((790,420,1130,760),fill=(238,246,253),outline=BLUE,width=5)
    draw.text((884,468),"agreement",font=_font(20,True),fill=BLUE)
    draw.text((874,535),"Ψ",font=_font(58,True),fill=INK)
    draw.text((873,616),f"{values['psi']:.3f}",font=_font(34,True),fill=GREEN)
    draw.text((825,678),"JSD-based consistency",font=_font(16),fill=MUTED)
    _rounded(draw,(550,830,1370,890),fill=PALE_GREEN,outline=GREEN,radius=14,width=2)
    draw.text((690,847),"VLM and geometry agree on power · line · pull",font=_font(18,True),fill=GREEN)
    return _back(pil)


def lattice_screen(pruned=False):
    frame=np.full((1080,1920,3),255,np.uint8)
    active=2
    subtitle=("Forbidden and improbable triples are removed before hypotheses are forwarded."
              if pruned else "The 4 × 3 × 4 lattice tests grasp, contact, and force combinations.")
    frame=header(frame,active,"Stage 2. Geometry checking and lattice pruning",subtitle)
    pil,draw=_pil(frame)
    card(draw,(65,250,1855,930),"3. Physics-consistency lattice  Λ","48 CANDIDATE TRIPLES",AMBER)
    cols=[(150,"Grasp type",("power","precision","pinch","hook"),BLUE),
          (720,"Contact type",("point","line","surface"),GREEN),
          (1290,"Force mode",("push","pull","torque","normal"),AMBER)]
    centers={}
    for x,title,items,color in cols:
        draw.text((x,325),title,font=_font(21,True),fill=color)
        for i,item in enumerate(items):
            y=385+i*95
            selected=item in ("power","line","pull")
            fill=PALE_GREEN if selected else WHITE
            outline=GREEN if selected else (194,207,218)
            _rounded(draw,(x,y,x+350,y+62),fill=fill,outline=outline,radius=12,width=3 if selected else 1)
            draw.text((x+28,y+17),item,font=_font(18,True),fill=GREEN if selected else INK)
            centers[item]=(x+175,y+31)
    # Selected valid path.
    for a,b in (("power","line"),("line","pull")):
        draw.line((*centers[a],*centers[b]),fill=GREEN,width=7)
    # Example rejected paths, dashed/faded when pruning is active.
    rejected=[("hook","surface","torque"),("power","surface","normal")]
    for triple in rejected:
        color=(225,190,190) if not pruned else RED
        width=3 if not pruned else 5
        for a,b in ((triple[0],triple[1]),(triple[1],triple[2])):
            draw.line((*centers[a],*centers[b]),fill=color,width=width)
        if pruned:
            x,y=centers[triple[2]]; draw.line((x-14,y-14,x+14,y+14),fill=RED,width=5); draw.line((x+14,y-14,x-14,y+14),fill=RED,width=5)
    if pruned:
        _rounded(draw,(125,815,1795,890),fill=(255,246,246),outline=RED,radius=14,width=2)
        draw.text((165,830),"Example pruning:",font=_font(17,True),fill=RED)
        draw.text((355,830),"hook + surface + torque → forbidden (Λ = 0)",font=_font(16),fill=INK)
        draw.text((980,830),"power + line + pull → valid (Λ = 1)",font=_font(16,True),fill=GREEN)
    else:
        draw.text((760,840),"Λ ∈ { 0, 0.45, 1 }",font=_font(22,True),fill=AMBER)
    return _back(pil)


def ranked_screen(values):
    frame=np.full((1080,1920,3),255,np.uint8)
    frame=header(frame,3,"Stage 2. Geometry checking and lattice pruning",
                 "Mechanically valid affordance hypotheses are ranked and forwarded to 3D lifting.")
    pil,draw=_pil(frame)
    card(draw,(180,270,1740,900),"Ranked affordance simplex  Δ⁺","STAGE 2 OUTPUT",GREEN)
    # Selected measured result.
    _rounded(draw,(240,370,1680,510),fill=PALE_GREEN,outline=GREEN,radius=18,width=3)
    draw.text((275,397),"1",font=_font(34,True),fill=GREEN)
    draw.text((355,392),"power  ·  line  ·  pull",font=_font(27,True),fill=INK)
    draw.text((355,438),f"geometry–VLM agreement Ψ = {values['psi']:.3f}   ·   lattice weight Λ = 1",font=_font(17),fill=MUTED)
    # Confidence bar uses agreement, not a fabricated hypothesis probability.
    draw.rounded_rectangle((1160,414,1580,438),radius=12,fill=(220,229,223))
    draw.rounded_rectangle((1160,414,1160+int(420*values['psi']),438),radius=12,fill=GREEN)
    draw.text((1595,410),f"{values['psi']:.3f}",font=_font(18,True),fill=GREEN)
    _rounded(draw,(240,560,1680,685),fill=WHITE,outline=(198,211,221),radius=16,width=2)
    draw.text((275,588),"A⁺",font=_font(29,True),fill=BLUE)
    draw.text((355,586),"Valid alternatives remain available for downstream repair",font=_font(21,True),fill=INK)
    draw.text((355,628),"No single semantic guess is committed before geometry and mechanics are checked.",font=_font(16),fill=MUTED)
    _rounded(draw,(505,755,1415,835),fill=(238,246,253),outline=BLUE,radius=16,width=2)
    draw.text((630,778),"Forward valid hypotheses  →  Stage 3: contact-frame lifting",font=_font(20,True),fill=BLUE)
    return _back(pil)


def load_values(path):
    data=json.loads(Path(path).read_text())
    tool=data["tool_primitives"][0]
    return {
        "psi":float(tool["geometry_vlm_agreement"]),
        "depth":float(tool["touch_xyz"][2]),
        "axis":tool["axis_direction_3d"],
        "normal":tool["contact_normal_3d"],
    }


def blend(a,b,p):
    p=_smoothstep(float(np.clip(p,0,1)))
    return cv2.addWeighted(b,p,a,1-p,0)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results-json",default="results/regiomanip2/color_000012_run3/interaction_lift/interaction_lift_results.json")
    ap.add_argument("--stage1-video",default="results/video/regiomanip_stage1_final/regiomanip_stage1_FINAL.mp4")
    ap.add_argument("--output-dir",default="results/video/regiomanip_stage2_final")
    ap.add_argument("--fps",type=int,default=15)
    ap.add_argument("--duration",type=float,default=12.0)
    args=ap.parse_args()

    out=Path(args.output_dir); frames=out/"frames"; frames.mkdir(parents=True,exist_ok=True)
    values=load_values(args.results_json)
    screens=[geometry_screen(values),agreement_screen(values),lattice_screen(False),lattice_screen(True),ranked_screen(values)]
    cap=cv2.VideoCapture(args.stage1_video); n=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)); cap.set(cv2.CAP_PROP_POS_FRAMES,max(0,n-1)); ok,last=cap.read(); cap.release()
    if not ok: last=np.full((1080,1920,3),255,np.uint8)
    last=cv2.resize(last,(1920,1080))
    schedule=[(0.0,2.7,0),(2.7,5.1,1),(5.1,7.3,2),(7.3,9.4,3),(9.4,args.duration,4)]
    total=int(args.duration*args.fps)
    for i in range(total):
        t=i/args.fps
        if t<.65:
            frame=blend(last,screens[0],t/.65)
        else:
            idx=next((j for j,(a,b,_) in enumerate(schedule) if a<=t<b),len(schedule)-1)
            a,b,si=schedule[idx]; frame=screens[si].copy()
            if idx>0 and t<a+.35: frame=blend(screens[schedule[idx-1][2]],screens[si],(t-a)/.35)
        cv2.imwrite(str(frames/f"{i:06d}.png"),frame)
    video=out/"regiomanip_stage2_geometry_lattice.mp4"
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),"-y","-framerate",str(args.fps),"-i",str(frames/"%06d.png"),"-c:v","libx264","-preset","slow","-crf","18","-pix_fmt","yuv420p","-movflags","+faststart",str(video)],check=True)
    manifest={"video":str(video),"fps":args.fps,"duration_seconds":args.duration,"resolution":[1920,1080],"geometry_vlm_agreement":values["psi"],"selected_triple":["power","line","pull"],"source":args.results_json}
    (out/"manifest.json").write_text(json.dumps(manifest,indent=2))
    print(json.dumps(manifest,indent=2))


if __name__=="__main__": main()
