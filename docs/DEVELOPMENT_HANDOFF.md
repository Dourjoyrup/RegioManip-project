# RegioManip Stage 1–3 — Codex handoff

Original snapshot: 2026-11-09  
Last updated: 2026-11-11

This is the portable continuation package for the RegioManip manuscript video. Copy the complete folder into a new workspace and ask the next Codex account to read this file before making changes.

## Current objective

Produce a publication-quality visual explanation of:

1. **Stage 1 — Region-level affordance grounding**
2. **Stage 2 — Geometry checking and lattice pruning**
3. **Stage 3 — Contact-frame lifting into SE(3)**

The current combined video is the accepted working baseline. Continue from it; do not redesign completed stages unless explicitly requested.

## Start here

- Complete final video: `results/video/regiomanip_stage1_stage2_stage3_final/regiomanip_stage1_stage2_stage3_FINAL.mp4`
- Manuscript: `Manuscript/regiomanip.pdf`
- Title page: kept privately (not in the public repository)
- Full reproduction script: `reproduce_final_video.sh`

Only the complete Stage 1–3 MP4 is packaged. Intermediate stage videos are intentionally omitted and can be regenerated with the reproduction script.

## Rendering source files

- Stage 1: `render_stage1_open3d_grounding.py`, `render_stage1_layers_together.py`, `render_contribution_video_step1.py`, and `render_stage1_3d_layer_decomposition.py`
- Stage 2: `render_stage2_two_page_motion.py` (current source), plus the geometry-lattice renderers
- Stage 3: `render_stage3_contact_frame_3d.py` (current source)

All renderer paths are under `scripts/`.

## Current visual narrative

### Stage 1

- RGB-D observation and language instruction.
- Stabilized calibrated 3D scene.
- Grounding DINO, SAM 2, borders, and medial skeletons.
- Grounded candidate geometry becomes a structured five-image VLM input.
- VLM output is presented as `RegionAffordanceTuple`.

### Stage 2

- Geometry check and geometry–VLM agreement.
- Grasp/contact/force lattice and physics pruning.
- Valid `power · line · pull` is forwarded as the ranked affordance hypothesis.
- Improbable and forbidden paths remain visible for comparison.

### Stage 3

- Raw RGB-D cloud is shown first.
- Depth outliers, sensor background, and disconnected mask islands are removed before region points are introduced.
- Clean regions are back-projected to metric 3D and G/C/T are displayed.
- Contact primitives are fitted, then removed before ranking.
- SE(3) orientation hypotheses are ranked at the same fixed T/C/G metric points.
- Only after ranking is the selected contact frame constructed.
- Final output shows `T*C ∈ SE(3)`, translation, confidence, and execution readiness.

## Approved visual rules

- 1920×1080 H.264 at 30 fps (≈95 s), with a white publication-style background.
- Semantic palette: work-end red-orange, handle green, target yellow/amber.
- Scientific values and stored SE(3) results are never altered for appearance.
- Stage 3 uses a right-side 3D scene and left-side explanatory text.
- Stage 3 order: back-project → fit primitives → rank hypotheses → construct selected frame.
- Ranking changes orientation while retaining the same measured origins.
- Exposure, line thickness, layout, transitions, and display-axis sign convention are visualization treatments only.

## Data and scientific code included

- `data/000/color_000012.png`
- `data/000/depth_000012.npy`
- Complete `results/regiomanip2/color_000012_run3/` evidence tree
- Depth cleaning, back-projection, primitive fitting, frame hypotheses, interaction lift, temporal filtering, and visualization modules
- Reconstructed assets under `results/video/assets/`

## Tested environment

Python 3.10 with OpenCV, NumPy, Pillow, Open3D, and imageio-ffmpeg. Open3D may report that X11 is unavailable and use headless Vulkan/llvmpipe; this is expected.

## Instructions for the next Codex account

1. Read this document and inspect the complete final MP4.
2. Treat the combined video and included renderers as the accepted baseline.
3. Preserve semantic colors, fixed metric points, and scientific values.
4. Make targeted edits, inspect representative frames, then rebuild the combined video.
5. Do not add generated frame directories to future archives; MP4s, manifests, scripts, inputs, and evidence are sufficient.
6. Continue from the present Stage 3 state.

## Suggested first message in a new session

> Read `README_HANDOFF.md`, inspect the manuscript and current Stage 1–3 final video, then continue the RegioManip publication video from the accepted state. Preserve all approved layouts, colors, scientific values, and the Stage 3 fixed-origin SE(3) ranking convention.

## Revision 2026-11-11 (presentation pass; no scientific values changed)

- Stage/page transitions fade through white; full layouts and subtitles are never superimposed.
- Shared header (`draw_header`) gives identical brand/title/subtitle placement in all stages.
- Stage 1: closer display camera (0.52 → 0.46 m), visible 4 px DINO brackets, 3 px skeletons, tighter intro cards; the "3D SOURCE" card now shows only the clean geometry crop plus a card-scale colour key.
- Stage 2: Ψ prints only the stored value (no count-up through intermediate numbers); lattice centred; minimum label size 15 px.
- Stage 3: wrapped captions (previously overflowed cards); camera dollies in on the cleaned regions after background suppression; larger G/C/T markers; connected step indicator; H1–H4 listed with their stored scores (all 0.680, a tie; H1 = stored best); final card shows readiness threshold 0.60 from the results JSON. Duration 14 → 15 s.
- `reproduce_final_video.sh` renders lossless stage clips in `WORK_DIR` and encodes the final MP4 once (CRF 16, H.264 High, faststart). No intermediate folders are created in the package.
- Added the previously missing dependency `scripts/render_stage1_open3d_rgbd.py`.
- Tested with Python 3.10 and Open3D 0.20.

## Revision 2026-11-11 b (redesign for clarity and pacing; no scientific values changed)

- Shared design system in `scripts/video_style.py`: global Stage 1 → 2 → 3 tracker, per-stage step indicator with check marks and arrows, narration bar with one sentence per step, title cards, shared arrows and check/cross badges, and a pacing timeline (`TimeMap`).
- Typography: Lato, with automatic per-glyph DejaVu fallback (∈, ✓).
- Opening overview page only (`scripts/render_title_cards.py`); stages are introduced by the top-right stage tracker, not full-page cards.
- Stage 1 and Stage 3 share one layout: explanation panel on the left (step, title, explanation, checklist or result, input → output), 3D view on the right.
- Stage 3: checklists per step; primitives appear one at a time with labels in 3D; hypotheses shown with axis icons and stored scores; labelled target and tool frames.
- Pacing: about 49 s → 95 s at 30 fps. Static holds are 2–3.5 s (opening page and final result about 4.5 s); held 3D views drift slowly (≤4°, back to the exact final pose) so they never freeze.
- Smoothness: every discrete 3D state change (background removal, borders, primitives, hypotheses, final frame) is cross-dissolved over 0.45 s; DINO brackets fade in from white. A frame-by-frame audit found no abrupt jumps or flash frames.
- Flash fix: headless Open3D occasionally returned an incomplete (mostly black or empty) frame. `GroundingRenderer._render_checked` now validates every 3D render and re-renders bad frames; the final video was scanned frame by frame and contains no flash frames.

## Scientific caveat

The videos explain stored experimental results. Display exposure, point size, cards, line thickness, transitions, and canonical display signs do not modify RGB-D data, masks, inferred assignments, scores, or stored transformations.
