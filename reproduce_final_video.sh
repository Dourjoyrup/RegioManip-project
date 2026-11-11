#!/usr/bin/env bash
# Rebuild the complete RegioManip Stage 1–3 video.
#
# All stage clips are rendered losslessly into WORK_DIR (outside the package)
# and the final MP4 is compressed exactly once.  Only the final MP4 is written
# inside the package.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
FPS="${FPS:-30}"
WORK_DIR="${WORK_DIR:-/tmp/regiomanip_stage1_stage2_stage3_rebuild}"
FINAL_DIR="results/video/regiomanip_stage1_stage2_stage3_final"
FINAL_MP4="$FINAL_DIR/regiomanip_stage1_stage2_stage3_FINAL.mp4"

export REGIOMANIP_LOSSLESS=1
export MPLCONFIGDIR="$WORK_DIR/mpl"
export OPEN3D_CPU_RENDERING=true LIBGL_ALWAYS_SOFTWARE=true EGL_PLATFORM=surfaceless

mkdir -p "$WORK_DIR" "$FINAL_DIR"
FFMPEG_BIN="$($PYTHON_BIN -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')"

# Opening overview page
"$PYTHON_BIN" scripts/render_title_cards.py --output-dir "$WORK_DIR/cards" --fps "$FPS"

# Stage 1a: RGB-D + language input, 3D grounding (DINO, SAM 2, borders, skeletons)
"$PYTHON_BIN" scripts/render_stage1_open3d_grounding.py \
  --output-dir "$WORK_DIR/grounding" --fps "$FPS" --duration 10 --intro-duration 3

# Stage 1b: structured 5-image VLM prompt and RegionAffordanceTuple
"$PYTHON_BIN" scripts/render_stage1_layers_together.py \
  --source-frame "$WORK_DIR/grounding/keyframes/grounding_4.png" \
  --output-dir "$WORK_DIR/layers" --fps "$FPS" --duration 9.2 --reveal-at 0.25

# Stage 2: geometry predictors, depth gate, agreement, physics lattice, ranked hypothesis
"$PYTHON_BIN" scripts/render_stage2_two_page_motion.py \
  --stage1-video "$WORK_DIR/layers/regiomanip_hold_then_all_five.mp4" \
  --output-dir "$WORK_DIR/stage2" --fps "$FPS" --duration 10.6

# Stage 3: back-project -> fit primitives -> rank SE(3) at fixed T/C/G -> construct frame
"$PYTHON_BIN" scripts/render_stage3_contact_frame_3d.py --output-dir "$WORK_DIR/stage3" --fps "$FPS"

# Single final encode (web/GitHub friendly: H.264 High, yuv420p, faststart, 2 s GOP)
"$FFMPEG_BIN" -y -v error \
  -i "$WORK_DIR/cards/card_opening.mp4" \
  -i "$WORK_DIR/grounding/regiomanip_grounding_sam_skeleton_3d.mp4" \
  -i "$WORK_DIR/layers/regiomanip_hold_then_all_five.mp4" \
  -i "$WORK_DIR/stage2/regiomanip_stage2_two_page.mp4" \
  -i "$WORK_DIR/stage3/regiomanip_stage3_contact_frame_3d.mp4" \
  -filter_complex '[0:v][1:v][2:v][3:v][4:v]concat=n=5:v=1:a=0[v]' -map '[v]' \
  -c:v libx264 -preset slow -crf 16 -tune animation -profile:v high -level 4.1 \
  -g "$((FPS * 2))" -pix_fmt yuv420p -movflags +faststart "$FINAL_MP4"

"$FFMPEG_BIN" -v error -i "$FINAL_MP4" -f null -
echo "Created: $FINAL_MP4"
echo "Intermediate clips and frames remain in $WORK_DIR (safe to delete)."
