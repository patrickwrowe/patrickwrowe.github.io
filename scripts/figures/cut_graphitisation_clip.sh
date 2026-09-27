#!/usr/bin/env bash
# Cut the archive anneal video to a web clip — restructure spec §7 step 4.
#
# Source: the 45 s, 1654 x 1612, 149 MB h264 file Patrick rendered in 2021 from the
# 1.0 g cm^-3, 3500 K graphitisation run. Re-encoded at 720 px wide, CRF 28, no audio,
# faststart, which lands well under the 15 MB target. The poster is the final frame.
#
# ffmpeg is not installed system-wide; the origins_md environment has one.
#   FFMPEG=... scripts/figures/cut_graphitisation_clip.sh
set -euo pipefail

FFMPEG="${FFMPEG:-/home/patrick/micromamba/envs/origins_md/bin/ffmpeg}"
SRC="/data/pr_archive/source_drives/sdb2_Patrick_4Tb_BU/Happy_Electron_Backup/Research/Carbon_Potential/2_Applications/1_Graphitisation/5_Visualisations/dens_1.0_3500K/Graphitisation_1.0.mp4"
OUT_DIR="public/video"
STEM="$OUT_DIR/graphitisation-1.0gcc-3500K"

mkdir -p "$OUT_DIR"
"$FFMPEG" -hide_banner -loglevel error -y -i "$SRC" \
  -vf "scale=720:-2,hue=s=0" -c:v libx264 -crf 28 -preset slow -pix_fmt yuv420p -an -movflags +faststart \
  "$STEM.mp4"
"$FFMPEG" -hide_banner -loglevel error -y -sseof -0.1 -i "$SRC" -frames:v 1 -vf "scale=720:-2,hue=s=0" -q:v 3 "$STEM.jpg"

SIZE=$(stat -c %s "$STEM.mp4")
echo "clip: $SIZE bytes"
if [ "$SIZE" -ge 15000000 ]; then
  echo "clip is over the 15 MB target; raise -crf or shorten with -ss/-t" >&2
  rm -f "$STEM.mp4"
  exit 1
fi
