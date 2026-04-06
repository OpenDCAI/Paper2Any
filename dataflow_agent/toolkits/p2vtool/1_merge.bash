#!/bin/bash

set -euo pipefail

SLIDE_DIR="$1"
VIDEO_DIR="$2"
OUTPUT_DIR="$3"
WIDTH="$4"
HEIGHT="$5"
NUM_VIDEO="$6"
VIDEO_PATH="$7"

mkdir -p "$OUTPUT_DIR"
LIST_FILE="$OUTPUT_DIR/list.txt"
: > "$LIST_FILE"

choose_video_encoder() {
  local preferred="${PAPER2VIDEO_TALKING_MERGE_VIDEO_CODEC:-auto}"
  if [[ "$preferred" == "auto" ]]; then
    if ffmpeg -encoders 2>/dev/null | grep -q 'h264_nvenc'; then
      echo "h264_nvenc"
    else
      echo "libx264"
    fi
    return 0
  fi
  echo "$preferred"
}

VIDEO_CODEC="$(choose_video_encoder)"

concat_page_clips() {
  if ffmpeg -y -f concat -safe 0 -i "$LIST_FILE" -c copy "$VIDEO_PATH"; then
    return 0
  fi

  echo "stream-copy concat failed, retrying with re-encode" >&2
  rm -f "$VIDEO_PATH"
  ffmpeg -y \
    -f concat -safe 0 -i "$LIST_FILE" \
    -c:v libx264 -preset ultrafast -crf 23 \
    -c:a aac \
    "$VIDEO_PATH"
}

for i in $(seq 1 "$NUM_VIDEO"); do
  slide_path="$SLIDE_DIR/$i.png"
  video_path="$VIDEO_DIR/$((i-1))/digit_person_withaudio.mp4"
  output_path="$OUTPUT_DIR/page_$(printf "%03d" "$i").mp4"

  if [[ ! -f "$slide_path" || ! -f "$video_path" ]]; then
    echo "missing input for page $i: slide=$slide_path video=$video_path" >&2
    exit 1
  fi

  duration="$(ffprobe -v error -show_entries format=duration -of default=noprint_wrappers=1:nokey=1 "$video_path")"
  echo "Processing page $i with codec=$VIDEO_CODEC"

  if [[ "$VIDEO_CODEC" == "h264_nvenc" ]]; then
    ffmpeg -y \
      -loop 1 -t "$duration" -i "$slide_path" \
      -i "$video_path" \
      -filter_complex "[1:v]scale=${WIDTH}:${HEIGHT}[avatar];[0:v][avatar]overlay=W-w-10:10,format=yuv420p[outv]" \
      -map "[outv]" -map 1:a \
      -c:v h264_nvenc -preset p3 -tune hq -cq 28 \
      -c:a copy -shortest "$output_path"
  else
    ffmpeg -y \
      -loop 1 -t "$duration" -i "$slide_path" \
      -i "$video_path" \
      -filter_complex "[1:v]scale=${WIDTH}:${HEIGHT}[avatar];[0:v][avatar]overlay=W-w-10:10,format=yuv420p[outv]" \
      -map "[outv]" -map 1:a \
      -c:v libx264 -preset ultrafast -crf 23 \
      -c:a aac -shortest "$output_path"
  fi

  if [[ ! -s "$output_path" ]]; then
    echo "failed to generate page clip: $output_path" >&2
    exit 1
  fi

  echo "file '$output_path'" >> "$LIST_FILE"
done

if [[ ! -s "$LIST_FILE" ]]; then
  echo "no generated page clips to concat" >&2
  exit 1
fi

concat_page_clips

if [[ ! -s "$VIDEO_PATH" ]]; then
  echo "concat output missing: $VIDEO_PATH" >&2
  exit 1
fi

echo "All page videos merged to: $VIDEO_PATH"
