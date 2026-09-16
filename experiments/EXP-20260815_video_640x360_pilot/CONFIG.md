# EXP-20260815 — 640×360 video transcode pilot

## Purpose

Measure the actual storage reduction of an independent 640×360 derivative and
check that the files remain readable before changing the R5 download policy.
This pilot is a capacity/format check, not a predictor or accuracy result.

## Execution boundary

- Runtime: remote `/root/autodl-tmp/scheduler`
- Input: ten existing 1280×720 Video-MME MP4s
- Output: `experiments/EXP-20260815_video_640x360_pilot/outputs/`
- Originals: read-only; no overwrite or deletion
- Collection: pilot-only; not merged into `trace_collection_v1_legacy_core`

## Encoding

```text
ffmpeg -vf scale=640:360:flags=lanczos -c:v libx264 -preset fast -crf 23
        -pix_fmt yuv420p -c:a aac -b:a 128k -movflags +faststart
```

## Acceptance checks

1. Every output is readable by `ffprobe` and reports 640×360.
2. Each output has a SHA256 and before/after byte count in the pilot manifest.
3. Originals remain byte-for-byte untouched.
4. The measured ratio is used for the R5 capacity decision; no unmeasured ratio
   is promoted to the formal download manifest.
