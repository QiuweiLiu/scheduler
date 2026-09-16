# EXP-20260815 — Result

## Status

Completed as a six-video valid pilot. Four planned candidates were not run;
one 1280×640 candidate was started but excluded because forcing 640×360 would
change its aspect ratio, and its partial output was removed.

## Remote artifacts

- Manifest: `/root/autodl-tmp/scheduler/experiments/EXP-20260815_video_640x360_pilot/pilot_manifest.tsv`
- Valid outputs: `/root/autodl-tmp/scheduler/experiments/EXP-20260815_video_640x360_pilot/outputs/`
- Encoder log: `/root/autodl-tmp/scheduler/experiments/EXP-20260815_video_640x360_pilot/ffmpeg_errors.log`

## Measured result

| quantity | value |
|---|---:|
| valid samples | 6 |
| input bytes | 1,068,528,024 |
| output bytes | 807,370,470 |
| aggregate output/input ratio | 0.755591 |
| aggregate saving | 24.4409% |
| mean per-video ratio | 0.757575 |
| per-video ratio range | 0.494763–0.960285 |

All six valid outputs report `640×360`, H.264 video, and no ffmpeg errors.

## Interpretation

With `libx264 -preset fast -crf 23`, the measured reduction is much smaller and
more variable than the earlier unverified 25%–40% output-size assumption. Some
source files were already low-bitrate, so downscaling saved only about 4%–9%;
the six-sample aggregate saved about 24.4%.

Applying the aggregate ratio only as a planning estimate gives about 23.38 GB
for the remaining 260 videos (about 21.77 GiB), not a formal capacity guarantee.
The R5 policy still needs a second, more aggressive encoding/quality pilot if
the goal is to fit traces and derived data comfortably in the remaining disk.
