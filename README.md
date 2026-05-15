# GoPro Family Trip Editor

Automated pipeline: **Qwen2.5-VL** analyzes your footage → **FFmpeg** stitches it.

## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.10+ | Already installed on most systems |
| FFmpeg | `brew install ffmpeg` / `apt install ffmpeg` / [ffmpeg.org](https://ffmpeg.org) |
| Ollama | [ollama.com](https://ollama.com) |
| Qwen model | `ollama pull qwen2.5vl:7b` (~5 GB) |

```bash
pip install requests
```

## Quickstart

```bash
# 1. Clone / copy this folder somewhere
cd gopro_editor

# 2. Point it at your footage folder
python pipeline.py \
  --input  /Volumes/GoPro/Hawaii2024 \
  --output ~/Desktop/hawaii_trip.mp4
```

That's it. The script will:
1. Find and sort all `.mp4`, `.jpg`, `.heic`, etc. in the folder
2. Send each clip to Qwen2.5-VL for analysis (~5–15s per clip)
3. Build a crossfade edit plan
4. Render the final video with FFmpeg

## Transitions

| Clip type | Transition |
|-----------|-----------|
| Video → Video | 0.8s crossfade dissolve |
| Image → anything | 0.5s fade-in from black |
| Image hold time | 4 seconds |

Edit `CROSSFADE_DURATION`, `FADE_IN_DURATION`, `IMAGE_HOLD_DURATION` at the top of `pipeline.py` to taste.

## The JSON cache — your edit lever

After Qwen runs, you'll find `hawaii_trip_analysis.json` next to the output file.
Open it and you'll see something like:

```json
[
  {
    "file": "GX010123.MP4",
    "description": "Kids playing on the beach, sunny day",
    "quality": "good",
    "keep": true,
    "trim_start_seconds": 2,
    "trim_end_seconds": 0,
    "suggested_speed": 1.0
  },
  {
    "file": "GX010124.MP4",
    "description": "Shaky walk from car to beach",
    "quality": "shaky",
    "keep": false,
    "trim_start_seconds": 0,
    "trim_end_seconds": 0,
    "suggested_speed": 1.5
  }
]
```

Edit this file, then rerun `pipeline.py` — it will skip Qwen and use your edits directly.

## Using with Claude Code

With `CLAUDE.md` in this folder, Claude Code already knows the project.
Open the folder in Claude Code and just say:

- *"Run the pipeline on /Volumes/Trip, output trip.mp4"*
- *"Show me what Qwen decided to skip"*
- *"The render failed — debug the ffmpeg command"*
- *"Speed up all driving clips to 2x"*
- *"Add a black title card at the start saying 'Costa Rica 2025'"*

## Flags

```
--input PATH          Footage folder (required)
--output PATH         Output .mp4 path (required)
--ollama-url URL      Ollama URL (default: http://localhost:11434)
--skip-analysis       Skip Qwen, stitch in order only
--clear-cache         Force re-analysis even if cache exists
--dry-run             Print FFmpeg command, don't render
```
