# GoPro Family Trip Editor — Claude Code Instructions

This project edits GoPro family trip footage into a single polished video.

## What this does
1. Scans a folder of GoPro video chunks + photos
2. Sorts chronologically (GoPro filename pattern -> file timestamp fallback)
3. Splits long videos into segments (every 30s + scene-cut detection)
4. Analyzes each segment with Qwen3-VL -> keep/skip/trim/speed JSON
5. Renders with FFmpeg using crossfades between segments and fade-ins for photos
6. Audio uses chained `acrossfade` to stay locked with video xfades (no drift)

## Two analysis backends

### `--backend ollama` (default, easy)
- Sends 3 extracted JPEG frames per segment to Ollama
- Setup: `ollama pull qwen3-vl:8b`
- Fast, runs as a separate server, simple

### `--backend transformers` (best quality, more setup)
- Sends actual video segments to Qwen3-VL via Hugging Face Transformers
- Model sees motion, not just snapshots -> better at detecting shakiness
- Setup:
  ```
  pip install -r requirements_transformers.txt
  ```
- Loads model into Python process (uses VRAM continuously)
- 8B model needs ~16 GB VRAM; with `--quantize-4bit` ~6 GB

## How to run

```bash
# Default - ollama backend with qwen3-vl:8b
python pipeline.py --input /path/to/footage --output trip.mp4

# Use the transformers backend with the 8B model
python pipeline.py --input /path/to/footage --output trip.mp4 --backend transformers

# Smaller / quantized model if VRAM is tight
python pipeline.py --input ./footage --output trip.mp4 \
  --backend transformers \
  --transformers-model "Qwen/Qwen3-VL-4B-Instruct" \
  --quantize-4bit

# Skip AI analysis entirely - just stitch in chronological order
python pipeline.py --input ./footage --output trip.mp4 --skip-analysis

# Re-run with cache wipe (after changing backend or footage)
python pipeline.py --input ./footage --output trip.mp4 --clear-cache

# Tune segment splitting and scene sensitivity
python pipeline.py --input ./footage --output trip.mp4 \
  --segment-interval 20 \
  --scene-threshold 0.25
```

## Backend comparison

| Aspect          | ollama                       | transformers                 |
|-----------------|------------------------------|------------------------------|
| Setup           | low                          | medium                       |
| Quality         | good                         | better (sees motion)         |
| Speed           | fast                         | slower per segment           |
| VRAM            | shared with ollama server    | held by Python process       |
| API             | per-segment HTTP call        | in-process inference         |

Both backends write the same JSON cache format, so you can switch backends
and the FFmpeg/stitching stage stays unchanged.

## Files

- `pipeline.py` - orchestrator (sort, segment, dispatch to backend, build FFmpeg)
- `analyzer_transformers.py` - transformers / Qwen3-VL backend
- `requirements_transformers.txt` - pip deps for transformers backend
- `<output>_analysis.json` - cached per-segment analysis (editable!)
- `ffmpeg_command.sh` - exact FFmpeg command used (for debugging)

## Editing the JSON cache manually

After any analysis run, open `<output>_analysis.json` and tweak:
- `"keep": false` to exclude a segment
- `"suggested_speed": 2.0` to speed up boring segments
- `"start"` / `"end"` / `"duration"` to fine-tune boundaries

Rerun without `--clear-cache` and the FFmpeg stage uses your edits.

## Common Claude Code requests

- "Run the pipeline on /path/to/Trip with the transformers backend"
- "Show me which segments got skipped and why"
- "Use the 4B model with 4-bit quantization to save VRAM"
- "Edit the cache JSON to keep all the dog clips at 1.5x speed"
- "The render failed - debug ffmpeg_command.sh"
- "Add a 2-second black title card at the start saying 'Hawaii 2024'"
