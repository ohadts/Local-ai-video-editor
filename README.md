# Family Trip AI Editor

Automated highlight-reel pipeline for footages. A vision-language model
(Qwen3-VL) watches every clip, decides what's worth keeping, and FFmpeg
stitches the result into a single video with crossfades.

```
footage folder -> scan + sort + dedupe -> AI analysis (per-segment) -> filter by score -> chunked render -> trip.mp4
```

## Quickstart

```bash
# Prereqs
pip install requests pillow
ollama pull qwen3-vl:8b          # ~5 GB

# Run
python pipeline.py \
  --input  /path/to/footage \
  --output /path/to/trip.mp4 \
  --min-score 7 \
  --max-duration 30
```

## Prerequisites

| Tool   | Install                                                        |
|--------|----------------------------------------------------------------|
| Python | 3.10+ (3.12 if you want the transformers backend with ROCm)    |
| FFmpeg | https://ffmpeg.org or `brew install ffmpeg`                    |
| Ollama | https://ollama.com  (then `ollama pull qwen3-vl:8b`)           |
| pip    | `pip install requests pillow`                                  |

For the optional transformers backend (better quality, sends actual video
clips to Qwen3-VL instead of extracted frames):

```bash
pip install -r requirements_transformers.txt
```

On AMD GPUs (Windows + Radeon RX 7000/9000 series), use AMD's official ROCm
PyTorch wheels — see the AMD ROCm section below.

## How it works

### 1. Discovery and sorting
The pipeline scans the input folder recursively for `.mp4 / .mov / .jpg / .heic` etc.

- **Photos** are sorted by file modification time (true capture order)
- **Photo bursts** within 4 seconds are collapsed to a single representative shot
- **Videos** use chapter+segment numbering (handles split clips correctly)
- Pipeline-managed folders (`_chunks/`, `_ffmpeg_work/`) are excluded from the scan
- Final order interleaves photos and videos by capture time

### 2. Segment splitting
Each video is split into ~30-second segments using fixed intervals plus any
scene-cut boundaries detected by ffprobe. So a 5-minute clip with a shaky
start, good middle, and boring end becomes 3-6 independently-judged segments.

### 3. AI analysis
For every segment, the chosen backend asks Qwen3-VL to return JSON:

```json
{
  "description": "Kids running on the beach at sunset",
  "has_people": true,
  "quality": "good",
  "highlight_score": 8,
  "keep": true,
  "suggested_speed": 1.0,
  "notes": ""
}
```

The prompt instructs the model to be ruthless — most footage gets score 0-4
and is cut. Only the strongest 7+ scores earn a slot.

Results are cached to `<output_stem>_analysis.json` and written **after every
segment** with atomic file-replace. Ctrl+C never corrupts the cache, and
restarting the pipeline resumes from where it stopped.

### 4. Filtering
Three layers on top of the model's keep/skip decision:

- `--min-score N` — drop anything below threshold N (default 7)
- `--max-segments N` — keep only top-N by score (chronological order preserved)
- `--max-duration MIN` — drop lowest-scored segments until total runtime fits

The filter never mutates the cache, so you can experiment freely with
different thresholds without re-analyzing.

### 5. Chunked render
Segments are rendered in batches (default 30 per chunk) into intermediate
`.mp4` files, then concatenated losslessly. Why:

- **Survives Windows command-line limits** (CreateProcess caps at ~32K chars)
- **Survives OOM** — failed chunks auto-subdivide and retry (see Recovery)
- **Resumes** — completed chunks are skipped on restart
- **Isolates failures** — one bad chunk doesn't lose hours of rendering

## All CLI flags

### Input / output (required)
```
--input  PATH       Folder containing footages + photos
--output PATH       Output .mp4 path
```

### Analysis backend
```
--backend {ollama, transformers}     Default: ollama
--ollama-url URL                     Default: http://localhost:11434
--transformers-model HF_ID           Default: Qwen/Qwen3-VL-8B-Instruct
--quantize-4bit                      4-bit NF4 quantization (saves VRAM)
--max-frames N                       Frames per segment (transformers only, default 16)
```

### Analysis control
```
--skip-analysis     Skip AI entirely — stitch full clips in order
--clear-cache       Delete cache and re-analyze from scratch
--dry-run           Build plan and print summary but don't render
--scene-threshold F Scene-cut sensitivity 0-1 (default 0.3, lower = more cuts)
--segment-interval S Seconds per analysis segment (default 30)
```

### Selection / tightening
```
--min-score F       Minimum highlight_score 0-10 to keep (default 7)
--max-segments N    Cap to top-N segments by score
--max-duration MIN  Cap final cut to N minutes (drops lowest-scored first)
```

### Rendering
```
--encoder {libx264, h264_amf, hevc_amf}   Default: libx264
--chunk-size N      Segments per render chunk (default 30)
```

### Tool paths
```
--ffmpeg  PATH      Override ffmpeg binary location
--ffprobe PATH      Override ffprobe binary location
```

Resolution order for the binaries: CLI flag -> `FFMPEG_PATH` / `FFPROBE_PATH`
environment variables -> PATH -> common Windows install locations.

## Two analysis backends

### `--backend ollama` (default)

Sends 3 extracted JPEG frames per segment to a running Ollama instance.

```bash
ollama pull qwen3-vl:8b
python pipeline.py --input ./footage --output trip.mp4
```

- Easy setup, runs as a separate server
- Frame-based (no temporal understanding)
- Fast: ~5-15 seconds per segment
- Works on Linux/Mac/Windows with NVIDIA, AMD, or CPU

### `--backend transformers`

Sends actual video segments to Qwen3-VL via Hugging Face Transformers.

```bash
pip install -r requirements_transformers.txt
python pipeline.py --input ./footage --output trip.mp4 --backend transformers
```

- Better quality (model sees motion, not snapshots)
- Slower: ~15-60 seconds per segment depending on GPU
- Loads model into Python process (~16 GB VRAM at fp16 for 8B)
- With `--quantize-4bit`, fits in ~6 GB VRAM

### AMD ROCm on Windows for transformers

If you have a supported AMD GPU (RX 7000/9000 series, RDNA 3/4) and want to
run the transformers backend on GPU:

```cmd
REM Requires Python 3.12 specifically — create a dedicated venv
python -m venv venv-rocm
venv-rocm\Scripts\activate

REM Install ROCm SDK packages, then PyTorch
pip install --no-cache-dir ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_core-7.2.1-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_devel-7.2.1-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm_sdk_libraries_custom-7.2.1-py3-none-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/rocm-7.2.1.tar.gz

pip install --no-cache-dir ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torch-2.9.1+rocm7.2.1-cp312-cp312-win_amd64.whl ^
  https://repo.radeon.com/rocm/windows/rocm-rel-7.2.1/torchvision-0.24.1+rocm7.2.1-cp312-cp312-win_amd64.whl

pip install transformers>=4.57 qwen-vl-utils>=0.0.14 accelerate pillow requests
```

Requires Radeon Adrenalin driver 26.2.2 or newer. ROCm 7.2.x on Windows is
still a preview — expect occasional hangs (see Recovery below).

## GPU-accelerated encoding (AMF)

On AMD GPUs, `--encoder h264_amf` or `--encoder hevc_amf` uses the hardware
video encoder via AMD Media Framework. Honest tradeoffs:

| Encoder      | Speed             | Quality   | File size | Use when                          |
|--------------|-------------------|-----------|-----------|-----------------------------------|
| `libx264`    | 0.3-1x realtime   | Excellent | Baseline  | Default; best for archival        |
| `h264_amf`   | 5-10x realtime    | Very good | ~20% larger at same visual quality | Fast family videos, you want speed |
| `hevc_amf`   | 3-5x realtime     | Very good | ~40% smaller | You want compact files, can give up some compatibility |

AMF works on Windows out of the box with current Radeon drivers — no extra
install needed. `h264_amf` chunks concat fine with `libx264` chunks (both
are H.264 in MP4), so you can mix them across a single run if needed.

## Recovery mechanisms

The pipeline assumes things will go wrong — long runs on consumer hardware
hit OOM, driver hangs, Ctrl+C, etc. Every stage is designed to survive.

### Analysis stage

- **Atomic cache writes** — JSON is written to `<file>.tmp` then renamed, so
  Ctrl+C during the write cannot corrupt the file.
- **Per-segment checkpointing** — cache writes after every successful
  segment, never just at the end. Worst-case loss from a crash: one segment.
- **Stable segment IDs** — segments are identified by `source_path + start_time`.
  Renaming the output file doesn't break resume; reordering source folders
  doesn't either (as long as files don't move).
- **Resume on restart** — running the same command after a crash skips any
  segment whose ID is already in the cache.

### Render stage

- **Chunked rendering** — segments are batched into separate render passes.
  Each chunk is its own ffmpeg invocation with its own output file.
- **Chunk resume** — finished chunks (`_chunks/chunk_NNN.mp4` larger than 1KB)
  are skipped on rerun.
- **Auto-subdivide on failure** — if a chunk fails for any reason (OOM,
  driver crash, exit code != 0), the renderer:
    1. Deletes the partial output
    2. Splits the chunk in half
    3. Renders each half independently with its own retry-on-failure
    4. Recurses down to single segments if needed
  Subchunks are named deterministically (`chunk_d1_chunk_001_n15.mp4`) so
  resume works after Ctrl+C even during a subdivide.
- **Hang detection** — output file size is monitored every 15 seconds. If no
  growth for 3 minutes (typical AMF driver hang), ffmpeg is killed and the
  chunk is subdivided. No more babysitting hung renders.
- **Lossless final concat** — chunk outputs are joined with the ffmpeg
  concat demuxer (no re-encode), so the final file has zero additional
  quality loss regardless of how many subdivisions happened.

### Path-shortening (Windows command-line limit)

Windows' `CreateProcess` rejects commands longer than ~32K characters. With
many segments and long source paths, the assembled ffmpeg command easily
blows past this. The pipeline applies four mitigations:

1. **Input deduplication** — each unique source video gets one `-i` even if
   many segments come from it; segments are addressed via `trim=start=...`
   inside the filtergraph.
2. **Common-prefix relative paths** — if all sources share a parent
   directory, ffmpeg runs with `cwd=common_parent` and inputs become
   relative. Saves thousands of characters.
3. **Hardlink fallback** — if (2) doesn't apply (cross-drive sources), the
   pipeline creates `_ffmpeg_work/` and hardlinks all sources there with
   short names like `v0001.mp4`. Hardlinks need either elevated privileges
   or Windows Developer Mode; falls back to copies (uses real disk space) if
   neither is available.
4. **Filter externalization** — the `filter_complex` string is always
   written to `filter_complex.txt` and passed via `-filter_complex_script`,
   regardless of total size.

## When things go wrong

### "WinError 206: command line too long"
Path-shortening should prevent this. If you still hit it:
- Make sure you have enough free space for `_ffmpeg_work/`
- Or enable Windows Developer Mode (Settings -> For developers) to let
  hardlinks work without admin
- Or move sources to a path with fewer nested folders

### Render hangs without error
Common with AMF on Windows. The new hang detection kills hung ffmpeg
processes after 3 minutes and triggers subdivide. If a chunk hangs
repeatedly, fall back to `--encoder libx264` (CPU; slower but reliable).

### OOM during render
The auto-subdivide handles this — let it run. If a single segment can't
render, that source file may be corrupt; check ffprobe on it directly.

### "Analysis unavailable" for every segment
Either the model isn't actually serving requests (check `ollama list`),
or the prompt is returning non-JSON. Check the warnings logged during
analysis.

### Output is too long / has too much mediocre footage
Either re-analyze with `--clear-cache` (newer prompts may be tighter),
or just turn up `--min-score 8` and lower `--max-duration` — the filter
is applied fresh on every run from the existing cache.

### `_ffmpeg_work/` is huge
If hardlinks failed and the pipeline fell back to copying, this folder
holds duplicates of all your source media. Enable Windows Developer Mode
(or run elevated once) to get real hardlinks. Safe to delete after the
render finishes.

## Editing the analysis JSON manually

The cache JSON is human-readable and editable. After a run, open it and
adjust per-segment fields:

```json
{
  "source_path": "D:\\users\\ohad\\Videos\\Trip1\\...\\GH010100.MP4",
  "source_type": "video",
  "start": 30.0,
  "end": 60.0,
  "duration": 30.0,
  "description": "Family walks along a beach at sunset",
  "highlight_score": 6,
  "keep": false,
  "suggested_speed": 1.0
}
```

Change `"keep": true` to force-include a segment, bump `"highlight_score"`
to keep it past `--min-score`, raise `"suggested_speed": 2.0` to fast-forward
a boring stretch.

Then rerun *without* `--clear-cache` — the FFmpeg stage will use your edits.

## Project files

```
pipeline.py                    Orchestrator (sort, analyze, filter, render)
analyzer_transformers.py       Optional Hugging Face Transformers backend
CLAUDE.md                      Claude Code project instructions
README.md                      This file
requirements_transformers.txt  pip deps for the transformers backend
.gitignore                     Excludes venvs, caches, output videos
```

After a run, you'll also see (excluded from git):

```
<output>_analysis.json       Cached per-segment analysis (editable)
_chunks/                     Intermediate render outputs
  chunk_000.mp4
  chunk_000_cmd.sh           Debug: exact ffmpeg command used
  chunk_000_filter.txt       Debug: externalized filter graph
  concat_list.txt            Final concat manifest
_ffmpeg_work/                Hardlinked/copied source media (if needed)
```

Delete `_chunks/` and `_ffmpeg_work/` after the render to reclaim disk space.