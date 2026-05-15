#!/usr/bin/env python3
"""
GoPro Family Trip Video Editor Pipeline  (segment-aware edition)
=================================================================
Stage 1: Scan & sort media files (GoPro filename -> mtime fallback)
Stage 2: Scene-detect each video -> split into segments
Stage 3: Analyze each segment with Qwen via Ollama -> per-segment JSON
Stage 4: Build FFmpeg filter_complex from kept segments
Stage 5: Render final video

Usage:
    python pipeline.py --input /path/to/footage --output trip.mp4
    python pipeline.py --input ./footage --output trip.mp4 --skip-analysis
    python pipeline.py --input ./footage --output trip.mp4 --clear-cache
    python pipeline.py --input ./footage --output trip.mp4 --dry-run
    python pipeline.py --input ./footage --output trip.mp4 --scene-threshold 0.2
"""

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import requests

# -----------------------------------------------------------------
# Constants  (tweak freely)
# -----------------------------------------------------------------
VIDEO_EXTENSIONS  = {".mp4", ".MP4", ".mov", ".MOV", ".avi", ".AVI",
                     ".mkv", ".MKV", ".m4v", ".M4V"}
IMAGE_EXTENSIONS  = {".jpg", ".jpeg", ".JPG", ".JPEG",
                     ".png", ".PNG", ".heic", ".HEIC"}

CROSSFADE_DURATION  = 0.8   # seconds - dissolve between segments
FADE_IN_DURATION    = 0.5   # seconds - fade-in for photos
IMAGE_HOLD_DURATION = 4.0   # seconds - how long each photo is shown
TARGET_FPS          = 30    # normalize all streams to this fps

# Scene detection: lower = more sensitive (detects more cuts)
SCENE_THRESHOLD      = 0.3
MIN_SEGMENT_DURATION = 2.0  # ignore segments shorter than this
SEGMENT_INTERVAL     = 30.0 # split every N seconds (primary splitting strategy)

# Frames sampled per segment for Qwen analysis
SEGMENT_FRAME_POSITIONS = [0.15, 0.5, 0.85]

QWEN_MODEL         = "qwen3-vl:8b"   # ollama tag — change to whatever you have pulled
DEFAULT_OLLAMA_URL = "http://localhost:11434"

# -----------------------------------------------------------------
# FFmpeg / FFprobe paths
# -----------------------------------------------------------------
# Override these if ffmpeg isn't on PATH. Either set them here directly:
#   FFMPEG_PATH  = r"C:\ffmpeg\bin\ffmpeg.exe"
#   FFPROBE_PATH = r"C:\ffmpeg\bin\ffprobe.exe"
# Or use the --ffmpeg / --ffprobe CLI flags at runtime, or set FFMPEG_PATH /
# FFPROBE_PATH environment variables.
FFMPEG_PATH  = os.environ.get("FFMPEG_PATH",  "ffmpeg")
FFPROBE_PATH = os.environ.get("FFPROBE_PATH", "ffprobe")


def _resolve_binary(name: str, configured: str) -> str:
    """
    Resolve an ffmpeg/ffprobe binary path.
    1) If `configured` is an existing file, use it directly.
    2) Otherwise look it up on PATH via shutil.which.
    3) On Windows, also try common install locations.
    Raises FileNotFoundError with an actionable message if nothing works.
    """
    import shutil
    p = Path(configured)
    if p.is_file():
        return str(p.resolve())

    found = shutil.which(configured)
    if found:
        return found

    # Common Windows install paths to probe automatically
    common = [
        rf"C:\ffmpeg\bin\{name}.exe",
        rf"C:\Program Files\ffmpeg\bin\{name}.exe",
        rf"C:\Program Files (x86)\ffmpeg\bin\{name}.exe",
        os.path.expandvars(rf"%LOCALAPPDATA%\Programs\ffmpeg\bin\{name}.exe"),
        os.path.expandvars(rf"%USERPROFILE%\scoop\apps\ffmpeg\current\bin\{name}.exe"),
    ]
    for cand in common:
        if Path(cand).is_file():
            return cand

    raise FileNotFoundError(
        f"Could not find {name}. Tried: {configured!r} (configured), PATH, and common Windows locations.\n"
        f"  Fix one of:\n"
        f"    1) Edit FFMPEG_PATH / FFPROBE_PATH at the top of pipeline.py\n"
        f"    2) Pass --ffmpeg / --ffprobe on the command line\n"
        f"    3) Set FFMPEG_PATH / FFPROBE_PATH environment variables"
    )

# -----------------------------------------------------------------
# 1.  MEDIA DISCOVERY & SORTING
# -----------------------------------------------------------------

GOPRO_PATTERN = re.compile(r"(?:GX|GH|GOPR|GP)(\d{4,8})(?:_(\d+))?", re.IGNORECASE)

# Photos taken within this many seconds of each other are considered a burst
# and collapsed to a single representative shot.
BURST_WINDOW_SECONDS = 4.0


def gopro_sort_key(path: Path):
    """
    For videos only: prefer GoPro chapter+segment numbering (handles split clips
    correctly), fall back to mtime. Photos are NOT sorted by this — they use
    pure mtime so the user's actual capture timeline is preserved.
    """
    m = GOPRO_PATTERN.search(path.stem)
    if m:
        return (0, int(m.group(1)), int(m.group(2) or 0), path.stat().st_mtime)
    return (1, 0, 0, path.stat().st_mtime)


def collapse_photo_bursts(photos: list) -> list:
    """
    Group photos taken within BURST_WINDOW_SECONDS of each other and keep
    only one representative per burst (the middle one — usually the steadiest).
    Input must already be sorted by mtime ascending.
    """
    if not photos:
        return []

    groups = [[photos[0]]]
    for p in photos[1:]:
        prev = groups[-1][-1]
        gap = p["path"].stat().st_mtime - prev["path"].stat().st_mtime
        if gap <= BURST_WINDOW_SECONDS:
            groups[-1].append(p)
        else:
            groups.append([p])

    representatives = []
    for g in groups:
        rep = g[len(g) // 2]   # middle of the burst
        if len(g) > 1:
            rep["_burst_size"] = len(g)
        representatives.append(rep)
    return representatives


# Directory names the pipeline itself creates — exclude from input scan so
# we never accidentally include our own chunks or hardlinked copies as inputs.
EXCLUDED_DIRS = {"_chunks", "_ffmpeg_work"}


def discover_media(input_dir: Path) -> list:
    raw_videos, raw_photos = [], []
    for p in sorted(input_dir.rglob("*")):
        if not p.is_file():
            continue
        # Skip anything inside a pipeline-managed subdirectory
        if any(part in EXCLUDED_DIRS for part in p.parts):
            continue
        if p.suffix in VIDEO_EXTENSIONS:
            raw_videos.append({"path": p, "type": "video"})
        elif p.suffix in IMAGE_EXTENSIONS:
            raw_photos.append({"path": p, "type": "image"})

    if not raw_videos and not raw_photos:
        print(f"[ERROR] No media found in {input_dir}")
        sys.exit(1)

    # Photos: sort by mtime (actual capture time), then collapse bursts
    raw_photos.sort(key=lambda m: m["path"].stat().st_mtime)
    photos_collapsed = collapse_photo_bursts(raw_photos)
    n_dropped = len(raw_photos) - len(photos_collapsed)

    # Videos: GoPro chapter ordering (handles split clips), with mtime fallback
    raw_videos.sort(key=lambda m: gopro_sort_key(m["path"]))

    # Final order: interleave by mtime so photos and videos appear in true
    # capture order across the whole trip
    media = sorted(
        raw_videos + photos_collapsed,
        key=lambda m: m["path"].stat().st_mtime,
    )

    print(f"\n  Found {len(media)} media items "
          f"({len(raw_videos)} videos, {len(photos_collapsed)} photos"
          + (f", {n_dropped} burst duplicates dropped" if n_dropped else "")
          + ")")
    for i, m in enumerate(media):
        burst = f"  [burst x{m['_burst_size']}]" if m.get("_burst_size") else ""
        print(f"    {i+1:3d}. [{m['type'][:3].upper()}] {m['path'].name}{burst}")
    return media


# -----------------------------------------------------------------
# 2.  SCENE DETECTION -> SEGMENTS
# -----------------------------------------------------------------

def get_video_duration(path: Path) -> float:
    try:
        r = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, timeout=20,
        )
        return float(r.stdout.strip())
    except Exception:
        return 0.0


def detect_cuts_via_ffprobe(path: Path, duration: float) -> list:
    """
    Try ffprobe histogram-based scene detection.
    Returns list of cut timestamps, or empty list if none found / command fails.
    GoPro continuous recordings rarely trigger this — it's a bonus on top of interval splitting.
    """
    try:
        r = subprocess.run(
            [FFPROBE_PATH, "-v", "quiet",
             "-select_streams", "v",
             "-vf", f"select='gt(scene,{SCENE_THRESHOLD})',showinfo",
             "-f", "null", "-",
             str(path)],
            capture_output=True, text=True, timeout=180,
        )
        cuts = []
        for line in r.stderr.splitlines():
            m = re.search(r"pts_time:([\d.]+)", line)
            if m:
                t = float(m.group(1))
                if t > 1.0 and t < duration - 1.0:   # ignore edges
                    cuts.append(t)
        return sorted(set(cuts))
    except Exception:
        return []


def build_segments_for_file(item: dict) -> list:
    """
    Split a video into analysis segments using two strategies combined:

    1. Fixed-interval split every SEGMENT_INTERVAL seconds (primary).
       GoPro clips are long continuous recordings — no hard cuts — so
       histogram scene detection finds nothing. Fixed intervals guarantee
       Qwen sees a representative window of every part of the clip.

    2. ffprobe scene detection cuts (secondary, bonus).
       If genuine hard cuts are found (e.g. camera was stopped/started),
       those boundaries are merged in so segments don't straddle a scene change.

    The two cut lists are merged, deduped, and filtered by MIN_SEGMENT_DURATION.
    """
    path  = item["path"]
    ftype = item["type"]

    if ftype == "image":
        return [{"source_path": str(path), "source_type": "image",
                 "seg_index": 0, "start": 0.0,
                 "end": IMAGE_HOLD_DURATION, "duration": IMAGE_HOLD_DURATION}]

    duration = get_video_duration(path)
    if duration <= 0:
        return []

    # 1. Fixed-interval boundaries
    interval_cuts = [i * SEGMENT_INTERVAL for i in range(int(duration / SEGMENT_INTERVAL) + 1)
                     if i * SEGMENT_INTERVAL < duration]

    # 2. Scene-detection boundaries (bonus — may be empty for GoPro)
    scene_cuts = detect_cuts_via_ffprobe(path, duration)
    if scene_cuts:
        print(f" (+{len(scene_cuts)} scene cuts)", end="", flush=True)

    # Merge, dedupe, sort
    all_cuts = sorted(set(interval_cuts + scene_cuts + [0.0, duration]))

    # Merge cuts that are too close together
    merged = [all_cuts[0]]
    for t in all_cuts[1:]:
        if t - merged[-1] >= MIN_SEGMENT_DURATION:
            merged.append(t)
    if merged[-1] != duration:
        merged.append(duration)

    segments = []
    for i in range(len(merged) - 1):
        start = merged[i]
        end   = merged[i + 1]
        dur   = end - start
        if dur < MIN_SEGMENT_DURATION:
            continue
        segments.append({
            "source_path": str(path),
            "source_type": "video",
            "seg_index":   i,
            "start":       round(start, 3),
            "end":         round(end,   3),
            "duration":    round(dur,   3),
        })

    return segments


# -----------------------------------------------------------------
# 3.  QWEN ANALYSIS  (per segment)
# -----------------------------------------------------------------

SEGMENT_PROMPT_TPL = (
    "You are an editor cutting a SHORT family trip highlight video. "
    "You are looking at {n} frames spanning a {duration:.1f}-second clip.\n\n"
    "Most footage gets cut. Only the very best ~10% earns a slot. "
    "Think of it like a 30-second trailer for the whole trip.\n\n"
    "Return ONLY valid JSON, no markdown:\n"
    '{{\n'
    '  "description": "one sentence about what is happening",\n'
    '  "has_people": true or false,\n'
    '  "quality": "good|shaky|dark|blurry|boring",\n'
    '  "highlight_score": 0,\n'
    '  "keep": true or false,\n'
    '  "suggested_speed": 1.0,\n'
    '  "notes": ""\n'
    '}}\n\n'
    "highlight_score is 0-10, calibrated like this:\n"
    "  0-2: cut without hesitation (boring, shaky, dark, transit, generic scenery)\n"
    "  3-4: nothing special — usually still cut from a tight highlight\n"
    "  5-6: pleasant but generic — keep ONLY if needed for narrative\n"
    "  7-8: genuinely good moment — clear action, emotion, or rare beauty\n"
    "  9-10: must-include — peak moments, the kind you'd show a friend\n\n"
    "Rules:\n"
    "  * keep=true ONLY if highlight_score >= 7\n"
    "  * Generic scenery, walking, driving, transit -> score 0-3 -> cut\n"
    "  * People doing nothing in particular -> score 3-5 -> cut\n"
    "  * People reacting, laughing, doing something specific -> 7+\n"
    "  * Animals, wildlife in clear shot -> 7+\n"
    "  * Truly striking scenery (rare light, unique landscape) -> 7+\n"
    "  * Notable narrative moment (arriving somewhere, finding something) -> 7+\n"
    "  * If two adjacent clips show the same activity, only one earns 7+\n"
    "  * Default to score 3 and keep=false. Be a critic, not a curator.\n\n"
    "suggested_speed (only matters if keep=true):\n"
    "  1.0 = peak moments, dialogue, action\n"
    "  1.5 = good activity with slow pace\n"
    "  2.0 = scenery without people\n"
    "  2.5-3.0 = beautiful but repetitive scenery"
)

IMAGE_PROMPT = (
    "You are an editor selecting photos for a SHORT family trip highlight.\n"
    "Most photos get cut. Be ruthless.\n\n"
    "Return ONLY valid JSON, no markdown:\n"
    '{{\n'
    '  "description": "one sentence describing the photo",\n'
    '  "has_people": true or false,\n'
    '  "quality": "good|dark|blurry|boring",\n'
    '  "highlight_score": 0,\n'
    '  "keep": true or false,\n'
    '  "suggested_speed": 1.0,\n'
    '  "notes": ""\n'
    '}}\n\n'
    "highlight_score is 0-10:\n"
    "  0-2: cut (blurry, dark, boring, accidental)\n"
    "  3-5: pleasant but generic - cut from a tight highlight\n"
    "  6-7: good photo - keep only if it adds variety\n"
    "  8-10: standout - clearly memorable, expressive, or striking\n\n"
    "Rules:\n"
    "  * keep=true ONLY if highlight_score >= 7\n"
    "  * Default to score 3 and keep=false\n"
    "  * People with clear expression/action -> 7+\n"
    "  * Truly striking scene composition -> 7+\n"
    "  * Generic landscape without people -> 3-5 -> cut"
)


def check_ollama(ollama_url: str) -> bool:
    try:
        r = requests.get(f"{ollama_url}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(QWEN_MODEL.split(":")[0] in m for m in models):
            print(f"[WARN] {QWEN_MODEL} not found. Available: {models}")
            print(f"       Run: ollama pull {QWEN_MODEL}")
            return False
        return True
    except Exception as e:
        print(f"[ERROR] Cannot reach Ollama at {ollama_url}: {e}")
        return False


def extract_frame_at(video_path: Path, timestamp: float):
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        subprocess.run(
            [FFMPEG_PATH, "-ss", str(timestamp), "-i", str(video_path),
             "-frames:v", "1", "-q:v", "6", "-vf", "scale=640:-2", "-y", tmp_path],
            capture_output=True, timeout=20, check=True,
        )
        return base64.b64encode(Path(tmp_path).read_bytes()).decode()
    except Exception:
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


def analyze_segment_ollama(seg: dict, ollama_url: str) -> dict:
    path  = Path(seg["source_path"])
    stype = seg["source_type"]

    if stype == "image":
        try:
            frames = [base64.b64encode(path.read_bytes()).decode()]
        except Exception:
            return _default_analysis()
        prompt = IMAGE_PROMPT
    else:
        frames = []
        for pos in SEGMENT_FRAME_POSITIONS:
            ts = seg["start"] + pos * seg["duration"]
            f  = extract_frame_at(path, ts)
            if f:
                frames.append(f)
        if not frames:
            return _default_analysis()
        prompt = SEGMENT_PROMPT_TPL.format(n=len(frames), duration=seg["duration"])

    payload = {
        "model":   QWEN_MODEL,
        "prompt":  prompt,
        "images":  frames,
        "stream":  False,
        "options": {"temperature": 0.1},
    }

    try:
        r = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=120)
        r.raise_for_status()
        raw = r.json().get("response", "{}")
        raw = re.sub(r"```(?:json)?|```", "", raw).strip()
        return json.loads(raw)
    except Exception as e:
        print(f" [WARN] Qwen: {e}")
        return _default_analysis()


def _default_analysis() -> dict:
    return {"description": "Analysis unavailable", "quality": "good",
            "keep": True, "suggested_speed": 1.0, "notes": ""}


def _seg_id(seg: dict) -> str:
    """Stable identifier for resume: source path + start time uniquely identifies a segment."""
    return f"{seg['source_path']}|{seg.get('start', 0):.3f}"


def run_analysis_stage(media: list, analyze_fn, cache_path: Path) -> list:
    """
    analyze_fn: callable taking a segment dict and returning an analysis dict.
                Created by main() to bind the chosen backend (ollama or transformers).

    Crash-resilient: writes cache after every segment, and on restart skips any
    segments that were already analyzed (matched by source path + start time).
    A crash partway through a long run loses at most one segment of work.
    """
    # Load any existing analysis (full or partial) for resume support
    cached_by_id = {}
    if cache_path.exists():
        try:
            with open(cache_path) as f:
                existing = json.load(f)
            cached_by_id = {_seg_id(s): s for s in existing if "description" in s}
            if cached_by_id:
                print(f"\n  Resuming with {len(cached_by_id)} segments already analyzed.")
        except Exception:
            cached_by_id = {}

    all_segments = []
    skipped_resume = 0

    for mi, item in enumerate(media):
        path  = item["path"]
        ftype = item["type"]
        print(f"\n  [{mi+1}/{len(media)}] {path.name}")

        if ftype == "video":
            print(f"         Detecting scenes... ", end="", flush=True)
            t0 = time.time()
            segments = build_segments_for_file(item)
            print(f"{len(segments)} segment(s)  ({time.time()-t0:.1f}s)")
        else:
            segments = build_segments_for_file(item)

        for si, seg in enumerate(segments):
            seg_id = _seg_id(seg)

            # Resume: reuse cached analysis for this segment if present
            if seg_id in cached_by_id:
                cached = cached_by_id[seg_id]
                # Preserve any keys we didn't have before (e.g. has_people)
                for k in ("description", "quality", "keep", "suggested_speed",
                          "notes", "has_people"):
                    if k in cached:
                        seg[k] = cached[k]
                all_segments.append(seg)
                skipped_resume += 1
                continue

            if ftype == "video":
                label = f"seg {si+1}/{len(segments)} [{seg['start']:.1f}s-{seg['end']:.1f}s dur={seg['duration']:.1f}s]"
            else:
                label = "photo"
            print(f"         Analyzing {label}... ", end="", flush=True)
            t0 = time.time()
            analysis = analyze_fn(seg)
            seg.update(analysis)
            status = "OK" if seg["keep"] else "SKIP"
            print(f"{status}  ({time.time()-t0:.1f}s)  {seg['description'][:55]}")
            all_segments.append(seg)

            # Checkpoint: write cache after every successful segment so a crash
            # doesn't lose progress. Use atomic write (temp file + rename) so
            # Ctrl+C during the write can never corrupt the cache.
            try:
                tmp_path = cache_path.with_suffix(cache_path.suffix + ".tmp")
                with open(tmp_path, "w") as f:
                    json.dump(all_segments, f, indent=2)
                os.replace(tmp_path, cache_path)   # atomic on Windows + POSIX
            except Exception as e:
                print(f"         [WARN] Could not write cache: {e}")

    if skipped_resume:
        print(f"\n  Reused {skipped_resume} cached segments from previous run.")
    print(f"\n  Analysis cached -> {cache_path.name}")
    return all_segments


def build_skip_analysis(media: list) -> list:
    segments = []
    for item in media:
        path  = item["path"]
        ftype = item["type"]
        if ftype == "image":
            seg = {"source_path": str(path), "source_type": "image",
                   "seg_index": 0, "start": 0.0,
                   "end": IMAGE_HOLD_DURATION, "duration": IMAGE_HOLD_DURATION}
        else:
            dur = get_video_duration(path)
            seg = {"source_path": str(path), "source_type": "video",
                   "seg_index": 0, "start": 0.0,
                   "end": round(dur, 3), "duration": round(dur, 3)}
        seg.update(_default_analysis())
        segments.append(seg)
    return segments


# -----------------------------------------------------------------
# 4.  BUILD FFMPEG COMMAND
# -----------------------------------------------------------------

def get_video_resolution(path: Path):
    try:
        r = subprocess.run(
            [FFPROBE_PATH, "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height", "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=15,
        )
        w, h = r.stdout.strip().split(",")
        return int(w), int(h)
    except Exception:
        return (1920, 1080)


def _shorten_paths_via_hardlinks(segments: list, output_path: Path) -> dict:
    """
    Shorten input paths for the ffmpeg command. Three strategies in order:

    1) Common-prefix relative paths (best, free, no system changes)
       If all sources share a common parent directory, change the working dir
       to that parent and use relative paths in the ffmpeg command.
       This costs zero disk space and needs no admin/developer mode.

    2) Hardlinks into _ffmpeg_work/  (requires same NTFS volume + admin OR dev mode)
       Tried as a fallback if relative-path approach can't shorten enough.

    3) Copies as last resort  (uses real disk space!)

    Returns a dict mapping original_path -> shortened_path. Also returns,
    via the global state, an optional cwd to chdir into before running ffmpeg.
    """
    global _RENDER_CWD
    _RENDER_CWD = None

    unique_paths = []
    for s in segments:
        if s.get("_kept_final", s.get("keep", True)):
            if s["source_path"] not in unique_paths:
                unique_paths.append(s["source_path"])

    if not unique_paths:
        return {}

    # ── Strategy 1: common-prefix relative paths ──────────────────────────
    try:
        common = os.path.commonpath(unique_paths)
        if common and Path(common).is_dir():
            mapping = {}
            for orig in unique_paths:
                rel = os.path.relpath(orig, common)
                mapping[orig] = rel
            avg_len = sum(len(v) for v in mapping.values()) / len(mapping)
            print(f"  Path shortening: common prefix = {common}")
            print(f"                   avg path now {avg_len:.0f} chars (was {sum(len(p) for p in unique_paths)/len(unique_paths):.0f})")
            _RENDER_CWD = common
            return mapping
    except (ValueError, OSError):
        pass   # different drives, etc. — fall through

    # ── Strategy 2 & 3: hardlinks / copies into _ffmpeg_work/ ─────────────
    work_dir = output_path.parent / "_ffmpeg_work"
    work_dir.mkdir(exist_ok=True)

    mapping = {}
    img_idx = vid_idx = 0
    n_linked = n_copied = 0
    for orig in unique_paths:
        orig_p = Path(orig)
        if not orig_p.is_file():
            continue
        # Determine type from extension
        if orig_p.suffix.lower() in (e.lower() for e in IMAGE_EXTENSIONS):
            short = work_dir / f"i{img_idx:04d}{orig_p.suffix.lower()}"
            img_idx += 1
        else:
            short = work_dir / f"v{vid_idx:04d}{orig_p.suffix.lower()}"
            vid_idx += 1

        if short.exists():
            # If a usable link/copy already exists from a prior run, reuse it
            if short.stat().st_size == orig_p.stat().st_size:
                mapping[orig] = str(short)
                continue
            short.unlink()

        try:
            os.link(orig, short)
            n_linked += 1
        except OSError:
            try:
                import shutil
                shutil.copy2(orig, short)
                n_copied += 1
            except Exception as e:
                print(f"  [WARN] Could not link/copy {orig_p.name}: {e}")
                continue
        mapping[orig] = str(short)

    if mapping:
        sample = next(iter(mapping.values()))
        print(f"  Path shortening: {n_linked} hardlinked, {n_copied} copied into {work_dir.name}/")
        if n_copied > 0:
            print(f"                   (hardlinks need admin or Windows Developer Mode;")
            print(f"                    enable it via Settings -> For developers to avoid copies)")
    return mapping


# Global set by _shorten_paths_via_hardlinks Strategy 1 — if non-None, ffmpeg
# is invoked with cwd=this so relative paths in the command line resolve.
_RENDER_CWD = None


def build_ffmpeg_command(segments: list, output_path: Path) -> list:
    kept    = [s for s in segments if s.get("keep", True)]
    skipped = [s for s in segments if not s.get("keep", True)]

    if not kept:
        print("[ERROR] All segments skipped - nothing to render.")
        sys.exit(1)

    print(f"\n  Edit plan: {len(kept)} segments kept, {len(skipped)} skipped")
    for s in skipped:
        name = Path(s["source_path"]).name
        print(f"     SKIP  {name}  [{s['start']:.1f}s-{s['end']:.1f}s]  {s.get('description','')[:50]}")

    target_w, target_h = 1920, 1080
    for s in kept:
        if s["source_type"] == "video":
            target_w, target_h = get_video_resolution(Path(s["source_path"]))
            break

    # Shorten paths via hardlinks/copies in a flat work directory.
    # The ffmpeg command then references short names (i0000.jpg, v0001.mp4, ...)
    # instead of long nested Windows paths repeated for every input.
    short_path_map = _shorten_paths_via_hardlinks(kept, output_path)

    def _src_path(orig: str) -> str:
        return short_path_map.get(orig, orig)

    inputs       = []
    filter_parts = []
    v_labels     = []
    a_labels     = []

    # ── Deduplicate video inputs ──────────────────────────────────────
    # Each unique video file is opened ONCE as a single -i, regardless of how
    # many segments come from it. The filtergraph references that one input by
    # index for all its segments. This keeps the command line short on Windows
    # (32,767 char limit) when there are hundreds of segments.
    # Images are still added per-segment because their input flags (-loop -t)
    # are tied to that specific use.
    video_input_idx = {}     # source_path str -> input index

    def _next_input_index() -> int:
        return inputs.count("-i")

    def _get_or_add_video_input(path_str: str) -> int:
        if path_str not in video_input_idx:
            video_input_idx[path_str] = _next_input_index()
            inputs.extend(["-i", path_str])
        return video_input_idx[path_str]

    for idx, seg in enumerate(kept):
        path  = Path(seg["source_path"])
        stype = seg["source_type"]
        speed = float(seg.get("suggested_speed", 1.0))
        start = seg["start"]
        dur   = seg["duration"]

        # ── Inputs (deduped for videos, shortened paths) ────────────────
        if stype == "image":
            input_idx = _next_input_index()
            inputs += ["-loop", "1", "-t", str(IMAGE_HOLD_DURATION),
                       "-i", _src_path(str(path))]
        else:
            input_idx = _get_or_add_video_input(_src_path(str(path)))

        # ── Video + Audio filter chains ───────────────────────────────────
        #
        # Correct order for speed-adjusted segments:
        #   Video: trim (exact window) → setpts reset → setpts speed → scale → fps/settb
        #   Audio: atrim (same exact window) → asetpts reset → atempo speed → aresample
        #
        # Key rules:
        #   1. trim/atrim use the same [trim_start, trim_start+dur] window → guarantees sync
        #   2. setpts speed BEFORE fps/settb — fps must see the already-sped frames
        #   3. atrim duration = dur (raw source seconds), atempo handles compression
        #   4. aresample after atempo keeps sample rate stable (atempo can drift it)

        in_label  = f"[{input_idx}:v]"
        out_label = f"[v{idx}]"

        if stype == "video":
            # No more pre-seek: every segment trims directly from t=0 of the source,
            # since the deduped input doesn't have a -ss before -i.
            trim_start = start

            vf_filters = [
                # Step 1: cut exact window, reset PTS to 0
                f"trim=start={trim_start:.3f}:duration={dur:.3f},setpts=PTS-STARTPTS",
            ]
            if speed != 1.0:
                # Step 2: speed BEFORE fps so fps sees the final frame rate
                vf_filters.append(f"setpts={1.0/speed:.4f}*PTS")
            vf_filters.append(
                # Step 3: normalize resolution + timebase
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,fps={TARGET_FPS},settb=1/{TARGET_FPS}"
            )
        else:
            # Image: no trim needed, just scale + fade
            vf_filters = [
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,fps={TARGET_FPS},settb=1/{TARGET_FPS}",
                f"fade=t=in:st=0:d={FADE_IN_DURATION}",
            ]

        filter_parts.append(in_label + ",".join(vf_filters) + out_label)
        v_labels.append(out_label)

        # Audio — simple, proven syntax. PTS-STARTPTS resets each segment's PTS
        # to zero. acrossfade (used later) needs this to align with video xfades.
        SR = 48000
        if stype == "image":
            filter_parts.append(
                f"anullsrc=channel_layout=stereo:sample_rate={SR},"
                f"atrim=duration={IMAGE_HOLD_DURATION:.3f},asetpts=PTS-STARTPTS"
                f"[a{idx}]"
            )
        else:
            trim_start  = start  # same: no pre-seek anymore
            in_label_a  = f"[{input_idx}:a]"
            out_label_a = f"[a{idx}]"
            af_filters  = [
                f"atrim=start={trim_start:.3f}:duration={dur:.3f}",
                "asetpts=PTS-STARTPTS",
            ]
            if speed > 2.0:
                af_filters.append(f"atempo=2.0,atempo={speed/2.0:.4f}")
            elif speed < 0.5:
                af_filters.append(f"atempo=0.5,atempo={speed*2.0:.4f}")
            elif speed != 1.0:
                af_filters.append(f"atempo={speed:.4f}")
            af_filters.append(f"aresample={SR}")
            filter_parts.append(in_label_a + ",".join(af_filters) + out_label_a)

        a_labels.append(f"[a{idx}]")

    # Sequential xfade
    if len(v_labels) == 1:
        filter_parts.append(f"{v_labels[0]}copy[vout]")
    else:
        prev_v     = v_labels[0]
        cumulative = 0.0
        for i, seg in enumerate(kept[:-1]):
            speed   = float(seg.get("suggested_speed", 1.0))
            seg_dur = (IMAGE_HOLD_DURATION if seg["source_type"] == "image"
                       else seg["duration"]) / speed
            cumulative += seg_dur
            offset     = max(0.05, cumulative - CROSSFADE_DURATION)
            next_v     = v_labels[i + 1]
            out_label  = "vout" if i == len(kept) - 2 else f"xf{i}"
            filter_parts.append(
                f"{prev_v}{next_v}"
                f"xfade=transition=fade:duration={CROSSFADE_DURATION}"
                f":offset={offset:.3f}[{out_label}]"
            )
            prev_v      = f"[{out_label}]"
            cumulative -= CROSSFADE_DURATION

    # Audio: chained acrossfade — must mirror the video xfade overlap.
    # If audio just used concat, video would lose 0.8s per transition (xfade overlap)
    # while audio kept full length, causing cumulative drift (~7s for 9 transitions).
    # acrossfade overlaps audio segments by the same CROSSFADE_DURATION → A/V stay locked.
    if len(a_labels) == 1:
        filter_parts.append(f"{a_labels[0]}acopy[aout]")
    else:
        prev_a = a_labels[0]
        for i in range(len(a_labels) - 1):
            next_a    = a_labels[i + 1]
            out_label = "aout" if i == len(a_labels) - 2 else f"af{i}"
            filter_parts.append(
                f"{prev_a}{next_a}"
                f"acrossfade=d={CROSSFADE_DURATION}:c1=tri:c2=tri"
                f"[{out_label}]"
            )
            prev_a = f"[{out_label}]"

    filter_complex = ";\n".join(filter_parts)

    cmd  = [FFMPEG_PATH, "-y"]
    cmd += inputs
    cmd += ["-filter_complex", filter_complex]
    cmd += ["-map", "[vout]", "-map", "[aout]"]
    cmd += ["-c:v", "libx264", "-preset", "slow", "-crf", "18",
            "-pix_fmt", "yuv420p",
            "-color_range", "tv", "-colorspace", "bt709",
            "-color_trc", "bt709", "-color_primaries", "bt709"]
    cmd += ["-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k"]
    cmd += ["-movflags", "+faststart"]
    cmd += [str(output_path)]
    return cmd


# -----------------------------------------------------------------
# 5.  EXECUTE
# -----------------------------------------------------------------

def run_ffmpeg(cmd: list, output_path: Path):
    """
    Execute the ffmpeg command. Always externalizes filter_complex to a sidecar
    file. Prints detailed size diagnostics so we can see where any remaining
    bulk lives.
    """
    SAFE_CMD_LIMIT = 30000
    cmd_file       = output_path.parent / "ffmpeg_command.sh"

    # ── Pre-externalization diagnostics ─────────────────────────────
    pre_len  = sum(len(a) + 3 for a in cmd)
    n_inputs = cmd.count("-i")
    n_args   = len(cmd)
    longest  = max((len(a), i) for i, a in enumerate(cmd))
    print(f"\n  Cmd size BEFORE externalization: {pre_len:,} chars  ({n_args} args, {n_inputs} inputs)")
    print(f"  Longest single arg: {longest[0]} chars  (arg #{longest[1]}, value preview: "
          f"{cmd[longest[1]][:80]!r})")

    # ── ALWAYS externalize filter_complex (costs nothing, helps everything) ──
    if "-filter_complex" in cmd:
        i = cmd.index("-filter_complex")
        filter_text = cmd[i + 1]
        filter_script = output_path.parent / "filter_complex.txt"
        with open(filter_script, "w", encoding="utf-8") as f:
            f.write(filter_text)
        cmd = cmd[:i] + ["-filter_complex_script", str(filter_script)] + cmd[i + 2:]
        post_len = sum(len(a) + 3 for a in cmd)
        print(f"  filter_complex externalized: {len(filter_text):,} chars -> {filter_script.name}")
        print(f"  Cmd size AFTER  externalization: {post_len:,} chars")
    else:
        post_len = pre_len

    # ── Final size check ────────────────────────────────────────────
    if post_len > SAFE_CMD_LIMIT:
        print(f"\n  [WARN] Command is still {post_len:,} chars after externalization.")
        print( "         The remaining bulk is likely the inputs section. Args > 50 chars:")
        for i, a in enumerate(cmd):
            if len(a) > 50:
                print(f"           [{i:3}] ({len(a):3} chars) {a[:100]}")
        # Show how many inputs and an example
        if n_inputs > 0:
            i_indices = [i for i, a in enumerate(cmd) if a == "-i"]
            sample_i  = cmd[i_indices[0] + 1] if i_indices else ""
            print(f"\n         {n_inputs} -i inputs, sample: {sample_i!r}  (len {len(sample_i)})")
            print(f"         Inputs total: ~{sum(len(cmd[i+1])+3 for i in i_indices):,} chars\n")

    with open(cmd_file, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n")
        f.write(" \\\n  ".join(cmd) + "\n")
    print(f"\n  Rendering...  (command -> {cmd_file.name})\n")

    try:
        subprocess.run(cmd, check=True)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n  Done!  {output_path.name}  ({size_mb:.1f} MB)")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] FFmpeg exited {e.returncode}. Check {cmd_file.name}.")
        sys.exit(1)
    except FileNotFoundError as e:
        # The cmd[0] is the resolved FFMPEG_PATH — show what was attempted plus the system error.
        print(f"\n[ERROR] Could not launch ffmpeg.")
        print(f"        Attempted path: {cmd[0]!r}")
        print(f"        System error:   {e}")
        print(f"        Path exists on disk? {Path(cmd[0]).is_file()}")
        sys.exit(1)
    except OSError as e:
        # Catches WinError 193 (not a valid Win32 application) and similar issues
        print(f"\n[ERROR] OS rejected ffmpeg launch.")
        print(f"        Attempted path: {cmd[0]!r}")
        print(f"        WinError / errno: {e.winerror if hasattr(e, 'winerror') else e.errno}")
        print(f"        Message: {e}")
        sys.exit(1)


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GoPro family trip editor - segment-aware AI pipeline")
    p.add_argument("--input",           required=True)
    p.add_argument("--output",          required=True)
    p.add_argument("--ollama-url",      default=DEFAULT_OLLAMA_URL)
    p.add_argument("--skip-analysis",   action="store_true",
                   help="Skip Qwen - stitch everything in order")
    p.add_argument("--clear-cache",     action="store_true",
                   help="Delete existing analysis cache and re-run")
    p.add_argument("--dry-run",         action="store_true",
                   help="Build FFmpeg command but don't execute")
    p.add_argument("--scene-threshold", type=float, default=SCENE_THRESHOLD,
                   help=f"Scene-cut sensitivity 0-1 (default {SCENE_THRESHOLD}). Lower = more cuts.")
    p.add_argument("--segment-interval", type=float, default=SEGMENT_INTERVAL,
                   help=f"Split videos every N seconds for analysis (default {SEGMENT_INTERVAL}s). "
                        "Shorter = finer-grained Qwen decisions but more API calls.")
    # ── Backend selection ──────────────────────────────────────────
    p.add_argument("--backend", choices=["ollama", "transformers"], default="ollama",
                   help="Analysis backend. ollama=easy setup, frame-based. "
                        "transformers=better quality, sends actual video segments to Qwen3-VL, "
                        "needs torch + transformers + qwen-vl-utils.")
    p.add_argument("--transformers-model", default="Qwen/Qwen3-VL-8B-Instruct",
                   help="HF model id for the transformers backend (default Qwen3-VL-8B-Instruct).")
    p.add_argument("--quantize-4bit", action="store_true",
                   help="Use 4-bit NF4 quantization for transformers backend (saves VRAM).")
    p.add_argument("--max-frames", type=int, default=16,
                   help="Frames per segment for the transformers backend (default 16).")
    # ── Highlight tightening ───────────────────────────────────────
    p.add_argument("--min-score", type=float, default=7.0,
                   help="Minimum highlight_score to keep (0-10). Default 7. "
                        "Higher = tighter cut. Overrides the model's keep flag.")
    p.add_argument("--max-segments", type=int, default=None,
                   help="Cap final cut to top-N segments by highlight_score. "
                        "Original chronological order is preserved.")
    p.add_argument("--max-duration", type=float, default=None,
                   help="Cap final cut to N minutes (after speed adjustment). "
                        "Lowest-scored segments are dropped first.")
    # ── Chunked rendering ──────────────────────────────────────────
    p.add_argument("--chunk-size", type=int, default=30,
                   help="Render in chunks of N segments. Default 30 — keep low (20-40) "
                        "to avoid out-of-memory errors when source footage is 4K. "
                        "Larger chunks render faster but use much more RAM per chunk.")
    # ── Encoder selection ──────────────────────────────────────────
    p.add_argument("--encoder", choices=["libx264", "h264_amf", "hevc_amf"],
                   default="libx264",
                   help="Video encoder. libx264 = CPU, best quality. "
                        "h264_amf = AMD GPU, ~5-10x faster, slightly lower quality. "
                        "hevc_amf = AMD GPU H.265, smaller files at similar quality but "
                        "slower than h264_amf and not as universally playable.")
    # ── ffmpeg / ffprobe paths ─────────────────────────────────────
    p.add_argument("--ffmpeg",  default=None,
                   help="Path to ffmpeg binary. Falls back to FFMPEG_PATH env var, then PATH, "
                        "then common Windows locations.")
    p.add_argument("--ffprobe", default=None,
                   help="Path to ffprobe binary (same fallback chain).")
    return p.parse_args()


def filter_by_score(segments: list, min_score: float, max_segments: int = None,
                    max_duration_minutes: float = None) -> list:
    """
    Apply highlight-score-based filtering on top of the model's keep/skip.
    Sets a transient `_kept_final` flag on each segment (True = include in render).
    Does NOT mutate the `keep` field so the cache stays untouched.

    Steps in order:
      1. Start from segments where keep != False (model's decision)
      2. Drop segments whose highlight_score is below min_score
         (segments without a score get score=5 as neutral default for legacy caches)
      3. If max_segments is set, take the top-N by score (chronological order kept)
      4. If max_duration is set, drop lowest-scored segments until total fits
    """
    DEFAULT_SCORE = 5.0

    # Step 0: initialize _kept_final from the model's `keep` decision
    for s in segments:
        score = s.get("highlight_score", DEFAULT_SCORE)
        try:
            score = float(score)
        except (TypeError, ValueError):
            score = DEFAULT_SCORE
        s["_score"]       = score
        s["_kept_final"]  = bool(s.get("keep", True))   # model said keep?

    # Step 1: score threshold
    for s in segments:
        if s["_score"] < min_score:
            s["_kept_final"] = False
    kept = [s for s in segments if s["_kept_final"]]
    print(f"\n  After min_score >= {min_score}: {len(kept)} segments")

    # Step 2: cap to top N
    if max_segments and len(kept) > max_segments:
        top_n_ids = {(s["source_path"], s.get("start", 0))
                     for s in sorted(kept, key=lambda x: -x["_score"])[:max_segments]}
        for s in segments:
            if s["_kept_final"] and (s["source_path"], s.get("start", 0)) not in top_n_ids:
                s["_kept_final"] = False
        kept = [s for s in segments if s["_kept_final"]]
        print(f"  After max_segments={max_segments}: {len(kept)} segments")

    # Step 3: duration cap (drop lowest-scored first)
    if max_duration_minutes:
        cap_seconds = max_duration_minutes * 60.0

        def _eff_dur(s):
            speed = float(s.get("suggested_speed", 1.0))
            d = s.get("duration", IMAGE_HOLD_DURATION) if s["source_type"] == "video" else IMAGE_HOLD_DURATION
            return d / speed

        total = sum(_eff_dur(s) for s in kept)
        if total > cap_seconds:
            print(f"  Total duration {total/60:.1f}min exceeds cap of {max_duration_minutes:.1f}min")
            ranked = sorted(kept, key=lambda x: x["_score"])
            removed = 0.0
            for s in ranked:
                if total - removed <= cap_seconds:
                    break
                s["_kept_final"] = False
                removed += _eff_dur(s)
            kept = [s for s in segments if s["_kept_final"]]
            new_total = sum(_eff_dur(s) for s in kept)
            print(f"  After max_duration: {len(kept)} segments, {new_total/60:.1f}min")

    return segments


def _encoder_flags(encoder: str) -> list:
    """
    Return the ffmpeg flags for the chosen video encoder.

    Quality notes:
      - libx264 -preset medium -crf 20 = high quality reference, CPU-bound
      - h264_amf -quality quality -qp 22 = ~5-10x faster on AMD GPU,
        slightly lower compression efficiency (file ~20% larger at same visual quality)
      - hevc_amf for smaller files via H.265, slower than h264_amf
    """
    if encoder == "h264_amf":
        return [
            "-c:v", "h264_amf",
            "-quality", "quality",      # speed | balanced | quality
            "-usage", "high_quality",
            "-rc", "cqp",
            "-qp_i", "20", "-qp_p", "22", "-qp_b", "24",
            "-b:v", "0",                # ignored under CQP but explicit
            "-pix_fmt", "yuv420p",
            "-color_range", "tv", "-colorspace", "bt709",
            "-color_trc", "bt709", "-color_primaries", "bt709",
        ]
    if encoder == "hevc_amf":
        return [
            "-c:v", "hevc_amf",
            "-quality", "quality",
            "-usage", "high_quality",
            "-rc", "cqp",
            "-qp_i", "22", "-qp_p", "24", "-qp_b", "26",
            "-pix_fmt", "yuv420p",
            "-color_range", "tv", "-colorspace", "bt709",
            "-color_trc", "bt709", "-color_primaries", "bt709",
            "-tag:v", "hvc1",           # Quicktime/iOS compat
        ]
    # Default: libx264
    return [
        "-c:v", "libx264", "-preset", "medium", "-crf", "20",
        "-pix_fmt", "yuv420p",
        "-color_range", "tv", "-colorspace", "bt709",
        "-color_trc", "bt709", "-color_primaries", "bt709",
    ]


def build_ffmpeg_command_for_chunk(chunk_segments: list, output_path: Path,
                                   short_path_map: dict, encoder: str = "libx264") -> list:
    """
    Same as build_ffmpeg_command but accepts a pre-shortened-path map and
    operates on a subset of segments. Used by render_in_chunks.
    """
    # All segments in chunk should be keep=true at this point
    kept = chunk_segments
    if not kept:
        return None

    target_w, target_h = 1920, 1080
    for s in kept:
        if s["source_type"] == "video":
            target_w, target_h = get_video_resolution(Path(s["source_path"]))
            break

    def _src_path(orig: str) -> str:
        return short_path_map.get(orig, orig)

    inputs       = []
    filter_parts = []
    v_labels     = []
    a_labels     = []

    video_input_idx = {}

    def _next_input_index() -> int:
        return inputs.count("-i")

    def _get_or_add_video_input(path_str: str) -> int:
        if path_str not in video_input_idx:
            video_input_idx[path_str] = _next_input_index()
            inputs.extend(["-i", path_str])
        return video_input_idx[path_str]

    for idx, seg in enumerate(kept):
        path  = Path(seg["source_path"])
        stype = seg["source_type"]
        speed = float(seg.get("suggested_speed", 1.0))
        start = seg["start"]
        dur   = seg["duration"]

        if stype == "image":
            input_idx = _next_input_index()
            inputs += ["-loop", "1", "-t", str(IMAGE_HOLD_DURATION),
                       "-i", _src_path(str(path))]
        else:
            input_idx = _get_or_add_video_input(_src_path(str(path)))

        in_label  = f"[{input_idx}:v]"
        out_label = f"[v{idx}]"

        if stype == "video":
            trim_start = start
            vf_filters = [
                f"trim=start={trim_start:.3f}:duration={dur:.3f},setpts=PTS-STARTPTS",
            ]
            if speed != 1.0:
                vf_filters.append(f"setpts={1.0/speed:.4f}*PTS")
            vf_filters.append(
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,fps={TARGET_FPS},settb=1/{TARGET_FPS}"
            )
        else:
            vf_filters = [
                f"scale={target_w}:{target_h}:force_original_aspect_ratio=decrease,"
                f"pad={target_w}:{target_h}:(ow-iw)/2:(oh-ih)/2,"
                f"setsar=1,fps={TARGET_FPS},settb=1/{TARGET_FPS}",
                f"fade=t=in:st=0:d={FADE_IN_DURATION}",
            ]
        filter_parts.append(in_label + ",".join(vf_filters) + out_label)
        v_labels.append(out_label)

        SR = 48000
        if stype == "image":
            filter_parts.append(
                f"anullsrc=channel_layout=stereo:sample_rate={SR},"
                f"atrim=duration={IMAGE_HOLD_DURATION:.3f},asetpts=PTS-STARTPTS"
                f"[a{idx}]"
            )
        else:
            in_label_a  = f"[{input_idx}:a]"
            out_label_a = f"[a{idx}]"
            af_filters  = [
                f"atrim=start={start:.3f}:duration={dur:.3f}",
                "asetpts=PTS-STARTPTS",
            ]
            if speed > 2.0:
                af_filters.append(f"atempo=2.0,atempo={speed/2.0:.4f}")
            elif speed < 0.5:
                af_filters.append(f"atempo=0.5,atempo={speed*2.0:.4f}")
            elif speed != 1.0:
                af_filters.append(f"atempo={speed:.4f}")
            af_filters.append(f"aresample={SR}")
            filter_parts.append(in_label_a + ",".join(af_filters) + out_label_a)
        a_labels.append(f"[a{idx}]")

    # Sequential xfade for video
    if len(v_labels) == 1:
        filter_parts.append(f"{v_labels[0]}copy[vout]")
    else:
        prev_v     = v_labels[0]
        cumulative = 0.0
        for i, seg in enumerate(kept[:-1]):
            speed   = float(seg.get("suggested_speed", 1.0))
            seg_dur = (IMAGE_HOLD_DURATION if seg["source_type"] == "image"
                       else seg["duration"]) / speed
            cumulative += seg_dur
            offset     = max(0.05, cumulative - CROSSFADE_DURATION)
            next_v     = v_labels[i + 1]
            out_l      = "vout" if i == len(kept) - 2 else f"xf{i}"
            filter_parts.append(
                f"{prev_v}{next_v}xfade=transition=fade:duration={CROSSFADE_DURATION}"
                f":offset={offset:.3f}[{out_l}]"
            )
            prev_v      = f"[{out_l}]"
            cumulative -= CROSSFADE_DURATION

    # acrossfade for audio (mirrors video xfade)
    if len(a_labels) == 1:
        filter_parts.append(f"{a_labels[0]}acopy[aout]")
    else:
        prev_a = a_labels[0]
        for i in range(len(a_labels) - 1):
            next_a = a_labels[i + 1]
            out_l  = "aout" if i == len(a_labels) - 2 else f"af{i}"
            filter_parts.append(
                f"{prev_a}{next_a}acrossfade=d={CROSSFADE_DURATION}:c1=tri:c2=tri[{out_l}]"
            )
            prev_a = f"[{out_l}]"

    filter_complex = ";\n".join(filter_parts)

    cmd  = [FFMPEG_PATH, "-y"]
    # Cap filter threads to reduce per-clip memory overhead. ffmpeg's default
    # spawns many parallel filter threads; with many inputs this multiplies the
    # working-set memory and triggers OOM on 4K source. 2 threads is plenty.
    cmd += ["-filter_threads", "2", "-filter_complex_threads", "2"]
    cmd += inputs
    cmd += ["-filter_complex", filter_complex]
    cmd += ["-map", "[vout]", "-map", "[aout]"]
    cmd += _encoder_flags(encoder)
    cmd += ["-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k"]
    cmd += ["-movflags", "+faststart"]
    cmd += [str(output_path)]
    return cmd


def _render_one_chunk(chunk_segments: list, chunk_out: Path, chunks_dir: Path,
                      short_path_map: dict, encoder: str, ci_label: str) -> bool:
    """
    Render a single chunk to disk. Returns True on success.
    On failure, returns False (caller decides whether to split-and-retry).
    """
    cmd = build_ffmpeg_command_for_chunk(chunk_segments, chunk_out, short_path_map,
                                          encoder=encoder)

    cmd_file = chunks_dir / f"{chunk_out.stem}_cmd.sh"
    if "-filter_complex" in cmd:
        i = cmd.index("-filter_complex")
        ftext = cmd[i + 1]
        fpath = chunks_dir / f"{chunk_out.stem}_filter.txt"
        with open(fpath, "w", encoding="utf-8") as f:
            f.write(ftext)
        cmd = cmd[:i] + ["-filter_complex_script", str(fpath)] + cmd[i + 2:]

    cmd_len = sum(len(a) + 3 for a in cmd)
    with open(cmd_file, "w", encoding="utf-8") as f:
        f.write("#!/bin/bash\n" + " \\\n  ".join(cmd) + "\n")
    print(f"        {ci_label} cmd size: {cmd_len:,} chars, {len(chunk_segments)} segs")

    try:
        run_cwd = _RENDER_CWD if _RENDER_CWD else None

        # Launch ffmpeg as a subprocess and monitor output file size.
        # If the output file size hasn't changed for HANG_TIMEOUT seconds and the
        # process is still running, assume it's hung (common with AMF on Windows)
        # and kill it. The chunk then gets subdivided automatically by the caller.
        HANG_TIMEOUT      = 180   # seconds without output progress
        HANG_CHECK_PERIOD = 15    # seconds between size checks

        proc = subprocess.Popen(cmd, cwd=run_cwd)
        last_size       = -1
        last_change_at  = time.time()
        while True:
            try:
                proc.wait(timeout=HANG_CHECK_PERIOD)
                break  # process exited
            except subprocess.TimeoutExpired:
                pass
            # Check size of (partially-written) output
            try:
                cur_size = chunk_out.stat().st_size if chunk_out.exists() else 0
            except OSError:
                cur_size = 0
            if cur_size != last_size:
                last_size      = cur_size
                last_change_at = time.time()
            elif time.time() - last_change_at > HANG_TIMEOUT:
                print(f"  [WARN] ffmpeg appears hung — no output progress for "
                      f"{HANG_TIMEOUT}s. Killing.")
                proc.kill()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    pass
                try:
                    chunk_out.unlink()
                except FileNotFoundError:
                    pass
                return False

        if proc.returncode != 0:
            print(f"  [WARN] {ci_label} failed (exit {proc.returncode}). See {cmd_file.name}.")
            try:
                chunk_out.unlink()
            except FileNotFoundError:
                pass
            return False
        return True
    except (FileNotFoundError, OSError) as e:
        print(f"  [ERROR] Could not launch ffmpeg: {e}")
        print(f"          See {cmd_file.name}")
        sys.exit(1)


def _render_with_subdivide(chunk_segments: list, base_out: Path, chunks_dir: Path,
                           short_path_map: dict, encoder: str, depth: int = 0) -> list:
    """
    Try to render chunk_segments as one file. On failure (typically OOM with many
    4K segments), split the chunk in half and recurse. Returns the list of output
    files produced (one if successful, multiple if subdivided).

    Bottoms out at single-segment chunks. If a single segment can't render, that's
    a hard failure and we abort.
    """
    if not chunk_segments:
        return []

    # Build a deterministic output name based on segment count + the first seg's
    # source path hash, so resume works even when we subdivide.
    suffix = ""
    if depth > 0:
        suffix = f"_d{depth}_{base_out.stem}_n{len(chunk_segments)}"
        out_path = chunks_dir / f"chunk{suffix}.mp4"
    else:
        out_path = base_out

    # Resume: skip already-rendered chunks
    if out_path.exists() and out_path.stat().st_size > 1024:
        print(f"  Resume: {out_path.name} already rendered ({len(chunk_segments)} segs)")
        return [out_path]

    label = f"depth={depth} {out_path.name}" if depth else out_path.name
    success = _render_one_chunk(chunk_segments, out_path, chunks_dir,
                                short_path_map, encoder, label)
    if success:
        return [out_path]

    # Failed — subdivide
    if len(chunk_segments) <= 1:
        print(f"  [ERROR] Single segment failed to render. Source may be corrupt: "
              f"{chunk_segments[0]['source_path']}")
        sys.exit(1)

    print(f"  Subdividing {len(chunk_segments)} segments into halves and retrying...")
    mid   = len(chunk_segments) // 2
    left  = _render_with_subdivide(chunk_segments[:mid],  base_out, chunks_dir,
                                    short_path_map, encoder, depth + 1)
    right = _render_with_subdivide(chunk_segments[mid:], base_out, chunks_dir,
                                    short_path_map, encoder, depth + 1)
    return left + right


def render_in_chunks(segments: list, output_path: Path, chunk_size: int,
                     encoder: str = "libx264"):
    """
    Render kept segments in batches, then concat the batch outputs losslessly.
    Survives Windows command-line limits and OOM crashes.

    On OOM, each chunk is automatically split in half and retried until it fits.
    The final concat list collects whatever pieces were actually produced.
    """
    kept = [s for s in segments if s.get("_kept_final", s.get("keep", True))]
    if not kept:
        print("[ERROR] No segments to render after filtering.")
        sys.exit(1)

    chunks_dir = output_path.parent / "_chunks"
    chunks_dir.mkdir(exist_ok=True)

    short_path_map = _shorten_paths_via_hardlinks(kept, output_path)

    n_chunks = (len(kept) + chunk_size - 1) // chunk_size
    print(f"\n  Rendering in {n_chunks} chunks of up to {chunk_size} segments each")
    print(f"  Chunk outputs in {chunks_dir.name}/  (auto-subdivide on OOM)")

    chunk_files = []
    for ci in range(n_chunks):
        chunk = kept[ci * chunk_size : (ci + 1) * chunk_size]
        base_out = chunks_dir / f"chunk_{ci:03d}.mp4"
        print(f"\n  [{ci+1}/{n_chunks}] Up to {len(chunk)} segments -> {base_out.name}")
        produced = _render_with_subdivide(chunk, base_out, chunks_dir,
                                          short_path_map, encoder)
        chunk_files.extend(produced)

    # Concat all (sub)chunks losslessly
    print(f"\n  Concatenating {len(chunk_files)} files into final output...")
    concat_list = chunks_dir / "concat_list.txt"
    with open(concat_list, "w", encoding="utf-8") as f:
        for cf in chunk_files:
            f.write(f"file '{cf.as_posix()}'\n")

    concat_cmd = [
        FFMPEG_PATH, "-y",
        "-f", "concat", "-safe", "0",
        "-i", str(concat_list),
        "-c", "copy",
        "-movflags", "+faststart",
        str(output_path),
    ]
    try:
        subprocess.run(concat_cmd, check=True)
        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n  Done!  {output_path.name}  ({size_mb:.1f} MB)")
        print(f"  Intermediate chunks remain in {chunks_dir.name}/ — delete to reclaim disk space.")
    except subprocess.CalledProcessError as e:
        print(f"\n[ERROR] Final concat failed (exit {e.returncode})")
        sys.exit(1)


def main():
    args = parse_args()

    global SCENE_THRESHOLD, SEGMENT_INTERVAL, FFMPEG_PATH, FFPROBE_PATH
    SCENE_THRESHOLD  = args.scene_threshold
    SEGMENT_INTERVAL = args.segment_interval

    # ── Resolve ffmpeg/ffprobe BEFORE any subprocess call ──────────
    # CLI flag wins, then existing constant (env-var or hardcoded), then fallback search.
    try:
        FFMPEG_PATH  = _resolve_binary("ffmpeg",  args.ffmpeg  or FFMPEG_PATH)
        FFPROBE_PATH = _resolve_binary("ffprobe", args.ffprobe or FFPROBE_PATH)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)
    print(f"  ffmpeg:  {FFMPEG_PATH}")
    print(f"  ffprobe: {FFPROBE_PATH}")

    # Also propagate to the transformers backend (it has its own ffmpeg call)
    try:
        import analyzer_transformers as _tx
        _tx.FFMPEG_PATH = FFMPEG_PATH
    except ImportError:
        pass   # transformers backend not used in this run

    input_dir   = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path  = output_path.parent / (output_path.stem + "_analysis.json")

    if args.clear_cache and cache_path.exists():
        cache_path.unlink()
        print("  Cleared analysis cache.")

    print("=" * 60)
    print("  GoPro Family Trip Editor  [segment-aware]")
    print("=" * 60)

    media = discover_media(input_dir)

    # ── Decide where segments come from ────────────────────────────
    # Priority: existing cache > skip-analysis (no AI) > run analysis with backend.
    # If cache exists, always use it (run_analysis_stage already handles resume
    # for partial caches). --skip-analysis only applies when there's no cache.
    if cache_path.exists():
        print(f"\n  Using existing analysis cache: {cache_path.name}")
        with open(cache_path) as f:
            segments = json.load(f)
    elif args.skip_analysis:
        print("\n  Skipping analysis - using full clips in order.")
        segments = build_skip_analysis(media)
    else:
        # ── Pick backend ─────────────────────────────────────────────
        if args.backend == "transformers":
            try:
                import analyzer_transformers as tx_backend
                tx_backend.FFMPEG_PATH = FFMPEG_PATH
            except ImportError as e:
                print(f"[ERROR] Could not import analyzer_transformers.py: {e}")
                sys.exit(1)
            if not tx_backend.setup(args.transformers_model, quantize_4bit=args.quantize_4bit):
                sys.exit(1)
            print(f"\n  Detecting scenes + analyzing with {args.transformers_model} "
                  f"(transformers, max_frames={args.max_frames})...")
            analyze_fn = lambda seg: tx_backend.analyze_segment(seg, max_frames=args.max_frames)
        else:
            # Default: ollama
            if not check_ollama(args.ollama_url):
                print("\n[TIP] Use --skip-analysis, switch to --backend transformers, "
                      "or pull the model first.")
                sys.exit(1)
            print(f"\n  Detecting scenes + analyzing with {QWEN_MODEL} (ollama)...")
            analyze_fn = lambda seg: analyze_segment_ollama(seg, args.ollama_url)

        segments = run_analysis_stage(media, analyze_fn, cache_path)

    kept    = [s for s in segments if s.get("keep", True)]
    skipped = [s for s in segments if not s.get("keep", True)]
    print(f"\n  {len(segments)} segments total -> {len(kept)} kept, {len(skipped)} skipped (model)")

    # Apply post-analysis filtering: score threshold, max_segments, max_duration
    segments = filter_by_score(
        segments,
        min_score=args.min_score,
        max_segments=args.max_segments,
        max_duration_minutes=args.max_duration,
    )

    final_kept = [s for s in segments if s.get("_kept_final", False)]
    if not final_kept:
        print("\n[ERROR] No segments left after filtering. Try lowering --min-score.")
        sys.exit(1)

    # Estimate final duration
    total_dur = sum(
        (IMAGE_HOLD_DURATION if s["source_type"] == "image" else s.get("duration", 0))
        / float(s.get("suggested_speed", 1.0))
        for s in final_kept
    )
    print(f"\n  Final cut: {len(final_kept)} segments, "
          f"~{total_dur/60:.1f} min total ({total_dur:.0f}s)")
    print(f"  Encoder:   {args.encoder}"
          + ("  (GPU, AMD AMF)" if args.encoder.endswith("_amf") else "  (CPU)"))

    if args.dry_run:
        print("\n  Dry run - skipping render.")
        return

    render_in_chunks(segments, output_path, chunk_size=args.chunk_size, encoder=args.encoder)


if __name__ == "__main__":
    main()
