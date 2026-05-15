#!/usr/bin/env python3
"""
analyzer_transformers.py
========================
Qwen3-VL backend using Hugging Face Transformers — sends actual video segments
(not just sampled frames) for true temporal understanding of motion and shakiness.

Trade-off vs ollama backend:
  + Better quality — model sees motion, not snapshots
  + Single call per segment instead of frame-extraction overhead
  - Heavier setup: torch + transformers + qwen-vl-utils
  - Loads model into the Python process (uses VRAM the whole run)

Setup:
    pip install -r requirements_transformers.txt

The 8B model needs about 16GB VRAM at full precision (bf16/fp16).
With 20GB VRAM the 8B fits comfortably; pass --quantize-4bit to
fit larger models like the 32B variant.
"""

import json
import re
import subprocess
import tempfile
from pathlib import Path

# Lazy globals — populated on setup() so importing this module is cheap
_model      = None
_processor  = None
_model_name = None
_torch      = None

# ffmpeg path — pipeline.py overrides this from its own resolved path before use
FFMPEG_PATH = "ffmpeg"


# ─────────────────────────────────────────────────────────────
# Prompts (same intent as ollama backend)
# ─────────────────────────────────────────────────────────────

SEGMENT_PROMPT = (
    "You are an editor cutting a SHORT family trip highlight video. "
    "Watch this segment carefully.\n\n"
    "Most footage gets cut. Only the very best ~10% earns a slot. "
    "Think of it like a 30-second trailer for the whole trip.\n\n"
    "Return ONLY valid JSON, no markdown:\n"
    '{\n'
    '  "description": "one sentence about what is happening",\n'
    '  "has_people": true or false,\n'
    '  "quality": "good|shaky|dark|blurry|boring",\n'
    '  "highlight_score": 0,\n'
    '  "keep": true or false,\n'
    '  "suggested_speed": 1.0,\n'
    '  "notes": ""\n'
    '}\n\n'
    "highlight_score is 0-10:\n"
    "  0-2: cut without hesitation (boring, shaky, dark, transit, generic scenery)\n"
    "  3-4: nothing special - usually still cut from a tight highlight\n"
    "  5-6: pleasant but generic - keep ONLY if needed for narrative\n"
    "  7-8: genuinely good moment - clear action, emotion, or rare beauty\n"
    "  9-10: must-include - peak moments, the kind you would show a friend\n\n"
    "Rules:\n"
    "  * keep=true ONLY if highlight_score >= 7\n"
    "  * Generic scenery, walking, driving, transit -> 0-3 -> cut\n"
    "  * People doing nothing in particular -> 3-5 -> cut\n"
    "  * People reacting, laughing, doing something specific -> 7+\n"
    "  * Animals, wildlife in clear shot -> 7+\n"
    "  * Truly striking scenery (rare light, unique landscape) -> 7+\n"
    "  * Notable narrative moment (arriving, finding) -> 7+\n"
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
    '{\n'
    '  "description": "one sentence describing the photo",\n'
    '  "has_people": true or false,\n'
    '  "quality": "good|dark|blurry|boring",\n'
    '  "highlight_score": 0,\n'
    '  "keep": true or false,\n'
    '  "suggested_speed": 1.0,\n'
    '  "notes": ""\n'
    '}\n\n'
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


# ─────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────

def setup(model_name: str = "Qwen/Qwen3-VL-8B-Instruct",
          quantize_4bit: bool = False) -> bool:
    """
    Load the Qwen3-VL model and processor into the current Python process.
    Returns True on success, False on failure (printing actionable errors).
    """
    global _model, _processor, _model_name, _torch

    if _model is not None and _model_name == model_name:
        return True

    # Lazy imports so users on the ollama backend never need these libs
    try:
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor
        _torch = torch
    except ImportError as e:
        print(f"[ERROR] transformers backend requires extra deps. Missing: {e.name}")
        print("        Install with:")
        print('          pip install "transformers>=4.57" "qwen-vl-utils>=0.0.14" \\')
        print('                      accelerate torch torchvision pillow')
        return False

    print(f"  Loading {model_name} via transformers (30-90s on first run)")

    # Diagnostic GPU detection — important to surface up front
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"  CUDA detected: {gpu_name}  ({total_vram:.1f} GB VRAM)")
    else:
        print("  [WARN] CUDA NOT DETECTED — model will run on CPU (extremely slow).")
        print("         Likely cause: torch was installed without CUDA support.")
        print("         To fix, reinstall PyTorch with CUDA:")
        print("           pip uninstall torch torchvision -y")
        print("           pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121")

    load_kwargs = {"dtype": "auto", "device_map": "auto"}

    if quantize_4bit:
        try:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
            print("  Using 4-bit NF4 quantization.")
        except ImportError:
            print("  [WARN] bitsandbytes not installed, loading without quantization.")
            print("         pip install bitsandbytes")

    try:
        _model = AutoModelForImageTextToText.from_pretrained(model_name, **load_kwargs)
        _model.eval()
        _processor  = AutoProcessor.from_pretrained(model_name)
        _model_name = model_name

        # Print VRAM usage after load
        if torch.cuda.is_available():
            mem_gb = torch.cuda.memory_allocated() / (1024**3)
            print(f"  Model loaded.  VRAM in use: {mem_gb:.1f} GB")
        else:
            print("  Model loaded on CPU.")
        return True

    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        print("        Make sure you have access to the model on Hugging Face.")
        return False


# ─────────────────────────────────────────────────────────────
# Per-segment analysis
# ─────────────────────────────────────────────────────────────

def _extract_segment_to_temp(source_path: Path, start: float, duration: float,
                             target_height: int = 360, target_fps: int = 4) -> Path:
    """
    Cut the requested time window into a low-resolution, low-fps temp .mp4.

    Re-encodes (not stream copy) because we MUST shrink the video — the model
    only needs ~16 sampled frames at modest resolution, but torchvision decodes
    the *entire* video into RAM before sampling. A 30s GoPro clip at 4K60 would
    decode to ~20GB of uint8 array. By pre-shrinking to 360p@4fps here we make
    torchvision's full decode fit comfortably in memory and run much faster.
    Audio is dropped — vision model never uses it.
    """
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.close()
    tmp_path = Path(tmp.name)

    subprocess.run(
        [
            FFMPEG_PATH, "-y",
            "-ss", f"{start:.3f}",
            "-i", str(source_path),
            "-t", f"{duration:.3f}",
            "-an",
            "-vf", f"scale=-2:{target_height},fps={target_fps}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            str(tmp_path),
        ],
        capture_output=True, timeout=300, check=True,
    )
    return tmp_path


def _load_image_as_pil(path: Path):
    """
    Decode JPEG/PNG/etc via Pillow, bypassing torchvision (whose ROCm wheel
    on Windows is built without libjpeg support).
    """
    from PIL import Image
    img = Image.open(path).convert("RGB")
    # Cap dimension so the image processor doesn't choke on giant 4K JPEGs
    max_dim = 1024
    if max(img.size) > max_dim:
        img.thumbnail((max_dim, max_dim))
    return img


def analyze_segment(seg: dict, max_frames: int = 16) -> dict:
    """
    Analyze one segment. Mirrors the ollama backend's interface — input is a
    segment dict (with source_path, source_type, start, duration), output is
    the analysis fields (description, quality, keep, suggested_speed, notes).
    """
    if _model is None:
        print("  [ERROR] setup() must be called before analyze_segment().")
        return _default_analysis()

    path  = Path(seg["source_path"])
    stype = seg["source_type"]
    temp_video = None

    if stype == "image":
        # Use PIL to decode (torchvision ROCm wheel lacks libjpeg)
        try:
            pil_img = _load_image_as_pil(path)
        except Exception as e:
            print(f"  [WARN] Image decode failed: {e}")
            return _default_analysis()
        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil_img},
                {"type": "text",  "text":  IMAGE_PROMPT},
            ],
        }]
    else:
        try:
            temp_video = _extract_segment_to_temp(path, seg["start"], seg["duration"])
        except Exception as e:
            print(f"  [WARN] Segment extraction failed: {e}")
            return _default_analysis()

        messages = [{
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": str(temp_video),
                    "max_frames": max_frames,
                    "max_pixels": 360 * 420,
                },
                {"type": "text", "text": SEGMENT_PROMPT},
            ],
        }]

    text = ""
    try:
        inputs = _processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        # Move tensors to model device
        inputs = {k: (v.to(_model.device) if hasattr(v, "to") else v)
                  for k, v in inputs.items()}

        with _torch.no_grad():
            output_ids = _model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )

        # Strip the prompt prefix
        prompt_len = inputs["input_ids"].shape[1]
        new_ids    = output_ids[:, prompt_len:]
        text       = _processor.batch_decode(
            new_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        # Free GPU tensors before parsing — reduces ROCm allocator pressure
        del inputs, output_ids, new_ids

        raw = re.sub(r"```(?:json)?|```", "", text).strip()
        return json.loads(raw)

    except json.JSONDecodeError:
        print(f"  [WARN] Model returned non-JSON. Raw: {text[:120]!r}")
        return _default_analysis()
    except Exception as e:
        print(f"  [WARN] Inference failed: {e}")
        return _default_analysis()
    finally:
        if temp_video is not None:
            try:
                temp_video.unlink()
            except Exception:
                pass
        # Aggressive cleanup between segments — ROCm on Windows leaks otherwise.
        # Without this, VRAM allocator fragments and eventually crashes the process.
        try:
            import gc
            gc.collect()
            if _torch is not None and _torch.cuda.is_available():
                _torch.cuda.empty_cache()
                _torch.cuda.synchronize()
        except Exception:
            pass


def _default_analysis() -> dict:
    return {
        "description":     "Analysis unavailable",
        "quality":         "good",
        "keep":            True,
        "suggested_speed": 1.0,
        "notes":           "",
    }
