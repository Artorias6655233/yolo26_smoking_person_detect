#!/usr/bin/env python3
"""Run the LoRA-fine-tuned Qwen3-VL-2B-Instruct adapter (trained by
tools/train_qwen3vl_lora.py) directly (no vLLM/llama.cpp server) over every
image in test/webImage/, using the exact same prompt/JSON-schema/drawing
convention as the live serving pipeline (vllm/scripts/judge_dataset.py), and
save annotated copies + a combined results.json into
test/qwen3-VL-2b-LoRA_test/.

Deliberately duplicates PROMPT_TMPL and the event-cleaning/drawing helpers
from judge_dataset.py byte-for-byte (same convention this repo already uses
between convert_yolo_to_qwen3vl_lora.py and judge_dataset.py) rather than
importing across the vllm/ boundary, so this script has no dependency on the
vllm/ tree and can't accidentally diverge from what training taught the model
to produce.

Usage:
  python tools/infer_qwen3vl_lora_webtest.py
  python tools/infer_qwen3vl_lora_webtest.py --adapter-dir runs/qwen3vl2b_lora/checkpoint-last
"""
import argparse
import json
import os
import subprocess
import time

# Same fragmentation fix as tools/train_qwen3vl_lora.py: images here vary
# just as wildly in resolution, and must be set before torch's CUDA
# allocator initializes.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image, ImageDraw, ImageFont, ImageFile
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration
from peft import PeftModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_ADAPTER_DIR = os.path.join(REPO_ROOT, "runs", "qwen3vl2b_lora", "adapter")
DEFAULT_IMAGES_DIR = os.path.join(REPO_ROOT, "test", "webImage")
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "test", "qwen3-VL-2b-LoRA_test")
FONT_CANDIDATES = [
    os.path.join(REPO_ROOT, "vllm", "observer", "fonts", "NotoSansCJK-Bold.ttc"),
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "C:\\Windows\\Fonts\\msyh.ttc",
    "C:\\Windows\\Fonts\\simhei.ttf",
]

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 384 * 28 * 28
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
COORD_SCALE = 1000
TARGET = "是否有人吸烟"

# --- byte-for-byte copies from vllm/scripts/judge_dataset.py (see module
# docstring for why this is a deliberate duplication, not drift) ---
PROMPT_TMPL = (
    "仔细观察图片，判断：{target}？只输出一个 JSON 对象，不要 markdown 代码块、不要解释、"
    '不要照抄格式说明：{{"found": true或false, '
    '"conclusion": "针对这个问题的直接结论，只讲与判断直接相关的证据，20字以内", '
    '"events": [{{"type": "人", "bbox_norm": [x1, y1, x2, y2]}}, '
    '{{"type": "支撑该判断的具体物品或动作，例如烟/手机/水杯", "bbox_norm": [x1, y1, x2, y2]}}]}} '
    "坐标是 0-1000 的归一化数值。画面里每有一个人，就给一条 type 为“人”、框住整个人的"
    "event；如果该判断的依据是这个人手持/靠近某个具体物品或做出某个具体动作（比如叼着烟、"
    "拿着手机），额外给一条只框住该物品/动作本身（不是整个人）的 event，type 用那个物品或"
    "动作的名字。没有相关目标时 found 为 false、events 为空数组。"
)
PERSON_COLOR = (255, 0, 0)
EVIDENCE_PALETTE = [(255, 215, 0), (255, 140, 0), (0, 120, 255), (170, 0, 255), (0, 160, 60)]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_free_mib():
    """Real system-wide free VRAM -- see train_qwen3vl_lora.py's identical
    helper for why this (not torch's own stats) is what we check."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=10,
        )
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return None


MIN_FREE_MIB = 700


def load_font(size):
    for p in FONT_CANDIDATES:
        try:
            if os.path.exists(p):
                return ImageFont.truetype(p, size)
        except OSError:
            continue
    return ImageFont.load_default()


def split_bboxes(b):
    if not isinstance(b, list) or not b or len(b) % 4 != 0:
        return []
    return [b[i:i + 4] for i in range(0, len(b), 4)]


def is_garbage_box(b):
    try:
        x1, y1, x2, y2 = [float(v) for v in b]
    except (TypeError, ValueError):
        return True
    x1 = max(0.0, min(x1, COORD_SCALE)); y1 = max(0.0, min(y1, COORD_SCALE))
    x2 = max(0.0, min(x2, COORD_SCALE)); y2 = max(0.0, min(y2, COORD_SCALE))
    if x2 <= x1 or y2 <= y1:
        return True
    area = (x2 - x1) * (y2 - y1)
    return area > 0.90 * COORD_SCALE * COORD_SCALE or area < 50


def is_placeholder_type(label: str) -> bool:
    return len(label) > 6
# --- end judge_dataset.py duplication ---


def clean_events(raw_events):
    """Same event-cleaning rule as judge_dataset.py's judge_image(): keep
    person boxes (minus garbage), keep evidence boxes that aren't a
    placeholder-text type and aren't just a duplicate of a person box, and
    require at least one surviving evidence box for `found` to be True."""
    person_boxes = []
    for e in raw_events:
        if str(e.get("type")) == "人":
            person_boxes.extend(b for b in split_bboxes(e.get("bbox_norm")) if not is_garbage_box(b))

    events = []
    has_evidence = False
    for e in raw_events:
        label = str(e.get("type", "?"))
        boxes = [b for b in split_bboxes(e.get("bbox_norm")) if not is_garbage_box(b)]
        if label != "人":
            if is_placeholder_type(label):
                continue
            boxes = [b for b in boxes if b not in person_boxes]
            if boxes:
                has_evidence = True
        if boxes:
            e = dict(e)
            e["_boxes"] = boxes
            events.append(e)
    return events, has_evidence


def draw_events(image: Image.Image, events, out_path: str):
    img = image.convert("RGB").copy()
    ow, oh = img.size
    draw = ImageDraw.Draw(img)
    lw = max(3, ow // 250)
    font = load_font(max(20, ow // 24))
    evidence_i = 0
    for ev in events:
        label = str(ev.get("type", "?"))
        if label == "人":
            color = PERSON_COLOR
        else:
            color = EVIDENCE_PALETTE[evidence_i % len(EVIDENCE_PALETTE)]
            evidence_i += 1
        for b in ev.get("_boxes", []):
            x1 = int(b[0] / COORD_SCALE * ow); y1 = int(b[1] / COORD_SCALE * oh)
            x2 = int(b[2] / COORD_SCALE * ow); y2 = int(b[3] / COORD_SCALE * oh)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=lw)
            try:
                fs = font.size
            except AttributeError:
                fs = 16
            ly = max(0, y1 - int(fs * 1.35))
            tb = draw.textbbox((x1, ly), label, font=font)
            draw.rectangle([tb[0] - 5, tb[1] - 3, tb[2] + 5, tb[3] + 3], fill=color)
            draw.text((x1, ly), label, fill=(255, 255, 255), font=font)
    save_image(img, out_path)


def save_image(img: Image.Image, out_path: str):
    ext = os.path.splitext(out_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        img.convert("RGB").save(out_path, "JPEG", quality=92)
    elif ext == ".png":
        img.save(out_path, "PNG")
    elif ext == ".webp":
        img.convert("RGB").save(out_path, "WEBP", quality=92)
    else:
        img.convert("RGB").save(out_path, "JPEG", quality=92)


def build_model(adapter_dir, load_in_4bit=True):
    proc_source = adapter_dir if os.path.exists(os.path.join(adapter_dir, "preprocessor_config.json")) else MODEL_ID
    log(f"loading processor from {proc_source} ...")
    processor = AutoProcessor.from_pretrained(proc_source, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    kwargs = dict(dtype=torch.bfloat16, device_map={"": 0})
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    log(f"loading base model {MODEL_ID} (4bit={load_in_4bit}) ...")
    base = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    log(f"attaching LoRA adapter from {adapter_dir} ...")
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    model.config.use_cache = True
    return processor, model


@torch.no_grad()
def run_one(processor, model, image_path, target=TARGET, max_new_tokens=256):
    image = Image.open(image_path).convert("RGB")
    conv = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT_TMPL.format(target=target)},
    ]}]
    inputs = processor.apply_chat_template(
        conv, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )
    device = next(model.parameters()).device
    inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in inputs.items()}

    t0 = time.time()
    gen = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    latency = time.time() - t0
    gen_only = gen[0][inputs["input_ids"].shape[1]:]
    text = processor.tokenizer.decode(gen_only, skip_special_tokens=True)

    try:
        m_start = text.index("{")
        m_end = text.rindex("}") + 1
        parsed = json.loads(text[m_start:m_end])
        parse_error = None
    except Exception as e:
        parsed = {"found": False, "conclusion": text.strip()[:200], "events": []}
        parse_error = str(e)

    raw_events = parsed.get("events") or []
    events, has_evidence = clean_events(raw_events)
    found = bool(parsed.get("found")) and has_evidence
    clean = [{"type": e.get("type"), "bbox_norm": e["_boxes"][0]} for e in events]

    return {
        "found": found,
        "raw_found": parsed.get("found"),
        "conclusion": parsed.get("conclusion"),
        "events": clean,
        "_draw_events": events,
        "_image": image,
        "latency_s": round(latency, 2),
        "raw_text": text,
        "parse_error": parse_error,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-dir", default=DEFAULT_ADAPTER_DIR)
    ap.add_argument("--images-dir", default=DEFAULT_IMAGES_DIR)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    processor, model = build_model(args.adapter_dir, load_in_4bit=not args.no_4bit)

    files = sorted(f for f in os.listdir(args.images_dir) if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    log(f"{len(files)} images in {args.images_dir}")

    records = []
    n_found = 0
    n_err = 0
    for i, fname in enumerate(files, 1):
        src_path = os.path.join(args.images_dir, fname)
        prefix = f"[{i}/{len(files)}] {fname}"
        try:
            result = run_one(processor, model, src_path, target=args.target, max_new_tokens=args.max_new_tokens)
        except Exception as e:
            log(f"{prefix}: ERROR {e}")
            records.append({"file": fname, "found": None, "error": str(e)})
            n_err += 1
            continue

        out_path = os.path.join(args.out_dir, fname)
        if result["_draw_events"]:
            draw_events(result["_image"], result["_draw_events"], out_path)
        else:
            save_image(result["_image"], out_path)

        if result["found"]:
            n_found += 1
        torch.cuda.empty_cache()
        free_mib = gpu_free_mib()
        if free_mib is not None and free_mib < MIN_FREE_MIB:
            log(f"[SAFETY] free VRAM {free_mib}MiB < {MIN_FREE_MIB}MiB threshold -- stopping early "
                f"to protect other GPU workloads (processed {i}/{len(files)}).")
            break
        log(f"{prefix}: found={result['found']} conclusion={result['conclusion']!r} "
            f"latency={result['latency_s']}s events={len(result['events'])} gpu_free={free_mib}MiB")
        records.append({
            "file": fname,
            "found": result["found"],
            "raw_found": result["raw_found"],
            "conclusion": result["conclusion"],
            "events": result["events"],
            "latency_s": result["latency_s"],
            "parse_error": result["parse_error"],
        })

    out_json = os.path.join(args.out_dir, "results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({
            "target": args.target,
            "adapter_dir": args.adapter_dir,
            "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": records,
        }, f, ensure_ascii=False, indent=2)

    n_false = len(records) - n_found - n_err
    log(f"done: {len(records)} images, found=true: {n_found}, found=false: {n_false}, errors: {n_err} -> {out_json}")


if __name__ == "__main__":
    main()
