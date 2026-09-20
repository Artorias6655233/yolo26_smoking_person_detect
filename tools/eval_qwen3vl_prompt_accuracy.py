#!/usr/bin/env python3
"""Baseline accuracy of the CURRENT prompt (vllm/scripts/judge_dataset.py's
PROMPT_TMPL, target="是否有人吸烟") against the base Qwen3-VL-2B-Instruct
model -- BEFORE any LoRA fine-tuning. Only the binary "found" (是否有人吸烟)
decision is scored; box/grounding quality is out of scope per this run's
purpose (see reports/).

Ground truth comes from smokingVSnotsmoking/validation_data/{smoking,
notsmoking} (200 + 200, Mendeley "Smoking vs Non-Smoking", CC BY 4.0) -- the
same labeled set the YOLO side of this repo already uses for its own
false-positive testing (see detect_and_box.py / README.md), so results are
directly comparable in spirit.

Deliberately duplicates PROMPT_TMPL and the event-cleaning helpers from
judge_dataset.py byte-for-byte (see tools/infer_qwen3vl_lora_webtest.py for
why this is intentional, not drift) -- this script has to ask the exact same
question the same way the live pipeline does.

Usage:
  python tools/eval_qwen3vl_prompt_accuracy.py
  python tools/eval_qwen3vl_prompt_accuracy.py --limit-per-class 50   # quick smoke test
"""
import argparse
import json
import os
import subprocess
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image, ImageFile
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_SMOKING_DIR = os.path.join(REPO_ROOT, "smokingVSnotsmoking", "validation_data", "smoking")
DEFAULT_NOTSMOKING_DIR = os.path.join(REPO_ROOT, "smokingVSnotsmoking", "validation_data", "notsmoking")
DEFAULT_OUT_JSON = os.path.join(REPO_ROOT, "reports", "qwen3vl_baseline_accuracy_results.json")

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 384 * 28 * 28
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
COORD_SCALE = 1000
TARGET = "是否有人吸烟"

# --- byte-for-byte copies from vllm/scripts/judge_dataset.py ---
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


def has_real_evidence(raw_events):
    person_boxes = []
    for e in raw_events:
        if str(e.get("type")) == "人":
            person_boxes.extend(b for b in split_bboxes(e.get("bbox_norm")) if not is_garbage_box(b))
    for e in raw_events:
        label = str(e.get("type", "?"))
        if label == "人" or is_placeholder_type(label):
            continue
        boxes = [b for b in split_bboxes(e.get("bbox_norm")) if not is_garbage_box(b)]
        boxes = [b for b in boxes if b not in person_boxes]
        if boxes:
            return True
    return False
# --- end judge_dataset.py duplication ---


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_free_mib():
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"], timeout=10)
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return None


def build_model(load_in_4bit=True):
    log(f"loading processor from {MODEL_ID} ...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)
    kwargs = dict(dtype=torch.bfloat16, device_map={"": 0})
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
    log(f"loading base model {MODEL_ID} (4bit={load_in_4bit}) ...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    model.config.use_cache = True
    return processor, model


@torch.no_grad()
def run_one(processor, model, image_path, target, max_new_tokens=256):
    image = Image.open(image_path).convert("RGB")
    conv = [{"role": "user", "content": [
        {"type": "image", "image": image},
        {"type": "text", "text": PROMPT_TMPL.format(target=target)},
    ]}]
    inputs = processor.apply_chat_template(
        conv, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt")
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
    found = bool(parsed.get("found")) and has_real_evidence(raw_events)
    return {
        "found": found,
        "raw_found": parsed.get("found"),
        "conclusion": parsed.get("conclusion"),
        "latency_s": round(latency, 2),
        "parse_error": parse_error,
    }


def list_images(d, limit=None):
    files = sorted(f for f in os.listdir(d) if os.path.splitext(f)[1].lower() in IMAGE_EXTS)
    return files[:limit] if limit else files


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--smoking-dir", default=DEFAULT_SMOKING_DIR)
    ap.add_argument("--notsmoking-dir", default=DEFAULT_NOTSMOKING_DIR)
    ap.add_argument("--out-json", default=DEFAULT_OUT_JSON)
    ap.add_argument("--target", default=TARGET)
    ap.add_argument("--no-4bit", action="store_true")
    ap.add_argument("--limit-per-class", type=int, default=None)
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    processor, model = build_model(load_in_4bit=not args.no_4bit)

    groups = [("smoking", args.smoking_dir, True), ("notsmoking", args.notsmoking_dir, False)]
    records = []
    tp = fp = tn = fn = n_err = 0
    for group_name, d, gt in groups:
        files = list_images(d, args.limit_per_class)
        log(f"{group_name}: {len(files)} images from {d}")
        for i, fname in enumerate(files, 1):
            src_path = os.path.join(d, fname)
            prefix = f"[{group_name} {i}/{len(files)}] {fname}"
            try:
                result = run_one(processor, model, src_path, target=args.target)
            except Exception as e:
                log(f"{prefix}: ERROR {e}")
                records.append({"group": group_name, "file": fname, "gt": gt, "pred": None, "error": str(e)})
                n_err += 1
                torch.cuda.empty_cache()
                continue

            pred = result["found"]
            if gt and pred:
                tp += 1
            elif gt and not pred:
                fn += 1
            elif not gt and pred:
                fp += 1
            else:
                tn += 1

            log(f"{prefix}: gt={gt} pred={pred} conclusion={result['conclusion']!r} latency={result['latency_s']}s")
            records.append({
                "group": group_name, "file": fname, "gt": gt, "pred": pred,
                "raw_found": result["raw_found"], "conclusion": result["conclusion"],
                "latency_s": result["latency_s"], "parse_error": result["parse_error"],
            })
            torch.cuda.empty_cache()
            free_mib = gpu_free_mib()
            if free_mib is not None and free_mib < 700:
                log(f"[SAFETY] free VRAM {free_mib}MiB < 700MiB -- stopping early "
                    f"(processed {i}/{len(files)} of {group_name}).")
                break

    n = tp + fp + tn + fn
    accuracy = (tp + tn) / n if n else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    recall = tp / (tp + fn) if (tp + fn) else float("nan")
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else float("nan")

    summary = {
        "target": args.target, "model": MODEL_ID, "quantization": "4bit-nf4" if not args.no_4bit else "bf16",
        "n_total": n, "n_errors": n_err,
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": records}, f, ensure_ascii=False, indent=2)

    log(f"done: n={n} acc={accuracy:.3f} precision={precision:.3f} recall={recall:.3f} f1={f1:.3f} "
        f"tp={tp} fp={fp} tn={tn} fn={fn} errors={n_err} -> {args.out_json}")


if __name__ == "__main__":
    main()
