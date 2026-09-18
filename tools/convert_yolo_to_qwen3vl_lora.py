#!/usr/bin/env python3
"""Convert Smoking_person.v3i.yolo26 (YOLO26 detection labels) into a LoRA
SFT dataset for vllm/models/qwen3-vl-2b-gguf, WITHOUT touching the source
dataset.

Output goes to Smoking_person.v3i.qwen3vl-lora/{train,valid,test}.jsonl, one
LLaMA-Factory-style ("messages" + "images") record per image. Images are not
copied: each record points back at the original file via a relative path, so
this only adds a few MB of text next to the (gitignored) image dataset.

The prompt/JSON-answer schema is copied verbatim from
vllm/scripts/judge_dataset.py's PROMPT_TMPL, so the LoRA is trained to
produce exactly what the live vLLM/llama.cpp observer pipeline already
expects (found/conclusion/events, bbox_norm in 0-1000) -- no prompt or
parser changes needed to swap in the fine-tuned adapter.

Usage:
  python3 tools/convert_yolo_to_qwen3vl_lora.py
"""
import argparse
import json
import os

import yaml

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC_DIR = os.path.join(REPO_ROOT, "Smoking_person.v3i.yolo26")
DST_DIR = os.path.join(REPO_ROOT, "Smoking_person.v3i.qwen3vl-lora")
SPLITS = ["train", "valid", "test"]
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
TARGET = "是否有人吸烟"

# Extra pure-negative images (person present, not smoking) sourced from the
# Mendeley "Smoking vs Non-Smoking" dataset (CC BY 4.0, same license as the
# YOLO dataset) -- see Smoking_person.v3i.qwen3vl-lora/README.md. Only the
# "notsmoking" folders are used: they have no bbox annotations, which is fine
# because found=false records carry no events anyway. The "smoking" folders
# are classification-only (no bbox) so they're skipped -- a found=true record
# needs a real evidence box per the live prompt contract, and we already have
# plenty of bbox-annotated positives from the YOLO dataset.
NEG_DIR = os.path.join(REPO_ROOT, "smokingVSnotsmoking")
NEG_SOURCES = {
    "train": os.path.join(NEG_DIR, "training_data", "notsmoking"),
    "valid": os.path.join(NEG_DIR, "validation_data", "notsmoking"),
}

# Kept byte-for-byte identical to vllm/scripts/judge_dataset.py's PROMPT_TMPL
# (calibrated 2026-09-15/17 against the live observer) -- this dataset exists
# to teach the model to answer this exact question in this exact format, not
# a paraphrase of it.
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

# YOLO class name -> the label the live prompt/observer uses for that thing.
# "Person" is the fixed "人" the prompt hardcodes; the rest are the "支撑该
# 判断的具体物品或动作" the prompt asks for by name.
CLASS_TO_TYPE = {
    "Person": "人",
    "Cigarette": "香烟",
    "Smoke": "烟雾",
    "Vape": "电子烟",
}

# Short human-readable phrase per evidence type, used to compose the <20-char
# "conclusion" field (mirrors the style of judge_dataset.py's own results).
EVIDENCE_PHRASE = {
    "香烟": "叼着香烟",
    "烟雾": "有烟雾",
    "电子烟": "使用电子烟",
}

NEGATIVE_CONCLUSION = "未见吸烟迹象"


def load_class_names(data_yaml_path):
    with open(data_yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    names = data["names"]
    if isinstance(names, dict):
        return {int(k): v for k, v in names.items()}
    return dict(enumerate(names))


def yolo_box_to_norm1000(cx, cy, w, h):
    x1 = (cx - w / 2) * 1000
    y1 = (cy - h / 2) * 1000
    x2 = (cx + w / 2) * 1000
    y2 = (cy + h / 2) * 1000
    x1 = max(0.0, min(x1, 1000.0))
    y1 = max(0.0, min(y1, 1000.0))
    x2 = max(0.0, min(x2, 1000.0))
    y2 = max(0.0, min(y2, 1000.0))
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def parse_label_file(path, class_names):
    person_boxes = []
    evidence_events = []  # [(type, bbox_norm)]
    if not os.path.exists(path):
        return person_boxes, evidence_events
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            cls_id = int(parts[0])
            cx, cy, w, h = (float(v) for v in parts[1:5])
            cls_name = class_names.get(cls_id)
            box = yolo_box_to_norm1000(cx, cy, w, h)
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            type_name = CLASS_TO_TYPE.get(cls_name)
            if type_name is None:
                continue
            if type_name == "人":
                person_boxes.append(box)
            else:
                evidence_events.append((type_name, box))
    return person_boxes, evidence_events


def build_answer(person_boxes, evidence_events):
    if not evidence_events:
        return {"found": False, "conclusion": NEGATIVE_CONCLUSION, "events": []}

    seen_types = []
    for type_name, _ in evidence_events:
        if type_name not in seen_types:
            seen_types.append(type_name)
    phrases = [EVIDENCE_PHRASE.get(t, t) for t in seen_types]
    conclusion = ("检测到" + "、".join(phrases))[:20]

    events = [{"type": "人", "bbox_norm": box} for box in person_boxes]
    events += [{"type": t, "bbox_norm": box} for t, box in evidence_events]
    return {"found": True, "conclusion": conclusion, "events": events}


def convert_split(split, class_names, src_dir, dst_dir):
    images_dir = os.path.join(src_dir, split, "images")
    labels_dir = os.path.join(src_dir, split, "labels")
    if not os.path.isdir(images_dir):
        print(f"[skip] {images_dir} not found")
        return 0

    records = []
    for fname in sorted(os.listdir(images_dir)):
        ext = os.path.splitext(fname)[1].lower()
        if ext not in IMAGE_EXTS:
            continue
        stem = os.path.splitext(fname)[0]
        label_path = os.path.join(labels_dir, stem + ".txt")
        person_boxes, evidence_events = parse_label_file(label_path, class_names)
        answer = build_answer(person_boxes, evidence_events)

        image_rel = os.path.relpath(os.path.join(images_dir, fname), dst_dir).replace(os.sep, "/")
        user_content = "<image>\n" + PROMPT_TMPL.format(target=TARGET)
        record = {
            "messages": [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": json.dumps(answer, ensure_ascii=False)},
            ],
            "images": [image_rel],
        }
        records.append(record)

    n_yolo = len(records)
    neg_dir = NEG_SOURCES.get(split)
    n_neg = 0
    if neg_dir and os.path.isdir(neg_dir):
        neg_answer = {"found": False, "conclusion": NEGATIVE_CONCLUSION, "events": []}
        user_content = "<image>\n" + PROMPT_TMPL.format(target=TARGET)
        for fname in sorted(os.listdir(neg_dir)):
            if os.path.splitext(fname)[1].lower() not in IMAGE_EXTS:
                continue
            image_rel = os.path.relpath(os.path.join(neg_dir, fname), dst_dir).replace(os.sep, "/")
            records.append({
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": json.dumps(neg_answer, ensure_ascii=False)},
                ],
                "images": [image_rel],
            })
            n_neg += 1
    elif neg_dir:
        print(f"[warn] negative source {neg_dir} not found, skipping")

    os.makedirs(dst_dir, exist_ok=True)
    out_path = os.path.join(dst_dir, f"{split}.jsonl")
    with open(out_path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    n_pos = sum(1 for r in records if '"found": true' in r["messages"][1]["content"])
    print(f"[{split}] {len(records)} samples ({n_pos} positive, {len(records) - n_pos} negative; "
          f"{n_yolo} from YOLO + {n_neg} extra negatives) -> {out_path}")
    return len(records)


def write_dataset_info(dst_dir):
    """LLaMA-Factory registry snippet -- merge this into LLaMA-Factory's own
    data/dataset_info.json (or point --dataset_dir at this folder and copy
    this file in as-is if nothing else lives there)."""
    info = {}
    for split in SPLITS:
        jsonl_path = os.path.join(dst_dir, f"{split}.jsonl")
        if not os.path.exists(jsonl_path):
            continue
        info[f"smoking_person_qwen3vl_{split}"] = {
            "file_name": f"{split}.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "messages", "images": "images"},
            "tags": {
                "role_tag": "role",
                "content_tag": "content",
                "user_tag": "user",
                "assistant_tag": "assistant",
            },
        }
    out_path = os.path.join(dst_dir, "dataset_info.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(info, f, ensure_ascii=False, indent=2)
    print(f"[info] wrote {out_path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-dir", default=SRC_DIR, help="source YOLO26 dataset root (untouched)")
    ap.add_argument("--dst-dir", default=DST_DIR, help="output LoRA dataset root")
    args = ap.parse_args()

    data_yaml = os.path.join(args.src_dir, "data.yaml")
    if not os.path.exists(data_yaml):
        raise SystemExit(
            f"{data_yaml} not found -- download the dataset first (see README.md)"
        )
    class_names = load_class_names(data_yaml)
    unknown = set(class_names.values()) - set(CLASS_TO_TYPE)
    if unknown:
        print(f"[warn] class(es) with no CLASS_TO_TYPE mapping, boxes will be dropped: {unknown}")

    total = 0
    for split in SPLITS:
        total += convert_split(split, class_names, args.src_dir, args.dst_dir)
    write_dataset_info(args.dst_dir)
    print(f"[done] {total} samples total -> {args.dst_dir}")


if __name__ == "__main__":
    main()
