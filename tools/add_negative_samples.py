"""
把 smokingVSnotsmoking/training_data/notsmoking 里的图片作为背景负样本
(无标注框)加入 YOLO 训练集,用来压低模型在人脸/手机/手指等场景下把非香烟
物体误判成 Cigarette 的概率。

背景:用 detect_and_box.py 在 validation_data 上测过，200 张不吸烟的图里有
11 张 (5.5%) 被误判出 Cigarette，原训练集里显然缺这类"长得像但不是香烟"的
负样本。YOLO 支持"背景图"训练——图片存在但没有对应标注框 (或标注文件为空)，
模型会学着不要在这类图上出框。

用法:
    python tools/add_negative_samples.py
"""
from pathlib import Path
import shutil

ROOT = Path(__file__).parent.parent
SRC_DIR = ROOT / "smokingVSnotsmoking" / "training_data" / "notsmoking"
DST_IMAGES = ROOT / "Smoking_person.v3i.yolo26" / "train" / "images"
DST_LABELS = ROOT / "Smoking_person.v3i.yolo26" / "train" / "labels"


def main() -> None:
    images = sorted(SRC_DIR.glob("*.jpg"))
    if not images:
        raise SystemExit(f"No images found under {SRC_DIR}")

    added, skipped = 0, 0
    for img_path in images:
        dst_img = DST_IMAGES / img_path.name
        dst_label = DST_LABELS / f"{img_path.stem}.txt"
        if dst_img.exists():
            skipped += 1
            continue
        shutil.copy2(img_path, dst_img)
        dst_label.touch()  # empty label file = background image, no objects to detect
        added += 1

    print(f"Added {added} negative background images ({skipped} already present, skipped).")
    print(f"train/images now has {len(list(DST_IMAGES.glob('*')))} files.")


if __name__ == "__main__":
    main()
