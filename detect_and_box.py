"""
用本仓库训练的 YOLO26 模型(models/*.pt)检测图片里的吸烟行为，把结果画框保存到新目录。

Cigarette 用红框标出；离某个 Cigarette 最近的 Person 框判定为"抽烟者"，用橙框标出；
其余 Person 用青色框标出。

本仓库训练出的模型本身就能同时识别 Person 和 Cigarette，不需要像原来那样另外接
人脸检测器/姿态模型去关联"这支烟是谁的"——直接用香烟框和人体框的几何距离判断即可。

用法:
    python detect_and_box.py --source test/webImage --output test/webImage_boxed
"""
import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CIGARETTE_COLOR = (0, 0, 255)   # 红色 (BGR) - 香烟
SMOKER_COLOR = (0, 165, 255)    # 橙色 - 判定为抽烟者的人
PERSON_COLOR = (255, 200, 0)    # 青色 - 其他人


def draw_box(img, xyxy, color, label):
    x1, y1, x2, y2 = (int(v) for v in xyxy)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    if label:
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(img, (x1, y1 - t_size[1] - 6), (x1 + t_size[0] + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA)


def nearest_person(cig_xyxy, person_boxes, max_norm_dist):
    """离香烟中心最近的人体框下标；按该人体框对角线归一化距离，超过阈值则视为关联不上。"""
    cx, cy = (cig_xyxy[0] + cig_xyxy[2]) / 2, (cig_xyxy[1] + cig_xyxy[3]) / 2
    best_idx, best_norm_dist = None, None
    for idx, (x1, y1, x2, y2) in enumerate(person_boxes):
        pcx, pcy = (x1 + x2) / 2, (y1 + y2) / 2
        diag = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
        dist = ((cx - pcx) ** 2 + (cy - pcy) ** 2) ** 0.5
        norm_dist = dist / max(diag, 1e-6)
        if best_norm_dist is None or norm_dist < best_norm_dist:
            best_idx, best_norm_dist = idx, norm_dist
    if best_idx is not None and best_norm_dist <= max_norm_dist:
        return best_idx
    return None


def main(opt):
    source = Path(opt.source)
    out_dir = Path(opt.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXTS)
    if not images:
        print(f"No images found in {source}")
        return

    model = YOLO(opt.weights)
    names = model.names
    cig_id = next(i for i, n in names.items() if n == "Cigarette")
    person_id = next(i for i, n in names.items() if n == "Person")

    n_with_cig = 0
    for img_path in images:
        im0 = cv2.imread(str(img_path))
        if im0 is None:
            print(f"[skip] failed to read {img_path}")
            continue

        result = model.predict(
            im0, conf=opt.conf_thres, iou=opt.iou_thres,
            classes=[cig_id, person_id], device=opt.device, verbose=False,
        )[0]

        cig_boxes, person_boxes = [], []
        for box in result.boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0])
            (cig_boxes if int(box.cls[0]) == cig_id else person_boxes).append((xyxy, conf))

        smoker_idx = {
            idx
            for xyxy, _ in cig_boxes
            for idx in [nearest_person(xyxy, [b for b, _ in person_boxes], opt.smoker_max_dist)]
            if idx is not None
        }

        for i, (xyxy, conf) in enumerate(person_boxes):
            color, label = (SMOKER_COLOR, f"Smoker {conf:.2f}") if i in smoker_idx else (PERSON_COLOR, f"Person {conf:.2f}")
            draw_box(im0, xyxy, color, label)
        for xyxy, conf in cig_boxes:
            draw_box(im0, xyxy, CIGARETTE_COLOR, f"Cigarette {conf:.2f}")

        if cig_boxes:
            n_with_cig += 1

        save_name = img_path.name if img_path.suffix.lower() == ".jpg" else f"{img_path.name}.jpg"
        save_path = out_dir / save_name
        cv2.imwrite(str(save_path), im0)
        print(f"{img_path.name}: {len(cig_boxes)} cigarette(s), {len(smoker_idx)} smoker(s) -> {save_path.name}")

    print(f"\nDone. {n_with_cig}/{len(images)} images had >=1 Cigarette detection. Annotated images saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=str, required=True, help="输入图片目录")
    parser.add_argument("--output", type=str, required=True, help="输出图片目录")
    parser.add_argument("--weights", type=str, default="models/smoking_person_yolo26n.pt", help="模型权重路径")
    parser.add_argument("--conf-thres", type=float, default=0.25, help="置信度阈值")
    parser.add_argument("--iou-thres", type=float, default=0.45, help="NMS IOU 阈值")
    parser.add_argument("--device", type=str, default="0", help="cuda device, 例如 0 或 cpu")
    parser.add_argument(
        "--smoker-max-dist", type=float, default=2.0,
        help="香烟中心到人体框中心的最大归一化距离(按人体框对角线归一化)，超过则不关联为该人抽烟",
    )
    opt = parser.parse_args()
    main(opt)
