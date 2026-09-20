#!/usr/bin/env python3
"""YOLO26 smoking-person detection service — HTTP image in, boxed detections out.

Same shape as vllm/observer/server.py, but local YOLO26 inference instead of a
remote VLM call, and images come from the HTTP request instead of a ROS camera
topic (no ROS dependency here).

POST /detect          JSON body, exactly one of image_base64 / image_path /
                       image_url -> runs the trained YOLO26 model, associates
                       each non-Person box (Cigarette/Smoke/Vape) with the
                       nearest Person (same rule as detect_and_box.py), and
                       answers with the boxes + a verdict.
POST /detect/upload    multipart/form-data file upload — curl -F convenience
                        wrapper around the same pipeline.
GET  /health           model/device status
GET  /classes          class id -> name map for the loaded weights
GET  /detections/<f>   fetch a saved annotated image

Env knobs:
  YOLO_PORT            default 9998
  YOLO_WEIGHTS         default models/smoking_person_yolo26n_v2.pt
  YOLO_DEVICE          default 0          (cuda device id, or "cpu")
  YOLO_CONF            default 0.25
  YOLO_IOU             default 0.45
  YOLO_SMOKER_MAX_DIST default 2.0        (see detect_and_box.py:nearest_person)
  YOLO_OUT_DIR         default runs/server_detections
  YOLO_MAX_IMAGE_MB    default 15         (reject bigger uploads/base64/downloads)

See README.md in this folder for usage examples and how to add new endpoints.
"""
import asyncio
import base64
import binascii
import os
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from ultralytics import YOLO

# detect_and_box.py lives at the repo root; reuse its box-drawing and
# person-association logic instead of re-implementing it here.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from detect_and_box import PERSON_COLOR, SMOKER_COLOR, draw_box, nearest_person  # noqa: E402

YOLO_PORT = int(os.environ.get("YOLO_PORT", "9998"))
YOLO_WEIGHTS = os.environ.get("YOLO_WEIGHTS", "models/smoking_person_yolo26n_v2.pt")
YOLO_DEVICE = os.environ.get("YOLO_DEVICE", "0")
YOLO_CONF = float(os.environ.get("YOLO_CONF", "0.25"))
YOLO_IOU = float(os.environ.get("YOLO_IOU", "0.45"))
YOLO_SMOKER_MAX_DIST = float(os.environ.get("YOLO_SMOKER_MAX_DIST", "2.0"))
YOLO_OUT_DIR = os.environ.get("YOLO_OUT_DIR", "runs/server_detections")
YOLO_MAX_IMAGE_MB = float(os.environ.get("YOLO_MAX_IMAGE_MB", "15"))

# Cigarette keeps detect_and_box.py's red; Smoke/Vape get their own colors so
# all four dataset classes (see data.yaml) are visually distinguishable.
EVIDENCE_COLORS = {
    "Cigarette": (0, 0, 255),
    "Smoke": (160, 160, 160),
    "Vape": (255, 0, 255),
}
DEFAULT_EVIDENCE_COLOR = (0, 255, 255)


# ---------------------------------------------------------------- model
app = FastAPI(title="yolo26-smoking-server")
model: Optional[YOLO] = None
ready = threading.Event()
infer_lock = threading.Lock()  # ultralytics predict() isn't safe for concurrent calls


def _load_model():
    global model
    model = YOLO(YOLO_WEIGHTS)
    ready.set()


@app.on_event("startup")
def _startup():
    threading.Thread(target=_load_model, daemon=True).start()


def _require_ready():
    if not ready.is_set():
        raise HTTPException(503, "模型加载中，请稍后重试")


# ---------------------------------------------------------------- image loading
def _decode_base64(data: str, max_bytes: int) -> bytes:
    if data.strip().startswith("data:") and "," in data:
        data = data.split(",", 1)[1]
    try:
        raw = base64.b64decode(data, validate=False)
    except binascii.Error as e:
        raise HTTPException(400, f"image_base64 解码失败: {e}")
    if len(raw) > max_bytes:
        raise HTTPException(413, f"图片超过大小限制 ({int(max_bytes)} bytes)")
    return raw


def _read_path_bytes(path: str, max_bytes: int) -> bytes:
    if not os.path.isfile(path):
        raise HTTPException(400, f"image_path 不存在: {path}")
    if os.path.getsize(path) > max_bytes:
        raise HTTPException(413, f"图片超过大小限制 ({int(max_bytes)} bytes)")
    with open(path, "rb") as f:
        return f.read()


def _fetch_url_bytes(url: str, max_bytes: int) -> bytes:
    scheme = urllib.parse.urlparse(url).scheme
    if scheme not in ("http", "https"):
        raise HTTPException(400, "image_url 仅支持 http/https")
    req = urllib.request.Request(url, headers={"User-Agent": "yolo26-server/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read(int(max_bytes) + 1)
    except urllib.error.URLError as e:
        raise HTTPException(502, f"下载 image_url 失败: {e}")
    if len(data) > max_bytes:
        raise HTTPException(413, f"图片超过大小限制 ({int(max_bytes)} bytes)")
    return data


class DetectRequest(BaseModel):
    image_base64: Optional[str] = None
    image_path: Optional[str] = None
    image_url: Optional[str] = None
    conf: Optional[float] = None
    iou: Optional[float] = None
    device: Optional[str] = None
    smoker_max_dist: Optional[float] = None
    save_image: bool = True
    return_image_base64: bool = False


def _resolve_image_bytes(req: DetectRequest, max_bytes: int) -> bytes:
    provided = [v for v in (req.image_base64, req.image_path, req.image_url) if v]
    if len(provided) != 1:
        raise HTTPException(400, "必须且只能提供 image_base64 / image_path / image_url 三者之一")
    if req.image_base64:
        return _decode_base64(req.image_base64, max_bytes)
    if req.image_path:
        return _read_path_bytes(req.image_path, max_bytes)
    return _fetch_url_bytes(req.image_url, max_bytes)


# ---------------------------------------------------------------- inference
def _run_detect(img_bytes: bytes, conf: float, iou: float, device: str,
                 smoker_max_dist: float, save_image: bool, return_image_base64: bool) -> dict:
    t0 = time.time()
    arr = np.frombuffer(img_bytes, dtype=np.uint8)
    im0 = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if im0 is None:
        raise HTTPException(400, "无法解码图片数据，确认是合法的图片格式 (jpg/png/...)")

    with infer_lock:
        result = model.predict(im0, conf=conf, iou=iou, device=device, verbose=False)[0]

    names = model.names
    boxes_by_cls = {}
    for box in result.boxes:
        cls_name = names[int(box.cls[0])]
        xyxy = [round(v, 1) for v in box.xyxy[0].tolist()]
        boxes_by_cls.setdefault(cls_name, []).append((xyxy, float(box.conf[0])))

    person_boxes = boxes_by_cls.pop("Person", [])
    evidence_items = [(cls, xyxy, c) for cls, lst in boxes_by_cls.items() for xyxy, c in lst]

    smoker_idx = set()
    evidences = []
    for cls, xyxy, c in evidence_items:
        linked = nearest_person(xyxy, [b for b, _ in person_boxes], smoker_max_dist)
        if linked is not None:
            smoker_idx.add(linked)
        evidences.append({"cls": cls, "conf": round(c, 4), "bbox_xyxy": xyxy, "linked_person_idx": linked})

    persons = [
        {"bbox_xyxy": xyxy, "conf": round(c, 4), "role": "smoker" if i in smoker_idx else "person"}
        for i, (xyxy, c) in enumerate(person_boxes)
    ]

    found = len(smoker_idx) > 0
    if found:
        ev_types = sorted({e["cls"] for e in evidences if e["linked_person_idx"] in smoker_idx})
        conclusion = f"检测到 {len(smoker_idx)} 位疑似吸烟者（依据：{'/'.join(ev_types)}）"
    elif evidences and persons:
        conclusion = "检测到吸烟相关证据，但未能关联到具体人物"
    elif persons:
        conclusion = f"检测到 {len(persons)} 位人物，未发现吸烟证据"
    else:
        conclusion = "画面中未检测到人物或吸烟证据"

    resp = {
        "found": found,
        "conclusion": conclusion,
        "num_persons": len(persons),
        "num_smokers": len(smoker_idx),
        "persons": persons,
        "evidences": evidences,
        "latency_s": round(time.time() - t0, 3),
        "model_weights": YOLO_WEIGHTS,
        "device": device,
    }

    if save_image or return_image_base64:
        annotated = im0.copy()
        for i, (xyxy, c) in enumerate(person_boxes):
            color, label = (SMOKER_COLOR, f"Smoker {c:.2f}") if i in smoker_idx else (PERSON_COLOR, f"Person {c:.2f}")
            draw_box(annotated, xyxy, color, label)
        for cls, xyxy, c in evidence_items:
            draw_box(annotated, xyxy, EVIDENCE_COLORS.get(cls, DEFAULT_EVIDENCE_COLOR), f"{cls} {c:.2f}")

        ok, buf = cv2.imencode(".jpg", annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
        if not ok:
            raise HTTPException(500, "标注图片编码失败")

        if save_image:
            os.makedirs(YOLO_OUT_DIR, exist_ok=True)
            fname = time.strftime("det_%Y%m%d_%H%M%S") + f"_{int(t0 * 1000) % 1000}.jpg"
            out_path = os.path.join(YOLO_OUT_DIR, fname)
            with open(out_path, "wb") as f:
                f.write(buf.tobytes())
            resp["annotated_image_path"] = out_path
            resp["_fname"] = fname
        if return_image_base64:
            resp["annotated_image_base64"] = base64.b64encode(buf.tobytes()).decode()

    return resp


async def _dispatch(request: Request, img_bytes: bytes, conf: Optional[float], iou: Optional[float],
                     device: Optional[str], smoker_max_dist: Optional[float],
                     save_image: bool, return_image_base64: bool) -> dict:
    max_bytes = YOLO_MAX_IMAGE_MB * 1024 * 1024
    if len(img_bytes) > max_bytes:
        raise HTTPException(413, f"图片超过大小限制 ({int(max_bytes)} bytes)")

    def work():
        return _run_detect(
            img_bytes,
            conf if conf is not None else YOLO_CONF,
            iou if iou is not None else YOLO_IOU,
            device or YOLO_DEVICE,
            smoker_max_dist if smoker_max_dist is not None else YOLO_SMOKER_MAX_DIST,
            save_image,
            return_image_base64,
        )

    resp = await asyncio.get_event_loop().run_in_executor(None, work)
    fname = resp.pop("_fname", None)
    if fname:
        resp["annotated_image_url"] = str(request.base_url).rstrip("/") + "/detections/" + fname
    return resp


# ---------------------------------------------------------------- API
@app.post("/detect")
async def detect(req: DetectRequest, request: Request):
    _require_ready()
    img_bytes = _resolve_image_bytes(req, YOLO_MAX_IMAGE_MB * 1024 * 1024)
    return await _dispatch(request, img_bytes, req.conf, req.iou, req.device,
                            req.smoker_max_dist, req.save_image, req.return_image_base64)


@app.post("/detect/upload")
async def detect_upload(
    request: Request,
    file: UploadFile = File(...),
    conf: Optional[float] = Form(None),
    iou: Optional[float] = Form(None),
    device: Optional[str] = Form(None),
    smoker_max_dist: Optional[float] = Form(None),
    save_image: bool = Form(True),
    return_image_base64: bool = Form(False),
):
    _require_ready()
    img_bytes = await file.read()
    return await _dispatch(request, img_bytes, conf, iou, device,
                            smoker_max_dist, save_image, return_image_base64)


@app.get("/health")
async def health():
    if not ready.is_set():
        return JSONResponse({"ok": False, "reason": "model loading"}, status_code=503)
    return {
        "ok": True,
        "weights": YOLO_WEIGHTS,
        "device": YOLO_DEVICE,
        "conf": YOLO_CONF,
        "iou": YOLO_IOU,
        "classes": model.names,
    }


@app.get("/classes")
async def classes():
    _require_ready()
    return {"classes": model.names}


@app.get("/detections/{fname}")
async def detection_image(fname: str):
    if "/" in fname or ".." in fname or not fname.endswith(".jpg"):
        raise HTTPException(404)
    path = os.path.join(YOLO_OUT_DIR, fname)
    if not os.path.exists(path):
        raise HTTPException(404, "no such detection image")
    return FileResponse(path, media_type="image/jpeg")


def main():
    uvicorn.run(app, host="0.0.0.0", port=YOLO_PORT, log_level="info")


if __name__ == "__main__":
    main()
