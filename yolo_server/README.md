# YOLO26 检测服务 (`yolo_server/server.py`)

把本仓库训练好的 YOLO26 权重（`models/*.pt`）包装成一个 HTTP 服务：客户端传一张
图片（本地路径 / base64 / URL / 文件上传），服务端跑一次推理，把 `Cigarette` /
`Smoke` / `Vape` 这类"证据"框关联到离它最近的 `Person` 框（复用
[`detect_and_box.py`](../detect_and_box.py) 里验证过的关联算法），返回结构化的
检测结果和一张画好框的标注图。

结构上参考了 [`vllm/observer/server.py`](../vllm/observer/server.py)：同样是
FastAPI + 后台线程加载模型 + `/health` + 存图片接口，区别是图片来自 HTTP 请求
本身，而不是订阅 ROS 摄像头话题——本仓库不依赖 ROS，`yolo_server` 也不需要。

## 1. 安装

```bash
pip install -r requirements.txt   # 已包含 fastapi / uvicorn / python-multipart
```

## 2. 启动

```bash
python yolo_server/server.py
# 或者用其他 host/port/权重跑：
YOLO_PORT=9998 YOLO_WEIGHTS=models/smoking_person_yolo26n_v2.pt YOLO_DEVICE=0 python yolo_server/server.py
```

模型在后台线程里异步加载，加载完成前 `/health` 返回 503，其余接口返回
`模型加载中，请稍后重试`。CPU 上也能跑（`YOLO_DEVICE=cpu`），只是慢一些。

### 环境变量

| 变量 | 默认值 | 说明 |
| --- | --- | --- |
| `YOLO_PORT` | `9998` | 监听端口 |
| `YOLO_WEIGHTS` | `models/smoking_person_yolo26n_v2.pt` | 权重路径 |
| `YOLO_DEVICE` | `0` | 推理设备，`0`/`1`(GPU id) 或 `cpu` |
| `YOLO_CONF` | `0.25` | 默认置信度阈值，可被单次请求覆盖 |
| `YOLO_IOU` | `0.45` | 默认 NMS IOU 阈值，可被单次请求覆盖 |
| `YOLO_SMOKER_MAX_DIST` | `2.0` | 证据框到人体框中心的最大归一化距离，超过则不关联（见 `detect_and_box.py:nearest_person`） |
| `YOLO_OUT_DIR` | `runs/server_detections` | 标注图片保存目录 |
| `YOLO_MAX_IMAGE_MB` | `15` | 单张图片大小上限（base64/上传/URL 下载都受限） |

## 3. API

### `POST /detect` — JSON 请求

`image_base64` / `image_path` / `image_url` 三选一：

```bash
# 本地路径（服务进程能访问到的路径，常用于同机调试）
curl -X POST http://localhost:9998/detect -H 'Content-Type: application/json' -d '{
  "image_path": "test/webImage/1.jpg"
}'

# base64（可以是纯 base64，也可以带 data:image/jpeg;base64, 前缀）
curl -X POST http://localhost:9998/detect -H 'Content-Type: application/json' -d "{
  \"image_base64\": \"$(base64 -w0 test/webImage/1.jpg)\"
}"

# 远程 URL（仅支持 http/https）
curl -X POST http://localhost:9998/detect -H 'Content-Type: application/json' -d '{
  "image_url": "https://example.com/pic.jpg"
}'
```

可选字段：`conf`、`iou`、`device`、`smoker_max_dist`（覆盖对应环境变量默认值）、
`save_image`（默认 `true`，是否落盘标注图）、`return_image_base64`（默认
`false`，是否把标注图直接以 base64 塞进响应体，适合拿不到服务器文件系统/URL 的
客户端）。

### `POST /detect/upload` — 表单文件上传

```bash
curl -X POST http://localhost:9998/detect/upload \
  -F "file=@test/webImage/1.jpg" \
  -F "conf=0.3" \
  -F "save_image=true"
```

字段同上（`conf`/`iou`/`device`/`smoker_max_dist`/`save_image`/
`return_image_base64`），走 `multipart/form-data` 而不是 JSON。

### 响应字段

```json
{
  "found": true,
  "conclusion": "检测到 1 位疑似吸烟者（依据：Cigarette）",
  "num_persons": 1,
  "num_smokers": 1,
  "persons": [
    {"bbox_xyxy": [0.0, 0.0, 508.8, 410.0], "conf": 0.8847, "role": "smoker"}
  ],
  "evidences": [
    {"cls": "Cigarette", "conf": 0.8368, "bbox_xyxy": [312.1, 137.0, 506.3, 220.2], "linked_person_idx": 0}
  ],
  "latency_s": 0.05,
  "model_weights": "models/smoking_person_yolo26n_v2.pt",
  "device": "0",
  "annotated_image_path": "runs/server_detections/det_20260920_103338_7.jpg",
  "annotated_image_url": "http://localhost:9998/detections/det_20260920_103338_7.jpg"
}
```

- `found`：是否存在被关联到某个人的吸烟证据（单纯检测到 `Person` 不算，单纯检测
  到没关联上任何人的 `Cigarette`/`Smoke`/`Vape` 也不算——判定规则与
  `detect_and_box.py` 的"抽烟者"关联逻辑一致）。
- `persons[].role`：`smoker`（关联到证据）或 `person`（普通行人）。
- `evidences[].linked_person_idx`：关联到的 `persons` 数组下标，`null` 表示没关
  联上任何人（比如烟雾飘在画面里但底下没有识别出人体框）。
- `save_image=false` 时不会有 `annotated_image_path`/`annotated_image_url`。

### `GET /health`

```bash
curl http://localhost:9998/health
# {"ok":true,"weights":"models/smoking_person_yolo26n_v2.pt","device":"0","conf":0.25,"iou":0.45,"classes":{"0":"Cigarette","1":"Person","2":"Smoke","3":"Vape"}}
```

### `GET /classes`

返回当前权重的 `class id -> 名称` 映射（换了权重、类别数不一样时用这个确认，而
不是硬编码 4 个类）。

### `GET /detections/{文件名}`

拉取某次检测保存下来的标注图（`YOLO_OUT_DIR` 下的文件，文件名做了路径穿越校
验，只接受不含 `/`、`..` 且以 `.jpg` 结尾的名字）。

## 4. 安全提示

- `image_path` 允许服务进程读取其能访问到的**任意本地文件**，这是为了同机调
  试方便；如果要把这个服务暴露到不受信任的网络，**务必去掉 `image_path` 这个
  分支**（或者加白名单目录校验），否则等于开了一个任意文件读取接口。
- `image_url` 只放行了 `http`/`https` 协议（挡掉 `file://` 之类的本地文件读
  取），但没有做内网地址（SSRF）过滤；生产环境如果要接受任意用户提供的 URL，
  建议再加一层对内网 IP 段的拒绝。
- 没有做鉴权，见下面"扩展"一节的示例。

## 5. 如何扩展 API

`server.py` 里所有推理逻辑都收敛在 `_run_detect()` 一个函数里，`/detect` 和
`/detect/upload` 只是两种拿到 `img_bytes` 的方式，最后都调用同一个
`_dispatch()`。新增功能时优先复用这条链路，而不是另起一套。

### 5.1 新增一个端点

比如加一个"只返回是否有人在抽烟，不要那么多字段"的极简端点，给下游脚本用：

```python
@app.post("/detect/simple")
async def detect_simple(req: DetectRequest, request: Request):
    _require_ready()
    img_bytes = _resolve_image_bytes(req, YOLO_MAX_IMAGE_MB * 1024 * 1024)
    full = await _dispatch(request, img_bytes, req.conf, req.iou, req.device,
                            req.smoker_max_dist, save_image=False, return_image_base64=False)
    return {"smoking": full["found"]}
```

### 5.2 批量检测：`/detect/batch`

多张图一次请求发过去，逐张跑 `_run_detect`（推理本身仍然串行，`infer_lock` 已
经保证了这一点，不用额外加锁）：

```python
class BatchDetectRequest(BaseModel):
    items: list[DetectRequest]

@app.post("/detect/batch")
async def detect_batch(req: BatchDetectRequest, request: Request):
    _require_ready()
    results = []
    for item in req.items:
        img_bytes = _resolve_image_bytes(item, YOLO_MAX_IMAGE_MB * 1024 * 1024)
        results.append(await _dispatch(request, img_bytes, item.conf, item.iou, item.device,
                                        item.smoker_max_dist, item.save_image, item.return_image_base64))
    return {"results": results}
```

### 5.3 热切换权重：`/reload`

现在权重在启动时加载一次，改权重要重启进程。要支持热切换，把加载逻辑包一层锁：

```python
model_lock = threading.Lock()

class ReloadRequest(BaseModel):
    weights: str

@app.post("/reload")
async def reload_model(req: ReloadRequest):
    global model, YOLO_WEIGHTS
    new_model = YOLO(req.weights)          # 加载失败会在这里直接抛异常 -> 500
    with model_lock, infer_lock:
        model = new_model
        YOLO_WEIGHTS = req.weights
    return {"ok": True, "weights": YOLO_WEIGHTS, "classes": model.names}
```

`_run_detect` 里读 `model`/`model.names` 的地方也要用 `model_lock` 包一下读
（当前单模型不热切换的场景下不需要，一旦加了 `/reload` 就必须加，否则会有另一
个请求正在用旧 model 对象推理时被换掉的竞态）。

### 5.4 加鉴权

在真正对外网开放之前，最简单的方式是一个 FastAPI 中间件校验固定 token（生产
环境建议换成更完整的方案，这里只是示例）：

```python
from fastapi import Header

API_TOKEN = os.environ.get("YOLO_API_TOKEN")  # 不设置就不校验，方便本机调试

@app.middleware("http")
async def check_token(request: Request, call_next):
    if API_TOKEN and request.url.path not in ("/health",):
        if request.headers.get("Authorization") != f"Bearer {API_TOKEN}":
            return JSONResponse({"detail": "unauthorized"}, status_code=401)
    return await call_next(request)
```

### 5.5 调整"是否算吸烟"的判定规则

判定规则全在 `_run_detect()` 里三个地方：

1. `nearest_person(xyxy, person_boxes, smoker_max_dist)` — 证据框和哪个人关
   联，改 `detect_and_box.py` 里的这个函数会同时影响 `detect_and_box.py` 和这
   个服务（两边共用一份逻辑，改的时候留意别只顾着一边的场景）。
2. `found = len(smoker_idx) > 0` — 目前"关联上至少一个人"就算 `found=True`；
   如果想要求"同一帧里至少两条独立证据才算"，改这一行的条件即可。
3. `conclusion` 的拼接逻辑——纯展示文案，改起来不影响 `found`/`persons`/
   `evidences` 这些结构化字段，下游脚本不用跟着改。

### 5.6 新增类别

数据集类别定义在 `Smoking_person.v3i.yolo26/data.yaml`（当前是 `Cigarette` /
`Person` / `Smoke` / `Vape`）。重新训练加了新类别后，`server.py` 不需要改代
码——`boxes_by_cls`/`evidence_items` 是按 `model.names` 动态取的，新类别会自动
被当成一种"证据"参与人物关联；只有想给它单独配色时才需要在 `EVIDENCE_COLORS`
里加一条，否则会落到 `DEFAULT_EVIDENCE_COLOR`（黄色）。
