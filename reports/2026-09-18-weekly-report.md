# 周报 — 吸烟行为检测(YOLO26)

日期:2026-09-18
仓库:[yolo26_smoking_person_detect](https://github.com/Artorias6655233/yolo26_smoking_person_detect)

## 本周目标

验证能否直接用现成数据集训出一个可用的吸烟行为检测模型(识别 Person / Cigarette / Smoke / Vape),并搭好可复现的训练流程和 GitHub 仓库。

## 完成情况

### 1. 仓库搭建

- 数据集:Roboflow `visionwork/smoking_person` v3,YOLO26 格式,5,658 张图(train 5,307 / valid 270 / test 81),4 类。
- 初始化 git 仓库,配好 README、LICENSE(MIT,并注明 `ultralytics` 依赖是 AGPL-3.0、数据集是 CC BY 4.0)、`.gitignore`、`requirements.txt`、训练脚本 `train.py`。

### 2. 首版训练(baseline)

- `yolo26n`,150 epoch(patience=50,第 87 轮触发 early stop),约 2 小时,RTX 3080。
- valid 集结果:

  | 类别 | mAP50-95 |
  | --- | --- |
  | 全部 | 0.415 |
  | Person | 0.712 |
  | Cigarette | 0.497 |
  | Smoke | 0.310 |
  | Vape | 0.297(仅 6 个验证样本,数据太少,仅供参考) |

- 用 53 张网络图片(`test/webImage`)做了人工抽查,吸烟场景基本能正确框出人和香烟。

### 3. 发现并修复:缺负样本导致误判

- 用 Kaggle `Smoking vs Not Smoking` 数据集(400 张标签明确的验证图,吸烟/不吸烟各 200)做了定量测试。
- **baseline 模型在不吸烟图片上有 5.5%(11/200)的误判率**——典型误判:自拍近景把眼睛框成香烟、打电话手机贴嘴边、抬手比划手指。根源是原训练集里几乎全是"确实在吸烟"的正样本,没有这类"长得像但不是"的负样本。
- 把该数据集里 805 张不吸烟图作为无标注背景图并入训练集,重新训练(v2)。结果:

  | 版本 | smoking 召回 | notsmoking 误判率 |
  | --- | --- | --- |
  | v1(无负样本) | 192/200(96%) | 11/200(5.5%) |
  | **v2(加负样本后)** | 192/200(96%) | **4/200(2%)** |

  召回率不变,误判率下降 63%。

### 4. 配套脚本

- `train.py` — 训练入口,自动把最优权重存到 `models/`(纳入版本控制,~5MB/份)。
- `detect_and_box.py` — 推理脚本,画出 Cigarette(红框)和判定出的"抽烟者"(橙框)。
- `tools/add_negative_samples.py` — 幂等地把负样本背景图合并进训练集。

全部已提交并 push 到 GitHub master。

## 已知局限 / 风险

- **Vape 类别样本极少**(仅 25 个标注框),模型基本学不到有效特征,检测结果不可信。
- Smoke 类别 mAP 相对偏低(0.27~0.31),复杂背景/半透明烟雾对模型仍有挑战。
- 目前只验证过 `yolo26n`(最小规模),尚未对比 `s`/`m` 规模是否有明显精度提升。
- 负样本测试集(`smokingVSnotsmoking/validation_data`)本质是人像分类数据,和实际部署场景(比如监控画面、多人环境)风格差异较大,误判率数字仅代表这一特定场景下的表现,不能直接外推到所有场景。

## 下周计划(待讨论优先级)

1. 针对 Vape / Smoke 类别补充标注数据,提升这两类的检测效果。
2. 评估 `yolo26s` 是否值得作为精度更高的备选版本。
3. 补充更多样化的负样本场景(如吃东西、喝水、拿吸管/口红等),让误判率测试更全面。
4. 明确模型的实际部署场景(实时视频流 / 单图审核),决定后续要不要做导出(ONNX/TensorRT)和推理性能优化。
