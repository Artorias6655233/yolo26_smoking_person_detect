# yolo26-smoking-person-detect

基于 [Ultralytics YOLO26](https://github.com/ultralytics/ultralytics) 训练的吸烟行为检测模型,识别画面中的**人 (Person)**、**香烟 (Cigarette)**、**烟雾 (Smoke)** 和 **电子烟 (Vape)**。

## 数据集

数据来自 Roboflow Universe 上的公开数据集 [`visionwork/smoking_person`](https://universe.roboflow.com/visionwork/smoking_person),版本 v3,以 YOLO26 格式导出。

| 类别 (class id) | 名称 | 标注框数量 |
| --- | --- | --- |
| 0 | Cigarette | 441 |
| 1 | Person | 5,269 |
| 2 | Smoke | 579 |
| 3 | Vape | 25 |

| 数据集划分 | 图片数量 |
| --- | --- |
| train | 5,307(+ 805 张负样本背景图,见下方) |
| valid | 270 |
| test | 81 |

- 许可证:**CC BY 4.0**(需保留原作者署名),原始来源见上方链接。
- **已知局限**:类别分布严重不均衡,`Person` 占绝大多数标注,而 `Vape` 仅有 25 个标注框,样本量过少,模型大概率无法学到该类别的有效特征,训练/评估时应重点关注 `Cigarette`、`Person`、`Smoke` 三类的表现,`Vape` 的检测结果仅供参考。

### 负样本增强(压低误判率)

原始数据集几乎全是"确实在吸烟"的正样本,缺少"长得像吸烟但其实不是"的负样本(比如近景自拍、打电话、抬手比划)。实测发现这会导致模型在这类场景里把眼睛、手机、手指误判成 `Cigarette`(详见下方"误判测试"一节)。

解决办法是把 Kaggle 上的 [`Smoking vs Not Smoking`](https://www.kaggle.com/datasets/sujaykapadnis/smoking) 数据集里 `training_data/notsmoking`(805 张,人物但不吸烟)作为**无标注背景图**加进训练集,教模型不要在这类图上出框。`validation_data/notsmoking`(200 张)特意不加入训练,留作误判率的长期回归测试集。

```bash
python tools/add_negative_samples.py
```

该脚本是幂等的(已存在的文件会跳过),把图片复制进 `Smoking_person.v3i.yolo26/train/images/`,并在 `train/labels/` 下为每张图创建一个空的 `.txt`(YOLO 里空标注 = 背景图,没有需要检测的目标)。

### 获取数据集

出于仓库体积考虑,原始图片未纳入版本控制(见 `.gitignore`)。请自行从上方 Roboflow 链接导出 **YOLO26** 格式,并解压到仓库根目录,保持以下目录结构(与 `data.yaml` 中的相对路径一致):

```
yolo26_smoking_person_detect/
└── Smoking_person.v3i.yolo26/
    ├── data.yaml
    ├── train/
    │   ├── images/
    │   └── labels/
    ├── valid/
    │   ├── images/
    │   └── labels/
    └── test/
        ├── images/
        └── labels/
```

同样出于体积考虑,用于负样本增强的 [`smokingVSnotsmoking`](https://www.kaggle.com/datasets/sujaykapadnis/smoking) 数据集也未纳入版本控制。如果要复现负样本增强(见上文),下载后解压到仓库根目录,保持 `smokingVSnotsmoking/training_data/notsmoking`、`smokingVSnotsmoking/validation_data/{smoking,notsmoking}` 这样的目录结构,再运行 `tools/add_negative_samples.py`。

## 环境要求

- Python >= 3.9
- 支持 CUDA 的 GPU(建议显存 >= 6GB;本项目在 RTX 3080 10GB 上验证过 `yolo26n`/`yolo26s` 规模)
- [PyTorch](https://pytorch.org/get-started/locally/)(需匹配本机 CUDA 版本单独安装)
- [Ultralytics](https://github.com/ultralytics/ultralytics) >= 8.4.0(已内置 YOLO26 模型定义)

## 安装

```bash
# 1. 先按本机 CUDA 版本安装匹配的 PyTorch,例如 CUDA 12.4:
pip install torch --index-url https://download.pytorch.org/whl/cu124

# 2. 安装其余依赖
pip install -r requirements.txt
```

## 训练

```bash
python train.py --epochs 150 --imgsz 640 --batch 16 --device 0
```

常用参数(完整列表见 `python train.py --help`):

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--data` | `Smoking_person.v3i.yolo26/data.yaml` | 数据集配置文件 |
| `--model` | `yolo26n.pt` | 起始权重/模型结构,首次运行会自动下载预训练权重 |
| `--epochs` | 150 | 训练轮数 |
| `--imgsz` | 640 | 输入分辨率 |
| `--batch` | 16 | batch size,显存不足时调小 |
| `--device` | `0` | 训练设备,`cpu` 或 GPU id |
| `--patience` | 50 | early stopping 的容忍轮数 |
| `--resume` | - | 从上次训练的 last checkpoint 继续 |

训练过程中的完整日志、中间 checkpoint 和可视化结果会保存在 `runs/detect/<name>/` 下(体积较大,已加入 `.gitignore`,不会被提交)。训练结束后,`train.py` 会自动把最优权重 `best.pt` 复制一份到 `models/<name>.pt`——这个目录**会**被提交到 git,作为可复现、可直接下载使用的最终模型。

## 评估

```bash
yolo detect val model=models/smoking_person_yolo26n_v2.pt data=Smoking_person.v3i.yolo26/data.yaml split=test
```

`models/` 下有三版权重,按训练先后保留,方便对比:

| 权重 | 模型规模 | 训练集 | valid mAP50-95 |
| --- | --- | --- | --- |
| `smoking_person_yolo26n.pt` | yolo26n | 原始 5,307 张(无负样本) | 0.415 |
| `smoking_person_yolo26n_v2.pt` | yolo26n | 加上 805 张负样本背景图 | 0.425 |
| `smoking_person_yolo26s.pt`(**推荐**) | yolo26s | 加上 805 张负样本背景图 | **0.451** |

`yolo26s`(约 20MB,10.0M 参数)相比 `yolo26n`(约 5MB,2.4M 参数)体积大 4 倍左右,在同样的数据集、超参数(`epochs=150 --batch 16 --imgsz 640 --patience 50`)下训练,valid mAP50-95 从 0.425 提升到 0.451(+6%),但推理速度和显存占用也相应增加,按需在精度和延迟之间取舍。`yolo26s` 的最优权重出现在第 48 轮(patience=50 早停于第 98 轮),各类别的 valid 表现:

| 类别 | Precision | Recall | mAP50 | mAP50-95 |
| --- | --- | --- | --- | --- |
| Cigarette | 0.763 | 0.867 | 0.852 | 0.507 |
| Person | 0.938 | 0.899 | 0.952 | 0.699 |
| Smoke | 0.644 | 0.452 | 0.524 | 0.279 |
| Vape | 0.601 | 0.500 | 0.428 | 0.318 |

## 误判(假阳性)测试

用 `detect_and_box.py` 在 `smokingVSnotsmoking/validation_data` 上分别跑 `smoking`/`notsmoking` 两组(各 200 张、有明确标签),量化"没有负样本会不会导致大量误判"这个问题:

```bash
python detect_and_box.py --source smokingVSnotsmoking/validation_data/smoking    --output test/smokingVSnotsmoking_boxed/smoking
python detect_and_box.py --source smokingVSnotsmoking/validation_data/notsmoking --output test/smokingVSnotsmoking_boxed/notsmoking
```

| 模型 | smoking 召回(检出 Cigarette) | notsmoking 误判率(误检出 Cigarette) |
| --- | --- | --- |
| v1(无负样本) | 192/200 (96%) | 11/200 (5.5%) |
| v2(加负样本后) | 192/200 (96%) | **4/200 (2%)** |

结论:v1 确实存在"缺负样本→误判"的问题,典型误判场景是近景自拍(把眼睛框成 Cigarette)、打电话(手机贴近嘴边)、抬手比划手指——这些跟原数据集里"人 + 烟"的取景差异很大,模型没学过要排除它们。把 Kaggle `notsmoking` 图片当无标注背景图加入训练后,召回率不变,误判率下降了 63%。

`detect_and_box.py` 用本仓库的模型同时检测 `Person` 和 `Cigarette`:Cigarette 画红框,离某个 Cigarette 最近的 Person 判定为"抽烟者"画橙框,其余 Person 画青色框。默认权重是 `models/smoking_person_yolo26n_v2.pt`,可用 `--weights` 换成别的权重对比效果。

## 目录结构

```
.
├── train.py                       # 训练脚本
├── detect_and_box.py              # 推理脚本:画出 Cigarette / 抽烟者 框
├── yolo_server/                   # HTTP 检测服务(FastAPI),见 yolo_server/README.md
├── tools/add_negative_samples.py  # 把负样本背景图合并进训练集
├── requirements.txt
├── models/                        # 最终训练好的权重(纳入版本控制)
│   ├── smoking_person_yolo26n.pt
│   ├── smoking_person_yolo26n_v2.pt
│   └── smoking_person_yolo26s.pt
├── Smoking_person.v3i.yolo26/     # 数据集(需自行下载,见上文,不纳入版本控制)
├── smokingVSnotsmoking/           # 负样本来源 + 误判测试集(需自行下载,不纳入版本控制)
└── runs/                          # 训练过程中的完整输出(不纳入版本控制)
```

## HTTP 检测服务

`yolo_server/server.py` 把 `models/*.pt` 包装成一个 FastAPI 服务：POST 图片
(本地路径/base64/URL/文件上传)，返回吸烟检测结果 + 标注图。详见
[`yolo_server/README.md`](yolo_server/README.md)。

```bash
python yolo_server/server.py
curl -X POST http://localhost:9998/detect -H 'Content-Type: application/json' \
  -d '{"image_path": "test/webImage/1.jpg"}'
```

## Qwen3-VL LoRA 微调数据集

`Smoking_person.v3i.qwen3vl-lora/` 是本数据集自动转换出的另一份格式，用于 LoRA 微调
`vllm/models/qwen3-vl-2b-gguf`，不修改原始 YOLO 数据。生成/更新：

```bash
python3 tools/convert_yolo_to_qwen3vl_lora.py
```

详见 [`Smoking_person.v3i.qwen3vl-lora/README.md`](Smoking_person.v3i.qwen3vl-lora/README.md)。

## 许可证

- 本仓库代码采用 [MIT License](LICENSE)。
- `ultralytics` 依赖库采用 **AGPL-3.0** 许可,闭源商用需向 Ultralytics 购买 Enterprise License,详见其[官方仓库](https://github.com/ultralytics/ultralytics)。
- 数据集采用 **CC BY 4.0**,由 Roboflow 用户 `visionwork` 提供,使用时请保留署名并遵守该许可条款。
