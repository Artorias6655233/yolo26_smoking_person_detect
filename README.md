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
| train | 5,307 |
| valid | 270 |
| test | 81 |

- 许可证:**CC BY 4.0**(需保留原作者署名),原始来源见上方链接。
- **已知局限**:类别分布严重不均衡,`Person` 占绝大多数标注,而 `Vape` 仅有 25 个标注框,样本量过少,模型大概率无法学到该类别的有效特征,训练/评估时应重点关注 `Cigarette`、`Person`、`Smoke` 三类的表现,`Vape` 的检测结果仅供参考。

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
yolo detect val model=models/smoking_person_yolo26n.pt data=Smoking_person.v3i.yolo26/data.yaml split=test
```

## 目录结构

```
.
├── train.py                       # 训练脚本
├── requirements.txt
├── models/                        # 最终训练好的权重(纳入版本控制)
│   └── smoking_person_yolo26n.pt
├── Smoking_person.v3i.yolo26/     # 数据集(需自行下载,见上文,不纳入版本控制)
└── runs/                          # 训练过程中的完整输出(不纳入版本控制)
```

## 许可证

- 本仓库代码采用 [MIT License](LICENSE)。
- `ultralytics` 依赖库采用 **AGPL-3.0** 许可,闭源商用需向 Ultralytics 购买 Enterprise License,详见其[官方仓库](https://github.com/ultralytics/ultralytics)。
- 数据集采用 **CC BY 4.0**,由 Roboflow 用户 `visionwork` 提供,使用时请保留署名并遵守该许可条款。
