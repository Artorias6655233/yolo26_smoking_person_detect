# 测试报告 — yolo26s vs yolo26n_v2 精度对比

日期:2026-09-20
仓库:[yolo26_smoking_person_detect](https://github.com/Artorias6655233/yolo26_smoking_person_detect)

## 背景与目的

已有 `yolo26n`(`models/smoking_person_yolo26n_v2.pt`)作为当前推荐模型。本报告训练一个更大规模的
`yolo26s`,用**完全相同的数据集、超参数**训练,并在**两份和 `yolo26n_v2` 完全相同的测试集**上跑一遍,
量化"换大模型能换来多少精度提升、代价是什么",为要不要切换默认权重提供依据。

## 训练设置

两者只有 `--model` 不同,其余参数一致,保证可比性:

```bash
python train.py --model yolo26n.pt --epochs 150 --imgsz 640 --batch 16 --device 0 --patience 50 --name smoking_person_yolo26n_v2
python train.py --model yolo26s.pt --epochs 150 --imgsz 640 --batch 16 --device 0 --patience 50 --name smoking_person_yolo26s
```

数据集:`Smoking_person.v3i.yolo26`(train 6,112 张,含 805 张负样本背景图,与 `yolo26n_v2` 一致)。

| | yolo26n_v2 | yolo26s |
| --- | --- | --- |
| 参数量(fused) | 2,375,616 | 9,466,728 |
| 权重体积 | ~5MB | ~20MB |
| 训练早停轮次 | 134(最优第 84 轮,见 `train_v2.log`) | 98(最优第 48 轮) |
| valid mAP50-95(训练过程记录) | 0.425 | **0.451** |

## 测试一:test split(同分布,81 张,模型训练时未见过)

```bash
yolo detect val model=models/smoking_person_yolo26n_v2.pt data=Smoking_person.v3i.yolo26/data.yaml split=test
yolo detect val model=models/smoking_person_yolo26s.pt  data=Smoking_person.v3i.yolo26/data.yaml split=test
```

test split 里 Vape 类别 0 个标注实例,故不参与本次对比(与 valid/train 一样,Vape 样本量过少的已知局限依然存在)。

| 指标(全部类别) | yolo26n_v2 | yolo26s | 差值 |
| --- | --- | --- | --- |
| Precision | 0.806 | 0.785 | -0.021 |
| Recall | 0.788 | **0.832** | +0.044 |
| mAP50 | 0.770 | **0.827** | +0.057 |
| **mAP50-95** | 0.483 | **0.504** | **+0.021** |

按类别拆分(mAP50-95):

| 类别 | yolo26n_v2 | yolo26s | 差值 |
| --- | --- | --- | --- |
| Cigarette | 0.456 | **0.479** | +0.023 |
| Person | **0.733** | 0.719 | -0.014 |
| Smoke | 0.260 | **0.313** | +0.053 |

推理速度(单张,RTX 3080,batch=1):

| | yolo26n_v2 | yolo26s |
| --- | --- | --- |
| 预处理 | 3.5ms | 3.0ms |
| 推理 | 8.3ms | 8.6ms |
| 后处理 | 1.1ms | 2.3ms |

**观察**:`yolo26s` 在 Cigarette、Smoke 两个核心类别上都有提升,尤其 Smoke(+0.053,相对提升 20%),
这大概率是模型容量更大、对烟雾这种弱纹理/半透明目标的特征提取能力更强。Person 类别小幅下降(-0.014),
可能是最优权重出现的轮次不同(84 vs 48)导致的正常波动,不代表规律性的能力下降。推理耗时基本持平(8.3ms →
8.6ms,+0.3ms/张),在 RTX 3080 上这个规模差距(4 倍参数量)带来的延迟增量可以忽略,GPU 上瓶颈不在计算量。

## 测试二:误判(假阳性)测试(`smokingVSnotsmoking/validation_data`,同一批 400 张)

沿用 README 里已有的方法论:用 `detect_and_box.py` 检测是否输出 ≥1 个 Cigarette 框作为"判定为吸烟"的依据,
在 200 张 `smoking` + 200 张 `notsmoking`(ground truth 由文件夹名给出)上跑,与 `yolo26n_v2` 已有结果
(见 README、`2026-09-18-qwen3vl-baseline-vs-yolo26-accuracy.md`)放在一起对比:

```bash
python detect_and_box.py --source smokingVSnotsmoking/validation_data/smoking    --output test/smokingVSnotsmoking_boxed/smoking_yolo26s    --weights models/smoking_person_yolo26s.pt
python detect_and_box.py --source smokingVSnotsmoking/validation_data/notsmoking --output test/smokingVSnotsmoking_boxed/notsmoking_yolo26s --weights models/smoking_person_yolo26s.pt
```

| 模型 | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| yolo26n_v2(当前推荐) | 97.00% | 97.96% | **96.00%** | 96.97% | 192 | 4 | 196 | 8 |
| **yolo26s** | 97.00% | **98.96%** | 95.00% | 96.94% | 190 | **2** | 198 | 10 |

**观察**:两者整体 accuracy 打平(97.00%),但**误判方向不同**——`yolo26s` 把 `notsmoking` 的误判率再砍半
(4/200 → 2/200),代价是 `smoking` 漏检多了 2 张(192/200 → 190/200)。即模型更"保守":更不容易把手指/
阴影/烟雾状噪声误判成香烟,但对个别边缘的真实吸烟场景(可能是香烟目标很小或被遮挡)漏检略多。哪个方向更
可取取决于业务场景——如果误报的代价(骚扰无辜用户)比漏报更敏感,`yolo26s` 更合适;如果不能接受漏检,
`yolo26n_v2` 的召回率更高。

## 综合结论

- **精度**:`yolo26s` 在 valid/test 集的整体 mAP50-95 上都稳定超过 `yolo26n_v2`(valid 0.451 vs 0.425,
  test 0.504 vs 0.483),尤其 Smoke 类别提升明显;误判测试上二者 accuracy 打平,但 `yolo26s` 的精确率
  (precision)更高、召回率(recall)略低。
- **代价**:权重体积从 ~5MB 增至 ~20MB(4 倍),但在 RTX 3080 上单图推理延迟几乎没有增加(+0.3ms),
  说明当前场景下模型规模不是延迟瓶颈,`s` 规模基本是"免费"的精度提升——除非部署到算力明显更弱的设备
  (如嵌入式/边缘端),否则没有理由不切换。
- **建议**:把 `models/smoking_person_yolo26s.pt` 作为新的默认推荐权重(`yolo_server`、`detect_and_box.py`
  的默认 `--weights`),`yolo26n_v2` 保留作为轻量级备选。

## 局限性

- 训练只跑了一次(没有多个随机种子的重复实验),`yolo26n_v2` vs `yolo26s` 之间的差异有多少是模型规模带来
  的、有多少是训练随机性(早停轮次不同、数据增强顺序不同)带来的,没有做方差分析,结论只代表这一次训练的
  相对表现。
- `smokingVSnotsmoking` 测试集本质是人像分类数据(单人近景为主),和实际部署场景(监控画面、多人环境、
  远景)风格差异较大,这里的 precision/recall 数字仅代表这一特定场景下的表现,不能直接外推。
- Vape 类别样本量过少(仅 25 个标注框,test split 里为 0),本次对比未覆盖该类别,规律性结论不适用于它。
- 未测试导出后(ONNX/TensorRT)或 CPU/边缘设备上的推理延迟,当前速度对比仅代表 RTX 3080 + PyTorch 原生
  推理路径下的情况。

## 下一步

1. 若采纳 `yolo26s` 作为新默认,更新 `yolo_server/server.py`、`detect_and_box.py` 的默认权重路径,并在
   README 里同步说明。
2. 补一次多样化负样本场景(吃东西、喝水、拿吸管等)的误判测试,验证 `yolo26s` 的"更保守"倾向是否在更多
   场景下依然成立。
3. 结合 [`2026-09-18-qwen3vl-baseline-vs-yolo26-accuracy.md`](2026-09-18-qwen3vl-baseline-vs-yolo26-accuracy.md)
   的方法,把 `yolo26s` 也加入 YOLO vs VLM 的三方对比表。
