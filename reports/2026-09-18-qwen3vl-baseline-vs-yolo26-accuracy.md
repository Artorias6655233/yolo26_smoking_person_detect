# 测试报告 — Qwen3-VL-2B(未微调) vs YOLO26 吸烟判定准确率

日期:2026-09-18
仓库:[yolo26_smoking_person_detect](https://github.com/Artorias6655233/yolo26_smoking_person_detect)

## 背景与目的

最终目标是对比三个方案在"图片中是否有人吸烟"这个二分类判断上的表现:

1. **YOLO26**(本仓库已训练,`models/smoking_person_yolo26n*.pt`)
2. **Qwen3-VL-2B-Instruct,未 LoRA 微调**(本报告)
3. **Qwen3-VL-2B-Instruct,LoRA 微调后**(训练中,结果另行补充)

本报告先跑通第 2 项,并把结果和已有的第 1 项数据放在一起对比,为后续 LoRA 微调提供一个"微调前"基线。
**只评估"是否有人吸烟"这个二元判断的准确率,不评估画框(bbox)的定位精度。**

## 测试方法

- **模型**:`Qwen/Qwen3-VL-2B-Instruct`(HuggingFace 原始权重,4bit NF4 量化,`bnb_4bit_compute_dtype=bfloat16`——
  和后续 LoRA 训练用的量化方式一致,方便微调前后直接对比),贪心解码(`do_sample=False`),不接 LoRA adapter。
- **Prompt**:直接复用线上 `vllm/observer` 正在用的那一套(`vllm/scripts/judge_dataset.py` 里的
  `PROMPT_TMPL`),`target` 固定为"是否有人吸烟"。模型返回 JSON:`{"found": bool, "conclusion": str,
  "events": [...]}`,判定规则和线上一致——`found` 为 true 且至少有一个非"人"的真实证据框(排除占位符文本、
  排除和人框重复的框)才算最终判定为"是"。
- **测试集**:`smokingVSnotsmoking/validation_data/{smoking,notsmoking}`,各 200 张,合计 400 张,ground
  truth 由文件夹名给出。**这是 YOLO 那边`detect_and_box.py`误判测试用的同一份数据**(见 README「误判
  (假阳性)测试」一节),所以两边数字可以直接放在一张表里比。
- **脚本**:[`tools/eval_qwen3vl_prompt_accuracy.py`](../tools/eval_qwen3vl_prompt_accuracy.py),逐张跑,
  记录 `found` 判定、模型给出的 `conclusion`、耗时;汇总输出到
  [`reports/qwen3vl_baseline_accuracy_results.json`](qwen3vl_baseline_accuracy_results.json)(含全部 400
  条明细,可复核)。复现:

  ```bash
  python tools/eval_qwen3vl_prompt_accuracy.py
  ```

## 结果

| 方案 | Accuracy | Precision | Recall | F1 | TP | FP | TN | FN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| YOLO26 v1(无负样本) | 95.25% | 94.58% | 96.00% | 95.29% | 192 | 11 | 189 | 8 |
| YOLO26 v2(加负样本,当前推荐) | **97.00%** | **97.96%** | 96.00% | 96.97% | 192 | 4 | 196 | 8 |
| **Qwen3-VL-2B(未微调,本报告)** | 96.00% | 92.59% | **100.00%** | 96.15% | 200 | **16** | 184 | **0** |

(YOLO 两行数字来自 README/`detect_and_box.py` 已有结果,换算出 accuracy/precision/F1;VLM 一行来自本次
400 张全量测试,原始记录见 `qwen3vl_baseline_accuracy_results.json`。)

**关键观察**:

- **召回率(Recall)拉满 100%**——400 张里没有一张漏判(FN=0),200 张确实在吸烟的图全部正确识别,比 YOLO
  两个版本(96%,各漏 8 张)都高。这大概率是 VLM 依赖"整体语义理解"而非"必须先精确检出小目标"的优势:
  YOLO 漏判的 8 张往往是香烟目标太小/被遮挡导致检测器直接没检出框,而 VLM 只要"看得出来这人在抽烟"就够。
- **精确率(Precision)明显偏低,92.59%,比 YOLO 两版都差**——400 张里 16 张不吸烟的图被误判为吸烟
  (FP=16),比 v2(4 张)高 4 倍,甚至比没加负样本的 v1(11 张)还高。**这不是随机噪声,而是集中在几类具体
  动作上**(见下方错误分析),说明未微调的 VLM 还没学会区分"像吸烟的手势"和"真的在吸烟"。
- 综合 accuracy(96.00%)介于 YOLO v1(95.25%)和 v2(97.00%)之间,F1(96.15%)也接近但不及 YOLO v2
  (96.97%)。
- 平均推理耗时 3.68s/张(min 0.81s,max 6.64s),RTX 3080、4bit 量化、单图单请求,无 batch。

## 错误分析:16 个假阳性

逐条看了这 16 张被误判的 `notsmoking` 图片,模型给出的 `conclusion` 高度集中在几类动作上:

| 误判原因 | 张数 | 典型 conclusion |
| --- | --- | --- |
| **咳嗽(手/拳头靠近嘴部)** | 9 | "男子正在咳嗽"、"男子正在咳嗽，用手捂住口鼻" |
| **使用吸入器(哮喘/鼻炎喷雾)** | 2 | "有人正在使用吸入器"、"老人手持蓝色吸入器" |
| **其他手靠近口鼻的动作**(捂鼻子、吐痰) | 2 | "男子用手捂住鼻子"、"人正在吐痰" |
| **看起来像真的在吸烟/有烟雾**(其余,存疑) | 3 | "画面中人物有烟雾冒出"、"男子嘴里叼着烟"、"人正在吸烟" |

**这不是巧合**:`smokingVSnotsmoking`(Mendeley "Smoking vs Non-Smoking")这份负样本数据集在设计时就**刻意
把咳嗽、用吸入器、喝水等"动作上容易和吸烟混淆"的照片放进 notsmoking 类**,目的就是测试模型有没有学到"手靠近
嘴部 ≠ 吸烟"这个区分。未微调的 Qwen3-VL-2B 在这批困难负样本上明显没有学到这个区分,尤其是"咳嗽"这个动作
(9/16,56% 的假阳性都是它)。

对比之下,YOLO26 走的是完全不同的路子——它只认"有没有检测到 Cigarette 这个小物体",不理解"咳嗽"这个
动作语义,所以不会被咳嗽误导;YOLO v2 的 4 个假阳性,大概率是烟雾/阴影/手指之类容易被误检成香烟的视觉
噪声,和 VLM 这种"语义误判"性质不同。

## 结论与下一步

- 未微调的 Qwen3-VL-2B **召回率已经优于 YOLO26**,但**精确率明显不如**,整体 accuracy/F1 略低于 YOLO26
  v2 当前最优版本。
- **假阳性高度集中在"咳嗽/用吸入器"等困难负样本上**,这正是 [`Smoking_person.v3i.qwen3vl-lora`](../Smoking_person.v3i.qwen3vl-lora/)
  数据集里已经并入的负样本类型([README](../Smoking_person.v3i.qwen3vl-lora/README.md)提到的
  "喝水、用吸入器、拿手机"等难负样本设计),理论上 LoRA 微调后应该能显著压低这部分误判、把 precision 拉
  上来,同时保住已经很高的 recall。
- 下一步:完成 LoRA 微调后,用**完全相同的方法**(同一 prompt、同一 400 张测试集、同一评分口径)跑一遍
  微调后的模型,把第三行数据补进上面的表格,形成 YOLO26 vs 未微调VLM vs 微调VLM 的完整三方对比。

## 局限性

- 和 YOLO 那份误判测试一样,`smokingVSnotsmoking` 本质是人像分类数据集(单人近景为主),和实际部署场景
  (监控画面、多人环境、远景)风格差异较大,这里的数字只代表这一特定场景下的相对表现,不能直接外推。
- 贪心解码(temperature=0)、单次采样,没有做多次采样取多数投票之类的方差分析。
- 4bit 量化可能相对 bf16/fp16 全精度有轻微精度损失,但因为后续 LoRA 训练和推理也用同样的量化方式,这里
  刻意保持一致,确保"微调前后"对比是公平的,而不是追求这个基线本身的最高精度。
