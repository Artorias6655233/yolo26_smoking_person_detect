#!/usr/bin/env python3
"""LoRA (QLoRA, 4-bit) fine-tune of Qwen3-VL-2B-Instruct on the
Smoking_person.v3i.qwen3vl-lora dataset.

This is a memory- and time-bounded proof-of-concept fine-tune meant to run
alongside another training job (a YOLO26 detector) sharing the same 10GB
GPU, so it deliberately:
  - loads the base model in 4-bit NF4 (bitsandbytes) to keep the frozen
    base weights small,
  - freezes the vision tower entirely (LoRA is only added to the language
    model's attention/MLP projections -- see TARGET_MODULES below; none of
    those name suffixes exist in the vision tower, see the sanity check in
    `build_model`),
  - uses batch_size=1 with gradient accumulation instead of a larger batch,
  - caps image resolution via the processor's min/max_pixels,
  - trains a small, fixed number of optimizer steps rather than a full
    multi-epoch run.

Usage:
  python tools/train_qwen3vl_lora.py
  python tools/train_qwen3vl_lora.py --max-steps 600 --grad-accum 16
"""
import argparse
import json
import os
import random
import subprocess
import sys
import time

# Must be set before torch initializes its CUDA caching allocator. Real
# training images have wildly varying resolutions/aspect ratios, which
# produces a different activation tensor shape on nearly every micro-batch;
# the default allocator can't reuse freed blocks across those shapes and
# just keeps reserving new ones (measured: max_reserved grew from ~4GB on a
# single fixed-image smoke test to ~7.7GB / 9.75GB-total-GPU after only 5
# accumulation windows of mixed-size images -- see PYTORCH_CUDA_ALLOC_CONF
# docs on expandable_segments, which is exactly the fragmentation this
# addresses). This one line plus the periodic empty_cache() + GPU safety
# check below are what make it safe to run next to the YOLO job.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
from PIL import Image, ImageFile
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen3VLForConditionalGeneration, get_cosine_schedule_with_warmup
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel

ImageFile.LOAD_TRUNCATED_IMAGES = True

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "Smoking_person.v3i.qwen3vl-lora")
OUT_DIR = os.path.join(REPO_ROOT, "runs", "qwen3vl2b_lora")

MODEL_ID = "Qwen/Qwen3-VL-2B-Instruct"

# Kept small on purpose: this is a 10GB card shared with a concurrent YOLO
# training job. 512*28*28 (~401408 px, e.g. ~634x634) is comfortably below
# what the 2B vision tower needs to still recognize a person/cigarette in
# these dataset images, while keeping the visual token count (and therefore
# attention memory) bounded. See README's calibration note in judge_dataset.py
# for the live-serving side (which resizes to width<=768 before sending).
MIN_PIXELS = 64 * 28 * 28
MAX_PIXELS = 256 * 28 * 28

# Same LoRA target-module suffixes as the language model's attention/MLP
# projections (Qwen3VLTextAttention / Qwen3VLTextMLP). The vision tower
# (Qwen3VLVisionAttention/MLP) uses different names ("qkv", "proj",
# "linear_fc1", "linear_fc2") so none of these suffixes can match it --
# verified in build_model() below with an assertion, not just by naming
# convention.
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def gpu_free_mib():
    """Real system-wide free VRAM (not just this process's torch stats) --
    the thing that actually matters for not starving the concurrent YOLO
    job. Returns None if nvidia-smi can't be queried (fails open so a
    transient nvidia-smi hiccup doesn't kill a multi-hour training run)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            timeout=10,
        )
        return int(out.decode().strip().splitlines()[0])
    except Exception:
        return None


MIN_FREE_MIB = 1500  # abort threshold: leave real headroom for the YOLO job / desktop


def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def build_model(load_in_4bit=True, resume_adapter=None):
    log("loading processor...")
    processor = AutoProcessor.from_pretrained(MODEL_ID, min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS)

    kwargs = dict(dtype=torch.bfloat16, device_map={"": 0})
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    log(f"loading base model (4bit={load_in_4bit})...")
    model = Qwen3VLForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
    model.config.use_cache = False

    if load_in_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()

    if resume_adapter and os.path.isdir(resume_adapter):
        log(f"resuming LoRA adapter from {resume_adapter}")
        model = PeftModel.from_pretrained(model, resume_adapter, is_trainable=True)
    else:
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=TARGET_MODULES,
        )
        model = get_peft_model(model, lora_config)

    model.print_trainable_parameters()
    lora_mod_names = [n for n, _ in model.named_modules() if "lora_" in n.lower()]
    vision_lora = [n for n in lora_mod_names if ".visual." in n]
    assert len(vision_lora) == 0, f"LoRA leaked into the (supposedly frozen) vision tower: {vision_lora[:5]}"
    log(f"LoRA attached to {len(lora_mod_names)} submodules, 0 in the vision tower (frozen as intended)")

    return processor, model


def strip_image_tag(text):
    return text.replace("<image>\n", "").replace("<image>", "")


def build_example_inputs(processor, record, data_dir):
    """Resolve the image, render the chat template for the full turn and for
    the prompt-only turn (to find the boundary for label masking), and return
    a dict of tensors ready to feed the model (batch size 1)."""
    img_rel = record["images"][0]
    img_path = os.path.normpath(os.path.join(data_dir, img_rel))
    image = Image.open(img_path).convert("RGB")

    user_text = strip_image_tag(record["messages"][0]["content"])
    assistant_text = record["messages"][1]["content"]

    full_conv = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
        {"role": "assistant", "content": [{"type": "text", "text": assistant_text}]},
    ]
    prompt_conv = [
        {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]},
    ]

    full_inputs = processor.apply_chat_template(
        full_conv, tokenize=True, add_generation_prompt=False, return_dict=True, return_tensors="pt"
    )
    prompt_inputs = processor.apply_chat_template(
        prompt_conv, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
    )

    full_ids = full_inputs["input_ids"][0]
    prompt_len = prompt_inputs["input_ids"].shape[1]
    prompt_len = min(prompt_len, full_ids.shape[0])

    labels = full_ids.clone()
    labels[:prompt_len] = -100

    batch = dict(full_inputs)
    batch["labels"] = labels.unsqueeze(0)
    return batch


def move_to_device(batch, device):
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


@torch.no_grad()
def diagnostic_eval(processor, model, valid_records, data_dir, n_pos=20, n_neg=20, seed=0):
    """Lightweight generation-based sanity check: does the fine-tuned model
    still distinguish positives from negatives, or has it collapsed to
    always/never predicting found=true? Not a full eval harness -- just
    enough signal to flag a regression before spending time on the full
    53-image inference run."""
    rng = random.Random(seed)
    pos = [r for r in valid_records if '"found": true' in r["messages"][1]["content"]]
    neg = [r for r in valid_records if '"found": false' in r["messages"][1]["content"]]
    rng.shuffle(pos)
    rng.shuffle(neg)
    sample = pos[:n_pos] + neg[:n_neg]
    rng.shuffle(sample)

    model.eval()
    model.config.use_cache = True
    device = next(model.parameters()).device
    results = []
    for rec in sample:
        gt_found = '"found": true' in rec["messages"][1]["content"]
        try:
            img_rel = rec["images"][0]
            img_path = os.path.normpath(os.path.join(data_dir, img_rel))
            image = Image.open(img_path).convert("RGB")
            user_text = strip_image_tag(rec["messages"][0]["content"])
            conv = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": user_text}]}]
            inputs = processor.apply_chat_template(
                conv, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
            inputs = move_to_device(inputs, device)
            gen = model.generate(**inputs, max_new_tokens=200, do_sample=False)
            gen_only = gen[0][inputs["input_ids"].shape[1]:]
            text = processor.tokenizer.decode(gen_only, skip_special_tokens=True)
            pred_found = None
            try:
                m = text[text.index("{"):text.rindex("}") + 1]
                parsed = json.loads(m)
                pred_found = bool(parsed.get("found"))
            except Exception:
                pass
            results.append({"image": img_rel, "gt_found": gt_found, "pred_found": pred_found, "raw": text[:300]})
        except Exception as e:
            results.append({"image": rec.get("images"), "gt_found": gt_found, "pred_found": None, "error": str(e)})
    model.config.use_cache = False
    model.train()

    n = len(results)
    n_correct = sum(1 for r in results if r["pred_found"] == r["gt_found"])
    n_pred_true = sum(1 for r in results if r["pred_found"] is True)
    n_pred_none = sum(1 for r in results if r["pred_found"] is None)
    n_gt_true = sum(1 for r in results if r["gt_found"] is True)
    summary = {
        "n": n,
        "accuracy": n_correct / n if n else None,
        "n_gt_true": n_gt_true,
        "n_gt_false": n - n_gt_true,
        "n_pred_true": n_pred_true,
        "n_pred_false_or_parsefail": n - n_pred_true,
        "n_parse_failures": n_pred_none,
    }
    return summary, results


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-steps", type=int, default=400, help="number of optimizer steps (not micro-batches)")
    ap.add_argument("--grad-accum", type=int, default=8, help="micro-batches (size 1) accumulated per optimizer step")
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup-steps", type=int, default=20)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--log-every", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-4bit", action="store_true", help="fall back to plain bf16 LoRA if bitsandbytes is unusable")
    ap.add_argument("--resume-adapter", type=str, default=None)
    ap.add_argument("--skip-eval", action="store_true")
    args = ap.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    processor, model = build_model(load_in_4bit=not args.no_4bit, resume_adapter=args.resume_adapter)
    device = next(model.parameters()).device

    train_records = load_jsonl(os.path.join(DATA_DIR, "train.jsonl"))
    valid_records = load_jsonl(os.path.join(DATA_DIR, "valid.jsonl"))
    log(f"train examples: {len(train_records)}, valid examples: {len(valid_records)}")

    order = list(range(len(train_records)))
    rng = random.Random(args.seed)
    rng.shuffle(order)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    try:
        import bitsandbytes as bnb
        optimizer = bnb.optim.PagedAdamW8bit(trainable_params, lr=args.lr)
        log("using PagedAdamW8bit optimizer")
    except Exception as e:
        log(f"falling back to torch.optim.AdamW ({e})")
        optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    scheduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=args.max_steps
    )

    log_path = os.path.join(OUT_DIR, "train_log.jsonl")
    log_f = open(log_path, "a", encoding="utf-8")

    model.train()
    data_ptr = 0
    step = 0
    t_start = time.time()
    running_loss = 0.0
    running_n = 0
    skipped = 0

    def emergency_stop(free_mib, where):
        log(f"[SAFETY] free VRAM {free_mib}MiB < {MIN_FREE_MIB}MiB threshold ({where}) -- stopping "
            f"training immediately to protect the concurrent GPU job. Saving checkpoint before exit.")
        try:
            ckpt_dir = os.path.join(OUT_DIR, "checkpoint-last")
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
        except Exception as e:
            log(f"[SAFETY] checkpoint save also failed ({e}); exiting anyway")
        log_f.close()
        sys.exit(1)

    while step < args.max_steps:
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0
        micro_done = 0
        for _ in range(args.grad_accum):
            if data_ptr >= len(order):
                rng.shuffle(order)
                data_ptr = 0
            idx = order[data_ptr]
            data_ptr += 1
            rec = train_records[idx]
            try:
                batch = build_example_inputs(processor, rec, DATA_DIR)
                batch = move_to_device(batch, device)
                out = model(**batch)
                loss = out.loss / args.grad_accum
                loss.backward()
                accum_loss += loss.item()
                micro_done += 1
                del batch, out, loss
            except Exception as e:
                skipped += 1
                log(f"[warn] skipping example {rec.get('images')}: {e}")
                continue
            finally:
                # Checked and cleared after EVERY micro-batch, not just every
                # optimizer step: real images vary wildly in resolution, so
                # reserved-but-fragmented memory (and real free VRAM) can
                # swing a lot within a single accumulation window. Measured
                # 2026-09-18: checking/clearing only once per optimizer step
                # (every grad-accum micro-batches) let real system-wide free
                # VRAM crash from ~2.3GB to ~0.4GB *within* one window before
                # the end-of-step check ever ran -- this is the fix.
                torch.cuda.empty_cache()
                free_mib = gpu_free_mib()
                if free_mib is not None and free_mib < MIN_FREE_MIB:
                    emergency_stop(free_mib, "mid-accumulation-window")
        if micro_done == 0:
            continue  # whole accumulation window failed, try again without wasting an optimizer step

        torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()
        step += 1
        running_loss += accum_loss
        running_n += 1

        # End-of-step check too (belt and suspenders around the per-micro-
        # batch one above).
        free_mib = gpu_free_mib()
        if free_mib is not None and free_mib < MIN_FREE_MIB:
            emergency_stop(free_mib, "end-of-step")

        if step % args.log_every == 0:
            avg_loss = running_loss / running_n
            elapsed = time.time() - t_start
            mem_alloc = torch.cuda.max_memory_allocated() / 1024**2
            mem_res = torch.cuda.max_memory_reserved() / 1024**2
            log(f"step {step}/{args.max_steps} loss={avg_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e} "
                f"elapsed={elapsed:.0f}s skipped={skipped} max_alloc={mem_alloc:.0f}MiB max_reserved={mem_res:.0f}MiB "
                f"gpu_free={free_mib}MiB")
            log_f.write(json.dumps({"step": step, "loss": avg_loss, "elapsed_s": elapsed,
                                     "max_alloc_mib": mem_alloc, "max_reserved_mib": mem_res,
                                     "gpu_free_mib": free_mib}) + "\n")
            log_f.flush()
            running_loss = 0.0
            running_n = 0
            torch.cuda.reset_peak_memory_stats()

        if step % args.save_every == 0 or step == args.max_steps:
            ckpt_dir = os.path.join(OUT_DIR, "checkpoint-last")
            model.save_pretrained(ckpt_dir)
            processor.save_pretrained(ckpt_dir)
            log(f"saved checkpoint to {ckpt_dir} (step {step})")

    log_f.close()

    final_dir = os.path.join(OUT_DIR, "adapter")
    model.save_pretrained(final_dir)
    processor.save_pretrained(final_dir)
    log(f"final adapter saved to {final_dir}")

    if not args.skip_eval:
        log("running diagnostic eval on a valid-set sample...")
        summary, results = diagnostic_eval(processor, model, valid_records, DATA_DIR)
        log(f"diagnostic eval summary: {summary}")
        with open(os.path.join(OUT_DIR, "eval_diagnostic.json"), "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "results": results}, f, ensure_ascii=False, indent=2)

    log("DONE")


if __name__ == "__main__":
    main()
