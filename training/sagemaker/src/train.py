#!/usr/bin/env python3
"""SageMaker training entrypoint for Gemma 4 E2B QLoRA on medical simplification.

Fixes applied after Oracle + Librarian diagnosis of Kaggle v7 NaN divergence:
- FastModel loader (Unsloth patches for Gemma 4 use_cache + fp16 audio mask)
- Explicit audio_tower / vision_tower / multi_modal_projector freeze (text-only task)
- attention_invalid_logits_value clamped to -1e4 (fp16 safety belt-and-suspenders)
- Dataset content converted to Gemma 4 multimodal format (list-of-dicts)
- get_chat_template("gemma-4") to disable thinking mode
- NaN guard callback (fail fast, don't burn 2h on divergence)
- bf16 on A10G/L4 (removes the fundamental fp16 overflow risk)
- Warmup 50 steps + cosine schedule (Oracle recommendation)
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", default=os.environ.get("SM_MODEL_DIR", "/opt/ml/model"))
    parser.add_argument("--training-dir", default=os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training"))
    parser.add_argument("--output-dir", default=os.environ.get("SM_OUTPUT_DATA_DIR", "/opt/ml/output/data"))
    parser.add_argument("--model-id", default="unsloth/gemma-4-E2B-it")
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--num-epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lr-scheduler", default="cosine")
    parser.add_argument("--seed", type=int, default=3407)
    parser.add_argument("--smoke-test-steps", type=int, default=0,
                        help="If >0, cap training at N steps instead of full epochs")
    parser.add_argument("--hf-repo", default=os.environ.get("HF_REPO", ""))
    parser.add_argument("--export-gguf", default="",
                        help="Quantization method for GGUF export (e.g. 'q4_k_m'). Empty = skip.")
    return parser.parse_args()


def banner(msg: str) -> None:
    line = "=" * 60
    print(f"\n{line}\n{msg}\n{line}", flush=True)


def freeze_multimodal(model) -> int:
    """Freeze audio_tower / vision_tower / multi_modal_projector.

    We are doing a text-only task. Keeping these trainable just gives
    the optimizer more ways to blow up on unused parameters.
    """
    frozen = 0
    total = 0
    freeze_tokens = ("audio_tower", "vision_tower", "multi_modal_projector")
    for name, p in model.named_parameters():
        total += 1
        if any(tok in name for tok in freeze_tokens) and p.requires_grad:
            p.requires_grad = False
            frozen += 1
    print(f"Multimodal freeze: {frozen} params frozen out of {total} total", flush=True)
    return frozen


def clamp_attention_invalid_value(model) -> None:
    """Clamp attention_invalid_logits_value from -1e9 to -1e4.

    -1e9 overflows fp16 max (65504). Even on bf16 this has no downside
    because the softmax treats -1e4 as effectively -inf.
    Unsloth v0.1.36-beta patches this for Gemma 4; we set it explicitly
    in case the container didn't pick up the patched loader.
    """
    if hasattr(model.config, "attention_invalid_logits_value"):
        old = model.config.attention_invalid_logits_value
        model.config.attention_invalid_logits_value = -1e4
        print(f"Clamped attention_invalid_logits_value: {old} -> -1e4", flush=True)


def to_multimodal_content(example: dict) -> dict:
    """Convert plain-string content to Gemma 4 multimodal list-of-dicts.

    Gemma 4's processor.apply_chat_template requires content as
    [{"type": "text", "text": "..."}] for user and model roles.
    System role stays a string because the chat template concatenates
    it with '\n\n' (list + str is a TypeError).
    """
    new_messages = []
    for msg in example["messages"]:
        content = msg["content"]
        if msg["role"] == "system":
            new_messages.append({"role": msg["role"], "content": content})
            continue
        if isinstance(content, str):
            content = [{"type": "text", "text": content}]
        new_messages.append({"role": msg["role"], "content": content})
    return {"messages": new_messages}


def install_nan_guard(trainer_cls):
    """Build a TrainerCallback class that stops training on NaN/Inf in logs."""
    from transformers import TrainerCallback

    class NaNGuard(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kw):
            if not logs:
                return
            for key, value in logs.items():
                if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
                    print(f"\n*** NaN/Inf in '{key}' = {value} at step {state.global_step}. Stopping. ***\n", flush=True)
                    control.should_training_stop = True
                    return

    return NaNGuard


def main() -> int:
    args = parse_args()

    banner("CogniTrace Gemma 4 E2B Medical QLoRA on SageMaker")
    for k, v in sorted(vars(args).items()):
        print(f"  {k}: {v}")

    import torch

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available.", file=sys.stderr)
        return 1

    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    bf16_supported = torch.cuda.is_bf16_supported()
    print(f"\nGPU: {gpu_name} ({gpu_mem_gb:.1f} GB), bf16={bf16_supported}", flush=True)
    if not bf16_supported:
        print("WARNING: bf16 NOT supported on this GPU. Falling back to fp16, NaN risk increases.", flush=True)

    banner("Loading model via Unsloth FastModel")
    try:
        from unsloth import FastModel
        loader = FastModel
        loader_name = "FastModel"
    except ImportError:
        from unsloth import FastVisionModel
        loader = FastVisionModel
        loader_name = "FastVisionModel"
    print(f"Using loader: {loader_name}", flush=True)

    model, tokenizer = loader.from_pretrained(
        model_name=args.model_id,
        max_seq_length=args.max_seq_length,
        load_in_4bit=True,
        dtype=None,
    )

    clamp_attention_invalid_value(model)

    banner("Applying LoRA adapters")
    try:
        model = loader.get_peft_model(
            model,
            finetune_vision_layers=False,
            finetune_language_layers=True,
            finetune_attention_modules=True,
            finetune_mlp_modules=True,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )
    except TypeError:
        model = loader.get_peft_model(
            model,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            use_gradient_checkpointing="unsloth",
            random_state=args.seed,
        )

    freeze_multimodal(model)

    try:
        from unsloth.chat_templates import get_chat_template
        tokenizer = get_chat_template(tokenizer, chat_template="gemma-4")
        print("Applied chat_template='gemma-4' (no thinking mode)", flush=True)
    except Exception as exc:
        print(f"WARNING: get_chat_template('gemma-4') failed ({exc!r}); using tokenizer default", flush=True)

    banner("Loading dataset")
    data_files = sorted(str(p) for p in Path(args.training_dir).glob("*.jsonl"))
    if not data_files:
        print(f"ERROR: no .jsonl files in {args.training_dir}", file=sys.stderr)
        return 1
    print(f"Data files: {data_files}", flush=True)

    from datasets import load_dataset
    raw = load_dataset("json", data_files=data_files, split="train")
    print(f"Raw pairs: {len(raw)}", flush=True)
    print(f"Sample raw:\n{json.dumps(raw[0], indent=2)[:800]}", flush=True)

    def format_example(example):
        messages = []
        for msg in example["messages"]:
            content = msg["content"]
            if msg["role"] != "system" and isinstance(content, str):
                content = [{"type": "text", "text": content}]
            messages.append({"role": msg["role"], "content": content})
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = raw.map(format_example, remove_columns=[c for c in raw.column_names if c != "messages"])
    print(f"\nFormatted sample text:\n{dataset[0]['text'][:800]}\n", flush=True)

    banner("SFTTrainer setup")
    from trl import SFTTrainer, SFTConfig

    NaNGuard = install_nan_guard(SFTTrainer)

    effective_max_steps = args.smoke_test_steps if args.smoke_test_steps > 0 else -1
    effective_epochs = 1 if args.smoke_test_steps > 0 else args.num_epochs

    sft_args = SFTConfig(
        output_dir="/opt/ml/output/data/checkpoints",
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum_steps,
        warmup_steps=args.warmup_steps,
        num_train_epochs=effective_epochs,
        max_steps=effective_max_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        logging_steps=10,
        save_steps=500,
        save_total_limit=2,
        optim="adamw_8bit",
        lr_scheduler_type=args.lr_scheduler,
        seed=args.seed,
        bf16=bf16_supported,
        fp16=not bf16_supported,
        report_to="none",
        packing=False,
        remove_unused_columns=False,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=sft_args,
        callbacks=[NaNGuard()],
    )

    banner(f"Starting training (smoke_test_steps={args.smoke_test_steps})")
    stats = trainer.train()

    print(f"\nFinal loss: {stats.training_loss:.4f}")
    print(f"Total steps: {stats.global_step}")

    if math.isnan(stats.training_loss) or math.isinf(stats.training_loss):
        print("*** Training diverged to NaN/Inf. NOT saving adapter. Exiting 1. ***", flush=True)
        return 1

    banner(f"Saving LoRA adapter to {args.model_dir}")
    os.makedirs(args.model_dir, exist_ok=True)
    model.save_pretrained(args.model_dir)
    tokenizer.save_pretrained(args.model_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    state_dump = Path(args.output_dir) / "trainer_state_summary.json"
    state_dump.write_text(json.dumps({
        "final_loss": float(stats.training_loss),
        "global_step": int(stats.global_step),
        "num_epochs": effective_epochs,
        "smoke_test_steps": args.smoke_test_steps,
    }, indent=2))
    print(f"Wrote summary to {state_dump}", flush=True)

    hf_token = os.environ.get("HF_TOKEN", "")
    if hf_token and args.hf_repo:
        banner(f"Pushing to HuggingFace: {args.hf_repo}")
        try:
            model.push_to_hub(args.hf_repo, token=hf_token)
            tokenizer.push_to_hub(args.hf_repo, token=hf_token)
            print(f"Pushed: https://huggingface.co/{args.hf_repo}", flush=True)
        except Exception as exc:
            print(f"HF push failed (non-fatal): {exc!r}", flush=True)
    else:
        print(f"\nSkipping HF push. HF_TOKEN set: {bool(hf_token)}, hf_repo: {args.hf_repo or '<unset>'}", flush=True)

    if args.export_gguf:
        banner(f"Exporting merged GGUF: {args.export_gguf}")
        try:
            model.save_pretrained_gguf(
                args.model_dir,
                tokenizer,
                quantization_method=args.export_gguf.lower(),
            )
            gguf_files = sorted(Path(args.model_dir).glob("*.gguf"))
            for gf in gguf_files:
                size_mb = gf.stat().st_size / (1024 ** 2)
                print(f"GGUF written: {gf.name} ({size_mb:.1f} MB)", flush=True)
        except Exception as exc:
            print(f"GGUF export FAILED (non-fatal): {exc!r}", flush=True)
            print("LoRA adapter still saved; export GGUF separately.", flush=True)

    banner("Training complete")
    return 0


if __name__ == "__main__":
    sys.exit(main())
