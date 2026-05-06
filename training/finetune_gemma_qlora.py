#!/usr/bin/env python3
"""Fine-tune Gemma 4 E2B on medical communication with Unsloth QLoRA.

Can be run as a standalone script or copy-pasted cell-by-cell into a Kaggle notebook.
Targets Kaggle free GPU tier (T4 or P100, ~15 GB VRAM).

Usage:
    python finetune_gemma_qlora.py
    python finetune_gemma_qlora.py --data training/data/medical_communication.jsonl
    python finetune_gemma_qlora.py --output outputs/lora_model --epochs 3
"""

import argparse
import os

# --- CONFIG (edit here or override via CLI flags) ---
MODEL_ID = "unsloth/gemma-4-E2B-it"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# LoRA
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Training
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
WARMUP_STEPS = 5
WEIGHT_DECAY = 0.01
OPTIMIZER = "adamw_8bit"
LR_SCHEDULER = "linear"
SEED = 3407

# Paths
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "medical_communication.jsonl")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", "lora_model")


# =============================================================================
# CELL 1: Install dependencies (Kaggle: uncomment, local: skip if already installed)
# =============================================================================

# Uncomment in Kaggle:
# import subprocess
# subprocess.run(["pip", "install", "unsloth[colab-new]", "trl", "datasets"], check=True)


# =============================================================================
# CELL 2: Load model with Unsloth
# =============================================================================

def load_model(model_id: str, max_seq_length: int, load_in_4bit: bool):
    """Load base model with Unsloth + 4-bit QLoRA."""
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=max_seq_length,
        dtype=None,           # auto-detect
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


# =============================================================================
# CELL 3: Add LoRA adapters
# =============================================================================

def add_lora(model):
    """Attach LoRA adapters with the training config."""
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth-patched, avoids use_cache bug
        random_state=SEED,
        use_rslora=False,
        loftq_config=None,
    )
    return model


# =============================================================================
# CELL 4: Load and format dataset
# =============================================================================

def load_dataset(data_path: str, tokenizer):
    """Load JSONL pairs and apply Gemma chat template."""
    from datasets import load_dataset as hf_load_dataset

    raw = hf_load_dataset("json", data_files=data_path, split="train")

    def format_example(example):
        # Each example has {"messages": [{"role": "user", ...}, {"role": "model", ...}]}
        messages = example["messages"]
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    dataset = raw.map(format_example, num_proc=1)
    print(f"Loaded {len(dataset)} training pairs from {data_path}")
    return dataset


# =============================================================================
# CELL 5: Train
# =============================================================================

def train(model, tokenizer, dataset, output_dir: str, num_epochs: int):
    """Run SFTTrainer with the configured hyperparameters."""
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from unsloth import is_bfloat16_supported

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,
        packing=False,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM_STEPS,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=num_epochs,
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim=OPTIMIZER,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER,
            seed=SEED,
            output_dir=output_dir,
            report_to="none",
        ),
    )

    print("Starting training...")
    trainer_stats = trainer.train()

    final_loss = trainer_stats.training_loss
    print(f"\nTraining complete.")
    print(f"Final training loss: {final_loss:.4f}")
    print(f"Total steps: {trainer_stats.global_step}")
    return trainer_stats


# =============================================================================
# CELL 6: Save LoRA adapter + tokenizer
# =============================================================================

def save_model(model, tokenizer, output_dir: str):
    """Save LoRA weights and tokenizer for export."""
    os.makedirs(output_dir, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"LoRA adapter saved to: {output_dir}")


# =============================================================================
# MAIN
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(description="QLoRA fine-tune Gemma 4 E2B on medical communication")
    parser.add_argument("--data", default=DATA_PATH, help="Path to training JSONL")
    parser.add_argument("--output", default=OUTPUT_DIR, help="Directory for LoRA adapter output")
    parser.add_argument("--epochs", type=int, default=NUM_EPOCHS, help="Number of training epochs")
    parser.add_argument("--model", default=MODEL_ID, help="Base model ID")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Training data not found: {args.data}\n"
            "Run curate_medical_data.py and format_instruction_pairs.py first."
        )

    print(f"Model:    {args.model}")
    print(f"Data:     {args.data}")
    print(f"Output:   {args.output}")
    print(f"Epochs:   {args.epochs}")
    print(f"LoRA r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGET_MODULES}")
    print()

    model, tokenizer = load_model(args.model, MAX_SEQ_LENGTH, LOAD_IN_4BIT)
    model = add_lora(model)
    dataset = load_dataset(args.data, tokenizer)
    train(model, tokenizer, dataset, args.output, args.epochs)
    save_model(model, tokenizer, args.output)


if __name__ == "__main__":
    main()
