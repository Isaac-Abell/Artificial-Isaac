"""
Train Model on Modal (Cloud GPU)
=================================
Fine-tune using Modal's cloud GPUs. Use this if you don't have a local GPU
or want to use a larger model (e.g. 32B on an A100/H100).

Prerequisites:
    1. pip install modal
    2. modal setup          # One-time auth
    3. Place training_data/dataset.jsonl in the project root

Usage:
    # Upload data and train (default: Qwen3-14B on L40S)
    modal run scripts/train_model_modal.py

    # Use a bigger model on A100 (Single GPU recommended for Unsloth)
    modal run scripts/train_model_modal.py --model "unsloth/Qwen3-32B-unsloth-bnb-4bit" --gpu "A100-80GB"

    # Download the finetuned model after training
    modal volume get artificial-you-vol finetuned_model/ ./finetuned_model/
"""

import modal
import os

# ──────────────────────── Config ────────────────────────

APP_NAME = "artificial-you-train"
VOLUME_NAME = "artificial-you-vol"

# Defaults (can be overridden via CLI args)
DEFAULT_MODEL = "unsloth/Qwen3-14B-unsloth-bnb-4bit"
DEFAULT_GPU = "A100-80GB"  # Options: "T4", "L40S", "A100", "A100-80GB", "H100"
DEFAULT_GPU_COUNT = 1 # Unsloth is optimized for single-GPU training.
DEFAULT_EPOCHS = 3

# Training hyperparameters (mirrors config.py)
CHAT_TEMPLATE = "chatml"
MAX_LENGTH = 2048
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
LORA_R = 64
LORA_ALPHA = 128
LORA_DROPOUT = 0
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10
OPTIMIZER = "adamw_8bit"
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"
PACKING = True
RANDOM_SEED = 67
LOGGING_STEPS = 1
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ──────────────────────── Modal Setup ────────────────────────

app = modal.App(APP_NAME)

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.4.0",
        "unsloth[cu124-torch250] @ git+https://github.com/unslothai/unsloth.git",
        "unsloth_zoo",
        "transformers>=4.51.0",
        "datasets",
        "accelerate",
        "peft",
        "bitsandbytes",
        "trl",
        "triton",
    )
)

# Persistent volume for model weights and data
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)
VOL_PATH = "/vol"

# ──────────────────────── Upload Data ────────────────────────

@app.function(image=image, volumes={VOL_PATH: vol}, timeout=300)
def upload_data(dataset_bytes: bytes):
    """Upload training dataset to the Modal volume."""
    import os
    os.makedirs(f"{VOL_PATH}/training_data", exist_ok=True)
    with open(f"{VOL_PATH}/training_data/dataset.jsonl", "wb") as f:
        f.write(dataset_bytes)
    vol.commit()
    print(f"✓ Uploaded dataset ({len(dataset_bytes):,} bytes)")


# ──────────────────────── Training ────────────────────────

@app.function(
    image=image,
    gpu=DEFAULT_GPU,
    volumes={VOL_PATH: vol},
    timeout=7200,  # 2 hours max
)
def train(
    model_id: str = DEFAULT_MODEL,
    epochs: int = DEFAULT_EPOCHS,
    gpu_count: int = DEFAULT_GPU_COUNT,
):
    """Fine-tune the model on Modal's cloud GPU."""
    import torch
    from unsloth import FastLanguageModel, is_bfloat16_supported
    from unsloth.chat_templates import get_chat_template
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import load_dataset

    if gpu_count > 1:
        print(f"⚠️ Warning: Unsloth is optimized for SINGLE-GPU training.")
        print(f"   Using {gpu_count} GPUs might be slower or more unstable for LoRA.")

    output_dir = f"{VOL_PATH}/finetuned_model"
    dataset_path = f"{VOL_PATH}/training_data/dataset.jsonl"

    print(f"🧠 Model: {model_id}")
    print(f"📊 Dataset: {dataset_path}")
    print(f"🎯 Epochs: {epochs}")
    print(f"💾 Output: {output_dir}")
    print(f"🖥️ GPU: {torch.cuda.get_device_name(0)}")
    print(f"📦 VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
    print()

    # 1. Load model
    print(f"Loading {model_id}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_id,
        max_seq_length=MAX_LENGTH,
        dtype=None,
        load_in_4bit=True,
    )

    # 2. Apply chat template
    print(f"Applying chat template: {CHAT_TEMPLATE}")
    tokenizer = get_chat_template(tokenizer, chat_template=CHAT_TEMPLATE)

    # 3. Add LoRA adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=RANDOM_SEED,
    )

    # 4. Load dataset
    print(f"Loading dataset from {dataset_path}...")
    dataset = load_dataset("json", data_files=dataset_path, split="train")

    def formatting_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo, tokenize=False, add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)

    # 5. Train
    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_LENGTH,
        dataset_num_proc=2,
        packing=PACKING,
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=epochs,
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=LOGGING_STEPS,
            optim=OPTIMIZER,
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            seed=RANDOM_SEED,
            output_dir=output_dir,
            report_to="none",
        ),
    )

    trainer.train()

    # 6. Save
    print(f"Saving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    vol.commit()

    print("✅ Training complete! Model saved to Modal volume.")
    print(f"   Download with: modal volume get {VOLUME_NAME} finetuned_model/ ./finetuned_model/")


# ──────────────────────── Entrypoint ────────────────────────

@app.local_entrypoint()
def main(
    model: str = DEFAULT_MODEL,
    gpu: str = DEFAULT_GPU,
    gpu_count: int = DEFAULT_GPU_COUNT,
    epochs: int = DEFAULT_EPOCHS,
):
    """Upload data and train on Modal."""
    from pathlib import Path

    dataset_path = Path("training_data/dataset.jsonl")
    if not dataset_path.exists():
        print(f"✗ Dataset not found at {dataset_path}")
        print("  Run 'python scripts/preprocess_data.py' first.")
        return

    # Upload dataset
    print(f"📤 Uploading {dataset_path}...")
    data = dataset_path.read_bytes()
    upload_data.remote(data)

    # Train
    print(f"\n🚀 Starting training on Modal ({gpu} x{gpu_count})...")
    print(f"   Model: {model}")
    print(f"   Epochs: {epochs}")
    print()
    train.remote(model_id=model, epochs=epochs, gpu_count=gpu_count)

    print(f"\n✅ Done! Download your model with:")
    print(f"   modal volume get {VOLUME_NAME} finetuned_model/ ./finetuned_model/")
