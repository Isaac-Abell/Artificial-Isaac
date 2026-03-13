"""
Train Model (Unsloth Optimized)
===========================
Trains the model using settings from config.py.
Uses standard chat format dataset with configurable chat templates.

Usage:
    python scripts/train_model.py
"""

import sys
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer
from transformers import TrainingArguments
from pathlib import Path

# Import config
from artificial_isaac.config import (
    BASE_MODEL_ID, CHAT_TEMPLATE, DATASET_OUTPUT, MODEL_OUTPUT_DIR,
    MAX_LENGTH, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LORA_R, LORA_ALPHA, EPOCHS, LEARNING_RATE, RANDOM_SEED,
    DATASET_NUM_PROC, PACKING, WARMUP_STEPS, LOGGING_STEPS,
    OPTIMIZER, WEIGHT_DECAY, LR_SCHEDULER_TYPE, TARGET_MODULES, USE_4BIT
)


def main():
    # 1. Load Model
    print(f"Loading {BASE_MODEL_ID}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_LENGTH,
        dtype=None,  # Auto-detects bfloat16
        load_in_4bit=USE_4BIT,
    )

    # 2. Apply Chat Template
    print(f"Applying chat template: {CHAT_TEMPLATE}")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template=CHAT_TEMPLATE,
    )

    # 3. Add LoRA Adapters
    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Saves VRAM
        random_state=RANDOM_SEED,
    )

    # 4. Load & Format Dataset (Standard format)
    print(f"Loading dataset from {DATASET_OUTPUT}...")
    dataset = load_dataset("json", data_files=str(DATASET_OUTPUT), split="train")

    # Apply chat template to format conversations
    def formatting_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            )
            for convo in convos
        ]
        return {"text": texts}

    dataset = dataset.map(formatting_func, batched=True, remove_columns=dataset.column_names)

    # 5. Training
    print("Starting Unsloth Training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_LENGTH,
        dataset_num_proc=DATASET_NUM_PROC,
        packing=PACKING,  # Combines short messages to speed up training
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=WARMUP_STEPS,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=LOGGING_STEPS,
            optim=OPTIMIZER, # Less memory
            weight_decay=WEIGHT_DECAY,
            lr_scheduler_type=LR_SCHEDULER_TYPE,
            seed=RANDOM_SEED,
            output_dir=str(MODEL_OUTPUT_DIR),
            report_to="none",
        ),
    )

    trainer.train()

    print(f"Saving to {MODEL_OUTPUT_DIR}...")
    model.save_pretrained(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
    
    print("Training complete!")


if __name__ == "__main__":
    main()