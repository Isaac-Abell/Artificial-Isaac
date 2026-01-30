"""
Train Model (Unsloth Optimized)
===========================
Trains the model using settings from config.py.
Uses ShareGPT format dataset with configurable chat templates.

Usage:
    python scripts/train_model.py
"""

import sys
import torch
from datasets import load_dataset
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template, standardize_sharegpt
from trl import SFTTrainer
from transformers import TrainingArguments
from pathlib import Path

# Import config
sys.path.append(str(Path(__file__).parent.parent))
from config import (
    BASE_MODEL_ID, CHAT_TEMPLATE, DATASET_OUTPUT, MODEL_OUTPUT_DIR,
    MAX_LENGTH, BATCH_SIZE, GRADIENT_ACCUMULATION_STEPS,
    LORA_R, LORA_ALPHA, EPOCHS, LEARNING_RATE
)


def main():
    # 1. Load Model
    print(f"Loading {BASE_MODEL_ID}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL_ID,
        max_seq_length=MAX_LENGTH,
        dtype=None,  # Auto-detects bfloat16
        load_in_4bit=True,
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
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        lora_alpha=LORA_ALPHA,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Saves massive VRAM
        random_state=3407,
    )

    # 4. Load & Format Dataset (ShareGPT format)
    print(f"Loading dataset from {DATASET_OUTPUT}...")
    dataset = load_dataset("json", data_files=str(DATASET_OUTPUT), split="train")
    
    # Standardize from ShareGPT format (from/value -> role/content)
    dataset = standardize_sharegpt(dataset)

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
        dataset_num_proc=2,
        packing=True,  # Combines short messages to speed up training 5x
        args=TrainingArguments(
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            warmup_steps=50,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            fp16=not is_bfloat16_supported(),
            bf16=is_bfloat16_supported(),
            logging_steps=10,
            optim="adamw_8bit",  # Uses less memory
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir=str(MODEL_OUTPUT_DIR),
            report_to="none",
        ),
    )

    trainer_stats = trainer.train()

    print(f"Saving to {MODEL_OUTPUT_DIR}...")
    model.save_pretrained(str(MODEL_OUTPUT_DIR))
    tokenizer.save_pretrained(str(MODEL_OUTPUT_DIR))
    
    print("Training complete!")


if __name__ == "__main__":
    main()