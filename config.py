"""
Global configuration for AI: Artificial Isaac
Edit these values to customize your training pipeline.
"""

from pathlib import Path

# ==============================
# IDENTITY & PATHS
# ==============================

# Replace with your own name as it appears in WhatsApp/Instagram
CHAT_OWNER = "Isaac Abell"

# Data paths
DATA_DIR = Path("data")
WHATSAPP_DIR = DATA_DIR / "whatsapp"
INSTAGRAM_DIR = DATA_DIR / "instagram" / "inbox"

# Output path (single unified dataset in ShareGPT format)
TRAINING_DATA_DIR = Path("training_data")
DATASET_OUTPUT = TRAINING_DATA_DIR / "dataset.jsonl"

# Model paths
MODEL_OUTPUT_DIR = Path("finetuned_model")
CHROMA_DB_DIR = Path("chroma_db")
RAG_DATA_DIR = Path("rag_data")
RAG_DATA_FILE = RAG_DATA_DIR / "biography.json"

# ==============================
# DATA PROCESSING
# ==============================

# Conversation grouping
SAME_CONVO_THRESHOLD_SECONDS = 1800  # 30 minutes - start new conversation
SAME_USER_THRESHOLD_SECONDS = 600    # 10 minutes - merge consecutive messages

# Token limits
HISTORY_MAX_TOKENS = 3000  # Maximum tokens per conversation
CONVO_MIN_TOKENS = 75      # Minimum tokens to include conversation

# ==============================
# MODEL CONFIGURATION
# ==============================

# Base model (Unsloth 4-bit quantized models recommended)
BASE_MODEL_ID = "unsloth/Qwen3-14B-unsloth-bnb-4bit"

# Tokenizer for preprocessing (should match base model family)
TOKENIZER_ID = "unsloth/Qwen3-14B-unsloth-bnb-4bit"

# Chat template for training (unsloth.chat_templates)
# Options: "chatml", "llama-3", "mistral", "gemma", "phi-3", "zephyr", "alpaca"
CHAT_TEMPLATE = "chatml" 

# ==============================
# TRAINING SETTINGS
# ==============================
RANDOM_SEED = 67
MAX_LENGTH = 2048
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8  # Effective batch size = 8 (smoother training)

# LoRA Optimization
LORA_R = 64            # Higher rank = more expressive adaptation
LORA_ALPHA = 128       # 2x LORA_R is the standard ratio
LORA_DROPOUT = 0       # Unsloth recommends 0 (uses its own regularization)

# Training Mechanics
EPOCHS = 3
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10      # Slightly more warmup for stability
LOGGING_STEPS = 1
OPTIMIZER = "adamw_8bit"
WEIGHT_DECAY = 0.01
LR_SCHEDULER_TYPE = "cosine"  # Cosine annealing > linear for personality fine-tuning

DATASET_NUM_PROC = 2
PACKING = True         # Faster training, fills GPU better
USE_BF16 = True            
USE_4BIT = True
QUANT_TYPE = "nf4"

# LoRA Target Modules (all attention + MLP = best style capture)
TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ==============================
# RAG CONFIGURATION
# ==============================

RAG_COLLECTION_NAME = "rag_data"
MAX_RAG_CONTEXT_TOKENS = 1024  # Max tokens for retrieved contexts
RAG_N_RESULTS = 3          # Number of contexts to retrieve

# ==============================
# INFERENCE
# ==============================

INFERENCE_MAX_TOKENS = 512
INFERENCE_TEMPERATURE = 0.7
INFERENCE_TOP_P = 0.9

# ==============================
# LOGGING
# ==============================

LOG_DIR = Path("logs")
RESULTS_DIR = Path("results")

# Create directories
for directory in [
    DATA_DIR, WHATSAPP_DIR, INSTAGRAM_DIR,
    TRAINING_DATA_DIR, MODEL_OUTPUT_DIR,
    CHROMA_DB_DIR, RAG_DATA_DIR,
    LOG_DIR, RESULTS_DIR
]:
    directory.mkdir(parents=True, exist_ok=True)