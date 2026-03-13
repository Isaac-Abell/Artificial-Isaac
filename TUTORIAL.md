# Complete Tutorial: Train Your Personal AI

This guide walks you through every step of creating your personal AI chatbot.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [Environment Setup](#environment-setup)
3. [Data Collection](#data-collection)
4. [Data Processing](#data-processing)
5. [RAG Data Entry](#rag-data-entry)
6. [RAG Setup](#rag-setup)
7. [Model Training (Local)](#model-training-local)
8. [Model Training (Cloud - Modal)](#model-training-cloud---modal)
9. [Testing](#testing)
10. [Inference](#inference)
11. [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Hardware Requirements

| GPU VRAM | Recommended Maximum Number of Model Parameters | Recommended Model (as of March 2026) | Training Time (est. for 150k tokens) |
| --- | --- | --- | --- |
| **8GB** | 4B | `unsloth/Qwen3-4B` | ~15-20 mins |
| **12–16GB** | 8B | `unsloth/Qwen3-8B-unsloth-bnb-4bit` | ~20-25 mins |
| **24-32GB** | 14B | `unsloth/Qwen3-14B-unsloth-bnb-4bit` | ~25-30 mins |

### Software Requirements

- **Python 3.11** (tested; other versions may work)
- **CUDA-capable NVIDIA GPU** with appropriate drivers
- **Git**

### Check Your Setup

```bash
# Check CUDA version
nvidia-smi

# Check Python version
python --version
```

---

## Environment Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Isaac-Abell/Artificial-Isaac.git
cd Artificial-Isaac
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/macOS:**
```bash
python -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

> ⚠️ **Warning** Order matters! Install PyTorch with CUDA *first*, then everything else. If you install the project dependencies first, pip will pull in the CPU-only version of PyTorch and the CUDA install will skip with "Requirement already satisfied."

```bash
# Step 1: Upgrade pip
pip install --upgrade pip

# Step 2: Install PyTorch with CUDA FIRST
# Visit https://pytorch.org/get-started/locally/ for your specific CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130

# Step 3: Install Triton
# Windows:
pip install triton-windows
# Linux:
pip install triton

# Step 4: Install Unsloth (must match your PyTorch + CUDA version)
pip install "unsloth[cu130-torch210] @ git+https://github.com/unslothai/unsloth.git"

# Step 5: Install project in editable mode
pip install -e .
```

### 4. Verify Installation

```bash
python -c "
import torch
import transformers
import chromadb
print(f'✓ PyTorch {torch.__version__}')
print(f'✓ CUDA: {torch.version.cuda}')
print(f'✓ GPU available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'✓ GPU: {torch.cuda.get_device_name(0)}')
print(f'✓ Transformers {transformers.__version__}')
"
```

You should see `GPU available: True`. If it says `False`, you likely installed the CPU-only torch — see [Troubleshooting](#pytorch-shows-cuda-false).

---

## Data Collection

### WhatsApp Export

1. Open WhatsApp on your phone
2. Select a 1-on-1 chat (not group chats)
3. Tap **⋮** → **More** → **Export chat**
4. Choose **"Without Media"**
5. Save the `.txt` file to `data/whatsapp/`
6. Repeat for all chats you want

**Tips:**
- More data = better results (aim for 5+ active chats)
- Quality over quantity — export chats where you're most active

### Instagram Export

1. Go to **Instagram** → **Settings** → **Privacy and security**
2. Click **Download your information**
3. **Format**: Select **JSON** (not HTML!)
4. **Date range**: All time
5. Wait for email (24–48 hours), download the ZIP
6. Extract and copy the `inbox/` contents to `data/instagram/inbox/`

---

## Data Processing

### 1. Configure Your Name

Edit `config.py`:

```python
CHAT_OWNER = "Your Full Name"  # Exactly as it appears in your chats
```

### 2. Run the Preprocessor

```bash
python scripts/preprocess_data.py
```

**Expected output:**
```
======================================================================
Unified Data Preprocessor (Hugging Face Format)
======================================================================

Chat owner: Your Name
Output: training_data\dataset.jsonl

Loading tokenizer...

Processing 4 WhatsApp file(s)...
  ✓ WhatsApp: 64 conversations
Processing 93 Instagram conversation(s)...
  ✓ Instagram: 562 conversations

======================================================================
Statistics:
======================================================================
  Total conversations:  626
  Total messages:       4,832
  Avg messages/convo:   7.7
  Total tokens:         187,432
  Avg tokens/convo:     299
======================================================================
```

---

## RAG Data Entry

The RAG (Retrieval-Augmented Generation) system gives your AI factual knowledge about you — things it can't learn from chat patterns alone (your name, job, hobbies, etc.).

### 1. Open the Survey Tool

Open `tools/rag_survey.html` in your browser. You can use:
- VS Code's "Open with Live Server" extension
- Any "View in Browser" extension
- Or just double-click the file

### 2. Answer Questions

The tool contains ~60 questions organized by category:
- Identity & Biography
- Professional & Career
- Skills & Expertise
- Interests & Hobbies
- Personality & Opinions
- Travel & Experiences

Only answered questions are included in the export.

### 3. Save Your Answers

Click **Save JSON** — your browser will download a `biography.json` file.

Move it to the project:
```bash
mkdir rag_data
move %USERPROFILE%\Downloads\biography.json rag_data\biography.json
```

### 4. Edit Later

To update answers, click **Load JSON**, select your existing `biography.json`, edit, and save again.

### Data Format

The file is a flat JSON array:
```json
[
  {
    "question": "What is your full name?",
    "answer": "Isaac Abell"
  },
  {
    "question": "Where did you grow up?",
    "answer": "I grew up in Seattle, WA. I loved the rain but hated the traffic."
  }
]
```

> **Note:** The questionnaire covers basic information, but adding your own custom questions and answers (either by tweaking the HTML or editing the JSON directly) builds a larger knowledge base for the RAG pipeline. This helps the AI actually knows what you know.

---

## RAG Setup

Index your biography data into ChromaDB:

```bash
python scripts/setup_rag.py
```

**Expected output:**
```
============================================================
RAG Setup — Indexing Q&A Data
============================================================

Data file: rag_data\biography.json
ChromaDB:  chroma_db

✓ Collection cleared
✓ Indexed 12 Q&A pairs from biography.json

✓ Total indexed chunks: 12
```

---

## Model Training (Local)

> **Note:** The default parameters in `config.py` are set for an RTX 5090 (32GB VRAM). If you are using a different GPU, review and adjust the configuration values (such as model size, batch size, and sequence length) to match your hardware. For a detailed explanation of all training parameters and troubleshooting memory errors, see **[CONFIGURATION_GUIDE.md](./CONFIGURATION_GUIDE.md)**.

### Choose Your Model

Edit `BASE_MODEL_ID` and `TOKENIZER_ID` in `config.py`:

| GPU VRAM | Model |
|----------|-------|
| 8GB | `unsloth/Qwen3-4B` |
| 12–16GB | `unsloth/Qwen3-8B-unsloth-bnb-4bit` |
| 24–32GB | `unsloth/Qwen3-14B-unsloth-bnb-4bit` |

### Start Training

```bash
python scripts/train_model.py
```

**What you'll see:**
```
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
Loading unsloth/Qwen3-14B-unsloth-bnb-4bit...
==((====))==  Unsloth 2026.3.4: Fast Qwen3 patching.
   \\   /|    NVIDIA GeForce RTX 5090. Max memory: 31.842 GB.
O^O/ \_/ \    Torch: 2.10.0+cu130. CUDA: 12.0. Triton: 3.6.0
\        /    Bfloat16 = TRUE.
 "-____-"     Free license: http://github.com/unslothai/unsloth

Trainable parameters = 159,383,552 of 14,770,033,664 (1.08% trained)
 25%|██▌       | 59/236 [06:00<18:00, 1.5s/it, loss=2.34]
```

---

## Model Training (Cloud - Modal)

If you don't have a local GPU or want to train a massive model (like 70B or 100B+), use the Modal script.

### 1. Simple Setup

```bash
pip install modal
modal setup
```

### 2. Run Cloud Training

```bash
# Deploys to Modal, uploads your data, and trains your configured GPU
modal run scripts/train_model_modal.py
```

### 3. Customizing Resources

You can specify the model and GPU type via command line:

```bash
# Train Qwen3-32B on an A100 80GB
modal run scripts/train_model_modal.py --model "unsloth/Qwen3-32B-unsloth-bnb-4bit" --gpu "A100-80GB"
```

### 4. Download Your Model

Once training is complete, download the adapter weights to your local machine:

```bash
modal volume get artificial-you-vol finetuned_model/ ./finetuned_model/
```

> **Note**: Unsloth is highly optimized for **single-GPU** training. Multi-GPU training for LoRA often introduces overhead that makes it slower or more unstable. For the best performance on Modal, we recommend using a single powerful GPU (like the L40S or A100-80GB) rather than multiple smaller ones.

### Which Epoch to Use?

| Epoch | Characteristics | Notes |
|-------|----------------|-------|
| 2–3 | More conservative, generalized | Best for stability |
| 3–4 | **Balanced personality + safety** | **Recommended** |
| 5+ | Strong personality, risk of overfitting | Use cautiously |

### Monitor GPU Usage

In another terminal:
```bash
nvidia-smi -l 1
```

---

## Testing

Run the test suite to verify your pipeline:

```bash
# Data processing + RAG tests (no GPU needed)
pytest tests/test_data_processing.py tests/test_rag.py -v

# Tokenizer tests (downloads tokenizer on first run)
pytest tests/test_tokenization.py -v

# All tests
pytest tests/ -v
```

### Test Coverage

| Test File | What It Tests |
|-----------|---------------|
| `test_data_processing.py` | JSON loading, validation, edge cases |
| `test_rag.py` | ChromaDB indexing, querying, context formatting |
| `test_tokenization.py` | Tokenizer loading, encode/decode, special chars |

---

## Inference

Start an interactive chat with your fine-tuned model:

```bash
python scripts/inference.py
```

```
================================================================================
INTERACTIVE TESTING MODE
Commands:
  - Type your question/prompt and press Enter
  - Type 'quit' or 'exit' to end
  - Type 'clear' to clear conversation history
================================================================================

You: What is your name
Assistant: Isaac

You: Do you mountain bike
Assistant: Yes

You: quit
Bye!
```

The inference script automatically retrieves relevant RAG context for each query.

---

## Troubleshooting

### PyTorch Shows CUDA False

**Cause:** CPU-only PyTorch was installed (usually because dependencies were installed before the CUDA version).

**Fix:**
```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

### Out of Memory (OOM) Errors

**Cause:** Model too large for your GPU.

**Solutions (in order):**
1. Switch to a smaller model (see GPU table above)
2. Reduce `MAX_LENGTH` in `config.py` (try 1024)
3. Reduce `LORA_R` (try 32 or 16)
4. Reduce `TARGET_MODULES` to `["q_proj", "v_proj"]`
5. Set `PACKING = False`

### Triton Import Error

**Windows:**
```bash
pip install triton-windows
```

**Linux:**
```bash
pip install triton
```

### Unsloth Import Error

```bash
pip install unsloth_zoo
```

Or for the full install with CUDA matching:
```bash
pip install "unsloth[cu130-torch210] @ git+https://github.com/unslothai/unsloth.git"
```

### Model Not Found

Make sure your `BASE_MODEL_ID` is a valid Hugging Face model. Common correct names:
- `unsloth/Qwen3-4B`
- `unsloth/Qwen3-8B-unsloth-bnb-4bit`
- `unsloth/Qwen3-14B-unsloth-bnb-4bit`

### Model Sounds Too Robotic

- Train for more epochs (try 4–5)
- Add more diverse training data (1000+ conversations)
- Check that `CHAT_OWNER` in `config.py` matches your name exactly

### Model Repeats Training Data (Overfitting)

- Use an earlier checkpoint (epoch 2–3)
- Reduce `EPOCHS` to 2–3
- Increase `LORA_DROPOUT` to 0.05

### ChromaDB Errors

```bash
# Clear database and re-index
rmdir /s /q chroma_db
python scripts/setup_rag.py
```

---

## Complete Command Sequence

For reference, here's the entire pipeline in order:

```bash
# 1. Setup
python -m venv venv
venv\Scripts\activate                    # Windows
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
pip install triton-windows               # or: pip install triton (Linux)
pip install "unsloth[cu130-torch210] @ git+https://github.com/unslothai/unsloth.git"
pip install -e .

# 2. Verify
python -c "import torch; print(torch.cuda.is_available())"

# 3. Process chat data
python scripts/preprocess_data.py

# 4. Fill out RAG survey (open tools/rag_survey.html in browser)
# Save biography.json to rag_data/

# 5. Index RAG data
python scripts/setup_rag.py

# 6. Train
python scripts/train_model.py

# 7. Test
pytest tests/ -v

# 8. Chat
python scripts/inference.py
```

---

**Happy training! 🚀**
