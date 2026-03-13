# Configuration Guide

This document explains the key parameters in `config.py` used for training and running your personal AI model. Use this guide to understand how to tune the model's performance and memory usage.

## Training Parameters

These settings control the mechanics of the training process.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MAX_LENGTH` | `2048` | The maximum context length (in tokens) the model can handle. Higher values act as "memory" for longer conversations but use significantly more VRAM. |
| `BATCH_SIZE` | `1` | How many examples are processed per GPU step. Keep this at 1 for consumer GPUs to save memory. |
| `GRADIENT_ACCUMULATION_STEPS` | `8` | Simulates a larger batch size by accumulating gradients over multiple steps before updating weights. `Batch Size * Accumulation Steps = Effective Batch Size`. |
| `EPOCHS` | `3` | Number of times the model sees the entire dataset. 1-2 epochs for subtle style, 3-4 for balanced, 5+ for strong style (risk of overfitting). |
| `LEARNING_RATE` | `2e-4` | How fast the model learns. Lower (e.g., 1e-4) is safer/slower; higher (e.g., 3e-4) is faster but can act erratic. |
| `PACKING` | `True` | Combines multiple short examples into a single sequence (up to `MAX_LENGTH`). Speeds up training significantly but uses slightly more VRAM. |

## LoRA (Low-Rank Adaptation) Settings

These control how "deeply" the model is modified.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LORA_R` | `64` | The "Rank" of the adapter. Higher = smarter/more complex behavior, but more VRAM. 16-32 is efficient; 64-128 is high performance. |
| `LORA_ALPHA` | `128` | Scaling factor for LoRA updates. Standard practice is `LORA_ALPHA = 2 * LORA_R`. |
| `TARGET_MODULES` | `[...]` | Which parts of the model to retrain. Training "all linear layers" (q, k, v, o, gate, up, down) yields the best results but uses the most memory. |

---

## Troubleshooting: Out of Memory (OOM)

If you see `CUDA out of memory` errors, try these solutions in order. They are ranked from **least impact on quality** to **"requires a different model."**

### Level 1: Minimal Impact

These settings reduce VRAM spikes with little to no effect on the final model quality.

1. **Disable Packing**
   - **Change:** `PACKING = False`
   - **Why:** When packing is on, the model constantly processes full 2048-token sequences. Disabling it means many batches will be shorter, reducing peak memory usage.
   - **Trade-off:** Training will be 2-3x slower.

2. **Verify Batch Size**
   - **Change:** Ensure `BATCH_SIZE = 1`
   - **Why:** Increasing batch size on consumer cards is the #1 cause of OOM.
   - **Trade-off:** None (standard for LoRA).

### Level 2: Moderate Impact

These reduce VRAM significantly but might slightly affect the model's ability to recall long contexts or complex nuances.

3. **Reduce Context Length**
   - **Change:** `MAX_LENGTH = 1024` (or `512` if desperate)
   - **Why:** Attention memory scales quadratically. Cutting length in half saves massive amounts of VRAM.
   - **Trade-off:** The model won't "see" as far back in a conversation during training.

4. **Lower LoRA Rank**
   - **Change:** `LORA_R = 32` (and `LORA_ALPHA = 64`) -> or even `16`/`32`.
   - **Why:** Fewer trainable parameters = less gradients to store.
   - **Trade-off:** The adapter becomes less expressive and might miss subtle patterns.

### Level 3: Large Impact

5. **Target Fewer Modules**
   - **Change:** `TARGET_MODULES = ["q_proj", "v_proj"]`
   - **Why:** Only trains the attention mechanism, ignoring the feed-forward networks (MLPs). Cuts trainable params by ~60%.
   - **Trade-off:** The model learns significantly less about your specific knowledge and vocabulary.

### Level 4: Significant Impact

If none of the above work, your GPU is probably unable fit the model you chose.

6. **Switch to a Smaller Model**
   - **Change:** `BASE_MODEL_ID ` to a model with less parameters
   - **Why:** Parameter count is the ultimate memory bottleneck.
   - **Trade-off:** Smaller models are less intelligent and nuanced.

| GPU VRAM | Recommended Model Size |
|----------|------------------------|
| **8GB** | 4B |
| **12-16GB** | 8B |
| **24GB+** | 14B |
