"""
Interactive Chat with Bot
===============================
Load the fine-tuned model and start an interactive
terminal session with RAG-enhanced personal knowledge retrieval.

Usage:
    python scripts/inference.py
"""
import sys
from pathlib import Path
import textwrap
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
sys.path.append(str(Path(__file__).parent.parent))
from rag.rag_helper import RAGHelper

from config import (
    BASE_MODEL_ID,
    MODEL_OUTPUT_DIR,
    USE_4BIT,
    QUANT_TYPE,
    USE_BF16,
    CHROMA_DB_DIR,
    RAG_COLLECTION_NAME,
    RAG_N_RESULTS,
    MAX_RAG_CONTEXT_TOKENS,
    INFERENCE_MAX_TOKENS,
    INFERENCE_TEMPERATURE,
    INFERENCE_TOP_P,
    USE_4BIT
)

# ----------------- Model Loader -----------------
def load_model(
    base_model_id=BASE_MODEL_ID,
    finetuned_model_path=MODEL_OUTPUT_DIR,
    use_4bit=USE_4BIT,
):
    print("🧠 Loading model...")
    if use_4bit:
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=USE_4BIT,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type=QUANT_TYPE,
            bnb_4bit_compute_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if USE_BF16 else torch.float16,
        )

    # Load fine-tuned adapter weights
    model = PeftModel.from_pretrained(base_model, str(finetuned_model_path))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("✅ Model loaded successfully!")
    return model, tokenizer


# ----------------- RAG Helper -----------------
rag_helper = RAGHelper(
    persist_directory=str(CHROMA_DB_DIR),
    collection_name=RAG_COLLECTION_NAME,
)

# ----------------- Prompt Formatter -----------------
def format_qwen_prompt(conversation_history):
    prompt = ""
    for msg in conversation_history:
        role = "user" if msg["role"] == "user" else "assistant"
        prompt += f"<|im_start|>{role}\n{msg['content']}<|im_end|>\n"
    prompt += "<|im_start|>assistant\n"
    return prompt


# ----------------- Generate Response -----------------
def generate_response(
    model,
    tokenizer,
    prompt,
    max_new_tokens=INFERENCE_MAX_TOKENS,
    temperature=INFERENCE_TEMPERATURE,
    top_p=INFERENCE_TOP_P,
):
    inputs = tokenizer(prompt, return_tensors="pt", padding=True)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    stop_tokens = ["<|im_start|>", "<|im_end|>", "User:", "\nUser:", "\n\nUser:"]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    response = tokenizer.decode(outputs[0], skip_special_tokens=False)

    if response.startswith(prompt):
        response = response[len(prompt):].strip()

    for stop_token in stop_tokens:
        if stop_token in response:
            response = response.split(stop_token)[0].strip()

    return response


# ----------------- Terminal Chat -----------------
def you_bot_chat(
    model,
    tokenizer,
    max_tokens=INFERENCE_MAX_TOKENS,
    top_rag=RAG_N_RESULTS,
    max_rag_tokens=MAX_RAG_CONTEXT_TOKENS,
):
    conversation_history = []
    print("=" * 80)
    print("INTERACTIVE TESTING MODE")
    print("Commands:")
    print("  - Type your question/prompt and press Enter")
    print("  - Type 'quit' or 'exit' to end")
    print("  - Type 'clear' to clear conversation history")
    print("=" * 80)

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["quit", "exit"]:
            print("👋 Bye!")
            break
        if user_input.lower() == "clear":
            conversation_history = []
            print("🧹 Conversation reset.\n")
            continue
        if not user_input:
            continue

        # Add user message
        conversation_history.append({"role": "user", "content": user_input})

        # Get RAG context
        rag_contexts = rag_helper.query_context(user_input, n_results=top_rag)
        formatted_context = ""
        tokens_accum = 0

        for ctx in rag_contexts:
            text = ctx["text"]
            token_count = len(tokenizer.encode(text, add_special_tokens=False))
            if tokens_accum + token_count > (max_rag_tokens // top_rag):
                tokens = tokenizer.encode(text, add_special_tokens=False)
                tokens = tokens[: (max_rag_tokens // top_rag) - tokens_accum]
                text = tokenizer.decode(tokens)
            formatted_context += f"[RAG] {text}\n\n"
            tokens_accum += len(tokenizer.encode(text, add_special_tokens=False))

        if formatted_context:
            formatted_context = (
                "Use the following context to answer if relevant:\n\n" + formatted_context
            )

        # Build full prompt
        final_prompt = format_qwen_prompt(conversation_history)
        if formatted_context:
            final_prompt = formatted_context + "\n" + final_prompt

        # Generate
        response = generate_response(model, tokenizer, final_prompt, max_new_tokens=max_tokens)
        response = response.strip()

        # Print and add to history
        print("Assistant: " + "\n".join(textwrap.wrap(response, width=120)) + "\n")
        conversation_history.append({"role": "assistant", "content": response})


# ----------------- Run -----------------
if __name__ == "__main__":
    model, tokenizer = load_model()
    you_bot_chat(model, tokenizer)
