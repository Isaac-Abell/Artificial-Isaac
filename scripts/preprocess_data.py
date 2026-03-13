"""
Unified Data Preprocessor
===========================
Converts WhatsApp and Instagram chat exports to standard Hugging Face format (role/content) for training.
This is the only preprocessing script needed - outputs universal format that
works with any model via Unsloth's chat template system.

Usage:
    python scripts/preprocess_data.py
"""

import json
import os
import sys
import zipfile
from pathlib import Path
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoTokenizer

# Add parent directory to path for config import
from artificial_isaac.config import (
    WHATSAPP_DIR, INSTAGRAM_DIR, DATASET_OUTPUT, CHAT_OWNER,
    SAME_CONVO_THRESHOLD_SECONDS, SAME_USER_THRESHOLD_SECONDS,
    HISTORY_MAX_TOKENS, CONVO_MIN_TOKENS, TOKENIZER_ID
)


def load_tokenizer():
    """Load tokenizer for token counting."""
    return AutoTokenizer.from_pretrained(
        TOKENIZER_ID,
        trust_remote_code=True,
        use_fast=True
    )


# =============================================================================
# WHATSAPP PROCESSING
# =============================================================================

def process_whatsapp_file(txt_path: Path, chat_owner: str, encoder) -> list:
    """
    Process a WhatsApp chat export file.
    
    Args:
        txt_path: Path to WhatsApp .txt file
        chat_owner: Your name in the chat
        encoder: Tokenizer for token counting
        
    Returns:
        List of conversations in ShareGPT format
    """
    from whatstk import WhatsAppChat
    
    chat = WhatsAppChat.from_source(filepath=str(txt_path))
    df = chat.df
    
    # Calculate time differences
    df["date_previous"] = df["date"].shift(periods=1)
    df["time_delta"] = (df["date"] - df["date_previous"]).dt.total_seconds().fillna(0)
    
    # Assign roles: you are "assistant", others are "user"
    df["role"] = df["username"].apply(lambda x: "assistant" if x == chat_owner else "user")
    
    # Filter out media messages
    df = df[
        (df["message"] != "<Media omitted>") & 
        (df["message"] != "<This message was edited>") & 
        (df["message"] != "<Deleted message>")
    ].copy()

    df["message"].replace("<This message was edited>", "", inplace=True)
    df["message"].replace("<Deleted message>", "", inplace=True)
    df["message"].replace("<Media omitted>", "", inplace=True)
    
    # Merge consecutive same-sender messages
    df = collapse_messages(df, SAME_USER_THRESHOLD_SECONDS)
    
    # Group into conversations
    return segment_conversations(df, encoder)


def process_whatsapp_dir(whatsapp_dir: Path, chat_owner: str, encoder) -> list:
    """Process all WhatsApp files in directory."""
    all_conversations = []
    
    # Extract zip files first
    for zip_path in whatsapp_dir.glob("*.zip"):
        print(f"Extracting {zip_path.name}...")
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(whatsapp_dir)
    
    txt_files = list(whatsapp_dir.glob("*.txt"))
    if not txt_files:
        return []
    
    print(f"\nProcessing {len(txt_files)} WhatsApp file(s)...")
    for txt_file in tqdm(txt_files, desc="WhatsApp"):
        try:
            conversations = process_whatsapp_file(txt_file, chat_owner, encoder)
            all_conversations.extend(conversations)
        except Exception as e:
            print(f"Error processing {txt_file.name}: {e}")
    
    return all_conversations


# =============================================================================
# INSTAGRAM PROCESSING
# =============================================================================

def load_instagram_messages(folder_path: Path) -> tuple:
    """Load all message_*.json files from an Instagram conversation folder."""
    all_messages = []
    participants = None
    
    for filename in sorted(os.listdir(folder_path)):
        if not (filename.startswith("message_") and filename.endswith(".json")):
            continue
        
        file_path = folder_path / filename
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            messages = data.get("messages", [])
            
            if participants is None:
                participants = data.get("participants", [])
            
            all_messages.extend(messages)
    
    is_group = len(participants) > 2 if participants else False
    return all_messages, is_group


def is_valid_message(content: str) -> bool:
    """Check if message content is valid and not a system message."""
    if not content:
        return False
    
    content_lower = content.lower()
    invalid_patterns = [
        "started a video chat", "liked a message", "sent an attachment",
        "reacted", "sent a photo", "sent a video", "sent a voice message",
        "started a call", "missed a call", "ended the call"
    ]
    
    for pattern in invalid_patterns:
        if pattern in content_lower:
            return False
    
    # Check for valid text content (not mostly emojis)
    try:
        content.encode('utf-8')
        text_chars = sum(c.isalnum() or c.isspace() for c in content)
        if len(content) > 0 and text_chars / len(content) < 0.3:
            return False
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False
    
    return True


def process_instagram_folder(folder_path: Path, chat_owner: str, encoder) -> list:
    """Process a single Instagram conversation folder."""
    messages, is_group = load_instagram_messages(folder_path)
    
    if is_group or not messages:
        return []
    
    # Sort by timestamp (oldest first)
    messages.sort(key=lambda m: m.get("timestamp_ms", 0))
    
    # Convert to DataFrame
    rows = []
    for msg in messages:
        content = msg.get("content", "").strip()
        if not is_valid_message(content):
            continue
        
        sender = msg.get("sender_name")
        timestamp_ms = msg.get("timestamp_ms", 0)
        
        rows.append({
            "username": sender,
            "message": content,
            "timestamp_s": timestamp_ms / 1000.0
        })
    
    if not rows:
        return []
    
    df = pd.DataFrame(rows)
    df["timestamp_previous"] = df["timestamp_s"].shift(periods=1)
    df["time_delta"] = (df["timestamp_s"] - df["timestamp_previous"]).fillna(0)
    df["role"] = df["username"].apply(lambda x: "assistant" if x == chat_owner else "user")
    
    # Merge consecutive same-sender messages
    df = collapse_messages(df, SAME_USER_THRESHOLD_SECONDS)
    
    return segment_conversations(df, encoder)


def process_instagram_dir(instagram_dir: Path, chat_owner: str, encoder) -> list:
    """Process all Instagram conversation folders."""
    if not instagram_dir.exists():
        return []
    
    all_conversations = []
    folder_list = [f for f in instagram_dir.iterdir() if f.is_dir()]
    
    if not folder_list:
        return []
    
    print(f"\nProcessing {len(folder_list)} Instagram conversation(s)...")
    for folder in tqdm(sorted(folder_list), desc="Instagram"):
        conversations = process_instagram_folder(folder, chat_owner, encoder)
        all_conversations.extend(conversations)
    
    return all_conversations


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def collapse_messages(df: pd.DataFrame, delta_threshold: int) -> pd.DataFrame:
    """Merge consecutive messages from the same sender within time threshold."""
    if len(df) == 0:
        return df
    
    new_data = []
    current_role = df.iloc[0]["role"]
    current_message = df.iloc[0]["message"]
    current_time_delta = df.iloc[0].get("time_delta", 0)
    
    for _, row in df.iloc[1:].iterrows():
        if row["role"] == current_role and row["time_delta"] < delta_threshold:
            current_message += "\n" + row["message"]
        else:
            new_data.append({
                "role": current_role,
                "message": current_message,
                "time_delta": current_time_delta
            })
            current_role = row["role"]
            current_message = row["message"]
            current_time_delta = row["time_delta"]
    
    new_data.append({
        "role": current_role,
        "message": current_message,
        "time_delta": current_time_delta
    })
    
    return pd.DataFrame(new_data)


def segment_conversations(df: pd.DataFrame, encoder) -> list:
    """
    Segment messages into conversations based on time gaps.
    Returns list of conversations in ShareGPT format.
    """
    conversations = []
    current_convo = []
    token_count = 0
    
    for _, row in df.iterrows():
        message_tokens = len(encoder.encode(row["message"]))
        
        # Check if we should start a new conversation
        if (row["time_delta"] < SAME_CONVO_THRESHOLD_SECONDS and 
            token_count + message_tokens < HISTORY_MAX_TOKENS):
            current_convo.append({"role": row["role"], "content": row["message"]})
            token_count += message_tokens
        else:
            if current_convo and token_count >= CONVO_MIN_TOKENS:
                conversations.append({"conversations": current_convo})
            current_convo = [{"role": row["role"], "content": row["message"]}]
            token_count = message_tokens
    
    # Don't forget the last conversation
    if current_convo and token_count >= CONVO_MIN_TOKENS:
        conversations.append({"conversations": current_convo})
    
    return conversations


# =============================================================================
# MAIN
# =============================================================================

def main():
    """Main preprocessing pipeline."""
    print("=" * 70)
    print("Unified Data Preprocessor (Hugging Face Format)")
    print("=" * 70)
    print(f"\nChat owner: {CHAT_OWNER}")
    print(f"Output: {DATASET_OUTPUT}\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    encoder = load_tokenizer()
    
    all_conversations = []
    
    # Process WhatsApp
    if WHATSAPP_DIR.exists():
        whatsapp_convos = process_whatsapp_dir(WHATSAPP_DIR, CHAT_OWNER, encoder)
        all_conversations.extend(whatsapp_convos)
        print(f"  ✓ WhatsApp: {len(whatsapp_convos)} conversations")
    else:
        print(f"WhatsApp directory not found: {WHATSAPP_DIR}")
    
    # Process Instagram
    if INSTAGRAM_DIR.exists():
        instagram_convos = process_instagram_dir(INSTAGRAM_DIR, CHAT_OWNER, encoder)
        all_conversations.extend(instagram_convos)
        print(f"  ✓ Instagram: {len(instagram_convos)} conversations")
    else:
        print(f"Instagram directory not found: {INSTAGRAM_DIR}")
    
    if not all_conversations:
        print("\n No conversations extracted!")
        print("   Please add WhatsApp .txt or Instagram JSON exports to:")
        print(f"   - WhatsApp: {WHATSAPP_DIR}")
        print(f"   - Instagram: {INSTAGRAM_DIR}")
        return
    
    # Write output in standard Hugging Face format
    DATASET_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with open(DATASET_OUTPUT, "w", encoding="utf-8") as f:
        for convo in all_conversations:
            f.write(json.dumps(convo, ensure_ascii=False) + "\n")
    
    # Statistics
    total_messages = sum(len(c["conversations"]) for c in all_conversations)
    total_tokens = sum(
        len(encoder.encode(msg["content"], add_special_tokens=False))
        for c in all_conversations
        for msg in c["conversations"]
    )
    print("\n" + "=" * 70)
    print("Statistics:")
    print("=" * 70)
    print(f"  Total conversations:  {len(all_conversations):,}")
    print(f"  Total messages:       {total_messages:,}")
    print(f"  Avg messages/convo:   {total_messages / len(all_conversations):.1f}")
    print(f"  Total tokens:         {total_tokens:,}")
    print(f"  Avg tokens/convo:     {total_tokens / len(all_conversations):.0f}")
    print(f"\nPreprocessing complete!")
    print(f"   Output saved to: {DATASET_OUTPUT}")
    print("=" * 70)   


if __name__ == "__main__":
    main()
