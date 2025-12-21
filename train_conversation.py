import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm

from tokenizer import CustomTokenizer
from builder import VelCoreModelBuilder, save_model, load_model

# ============================================================
# CONFIG
# ============================================================

HF_TOKEN_PATH = r"D:/model/code.txt"
LITE_MODEL_PATH = "Lite-model.pt"
PRO_MODEL_PATH = "Pro-model.pt"
TOKENIZER_PATH = "custom_tokenizer.json"

BATCH_SIZE = 4
EPOCHS = 10
LR = 3e-5
MAX_LENGTH = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD HF TOKEN
# ============================================================

if not os.path.exists(HF_TOKEN_PATH):
    raise FileNotFoundError(f"HF token not found: {HF_TOKEN_PATH}")

with open(HF_TOKEN_PATH, "r", encoding="utf-8-sig") as f:
    HF_TOKEN = f.read().strip()

if not HF_TOKEN:
    raise ValueError("HF token file is empty")

print("✓ Hugging Face token loaded")

# ============================================================
# DATASET
# ============================================================

def format_item(item):
    if "conversation" in item:
        return "\n".join(
            f"{t.get('role', 'user').capitalize()}: {t.get('content', '')}"
            for t in item["conversation"]
        )
    if "text" in item:
        return item["text"]
    if "prompt" in item and "response" in item:
        return f"User: {item['prompt']}\nAssistant: {item['response']}"
    if "instruction" in item and "output" in item:
        prompt = item["instruction"]
        if item.get("input"):
            prompt += f"\nInput: {item['input']}"
        return f"User: {prompt}\nAssistant: {item['output']}"
    return None

class ConversationDataset(Dataset):
    def __init__(self, tokenizer, split="train", max_samples=5000):
        self.tokenizer = tokenizer
        self.samples = []

        print(f"Loading conversational data...")
        conv_dataset = load_dataset(
            "Xerv-AI/Simple-English-Conversation",
            split=split,
            token=HF_TOKEN
        )

        for item in conv_dataset:
            text = format_item(item)
            if text:
                self.samples.append(text)

        print(f"Loading TinyStories for coherence (max {max_samples} samples)...")
        story_dataset = load_dataset(
            "roneneldan/TinyStories",
            split="train",
            streaming=True,
            token=HF_TOKEN
        )

        count = 0
        for item in story_dataset:
            if count >= max_samples:
                break
            text = format_item(item)
            if text:
                self.samples.append(text)
                count += 1

        # critical clarification supervision
        self.samples.extend([
            "User: ??\nAssistant: I do not understand. Please clarify.",
            "User: Explain this\nAssistant: Please give more details.",
            "User: Then explain briefate today\nAssistant: I do not understand the question."
        ])

        print(f"✓ Total loaded: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return torch.tensor(
            self.tokenizer.encode(self.samples[idx]),
            dtype=torch.long
        )

# ============================================================
# COLLATE
# ============================================================

def collate_fn(batch):
    max_len = min(max(len(x) for x in batch), MAX_LENGTH)

    input_ids, attention_mask = [], []

    for seq in batch:
        pad_len = max_len - len(seq)
        input_ids.append(
            torch.cat([seq, torch.zeros(pad_len, dtype=torch.long)])
        )
        attention_mask.append(
            torch.cat([torch.ones(len(seq)), torch.zeros(pad_len)])
        )

    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask).bool()
    }

# ============================================================
# LOSS
# ============================================================

def causal_lm_loss(logits, input_ids):
    logits = logits[:, :-1, :]
    targets = input_ids[:, 1:]
    return nn.CrossEntropyLoss(ignore_index=0)(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1)
    )

# ============================================================
# TRAIN
# ============================================================

def train(model_type="pro"):
    print("\n" + "=" * 70)
    print(f" VELCORE TRAINING — {model_type.upper()} MODEL")
    print("=" * 70)

    model_path = PRO_MODEL_PATH if model_type == "pro" else LITE_MODEL_PATH

    # ---------------- TOKENIZER (Always Rebuild for New Data) ----------------
    print("Rebuilding tokenizer for expanded dataset...")
    
    # Collect all text to build fresh vocab
    dataset_for_vocab = ConversationDataset(None) # Temporary for collecting text
    all_texts = dataset_for_vocab.samples
    
    tokenizer = CustomTokenizer(max_length=MAX_LENGTH)
    tokenizer.build_vocab(all_texts, min_freq=2)
    tokenizer.save(TOKENIZER_PATH)
    print(f"✓ Built fresh tokenizer. Vocab size: {len(tokenizer.vocab)}")

    # ---------------- DATASET ----------------
    dataset = ConversationDataset(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    # ---------------- MODEL ----------------
    if os.path.exists(model_path):
        model, _ = load_model(model_path, tokenizer, mode=model_type)
        print(f"✓ Loaded {model_type} model (resume)")
    else:
        if model_type == "pro":
            model = VelCoreModelBuilder.build_pro_model(len(tokenizer.vocab))
        else:
            model = VelCoreModelBuilder.build_lite_model(len(tokenizer.vocab))
        print(f"✓ Built new {model_type} model")

    model = model.to(DEVICE)
    model.train()

    optimizer = AdamW(model.parameters(), lr=LR)

    # ---------------- TRAIN LOOP ----------------
    for epoch in range(EPOCHS):
        total_loss = 0
        bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in bar:
            input_ids = batch["input_ids"].to(DEVICE)
            attention_mask = batch["attention_mask"].to(DEVICE)

            # Create dummy images for text-only training
            dummy_images = torch.zeros((input_ids.shape[0], 3, 224, 224), device=DEVICE)

            optimizer.zero_grad()
            outputs = model(text_input=input_ids, image_input=dummy_images)
            loss = causal_lm_loss(outputs["reasoning_logits"], input_ids)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            bar.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(dataloader)
        print(f"\nEpoch {epoch+1} | Avg Loss: {avg_loss:.4f}")

        save_model(model, tokenizer, model_path, epoch=epoch + 1)
        print(f"✓ Checkpoint updated: {model_path} (Epoch {epoch+1})")

    print("\n✅ TRAINING COMPLETE")

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    # Train Lite Model
    train(model_type="lite")
    
    # Train Pro Model
    train(model_type="pro")
