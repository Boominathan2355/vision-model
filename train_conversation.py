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

HF_TOKEN_PATH = r"D:/model/token.txt"
MODEL_PATH = "Pro-model.pt"
TOKENIZER_PATH = "custom_tokenizer.json"

BATCH_SIZE = 4
EPOCHS = 2
LR = 3e-5
MAX_LENGTH = 512

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# LOAD HF TOKEN
# ============================================================

if not os.path.exists(HF_TOKEN_PATH):
    raise FileNotFoundError(f"HF token not found: {HF_TOKEN_PATH}")

with open(HF_TOKEN_PATH, "r", encoding="utf-8") as f:
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
    def __init__(self, tokenizer, split="train"):
        self.tokenizer = tokenizer
        self.samples = []

        dataset = load_dataset(
            "Xerv-AI/Simple-English-Conversation",
            split=split,
            token=HF_TOKEN
        )

        for item in dataset:
            text = format_item(item)
            if text:
                self.samples.append(text)

        # critical clarification supervision
        self.samples.extend([
            "User: ??\nAssistant: I do not understand. Please clarify.",
            "User: Explain this\nAssistant: Please give more details.",
            "User: Then explain briefate today\nAssistant: I do not understand the question."
        ])

        print(f"✓ Loaded {len(self.samples)} samples")

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

def train():
    print("\n" + "=" * 70)
    print(" VELCORE TRAINING — HF AUTH DATASET")
    print("=" * 70)

    # ---------------- TOKENIZER ----------------
    if os.path.exists(TOKENIZER_PATH):
        tokenizer = CustomTokenizer(vocab_path=TOKENIZER_PATH, max_length=MAX_LENGTH)
        print("✓ Loaded tokenizer")
    else:
        raw = load_dataset(
            "Xerv-AI/Simple-English-Conversation",
            split="train",
            token=HF_TOKEN
        )

        texts = []
        for item in raw:
            text = format_item(item)
            if text:
                texts.append(text)

        texts.extend([
            "I do not understand. Please clarify.",
            "Please give more details."
        ])

        tokenizer = CustomTokenizer(max_length=MAX_LENGTH)
        tokenizer.build_vocab(texts)
        tokenizer.save(TOKENIZER_PATH)
        print("✓ Built tokenizer")

    print(f"Vocab size: {len(tokenizer.vocab)}")

    # ---------------- DATASET ----------------
    dataset = ConversationDataset(tokenizer)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=collate_fn
    )

    # ---------------- MODEL ----------------
    if os.path.exists(MODEL_PATH):
        model, _ = load_model(MODEL_PATH, tokenizer, mode="pro")
        print("✓ Loaded model (resume)")
    else:
        model = VelCoreModelBuilder.build_pro_model(len(tokenizer.vocab))
        print("✓ Built new model")

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

        save_model(model, tokenizer, MODEL_PATH)
        print("✓ Checkpoint saved")

    print("\n✅ TRAINING COMPLETE")

# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    train()
