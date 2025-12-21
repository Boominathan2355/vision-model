import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from datasets import load_dataset
from tqdm import tqdm
import os
import math
import json
from PIL import Image
from typing import Dict, List

from tokenizer import CustomTokenizer
from builder import VelCoreModelBuilder, save_model, load_model

# ==================== Visual Instruction Dataset ====================
class VisualReasoningDataset(Dataset):
    """
    Multimodal dataset for real visual training.
    Uses VQAv2-small: [image, question, multiple_choice_answer]
    """
    def __init__(self, tokenizer: CustomTokenizer, max_samples: int = 1000):
        self.tokenizer = tokenizer
        self.system_prompt = (
            "You are VelCore, a vision-aware assistant. "
            "Examine the provided image and answer the user's question accurately."
        )
        
        print(f"Loading VQAv2-small visual dataset...")
        try:
            # Using validation split as it's often more cleaned for small runs
            self.ds = load_dataset("merve/vqav2-small", split="validation")
            print(f"✓ Loaded {len(self.ds)} visual samples")
        except Exception as e:
            print(f"Error loading dataset: {e}. Falling back to dummy data.")
            self.ds = []

        self.max_samples = min(len(self.ds), max_samples) if self.ds else 0
        
        # Image transformation for ViT
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return self.max_samples

    def __getitem__(self, idx):
        item = self.ds[idx]
        image = item['image'].convert('RGB')
        question = item['question']
        answer = item['multiple_choice_answer']
        
        # Transform image
        image_tensor = self.transform(image)
        
        # Format as Anthropic-style conversational block
        # The [IMG] token signals the model to attend to the visual embedding
        formatted_text = (
            f"System: {self.system_prompt}\n"
            f"User: [IMG] {question}\n"
            f"Assistant: [THINK_START] I see the image. {question} [THINK_END] {answer}"
        )
        
        tokens = self.tokenizer.encode(formatted_text)
        
        return {
            'text_tokens': torch.tensor(tokens, dtype=torch.long),
            'image': image_tensor,
            'text': formatted_text
        }

def collate_fn(batch):
    return {
        'text_tokens': torch.stack([item['text_tokens'] for item in batch]),
        'images': torch.stack([item['image'] for item in batch]),
        'texts': [item['text'] for item in batch]
    }

# ==================== Anthropic Training Method ====================
def train_visual():
    # 1. Config
    EPOCHS = 5
    LR = 3e-5
    BATCH_SIZE = 2
    GRAD_ACCUM = 8 # Simulating batch size of 16
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print(f"\n[START] Visual Reasoning Training on {DEVICE}")
    
    # 2. Setup
    tokenizer = CustomTokenizer(max_length=512)
    # Build vocab if needed (using sample from dataset)
    ds_temp = load_dataset("merve/vqav2-small", split="validation")
    sample_texts = [f"System User Assistant [IMG] [THINK_START] [THINK_END] {x['question']} {x['multiple_choice_answer']}" for x in list(ds_temp)[:100]]
    tokenizer.build_vocab(sample_texts)
    
    dataset = VisualReasoningDataset(tokenizer, max_samples=1000)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    
    # Load Pro model for better visual capacity
    model_path = 'Pro-model.pt'
    if os.path.exists(model_path):
        model, _ = load_model(model_path, tokenizer, mode='pro')
    else:
        model = VelCoreModelBuilder.build_pro_model(len(tokenizer.vocab))
    model.to(DEVICE)
    
    # Advanced Weight Decay
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if any(x in n for x in ["bias", "Norm", "norm"]): no_decay.append(p)
        else: decay.append(p)
    
    optimizer = AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0}
    ], lr=LR, betas=(0.9, 0.95))
    
    # Scheduler
    total_steps = len(loader) * EPOCHS
    warmup = int(total_steps * 0.1)
    def lr_lambda(s):
        if s < warmup: return s / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / (total_steps - warmup)))
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Precision
    dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    
    # 3. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for i, batch in enumerate(pbar):
            ids = batch['text_tokens'].to(DEVICE)
            imgs = batch['images'].to(DEVICE)
            
            with torch.autocast(device_type=DEVICE, dtype=dtype):
                # Standard causal LM loss
                out = model(ids, imgs)['reasoning_logits']
                loss = nn.functional.cross_entropy(
                    out[:, :-1].reshape(-1, out.size(-1)),
                    ids[:, 1:].reshape(-1),
                    ignore_index=0
                )
                loss = loss / GRAD_ACCUM
            
            loss.backward()
            
            if (i + 1) % GRAD_ACCUM == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                
            total_loss += loss.item() * GRAD_ACCUM
            pbar.set_postfix(loss=f"{loss.item() * GRAD_ACCUM:.4f}")
            
        print(f"✓ Epoch {epoch+1} Complete. Avg Loss: {total_loss/len(loader):.4f}")
        save_model(model, tokenizer, 'Visual-Pro-model.pt', epoch=epoch+1)
        
    print("\n✅ VISUAL TRAINING COMPLETE. Model saved as 'Visual-Pro-model.pt'")

if __name__ == "__main__":
    train_visual()
