import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from tqdm import tqdm
import json
import os
from typing import Dict, List
from tokenizer import CustomTokenizer
from builder import ThiranModelBuilder, save_model

# ==================== Quick Training Dataset ====================
class QuickTrainDataset(Dataset):
    """Quick training dataset for demonstration"""
    
    def __init__(self, tokenizer: CustomTokenizer, num_samples: int = 50):
        self.tokenizer = tokenizer
        self.data = self._create_data(num_samples)
        print(f"\n[DATASET] Created {len(self.data)} training samples")
    
    def _create_data(self, num_samples: int) -> List[Dict]:
        """Create synthetic training data"""
        data = [
            {"text": "What is in this image?"},
            {"text": "Can you describe this visual content?"},
            {"text": "Analyze the image for me."},
            {"text": "Tell me about what you see."},
            {"text": "What objects are visible?"},
            {"text": "Describe the visual elements."},
            {"text": "Explain this image to me."},
            {"text": "What is the main focus?"},
            {"text": "How would you analyze this?"},
            {"text": "Give me details about this."},
        ]
        # Repeat to create more samples
        extended = data * (max(1, num_samples // len(data)) + 1)
        return extended[:num_samples]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        text = self.data[idx]['text']
        tokens = self.tokenizer.encode(text)
        return {'input_ids': torch.tensor(tokens, dtype=torch.long)}


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding"""
    if not batch:
        return {'input_ids': torch.tensor([])}
    
    max_len = max(len(item['input_ids']) for item in batch) if batch else 1
    max_len = min(max_len, 512)
    
    input_ids = []
    for item in batch:
        ids = item['input_ids']
        if len(ids) > max_len:
            ids = ids[:max_len]
        pad_len = max_len - len(ids)
        padded_ids = torch.cat([ids, torch.zeros(pad_len, dtype=torch.long)])
        input_ids.append(padded_ids)
    
    return {'input_ids': torch.stack(input_ids) if input_ids else torch.tensor([])}


def train_epoch(model, train_loader, optimizer, device, epoch, num_epochs, model_name="Model") -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch + 1}/{num_epochs}")
    
    for batch in pbar:
        if len(batch['input_ids']) == 0:
            continue
        
        input_ids = batch['input_ids'].to(device)
        optimizer.zero_grad()
        
        try:
            image_tensor = torch.randn(input_ids.size(0), 3, 224, 224, device=device)
            outputs = model(input_ids, image_tensor)
            loss = outputs['logits'].mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{total_loss / num_batches:.4f}'})
        except Exception as e:
            pbar.write(f"Error: {str(e)[:80]}")
            continue
    
    return total_loss / num_batches if num_batches > 0 else 0.0


def main():
    print("\n" + "=" * 70)
    print("THIRAN QUICK TRAINING")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n[CONFIG]")
    print(f"  - Device: {device}")
    print(f"  - Batch size: 2")
    print(f"  - Epochs: 1")
    
    # Initialize tokenizer
    print("\n[STEP 1] Initializing tokenizer...")
    tokenizer = CustomTokenizer(max_length=512)
    tokenizer.build_vocab([
        "What is in this image?",
        "Can you describe this?",
        "Analyze the content.",
    ], min_freq=1)
    print(f"✓ Tokenizer ready - Vocab size: {len(tokenizer.vocab)}")
    
    # Create dataset
    print("\n[STEP 2] Creating dataset...")
    dataset = QuickTrainDataset(tokenizer, num_samples=20)
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)
    print(f"✓ DataLoader created with {len(train_loader)} batches")
    
    # Build models
    print("\n[STEP 3] Building models...")
    pro_model = ThiranModelBuilder.build_pro_model(len(tokenizer.vocab)).to(device)
    lite_model = ThiranModelBuilder.build_lite_model(len(tokenizer.vocab)).to(device)
    
    # Training
    print("\n[STEP 4] Starting training...")
    
    print("\n" + "-" * 70)
    print("TRAINING PRO MODEL")
    print("-" * 70)
    pro_optimizer = AdamW(pro_model.parameters(), lr=1e-4)
    loss = train_epoch(pro_model, train_loader, pro_optimizer, device, 0, 1, "Pro")
    print(f"✓ Pro Model - Loss: {loss:.4f}")
    
    print("\n" + "-" * 70)
    print("TRAINING LITE MODEL")
    print("-" * 70)
    lite_optimizer = AdamW(lite_model.parameters(), lr=1e-4)
    loss = train_epoch(lite_model, train_loader, lite_optimizer, device, 0, 1, "Lite")
    print(f"✓ Lite Model - Loss: {loss:.4f}")
    
    # Save
    print("\n[STEP 5] Saving models...")
    save_model(pro_model, tokenizer, 'thiran_pro_model.pt')
    save_model(lite_model, tokenizer, 'thiran_lite_model.pt')
    tokenizer.save('custom_tokenizer.json')
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETED")
    print("=" * 70)
    print(f"  - Pro Model: thiran_pro_model.pt")
    print(f"  - Lite Model: thiran_lite_model.pt")
    print(f"  - Tokenizer: custom_tokenizer.json")
    print(f"  - Device: {device}")


if __name__ == "__main__":
    main()
