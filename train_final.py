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

# ==================== Indic Dataset with Token ====================
class IndicInstructDataset(Dataset):
    """
    Indic Instruct dataset with HF token authentication
    """
    
    def __init__(self, tokenizer: CustomTokenizer, max_samples: int = None, language: str = "en", hf_token: str = None):
        """
        Initialize dataset
        """
        self.tokenizer = tokenizer
        self.max_samples = max_samples
        self.language = language
        
        print(f"\n[DATASET] Loading Indic Instruct data ({language} only)...")
        
        self.filtered_data = []
        
        # Try multiple methods to load the dataset
        try:
            print("  - Method 1: Using datasets library with token...")
            from datasets import load_dataset
            
            dataset = load_dataset(
                "ai4bharat/indic-instruct-data-v0.1",
                "anudesh",
                token=hf_token
            )
            
            # Get the data
            if isinstance(dataset, dict):
                data = dataset.get('train', dataset)
            else:
                data = dataset
            
            # Convert to list of dicts
            if hasattr(data, '__iter__'):
                self.filtered_data = [dict(item) for item in data]
            else:
                self.filtered_data = list(data)
                
            print(f"✓ Total samples loaded: {len(self.filtered_data)}")
            
        except Exception as e:
            print(f"  - Method 1 failed: {str(e)[:100]}")
            
            try:
                print("  - Method 2: Loading Parquet files directly...")
                from huggingface_hub import hf_hub_download
                import pandas as pd
                
                # Download the Parquet file for English
                print(f"  - Downloading English Parquet file...")
                file_path = hf_hub_download(
                    repo_id="ai4bharat/indic-instruct-data-v0.1",
                    filename="anudesh/en-00000-of-00001.parquet",
                    repo_type="dataset",
                    token=hf_token
                )
                
                print(f"  - Successfully downloaded: {file_path}")
                
                # Load Parquet file
                df = pd.read_parquet(file_path)
                print(f"  - Parquet file loaded with {len(df)} rows")
                print(f"  - Columns: {list(df.columns)}")
                
                # Convert to list of dicts
                self.filtered_data = df.to_dict('records')
                print(f"✓ Total samples loaded: {len(self.filtered_data)}")
                    
            except Exception as e2:
                print(f"  - Method 2 failed: {str(e2)[:100]}")
                print("  - Using synthetic data instead...")
                self.filtered_data = self._create_synthetic_data()
        
        # Filter for English
        self.filtered_data = self._filter_by_language()
        
        # Limit samples
        if max_samples and len(self.filtered_data) > max_samples:
            self.filtered_data = self.filtered_data[:max_samples]
        
        print(f"✓ After {language} filtering: {len(self.filtered_data)} samples")
    
    def _create_synthetic_data(self) -> List[Dict]:
        """Create synthetic training data"""
        return [
            {"instruction": "What is in this image?", "response": "The image contains visual elements for analysis.", "language": "en"},
            {"instruction": "Can you describe this?", "response": "This represents complex visual information.", "language": "en"},
            {"instruction": "Analyze the content.", "response": "The content demonstrates multimodal understanding.", "language": "en"},
            {"instruction": "What do you see?", "response": "I observe various elements and patterns.", "language": "en"},
            {"instruction": "Explain this visual.", "response": "This visual shows integrated information.", "language": "en"},
        ]
    
    def _filter_by_language(self) -> List[Dict]:
        """Filter for English language"""
        filtered = []
        for sample in self.filtered_data:
            if isinstance(sample, dict):
                lang = str(sample.get('language', 'en')).lower()
                if lang == self.language or lang.startswith(self.language):
                    filtered.append(sample)
        return filtered
    
    def __len__(self) -> int:
        return len(self.filtered_data)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = self.filtered_data[idx]
        instruction = sample.get('instruction', '')
        response = sample.get('response', '')
        combined_text = f"{instruction} {response}"
        tokens = self.tokenizer.encode(combined_text)
        return {'input_ids': torch.tensor(tokens, dtype=torch.long)}


# ==================== Training Utilities ====================
class TrainingConfig:
    """Training configuration"""
    def __init__(self):
        self.batch_size = 4
        self.learning_rate = 2e-4
        self.num_epochs = 2
        self.max_grad_norm = 1.0
        self.num_workers = 0
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.log_steps = 10


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Collate batch with padding"""
    if not batch or len(batch) == 0:
        return {'input_ids': torch.tensor([])}
    
    max_len = max(len(item['input_ids']) for item in batch)
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


def train_epoch(model, train_loader, optimizer, config, epoch, model_name="Model") -> float:
    """Train for one epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"{model_name} Epoch {epoch + 1}/{config.num_epochs}")
    
    for batch in pbar:
        if len(batch['input_ids']) == 0:
            continue
        
        input_ids = batch['input_ids'].to(config.device)
        optimizer.zero_grad()
        
        try:
            image_tensor = torch.randn(input_ids.size(0), 3, 224, 224, device=config.device)
            outputs = model(input_ids, image_tensor)
            loss = outputs['logits'].mean()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.max_grad_norm)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            pbar.set_postfix({'loss': f'{total_loss / num_batches:.4f}'})
        except Exception as e:
            print(f"\n  Error: {str(e)[:100]}")
            continue
    
    return total_loss / num_batches if num_batches > 0 else 0.0


# ==================== Main ====================
def main():
    print("\n" + "=" * 70)
    print("THIRAN MODEL TRAINING WITH INDIC INSTRUCT DATASET")
    print("=" * 70)
    
    config = TrainingConfig()
    print(f"\n[CONFIG]")
    print(f"  - Device: {config.device}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Learning rate: {config.learning_rate}")
    print(f"  - Epochs: {config.num_epochs}")
    
    # Initialize tokenizer
    print("\n[STEP 1] Initializing tokenizer...")
    tokenizer = CustomTokenizer(max_length=512)
    sample_texts = [
        "What is in this image?",
        "Can you analyze this visual content?",
        "Describe what you see here.",
    ]
    tokenizer.build_vocab(sample_texts, min_freq=1)
    print(f"✓ Tokenizer ready - Vocab size: {len(tokenizer.vocab)}")
    
    # Load dataset with token
    print("\n[STEP 2] Loading Indic Instruct dataset...")
    hf_token = "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"
    dataset = IndicInstructDataset(
        tokenizer=tokenizer,
        max_samples=500,
        language="en",
        hf_token=hf_token
    )
    
    # Build models
    print("\n[STEP 3] Building models...")
    pro_model = ThiranModelBuilder.build_pro_model(len(tokenizer.vocab)).to(config.device)
    lite_model = ThiranModelBuilder.build_lite_model(len(tokenizer.vocab)).to(config.device)
    
    # Create dataloaders
    print("\n[STEP 4] Setting up training...")
    train_loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn)
    print(f"✓ DataLoader created with {len(train_loader)} batches")
    
    pro_optimizer = AdamW(pro_model.parameters(), lr=config.learning_rate)
    lite_optimizer = AdamW(lite_model.parameters(), lr=config.learning_rate)
    
    # Training
    print("\n[STEP 5] Starting training...")
    
    print("\n" + "-" * 70)
    print("TRAINING PRO MODEL")
    print("-" * 70)
    for epoch in range(config.num_epochs):
        loss = train_epoch(pro_model, train_loader, pro_optimizer, config, epoch, "Pro")
        print(f"✓ Pro Epoch {epoch + 1} - Loss: {loss:.4f}")
    
    print("\n" + "-" * 70)
    print("TRAINING LITE MODEL")
    print("-" * 70)
    for epoch in range(config.num_epochs):
        loss = train_epoch(lite_model, train_loader, lite_optimizer, config, epoch, "Lite")
        print(f"✓ Lite Epoch {epoch + 1} - Loss: {loss:.4f}")
    
    # Save models
    print("\n[STEP 6] Saving trained models...")
    save_model(pro_model, tokenizer, 'thiran_pro_model.pt')
    save_model(lite_model, tokenizer, 'thiran_lite_model.pt')
    tokenizer.save('custom_tokenizer.json')
    
    print("\n" + "=" * 70)
    print("✓ TRAINING COMPLETED")
    print("=" * 70)
    print(f"  - Pro Model: thiran_pro_model.pt")
    print(f"  - Lite Model: thiran_lite_model.pt")
    print(f"  - Tokenizer: custom_tokenizer.json")
    print(f"  - Total samples trained: {len(dataset)}")
    print(f"  - Device: {config.device}")


if __name__ == "__main__":
    main()
