import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
import os
from typing import Dict, List
import json

from tokenizer import CustomTokenizer
from builder import VelCoreModelBuilder, save_model, load_model


class ConversationDataset(Dataset):
    """Dataset loader for Simple-English-Conversation with synthetic images"""
    
    def __init__(self, tokenizer: CustomTokenizer, max_samples: int = 500, split: str = "train"):
        """
        Args:
            tokenizer: CustomTokenizer instance
            max_samples: Maximum number of samples to use
            split: Dataset split to use
        """
        self.tokenizer = tokenizer
        
        print(f"Loading Simple-English-Conversation dataset (max {max_samples} samples)...")
        
        # Load dataset from Hugging Face
        try:
            dataset = load_dataset(
                "Xerv-AI/Simple-English-Conversation",
                split=split
            )
            
            # Process samples
            self.samples = []
            count = 0
            
            # fast inspection to detect keys
            if len(dataset) > 0:
                print(f"Dataset keys detected: {dataset[0].keys()}")

            for item in dataset:
                if count >= max_samples:
                    break
                    
                # Robust field extraction for unknown dataset structure
                # We try to find text content in common fields
                text = ""
                
                # Check for direct text field
                if 'text' in item and item['text']:
                    text = item['text']
                
                # Check for prompt/response pairs
                elif 'prompt' in item and 'response' in item:
                    text = f"User: {item['prompt']} Assistant: {item['response']}"
                elif 'instruction' in item and 'output' in item:
                    text = f"User: {item['instruction']} Assistant: {item['output']}"
                elif 'question' in item and 'answer' in item:
                    text = f"Question: {item['question']} Answer: {item['answer']}"
                    
                # Check for conversation/messages format
                elif 'conversation' in item and isinstance(item['conversation'], list):
                    parts = []
                    for turn in item['conversation']:
                        if isinstance(turn, dict):
                            role = turn.get('role', '')
                            content = turn.get('content', '')
                            parts.append(f"{role.capitalize()}: {content}")
                        elif isinstance(turn, str):
                            parts.append(turn)
                    text = " ".join(parts)
                
                # Fallback: join all string values
                if not text:
                    parts = []
                    for k, v in item.items():
                        if isinstance(v, str) and len(v) < 1000: # Limit length of ignored metadata
                            parts.append(f"{k.capitalize()}: {v}")
                    if parts:
                        text = " ".join(parts)
                
                if text.strip():  # Only add if we have content
                    self.samples.append({
                        'text': text,
                        # Keep original items for potential future use or debugging
                        'raw': item
                    })
                    count += 1
            
            print(f"✓ Loaded {len(self.samples)} samples from Simple-English-Conversation")
            
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")
            print("Using fallback sample data...")
            # Fallback to sample data
            self.samples = [
                {
                    'text': "User: Hello there. Assistant: Hi! How can I help you today?",
                    'raw': {}
                }
            ] * max_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Generate synthetic image (random tensor for vision-language training)
        image = torch.randn(3, 224, 224)
        
        # Generic label (0) for conversation
        label = 0
        
        return {
            'text_tokens': torch.tensor(tokens, dtype=torch.long),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'text': text
        }



def collate_fn(batch):
    """Custom collate function for batching"""
    max_len = max(len(item['text_tokens']) for item in batch)
    
    # Pad text tokens
    text_tokens = []
    for item in batch:
        tokens = item['text_tokens']
        padded = torch.cat([
            tokens,
            torch.zeros(max_len - len(tokens), dtype=torch.long)
        ])
        text_tokens.append(padded)
    
    return {
        'text_tokens': torch.stack(text_tokens),
        'images': torch.stack([item['image'] for item in batch]),
        'labels': torch.stack([item['label'] for item in batch]),
        'texts': [item['text'] for item in batch]
    }


class TrainingConfig:
    """Configuration for training"""
    def __init__(self):
        self.model_type = 'pro'  # 'lite' or 'pro'
        self.batch_size = 8
        self.num_epochs = 20
        self.learning_rate = 1e-4
        self.max_samples = 2000  # Default to 2000
        self.save_every_epoch = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.warmup_steps = 10
        self.resume_from_checkpoint = True
        
        # Loss weights
        self.classification_weight = 0.3
        self.language_model_weight = 0.7
        self.reconstruction_weight = 0.1


def compute_loss(outputs: Dict, labels: torch.Tensor, text_tokens: torch.Tensor, 
                 images: torch.Tensor, config: TrainingConfig, debug: bool = False) -> Dict[str, torch.Tensor]:
    """Compute combined loss for multimodal training"""
    
    if debug:
        print(f"\n[DEBUG] Loss Computation:")
        print(f"  Logits shape: {outputs['logits'].shape}")
        
    # Classification loss with label smoothing
    classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(outputs['logits'], labels)
    classification_loss = torch.clamp(classification_loss, max=10.0)
    
    # Language modeling loss (predict next token)
    lm_logits = outputs['reasoning_logits'][:, :-1, :].contiguous()
    lm_targets = text_tokens[:, 1:].contiguous()
    
    lm_loss = nn.CrossEntropyLoss(ignore_index=0)(
        lm_logits.view(-1, lm_logits.size(-1)),
        lm_targets.view(-1)
    )
    
    lm_loss = torch.clamp(lm_loss, max=10.0)
    
    # Combined loss
    total_loss = (config.classification_weight * classification_loss + 
                  config.language_model_weight * lm_loss)
    
    # Image reconstruction loss (Pro model only)
    if 'reconstructed_image' in outputs and config.model_type == 'pro':
        target_images = images.view(images.size(0), -1)
        reconstructed = outputs['reconstructed_image']
        reconstruction_loss = nn.MSELoss()(reconstructed, target_images)
        reconstruction_loss = torch.clamp(reconstruction_loss, max=10.0)
        total_loss += config.reconstruction_weight * reconstruction_loss
    else:
        reconstruction_loss = torch.tensor(0.0)
    
    total_loss = torch.clamp(total_loss, max=15.0)
    
    return {
        'total_loss': total_loss,
        'classification_loss': classification_loss,
        'lm_loss': lm_loss,
        'reconstruction_loss': reconstruction_loss
    }


def train_epoch(model, dataloader, optimizer, config: TrainingConfig, epoch: int):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    for batch_idx, batch in enumerate(progress_bar):
        text_tokens = batch['text_tokens'].to(config.device)
        images = batch['images'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        debug_mode = (batch_idx == 0 and epoch == 0)
        
        optimizer.zero_grad()
        outputs = model(text_tokens, images)
        losses = compute_loss(outputs, labels, text_tokens, images, config, debug=debug_mode)
        
        if torch.isnan(losses['total_loss']):
            continue
        
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += losses['total_loss'].item()
        
        progress_bar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'cls': f"{losses['classification_loss'].item():.4f}",
            'lm': f"{losses['lm_loss'].item():.4f}"
        })
    
    avg_loss = total_loss / max(1, len(dataloader))
    return avg_loss


def train_single_model(model_type: str, tokenizer, dataset, config: TrainingConfig):
    """Train a single model type (pro or lite)"""
    
    print("\n" + "=" * 70)
    print(f" TRAINING {model_type.upper()} MODEL")
    print("=" * 70)
    
    config.model_type = model_type
    vocab_size = len(tokenizer.vocab)
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Build or load model
    checkpoint_name = f'VelCore-{model_type.capitalize()}_conversation.pt'
    
    if config.resume_from_checkpoint and os.path.exists(checkpoint_name):
        print(f"\n[LOAD] Loading existing {model_type} model from checkpoint...")
        try:
            model = load_model(checkpoint_name, tokenizer, mode=model_type)
            print(f"  ✓ Model loaded from checkpoint")
        except Exception as e:
            print(f"  ⚠ Failed to load checkpoint: {e}")
            if model_type == 'pro':
                model = VelCoreModelBuilder.build_pro_model(vocab_size)
            else:
                model = VelCoreModelBuilder.build_lite_model(vocab_size)
    else:
        print(f"\n[BUILD] Building new {model_type} model with vocab size {vocab_size}...")
        if model_type == 'pro':
            model = VelCoreModelBuilder.build_pro_model(vocab_size)
        else:
            model = VelCoreModelBuilder.build_lite_model(vocab_size)
    
    model = model.to(config.device)
    
    optimizer = AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    
    print(f"\n[TRAIN] Starting training for {config.num_epochs} epochs...")
    
    best_loss = float('inf')
    training_history = []
    
    for epoch in range(config.num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, config, epoch)
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs} - Average Loss: {avg_loss:.4f}")
        training_history.append({'epoch': epoch+1, 'loss': avg_loss, 'model': model_type})
        
        if config.save_every_epoch or avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, tokenizer, checkpoint_name)
            print(f"  ✓ Saved checkpoint: {checkpoint_name}")
    
    return training_history, best_loss


def main():
    """Main training function"""
    
    print("\n" + "=" * 70)
    print(" VELCORE MODEL TRAINING - Simple-English-Conversation")
    print(" Training with conversation dataset")
    print("=" * 70)
    
    config = TrainingConfig()
    
    # Step 1: Load dataset to get all text samples for vocab
    print(f"\n[STEP 1] Loading dataset for vocabulary...")
    
    try:
        dataset_raw = load_dataset("Xerv-AI/Simple-English-Conversation", split="train")
        
        # Extract texts for vocabulary
        all_texts = []
        for i, item in enumerate(dataset_raw):
            if i >= config.max_samples:
                break
                
            # Same robust text extraction as Dataset class
            text = ""
            if 'text' in item and item['text']:
                text = item['text']
            elif 'prompt' in item and 'response' in item:
                text = f"User: {item['prompt']} Assistant: {item['response']}"
            elif 'instruction' in item and 'output' in item:
                text = f"User: {item['instruction']} Assistant: {item['output']}"
            
            if text:
                all_texts.append(text)
        
        print(f"  ✓ Loaded {len(all_texts)} text samples for vocabulary building")
        
    except Exception as e:
        print(f"  Warning: Could not load dataset: {e}")
        all_texts = ["Hello world user assistant conversation"]
    
    # Step 2: Build tokenizer
    print(f"\n[STEP 2] Building tokenizer vocabulary...")
    tokenizer = CustomTokenizer(max_length=512)
    
    all_texts.extend([
        "User Assistant Question Answer System",
        "Hello Hi Thanks Goodbye Please",
        "Conversation Dialogue Chat Interaction"
    ])
    
    tokenizer.build_vocab(all_texts, min_freq=1)
    print(f"  ✓ Tokenizer created with vocab size: {len(tokenizer.vocab)}")
    
    # Step 3: Create dataset
    print(f"\n[STEP 3] Creating training dataset...")
    dataset = ConversationDataset(tokenizer, max_samples=config.max_samples, split="train")
    
    # Step 4: Train BOTH models
    all_history = []
    
    # Train Pro model
    train_single_model('pro', tokenizer, dataset, config)
    
    # Train Lite model
    train_single_model('lite', tokenizer, dataset, config)
    
    # Save tokenizer
    tokenizer.save('conversation_tokenizer.json')
    print(f"\n[FINAL] Saved tokenizer to conversation_tokenizer.json")
    print(" ✓ ALL TRAINING COMPLETED!")


if __name__ == "__main__":
    main()
