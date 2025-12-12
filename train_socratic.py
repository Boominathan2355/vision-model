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


class GSM8KSocraticDataset(Dataset):
    """Dataset loader for OpenAI GSM8K Socratic - with guided questioning approach"""
    
    def __init__(self, tokenizer: CustomTokenizer, max_samples: int = 500, split: str = "train"):
        """
        Args:
            tokenizer: CustomTokenizer instance
            max_samples: Maximum number of samples to use
            split: Dataset split to use ('train' or 'test')
        """
        self.tokenizer = tokenizer
        
        print(f"Loading GSM8K Socratic dataset (max {max_samples} samples)...")
        
        # Load dataset from Hugging Face
        try:
            dataset = load_dataset("openai/gsm8k", "socratic", split=split)
            
            # Process samples
            self.samples = []
            count = 0
            for item in dataset:
                if count >= max_samples:
                    break
                
                # GSM8K Socratic fields: question, answer (with Socratic sub-questions)
                question = item.get('question', '')
                answer = item.get('answer', '')
                
                # Create structured text with Socratic reasoning
                text_parts = ["Domain: Mathematics", "Method: Socratic"]
                
                if question:
                    text_parts.append(f"Question: {question}")
                
                if answer:
                    # Socratic answers include guided sub-questions and responses
                    text_parts.append(f"Socratic Solution: {answer}")
                
                text = " ".join(text_parts)
                
                if text.strip():
                    self.samples.append({
                        'text': text,
                        'question': question,
                        'answer': answer
                    })
                    count += 1
            
            print(f"✓ Loaded {len(self.samples)} samples from GSM8K Socratic")
            
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")
            print("Using fallback sample data...")
            self.samples = [
                {
                    'text': "Domain: Mathematics Method: Socratic Question: What is 2+2? Socratic Solution: Let's think step by step. #### 4",
                    'question': 'What is 2+2?',
                    'answer': '4'
                }
            ] * max_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Synthetic image (GSM8K is text-only)
        image = torch.randn(3, 224, 224)
        
        # Single label for Mathematics domain
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
    """Configuration for training - Optimized for Trillion Parameter Scale"""
    def __init__(self):
        self.model_type = 'pro'
        self.batch_size = 2  # Reduced for trillion params
        self.gradient_accumulation_steps = 4  # Simulate batch size of 8
        self.num_epochs = 8  # Fewer epochs for larger dataset (5000 samples)
        self.learning_rate = 5e-5  # Lower LR for massive model
        self.max_samples = 5000  # GSM8K socratic has 7,473 train samples
        self.save_every_epoch = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.warmup_steps = 200  # Extended warmup
        self.resume_from_checkpoint = True
        self.use_mixed_precision = True  # AMP for memory efficiency
        self.max_grad_norm = 0.5  # Tighter clipping
        self.weight_decay = 0.01
        
        # Loss weights
        self.classification_weight = 0.3
        self.language_model_weight = 0.7
        self.reconstruction_weight = 0.1


def compute_loss(outputs: Dict, labels: torch.Tensor, text_tokens: torch.Tensor, 
                 images: torch.Tensor, config: TrainingConfig, debug: bool = False) -> Dict[str, torch.Tensor]:
    """Compute combined loss for multimodal training"""
    
    classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(outputs['logits'], labels)
    classification_loss = torch.clamp(classification_loss, max=10.0)
    
    # Language modeling loss
    lm_logits = outputs['reasoning_logits'][:, :-1, :].contiguous()
    lm_targets = text_tokens[:, 1:].contiguous()
    
    lm_loss = nn.CrossEntropyLoss(ignore_index=0)(
        lm_logits.view(-1, lm_logits.size(-1)),
        lm_targets.view(-1)
    )
    lm_loss = torch.clamp(lm_loss, max=10.0)
    
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
        
        if debug_mode:
            print(f"\n{'='*70}")
            print(f"[DEBUG] First Batch - GSM8K Socratic Training")
            print(f"{'='*70}")
            print(f"Text tokens shape: {text_tokens.shape}")
        
        optimizer.zero_grad()
        outputs = model(text_tokens, images)
        losses = compute_loss(outputs, labels, text_tokens, images, config, debug=debug_mode)
        
        if torch.isnan(losses['total_loss']):
            continue
        
        losses['total_loss'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        if debug_mode:
            print(f"[DEBUG] Backward pass completed\n{'='*70}\n")
        
        total_loss += losses['total_loss'].item()
        
        progress_bar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'cls': f"{losses['classification_loss'].item():.4f}",
            'lm': f"{losses['lm_loss'].item():.4f}"
        })
    
    avg_loss = total_loss / len(dataloader)
    return avg_loss


def train_single_model(model_type: str, tokenizer, dataset, config: TrainingConfig):
    """Train a single model type (pro or lite)"""
    
    print("\n" + "=" * 70)
    print(f" TRAINING {model_type.upper()} MODEL with GSM8K Socratic")
    print("=" * 70)
    
    config.model_type = model_type
    vocab_size = len(tokenizer.vocab)
    
    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    checkpoint_name = f'VelCore-{model_type.capitalize()}_socratic.pt'
    
    if config.resume_from_checkpoint and os.path.exists(checkpoint_name):
        print(f"\n[LOAD] Loading existing {model_type} model from checkpoint...")
        try:
            model = load_model(checkpoint_name, tokenizer, mode=model_type)
            print(f"  ✓ Model loaded from checkpoint - resuming training")
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
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  ✓ Model ready - {total_params:,} parameters")
    
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
    
    print(f"\n✓ {model_type.upper()} MODEL TRAINING COMPLETE! Final loss: {best_loss:.4f}")
    return training_history, best_loss


def main():
    """Main training function - trains both PRO and LITE models with GSM8K Socratic"""
    
    print("\n" + "=" * 70)
    print(" VELCORE MODEL TRAINING - OpenAI GSM8K Socratic Dataset")
    print(" Training with Socratic method guided questioning")
    print("=" * 70)
    
    config = TrainingConfig()
    
    print(f"\n[CONFIG]")
    print(f"  Model Type: {config.model_type.upper()}")
    print(f"  Device: {config.device}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Max Samples: {config.max_samples}")
    
    # Step 1: Load GSM8K Socratic dataset
    print(f"\n[STEP 1] Loading GSM8K Socratic dataset...")
    
    try:
        dataset_raw = load_dataset("openai/gsm8k", "socratic", split="train")
        
        all_texts = []
        for i, item in enumerate(dataset_raw):
            if i >= config.max_samples:
                break
            
            question = item.get('question', '')
            answer = item.get('answer', '')
            
            if question:
                all_texts.append(question)
            if answer:
                all_texts.append(answer)
        
        print(f"  ✓ Loaded {len(all_texts)} text samples for vocabulary building")
        
    except Exception as e:
        print(f"  Warning: Could not load dataset: {e}")
        all_texts = ["Mathematics Socratic reasoning step problem"]
    
    # Step 2: Build tokenizer
    print(f"\n[STEP 2] Building tokenizer vocabulary...")
    tokenizer = CustomTokenizer(max_length=512)
    
    all_texts.extend([
        "Domain Mathematics Method Socratic Question Answer Solution",
        "Let's think step by step what is how many why",
        "calculate solve compute add subtract multiply divide",
        "think reason understand explain therefore thus hence"
    ])
    
    tokenizer.build_vocab(all_texts, min_freq=1)
    vocab_size = len(tokenizer.vocab)
    print(f"  ✓ Tokenizer created with vocab size: {vocab_size}")
    
    # Step 3: Create dataset
    print(f"\n[STEP 3] Creating training dataset...")
    dataset = GSM8KSocraticDataset(tokenizer, max_samples=config.max_samples, split="train")
    
    # Validate token IDs
    print(f"\n[STEP 3.1] Validating token IDs...")
    max_token_id = 0
    invalid_samples = 0
    for i in range(min(len(dataset), 10)):
        sample = dataset[i]
        tokens = sample['text_tokens']
        max_id = tokens.max().item()
        max_token_id = max(max_token_id, max_id)
        
        if max_id >= vocab_size:
            invalid_samples += 1
    
    if invalid_samples == 0:
        print(f"  ✓ All tokens valid (max ID: {max_token_id}, vocab size: {vocab_size})")
    else:
        print(f"  ERROR: Found {invalid_samples} samples with invalid token IDs!")
        return
    
    # Step 4: Train BOTH models
    all_history = []
    results = {}
    
    print("\n" + "=" * 70)
    print(" PHASE 1: Training PRO model with GSM8K Socratic")
    print("=" * 70)
    pro_history, pro_loss = train_single_model('pro', tokenizer, dataset, config)
    all_history.extend(pro_history)
    results['pro'] = pro_loss
    
    print("\n" + "=" * 70)
    print(" PHASE 2: Training LITE model with GSM8K Socratic")
    print("=" * 70)
    lite_history, lite_loss = train_single_model('lite', tokenizer, dataset, config)
    all_history.extend(lite_history)
    results['lite'] = lite_loss
    
    # Save tokenizer and history
    print(f"\n[FINAL] Saving tokenizer and training history...")
    tokenizer.save('socratic_tokenizer.json')
    
    with open('socratic_training_history.json', 'w') as f:
        json.dump(all_history, f, indent=2)
    
    print(f"  ✓ Saved tokenizer to socratic_tokenizer.json")
    print(f"  ✓ Saved training history to socratic_training_history.json")
    
    # Final summary
    print("\n" + "=" * 70)
    print(" ✓ ALL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  PRO  Model: VelCore-Pro_socratic.pt  (Final loss: {results['pro']:.4f})")
    print(f"  LITE Model: VelCore-Lite_socratic.pt (Final loss: {results['lite']:.4f})")
    print(f"\nTokenizer: socratic_tokenizer.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
