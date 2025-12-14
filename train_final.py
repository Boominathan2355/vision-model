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


class TuringReasoningDataset(Dataset):
    """Dataset loader for Turing-Open-Reasoning with synthetic images"""
    
    def __init__(self, tokenizer: CustomTokenizer, max_samples: int = 500, split: str = "train"):
        """
        Args:
            tokenizer: CustomTokenizer instance
            max_samples: Maximum number of samples to use
            split: Dataset split to use ('train' or other available splits)
        """
        self.tokenizer = tokenizer
        
        print(f"Loading Turing-Open-Reasoning dataset (max {max_samples} samples)...")
        
        # Load token
        token = None
        if os.path.exists("token.txt"):
            try:
                with open("token.txt", "r") as f:
                    token = f.read().strip()
            except:
                pass

        # Load dataset from Hugging Face
        try:
            dataset = load_dataset(
                "open-thoughts/Turing-Open-Reasoning",
                split=split,
                token=token
            )
            
            # Process samples
            self.samples = []
            count = 0
            for item in dataset:
                if max_samples is not None and count >= max_samples:
                    break
                    
                # Extract fields
                question = item.get('question', '')
                answer = item.get('answer', '')
                domain = item.get('domain', '')
                subdomain = item.get('sub-domain', '')
                code = item.get('code', '')
                
                # Create comprehensive text combining question, answer, and context
                # This encourages the model to learn reasoning patterns
                text_parts = []
                
                # Add domain context
                if domain:
                    text_parts.append(f"Domain: {domain}")
                if subdomain:
                    text_parts.append(f"Sub-domain: {subdomain}")
                
                # Add question
                if question:
                    text_parts.append(f"Question: {question}")
                
                # Add code if available (for computational reasoning)
                if code and code.strip():
                    text_parts.append(f"Code: {code}")
                
                # Add answer
                if answer:
                    text_parts.append(f"Answer: {answer}")
                
                # Combine all parts
                text = " ".join(text_parts)
                
                if text.strip():  # Only add if we have content
                    self.samples.append({
                        'text': text,
                        'domain': domain,
                        'question': question,
                        'answer': answer
                    })
                    count += 1
            
            print(f"✓ Loaded {len(self.samples)} samples from Turing-Open-Reasoning")
            
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")
            print("Using fallback sample data...")
            # Fallback to sample data
            self.samples = [
                {
                    'text': "Domain: Mathematics Question: What is 2+2? Answer: 4",
                    'domain': 'Mathematics',
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
        
        # Generate synthetic image (random tensor for vision-language training)
        # In real scenario, you'd load actual images or diagrams related to the problem
        image = torch.randn(3, 224, 224)
        
        # Create a label based on domain (if available)
        # Map domains to numeric labels for classification
        domain_to_label = {
            'Mathematics': 0,
            'Physics': 1,
            'Chemistry': 2,
            'Biology': 3,
            'Computer Science': 4,
            'Engineering': 5,
        }
        label = domain_to_label.get(sample.get('domain', ''), 0)
        
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
    """Configuration for training - Optimized for 600M/300M Parameter Scale"""
    def __init__(self):
        self.model_type = 'pro'  # 'lite' or 'pro'
        self.batch_size = 2  # Reduced for trillion params (48GB+ VRAM required)
        self.gradient_accumulation_steps = 4  # Simulate batch size of 8
        self.num_epochs = 15  # Reasonable for trillion scale
        self.learning_rate = 5e-5  # Lower LR for massive model
        self.max_samples = None  # None = use full dataset
        self.save_every_epoch = True
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.warmup_steps = 200  # Extended warmup for stability with huge model
        self.resume_from_checkpoint = True
        self.use_mixed_precision = True  # AMP for memory efficiency
        self.max_grad_norm = 0.5  # Tighter clipping for large models
        self.weight_decay = 0.01
        
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
        print(f"  Labels shape: {labels.shape}")
        print(f"  Logits min/max: {outputs['logits'].min():.4f} / {outputs['logits'].max():.4f}")
        print(f"  Logits contains NaN: {torch.isnan(outputs['logits']).any()}")
        print(f"  Logits contains Inf: {torch.isinf(outputs['logits']).any()}")
    
    # Classification loss with label smoothing for stability
    classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(outputs['logits'], labels)
    
    if debug:
        print(f"  Classification loss (before clamp): {classification_loss.item():.4f}")
        print(f"  Classification loss is NaN: {torch.isnan(classification_loss).any()}")
    
    # Clamp classification loss to prevent explosion
    classification_loss = torch.clamp(classification_loss, max=10.0)
    
    # Language modeling loss (predict next token)
    # Shift tokens for causal language modeling
    lm_logits = outputs['reasoning_logits'][:, :-1, :].contiguous()
    lm_targets = text_tokens[:, 1:].contiguous()
    
    if debug:
        print(f"  LM Logits shape: {lm_logits.shape}")
        print(f"  LM Targets shape: {lm_targets.shape}")
        print(f"  LM Logits min/max: {lm_logits.min():.4f} / {lm_logits.max():.4f}")
        print(f"  LM Logits contains NaN: {torch.isnan(lm_logits).any()}")
        print(f"  LM Targets unique values: {torch.unique(lm_targets[:10])}")
    
    # Only compute loss on non-padding tokens
    lm_loss = nn.CrossEntropyLoss(ignore_index=0)(
        lm_logits.view(-1, lm_logits.size(-1)),
        lm_targets.view(-1)
    )
    
    if debug:
        print(f"  LM loss (before clamp): {lm_loss.item():.4f}")
        print(f"  LM loss is NaN: {torch.isnan(lm_loss).any()}")
    
    # Clamp LM loss to prevent explosion
    lm_loss = torch.clamp(lm_loss, max=10.0)
    
    # Combined loss with weighted sum
    total_loss = (config.classification_weight * classification_loss + 
                  config.language_model_weight * lm_loss)
    
    if debug:
        print(f"  Total loss (before clamp): {total_loss.item():.4f}")
        print(f"  Total loss is NaN: {torch.isnan(total_loss).any()}")
    
    # Image reconstruction loss (Pro model only)
    if 'reconstructed_image' in outputs and config.model_type == 'pro':
        target_images = images.view(images.size(0), -1)
        reconstructed = outputs['reconstructed_image']
        reconstruction_loss = nn.MSELoss()(reconstructed, target_images)
        reconstruction_loss = torch.clamp(reconstruction_loss, max=10.0)
        total_loss += config.reconstruction_weight * reconstruction_loss
    else:
        reconstruction_loss = torch.tensor(0.0)
    
    # Final clamp on total loss
    total_loss = torch.clamp(total_loss, max=15.0)
    
    return {
        'total_loss': total_loss,
        'classification_loss': classification_loss,
        'lm_loss': lm_loss,
        'reconstruction_loss': reconstruction_loss
    }


def train_epoch(model, dataloader, optimizer, config: TrainingConfig, epoch: int):
    """Train for one epoch with gradient accumulation and mixed precision"""
    model.train()
    
    total_loss = 0
    accumulated_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    # Setup mixed precision if enabled
    scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
    
    for batch_idx, batch in enumerate(progress_bar):
        # Move to device
        text_tokens = batch['text_tokens'].to(config.device)
        images = batch['images'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        # Debug first batch only
        debug_mode = (batch_idx == 0 and epoch == 0)
        
        if debug_mode:
            print(f"\n{'='*70}")
            print(f"[DEBUG] First Batch of Training")
            print(f"{'='*70}")
            print(f"Text tokens shape: {text_tokens.shape}")
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Text tokens sample (first 20): {text_tokens[0, :20]}")
            print(f"Labels: {labels}")
            print(f"Images min/max: {images.min():.4f} / {images.max():.4f}")
            print(f"Gradient Accumulation Steps: {config.gradient_accumulation_steps}")
            print(f"Mixed Precision: {config.use_mixed_precision}")
        
        # Forward pass with mixed precision
        try:
            if config.use_mixed_precision:
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    outputs = model(text_tokens, images)
                    losses = compute_loss(outputs, labels, text_tokens, images, config, debug=debug_mode)
                    loss = losses['total_loss'] / config.gradient_accumulation_steps
            else:
                outputs = model(text_tokens, images)
                losses = compute_loss(outputs, labels, text_tokens, images, config, debug=debug_mode)
                loss = losses['total_loss'] / config.gradient_accumulation_steps
            
            if debug_mode:
                print(f"[DEBUG] Forward pass completed")
                print(f"  Output keys: {outputs.keys()}")
            
            # Check for NaN before backward pass
            if torch.isnan(loss):
                if debug_mode:
                    print(f"\n[ERROR] NaN detected in loss before backward pass!")
                    print(f"  Skipping this batch...")
                continue
            
            # Backward pass with gradient accumulation
            if config.use_mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item()
            
            # Update weights after accumulation
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                # Gradient clipping for stability
                if config.use_mixed_precision:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                
                if debug_mode:
                    # Check for NaN gradients
                    has_nan_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"  NaN gradient in: {name}")
                            has_nan_grad = True
                    if not has_nan_grad:
                        print(f"  All gradients are valid (no NaN)")
                
                # Optimizer step
                if config.use_mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()
                accumulated_loss = 0
            
            if debug_mode:
                print(f"[DEBUG] Backward pass completed")
                print(f"{'='*70}\n")
            
            # Update metrics
            total_loss += losses['total_loss'].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'cls': f"{losses['classification_loss'].item():.4f}",
                'lm': f"{losses['lm_loss'].item():.4f}"
            })
        
        except RuntimeError as e:
            print(f"\nError in batch {batch_idx}: {e}")
            print(f"Text shape: {text_tokens.shape}, Image shape: {images.shape}")
            optimizer.zero_grad()
            continue
    
    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


def train_single_model(model_type: str, tokenizer, dataset, config: TrainingConfig):
    """Train a single model type (pro or lite) - Trillion Parameter Optimized"""
    
    print("\n" + "=" * 70)
    print(f" TRAINING {model_type.upper()} MODEL (TRILLION PARAMETER SCALE)")
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
    if model_type == 'pro':
        checkpoint_name = 'Pro-model.pt'
    else:
        checkpoint_name = 'Lite-model.pt'
    
    if config.resume_from_checkpoint and os.path.exists(checkpoint_name):
        print(f"\n[LOAD] Loading existing {model_type} model from checkpoint...")
        print(f"  Checkpoint found: {checkpoint_name}")
        try:
            model, _ = load_model(checkpoint_name, tokenizer, mode=model_type)
            print(f"  ✓ Model loaded from checkpoint - resuming training")
        except Exception as e:
            print(f"  ⚠ Failed to load checkpoint: {e}")
            print(f"  Building new model instead...")
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
    
    # Setup optimizer with weight decay for regularization
    optimizer = AdamW(
        model.parameters(), 
        lr=config.learning_rate, 
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)  # Conservative momentum for large models
    )
    
    # Training loop
    print(f"\n[TRAIN] Starting training for {config.num_epochs} epochs...")
    print(f"  ├─ Batch Size: {config.batch_size}")
    print(f"  ├─ Gradient Accumulation: {config.gradient_accumulation_steps}x (eff. batch: {config.batch_size * config.gradient_accumulation_steps})")
    print(f"  ├─ Learning Rate: {config.learning_rate}")
    print(f"  ├─ Warmup Steps: {config.warmup_steps}")
    print(f"  ├─ Max Grad Norm: {config.max_grad_norm}")
    print(f"  ├─ Mixed Precision: {config.use_mixed_precision}")
    print(f"  └─ Device: {config.device}")
    
    best_loss = float('inf')
    training_history = []
    
    for epoch in range(config.num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, config, epoch)
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs} - Average Loss: {avg_loss:.4f}")
        training_history.append({'epoch': epoch+1, 'loss': avg_loss, 'model': model_type})
        
        # Save checkpoint
        if config.save_every_epoch or avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, tokenizer, checkpoint_name)
            print(f"  ✓ Saved checkpoint: {checkpoint_name}")
    
    print(f"\n✓ {model_type.upper()} MODEL TRAINING COMPLETE! Final loss: {best_loss:.4f}")
    return training_history, best_loss


def main():
    """Main training function - trains both PRO and LITE models"""
    
    print("\n" + "=" * 70)
    print(" VELCORE MODEL TRAINING - Turing-Open-Reasoning Dataset")
    print(" Training BOTH Pro and Lite models")
    print("=" * 70)
    
    # Configuration
    config = TrainingConfig()
    
    print(f"\n[CONFIG]")
    print(f"  Model Type: {config.model_type.upper()}")
    print(f"  Device: {config.device}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Max Samples: {config.max_samples}")
    
    # Step 1: Load dataset to get all text samples
    print(f"\n[STEP 1] Loading Turing-Open-Reasoning dataset...")
    
    try:
        # Load token
        token = None
        if os.path.exists("token.txt"):
            try:
                with open("token.txt", "r") as f:
                    token = f.read().strip()
            except:
                pass

        dataset_raw = load_dataset("open-thoughts/Turing-Open-Reasoning", split="train", token=token)
        
        # Extract all texts from dataset
        all_texts = []
        for i, item in enumerate(dataset_raw):
            if config.max_samples is not None and i >= config.max_samples:
                break
            
            # Extract fields
            question = item.get('question', '')
            answer = item.get('answer', '')
            domain = item.get('domain', '')
            subdomain = item.get('sub-domain', '')
            
            # Combine fields for vocabulary
            if question:
                all_texts.append(question)
            if answer:
                all_texts.append(answer)
            if domain:
                all_texts.append(domain)
            if subdomain:
                all_texts.append(subdomain)
        
        print(f"  ✓ Loaded {len(all_texts)} text samples for vocabulary building")
        
    except Exception as e:
        print(f"  Warning: Could not load dataset: {e}")
        print(f"  Using fallback vocabulary...")
        all_texts = ["Mathematics Physics Chemistry Biology Question Answer Domain"]
    
    # Step 2: Build tokenizer with comprehensive vocabulary
    print(f"\n[STEP 2] Building tokenizer vocabulary...")
    tokenizer = CustomTokenizer(max_length=512)
    
    # Add domain and reasoning keywords
    all_texts.extend([
        "Domain Question Answer Mathematics Physics Chemistry Biology",
        "Computer Science Engineering Algebra Calculus Geometry",
        "What is the solution Calculate solve compute determine find",
        "Step by step reasoning analysis explanation proof"
    ])
    
    # Build vocabulary with min_freq=1 to include all words
    tokenizer.build_vocab(all_texts, min_freq=1)
    vocab_size = len(tokenizer.vocab)
    print(f"  ✓ Tokenizer created with vocab size: {vocab_size}")
    
    # Step 3: Load dataset with proper tokenizer
    print(f"\n[STEP 3] Creating training dataset...")
    dataset = TuringReasoningDataset(tokenizer, max_samples=config.max_samples, split="train")
    
    # Validate token IDs in dataset
    print(f"\n[STEP 3.1] Validating token IDs...")
    max_token_id = 0
    invalid_samples = 0
    for i in range(min(len(dataset), 10)):  # Check first 10 samples
        sample = dataset[i]
        tokens = sample['text_tokens']
        max_id = tokens.max().item()
        max_token_id = max(max_token_id, max_id)
        
        if max_id >= vocab_size:
            print(f"  WARNING: Sample {i} has token ID {max_id} >= vocab_size {vocab_size}")
            invalid_samples += 1
    
    if invalid_samples == 0:
        print(f"  ✓ All tokens valid (max ID: {max_token_id}, vocab size: {vocab_size})")
    else:
        print(f"  ERROR: Found {invalid_samples} samples with invalid token IDs!")
        print(f"  This will cause NaN during training. Aborting...")
        return
    
    # Step 4: Train BOTH models
    all_history = []
    results = {}
    
    # Train Pro model
    print("\n" + "=" * 70)
    print(" PHASE 1: Training PRO model")
    print("=" * 70)
    pro_history, pro_loss = train_single_model('pro', tokenizer, dataset, config)
    all_history.extend(pro_history)
    results['pro'] = pro_loss
    
    # Train Lite model
    print("\n" + "=" * 70)
    print(" PHASE 2: Training LITE model")
    print("=" * 70)
    lite_history, lite_loss = train_single_model('lite', tokenizer, dataset, config)
    all_history.extend(lite_history)
    results['lite'] = lite_loss
    
    # Save tokenizer and training history
    print(f"\n[FINAL] Saving tokenizer and training history...")
    tokenizer.save('custom_tokenizer.json')
    
    with open('training_history.json', 'w') as f:
        json.dump(all_history, f, indent=2)
    
    print(f"  ✓ Saved tokenizer to custom_tokenizer.json")
    print(f"  ✓ Saved training history to training_history.json")
    
    # Final summary
    print("\n" + "=" * 70)
    print(" ✓ ALL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  PRO  Model: Pro-model.pt  (Final loss: {results['pro']:.4f})")
    print(f"  LITE Model: Lite-model.pt (Final loss: {results['lite']:.4f})")
    print(f"\nTokenizer: custom_tokenizer.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()

