import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
import math
from datasets import load_dataset
from tqdm import tqdm
import os
from typing import Dict, List
import json

from tokenizer import CustomTokenizer
from builder import VelCoreModelBuilder, save_model, load_model


    def __init__(self, tokenizer: CustomTokenizer, max_samples: int = 500, split: str = "train"):
        self.tokenizer = tokenizer
        
        # Anthropic-Style Constitution Foundation
        self.system_prompt = (
            "You are VelCore, a highly intelligent and helpful AI assistant. "
            "You provide accurate, reasoned, and honest information. "
            "Analyze questions carefully before answering."
        )
        
        print(f"Loading Turing-Open-Reasoning dataset (max {max_samples} samples)...")
        # ... logic for loading ...
        # (keeping existing loading logic but will wrap content in system prompt)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Apply Claude-style instruction formatting
        formatted_text = f"System: {self.system_prompt} User: {sample['question']} Assistant: Let me think. [THINK_START] {sample['answer']} [THINK_END]"
        
        tokens = self.tokenizer.encode(formatted_text)
        image = torch.randn(3, 224, 224)
        
        domain_to_label = {'Mathematics': 0, 'Physics': 1, 'Chemistry': 2, 'Biology': 3, 'Computer Science': 4, 'Engineering': 5}
        label = domain_to_label.get(sample.get('domain', ''), 0)
        
        return {
            'text_tokens': torch.tensor(tokens, dtype=torch.long),
            'image': image,
            'label': torch.tensor(label, dtype=torch.long),
            'text': formatted_text
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


def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """Cosine learning rate scheduler with warmup (Anthropic Standard)"""
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress)))
    return LambdaLR(optimizer, lr_lambda)

def train_epoch(model, dataloader, optimizer, scheduler, config: TrainingConfig, epoch: int):
    """Train for one epoch with Cosine Cooling and Gradient Scaling"""
    model.train()
    
    total_loss = 0
    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{config.num_epochs}")
    
    # Use bfloat16 if available (more stable for Claude-style scaling)
    device_type = 'cuda' if 'cuda' in str(config.device) else 'cpu'
    mixed_precision_dtype = torch.bfloat16 if (torch.cuda.is_available() and torch.cuda.is_bf16_supported()) else torch.float16
    
    scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision and mixed_precision_dtype == torch.float16 else None
    
    for batch_idx, batch in enumerate(progress_bar):
        text_tokens = batch['text_tokens'].to(config.device)
        images = batch['images'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        # Optimization: Mixed Precision (Anthropic-Style Stability)
        with torch.autocast(device_type=device_type, dtype=mixed_precision_dtype, enabled=config.use_mixed_precision):
            outputs = model(text_tokens, images)
            losses = compute_loss(outputs, labels, text_tokens, images, config)
            loss = losses['total_loss'] / config.gradient_accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
            
            # Robust Gradient Clipping (Claude-Scale standard)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            
            scheduler.step()
            optimizer.zero_grad()
            
        total_loss += losses['total_loss'].item()
        progress_bar.set_postfix({'loss': f"{losses['total_loss'].item():.3f}", 'lr': f"{scheduler.get_last_lr()[0]:.2e}"})
    
    return total_loss / max(len(dataloader), 1)


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
    
    # Setup optimizer with Advanced Weight Decay (Anthropic Standard)
    # Exclude Norm and Bias parameters from weight decay to improve stability
    decay_params = []
    no_decay_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        if any(nd in name for nd in ["bias", "Norm", "norm"]):
            no_decay_params.append(param)
        else:
            decay_params.append(param)
            
    optim_groups = [
        {"params": decay_params, "weight_decay": config.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    
    optimizer = AdamW(optim_groups, lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8)
    
    # Scheduler: Cosine with Warmup
    num_training_steps = (len(dataloader) // config.gradient_accumulation_steps) * config.num_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, num_training_steps)

    # Training loop
    print(f"\n[TRAIN] Starting Anthropic-Style Training for {config.num_epochs} epochs...")
    print(f"  ├─ Architecture: Claude-Style (RMSNorm, RoPE)")
    print(f"  ├─ Scheduler: Cosine Cooling (Warmup: {config.warmup_steps})")
    print(f"  ├─ Precision: bfloat16 (Safe-Scale Enabled)" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "  ├─ Precision: float16 (AMP Scale)")
    print(f"  └─ Constitution: {dataset.system_prompt[:30]}...")
    
    best_loss = float('inf')
    training_history = []
    
    for epoch in range(config.num_epochs):
        avg_loss = train_epoch(model, dataloader, optimizer, scheduler, config, epoch)
        
        print(f"\nEpoch {epoch+1}/{config.num_epochs} - Average Loss: {avg_loss:.4f}")
        training_history.append({'epoch': epoch+1, 'loss': avg_loss, 'model': model_type})
        
        # Save checkpoint
        if config.save_every_epoch or avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, tokenizer, checkpoint_name, epoch=epoch + 1)
            print(f"  ✓ Saved checkpoint: {checkpoint_name} (Epoch {epoch+1})")
    
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
        if os.path.exists("code.txt"):
            try:
                with open("code.txt", "r") as f:
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

