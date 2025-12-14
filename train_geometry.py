import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from datasets import load_dataset
from tqdm import tqdm
import os
from typing import Dict, List
import json
from PIL import Image
import torchvision.transforms as transforms

from tokenizer import CustomTokenizer
from builder import VelCoreModelBuilder, save_model, load_model


class Geometry3KDataset(Dataset):
    """Dataset loader for geometry3k with REAL images"""
    
    def __init__(self, tokenizer: CustomTokenizer, max_samples: int = 500, split: str = "train"):
        """
        Args:
            tokenizer: CustomTokenizer instance
            max_samples: Maximum number of samples to use
            split: Dataset split to use ('train', 'validation', or 'test')
        """
        self.tokenizer = tokenizer
        
        # Image transform for preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"Loading geometry3k dataset (max {max_samples} samples)...")
        
        # Load dataset from Hugging Face
        try:
            dataset = load_dataset("hiyouga/geometry3k", split=split)
            
            # Process samples
            self.samples = []
            count = 0
            for item in dataset:
                if count >= max_samples:
                    break
                
                # Extract fields from geometry3k
                # Fields may include: image, question, choices, answer, etc.
                question = item.get('question', '')
                choices = item.get('choices', [])
                answer = item.get('answer', '')
                image = item.get('image', None)  # PIL Image
                
                # Create text combining question, choices, and answer
                text_parts = ["Domain: Geometry"]
                
                if question:
                    text_parts.append(f"Question: {question}")
                
                if choices:
                    if isinstance(choices, list):
                        choices_str = " | ".join([f"({chr(65+i)}) {c}" for i, c in enumerate(choices)])
                        text_parts.append(f"Choices: {choices_str}")
                
                if answer is not None:
                    text_parts.append(f"Answer: {answer}")
                
                text = " ".join(text_parts)
                
                if text.strip() and image is not None:
                    self.samples.append({
                        'text': text,
                        'image': image,  # Store PIL Image
                        'question': question,
                        'answer': answer
                    })
                    count += 1
            
            print(f"✓ Loaded {len(self.samples)} samples from geometry3k with REAL images")
            
        except Exception as e:
            print(f"Warning: Could not load dataset: {e}")
            print("Using fallback sample data with synthetic images...")
            self.samples = [
                {
                    'text': "Domain: Geometry Question: What is the area of a circle with radius 5? Answer: 78.54",
                    'image': None,
                    'question': 'What is the area of a circle with radius 5?',
                    'answer': '78.54'
                }
            ] * max_samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        text = sample['text']
        
        # Tokenize text
        tokens = self.tokenizer.encode(text)
        
        # Process real image or create synthetic
        if sample['image'] is not None:
            try:
                image = sample['image']
                # Convert to RGB if needed
                if image.mode != 'RGB':
                    image = image.convert('RGB')
                image_tensor = self.transform(image)
            except Exception as e:
                print(f"Warning: Failed to process image: {e}")
                image_tensor = torch.randn(3, 224, 224)
        else:
            # Fallback to synthetic image
            image_tensor = torch.randn(3, 224, 224)
        
        # For geometry3k, we use a single label (Geometry = 0)
        # Could extend to classify by problem type if available
        label = 0
        
        return {
            'text_tokens': torch.tensor(tokens, dtype=torch.long),
            'image': image_tensor,
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
        self.batch_size = 2  # Reduced for trillion params
        self.gradient_accumulation_steps = 4  # Simulate batch size of 8
        self.num_epochs = 12  # Medium epochs for 2000 samples
        self.learning_rate = 5e-5  # Lower LR for massive model
        self.max_samples = 2000  # geometry3k has 2,101 train samples
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
    
    if debug:
        print(f"\n[DEBUG] Loss Computation:")
        print(f"  Logits shape: {outputs['logits'].shape}")
        print(f"  Labels shape: {labels.shape}")
        print(f"  Logits min/max: {outputs['logits'].min():.4f} / {outputs['logits'].max():.4f}")
        print(f"  Logits contains NaN: {torch.isnan(outputs['logits']).any()}")
        print(f"  Logits contains Inf: {torch.isinf(outputs['logits']).any()}")
    
    # Classification loss with label smoothing
    classification_loss = nn.CrossEntropyLoss(label_smoothing=0.1)(outputs['logits'], labels)
    
    if debug:
        print(f"  Classification loss (before clamp): {classification_loss.item():.4f}")
        print(f"  Classification loss is NaN: {torch.isnan(classification_loss).any()}")
    
    classification_loss = torch.clamp(classification_loss, max=10.0)
    
    # Language modeling loss
    lm_logits = outputs['reasoning_logits'][:, :-1, :].contiguous()
    lm_targets = text_tokens[:, 1:].contiguous()
    
    if debug:
        print(f"  LM Logits shape: {lm_logits.shape}")
        print(f"  LM Targets shape: {lm_targets.shape}")
        print(f"  LM Logits min/max: {lm_logits.min():.4f} / {lm_logits.max():.4f}")
        print(f"  LM Logits contains NaN: {torch.isnan(lm_logits).any()}")
    
    lm_loss = nn.CrossEntropyLoss(ignore_index=0)(
        lm_logits.view(-1, lm_logits.size(-1)),
        lm_targets.view(-1)
    )
    
    if debug:
        print(f"  LM loss (before clamp): {lm_loss.item():.4f}")
        print(f"  LM loss is NaN: {torch.isnan(lm_loss).any()}")
    
    lm_loss = torch.clamp(lm_loss, max=10.0)
    
    # Combined loss
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
    
    scaler = torch.cuda.amp.GradScaler() if config.use_mixed_precision else None
    
    for batch_idx, batch in enumerate(progress_bar):
        text_tokens = batch['text_tokens'].to(config.device)
        images = batch['images'].to(config.device)
        labels = batch['labels'].to(config.device)
        
        debug_mode = (batch_idx == 0 and epoch == 0)
        
        if debug_mode:
            print(f"\n{'='*70}")
            print(f"[DEBUG] First Batch - Geometry Training (Trillion Param Optimized)")
            print(f"{'='*70}")
            print(f"Text tokens shape: {text_tokens.shape}")
            print(f"Images shape: {images.shape}")
            print(f"Labels shape: {labels.shape}")
            print(f"Gradient Accumulation: {config.gradient_accumulation_steps}x")
            print(f"Mixed Precision: {config.use_mixed_precision}")
        
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
            
            if torch.isnan(loss):
                if debug_mode:
                    print(f"\n[ERROR] NaN detected in loss!")
                continue
            
            if config.use_mixed_precision:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            accumulated_loss += loss.item()
            
            if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                if config.use_mixed_precision:
                    scaler.unscale_(optimizer)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
                
                if debug_mode:
                    has_nan_grad = False
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"  NaN gradient in: {name}")
                            has_nan_grad = True
                    if not has_nan_grad:
                        print(f"  All gradients are valid (no NaN)")
                
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
        
        except RuntimeError as e:
            print(f"\nError in batch {batch_idx}: {e}")
            optimizer.zero_grad()
            continue
        
        total_loss += losses['total_loss'].item()
        
        progress_bar.set_postfix({
            'loss': f"{losses['total_loss'].item():.4f}",
            'cls': f"{losses['classification_loss'].item():.4f}",
            'lm': f"{losses['lm_loss'].item():.4f}"
        })
    
    avg_loss = total_loss / max(len(dataloader), 1)
    return avg_loss


def train_single_model(model_type: str, tokenizer, dataset, config: TrainingConfig):
    """Train a single model type (pro or lite) - Trillion Parameter Optimized"""
    
    print("\n" + "=" * 70)
    print(f" TRAINING {model_type.upper()} MODEL with Geometry3K (TRILLION PARAMETER SCALE)")
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
    
    checkpoint_name = f'VelCore-{model_type.capitalize()}.pt'
    
    if config.resume_from_checkpoint and os.path.exists(checkpoint_name):
        print(f"\n[LOAD] Loading existing {model_type} model from checkpoint...")
        print(f"  Checkpoint found: {checkpoint_name}")
        try:
            model = load_model(checkpoint_name, tokenizer, mode=model_type)
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
    
    optimizer = AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95)
    )
    
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
        
        if config.save_every_epoch or avg_loss < best_loss:
            best_loss = avg_loss
            save_model(model, tokenizer, checkpoint_name)
            print(f"  ✓ Saved checkpoint: {checkpoint_name}")
    
    print(f"\n✓ {model_type.upper()} MODEL TRAINING COMPLETE! Final loss: {best_loss:.4f}")
    return training_history, best_loss


def main():
    """Main training function - trains both PRO and LITE models with Geometry3K"""
    
    print("\n" + "=" * 70)
    print(" VELCORE MODEL TRAINING - Geometry3K Dataset")
    print(" Training with REAL geometry diagram images")
    print("=" * 70)
    
    config = TrainingConfig()
    
    print(f"\n[CONFIG]")
    print(f"  Model Type: {config.model_type.upper()}")
    print(f"  Device: {config.device}")
    print(f"  Batch Size: {config.batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Max Samples: {config.max_samples}")
    
    # Step 1: Load geometry3k dataset
    print(f"\n[STEP 1] Loading Geometry3K dataset...")
    
    try:
        dataset_raw = load_dataset("hiyouga/geometry3k", split="train")
        
        all_texts = []
        for i, item in enumerate(dataset_raw):
            if i >= config.max_samples:
                break
            
            question = item.get('question', '')
            choices = item.get('choices', [])
            answer = item.get('answer', '')
            
            if question:
                all_texts.append(question)
            if isinstance(choices, list):
                for c in choices:
                    if c:
                        all_texts.append(str(c))
            if answer:
                all_texts.append(str(answer))
        
        print(f"  ✓ Loaded {len(all_texts)} text samples for vocabulary building")
        
    except Exception as e:
        print(f"  Warning: Could not load dataset: {e}")
        print(f"  Using fallback vocabulary...")
        all_texts = ["Geometry angle triangle circle area perimeter"]
    
    # Step 2: Build tokenizer
    print(f"\n[STEP 2] Building tokenizer vocabulary...")
    tokenizer = CustomTokenizer(max_length=512)
    
    all_texts.extend([
        "Domain Geometry Question Answer Triangle Circle Rectangle",
        "angle degree radian perpendicular parallel",
        "area perimeter circumference diameter radius",
        "theorem proof solve calculate find determine"
    ])
    
    tokenizer.build_vocab(all_texts, min_freq=1)
    vocab_size = len(tokenizer.vocab)
    print(f"  ✓ Tokenizer created with vocab size: {vocab_size}")
    
    # Step 3: Create dataset with REAL images
    print(f"\n[STEP 3] Creating training dataset with REAL images...")
    dataset = Geometry3KDataset(tokenizer, max_samples=config.max_samples, split="train")
    
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
    
    print("\n" + "=" * 70)
    print(" PHASE 1: Training PRO model with Geometry3K")
    print("=" * 70)
    pro_history, pro_loss = train_single_model('pro', tokenizer, dataset, config)
    all_history.extend(pro_history)
    results['pro'] = pro_loss
    
    print("\n" + "=" * 70)
    print(" PHASE 2: Training LITE model with Geometry3K")
    print("=" * 70)
    lite_history, lite_loss = train_single_model('lite', tokenizer, dataset, config)
    all_history.extend(lite_history)
    results['lite'] = lite_loss
    
    # Save tokenizer and history
    print(f"\n[FINAL] Saving tokenizer and training history...")
    tokenizer.save('geometry_tokenizer.json')
    
    with open('geometry_training_history.json', 'w') as f:
        json.dump(all_history, f, indent=2)
    
    print(f"  ✓ Saved tokenizer to geometry_tokenizer.json")
    print(f"  ✓ Saved training history to geometry_training_history.json")
    
    # Final summary
    print("\n" + "=" * 70)
    print(" ✓ ALL TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 70)
    print(f"\nResults:")
    print(f"  PRO  Model: VelCore-Pro_geometry.pt  (Final loss: {results['pro']:.4f})")
    print(f"  LITE Model: VelCore-Lite_geometry.pt (Final loss: {results['lite']:.4f})")
    print(f"\nTokenizer: geometry_tokenizer.json")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
