"""
VELCORE UNIFIED COLAB TRAINER
Version: Claude-Style Upgrade (RMSNorm, RoPE, ViT) + TinyStories Dataset
========================================================================
This single file contains the entire Vel Insights architecture, tokenizer, and training logic.
Upload this to Google Colab and run to start training.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchvision import models, transforms
from PIL import Image
from typing import Dict, Optional, List
import math
import json
import re
import argparse
from tqdm import tqdm
from datasets import load_dataset

# ============================================================
# COLAB PRE-FLIGHT (Install dependencies)
# ============================================================
def install_requirements():
    try:
        import datasets
        import tqdm
    except ImportError:
        print("Installing dependencies...")
        import subprocess
        subprocess.check_call(["pip", "install", "datasets", "tqdm", "torchvision", "pillow", "tokenizers"])
        print("✓ Dependencies installed")

# ============================================================
# CUSTOM TOKENIZER
# ============================================================
try:
    from tokenizers import Tokenizer, models as tk_models, trainers, pre_tokenizers, decoders, processors
    TOKENIZERS_AVAILABLE = True
except ImportError:
    TOKENIZERS_AVAILABLE = False

class CustomTokenizer:
    def __init__(self, vocab_path: str = None, max_length: int = 512, vocab_size: int = 32000):
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.special_tokens = ['[THINK_START]', '[THINK_END]', '[REASON_START]', '[REASON_END]', '[PAD]', '[UNK]', '[CLS]', '[SEP]', '[IMG]']
        
        if TOKENIZERS_AVAILABLE:
            if vocab_path and os.path.exists(vocab_path):
                self.tokenizer = Tokenizer.from_file(vocab_path)
            else:
                self.tokenizer = Tokenizer(tk_models.BPE(unk_token="[UNK]"))
                self.tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
                self.tokenizer.decoder = decoders.ByteLevel()
                self.tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
        else:
            self.tokenizer = None
            
        self.vocab = {}
        self.id_to_token = {}
        if vocab_path and os.path.exists(vocab_path): self.load(vocab_path)

    def build_vocab(self, texts: List[str], min_freq: int = 2):
        if TOKENIZERS_AVAILABLE:
            trainer = trainers.BpeTrainer(vocab_size=self.vocab_size, min_frequency=min_freq, special_tokens=self.special_tokens)
            self.tokenizer.train_from_iterator(texts, trainer=trainer)
            self.vocab = self.tokenizer.get_vocab()
            self.id_to_token = {v: k for k, v in self.vocab.items()}
            print(f"✓ BPE Tokenizer trained ({len(self.vocab)} tokens)")
        else:
            self.vocab = {t: i for i, t in enumerate(self.special_tokens)}
            print("⚠ Fallback word-level used (tokenizers missing)")

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        if TOKENIZERS_AVAILABLE and self.tokenizer:
            tokens = self.tokenizer.encode(text).ids
            if add_special_tokens: tokens = [self.vocab.get('[CLS]', 6)] + tokens + [self.vocab.get('[SEP]', 7)]
        else:
            tokens = [self.vocab.get(w, 5) for w in text.split()]
        tokens = tokens[:self.max_length]
        return tokens + [self.vocab.get('[PAD]', 4)] * (self.max_length - len(tokens))

    def decode(self, tokens: List[int]) -> str:
        clean = [t for t in tokens if t in self.id_to_token and not (self.id_to_token[t].startswith('[') and self.id_to_token[t].endswith(']'))]
        return self.tokenizer.decode(clean) if TOKENIZERS_AVAILABLE and self.tokenizer else ""

    def save(self, path: str):
        if TOKENIZERS_AVAILABLE and self.tokenizer: self.tokenizer.save(path)
        else:
            with open(path, 'w') as f: json.dump({'vocab': self.vocab}, f)

    def load(self, path: str):
        if TOKENIZERS_AVAILABLE:
            self.tokenizer = Tokenizer.from_file(path)
            self.vocab = self.tokenizer.get_vocab()
            self.id_to_token = {v: k for k, v in self.vocab.items()}
        else:
            with open(path, 'r') as f: self.vocab = json.load(f).get('vocab', {})
            self.id_to_token = {v: k for k, v in self.vocab.items()}

# ============================================================
# ADVANCED MODULES (RMSNorm, RoPE, ViT Integration)
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    def forward(self, x):
        return self._norm(x.float()).type_as(x) * self.weight

def precompute_freqs_cis(dim: int, end: int):
    freqs = 1.0 / (10000.0 ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    return torch.polar(torch.ones_like(freqs), freqs)

def apply_rotary_emb(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    ndim = xq_.ndim
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(xq_.shape)]
    freqs_cis = freqs_cis.view(*shape)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RoPEAttention(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x, freqs_cis=None, mask=None):
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_heads, self.head_dim)
        if freqs_cis is not None: xq, xk = apply_rotary_emb(xq, xk, freqs_cis)
        xq, xk, xv = [t.transpose(1, 2) for t in [xq, xk, xv]]
        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        if mask is not None: scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2), -1e9)
        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        output = torch.matmul(scores, xv)
        return self.wo(output.transpose(1, 2).contiguous().view(bsz, seqlen, -1))

class ThinkingLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.attention = RoPEAttention(hidden_size, num_heads)
        self.norm1, self.norm2 = RMSNorm(hidden_size), RMSNorm(hidden_size)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size*4), nn.GELU(), nn.Linear(hidden_size*4, hidden_size))
    def forward(self, x, freqs_cis=None, mask=None):
        x = self.norm1(x + self.attention(x, freqs_cis, mask))
        return self.norm2(x + self.ffn(x))

class ReasoningModule(nn.Module):
    def __init__(self, hidden_size, num_steps=3, num_heads=8):
        super().__init__()
        self.layers = nn.ModuleList([ThinkingLayer(hidden_size, num_heads) for _ in range(num_steps)])
        self.gate = nn.Sequential(nn.Linear(hidden_size*2, hidden_size), nn.Sigmoid())
    def forward(self, x, context, freqs_cis=None, mask=None):
        reason_state = x
        for layer in self.layers:
            g = self.gate(torch.cat([reason_state, context[:, :x.shape[1]]], dim=-1))
            reason_state = g * layer(reason_state, freqs_cis, mask) + (1-g) * x
        return reason_state

class SimpleFusionLayer(nn.Module):
    def __init__(self, hidden_size, num_heads=8):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(hidden_size, num_heads, batch_first=True)
        self.norm1, self.norm2 = RMSNorm(hidden_size), RMSNorm(hidden_size)
        self.ffn = nn.Sequential(nn.Linear(hidden_size, hidden_size*2), nn.GELU(), nn.Linear(hidden_size*2, hidden_size))
    def forward(self, t, v):
        a, _ = self.cross_attn(t, v, v)
        t = self.norm1(t + 0.1 * a)
        return self.norm2(t + 0.1 * self.ffn(t))

class MultimodalEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.text_emb = nn.Embedding(config['vocab_size'], config['hidden_size'], padding_idx=0)
        self.text_pos = nn.Parameter(torch.randn(1, 8192, config['hidden_size']) * 0.02)
        self.emb_norm = RMSNorm(config['hidden_size'])
        self.visual_encoder = models.vit_b_32(weights=None) if config['mode'] == 'lite' else models.vit_b_16(weights=None)
        self.visual_encoder.heads = nn.Identity()
        self.visual_proj = nn.Linear(768, config['hidden_size'])
        self.fusion_layers = nn.ModuleList([SimpleFusionLayer(config['hidden_size'], config['num_heads']) for _ in range(config['num_fusion_layers'])])
        self.reasoning = ReasoningModule(config['hidden_size'], config['reasoning_steps'], config['num_heads'])
        self.config = config

    def forward(self, t_in, i_in):
        t = self.emb_norm(self.text_emb(t_in) + self.text_pos[:, :t_in.shape[1], :])
        v = self.visual_proj(self.visual_encoder(i_in)).unsqueeze(1)
        f = t
        for layer in self.fusion_layers: f = layer(f, v)
        f = torch.cat([f, v], dim=1)
        freqs_cis = precompute_freqs_cis(self.config['hidden_size'] // self.config['num_heads'], t_in.shape[1]).to(t_in.device)
        mask = (t_in == 0).to(torch.bool) if (t_in == 0).any() else None
        r = self.reasoning(f[:, :t_in.shape[1]], f, freqs_cis, mask)
        return {'reasoned_features': r}

class VelCoreModel(nn.Module):
    def __init__(self, mode='lite', vocab_size=1000):
        super().__init__()
        self.mode = mode
        self.config = {
            'hidden_size': 512 if mode == 'lite' else 1024,
            'vocab_size': vocab_size,
            'num_heads': 8 if mode == 'lite' else 16,
            'num_fusion_layers': 3 if mode == 'lite' else 6,
            'reasoning_steps': 4 if mode == 'lite' else 8,
            'mode': mode
        }
        self.encoder = MultimodalEncoder(self.config)
        self.gen_norm = RMSNorm(self.config['hidden_size'])
        self.gen_head = nn.Linear(self.config['hidden_size'], vocab_size)
    def forward(self, text_input, image_input):
        enc = self.encoder(text_input, image_input)
        logits = self.gen_head(self.gen_norm(enc['reasoned_features']))
        return {'reasoning_logits': logits}

# ============================================================
# DATASET & UTILS
# ============================================================
# Anthropic-Style Constitution Foundation
SYSTEM_PROMPT = (
    "You are VelCore, a highly intelligent and helpful AI assistant. "
    "You provide accurate, reasoned, and honest information. "
    "Analyze tasks carefully before responding."
)

def format_item(item):
    """Formats raw data into Anthropic-style instruction blocks"""
    if "conversation" in item:
        history = "\n".join(f"{t.get('role','').capitalize()}: {t.get('content','')}" for t in item["conversation"])
        return f"System: {SYSTEM_PROMPT}\n{history}"
    if "text" in item: 
        return f"System: {SYSTEM_PROMPT}\nContext: {item['text']}\nAssistant: Let me think. [THINK_START] {item['text'][:100]}... [THINK_END]"
    if "prompt" in item: 
        return f"System: {SYSTEM_PROMPT}\nUser: {item['prompt']}\nAssistant: [THINK_START] Analysis. [THINK_END] {item['response']}"
    return None

class ConversationDataset(Dataset):
    def __init__(self, tokenizer, hf_token, max_samples=5000):
        self.tokenizer = tokenizer
        self.samples = []
        print("Loading TinyStories...")
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True, token=hf_token)
        for i, item in enumerate(ds):
            if i >= max_samples: break
            self.samples.append(item['text'])
        print(f"✓ Loaded {len(self.samples)} samples")

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        return torch.tensor(self.tokenizer.encode(self.samples[idx]), dtype=torch.long)

def collate_fn(batch):
    max_len = max(len(x) for x in batch)
    ids = torch.stack([torch.cat([x, torch.zeros(max_len - len(x), dtype=torch.long)]) for x in batch])
    return {"input_ids": ids}

# ============================================================
# TRAINING LOGIC
# ============================================================
def run_training(hf_token, mode='lite', epochs=5, force_device=None):
    install_requirements()
    
    # Mount Google Drive if in Colab
    if os.path.exists('/content'):
        try:
            from google.colab import drive
            print("Mounting Google Drive...")
            drive.mount('/content/drive')
        except ImportError:
            print("Not in Colab or Drive mounting failed.")

    if force_device:
        device = torch.device(force_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"Using device: {device}")
    
    model_path = f"{mode}-model.pt"
    tokenizer_path = "tokenizer.json"

    # 1. Tokenizer (Load or Build)
    tokenizer = CustomTokenizer()
    retrain = False
    if os.path.exists(tokenizer_path):
        use_ext = input(f"Existing tokenizer found at {tokenizer_path}. Retrain for new dataset? (y/n) [n]: ").strip().lower() == 'y'
        if use_ext: retrain = True
        else:
            print(f"✓ Loading existing tokenizer: {tokenizer_path}")
            tokenizer.load(tokenizer_path)
    else: retrain = True

    if retrain:
        print("Building new tokenizer from dataset...")
        ds = load_dataset("roneneldan/TinyStories", split="train", streaming=True, token=hf_token)
        texts = [item['text'] for _, item in zip(range(1000), ds)]
        tokenizer.build_vocab(texts)
        tokenizer.save(tokenizer_path)
    
    # 2. Model (Build and optionally Load)
    model = VelCoreModel(mode=mode, vocab_size=len(tokenizer.vocab)).to(device)
    
    if os.path.exists(model_path):
        print(f"✓ Resuming from checkpoint: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats (Unified vs Builder)
        state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
        
        # Vocab Adaptation: Resize if saved vocab size differs from current
        current_vocab_size = len(tokenizer.vocab)
        
        # 1. Resize Text Embedding layer
        if 'encoder.text_emb.weight' in state_dict:
            saved_size = state_dict['encoder.text_emb.weight'].shape[0]
            if saved_size != current_vocab_size:
                print(f"  ⚠ Resizing text_emb: {saved_size} -> {current_vocab_size}")
                old_weight = state_dict['encoder.text_emb.weight']
                new_weight = model.encoder.text_emb.weight.data.clone()
                min_v = min(saved_size, current_vocab_size)
                new_weight[:min_v] = old_weight[:min_v]
                state_dict['encoder.text_emb.weight'] = new_weight

        # 2. Resize Generation Head layer
        if 'gen_head.weight' in state_dict:
            saved_size = state_dict['gen_head.weight'].shape[0]
            if saved_size != current_vocab_size:
                print(f"  ⚠ Resizing gen_head: {saved_size} -> {current_vocab_size}")
                # Weight
                old_w = state_dict['gen_head.weight']
                new_w = model.gen_head.weight.data.clone()
                min_v = min(saved_size, current_vocab_size)
                new_w[:min_v] = old_w[:min_v]
                state_dict['gen_head.weight'] = new_w
                # Bias
                if 'gen_head.bias' in state_dict:
                    old_b = state_dict['gen_head.bias']
                    new_b = model.gen_head.bias.data.clone()
                    new_b[:min_v] = old_b[:min_v]
                    state_dict['gen_head.bias'] = new_b
        
        msg = model.load_state_dict(state_dict, strict=False)
        if msg.missing_keys: print(f"  ⚠ Missing keys: {len(msg.missing_keys)}")
        if msg.unexpected_keys: print(f"  ⚠ Unexpected keys: {len(msg.unexpected_keys)}")
    else:
        print(f"✓ Created new {mode} model")

    # Optimization: Advanced Weight Decay (Anthropic Standard)
    decay = []
    no_decay = []
    for n, p in model.named_parameters():
        if not p.requires_grad: continue
        if any(nd in n for nd in ["bias", "Norm", "norm"]): no_decay.append(p)
        else: decay.append(p)
    
    optimizer = AdamW([
        {"params": decay, "weight_decay": 0.01},
        {"params": no_decay, "weight_decay": 0.0}
    ], lr=5e-5, betas=(0.9, 0.95))
    
    print("Loading dataset...")
    dataset = ConversationDataset(tokenizer, hf_token)
    # Reduced batch size for T4 compatibility
    batch_size = 2
    grad_accum_steps = 4 
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    # Scheduler: Cosine with Warmup (Anthropic Scale Standard)
    num_steps = len(loader) * epochs
    warmup = int(num_steps * 0.1)
    def lr_lambda(s):
        if s < warmup: return s / warmup
        return 0.5 * (1.0 + math.cos(math.pi * (s - warmup) / (num_steps - warmup)))
    scheduler = LambdaLR(optimizer, lr_lambda)

    # Use bfloat16 if GPU supports it (more stable)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    # Memory Optimization: Clear cache before starting
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}")
        
        for i, batch in enumerate(pbar):
            ids = batch["input_ids"].to(device)
            # Efficient zeroing: use simpler constant for dummy images
            images = torch.zeros(ids.shape[0], 3, 224, 224, device=device)
            
            with torch.autocast(device_type='cuda' if 'cuda' in str(device) else 'cpu', dtype=dtype):
                out = model(ids, images)['reasoning_logits']
                loss = F.cross_entropy(out[:, :-1].reshape(-1, out.size(-1)), ids[:, 1:].reshape(-1), ignore_index=0)
                loss = loss / grad_accum_steps
            
            loss.backward()
            
            if (i + 1) % grad_accum_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            total_loss += loss.item() * grad_accum_steps
            pbar.set_postfix(loss=f"{loss.item() * grad_accum_steps:.4f}", lr=f"{scheduler.get_last_lr()[0]:.2e}")

        # Periodically clear VRAM between epochs
        if torch.cuda.is_available(): torch.cuda.empty_cache()
        
        # Save comprehensive checkpoint (Standardized with Builder format)
        torch.save({
            'state_dict': model.state_dict(),
            'vocab': tokenizer.vocab,
            'epoch': epoch + 1,
            'mode': mode,
            'config': model.config
        }, model_path)
        tokenizer.save(tokenizer_path)
        
        print(f"Epoch {epoch+1} done. Avg Loss: {total_loss/len(loader):.4f}")
        print(f"✓ Checkpoints updated: {model_path}, {tokenizer_path}")

        # --- Google Drive Sync ---
        drive_path = "/content/drive/MyDrive/velcore_models"
        if os.path.exists("/content/drive"):
            os.makedirs(drive_path, exist_ok=True)
            try:
                import shutil
                shutil.copy(model_path, os.path.join(drive_path, model_path))
                shutil.copy(tokenizer_path, os.path.join(drive_path, tokenizer_path))
                print(f"💾 Synced to Drive: {drive_path}")
            except Exception as e:
                print(f"⚠ Drive sync failed: {e}")


# ============================================================
# MAIN ENTRY FOR COLAB
# ============================================================
if __name__ == "__main__":
    print("Welcome to VelCore Colab Trainer!")
    token = input("Enter your Hugging Face Token: ").strip()
    if not token:
        print("Error: HF Token required.")
    else:
        m = input("Model mode (lite/pro) [lite]: ").strip().lower() or "lite"
        e = int(input("Epochs [5]: ").strip() or "5")
        d = input("Device (cuda/cpu) [auto]: ").strip().lower() or None
        run_training(token, mode=m, epochs=e, force_device=d)
