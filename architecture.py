import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Dict, Optional
import math
import torch.nn.functional as F

# ==================== Image Preprocessor ====================
class ImagePreprocessor:
    def __init__(self, image_size: int = 224):
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                              std=[0.229, 0.224, 0.225])
        ])
    
    def process(self, image_path: str) -> torch.Tensor:
        """Process image from path"""
        image = Image.open(image_path).convert('RGB')
        return self.transform(image).unsqueeze(0)

# ==================== Advanced Modules ====================
class RMSNorm(nn.Module):
    """Root Mean Square Normalization for faster, more stable training"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    """Precompute the frequency tensor for RoPE"""
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)
    freqs = torch.outer(t, freqs).float()
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    """Reshape freqs_cis for broadcasting with x"""
    ndim = x.ndim
    assert 0 <= 1 < ndim
    assert freqs_cis.shape == (x.shape[1], x.shape[-1])
    shape = [d if i == 1 or i == ndim - 1 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)

def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    """Apply rotary embeddings to query and key tensors"""
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

class RoPEAttention(nn.Module):
    """Multi-head attention with Rotary Positional Embeddings"""
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert self.head_dim * num_heads == hidden_size

        self.wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wv = nn.Linear(hidden_size, hidden_size, bias=False)
        self.wo = nn.Linear(hidden_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        bsz, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        xq = xq.view(bsz, seqlen, self.num_heads, self.head_dim)
        xk = xk.view(bsz, seqlen, self.num_heads, self.head_dim)
        xv = xv.view(bsz, seqlen, self.num_heads, self.head_dim)

        if freqs_cis is not None:
            xq, xk = apply_rotary_emb(xq, xk, freqs_cis)

        xq = xq.transpose(1, 2)
        xk = xk.transpose(1, 2)
        xv = xv.transpose(1, 2)

        scores = torch.matmul(xq, xk.transpose(2, 3)) / math.sqrt(self.head_dim)
        
        if mask is not None:
            # Convert boolean mask to additive mask
            # mask: [batch, seqlen] -> [batch, 1, 1, seqlen]
            mask_expanded = mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask_expanded, -1e9)

        scores = F.softmax(scores.float(), dim=-1).type_as(xq)
        scores = self.dropout(scores)
        
        output = torch.matmul(scores, xv)
        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, -1)
        return self.wo(output)

# ==================== Model Architecture ====================
class ThinkingLayer(nn.Module):
    """Layer for thinking/reasoning process with RoPE"""
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attention = RoPEAttention(hidden_size, num_heads, dropout=dropout)
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention for thinking with RoPE
        attn_out = self.attention(x, freqs_cis, mask)
        x = self.norm1(x + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x

class ReasoningModule(nn.Module):
    """Module for multi-step reasoning"""
    def __init__(self, hidden_size: int, num_steps: int = 3, num_heads: int = 8):
        super().__init__()
        self.num_steps = num_steps
        self.hidden_size = hidden_size
        self.thinking_layers = nn.ModuleList([
            ThinkingLayer(hidden_size, num_heads) for _ in range(num_steps)
        ])
        
        # Context projection to match reasoning state
        self.context_projection = nn.Linear(hidden_size, hidden_size)
        
        # Reasoning gate
        self.reason_gate = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Sigmoid()
        )
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, freqs_cis: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Initial reasoning state
        reason_state = x
        
        # Project context to match reasoning state shape
        if context.shape[1] != reason_state.shape[1]:
            context_proj = context.mean(dim=1, keepdim=True).expand_as(reason_state)
        else:
            context_proj = context
        
        context_proj = self.context_projection(context_proj)
        
        for layer in self.thinking_layers:
            # Combine with context
            combined = torch.cat([reason_state, context_proj], dim=-1)
            gate = self.reason_gate(combined)
            
            # Thinking step with RoPE
            reason_state = layer(reason_state, freqs_cis, mask)
            
            # Gated update
            reason_state = gate * reason_state + (1 - gate) * x
        
        return reason_state


class SimpleFusionLayer(nn.Module):
    """
    Simplified fusion layer for stable training.
    Uses simple cross-attention instead of full TransformerEncoder.
    """
    
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Simple cross-attention
        self.cross_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feed-forward with GELU activation
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout)
        )
        
        self.norm1 = RMSNorm(hidden_size)
        self.norm2 = RMSNorm(hidden_size)
        
        # Gradient scaling for stability
        self.scale = nn.Parameter(torch.ones(1) * 0.1)
    
    def forward(self, text_features: torch.Tensor, visual_features: torch.Tensor) -> torch.Tensor:
        """
        Fuse text and visual features.
        
        Args:
            text_features: [B, seq_len, hidden_size]
            visual_features: [B, 1, hidden_size] or [B, num_patches, hidden_size]
        
        Returns:
            fused_features: [B, seq_len + visual_len, hidden_size]
        """
        # Cross-attention: text attends to visual
        attn_out, _ = self.cross_attn(text_features, visual_features, visual_features)
        
        # Residual connection with scaling (for gradient stability)
        text_features = self.norm1(text_features + self.scale * attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(text_features)
        text_features = self.norm2(text_features + self.scale * ffn_out)
        
        # Concatenate with visual features
        fused = torch.cat([text_features, visual_features], dim=1)
        
        return fused

class MultimodalEncoder(nn.Module):
    """Encoder for both text and images with numerical stability"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_size = config['hidden_size']
        
        # Text encoder with proper initialization
        self.text_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'], padding_idx=0)
        # Initialize embeddings with smaller std for stability
        nn.init.normal_(self.text_embedding.weight, mean=0.0, std=0.02)
        
        self.text_pos_encoding = nn.Parameter(torch.randn(1, config['max_seq_len'], config['hidden_size']) * 0.02)
        self.embedding_norm = RMSNorm(config['hidden_size'], eps=1e-6)
        
        # Image encoder - Vision Transformer (Claude-style)
        if config['mode'] == 'pro':
            self.visual_encoder = models.vit_b_16(weights=None)
            # Remove classification heads
            self.visual_encoder.heads = nn.Identity()
            visual_features = 768
        else:  # lite
            self.visual_encoder = models.vit_b_32(weights=None)
            # Remove classification heads
            self.visual_encoder.heads = nn.Identity()
            visual_features = 768
        
        self.visual_projection = nn.Linear(visual_features, config['hidden_size'])
        # Initialize projection properly
        nn.init.xavier_uniform_(self.visual_projection.weight)
        nn.init.zeros_(self.visual_projection.bias)
        self.visual_norm = RMSNorm(config['hidden_size'], eps=1e-6)
        
        # Layer norms before attention for stability
        self.pre_cross_attn_norm_text = RMSNorm(config['hidden_size'], eps=1e-6)
        self.pre_cross_attn_norm_visual = RMSNorm(config['hidden_size'], eps=1e-6)
        
        # Cross-modal attention with proper dropout
        self.cross_attn = nn.MultiheadAttention(
            config['hidden_size'], 
            config['num_heads'], 
            batch_first=True,
            dropout=0.1
        )
        self.cross_attn_norm = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        
        # Simplified fusion layers (stable alternative to TransformerEncoder)
        self.fusion_layers = nn.ModuleList([
            SimpleFusionLayer(
                hidden_size=config['hidden_size'],
                num_heads=config['num_heads'],
                dropout=0.1
            )
            for _ in range(config['num_fusion_layers'])
        ])
        
        # Final fusion projection
        self.fusion_projection = nn.Sequential(
            RMSNorm(config['hidden_size']),
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Thinking and reasoning module
        self.reasoning = ReasoningModule(
            config['hidden_size'],
            num_steps=config['reasoning_steps'],
            num_heads=config['num_heads']
        )
        
        # Apply weight initialization
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with proper schemes for numerical stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, RMSNorm):
                nn.init.ones_(module.weight)
    
    def forward(self, text_input: torch.Tensor, image_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = text_input.shape[0]
        seq_len = text_input.shape[1]
        
        # Encode text with normalization
        text_emb = self.text_embedding(text_input)
        text_emb = text_emb + self.text_pos_encoding[:, :seq_len, :]
        text_emb = self.embedding_norm(text_emb)
        # Clamp to prevent extreme values
        text_emb = torch.clamp(text_emb, min=-10.0, max=10.0)
        
        # Encode image with normalization (Vision Transformer)
        visual_emb = self.visual_encoder(image_input)
        visual_emb = self.visual_projection(visual_emb)
        visual_emb = self.visual_norm(visual_emb)
        visual_emb = torch.clamp(visual_emb, min=-10.0, max=10.0)
        visual_emb = visual_emb.unsqueeze(1)  # Add sequence dimension
        
        # Normalize before cross-attention
        text_normed = self.pre_cross_attn_norm_text(text_emb)
        visual_normed = self.pre_cross_attn_norm_visual(visual_emb)
        
        # Cross-modal attention with residual
        cross_out, _ = self.cross_attn(
            text_normed, visual_normed, visual_normed
        )
        cross_out = self.cross_attn_norm(cross_out)
        # Residual connection
        cross_out = text_emb + 0.1 * cross_out
        
        # Apply simplified fusion layers iteratively with clamping
        fused = cross_out
        for fusion_layer in self.fusion_layers:
            fused = fusion_layer(fused, visual_emb)
            # Clamp after each fusion layer
            fused = torch.clamp(fused, min=-10.0, max=10.0)
        
        # Final projection with clamping
        fused = self.fusion_projection(fused[:, :seq_len, :])
        fused = torch.clamp(fused, min=-10.0, max=10.0)
        fused = torch.cat([fused, visual_emb], dim=1)
        
        # Thinking and reasoning with mask and RoPE
        # 1. Precompute RoPE frequencies for current sequence
        head_dim = self.hidden_size // self.config['num_heads']
        freqs_cis = precompute_freqs_cis(head_dim, seq_len).to(text_input.device)
        
        # 2. Reasoning with mask
        text_mask = (text_input == 0).to(torch.bool) if (text_input == 0).any() else None
        reasoned = self.reasoning(fused[:, :seq_len], fused, freqs_cis, text_mask)
        reasoned = torch.clamp(reasoned, min=-10.0, max=10.0)
        
        return {
            'text_features': text_emb,
            'visual_features': visual_emb,
            'fused_features': fused,
            'reasoned_features': reasoned,
            'freqs_cis': freqs_cis
        }


class ImageGenerator(nn.Module):
    """Text-to-Image Generator using progressive upsampling"""
    def __init__(self, hidden_size: int = 768, latent_dim: int = 512):
        super().__init__()
        self.hidden_size = hidden_size
        self.latent_dim = latent_dim
        
        # Project text features to latent space
        self.text_projector = nn.Sequential(
            nn.Linear(hidden_size, latent_dim),
            RMSNorm(latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, latent_dim * 4 * 4)  # 4x4 spatial
        )
        
        # Progressive upsampling decoder: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128 -> 224x224
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(latent_dim, 512, 4, 2, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # 64x64 -> 128x128
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            # 128x128 -> 256x256
            nn.ConvTranspose2d(32, 16, 4, 2, 1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            # Final conv to 3 channels (RGB)
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
        # Resize layer to get exact 224x224
        self.resize = nn.AdaptiveAvgPool2d((224, 224))
        
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Generate image from text features
        Args:
            text_features: [batch, hidden_size] - pooled text representation
        Returns:
            generated_image: [batch, 3, 224, 224] - RGB image in [-1, 1] range
        """
        batch_size = text_features.size(0)
        
        # Project to latent and reshape to 4x4 spatial
        latent = self.text_projector(text_features)
        latent = latent.view(batch_size, self.latent_dim, 4, 4)
        
        # Progressive upsampling
        image = self.decoder(latent)
        
        # Resize to exact 224x224
        image = self.resize(image)
        
        return image

class VelCoreModel(nn.Module):
    """Main VelCore model with pro and lite versions - 600M/300M Parameter Scale with Thinking"""
    def __init__(self, mode: str = 'pro', tokenizer_vocab_size: int = None):
        super().__init__()
        self.mode = mode
        
        # Configuration - 600M/300M Parameter Scale with Thinking
        if mode == 'pro':
            config = {
                'hidden_size': 1024,
                'vocab_size': tokenizer_vocab_size or 1000,
                'max_seq_len': 8192,
                'num_heads': 16,  # 1024 / 64 = 16 heads
                'num_fusion_layers': 6,
                'reasoning_steps': 8,
                'mode': 'pro'
            }
        else:  # lite
            config = {
                'hidden_size': 512,
                'vocab_size': tokenizer_vocab_size or 1000,
                'max_seq_len': 8192,
                'num_heads': 8,   # 512 / 64 = 8 heads
                'num_fusion_layers': 3,
                'reasoning_steps': 4,
                'mode': 'lite'
            }
        
        self.config = config
        
        # Encoder
        self.encoder = MultimodalEncoder(config)
        
        # Task heads with proper initialization
        self.classification_head = nn.Sequential(
            RMSNorm(config['hidden_size'], eps=1e-6),
            nn.Linear(config['hidden_size'], config['hidden_size'] // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(config['hidden_size'] // 2, config.get('num_classes', 10))
        )
        # Initialize classification head
        for module in self.classification_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.02)
                nn.init.zeros_(module.bias)
        
        # Text generation head (for reasoning output) with normalization
        self.generation_norm = RMSNorm(config['hidden_size'], eps=1e-6)
        self.generation_head = nn.Linear(config['hidden_size'], config['vocab_size'])
        # Initialize with small weights to prevent large logits
        nn.init.normal_(self.generation_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.generation_head.bias)
        
        # Image reconstruction head (for multimodal understanding)
        if mode == 'pro':
            self.image_reconstruction = nn.Sequential(
                nn.Linear(config['hidden_size'], 512),
                nn.ReLU(),
                nn.Linear(512, 1024),
                nn.ReLU(),
                nn.Linear(1024, 224 * 224 * 3),
                nn.Tanh()
            )
            
            # Image Generator (text-to-image)
            self.image_generator = ImageGenerator(
                hidden_size=config['hidden_size'],
                latent_dim=512
            )
    
    def forward(self, text_input: torch.Tensor, image_input: torch.Tensor) -> Dict:
        # Encode inputs
        encodings = self.encoder(text_input, image_input)
        
        # Get reasoning features with clamping
        reasoned = encodings['reasoned_features']
        reasoned = torch.clamp(reasoned, min=-10.0, max=10.0)
        
        # CLS token for classification (first token)
        cls_token = reasoned[:, 0, :]
        cls_token = torch.clamp(cls_token, min=-10.0, max=10.0)
        
        # Generate outputs with normalization
        classification_logits = self.classification_head(cls_token)
        # Scale down logits to prevent overflow in softmax
        classification_logits = torch.clamp(classification_logits, min=-20.0, max=20.0)
        
        # Generation logits with normalization
        reasoned_normed = self.generation_norm(reasoned)
        generation_logits = self.generation_head(reasoned_normed)
        generation_logits = torch.clamp(generation_logits, min=-20.0, max=20.0)
        
        outputs = {
            'logits': classification_logits,
            'reasoning_logits': generation_logits,
            'features': encodings
        }
        
        # Image reconstruction for pro version
        if self.mode == 'pro':
            outputs['reconstructed_image'] = self.image_reconstruction(
                encodings['visual_features'].squeeze(1)
            )
        
        return outputs
    
    def think(self, text_input: torch.Tensor, image_input: torch.Tensor) -> str:
        """Generate thinking/reasoning output"""
        with torch.no_grad():
            outputs = self.forward(text_input, image_input)
            reasoning_logits = outputs['reasoning_logits']
            
            # Convert reasoning tokens to text
            reasoning_tokens = torch.argmax(reasoning_logits, dim=-1)
            
            return reasoning_tokens
    
    def generate_image(self, text_input: torch.Tensor) -> torch.Tensor:
        """
        Generate image from text input (PRO mode only)
        Args:
            text_input: [batch, seq_len] - tokenized text
        Returns:
            generated_image: [batch, 3, 224, 224] - RGB image in [-1, 1] range
        """
        if self.mode != 'pro':
            raise ValueError("Image generation is only available in PRO mode")
        
        with torch.no_grad():
            # Create dummy image for encoder (we only need text features)
            batch_size = text_input.size(0)
            device = text_input.device
            dummy_image = torch.zeros(batch_size, 3, 224, 224, device=device)
            
            # Get text features from encoder
            encodings = self.encoder(text_input, dummy_image)
            
            # Use pooled text features (mean pooling)
            text_features = encodings['text_features'].mean(dim=1)  # [batch, hidden]
            
            # Generate image
            generated = self.image_generator(text_features)
            
            return generated
