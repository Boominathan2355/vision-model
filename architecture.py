import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from typing import Dict, Optional

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

# ==================== Model Architecture ====================
class ThinkingLayer(nn.Module):
    """Layer for thinking/reasoning process"""
    def __init__(self, hidden_size: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.mha = nn.MultiheadAttention(hidden_size, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Self-attention for thinking
        attn_out, _ = self.mha(x, x, x, key_padding_mask=mask)
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
    
    def forward(self, x: torch.Tensor, context: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Initial reasoning state
        reason_state = x
        batch_size = x.shape[0]
        
        # Project context to match reasoning state shape
        # Take mean across sequence dimension if needed
        if context.shape[1] != reason_state.shape[1]:
            context_proj = context.mean(dim=1, keepdim=True).expand_as(reason_state)
        else:
            context_proj = context
        
        context_proj = self.context_projection(context_proj)
        
        for i, layer in enumerate(self.thinking_layers):
            # Combine with context
            combined = torch.cat([reason_state, context_proj], dim=-1)
            gate = self.reason_gate(combined)
            
            # Thinking step
            reason_state = layer(reason_state, mask)
            
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
        
        # Layer normalization for stability
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
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
        self.embedding_norm = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        
        # Image encoder - initialized from scratch (no pretrained weights)
        if config['mode'] == 'pro':
            self.visual_encoder = models.resnet50(weights=None)
            visual_features = 2048
        else:  # lite
            self.visual_encoder = models.resnet18(weights=None)
            visual_features = 512
        
        # Remove classification head
        self.visual_encoder = nn.Sequential(*list(self.visual_encoder.children())[:-1])
        self.visual_projection = nn.Linear(visual_features, config['hidden_size'])
        # Initialize projection properly
        nn.init.xavier_uniform_(self.visual_projection.weight)
        nn.init.zeros_(self.visual_projection.bias)
        self.visual_norm = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        
        # Layer norms before attention for stability
        self.pre_cross_attn_norm_text = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        self.pre_cross_attn_norm_visual = nn.LayerNorm(config['hidden_size'], eps=1e-6)
        
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
            nn.LayerNorm(config['hidden_size']),
            nn.Linear(config['hidden_size'], config['hidden_size']),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Thinking and reasoning module
        self.reasoning = ReasoningModule(
            config['hidden_size'],
            num_steps=config['reasoning_steps']
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
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(self, text_input: torch.Tensor, image_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = text_input.shape[0]
        
        # Encode text with normalization
        text_emb = self.text_embedding(text_input)
        text_emb = text_emb + self.text_pos_encoding[:, :text_input.shape[1], :]
        text_emb = self.embedding_norm(text_emb)
        # Clamp to prevent extreme values
        text_emb = torch.clamp(text_emb, min=-10.0, max=10.0)
        
        # Encode image with normalization
        visual_features = self.visual_encoder(image_input)
        visual_features = visual_features.view(batch_size, -1)
        visual_emb = self.visual_projection(visual_features)
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
        fused = self.fusion_projection(fused[:, :text_emb.shape[1], :])
        fused = torch.clamp(fused, min=-10.0, max=10.0)
        fused = torch.cat([fused, visual_emb], dim=1)
        
        # Thinking and reasoning with mask
        text_mask = (text_input == 0)  # True for padding tokens
        reasoned = self.reasoning(fused[:, :text_emb.shape[1]], fused, text_mask)
        reasoned = torch.clamp(reasoned, min=-10.0, max=10.0)
        
        return {
            'text_features': text_emb,
            'visual_features': visual_emb,
            'fused_features': fused,
            'reasoned_features': reasoned
        }

class ThiranModel(nn.Module):
    """Main Thiran model with pro and lite versions"""
    def __init__(self, mode: str = 'pro', tokenizer_vocab_size: int = None):
        super().__init__()
        self.mode = mode
        
        # Configuration
        if mode == 'pro':
            config = {
                'hidden_size': 768,
                'vocab_size': tokenizer_vocab_size or 1000,
                'max_seq_len': 512,
                'num_heads': 12,
                'num_fusion_layers': 6,
                'reasoning_steps': 4,
                'mode': 'pro'
            }
        else:  # lite
            config = {
                'hidden_size': 384,
                'vocab_size': tokenizer_vocab_size or 1000,
                'max_seq_len': 512,
                'num_heads': 8,
                'num_fusion_layers': 3,
                'reasoning_steps': 2,
                'mode': 'lite'
            }
        
        self.config = config
        
        # Encoder
        self.encoder = MultimodalEncoder(config)
        
        # Task heads with proper initialization
        self.classification_head = nn.Sequential(
            nn.LayerNorm(config['hidden_size'], eps=1e-6),
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
        self.generation_norm = nn.LayerNorm(config['hidden_size'], eps=1e-6)
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
