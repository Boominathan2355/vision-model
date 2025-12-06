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

class MultimodalEncoder(nn.Module):
    """Encoder for both text and images"""
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        self.hidden_size = config['hidden_size']
        
        # Text encoder
        self.text_embedding = nn.Embedding(config['vocab_size'], config['hidden_size'])
        self.text_pos_encoding = nn.Parameter(torch.randn(1, config['max_seq_len'], config['hidden_size']))
        
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
        
        # Cross-modal attention
        self.cross_attn = nn.MultiheadAttention(
            config['hidden_size'], 
            config['num_heads'], 
            batch_first=True
        )
        
        # Fusion layers
        self.fusion_layers = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=config['hidden_size'],
                nhead=config['num_heads'],
                dim_feedforward=config['hidden_size'] * 4,
                batch_first=True
            ),
            num_layers=config['num_fusion_layers']
        )
        
        # Thinking and reasoning module
        self.reasoning = ReasoningModule(
            config['hidden_size'],
            num_steps=config['reasoning_steps']
        )
    
    def forward(self, text_input: torch.Tensor, image_input: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size = text_input.shape[0]
        
        # Encode text
        text_emb = self.text_embedding(text_input)
        text_emb = text_emb + self.text_pos_encoding[:, :text_input.shape[1], :]
        
        # Encode image
        with torch.no_grad():
            visual_features = self.visual_encoder(image_input)
        visual_features = visual_features.view(batch_size, -1)
        visual_emb = self.visual_projection(visual_features)
        visual_emb = visual_emb.unsqueeze(1)  # Add sequence dimension
        
        # Cross-modal attention
        cross_out, _ = self.cross_attn(
            text_emb, visual_emb, visual_emb
        )
        
        # Combine features
        combined = torch.cat([cross_out, visual_emb], dim=1)
        
        # Fusion
        fused = self.fusion_layers(combined)
        
        # Thinking and reasoning
        text_mask = (text_input != 0)  # Assuming 0 is padding
        reasoned = self.reasoning(fused[:, :text_emb.shape[1]], fused, text_mask)
        
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
        
        # Task heads
        self.classification_head = nn.Sequential(
            nn.Linear(config['hidden_size'], config['hidden_size'] // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(config['hidden_size'] // 2, config.get('num_classes', 10))
        )
        
        # Text generation head (for reasoning output)
        self.generation_head = nn.Linear(config['hidden_size'], config['vocab_size'])
        
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
        
        # Get reasoning features
        reasoned = encodings['reasoned_features']
        
        # CLS token for classification (first token)
        cls_token = reasoned[:, 0, :]
        
        # Generate outputs
        outputs = {
            'logits': self.classification_head(cls_token),
            'reasoning_logits': self.generation_head(reasoned),
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
