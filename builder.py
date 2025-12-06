import torch
import torch.nn as nn
from architecture import ThiranModel

# ==================== Model Builder ====================
class ThiranModelBuilder:
    """Builder for Thiran models"""
    @staticmethod
    def build_pro_model(tokenizer_vocab_size: int) -> ThiranModel:
        """Build pro version of Thiran model"""
        model = ThiranModel(mode='pro', tokenizer_vocab_size=tokenizer_vocab_size)
        
        # Initialize weights
        model.apply(ThiranModelBuilder._init_weights)
        
        print(f"✓ Pro Model built")
        print(f"  - Mode: Pro")
        print(f"  - Hidden size: 768")
        print(f"  - Vocab size: {tokenizer_vocab_size}")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    @staticmethod
    def build_lite_model(tokenizer_vocab_size: int) -> ThiranModel:
        """Build lite version of Thiran model"""
        model = ThiranModel(mode='lite', tokenizer_vocab_size=tokenizer_vocab_size)
        
        # Initialize weights
        model.apply(ThiranModelBuilder._init_weights)
        
        print(f"✓ Lite Model built")
        print(f"  - Mode: Lite")
        print(f"  - Hidden size: 384")
        print(f"  - Vocab size: {tokenizer_vocab_size}")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        return model
    
    @staticmethod
    def _init_weights(module):
        """Initialize model weights"""
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

def save_model(model: ThiranModel, tokenizer, path: str = 'thiran_model.pt'):
    """Save model with configuration and weights"""
    save_data = {
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab': tokenizer.vocab,
        'tokenizer_max_length': tokenizer.max_length,
        'tokenizer_thinking_tokens': tokenizer.thinking_tokens,
        'mode': model.mode,
        'config': model.config,
        'model_weights': {name: param.clone().detach() for name, param in model.named_parameters()}
    }
    
    torch.save(save_data, path)
    print(f"✓ Model saved to {path}")
    print(f"  - Model state dict: {len(save_data['model_state_dict'])} tensors")
    print(f"  - Model weights: {len(save_data['model_weights'])} parameters")
    print(f"  - Tokenizer vocab: {len(tokenizer.vocab)} tokens")

def load_model(path: str, tokenizer, mode: str = None) -> ThiranModel:
    """Load saved model with weights"""
    save_data = torch.load(path, map_location='cpu')
    
    # Determine mode
    mode = mode or save_data.get('mode', 'pro')
    vocab_size = len(tokenizer.vocab)
    
    # Build appropriate model
    if mode == 'pro':
        model = ThiranModelBuilder.build_pro_model(vocab_size)
    else:
        model = ThiranModelBuilder.build_lite_model(vocab_size)
    
    # Load weights from state dict
    model.load_state_dict(save_data['model_state_dict'])
    
    print(f"✓ Model loaded from {path}")
    print(f"  - Mode: {mode}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model
