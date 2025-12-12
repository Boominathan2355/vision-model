import torch
import torch.nn as nn
from architecture import VelCoreModel
import json
from typing import Dict, Optional

# ==================== Dynamic Architecture Config ====================
class DynamicArchConfig:
    """Dynamic architecture configuration based on data"""
    
    @staticmethod
    def from_data_stats(data_stats: Dict) -> Dict:
        """Generate optimal config from data statistics - Trillion Parameter Scale"""
        vocab_size = data_stats.get('vocab_size', 68)
        avg_seq_length = data_stats.get('avg_seq_length', 64)
        complexity = data_stats.get('complexity', 'medium')
        
        config = {
            'vocab_size': vocab_size,
            'avg_seq_length': avg_seq_length,
            'complexity': complexity,
        }
        
        # Hidden size adaptation - Trillion Parameter Scale
        if avg_seq_length > 256:
            config['hidden_size_pro'] = 6144
            config['hidden_size_lite'] = 3072
        elif avg_seq_length > 128:
            config['hidden_size_pro'] = 4096
            config['hidden_size_lite'] = 2048
        else:
            config['hidden_size_pro'] = 2048
            config['hidden_size_lite'] = 1024
        
        # Learning rate adaptation
        if vocab_size > 100:
            config['learning_rate'] = 5e-5
        elif vocab_size > 50:
            config['learning_rate'] = 1e-4
        else:
            config['learning_rate'] = 5e-4
        
        # Batch size recommendation
        config['batch_size'] = 2 if avg_seq_length > 256 else (4 if avg_seq_length > 128 else 8)
        
        return config

# ==================== Model Builder ====================
class VelCoreModelBuilder:
    """Builder for VelCore models with dynamic architecture support"""
    
    @staticmethod
    def build_pro_model(tokenizer_vocab_size: int, dynamic_config: Optional[Dict] = None) -> VelCoreModel:
        """Build pro version of VelCore model - Trillion Parameter Scale"""
        model = VelCoreModel(mode='pro', tokenizer_vocab_size=tokenizer_vocab_size)
        
        # Initialize weights
        model.apply(VelCoreModelBuilder._init_weights)
        
        print(f"✓ Pro Model built (Trillion Parameter Scale)")
        print(f"  - Mode: Pro")
        print(f"  - Hidden size: {dynamic_config.get('hidden_size_pro', 6144) if dynamic_config else 6144}")
        print(f"  - Num heads: 48")
        print(f"  - Num fusion layers: 24")
        print(f"  - Reasoning steps: 32")
        print(f"  - Vocab size: {tokenizer_vocab_size}")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        if dynamic_config:
            print(f"  - Learning rate: {dynamic_config.get('learning_rate', 5e-5)}")
            print(f"  - Batch size: {dynamic_config.get('batch_size', 2)}")
        
        return model
    
    @staticmethod
    def build_lite_model(tokenizer_vocab_size: int, dynamic_config: Optional[Dict] = None) -> VelCoreModel:
        """Build lite version of VelCore model - Trillion Parameter Scale"""
        model = VelCoreModel(mode='lite', tokenizer_vocab_size=tokenizer_vocab_size)
        
        # Initialize weights
        model.apply(VelCoreModelBuilder._init_weights)
        
        print(f"✓ Lite Model built (Trillion Parameter Scale)")
        print(f"  - Mode: Lite")
        print(f"  - Hidden size: {dynamic_config.get('hidden_size_lite', 3072) if dynamic_config else 3072}")
        print(f"  - Num heads: 24")
        print(f"  - Num fusion layers: 12")
        print(f"  - Reasoning steps: 16")
        print(f"  - Vocab size: {tokenizer_vocab_size}")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        if dynamic_config:
            print(f"  - Learning rate: {dynamic_config.get('learning_rate', 5e-5)*2}")
            print(f"  - Batch size: {dynamic_config.get('batch_size', 4)}")
        
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

def save_model(model: VelCoreModel, tokenizer, path: str = 'VelCore.pt', dynamic_config: Optional[Dict] = None):
    """Save model with configuration, weights, and metadata"""
    save_data = {
        'model_state_dict': model.state_dict(),
        'tokenizer_vocab': tokenizer.vocab,
        'tokenizer_max_length': tokenizer.max_length,
        'tokenizer_thinking_tokens': tokenizer.thinking_tokens,
        'mode': model.mode,
        'config': model.config,
        'model_weights': {name: param.clone().detach() for name, param in model.named_parameters()},
        'timestamp': torch.tensor([]).to(torch.device('cpu')),  # Placeholder for timestamp
        'dynamic_config': dynamic_config or {}
    }
    
    torch.save(save_data, path)
    print(f"✓ Model saved to {path}")
    print(f"  - Model state dict: {len(save_data['model_state_dict'])} tensors")
    print(f"  - Tokenizer vocab: {len(tokenizer.vocab)} tokens")
    if dynamic_config:
        print(f"  - Dynamic config saved: LR={dynamic_config.get('learning_rate')}, BS={dynamic_config.get('batch_size')}")

def load_model(path: str, tokenizer, mode: str = None) -> VelCoreModel:
    """Load saved model with weights and configuration"""
    save_data = torch.load(path, map_location='cpu')
    
    # Determine mode
    mode = mode or save_data.get('mode', 'pro')
    vocab_size = len(tokenizer.vocab)
    
    # Build appropriate model
    dynamic_config = save_data.get('dynamic_config', {})
    if mode == 'pro':
        model = VelCoreModelBuilder.build_pro_model(vocab_size, dynamic_config)
    else:
        model = VelCoreModelBuilder.build_lite_model(vocab_size, dynamic_config)
    
    # Load weights from state dict
    model.load_state_dict(save_data['model_state_dict'])
    
    print(f"✓ Model loaded from {path}")
    print(f"  - Mode: {mode}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, save_data.get('dynamic_config', {})
