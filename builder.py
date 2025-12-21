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
        
        # Hidden size adaptation - 600M/300M Parameter Scale with Thinking
        if avg_seq_length > 256:
            config['hidden_size_pro'] = 1024
            config['hidden_size_lite'] = 512
        elif avg_seq_length > 128:
            config['hidden_size_pro'] = 768
            config['hidden_size_lite'] = 384
        else:
            config['hidden_size_pro'] = 512
            config['hidden_size_lite'] = 256
        
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
        """Build pro version of VelCore model - 600M Parameter Scale"""
        model = VelCoreModel(mode='pro', tokenizer_vocab_size=tokenizer_vocab_size)
        
        # Initialize weights
        model.apply(VelCoreModelBuilder._init_weights)
        
        print(f"✓ Pro Model built (600M Parameter Scale with Thinking)")
        print(f"  - Mode: Pro")
        print(f"  - Hidden size: {dynamic_config.get('hidden_size_pro', 1024) if dynamic_config else 1024}")
        print(f"  - Num heads: 16")
        print(f"  - Num fusion layers: 6")
        print(f"  - Reasoning steps: 8")
        print(f"  - Vocab size: {tokenizer_vocab_size}")
        print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        if dynamic_config:
            print(f"  - Learning rate: {dynamic_config.get('learning_rate', 5e-5)}")
            print(f"  - Batch size: {dynamic_config.get('batch_size', 2)}")
        
        return model
    
    @staticmethod
    def build_lite_model(tokenizer_vocab_size: int, dynamic_config: Optional[Dict] = None) -> VelCoreModel:
        """Build lite version of VelCore model - 300M Parameter Scale"""
        model = VelCoreModel(mode='lite', tokenizer_vocab_size=tokenizer_vocab_size)
        
        # Initialize weights
        model.apply(VelCoreModelBuilder._init_weights)
        
        print(f"✓ Lite Model built (300M Parameter Scale with Thinking)")
        print(f"  - Mode: Lite")
        print(f"  - Hidden size: {dynamic_config.get('hidden_size_lite', 512) if dynamic_config else 512}")
        print(f"  - Num heads: 8")
        print(f"  - Num fusion layers: 3")
        print(f"  - Reasoning steps: 4")
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

def save_model(model: VelCoreModel, tokenizer, path: str = 'Pro-model.pt', epoch: Optional[int] = None, dynamic_config: Optional[Dict] = None):
    """Save model with configuration, weights, and metadata"""
    save_data = {
        'state_dict': model.state_dict(),
        'vocab': tokenizer.vocab,
        'tokenizer_max_length': tokenizer.max_length,
        'tokenizer_thinking_tokens': tokenizer.thinking_tokens,
        'mode': model.mode,
        'config': model.config,
        'epoch': epoch,
        'timestamp': torch.tensor([]).to(torch.device('cpu')),  # Placeholder for timestamp
        'dynamic_config': dynamic_config or {}
    }
    
    torch.save(save_data, path)
    print(f"✓ Model saved to {path}")
    print(f"  - State dict: {len(save_data['state_dict'])} tensors")
    print(f"  - Tokenizer vocab: {len(tokenizer.vocab)} tokens")
    if epoch is not None:
        print(f"  - Epoch: {epoch}")
    if dynamic_config:
        print(f"  - Dynamic config saved: LR={dynamic_config.get('learning_rate')}, BS={dynamic_config.get('batch_size')}")

def load_model(path: str, tokenizer, mode: str = None) -> VelCoreModel:
    """
    Load saved model with weights and configuration
    Auto-resizes embeddings if vocabulary size mismatches
    """
    try:
        save_data = torch.load(path, map_location='cpu')
    except Exception as e:
        print(f"Error loading file {path}: {e}")
        raise e
    
    # Determine mode
    mode = mode or save_data.get('mode', 'pro')
    vocab_size = len(tokenizer.vocab)
    
    # Build appropriate model with CURRENT vocab size
    dynamic_config = save_data.get('dynamic_config', {})
    if mode == 'pro':
        model = VelCoreModelBuilder.build_pro_model(vocab_size, dynamic_config)
    else:
        model = VelCoreModelBuilder.build_lite_model(vocab_size, dynamic_config)
    
    # Process state dict (handle both old and new keys)
    state_dict = save_data.get('state_dict', save_data.get('model_state_dict'))
    if state_dict is None:
        state_dict = save_data
    
    # Check text_embedding mismatch (handle multiple naming conventions)
    # Architecture: encoder.text_embedding.weight
    # Unified Colab: encoder.text_emb.weight
    emb_key = None
    if 'encoder.text_embedding.weight' in state_dict: emb_key = 'encoder.text_embedding.weight'
    elif 'encoder.text_emb.weight' in state_dict: emb_key = 'encoder.text_emb.weight'
    
    if emb_key:
        saved_vocab_size = state_dict[emb_key].shape[0]
        if saved_vocab_size != vocab_size:
            print(f"  ⚠ Vocab size mismatch: Saved={saved_vocab_size}, Current={vocab_size}. Resizing...")
            
            # 1. Resize Encoder Embeddings (handle both text_embedding and text_emb)
            old_emb = state_dict[emb_key]
            
            # Find the active attribute name in the model
            model_emb = getattr(model.encoder, 'text_embedding', getattr(model.encoder, 'text_emb', None))
            if model_emb:
                new_emb = model_emb.weight.data.clone()
                min_vocab = min(saved_vocab_size, vocab_size)
                new_emb[:min_vocab] = old_emb[:min_vocab]
                state_dict[emb_key] = new_emb
            
            # 2. Resize Generation Head (handle generation_head and gen_head)
            head_weight_key = 'generation_head.weight' if 'generation_head.weight' in state_dict else ('gen_head.weight' if 'gen_head.weight' in state_dict else None)
            head_bias_key = 'generation_head.bias' if 'generation_head.bias' in state_dict else ('gen_head.bias' if 'gen_head.bias' in state_dict else None)
            
            if head_weight_key:
                old_head = state_dict[head_weight_key]
                model_head = getattr(model, 'generation_head', getattr(model, 'gen_head', None))
                if model_head:
                    new_head = model_head.weight.data.clone()
                    new_head[:min_vocab] = old_head[:min_vocab]
                    state_dict[head_weight_key] = new_head
                
            if head_bias_key:
                old_bias = state_dict[head_bias_key]
                model_head = getattr(model, 'generation_head', getattr(model, 'gen_head', None))
                if model_head:
                    new_bias = model_head.bias.data.clone()
                    new_bias[:min_vocab] = old_bias[:min_vocab]
                    state_dict[head_bias_key] = new_bias

    # Load weights
    try:
        msg = model.load_state_dict(state_dict, strict=False)
        if msg.missing_keys:
            print(f"  ⚠ Missing keys during load: {len(msg.missing_keys)}")
        if msg.unexpected_keys:
            print(f"  ⚠ Unexpected keys during load: {len(msg.unexpected_keys)}")
        print(f"✓ Model loaded from {path} (with vocab adaptation)")
    except Exception as e:
        print(f"⚠ Loading failed: {e}")
        
    print(f"  - Mode: {mode}")
    print(f"  - Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model, save_data.get('dynamic_config', {})
